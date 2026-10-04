"""Config integrity: credential-preserving saves, honest parsing, no leaks.

Every defect here shares one theme: ``config.json`` is not a settings file, it
is the operator's live credential store (account emails, plaintext passwords,
64-char bearer tokens, API keys). So the bar for each fix is not "the value is
right" but "a wrong value, a crash or a stray log line must not cost the
operator their accounts".

Covered:
  1. ``banned: "false"`` loaded as a permanent ban (bool("false") is True).
  2. Save-crash safety of the degraded (os.replace blocked) write path.
  3. The atomic happy path still round-trips and leaves no .tmp residue.
  4. ``port`` range validation (0 / 70000 / negative / junk / valid).
  5. repr() of Settings/AccountConfig leaking the secrets it holds.
  6. Duplicate accounts loading silently into the pool.
  7. A single-entry ``providers`` list being ignored without a word.
  8. A realistic save/load round trip still works.

Idioms mirror the existing atomic-save tests in tests/test_reliability.py
(`test_save_is_atomic_and_leaves_no_tmp`, `test_concurrent_saves_never_corrupt`):
tmp_path for the config, real Settings.save(), and assertions on the actual
bytes on disk rather than on internal calls.
"""

from __future__ import annotations

import json
import os

import pytest

from deepseaport import config as config_mod
from deepseaport.config import (
    DEFAULT_PORT,
    PORT_LIMIT,
    AccountConfig,
    Settings,
    load_settings,
)
from support import settings
from functools import partial
from deepseaport.core import pools
from deepseaport.core import registry as registry_mod

_TOKEN = "T" * 64


def _write(path, payload: dict) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _clean_env(monkeypatch) -> None:
    """Drop every env input load_settings reads, so the file is the only source."""
    for name in list(os.environ):
        if name.startswith(("DEEPSEAPORT_", "DEEPSEEK_")):
            monkeypatch.delenv(name, raising=False)


# Default kwargs differ per file; the shared base dict lives in
# support.py. Every list here is rebuilt per call on purpose.
_settings = partial(settings, log_level="INFO", warmup_on_startup=True)


# 1. banned is a real boolean, whatever the JSON held ------------------------


@pytest.mark.parametrize("raw,expected", [
    (True, True),
    (False, False),
    ("true", True),
    ("false", False),
    ("TRUE", True),
    ("False", False),
    ("  yes  ", True),
    ("  NO\t", False),
    ("1", True),
    ("0", False),
    ("on", True),
    ("maybe", False),          # unreadable must not read as banned
    (1, True),
    (0, False),
    (None, False),             # explicit null, same as missing
])
def test_banned_accepts_every_spelling(tmp_path, raw, expected):
    """"banned": "false" used to load as banned=True with banned_until=0.0 -
    "expiry unknown", so __post_init__ never cleared it and the account stayed
    (BANNED) forever, with no date and no way back except the unblock endpoint.
    """
    p = _write(tmp_path / "c.json", {"accounts": [
        {"email": "a@x.com", "token": "t1", "banned": raw, "banned_until": 0.0}]})
    assert load_settings(p).accounts[0].banned is expected


@pytest.mark.parametrize("raw", ["false", "False", "0", "no", "", False, 0, None])
def test_falsy_banned_leaves_no_unexpirable_ban(tmp_path, raw):
    """The regression in full: the marker is False AND there is no expiry for
    __post_init__ to clear, which is what made the old behaviour permanent."""
    p = _write(tmp_path / "c.json", {"accounts": [
        {"email": "a@x.com", "token": "t1", "banned": raw}]})
    acct = load_settings(p).accounts[0]
    assert acct.banned is False
    assert acct.banned_until == 0.0


def test_banned_missing_key_is_not_banned(tmp_path):
    p = _write(tmp_path / "c.json", {"accounts": [{"email": "a@x.com", "token": "t1"}]})
    assert load_settings(p).accounts[0].banned is False


def test_banned_true_string_still_bans(tmp_path):
    """The other direction: the fix must not make bans unreadable."""
    p = _write(tmp_path / "c.json", {"accounts": [
        {"email": "a@x.com", "token": "t1", "banned": "true",
         "banned_until": 4_000_000_000.0}]})
    assert load_settings(p).accounts[0].banned is True


def test_expired_ban_is_still_cleared(tmp_path):
    """Pre-existing behaviour, pinned: __post_init__ expires a past ban."""
    p = _write(tmp_path / "c.json", {"accounts": [
        {"email": "a@x.com", "token": "t1", "banned": True, "banned_until": 1.0}]})
    acct = load_settings(p).accounts[0]
    assert acct.banned is False and acct.banned_until == 0.0


# 2. save-crash safety --------------------------------------------------------


def _block_replace(monkeypatch) -> None:
    """Force the degraded in-place path the way a Windows AV handle does."""
    def blocked(*_args, **_kwargs):
        raise PermissionError(32, "The process cannot access the file")
    monkeypatch.setattr(config_mod.os, "replace", blocked)


def test_degraded_save_failure_keeps_previous_bytes(tmp_path, monkeypatch):
    """A failure *inside* the degraded write must leave the previous config.

    This is the guarantee the old `open(target, "w")` fallback could not make:
    "w" truncates at open, so the file was already 0 bytes before the first
    byte of the new content existed. Here the write lands half the payload and
    then fails the way a full disk does.
    """
    cfg = tmp_path / "config.json"
    s = Settings(keys=["sk-live"], accounts=[
        AccountConfig(email="a@x.com", password="hunter2", token=_TOKEN)],
        active_account="a@x.com", config_path=str(cfg))
    s.save()
    good = cfg.read_bytes()
    assert good, "precondition: a good config on disk"

    _block_replace(monkeypatch)
    real_write = config_mod.os.write
    state = {"n": 0}

    def flaky_write(fd, data):
        # One os.write per _write_all call, in a fixed order: 1 the .bak copy,
        # 2 the in-place config write, 3 the rollback.
        state["n"] += 1
        if state["n"] == 2:
            real_write(fd, bytes(data[: len(data) // 2]))
            raise OSError(28, "No space left on device")
        return real_write(fd, data)

    monkeypatch.setattr(config_mod.os, "write", flaky_write)
    s.accounts[0].email = "b@x.com"
    with pytest.raises(OSError):
        s.save()
    monkeypatch.undo()

    assert cfg.stat().st_size == len(good), "the file shrank mid-save"
    assert cfg.read_bytes() == good, "previous good config was not restored"
    assert (tmp_path / "config.json.bak").read_bytes() == good, \
        "no .bak copy of the last good state"


def test_degraded_save_crash_never_produces_zero_bytes(tmp_path):
    """Even when the rollback cannot finish either, the file is not zeroed.

    This is the hard floor: partial new content (possibly invalid JSON, which
    load_settings already treats as an empty config) is recoverable from the
    .bak, an empty credential store is not.
    """
    cfg = tmp_path / "config.json"
    s = Settings(accounts=[AccountConfig(email="a@x.com", password="pw",
                                         token=_TOKEN)], config_path=str(cfg))
    s.save()
    good = cfg.read_bytes()

    _block_replace(monkeypatch := pytest.MonkeyPatch())
    real_write = config_mod.os.write
    state = {"n": 0}

    def half_then_die(fd, data):
        state["n"] += 1
        if state["n"] == 1:            # the .bak copy still lands
            return real_write(fd, data)
        real_write(fd, bytes(data[: max(1, len(data) // 2)]))
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(config_mod.os, "write", half_then_die)
    s.accounts[0].email = "b@x.com"
    with pytest.raises(OSError):
        s.save()
    monkeypatch.undo()

    assert cfg.stat().st_size > 0, "config.json was reduced to 0 bytes"
    # The previous good state is still one file away.
    assert (tmp_path / "config.json.bak").read_bytes() == good


def test_degraded_save_shortening_does_not_leave_old_tail(tmp_path):
    """ftruncate runs only after the new content is durable, so shrinking the
    config cannot leave `new + old_tail` (a torn file) behind."""
    cfg = tmp_path / "config.json"
    s = Settings(accounts=[
        AccountConfig(email=f"pad-{i}@x.com", password="pw", token=f"tok{i}" * 8)
        for i in range(20)], config_path=str(cfg))
    s.save()
    assert len(cfg.read_bytes()) > 2000

    _block_replace(monkeypatch := pytest.MonkeyPatch())
    s.accounts = [AccountConfig(email="only@x.com", token="t1")]
    s.save()
    monkeypatch.undo()

    reloaded = json.loads(cfg.read_text(encoding="utf-8"))
    assert [a["email"] for a in reloaded["accounts"]] == ["only@x.com"]


def test_degraded_save_writes_config_and_baks(tmp_path):
    """The degraded path still persists the new state (it is a fallback, not a
    no-op) and leaves the previous content in <name>.bak (already gitignored via
    *.bak, so nothing new to clean up)."""
    cfg = tmp_path / "config.json"
    s = Settings(accounts=[AccountConfig(email="a@x.com", token="t1")],
                 config_path=str(cfg))
    s.save()
    first = cfg.read_bytes()

    _block_replace(monkeypatch := pytest.MonkeyPatch())
    s.accounts.append(AccountConfig(email="b@x.com", token="t2"))
    s.save()
    monkeypatch.undo()

    reloaded = json.loads(cfg.read_text(encoding="utf-8"))
    assert [a["email"] for a in reloaded["accounts"]] == ["a@x.com", "b@x.com"]
    assert (tmp_path / "config.json.bak").read_bytes() == first
    assert list(tmp_path.glob("*.tmp")) == []


# 3. atomic happy path --------------------------------------------------------


def test_atomic_save_roundtrips_and_leaves_no_residue(tmp_path):
    cfg = tmp_path / "config.json"
    s = Settings(accounts=[AccountConfig(email="a@x.com", token="t1")],
                 config_path=str(cfg))
    s.save()
    assert json.loads(cfg.read_text(encoding="utf-8"))["accounts"][0]["email"] == "a@x.com"
    assert list(tmp_path.glob("*.tmp")) == []
    # The happy path is atomic, so it needs no backup file.
    assert not (tmp_path / "config.json.bak").exists()


# 4. port range validation ----------------------------------------------------


def test_port_zero_falls_back_to_default(monkeypatch, tmp_path, caplog):
    _clean_env(monkeypatch)
    p = _write(tmp_path / "c.json", {"port": 0})
    with caplog.at_level("WARNING", logger="deepseaport.config"):
        s = load_settings(p)
    assert s.port == DEFAULT_PORT
    assert any("port" in r.getMessage() for r in caplog.records)


def test_port_above_range_is_clamped(monkeypatch, tmp_path, caplog):
    """70000 reaches socket.bind() and raises OverflowError, which is not an
    OSError and so escapes the bind path's handlers as a traceback."""
    _clean_env(monkeypatch)
    p = _write(tmp_path / "c.json", {"port": 70000})
    with caplog.at_level("WARNING", logger="deepseaport.config"):
        s = load_settings(p)
    assert s.port == PORT_LIMIT == 65535
    assert any("port" in r.getMessage() for r in caplog.records)


def test_port_negative_and_junk_are_sane(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    assert load_settings(_write(tmp_path / "a.json", {"port": -1})).port == DEFAULT_PORT
    assert load_settings(_write(tmp_path / "b.json", {"port": "nope"})).port == DEFAULT_PORT
    assert load_settings(_write(tmp_path / "c.json", {})).port == DEFAULT_PORT


def test_port_valid_values_untouched(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    assert load_settings(_write(tmp_path / "a.json", {"port": 8080})).port == 8080
    assert load_settings(_write(tmp_path / "b.json", {"port": 65535})).port == 65535


def test_port_env_is_range_checked_too(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    p = _write(tmp_path / "c.json", {"port": 8080})
    monkeypatch.setenv("DEEPSEAPORT_PORT", "70000")
    assert load_settings(p).port == 65535
    monkeypatch.setenv("DEEPSEAPORT_PORT", "0")
    assert load_settings(p).port == DEFAULT_PORT


def test_port_clamp_does_not_rewrite_the_operator_file(monkeypatch, tmp_path):
    """RUNTIME-ONLY, like the max_retries clamp: save() runs on unrelated
    actions, so persisting the corrected number would silently rewrite a
    config the operator never asked to change."""
    _clean_env(monkeypatch)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"port": 70000, "keys": ["sk"]}), encoding="utf-8")
    s = load_settings(str(cfg))
    s.save()
    assert json.loads(cfg.read_text(encoding="utf-8"))["port"] == 70000


# 5. repr must not leak --------------------------------------------------------


def test_accountconfig_repr_hides_password_and_token():
    acct = AccountConfig(email="a@x.com", mobile="+48123", password="hunter2",
                         token=_TOKEN, token_expires_at=123.0, banned=True,
                         banned_until=4_000_000_000.0, provider="qwen")
    text = repr(acct)
    assert "hunter2" not in text
    assert _TOKEN not in text
    assert "T" * 10 not in text, "no token prefix either"
    # ... while the fields a debugging print is actually for stay readable.
    for useful in ("a@x.com", "+48123", "qwen", "banned=True", "123.0"):
        assert useful in text, useful
    assert "password=<set>" in text and "token=<set>" in text
    assert str(acct) == text, "no __str__ bypass"


def test_accountconfig_repr_marks_empty_secrets():
    text = repr(AccountConfig(email="a@x.com"))
    assert "password=<empty>" in text and "token=<empty>" in text


def test_settings_repr_hides_keys_and_accounts():
    s = Settings(keys=["sk-super-secret-key"], accounts=[
        AccountConfig(email="a@x.com", password="hunter2", token=_TOKEN)],
        active_account="a@x.com", port=5001, config_path="cfg.json",
        provider="qwen", providers=["qwen", "opencode"])
    text = repr(s)
    for secret in ("sk-super-secret-key", "hunter2", _TOKEN, "T" * 10):
        assert secret not in text, secret
    assert "<set x1>" in text
    # Non-secret context survives: which accounts, which provider, which file.
    for useful in ("a@x.com", "qwen", "cfg.json", "5001", "active_account"):
        assert useful in text, useful
    assert str(s) == text


def test_settings_repr_does_not_break_equality_or_save(tmp_path):
    """repr=False must not disturb the dataclass features the code relies on."""
    s = Settings(accounts=[AccountConfig(email="a@x.com", token="t1")],
                 config_path=str(tmp_path / "c.json"))
    assert s == Settings(accounts=[AccountConfig(email="a@x.com", token="t1")],
                         config_path=str(tmp_path / "c.json"))
    s.save()
    assert json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))["port"] == DEFAULT_PORT


# 6. duplicate accounts ---------------------------------------------------------


def test_load_dedupes_same_identity_and_keeps_first(tmp_path, caplog):
    p = _write(tmp_path / "c.json", {"accounts": [
        {"email": "first@x.com", "token": "tok1", "password": "keep-me"},
        {"email": "FIRST@x.com", "token": "tok2"},      # same email, case-folded
        {"email": "third@x.com", "token": "tok3"},
        {"email": "third@x.com", "token": "tok3"},       # exact repeat
    ]})
    with caplog.at_level("WARNING", logger="deepseaport.config"):
        s = load_settings(p)
    assert [a.email for a in s.accounts] == ["first@x.com", "third@x.com"]
    assert s.accounts[0].token == "tok1", "the FIRST occurrence wins"
    assert s.accounts[0].password == "keep-me", "with its credentials intact"
    assert any("duplicate account" in r.getMessage() for r in caplog.records)


def test_load_dedupes_on_token_identity(tmp_path, caplog):
    p = _write(tmp_path / "c.json", {"accounts": [
        {"email": "a@x.com", "token": "shared-token"},
        {"email": "b@x.com", "token": "shared-token"},
    ]})
    with caplog.at_level("WARNING", logger="deepseaport.config"):
        s = load_settings(p)
    assert [a.email for a in s.accounts] == ["a@x.com"]
    assert any("duplicate account" in r.getMessage() for r in caplog.records)


def test_load_keeps_distinct_accounts(tmp_path):
    """Same email on different providers is two credentials; distinct mobiles
    and half-filled entries must survive."""
    p = _write(tmp_path / "c.json", {"accounts": [
        {"email": "a@x.com", "token": "t1"},
        {"email": "a@x.com", "token": "t2", "provider": "qwen"},
        {"email": "b@x.com", "mobile": "48123", "token": "t3"},
        {"email": "c@x.com"},
        {"mobile": "99999", "token": "t4"},
    ]})
    s = load_settings(p)
    assert [a.identifier for a in s.accounts] == [
        "a@x.com", "a@x.com", "b@x.com", "c@x.com", "99999"]
    assert s.accounts[0].provider == "deepseek"
    assert s.accounts[1].provider == "qwen"


def test_pool_rejects_what_load_dedupes(tmp_path):
    """The runtime gate and the load gate must agree on identity, or a
    duplicate walks in through one of them."""
    from deepseaport.accounts import AccountPool
    p = _write(tmp_path / "c.json", {"accounts": [
        {"email": "a@x.com", "token": "t1"},
        {"email": "b@x.com", "token": "shared"},
        {"email": "c@x.com", "token": "shared"},
    ]})
    loaded = load_settings(p).accounts
    assert len(loaded) == 2
    pool = AccountPool(loaded)
    assert len(pool) == 2
    with pytest.raises(ValueError):
        pool.add(AccountConfig(email="a@x.com", token="t9"))


def test_load_warns_but_does_not_truncate_over_the_account_limit(tmp_path, caplog):
    """MAX_ACCOUNTS is an admission cap on POST /v1/accounts. Truncating at load
    would delete credentials on the next save(), which rewrites the file from
    settings.accounts - so an over-limit config is reported, never pruned."""
    p = _write(tmp_path / "c.json", {"accounts": [
        {"email": f"a{i}@x.com", "token": f"t{i}"} for i in range(config_mod.MAX_ACCOUNTS_LIMIT + 3)]})
    with caplog.at_level("WARNING", logger="deepseaport.config"):
        s = load_settings(p)
    assert len(s.accounts) == config_mod.MAX_ACCOUNTS_LIMIT + 3
    assert any(str(config_mod.MAX_ACCOUNTS_LIMIT) in r.getMessage()
               for r in caplog.records)


# 7. single-entry providers list ------------------------------------------------


def test_single_provider_entry_is_ignored_and_reported(monkeypatch, caplog):
    """Multi mode is defined as 2+ ids (ARCHITECTURE.md) and pinned by
    tests/test_multi_provider.py, so one entry still loses to `provider` - but
    the ignored entry is now named instead of silently dropped."""
    monkeypatch.setattr(pools, "_WARNED_IGNORED_PROVIDERS", set())
    s = _settings(provider="deepseek", providers=["qwen"])
    with caplog.at_level("WARNING", logger="deepseaport.core.pools"):
        ids = pools.active_provider_ids(s)
    assert ids == ["deepseek"]
    assert pools.is_multi(s) is False
    messages = [r.getMessage() for r in caplog.records]
    assert any("qwen" in m and "deepseek" in m for m in messages), messages


def test_provider_list_warns_once(monkeypatch, caplog):
    """active_provider_ids is on the per-request path; a warning per call would
    bury the log."""
    monkeypatch.setattr(pools, "_WARNED_IGNORED_PROVIDERS", set())
    s = _settings(provider="deepseek", providers=["qwen"])
    with caplog.at_level("WARNING", logger="deepseaport.core.pools"):
        for _ in range(5):
            pools.active_provider_ids(s)
    assert sum("single provider" in r.getMessage() for r in caplog.records) == 1


def test_matching_single_entry_is_not_warned_about(monkeypatch, caplog):
    """providers=[provider] is redundant, not wrong: nothing was ignored, so
    there is nothing to report."""
    monkeypatch.setattr(pools, "_WARNED_IGNORED_PROVIDERS", set())
    with caplog.at_level("WARNING", logger="deepseaport.core.pools"):
        assert pools.active_provider_ids(
            _settings(provider="deepseek", providers=["deepseek"])) == ["deepseek"]
    assert not [r for r in caplog.records if "single provider" in r.getMessage()]


def test_empty_and_multi_providers_unchanged(monkeypatch):
    monkeypatch.setattr(pools, "_WARNED_IGNORED_PROVIDERS", set())
    # documented "empty = use provider only"
    assert pools.active_provider_ids(_settings(provider="deepseek", providers=[])) == ["deepseek"]
    assert pools.active_provider_ids(_settings(provider="deepseek")) == ["deepseek"]
    # 2+ unchanged, incl. case folding, dedupe and blank entries
    assert pools.active_provider_ids(
        _settings(provider="deepseek", providers=["deepseek", "qwen", "DEEPSEEK", ""])) == [
        "deepseek", "qwen"]
    assert pools.is_multi(_settings(providers=["deepseek", "qwen"])) is True


# 8. realistic round trip --------------------------------------------------------


def test_full_config_round_trip(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    cfg = tmp_path / "config.json"
    original = {
        "keys": ["sk-one", "sk-two"],
        "accounts": [
            {"email": "a@x.com", "mobile": "", "password": "pw-a",
             "token": "tok-a" * 10, "token_expires_at": 4_000_000_000.0,
             "banned": True, "banned_until": 4_000_000_000.0, "provider": "deepseek"},
            {"email": "b@x.com", "mobile": "+48123456789", "password": "pw-b",
             "token": "tok-b" * 10, "token_expires_at": 0.0,
             "banned": False, "banned_until": 0.0, "provider": "qwen"},
        ],
        "active_account": "a@x.com",
        "obscura_bin": "C:/tools/obscura.exe",
        "obscura_profile": "C:/profiles/one",
        "browser_bin": "auto",
        "browser_headless": True,
        "port": 5123,
        "listen": True,
        "enable_tools": True,
        "warmup_on_startup": False,
        "auto_delete_session": True,
        "max_retries": 7,
        "parallel_challenge_fetch": False,
        "tool_args_max_chars": 50_000,
        "forgiving_toolcalls": True,
        "use_multiple_accounts": True,
        "log_level": "DEBUG",
        "stream_mode": "buffered",
        "chat_model": "qwen-max",
        "enable_append": True,
        "append_top": "top line",
        "append_bottom": "bottom line",
        "send_dry_run_to_frontend": False,
        "provider": "deepseek",
        "providers": ["deepseek", "qwen"],
    }
    cfg.write_text(json.dumps(original), encoding="utf-8")

    s = load_settings(str(cfg))
    assert s.port == 5123 and s.listen is True
    assert s.max_retries == 5, "still clamped at load"
    assert [a.identifier for a in s.accounts] == ["a@x.com", "b@x.com"]
    assert s.accounts[0].banned is True and s.accounts[1].banned is False
    assert pools.active_provider_ids(s) == ["deepseek", "qwen"]

    s.save()
    reloaded = load_settings(str(cfg))
    assert reloaded == s
    # Credentials survive verbatim (they are the whole point of the file).
    assert reloaded.accounts[0].password == "pw-a"
    assert reloaded.accounts[0].token == "tok-a" * 10
    assert reloaded.keys == ["sk-one", "sk-two"]
    on_disk = json.loads(cfg.read_text(encoding="utf-8"))
    assert on_disk["max_retries"] == 7, "the clamp must not be persisted"
    assert on_disk["provider"] == "deepseek"
    assert list(tmp_path.glob("*.tmp")) == []


# 9. a provider id must be checked against the registry ------------------------
#
# `provider` loaded with no `valid=` tuple while `log_level` and `stream_mode`
# both have one, so it was the one config key nothing validated. "provider":
# "glm" produced a server that reported ready, answered /v1/models with
# DeepSeek's catalogue and routed every chat request to DeepSeek: no error, no
# log line, no 400.


def test_unknown_provider_ids_are_named_and_replaced_at_load(tmp_path, caplog):
    p = _write(tmp_path / "c.json",
               {"provider": "glm", "providers": ["deepseek", "quwn"]})
    with caplog.at_level("WARNING", logger="deepseaport.config"):
        s = load_settings(p)

    assert s.provider == "deepseek"
    # The unknown ENTRY goes too, not just the scalar: 2+ ids is what enables
    # multi mode, so leaving "quwn" in place would build a pool and a model
    # list for a provider that cannot exist.
    assert s.providers == ["deepseek"]

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "glm" in messages, "the rejected scalar id was not named"
    assert "quwn" in messages, "the rejected list entry was not named"
    assert "deepseek" in messages, "the operator was not told what to use"


def test_an_unknown_provider_is_never_silently_accepted(tmp_path, caplog):
    """The regression itself: no warning means the operator cannot see the typo."""
    p = _write(tmp_path / "c.json", {"provider": "typo"})
    with caplog.at_level("WARNING", logger="deepseaport.config"):
        s = load_settings(p)

    assert s.provider == "deepseek"
    assert any("typo" in r.getMessage() for r in caplog.records), (
        "the bad id was substituted with nothing logged")


def test_an_unknown_provider_from_the_env_is_named_too(tmp_path, monkeypatch,
                                                        caplog):
    monkeypatch.setenv("DEEPSEAPORT_PROVIDER", "quwn")
    p = _write(tmp_path / "c.json", {})
    with caplog.at_level("WARNING", logger="deepseaport.config"):
        s = load_settings(p)

    assert s.provider == "deepseek"
    assert "quwn" in caplog.text


def test_registered_providers_are_left_alone(tmp_path):
    p = _write(tmp_path / "c.json",
               {"provider": "qwen", "providers": ["deepseek", "qwen"]})
    s = load_settings(p)
    assert s.provider == "qwen"
    assert s.providers == ["deepseek", "qwen"]


def test_an_omitted_provider_is_still_the_documented_default(tmp_path, caplog):
    """Validation must not turn "no provider configured" into an error."""
    p = _write(tmp_path / "c.json", {"accounts": []})
    with caplog.at_level("WARNING", logger="deepseaport.config"):
        s = load_settings(p)

    assert s.provider == "deepseek"
    assert s.providers == []
    assert not caplog.records, [r.getMessage() for r in caplog.records]


def test_a_registry_that_cannot_be_asked_leaves_the_config_alone(monkeypatch,
                                                                 tmp_path):
    """Unable to answer "is this registered?" must not flatten a live config.

    ``core.registry`` is reached through a lazy import, and a deployment where
    it fails must keep the ids it was given rather than be silently moved to
    deepseek -- "cannot answer" is not the same verdict as "nothing exists".
    """
    def _boom():
        raise RuntimeError("registry exploded")

    monkeypatch.setattr(registry_mod, "registered", _boom)
    p = _write(tmp_path / "c.json", {"provider": "future-provider",
                                     "providers": ["future-provider"]})
    s = load_settings(p)

    assert s.provider == "future-provider"
    assert s.providers == ["future-provider"]