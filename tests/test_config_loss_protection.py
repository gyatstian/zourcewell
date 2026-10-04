"""A config.json that fails to parse must never be silently treated as empty.

An unparsable config is not an empty config. Loading it as `{}` made the
in-memory Settings empty, and the next save() -- any account action, any TUI
toggle, any ban persist -- rewrote the file with `"accounts": []`, destroying
every email, password and 64-char bearer token in it. A UTF-8 BOM (what
Windows editors add) was enough to trigger it.

Also covers the present-but-empty env override, which silently blanked
`obscura_bin` and so let the WAF path resolve a *different* binary than the
operator pinned.

And the second way a save destroys accounts without any error at all: the
accounts SNAPSHOT was taken outside `_CONFIG_SAVE_LOCK`, so two concurrent
mutators each wrote a payload built from a list one of them had already
outgrown. The file stayed valid JSON and lost an account.
"""

import json
import threading

import pytest

from deepseaport import config as cfg
from deepseaport.config import AccountConfig, Settings, load_settings


def _write(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def _account(email="real@x.com"):
    return {"email": email, "password": "hunter2", "token": "T" * 64}


@pytest.fixture(autouse=True)
def _clear_latch():
    cfg._UNPARSABLE_CONFIG.clear()
    yield
    cfg._UNPARSABLE_CONFIG.clear()


# --- unparsable must not destroy -------------------------------------------

def test_bom_is_stripped_and_accounts_survive(tmp_path):
    """A BOM is the most likely way a hand-edited config stops parsing."""
    p = tmp_path / "config.json"
    p.write_bytes(b"\xef\xbb\xbf" + json.dumps({"accounts": [_account()]}).encode())

    s = load_settings(p)

    assert len(s.accounts) == 1
    assert s.accounts[0].token == "T" * 64
    # Repaired, not latched -- a save is legitimate and must work.
    assert str(p) not in cfg._UNPARSABLE_CONFIG
    s.port = 5099
    s.save()
    assert len(load_settings(p).accounts) == 1


def test_truncated_config_refuses_to_be_saved_over(tmp_path):
    """The regression: this used to overwrite the file with an empty one."""
    p = tmp_path / "config.json"
    p.write_bytes(json.dumps({"accounts": [_account()]}).encode()[:40])

    s = load_settings(p)

    assert s.accounts == []          # app still starts
    assert str(p) in cfg._UNPARSABLE_CONFIG
    with pytest.raises(RuntimeError, match="refusing to save"):
        s.save()

    # The operator's bytes are still on disk, untouched.
    assert p.exists()
    assert p.read_bytes() != b""
    assert p.read_bytes().endswith(b"}") is False or len(p.read_bytes()) == 40


def test_garbage_config_refuses_to_be_saved_over(tmp_path):
    p = tmp_path / "config.json"
    p.write_text("this is not json at all", encoding="utf-8")

    s = load_settings(p)
    assert s.accounts == []
    with pytest.raises(RuntimeError):
        s.save()
    assert p.read_text(encoding="utf-8") == "this is not json at all"


def test_json_array_is_not_a_config(tmp_path):
    """Valid JSON, wrong shape -- same danger, so the same guard."""
    p = tmp_path / "config.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")

    s = load_settings(p)
    assert s.accounts == []
    with pytest.raises(RuntimeError):
        s.save()


def test_fixing_the_file_re_enables_saving(tmp_path):
    p = tmp_path / "config.json"
    p.write_text("{broken", encoding="utf-8")
    load_settings(p)
    assert str(p) in cfg._UNPARSABLE_CONFIG

    _write(p, {"accounts": [_account("back@x.com")], "port": 5099})
    s = load_settings(p)

    assert str(p) not in cfg._UNPARSABLE_CONFIG
    assert len(s.accounts) == 1
    s.port = 5099
    s.save()  # must not raise
    assert len(load_settings(p).accounts) == 1


def test_missing_and_blank_config_are_still_fresh_installs(tmp_path):
    """A missing or empty file is legitimate, not an error."""
    missing = tmp_path / "nope.json"
    assert load_settings(missing).accounts == []
    assert str(missing) not in cfg._UNPARSABLE_CONFIG

    blank = tmp_path / "config.json"
    blank.write_text("   \n", encoding="utf-8")
    assert load_settings(blank).accounts == []
    assert str(blank) not in cfg._UNPARSABLE_CONFIG
    blank.parent.joinpath("x")
    load_settings(blank).save()  # must not raise


def test_latch_is_per_path(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text("{broken", encoding="utf-8")
    _write(b, {"accounts": [_account()]})

    load_settings(a)
    load_settings(b)

    assert str(a) in cfg._UNPARSABLE_CONFIG
    assert str(b) not in cfg._UNPARSABLE_CONFIG
    load_settings(b).save()  # unaffected by a's latch


# --- present-but-empty env override ---------------------------------------

def test_empty_env_var_does_not_blank_config(tmp_path, monkeypatch):
    p = _write(tmp_path / "config.json",
               {"obscura_bin": "C:/my/obscura.exe", "obscura_profile": "C:/my/prof"})
    monkeypatch.setenv("OBSCURA_BIN", "")
    monkeypatch.setenv("OBSCURA_PROFILE", "   ")

    s = load_settings(p)

    assert s.obscura_bin == "C:/my/obscura.exe"
    assert s.obscura_profile == "C:/my/prof"


def test_real_env_var_still_overrides(tmp_path, monkeypatch):
    p = _write(tmp_path / "config.json", {"obscura_bin": "C:/my/obscura.exe"})
    monkeypatch.setenv("OBSCURA_BIN", "D:/other/obscura.exe")

    assert load_settings(p).obscura_bin == "D:/other/obscura.exe"


def test_env_var_wins_over_missing_config_value(tmp_path, monkeypatch):
    p = _write(tmp_path / "config.json", {})
    monkeypatch.setenv("OBSCURA_BIN", "D:/only/here.exe")

    assert load_settings(p).obscura_bin == "D:/only/here.exe"


# --- the unlocked snapshot lost a whole account -------------------------------


class _TrackedLock:
    """``threading.Lock`` stand-in that can say whether the CALLER holds it.

    ``Settings._CONFIG_SAVE_LOCK`` is a plain Lock, so a test cannot ask it
    anything -- including the one question the race below turns on: has
    ``save()`` taken the lock yet, or is it still building the payload? Being
    able to substitute the lock is the only instrumentation this test needs.

    Owner tracking (rather than a counter) because the answer is per-thread:
    the main thread is the writer that has to be kept out.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owner: int | None = None

    def __enter__(self):
        self._lock.acquire()
        self._owner = threading.get_ident()
        return self

    def __exit__(self, *exc):
        self._owner = None
        self._lock.release()

    def held(self) -> bool:
        return self._owner == threading.get_ident()


def test_two_concurrent_adds_both_survive_the_save(tmp_path, monkeypatch):
    """The regression: the accounts snapshot was taken OUTSIDE the lock.

    A read-modify-write whose read is unlocked drops a record the way a torn
    write drops bytes -- except the output is still valid JSON and nobody is
    told. Thread A snapshots two accounts, thread B appends a third and writes
    three, then A takes the lock and atomically replaces the file with its
    stale two-account payload. Every mutator routes through ``Settings.save()``
    (``accounts.add_account`` / ``set_account_token`` / ``remove_account``,
    ``server_routes``' POST/DELETE/unblock, and ``core.engine``'s
    ``ban_403_for_item`` / ``switch_current_after_ban_failover``), so this is
    the one place the fix belongs.

    Deterministic, not timing. ``json.dumps`` is the last step before the
    write, so hooking it parks writer A in exactly the window the fix closes.
    A waits there ONLY when the lock is not yet held: holding it there IS the
    fix, and then the second writer (this thread) can never make progress, so
    waiting would deadlock. That condition -- not a sleep -- is what makes this
    red on the old code and green on the new.
    """
    path = tmp_path / "config.json"
    settings = Settings(config_path=str(path))

    tracked = _TrackedLock()
    monkeypatch.setattr(cfg, "_CONFIG_SAVE_LOCK", tracked)

    real_dumps = cfg.json.dumps
    on_writer_a = threading.local()
    a_parked = threading.Event()      # A is at the serialize step
    b_written = threading.Event()     # B's save is on disk

    def hooked_dumps(*args, **kwargs):
        out = real_dumps(*args, **kwargs)
        if getattr(on_writer_a, "yes", False):
            a_parked.set()
            if not tracked.held():
                b_written.wait(30)
        return out

    monkeypatch.setattr(cfg.json, "dumps", hooked_dumps)

    def writer_a() -> None:
        on_writer_a.yes = True
        settings.accounts.append(AccountConfig(email="a@x.com", token="A" * 64))
        settings.save()

    a = threading.Thread(target=writer_a, name="writer-a")
    a.start()
    assert a_parked.wait(30), "writer A never reached the serialize step"

    # Writer B, on this thread: it must be able to complete, which is exactly
    # what the old unlocked snapshot made impossible.
    settings.accounts.append(AccountConfig(email="b@x.com", token="B" * 64))
    settings.save()
    b_written.set()

    a.join(30)
    assert not a.is_alive(), "writer A hung"

    emails = sorted(entry["email"] for entry in
                    json.loads(path.read_text(encoding="utf-8"))["accounts"])
    assert emails == ["a@x.com", "b@x.com"], (
        f"a concurrent add was silently lost on the way to disk: {emails}")


def test_save_lock_is_plain_so_a_nested_acquisition_would_deadlock():
    """``save()`` must acquire ``_CONFIG_SAVE_LOCK`` exactly once.

    The fix is "move the ``with`` up", never "add a second one": the lock is a
    plain ``Lock``, so an acquisition nested inside the first self-deadlocks
    instead of re-entering. Pin the property that makes the move the only way
    to do it, so a future edit reaches for ``RLock`` deliberately (or not).
    """
    assert not hasattr(cfg._CONFIG_SAVE_LOCK, "_is_owned"), (
        "_CONFIG_SAVE_LOCK became reentrant; save() may now nest acquisitions")
