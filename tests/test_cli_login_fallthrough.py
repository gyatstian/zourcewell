"""F6: a non-DeepSeek login failure must never fall through to DeepSeek.

``cli._obscura_login_token`` resolved ``registry.get(pid).login`` for a
non-deepseek pid inside a bare ``except: pass`` and then ran the DeepSeek
Obscura driver unconditionally. A ~30s chat.deepseek.com browser login opened
with the operator's credentials, its userToken was stored under the OTHER
provider's id, and the command reported "login ok, token saved" with rc 0.
Reachable in two shapes: opencode's ``login`` raises NotImplementedError by
design, and a failed Qwen signin raises.

F13 lives here too: the same command's SUCCESS path reported a clean
"login ok, token saved" while the record it had just updated kept a persisted
ban, which the pool honours by skipping the account -- so `chat` answered "no
account free" for an account `accounts list` showed as usable.
"""

from __future__ import annotations

import pytest
from support import on_disk, settings, write_config

from deepseaport.cli import main as cli_main


def _raising_login(monkeypatch, exc: BaseException) -> list:
    """Registry whose ``qwen`` provider raises from ``login``; returns its calls."""
    import deepseaport.core.registry as reg

    calls: list = []

    class _FakeQwen:
        def login(self, email, password, settings):  # noqa: A002 (registry seam)
            calls.append((email, password, settings))
            raise exc

    real_get = reg.get
    monkeypatch.setattr(reg, "get",
                        lambda pid: _FakeQwen() if pid == "qwen" else real_get(pid))
    return calls


def _spy_deepseek_driver(monkeypatch, token: str = "ds-token") -> list:
    """The shipped DeepSeek browser driver, spied on at its import site."""
    import deepseaport.providers.deepseek.auth as ds_auth

    ran: list = []
    monkeypatch.setattr(ds_auth, "obscura_login_token",
                        lambda e, p, s: ran.append((e, p)) or (token, False, ""))
    return ran


@pytest.mark.parametrize("exc, expected", [
    (NotImplementedError(), "NotImplementedError"),
    (RuntimeError("signin rejected"), "signin rejected"),
])
def test_non_deepseek_login_failure_never_runs_the_deepseek_driver(
        monkeypatch, tmp_path, capsys, exc, expected):
    """opencode raises by design, a failed Qwen signin raises: both must be
    REPORTED (Evidence: + exit 3) instead of silently becoming a DeepSeek
    browser login whose token is filed under qwen."""
    _raising_login(monkeypatch, exc)
    ran = _spy_deepseek_driver(monkeypatch)
    # Flag-less: settings.provider drives the pid, which is the `deepseaport
    # login` form a qwen-configured deployment (and the TUI's auto-login
    # settings) reaches the wrapper with.
    cfg = write_config(tmp_path, {"keys": [], "provider": "qwen", "accounts": []})

    rc = cli_main(["--config", cfg, "login", "--email", "a@x.com",
                   "--password", "pw"])

    assert rc == 3
    assert ran == [], "the ~30s DeepSeek browser login ran for a qwen login"
    out = capsys.readouterr().out
    assert "login failed" in out and expected in out
    assert "Evidence:" in out
    assert "token saved" not in out  # it never authenticated anything
    assert on_disk(cfg)["accounts"] == [], "a DeepSeek token was stored under qwen"


def test_unregistered_non_deepseek_provider_is_reported_not_retried(
        monkeypatch):
    """``registry.get(pid) is None`` is the same refusal, not a DeepSeek login.

    Reached through the seam directly: ``config.load_settings`` normalizes an
    unknown ``provider`` back to deepseek with a warning, so a settings value is
    never the door here.
    """
    import deepseaport.cli as cli_mod
    import deepseaport.core.registry as reg

    real_get = reg.get
    monkeypatch.setattr(reg, "get",
                        lambda pid: None if pid == "qwen" else real_get(pid))
    ran = _spy_deepseek_driver(monkeypatch)

    token, banned, detail = cli_mod._obscura_login_token(
        "a@x.com", "pw", settings(provider="qwen"), "qwen")

    assert (token, banned) == (None, False)
    assert "qwen login failed" in detail and "no provider registered" in detail
    assert ran == []


def test_deepseek_login_still_drives_obscura_and_keeps_the_shim_identity(
        monkeypatch):
    """The pid=="deepseek" path must stay the direct browser driver.

    Two contracts in one: the driver is called with the historical 3-arg form,
    and ``_obscura_login_token`` is still the object ``_CANONICAL_LOGIN_DRIVER``
    points at -- ``cmd_login`` compares identity to decide whether its
    empty-credential risk guard applies, so a wrapper here would disable it
    silently.
    """
    import deepseaport.cli as cli_mod

    ran = _spy_deepseek_driver(monkeypatch, token="ds-token")

    assert cli_mod._obscura_login_token is cli_mod._CANONICAL_LOGIN_DRIVER
    assert cli_mod._obscura_login_token("a@x.com", "pw",
                                        settings(provider="deepseek")) == (
        "ds-token", False, "")
    assert ran == [("a@x.com", "pw")]


def test_non_deepseek_login_success_is_still_returned_and_stored(
        monkeypatch, tmp_path):
    """Control: the fix must not cost the working registry login its token."""
    import deepseaport.core.registry as reg

    class _FakeQwen:
        def login(self, email, password, settings):  # noqa: A002
            return ("qwen-jwt", False, "")

    real_get = reg.get
    monkeypatch.setattr(reg, "get",
                        lambda pid: _FakeQwen() if pid == "qwen" else real_get(pid))
    ran = _spy_deepseek_driver(monkeypatch)
    cfg = write_config(tmp_path, {"keys": [], "provider": "qwen", "accounts": []})

    assert cli_main(["--config", cfg, "login", "--email", "q@x.com",
                     "--password", "pw"]) == 0

    acc = on_disk(cfg)["accounts"][0]
    assert (acc["token"], acc["provider"]) == ("qwen-jwt", "qwen")
    assert ran == []


# F13: a fresh token must not read as "usable" while a persisted ban survives --


def test_relogin_over_a_persisted_ban_says_so_and_keeps_the_ban(
        monkeypatch, tmp_path, capsys):
    """The record keeps ``banned``/``banned_until`` (``_upsert_account`` only
    writes the token) and the pool skips it, so "token saved" alone is a lie;
    the operator needs the one command that clears it -- and no auto-clear,
    because a suspended account can legitimately complete a login."""
    import deepseaport.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_obscura_login_token",
                        lambda e, p, s: ("fresh-tok", False, ""))
    cfg = write_config(tmp_path, {"keys": [], "accounts": [
        {"email": "a@x.com", "token": "old-tok", "provider": "deepseek",
         "banned": True, "banned_until": 0.0},
    ]})

    rc = cli_main(["--config", cfg, "login", "--email", "a@x.com",
                   "--password", "pw"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "login ok, token saved" in out
    assert "ban" in out.lower() and "accounts unblock a@x.com" in out
    acc = on_disk(cfg)["accounts"][0]
    assert acc["token"] == "fresh-tok"
    assert acc["banned"] is True, "the persisted ban was auto-cleared"


def test_relogin_over_a_clean_account_prints_no_ban_line(
        monkeypatch, tmp_path, capsys):
    """Control: nothing to warn about, nothing printed."""
    import deepseaport.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_obscura_login_token",
                        lambda e, p, s: ("fresh-tok", False, ""))
    cfg = write_config(tmp_path, {"keys": [], "accounts": [
        {"email": "a@x.com", "token": "old-tok", "provider": "deepseek"},
    ]})

    assert cli_main(["--config", cfg, "login", "--email", "a@x.com",
                     "--password", "pw"]) == 0
    assert "unblock" not in capsys.readouterr().out


def test_banned_login_with_a_token_keeps_its_banned_branch(
        monkeypatch, tmp_path, capsys):
    """The widened success path must not swallow the driver's OWN ban report:
    login_banned still wins, with its exit 3 and its persisted markers."""
    import deepseaport.accounts as acc_mod
    import deepseaport.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_obscura_login_token",
                        lambda e, p, s: ("tok-banned", True, "suspended"))
    monkeypatch.setattr(acc_mod, "check_ban_for_token",
                        lambda token, timeout=12, provider="deepseek": (True, None))
    cfg = write_config(tmp_path, {"keys": [], "accounts": []})

    assert cli_main(["--config", cfg, "login", "--email", "b@x.com",
                     "--password", "pw"]) == 3
    out = capsys.readouterr().out
    assert "BANNED" in out
    assert on_disk(cfg)["accounts"][0]["banned"] is True
