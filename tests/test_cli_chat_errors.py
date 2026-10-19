"""F14: ``deepseaport chat`` printed a raw traceback as its whole UX.

``cmd_chat`` had two provider calls. The accountless branch (opencode ran with
``pool=None``) sat outside any ``try`` at all, and the account-holding branch
caught only ``(ValueError, HTTPException, TimeoutError)`` -- so a provider
engine failure (deepseek's create_session / pow RuntimeErrors, opencode's serve
failures, anything a helper raises) escaped to the terminal, and the frozen exe
then appended "Press Enter to close..." to the traceback.

Both now take ``_chat_fail``: one actionable line, exit 1, no traceback. The
HTTP and timeout wordings it already owned are pinned here so widening the catch
could not flatten them into the generic line.
"""

from __future__ import annotations

import types

import pytest
from support import write_config

from deepseaport.cli import main as cli_main


def _fake_bridge(monkeypatch, *, forbidden: bool = False):
    """create_app + ObscuraBridge stand-ins; the bridge only counts its warmups."""
    import deepseaport.core.obscura_bridge as bridge_mod
    import deepseaport.server as srv_mod

    monkeypatch.setattr(srv_mod, "create_app",
                        lambda settings=None: types.SimpleNamespace(
                            state=types.SimpleNamespace(bridge=None)))
    if forbidden:
        monkeypatch.setattr(bridge_mod, "ObscuraBridge",
                            lambda *a, **k: pytest.fail("no WAF bridge here"))
    else:
        class _FakeObscuraBridge:
            def __init__(self, *a, **k):
                self.state = types.SimpleNamespace(user_agent="ua", cookies={})

            def warmup(self, timeout=45):
                return {}

        monkeypatch.setattr(bridge_mod, "ObscuraBridge", _FakeObscuraBridge)
    monkeypatch.setattr(bridge_mod, "register_default_bridge", lambda *a, **k: None)


def _qwen_config(tmp_path) -> str:
    return write_config(tmp_path, {"keys": [], "accounts": [
        {"email": "a@x.com", "token": "qwen-jwt", "provider": "qwen"}]})


def _fail_from_qwen(monkeypatch, exc: BaseException):
    import deepseaport.accounts as acc_mod
    import deepseaport.providers.qwen.provider as qw_prov

    monkeypatch.setattr(acc_mod.AccountPool, "acquire",
                        lambda self, timeout=30, allow_failover=True: object())

    def _boom(self, *a, **k):
        raise exc

    monkeypatch.setattr(qw_prov.QwenProvider, "complete", _boom)


def test_chat_account_holding_failure_is_one_line_not_a_traceback(
        monkeypatch, tmp_path, capsys):
    _fake_bridge(monkeypatch)
    _fail_from_qwen(monkeypatch, RuntimeError("upstream exploded"))

    cfg = _qwen_config(tmp_path)
    rc = cli_main(["--config", cfg, "chat", "--provider", "qwen",
                   "--model", "qwen3.7-plus", "--prompt", "hi"])

    assert rc == 1
    out = capsys.readouterr().out
    assert "chat failed" in out and "upstream exploded" in out
    assert "Traceback" not in out
    assert len([ln for ln in out.splitlines() if "chat failed" in ln]) == 1


def test_chat_accountless_provider_failure_is_one_line_not_a_traceback(
        monkeypatch, tmp_path, capsys):
    """The opencode branch (no pool, no slot) had no ``try`` around it at all."""
    import deepseaport.providers.opencode.provider as oc_prov

    _fake_bridge(monkeypatch, forbidden=True)

    def _boom(self, *a, **k):
        raise RuntimeError("opencode serve died")

    monkeypatch.setattr(oc_prov.OpencodeProvider, "complete", _boom)

    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    rc = cli_main(["--config", cfg, "chat", "--provider", "opencode",
                   "--model", "opencode/muse-spark-1.3-contributor-free",
                   "--prompt", "hi"])

    assert rc == 1
    out = capsys.readouterr().out
    assert "chat failed" in out and "opencode serve died" in out
    assert "Traceback" not in out


def test_chat_keeps_the_http_exception_wording(monkeypatch, tmp_path, capsys):
    """Control for the widened catch: HTTPException still names its status."""
    from fastapi import HTTPException

    _fake_bridge(monkeypatch)
    _fail_from_qwen(monkeypatch, HTTPException(status_code=404,
                                               detail="model not found"))

    cfg = _qwen_config(tmp_path)
    rc = cli_main(["--config", cfg, "chat", "--provider", "qwen",
                   "--model", "qwen3.7-plus", "--prompt", "hi"])

    assert rc == 1
    assert "HTTP 404: model not found" in capsys.readouterr().out


def test_chat_keeps_the_busy_pool_wording(monkeypatch, tmp_path, capsys):
    """Control: a full pool still gets the actionable timeout line."""
    import deepseaport.accounts as acc_mod

    _fake_bridge(monkeypatch)

    def _timeout(self, timeout=30, allow_failover=True):
        raise TimeoutError("no slot")

    monkeypatch.setattr(acc_mod.AccountPool, "acquire", _timeout)

    cfg = _qwen_config(tmp_path)
    rc = cli_main(["--config", cfg, "chat", "--provider", "qwen",
                   "--model", "qwen3.7-plus", "--prompt", "hi"])

    assert rc == 1
    out = capsys.readouterr().out
    assert "timed out waiting for a free account slot" in out
    assert "Traceback" not in out


def test_chat_success_still_returns_0(monkeypatch, tmp_path, capsys):
    """Control for the other direction: the widened catch must not swallow a
    successful reply."""
    import deepseaport.accounts as acc_mod
    import deepseaport.providers.qwen.provider as qw_prov

    _fake_bridge(monkeypatch)
    monkeypatch.setattr(acc_mod.AccountPool, "acquire",
                        lambda self, timeout=30, allow_failover=True: object())
    monkeypatch.setattr(qw_prov.QwenProvider, "complete",
                        lambda self, *a, **k: {"content": "qwen-answer",
                                               "thinking": ""})

    cfg = _qwen_config(tmp_path)
    rc = cli_main(["--config", cfg, "chat", "--provider", "qwen",
                   "--model", "qwen3.7-plus", "--prompt", "hi"])

    assert rc == 0
    assert "qwen-answer" in capsys.readouterr().out
