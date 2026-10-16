"""Gap tests for tool-parser bounds and provider tool-surface contracts.

Groups:

* ``MAX_PARAMS_PER_INVOKE`` must bound how many parameters are *scanned*,
  never the value range of the last accepted one (H1).
* The opencode per-message tool map must not re-grant native bash on top of
  the managed agent's ``bash: "deny"`` permission (H2).
* Dead provider surface stays deleted (H3): no dead `*_models` diagnostics
  helper, no dead stop helper; `snapshot_verdict` is live and must remain.
* ``MAX_ARGS_CHARS`` is an import-time snapshot, not a live env read, and
  the request path honours ``Settings.tool_args_max_chars`` instead (H4).

Marker-based construction (chr(60)/chr(62)/chr(0xFF5C)) is used throughout
so the DSML dialects are assembled without literal tags in this file, the
same trick the existing dialect tests in test_tools_gaps.py use.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

from deepseaport.core import tools as T
from deepseaport.core.tools import common as TC
from deepseaport.providers.deepseek import login_policy as LP
from deepseaport.providers.qwen import client as QC


_LT = chr(60)
_GT = chr(62)
_FW = chr(0xFF5C)
_CAP = TC.MAX_PARAMS_PER_INVOKE
_READ = [{"type": "function", "function": {"name": "read"}}]


def _invoke(n, marker="||DSML||", wrapper=True, named_close=False):
    """Invoke carrying N `<parameter name="pI" string="true">vI</parameter>`."""
    generic = _LT + "/" + _GT
    pclose = (_LT + "/" + marker + "parameter>") if marker else generic
    iclose = (_LT + "/" + marker + "invoke>") if marker else generic
    parts = []
    if wrapper:
        parts.append(_LT + marker + "tool_calls" + _GT)
    parts.append(_LT + marker + 'invoke name="read"' + _GT)
    for i in range(n):
        parts.append(_LT + marker + f'parameter name="p{i}" string="true"' + _GT)
        parts.append(f"v{i}" + pclose)
    parts.append(iclose if named_close else generic)
    if wrapper:
        parts.append(_LT + "/" + marker + "tool_calls" + _GT)
    return "".join(parts)


# --- H1: the per-invoke parameter cap bounds the scan, not the value range --

@pytest.mark.parametrize("n", [_CAP, _CAP + 1, _CAP + 6, 200])
def test_param_cap_keeps_every_accepted_value_intact(n):
    """The cap-truncated reply keeps the first `_CAP` values byte for byte."""
    calls, _ = T.parse_tool_calls(_invoke(n), _READ)
    assert calls is not None and len(calls) == 1
    args = json.loads(calls[0]["function"]["arguments"])
    assert len(args) == _CAP
    # Keys, order and values: the 64th parameter is the one that used to
    # come back as '' because the cap narrowed its value_limit.
    assert list(args) == [f"p{i}" for i in range(_CAP)]
    assert args == {f"p{i}": f"v{i}" for i in range(_CAP)}
    assert args[f"p{_CAP - 1}"] == f"v{_CAP - 1}"
    # Parameters past the cap are dropped, not merged into the last value.
    assert not any(k.startswith("p") and k[1:].isdigit()
                   and int(k[1:]) >= _CAP for k in args)


@pytest.mark.parametrize("marker,forgiving,wrapped,named_close", [
    ("||DSML||", False, True, False),      # canonical DSML
    ("", False, False, False),             # bare <invoke>/<parameter>
    ("||DSML||", False, False, True),      # explicit </invoke> close
    (chr(0xFF5C) * 2 + "DSML" + chr(0xFF5C) * 2, False, True, False),
    (chr(0xFF5C) + "DSML" + chr(0xFF5C), True, False, True),  # forgiving
    # the same single-pipe marker with a generic close: a distinct parse
    # path from the named close above, so it stays its own row.
    (chr(0xFF5C) + "DSML" + chr(0xFF5C), True, False, False),
])
@pytest.mark.parametrize("n", [_CAP, _CAP + 1, _CAP + 6, 200])
def test_param_cap_bound_holds_in_every_dialect(marker, forgiving, wrapped,
                                                named_close, n):
    calls, _ = T.parse_tool_calls(
        _invoke(n, marker, wrapper=wrapped, named_close=named_close),
        _READ, forgiving=forgiving)
    assert calls is not None and len(calls) == 1
    args = json.loads(calls[0]["function"]["arguments"])
    assert len(args) == _CAP
    assert args[f"p{_CAP - 1}"] == f"v{_CAP - 1}"
    assert f"p{_CAP}" not in args


@pytest.mark.parametrize("n", [0, 1, 2, 17, _CAP - 1, _CAP])
def test_under_the_cap_nothing_changes(n):
    """Below/at the cap the parse is the plain whole-block read."""
    calls, rest = T.parse_tool_calls(_invoke(n), _READ)
    assert calls is not None
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {f"p{i}": f"v{i}" for i in range(n)}
    assert rest == ""


def test_param_cap_is_the_documented_hardcoded_64():
    """Pins the cap's VALUE and its shape, not its configurability.

    This was ``test_cap_is_configured_not_hardcoded``, which was wrong twice:
    the cap is a plain module constant (``MAX_PARAMS_PER_INVOKE = 64``, with
    no ``_int_env`` behind it, unlike ``MAX_ARGS_CHARS`` and
    ``MAX_BARE_OBJECTS``), and its second assert was ``X is X`` -- a
    module-level alias compared against itself, true by construction and
    therefore immune to a revert.

    What is worth pinning is that the constant is a plain ``int`` and that
    every dialect honours it, so a change has to be a deliberate edit to
    ``common.py`` and not an accident. The behavioural coverage is the
    parametrised H1 tests above, which derive their cases from this constant.
    """
    assert TC.MAX_PARAMS_PER_INVOKE == 64
    assert type(TC.MAX_PARAMS_PER_INVOKE) is int
    # No env knob behind it: unlike the sibling caps, this one is a literal.
    import inspect as _inspect
    source = _inspect.getsource(TC)
    assert "MAX_PARAMS_PER_INVOKE = 64" in source, \
        "the cap became configurable (or moved); update this test and the docs"


# --- H2: native tools must stay bounded by the managed permission map ------
# opencode v2 removed the per-message tool map, so there is no override left
# that could re-grant a native tool: this map is the only authority.

def test_managed_permission_map_still_denies_bash():
    from deepseaport.providers.opencode import managed_serve as MS

    cfg = json.loads(MS.managed_config_content())
    assert cfg["permission"]["bash"] == "deny"
    assert cfg["permission"]["read"] == "allow"


# --- H3: the live provider helper stays live ------------------------------


def test_qwen_stop_url_constant_survives_as_documented_protocol():
    """The endpoint constant is part of the documented upstream surface even
    though the unused POST helper is gone."""
    from deepseaport.providers.qwen import protocol as QP

    assert "CHAT_COMPLETIONS_STOP_URL" in QP.__all__
    assert QP.CHAT_COMPLETIONS_STOP_URL.endswith("/chat/completions/stop")
    # Sibling best-effort helpers and their shared query builder stay live.
    assert callable(QC.delete_chat)
    assert callable(QC.fetch_models)
    assert QC._with_chat_id("u", {"chat_id": "a b"}) == "u?chat_id=a%20b"


def test_snapshot_verdict_is_live_and_shared():
    """Kept: real_browser.RealBrowser is its real caller (login_policy is the
    single source of truth for both browser drivers)."""
    verdict = LP.snapshot_verdict
    assert verdict({"body": "Please complete the security check",
                    "captcha_visible": False}) == ("captcha", "security check")
    assert verdict({"body": "hello", "captcha_visible": True}) == (
        "captcha", "visible challenge overlay")
    assert verdict({"body": "wrong password"})[0] == "credential"
    assert verdict(None) == ("", "")

    from deepseaport.providers.deepseek import real_browser

    src = pathlib.Path(real_browser.__file__).read_text(encoding="utf-8")
    assert "snapshot_verdict" in src, "the CDP driver must keep calling it"


# --- H4: MAX_ARGS_CHARS is an import-time snapshot, not a live env read ----

_SUBPROCESS_PRELUDE = textwrap.dedent(
    """
    import json, os, sys
    import deepseaport
    from deepseaport.core.tools import common as TC
    from deepseaport import config
    print(json.dumps({
        "tree": os.path.dirname(deepseaport.__file__),
        "max_args_chars": TC.MAX_ARGS_CHARS,
        "repair_budget": TC._LENIENT_REPAIR_BUDGET,
        "settings": config.load_settings().tool_args_max_chars,
    }))
    """
)


def _subprocess_env(**overrides):
    """Env for a child that must import THIS checkout's src/."""
    import deepseaport

    src_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(deepseaport.__file__))))
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_root + (os.pathsep + existing if existing else "")
    env.update({k: str(v) for k, v in overrides.items()})
    return env


def _run_probe(**env_overrides):
    proc = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_PRELUDE],
        capture_output=True, text=True, timeout=120,
        env=_subprocess_env(**env_overrides), cwd=os.getcwd())
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    return data


def test_max_args_chars_snapshots_env_set_before_import():
    """Set before import: the snapshot picks it up (documented semantics)."""
    data = _run_probe(DEEPSEAPORT_MAX_ARGS_CHARS="12345")
    assert data["max_args_chars"] == 12345
    assert data["repair_budget"] == 12345 + 16_384
    # load_settings reads the same variable per call, so the two agree here.
    assert data["settings"] == 12345


def test_max_args_chars_ignores_env_set_after_import():
    """Set AFTER import: the snapshot stays put, Settings follows the env.

    This is the divergence the comment in common.py documents. If someone
    later makes MAX_ARGS_CHARS live, this test fails and the comment (and
    parser/__init__.py + prompt.py, which are outside this change's scope)
    must be revisited in the same commit.
    """
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(
            """
            import json, os
            from deepseaport.core.tools import common as TC
            before = TC.MAX_ARGS_CHARS
            os.environ["DEEPSEAPORT_MAX_ARGS_CHARS"] = "777"
            import deepseaport.config as config
            print(json.dumps({
                "before": before,
                "after": TC.MAX_ARGS_CHARS,
                "settings": config.load_settings().tool_args_max_chars,
            }))
            """
        )],
        capture_output=True, text=True, timeout=120,
        env=_subprocess_env(), cwd=os.getcwd())
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["before"] == 200_000
    assert data["after"] == 200_000, "MAX_ARGS_CHARS must stay a snapshot"
    assert data["settings"] == 777, "load_settings must read the env per call"


def test_max_args_chars_default_when_env_absent():
    data = _run_probe(DEEPSEAPORT_MAX_ARGS_CHARS="")
    assert data["max_args_chars"] == 200_000
    assert data["settings"] == 200_000


def test_explicit_max_args_chars_wins_over_the_snapshot():
    """The live path: the request layer passes Settings.tool_args_max_chars."""
    tools = [{"type": "function", "function": {"name": "write"}}]
    payload = json.dumps({"path": "a.txt", "content": "x" * 4_000})
    text = json.dumps({"tool_calls": [{"function": {"name": "write",
                                                    "arguments": payload}}]})
    # 3k limit: the 4k payload is dropped.
    assert T.parse_tool_calls(text, tools, max_args_chars=3_000)[0] is None
    # 8k limit: same text is accepted, regardless of the snapshot.
    calls, _ = T.parse_tool_calls(text, tools, max_args_chars=8_000)
    assert calls is not None
    assert len(json.loads(calls[0]["function"]["arguments"])["content"]) == 4_000


def test_parse_tool_calls_uses_the_snapshot_when_no_limit_is_passed():
    """No explicit limit -> the import-time default is what applies."""
    tools = [{"type": "function", "function": {"name": "write"}}]
    payload = json.dumps({"path": "a.txt", "content": "x" * 4_000})
    text = json.dumps({"tool_calls": [{"function": {"name": "write",
                                                    "arguments": payload}}]})
    calls, _ = T.parse_tool_calls(text, tools)
    if TC.MAX_ARGS_CHARS >= len(payload):
        assert calls is not None
    else:
        assert calls is None


def test_max_args_chars_name_is_importable_from_the_public_surface():
    import deepseaport.core.tools as TS

    assert TS.MAX_ARGS_CHARS == TC.MAX_ARGS_CHARS
    assert TC._int_env("DEEPSEAPORT_TEST_ABSENT_ARGS", 200_000) == 200_000
