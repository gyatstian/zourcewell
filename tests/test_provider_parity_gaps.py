"""Cross-provider parity gaps: the places where provider-neutral code had one
provider's semantics baked in (usage numbers, the 403 mapping), plus the
logging/compat-surface contracts. Offline only, no network.

Each test names the defect it reproduces; the pre-change failure is quoted in
the docstring. Providers exercised here: deepseek (reference) and qwen.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from deepseaport.core import transport as T
from deepseaport.core.openai import build_usage, estimate_tokens
from deepseaport.providers.deepseek import client as DS
from deepseaport.providers.qwen import client as QC
from deepseaport.providers.qwen import completion_service as QCS
from deepseaport.providers.qwen import sse_parser as QSP


# --- F2: the usage number a provider reports must be COMPLETION tokens ------


def test_qwen_usage_event_is_completion_count():
    """F2. Pre-change: the parser preferred ``total_tokens``.

    README live sample: input 832 / output 1 / total 833 -> the old parser
    emitted "833", which build_usage then reported as completion_tokens.
    """
    line = ('data: {"choices":[{"delta":{"phase":"answer","content":"Hello",'
            '"status":"typing"}}],"usage":{"input_tokens":832,"output_tokens":1,'
            '"characters":0,"total_tokens":833}}')
    evs = list(QSP.parse_sse_line(line))
    assert [(e.kind, e.text) for e in evs] == [("content", "Hello"), ("usage", "1")]


def test_qwen_live_sample_builds_approx_one_completion_token():
    """The end-to-end symptom: 833 phantom completion tokens on a 1-token reply."""
    line = ('data: {"choices":[{"delta":{"phase":"answer","content":"Hello",'
            '"status":"typing"}}],"usage":{"input_tokens":832,"output_tokens":1,'
            '"characters":0,"total_tokens":833}}')
    usage_total = max(int(e.text) for e in QSP.parse_sse_line(line)
                      if e.kind == "usage")
    prep = {"prompt": "x" * 100, "model": "qwen3.7-plus"}
    usage = build_usage(prep, "Hello", "", usage_total)
    # Pre-change: {"prompt_tokens": 25, "completion_tokens": 833, "total_tokens": 858}
    assert usage["completion_tokens"] == 1
    assert usage["prompt_tokens"] == estimate_tokens("x" * 100)
    assert usage["total_tokens"] == usage["prompt_tokens"] + 1


def test_stream_usage_chunk_reports_completion_count():
    """Same numbers on the streaming path (stream_options.include_usage)."""
    from deepseaport.core import engine

    prep = {"prompt": "y" * 40, "model": "qwen3.7-plus"}
    raw = engine.usage_chunk(
        "chatcmpl-1", 1, "qwen3.7-plus", prep, "Hello", "", 1).decode("utf-8")
    chunk = json.loads(raw.split("data: ", 1)[1])
    assert chunk["usage"]["completion_tokens"] == 1
    assert chunk["usage"]["total_tokens"] == chunk["usage"]["prompt_tokens"] + 1


def test_build_usage_treats_its_input_as_completion_tokens():
    """The contract F2 depends on: usage_total IS the completion count.

    (Correct for DeepSeek's accumulated_token_usage; wrong for any provider
    whose reported total is input+output, which is why the Qwen parser must
    emit output_tokens.)
    """
    prep = {"prompt": "p" * 80}
    assert build_usage(prep, "", "", 42)["completion_tokens"] == 42
    # Falsy / garbage input falls back to the len//4 estimate, never raises.
    assert build_usage(prep, "a" * 40, "", 0)["completion_tokens"] == 10
    assert build_usage(prep, "a" * 40, "", None)["completion_tokens"] == 10
    assert build_usage(prep, "a" * 40, "", "nope")["completion_tokens"] == 10
    assert build_usage(prep, "", "", 7)["total_tokens"] == (
        estimate_tokens("p" * 80) + 7)


def test_qwen_usage_all_three_key_shapes():
    """Live (input/output/total), classic (prompt/completion/total), total-only."""
    cases = {
        '{"input_tokens":5,"output_tokens":7,"total_tokens":12}': "7",
        '{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12}': "7",
        '{"input_tokens":5,"total_tokens":12}': "12",
        '{"completion_tokens":0,"total_tokens":9}': "0",
    }
    for usage, want in cases.items():
        line = 'data: {"choices":[{"delta":{"content":"x"}}],"usage":%s}' % usage
        evs = [(e.kind, e.text) for e in QSP.parse_sse_line(line)]
        assert evs == [("content", "x"), ("usage", want)], usage


def test_deepseek_usage_still_feeds_accumulated_total():
    """DeepSeek reports an accumulated output total, so it passes its own
    usage_total straight through — unchanged by the Qwen-side fix."""
    from deepseaport.providers.deepseek import sse_parser as DSP

    line = 'data: {"p":"response/accumulated_token_usage","v":99}'
    kinds = [e for e in DSP.StreamParser().feed(line) if e.kind == "usage"]
    assert kinds, "expected a usage event from the deepseek parser"
    prep = {"prompt": "q" * 8}
    assert build_usage(prep, "hi", "", int(kinds[0].text))["completion_tokens"] == 99


# --- F3: the 403 meaning is per-provider ------------------------------------


class _Resp:
    """Minimal response double (status + drained body)."""

    def __init__(self, status_code=403, text="", content_type="text/html"):
        self.status_code = status_code
        self._text = text
        self.headers = {"content-type": content_type}
        self.closed = False

    @property
    def text(self):
        return self._text

    def iter_lines(self):
        return iter(())

    def close(self):
        self.closed = True


def _client_events(client, resp, monkeypatch):
    class _S:
        def post(self, *a, **k):
            return resp

    monkeypatch.setattr(client, "_SESSION", _S())
    return list(client.stream_completion({"h": "1"}, {"prompt": "hi"}))


def test_403_parity_deepseek_waf_vs_qwen_plain(monkeypatch):
    """F3. Pre-change: both providers emitted HTTP_403_WAF for a 403, so a real
    Qwen 403 (suspension body) was classified as WAF and never cooled down."""
    deepseek = _client_events(DS, _Resp(403, "blocked"), monkeypatch)
    assert [e.code for e in deepseek] == ["HTTP_403_WAF"]
    qwen = _client_events(QC, _Resp(403, "account suspended"), monkeypatch)
    assert [e.code for e in qwen] == ["HTTP_403"]
    assert [e.code for e in qwen] != ["HTTP_403_WAF"]


def test_403_mapping_is_injectable_and_only_403_is_mapped():
    """F3, unit level: the 403 code comes from the INJECTED table, and no other
    status is ever mapped.

    The pre-change version of this test looped over non-403 statuses only and
    so proved nothing about 403 -- it asserted the function's own
    ``HTTP_{status}`` fallback. It also survived a revert of the F3 fix, which
    is why it is being replaced rather than kept. The 403 half below fails the
    moment `http_error_code` goes back to a hardcoded ``HTTP_403_WAF``.
    """
    # The default (no table) is DeepSeek: 403 is the AWS-WAF challenge.
    assert T.http_error_code(403) == "HTTP_403_WAF"
    assert T.http_error_code(403, None) == "HTTP_403_WAF"
    assert T.http_error_code(403, T._DEEPSEEK_FORBIDDEN) == "HTTP_403_WAF"

    # Qwen passes an EMPTY table, which is how it opts OUT of the WAF label.
    # This is the line that a hardcoded 403 mapping cannot satisfy.
    assert T.http_error_code(403, {}) == "HTTP_403"
    assert T.http_error_code(403, {}) != T.http_error_code(403)

    # A provider that names its own 403 code is honoured verbatim.
    assert T.http_error_code(403, {403: "CUSTOM_403"}) == "CUSTOM_403"

    # Only 403 is a mapping key: every other status falls through to the
    # generic form under every table, so the injection cannot accidentally
    # reclassify a neighbouring status.
    for status in (400, 401, 404, 422, 429, 500, 502, 503):
        assert T.http_error_code(status) == f"HTTP_{status}"
        assert T.http_error_code(status, {}) == f"HTTP_{status}"
        assert T.http_error_code(status, {403: "CUSTOM_403"}) == f"HTTP_{status}"

    # Junk input degrades instead of raising: a non-int status, a None table
    # and a non-dict table all produce the generic code.
    assert T.http_error_code("nope", {}) == "HTTP_nope"
    assert T.http_error_code(None, {}) == "HTTP_None"
    assert T.http_error_code(403, None) == "HTTP_403_WAF"


def test_qwen_403_suspension_end_to_end_verdict(monkeypatch):
    """A Qwen 403 whose body says the account is suspended -> BAN, not WAF.

    Pre-change the code was HTTP_403_WAF and classify_error matched the WAF
    branch on the "403" substring, so the account got re-primes and a 502
    instead of a 403 + cooldown + failover.
    """
    evs = _client_events(QC, _Resp(403, "your account has been suspended"), monkeypatch)
    err = f"{evs[0].code}: {evs[0].message}"
    assert QCS.classify_error(err) is QCS.RetryKind.BAN
    assert QCS._TERMINAL_HANDLERS[QCS.RetryKind.BAN] is QCS._terminal_ban


def test_qwen_waf_is_200_html_not_403(monkeypatch):
    """Qwen's real WAF shape (200 + text/html) keeps its own detection."""
    evs = _client_events(QC, _Resp(200, "<html>hi</html>",
                                   content_type="text/html"), monkeypatch)
    assert [e.code for e in evs] == ["WAF_CHALLENGE"]
    assert QCS.classify_error("WAF_CHALLENGE: aliyun_waf_aa") is QCS.RetryKind.WAF


# --- F10: a provider reads its OWN models.py ---------------------------------


def test_qwen_provider_reads_its_own_models_module(monkeypatch):
    """F10. provider.models() returns its own MODELS_SPEC, so a filled-in
    models.py shows up through the provider and the catalogue never stays
    permanently empty.
    """
    import importlib

    from deepseaport.core.provider import ModelSpec

    mod = importlib.import_module("deepseaport.providers.qwen.provider")
    prov = mod.create_provider()
    assert prov.id == "qwen"
    spec = ModelSpec(id="qwen-probe", thinking=True, search=False,
                     model_type="default", owned_by="qwen")
    monkeypatch.setattr(mod, "MODELS_SPEC", {"qwen-probe": spec,
                                             "qwen-probe-2": spec})
    assert prov.models() == {"qwen-probe": spec, "qwen-probe-2": spec}
    assert prov.tuple_spec()["qwen-probe"] == (True, False, "default")
    # Copy, not the module dict itself (callers may not mutate the catalogue).
    got = prov.models()
    got["injected"] = spec
    assert "injected" not in prov.models()


def test_provider_modules_are_import_safe():
    """Importing any provider module must not touch the network, the
    filesystem or the process env.

    Any provider module doing I/O at import time is a real bug, and
    the socket guard below is what turns it into a test failure rather than a
    slow suite and an unexplained egress from the test runner.
    """
    code = textwrap.dedent(
        """
        import os, socket, sys
        sys.path.insert(0, %r)
        os.environ["DS_PARITY_SENTINEL"] = "1"
        real_conn = socket.socket.connect
        def _boom(self, *a, **k):
            raise AssertionError("import must not open a socket")
        socket.socket.connect = _boom
        import deepseaport.providers.qwen.provider as q
        import deepseaport.providers.qwen.models as qm
        import deepseaport.providers.deepseek.provider as d
        import deepseaport.providers.deepseek.models as dm
        import deepseaport.providers.qwen.protocol
        import deepseaport.providers.qwen.sse_parser
        import deepseaport.providers.qwen.client
        import deepseaport.providers.qwen.completion_service
        socket.socket.connect = real_conn
        assert q.QwenProvider().models() == dict(qm.MODELS_SPEC)
        assert d.DeepSeekProvider().models() == dict(dm.MODELS_SPEC)
        print("OK")
        """
    ) % str(_SRC_DIR)
    out = _run_py(code)
    assert out.strip().endswith("OK"), out


# --- compat surface ---------------------------------------------------------


def test_importing_registry_does_not_pull_in_server():
    code = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, %r)
        import deepseaport.core.registry  # noqa: F401
        print("deepseaport.server" in sys.modules)
        """
    ) % str(_SRC_DIR)
    out = _run_py(code)
    assert out.strip().splitlines()[-1] == "False", out


def test_server_reexports_still_resolve():
    """server exposes only what it owns: entry point, catalogue, 3 seams.

    The 57-name FROZEN COMPAT block is gone -- every one of those names is
    imported from the module that defines it. What is pinned here is that the
    thin surface still resolves, that the three seams the route layer looks up
    through this module are the REAL objects (not copies), and that the
    F11c logging move did not shadow anything."""
    import deepseaport.core.openai as core_openai
    import deepseaport.core.tools as core_tools
    import deepseaport.providers.deepseek.completion_service as ds_completion
    import deepseaport.server as S
    import deepseaport.server_routes as sr

    for name in ("MODELS", "active_models", "create_app",
                 "register_routes", "_pool", "_openai_response",
                 "parse_tool_calls", "Settings", "load_settings",
                 "apply_log_level", "lifespan", "configure_logging"):
        assert getattr(S, name, None) is not None, name
    # Seams must be the same objects, not aliases that can drift.
    assert S._openai_response is core_openai._openai_response
    assert ds_completion._openai_response is S._openai_response
    assert S.parse_tool_calls is core_tools.parse_tool_calls
    assert S._pool is sr._pool


# --- F11c: importing/creating the app must not reconfigure global logging ---


_IMPORT_SIDE_EFFECT_PROBE = textwrap.dedent(
    """
    import logging, sys
    sys.path.insert(0, %r)
    root = logging.getLogger()
    before = (root.level, len(root.handlers))
    import deepseaport.server  # noqa: F401
    after = (root.level, len(root.handlers))
    print(before)
    print(after)
    """
)


def test_importing_server_has_no_logging_side_effect():
    """F11c. Pre-change: a module-level logging.basicConfig ran on import, so
    the root logger level and handler list changed for every importer (in a
    fresh interpreter: level 30->20, 0->1 handlers)."""
    out = _run_py(_IMPORT_SIDE_EFFECT_PROBE % str(_SRC_DIR))
    before, after = out.strip().splitlines()[-2:]
    assert before == after, f"import mutated global logging: {before} -> {after}"
    assert before == "(30, 0)"


def test_creating_app_leaves_root_logger_level_untouched(monkeypatch):
    """create_app calls basicConfig, which no-ops once the root is configured,
    so an embedding host (or chat_ui, which forces the root to ERROR) keeps its
    own level. The caller can still opt in explicitly."""
    from deepseaport.server import configure_logging, create_app
    from deepseaport.config import Settings

    root = logging.getLogger()
    saved_level, saved_handlers = root.level, list(root.handlers)
    calls: list[dict] = []
    monkeypatch.setattr(logging, "basicConfig",
                        lambda **kw: calls.append(kw))
    try:
        # Simulate a host that already configured logging (chat_ui's case).
        root.setLevel(logging.ERROR)
        create_app(Settings(accounts=[], provider="deepseek", config_path=""))
        assert root.level == logging.ERROR
        assert len(calls) == 1
        assert calls[0]["format"] == (
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        # Opt-in: force replaces the existing configuration.
        calls.clear()
        configure_logging(force=True)
        assert calls[0]["force"] is True
    finally:
        root.setLevel(saved_level)
        root.handlers[:] = saved_handlers


def test_chat_log_survives_app_creation(monkeypatch):
    """chat_ui silences the root BEFORE create_app; that must still hold."""
    from deepseaport import chat_repl
    from deepseaport.config import Settings
    from deepseaport.server import create_app

    root = logging.getLogger()
    saved_level, saved_handlers = root.level, list(root.handlers)
    try:
        root.setLevel(logging.ERROR)
        saved = chat_repl._silence_chat_logs()
        try:
            assert root.level == logging.ERROR
            create_app(Settings(accounts=[], provider="deepseek", config_path=""))
            assert root.level == logging.ERROR
        finally:
            chat_repl._restore_chat_logs(saved)
    finally:
        root.setLevel(saved_level)
        root.handlers[:] = saved_handlers


def test_configure_logging_is_a_noop_on_a_configured_root(monkeypatch):
    """`basicConfig` must short-circuit on an already-configured root.

    The "already configured" precondition is established HERE, by attaching a
    handler if the run has none. It used to be `if not root.handlers:
    pytest.skip(...)` -- a dead early-out, because pytest's own logging plugin
    attaches a root handler for the whole session, so the branch never fired.
    That is worse than no branch: had it ever fired, the test would have
    silently skipped instead of failing, and the property it names (a no-op
    `configure_logging` on a configured root) would go unpinned.
    """
    from deepseaport import server as S

    root = logging.getLogger()
    saved_level, saved_handlers = root.level, list(root.handlers)
    sentinel = logging.NullHandler()
    added_here = not root.handlers
    if added_here:
        root.addHandler(sentinel)
    try:
        root.setLevel(logging.WARNING)
        calls: list[dict] = []
        monkeypatch.setattr(
            logging, "basicConfig",
            lambda **kw: (calls.append(kw), _stdlib_basic_config(**kw))[1])
        S.configure_logging()
        assert calls, "configure_logging must delegate to basicConfig"
        assert root.level == logging.WARNING, \
            "configure_logging overwrote the level of a configured root"
        # Full restore: level AND handler list, so a failure above cannot leak
        # logging state into the rest of the session.
    finally:
        root.setLevel(saved_level)
        root.handlers[:] = saved_handlers
        if added_here:
            root.removeHandler(sentinel)


def _stdlib_basic_config(**kw) -> None:
    """Mirror logging.basicConfig: return early when the root is configured."""
    root = logging.getLogger()
    if kw.get("force"):
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in kw.get("handlers") or []:
            root.addHandler(h)
    elif root.handlers:
        return
    else:
        root.addHandler(logging.StreamHandler())
    level = kw.get("level")
    if level is not None:
        root.setLevel(level)


# --- openai.py dead-code cleanup (F11a/b) ------------------------------------


def test_engine_wants_usage_is_the_usage_consumer():
    """F11b: `openai.should_emit_usage` was a duplicate of this predicate; the
    consumer is core.engine.wants_usage and it must keep reading the request."""
    from deepseaport.core import engine
    assert engine.wants_usage({"stream_options": {"include_usage": True}}) is True
    assert engine.wants_usage({}) is False


def test_openai_response_usage_is_built_once():
    """F11a: build_usage recomputes both counts; the duplicate pt/ct locals
    were dead stores."""
    from deepseaport.core.openai import _openai_response

    prep = {"model": "deepseek-flash", "prompt": "z" * 40, "tools": None}
    out = _openai_response(prep, {"content": "hello", "thinking": "",
                                  "usage_total": 9})
    assert out["usage"] == build_usage(prep, "hello", "", 9)
    assert out["usage"]["completion_tokens"] == 9
    # Without an upstream total the estimate is used, once.
    out2 = _openai_response(prep, {"content": "hello", "thinking": "",
                                   "usage_total": 0})
    assert out2["usage"] == build_usage(prep, "hello", "", 0)


# --- helpers ----------------------------------------------------------------


_SRC_DIR = Path(__file__).resolve().parents[1] / "src"


def _run_py(code: str) -> str:
    """Run `code` in a fresh interpreter against THIS tree (no pytest root
    logger, no ambient config), return stdout."""
    env = dict(os.environ)
    env.pop("PYTEST_CURRENT_TEST", None)
    env["PYTHONPATH"] = str(_SRC_DIR)
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        timeout=120, env=env, cwd=str(_SRC_DIR.parent))
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout
