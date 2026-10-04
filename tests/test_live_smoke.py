"""Live smoke suite: the only tests in this repo that can see an upstream drift.

Why this file exists
--------------------
``deepseaport`` reverse-engineers **undocumented** web APIs: DeepSeek
(``chat.deepseek.com`` behind AWS WAF + a wasm proof-of-work), Qwen
(``chat.qwen.ai`` behind Aliyun WAF) and a local ``opencode`` CLI.
Every other test in this repo runs
against monkeypatched stubs, so the whole test suite is structurally blind
to the one defect class that reading code can never find: **the upstream
changed its wire format and the bridge still answers 200**.

``ARCHITECTURE.md`` records that class of defect biting for real: "a stateless
parser silently truncates replies (this bug actually shipped once -- reply
came back 'web' instead of 'web2api-ok')". A stub cannot reproduce that,
because the stub *is* the thing that broke. Hence this file: a small, fast,
periodic suite that a maintainer runs with a real account, and that fails
loudly, with a diagnosis, the moment an upstream moves.

Design rules
------------
* **The three provider probes are ``@pytest.mark.live``; the two meta-tests
  deliberately are not.** ``tests/conftest.py`` skips the live marker unless
  the command line positively selects it, and it rewrites
  ``DEEPSEAPORT_CONFIG``/``DEEPSEAPORT_DATA`` to a temp tree and installs a
  non-loopback socket guard for every other run. Selecting ``-m live`` lifts
  both and uses the operator's real config, so the probes add zero risk to
  ``python -m pytest -m "not live" -q`` (what CI runs) and zero network to
  the default suite. The meta-tests -- the credential-leak check over this
  file's own diagnostics, and the gate-wording check -- are pure and hermetic,
  so they stay unmarked and therefore stay in CI. A policy test that only
  runs when someone has opted into the live tier is a policy test nobody runs.
* **No import-time side effects.** ``load_settings()`` is called inside the
  gate, never at module scope, so ``--collect-only`` on a machine with no
  config reads no file and opens no socket. Same discipline as
  ``tests/test_speed.py:308`` ("imported late so module import never fails").
* **The gate skips, it never fails.** A maintainer with no account, no
  ``obscura``, no ``opencode`` binary, or a Qwen-only config must get
  ``pytest.skip`` with a precise, actionable reason -- never a red suite.
  The converse also holds: **no network call happens before the gate
  passes**, so a skipped run costs nothing upstream.
* **A skip reason must name the PREREQUISITE, not the symptom.** A stale
  token can look perfectly healthy to every readiness check this gate can
  make (bound, has a token, not banned) while the chat path refuses it.
  That is why nothing here keys off "the token is present".
* **Probes assert invariants, not "it didn't crash".** Every probe names the
  drift it is built to catch in its docstring, and the assertions are the
  ones whose failure is *diagnostic* (a missing terminal token, a second
  ``[DONE]``, a ``tool_calls`` delta without ``index``).
* **A failure must say what drifted**: which provider, which model, the HTTP
  status, a redacted excerpt of the raw body / SSE frames, and which
  assertion tripped. Every diagnostic goes through
  ``core.engine.scrub_upstream_text`` -- the project's established scrubber
  (``tests/test_secret_and_leak_gaps.py`` pins it) -- so a live failure can
  never print a bearer token or a Qwen JWT into a CI log. That policy is
  asserted about *this file's* formatters at the bottom of this file.

Cost
----
Roughly one upstream round trip per probe per active provider, plus one
extra for the tool round trip (stream turn + non-stream turn + follow-up).
With DeepSeek + Qwen + opencode all active that is ~15 requests: a
couple of minutes. The opencode probe additionally pays the managed-serve boot
once (``providers/opencode/README.md``: the first request boots an isolated
``opencode serve``), which is why its gate does that boot explicitly and the
probes carry a generous client timeout.

Run
---
    python -m pytest -m live -q tests/test_live_smoke.py
    python -m pytest -m live -q tests/test_live_smoke.py -k deepseek
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import threading
import time

import pytest

# The scrubber is the project's redaction policy for anything an upstream
# said back to us: two shapes (a long opaque run, a dot-framed JWT), plus a
# hard length clamp. See core/engine.py:scrub_upstream_text.
from deepseaport.core.engine import scrub_upstream_text

# ---------------------------------------------------------------------------
# Probe vocabulary
# ---------------------------------------------------------------------------

# The single most valuable assertion in this file.
#
# A stateless SSE parser drops every continuation chunk (DeepSeek's bare
# ``data: {"v":"..."}`` frames carry no ``p`` path), so a truncated reply is
# indistinguishable from a short one at the HTTP layer: status 200, a
# non-empty string, a plausible ``finish_reason``. The historical shipped bug
# was exactly that -- "web" instead of "web2api-ok".
#
# So the prompt is built around a token that can only be in the reply if the
# WHOLE reply arrived. A reply missing the tail cannot contain it, whatever
# the model was asked. A nonce-shaped token (not a word) keeps the model from
# "helpfully" completing it, and a fixed value keeps the suite deterministic.
_ANSWER_TOKEN = "dsptlivesmoke-7q2-ok"

_CHAT_PROMPT = (
    "Reply with one short sentence of your own, then on a new line the exact "
    f"token {_ANSWER_TOKEN} and nothing after it."
)

# Bodies an upstream returns *instead of* an answer. These are not answers,
# so a reply containing one is drift, not a model quirk: WAF/interstitial
# HTML leaking through the SSE parser, or the bridge reflecting a refusal
# page as content.
_KNOWN_FAILURE_MARKERS = (
    "<html",
    "<!doctype",
    "just a moment",
    "checking your browser",
    "verify you are human",
    "enable javascript",
    "aliyun_waf",
    "aliyuncaptcha",
    "nc-container",
    "access denied",
    "captcha",
)

# Client-side ceiling, applied to the TestClient ONCE (see ``_build_gate``)
# rather than per request: starlette's TestClient deprecates a per-request
# ``timeout=`` argument, and the client-level knob is the supported one.
#
# Deliberately generous: the bridge owns the decision of when a turn is dead
# (its own consumer watchdog fires at ~130 s in `live` stream mode and ~390 s
# in `buffered`, then emits an in-band ``data: {"error": ...}`` frame this file
# detects and reports). Cutting a healthy-but-slow turn short at the test
# client would replace a diagnosable in-band error with an opaque httpx
# timeout, which is the one failure mode a smoke suite must not add.
_REQ_TIMEOUT = 420.0

# Diagnostic excerpt caps. Bodies and SSE frames are what a maintainer reads
# to see what the upstream actually said, so they get more room than a log
# line -- but still bounded, because a broken upstream can return megabytes.
_BODY_CAP = 700
_FRAME_CAP = 1600

# Whitespace-insensitive comparison: DeepSeek streams token-by-token and the
# buffered path re-emits the same text in one delta, and Qwen's `delta.phase`
# mapping can split a word across chunks. A reply's *characters* must match;
# its whitespace need not.
_WS = re.compile(r"\s+")


def _squash(text) -> str:
    """Whitespace-insensitive form used to compare two renderings of a reply."""
    return _WS.sub("", str(text or ""))


def _redact(text, cap: int = _BODY_CAP) -> str:
    """Bound + de-tokenize an upstream excerpt. Never raises."""
    return scrub_upstream_text(text, cap)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _diag(kind: str, provider: str, model: str, *, status=None, body=None,
          note: str = "", frames=None) -> str:
    """One drift report: what was asked, what came back, which check failed.

    Every field that could carry upstream text is scrubbed here, not at the
    call site, so a new probe cannot forget. ``kind`` names the assertion
    that tripped (it is the first thing a maintainer reads); the rest is the
    evidence needed to act on it.
    """
    parts = [f"[live-smoke] {kind}",
             f"provider={provider}", f"model={model}"]
    if status is not None:
        parts.append(f"http_status={status}")
    if note:
        parts.append(f"note={_redact(note, 300)}")
    if body is not None:
        parts.append(f"body={_redact(body, _BODY_CAP)}")
    if frames is not None:
        parts.append(f"sse_frames={_redact(frames, _FRAME_CAP)}")
    return " | ".join(parts)


def _fail(kind: str, provider: str, model: str, **kw) -> "AssertionError":
    """Build the AssertionError a failing probe raises (collected, not raised).

    Returning rather than raising keeps the decision in the collector: one
    live run should report every drifted provider, not stop at the first.
    """
    return AssertionError(_diag(kind, provider, model, **kw))


# ---------------------------------------------------------------------------
# SSE parsing (the bridge's own wire format, not the upstream's)
# ---------------------------------------------------------------------------


def _collect_stream(raw: str) -> dict:
    """Parse a ``text/event-stream`` body into the facts the probes assert on.

    Deliberately strict about framing and deliberately forgiving about
    content: this is a conformance reader for the OpenAI-shaped SSE the
    bridge emits (``core.engine.stream_chunk`` / ``usage_chunk`` /
    ``sse_error_frame``), so a violation here is a bridge regression, and a
    violation of the *content* invariants below is an upstream one.
    """
    out: dict = {
        "chunks": [], "dones": 0, "bad_lines": [], "errors": [],
        "content": "", "reasoning": "", "tool_calls": [],
        "finish": None, "usage": None, "ids": set(), "createds": set(),
        "models": set(),
    }
    for line in raw.splitlines():
        if not line.strip():
            continue
        if not line.startswith("data: "):
            # No `event:` lines, no comment lines, no bare JSON: every frame
            # the bridge writes is `data: <payload>` (engine.py:929, :983,
            # :1015). Anything else means the framing changed.
            out["bad_lines"].append(line)
            continue
        payload = line[len("data: "):].strip()
        if payload == "[DONE]":
            out["dones"] += 1
            continue
        try:
            chunk = json.loads(payload)
        except ValueError:
            out["bad_lines"].append(line)
            continue
        if not isinstance(chunk, dict):
            out["bad_lines"].append(line)
            continue
        out["chunks"].append(chunk)
        if isinstance(chunk.get("id"), str):
            out["ids"].add(chunk["id"])
        if isinstance(chunk.get("created"), int):
            out["createds"].add(chunk["created"])
        if isinstance(chunk.get("model"), str):
            out["models"].add(chunk["model"])
        if chunk.get("error"):
            # An in-band error frame is how the bridge reports a mid-stream
            # failure (engine.sse_error_frame). It is never an answer.
            out["errors"].append(chunk["error"])
        if isinstance(chunk.get("usage"), dict):
            out["usage"] = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            if choice.get("finish_reason"):
                out["finish"] = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                continue
            if isinstance(delta.get("content"), str):
                out["content"] += delta["content"]
            if isinstance(delta.get("reasoning_content"), str):
                out["reasoning"] += delta["reasoning_content"]
            for call in delta.get("tool_calls") or []:
                if isinstance(call, dict):
                    out["tool_calls"].append(call)
    return out


# ---------------------------------------------------------------------------
# Readiness gate
# ---------------------------------------------------------------------------


class _Target:
    """One provider this run will probe, plus the models it will use.

    ``chat_model`` is the cheapest model that still answers (no thinking, no
    search) so the bulk of the suite stays fast, falling back to the
    provider's own default when every model it serves thinks (Qwen, opencode).
    ``reasoner_model`` is the first thinking-capable model, which is the same
    id as ``chat_model`` for exactly those two providers -- a distinct request
    is still made, because the reasoning probe asks a different question.
    """

    __slots__ = ("provider", "chat_model", "reasoner_model", "bridge")

    def __init__(self, provider: str, chat_model: str, reasoner_model: str,
                 bridge=None) -> None:
        self.provider = provider
        self.chat_model = chat_model
        self.reasoner_model = reasoner_model
        self.bridge = bridge


class _Live:
    """Session-scoped live environment: settings, app, client, targets.

    Built once per session and reused, because the expensive part of the gate
    (an Obscura WAF prime, or the opencode managed-serve boot) is a *machine*
    state, not a per-test one. Re-priming it per probe would multiply the
    runtime by the number of probes for no extra coverage.
    """

    def __init__(self, settings, app, client, targets, notes) -> None:
        self.settings = settings
        self.app = app
        self.client = client
        self.targets = targets
        self.notes = notes
        # Memoized non-stream responses, keyed by the full request (see
        # ``_nonstream_json``). The stream probe is the only consumer, and it
        # is what keeps the file at ~one upstream turn per probe instead of
        # two.
        self.nonstream: dict = {}
        self.lock = threading.Lock()


_GATE: _Live | None = None
_GATE_LOCK = threading.Lock()


def _pick_chat_model(provider) -> str:
    """Cheapest answering model: no thinking, no search, else the default."""
    specs = provider.models()
    for mid, spec in specs.items():
        if not spec.thinking and not spec.search:
            return mid
    return provider.default_model() or next(iter(specs), "")


def _pick_reasoner_model(provider) -> str:
    """First thinking-capable model; "" when the provider serves none."""
    for mid, spec in provider.models().items():
        if spec.thinking:
            return mid
    return ""


def _binary_ok(path: str) -> bool:
    """True when a resolved binary is a real file or resolvable on PATH.

    ``ObscuraBridge._discover`` returns the bare string ``"obscura"`` when it
    finds nothing (core/obscura_bridge.py:304), so an unresolved binary looks
    like a plausible command name. Both halves have to be checked or the gate
    would proceed to spawn a process that does not exist.
    """
    raw = str(path or "").strip()
    if not raw:
        return False
    return bool(shutil.which(raw)) or os.path.isfile(raw)


def _ready_accounts(settings, pid: str):
    """Accounts bound to ``pid`` that could serve a request, and why not.

    Mirrors what the route layer will actually build: ``server_routes._pool``
    takes ``pools.accounts_for_provider`` and never falls back to every
    account, so a provider with no bound account is a 503 rather than a silent
    cross-provider borrow. That borrow would hand one upstream another's live
    credentials (a Qwen ES256 JWT replayed to chat.deepseek.com).
    """
    from deepseaport.core import pools

    bound = pools.accounts_for_provider(settings, pid)
    if not bound:
        return [], (f"no account is bound to provider {pid!r} "
                    f"(set accounts[].provider, or provider={pid!r})")
    with_token = [a for a in bound if str(getattr(a, "token", "") or "").strip()]
    if not with_token:
        return [], (f"no {pid} account has a token "
                    f"(run: deepseaport login --provider {pid})")
    now = time.time()
    usable = [a for a in with_token
              if not getattr(a, "banned", False)
              or float(getattr(a, "banned_until", 0.0) or 0.0) > now]
    if not usable:
        return [], (f"every {pid} account with a token is currently banned "
                    f"(deepseaport accounts unblock)")
    return usable, ""


def _prime_waf(provider, bridge) -> tuple[bool, str]:
    """Warm the WAF jar this provider needs; False + a reason if it will not.

    Per provider, deliberately. DeepSeek's gate is the ``aws-waf-token``
    cookie: on that host a 403 *is* the AWS challenge
    (``ARCHITECTURE.md``, "A 403 means different things per provider"), so the
    cookie is the whole readiness signal. Qwen's WAF is Aliyun and answers
    HTTP 200 + HTML, and its solved jar is the 64-char ``acw_tc`` plus
    ``x-ap``/``sca``/``cna``/``tfstk``/``isg`` harvested by
    ``prime_qwen_bridge`` -- ``has_waf_token()`` would be False there forever
    and the Qwen probe could never run.
    """
    pid = getattr(provider, "id", "")
    try:
        if pid == "qwen":
            provider.warmup(bridge)
            from deepseaport.providers.qwen import client as QC
            return (bool(QC.qwen_waf_cookie_header()),
                    "qwen Aliyun jar: no acw_tc after prime_qwen_bridge "
                    "(obscura cannot solve chat.qwen.ai's challenge)")
        bridge.warmup()
        return (bool(bridge.has_waf_token()),
                "no aws-waf-token after Obscura warmup (challenge unsolved, "
                "or the profile is locked by another process)")
    except Exception as exc:  # noqa: BLE001 - a gate never raises at the operator
        return False, f"WAF prime failed: {type(exc).__name__}: {_redact(exc, 200)}"


def _prime_opencode() -> tuple[bool, str]:
    """Make sure a local ``opencode serve`` answers before probing it.

    ``providers/opencode/README.md``: the first request boots a managed
    isolated ``opencode serve`` (its own port, its own generated config), and
    that boot is slow. Doing it here -- once, explicitly -- means the probe
    timeouts measure the model, not the installer. An operator's own serve
    (``DEEPSEAPORT_OPENCODE_BASE``) is used as-is and needs no binary.
    """
    from deepseaport.providers.opencode import client as OC
    from deepseaport.providers.opencode import managed_serve as MS
    from deepseaport.providers.opencode import protocol as P

    own_base = (os.environ.get("DEEPSEAPORT_OPENCODE_BASE", "") or "").strip()
    binary = shutil.which("opencode")
    if not own_base and not binary:
        return False, ("the `opencode` binary is not on PATH and "
                       "DEEPSEAPORT_OPENCODE_BASE is unset")
    if OC.health(timeout=10):
        return True, ""
    if own_base or not MS.managed_enabled():
        return False, (f"opencode serve is not answering at {P.base_url()} "
                       f"and this run cannot boot one")
    if not MS.ensure(timeout=90):
        return False, (f"managed opencode serve failed to boot "
                       f"(see the managed serve log for the child's stderr)")
    return (True, "") if OC.health(timeout=10) else (
        False, f"opencode serve still not answering at {P.base_url()} after boot")


def _build_gate() -> _Live:
    """Resolve the operator's environment, or explain precisely what is missing."""
    import httpx

    from deepseaport.config import load_settings
    from deepseaport.core import pools, registry
    from deepseaport.core.obscura_bridge import ObscuraBridge
    from deepseaport.server import create_app

    try:
        settings = load_settings()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"config unreadable: {type(exc).__name__}: {_redact(exc, 200)}")

    active = pools.active_provider_ids(settings)
    notes: list[str] = []
    targets: list[_Target] = []
    bridge: ObscuraBridge | None = None

    for pid in active:
        provider = registry.get(pid)
        if provider is None:
            notes.append(f"{pid}: not a registered provider")
            continue

        if getattr(provider, "requires_accounts", True):
            usable, why = _ready_accounts(settings, pid)
            if not usable:
                notes.append(f"{pid}: {why}")
                continue

        if getattr(provider, "needs_bridge", True):
            # One bridge for the whole run: it owns a single persistent Obscura
            # profile, and two bridges over one --storage-dir race the cookie
            # jar (core/obscura_bridge.py:235).
            if bridge is None:
                bridge = ObscuraBridge(settings.obscura_bin,
                                       settings.obscura_profile)
                if not _binary_ok(bridge.binary):
                    notes.append(
                        f"{pid}: obscura binary not found "
                        f"(resolved {bridge.binary!r}; set obscura_bin, "
                        f"OBSCURA_BIN, or put it on PATH)")
                    bridge = None
                    continue
            primed, why = _prime_waf(provider, bridge)
            if not primed:
                notes.append(f"{pid}: {why}")
                continue
        elif pid == "opencode":
            primed, why = _prime_opencode()
            if not primed:
                notes.append(f"opencode: {why}")
                continue

        chat_model = _pick_chat_model(provider)
        if not chat_model:
            notes.append(f"{pid}: the provider serves no model")
            continue
        targets.append(_Target(pid, chat_model,
                               _pick_reasoner_model(provider), bridge))

    if not targets:
        # Every provider that was asked for and could not be probed, each with
        # its own reason. One line naming the blocker beats "no live provider
        # is ready", which sends the maintainer hunting.
        pytest.skip("no live provider is ready -- " + "; ".join(notes)
                    if notes else "no live provider is ready -- config names "
                                   "no active provider")

    app = create_app(settings)
    if bridge is not None:
        # No lifespan is run (the TestClient is used without `with`, exactly
        # as tests/test_speed.py does), so the bridge is installed by hand and
        # published process-wide: ``providers/*/protocol.bridge_headers`` and
        # the ban probes resolve it through ``get_default_bridge()``, not from
        # app.state (see server_concurrency.lifespan, which does both).
        app.state.bridge = bridge
        from deepseaport.core.obscura_bridge import register_default_bridge
        register_default_bridge(bridge)

    from fastapi.testclient import TestClient
    client = TestClient(app)
    # Client-level, not per-request: starlette's TestClient deprecates a
    # per-request `timeout=` (its own warning), and this is the supported knob.
    # `stream=` is set explicitly so a long stream is bounded by the read
    # timeout per read rather than as one total budget -- a smoke run must not
    # cut a slow-but-alive stream off at a fixed wall clock.
    client.timeout = httpx.Timeout(connect=30.0, read=_REQ_TIMEOUT,
                                   write=_REQ_TIMEOUT, pool=30.0)
    return _Live(settings, app, client, targets, notes)


def _live() -> _Live:
    """The session's live environment, built on first use."""
    global _GATE
    with _GATE_LOCK:
        if _GATE is None:
            _GATE = _build_gate()
        return _GATE


def _auth_headers(settings) -> dict:
    """Bearer header when keys are configured; {} for the auth-disabled default."""
    keys = list(getattr(settings, "keys", []) or [])
    return {"Authorization": f"Bearer {keys[0]}"} if keys else {}


def _target(pid: str) -> _Target:
    """The target for ``pid``, or a skip naming why this run cannot probe it.

    The reason is the gate's own note for that provider, not a generic
    "not ready": a maintainer running only DeepSeek should see a skip on the
    Qwen test that says Qwen has no account, not one that says the suite
    could not start.
    """
    live = _live()
    for target in live.targets:
        if target.provider == pid:
            return target
    for note in live.notes:
        if note.startswith(f"{pid}:"):
            pytest.skip(f"live smoke: {note}")
    pytest.skip(f"live smoke: provider {pid!r} is not active in this config "
                f"(active: {', '.join(p.provider for p in live.targets) or 'none'})")


# ---------------------------------------------------------------------------
# Collector: report every drifted provider in one run
# ---------------------------------------------------------------------------


class _Probes:
    """Runs each probe and gathers failures instead of stopping at the first.

    A maintainer running the suite after a suspected upstream change wants the
    whole picture in one run: "DeepSeek truncated, Qwen lost its reasoning
    delta" is actionable, "the first one failed" is a second trip. Failures
    are still plain AssertionErrors carrying this file's full diagnostics.
    """

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.ran = 0

    def run(self, label: str, fn) -> None:
        self.ran += 1
        try:
            fn()
        except AssertionError as exc:
            self.failures.append(f"{label}: {exc}")

    def report(self) -> None:
        if not self.failures:
            return
        raise AssertionError(
            f"{len(self.failures)} of {self.ran} live probe(s) drifted:\n  - "
            + "\n  - ".join(self.failures))


def _skip_unless(condition: bool, reason: str) -> None:
    if not condition:
        pytest.skip(reason)


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def _nonstream_json(live: _Live, target: _Target, model: str,
                    body: dict) -> dict:
    """One non-stream completion, memoized per (provider, model, request).

    ``stream=False`` is the shape an agent harness uses for the first turn and
    the shape ``_openai_response`` shapes, so it is the one that has to stay
    correct on its own; the streaming path is a separate framing of the same
    engine and is probed separately.

    The memo exists for exactly one caller: the stream probe needs the
    non-stream rendering of the SAME request to compare deltas against, and
    without it that comparison would cost a second upstream turn per provider
    (and could legitimately differ, since these are sampled completions).
    Keyed on the whole request rather than on (provider, model) because a
    plain chat turn and a tool turn share a model and mean different things
    by it -- see ``_Live.nonstream``.
    """
    key = (target.provider, model, json.dumps(body, sort_keys=True))
    with live.lock:
        hit = live.nonstream.get(key)
    if hit is not None:
        return hit
    status, text = _post_json(live, target, model, body)
    if status != 200:
        raise _fail("non-stream request did not return 200", target.provider,
                    model, status=status, body=text)
    try:
        data = json.loads(text)
    except ValueError:
        raise _fail("non-stream response is not JSON", target.provider, model,
                    status=status, body=text) from None
    with live.lock:
        live.nonstream[key] = data
    return data


def _post_json(live: _Live, target: _Target, model: str, body: dict) -> tuple:
    """One non-stream POST, as ``(status, text)``.

    Every transport-level failure is folded into the same drift report as an
    HTTP error. A smoke suite that dies on an httpx traceback tells the
    maintainer nothing about which upstream moved; one that says "status
    502, body ..." tells them everything.
    """
    payload = dict(body)
    payload["model"] = model
    payload["stream"] = False
    try:
        resp = live.client.post("/v1/chat/completions", json=payload,
                                headers=_auth_headers(live.settings))
    except Exception as exc:  # noqa: BLE001 - reported, never raised raw
        raise _fail("the request did not complete", target.provider, model,
                    note=f"{type(exc).__name__}: {exc}") from None
    return resp.status_code, resp.text


def _post_stream(live: _Live, target: _Target, model: str, body: dict) -> str:
    """One streaming completion; returns the raw SSE body for conformance."""
    payload = dict(body)
    payload["model"] = model
    payload["stream"] = True
    try:
        with live.client.stream("POST", "/v1/chat/completions", json=payload,
                                headers=_auth_headers(live.settings)) as resp:
            raw = resp.read().decode("utf-8", "replace")
            status = resp.status_code
    except Exception as exc:  # noqa: BLE001 - reported as drift, never raised raw
        # A transport-level failure (client timeout, connection reset) is not
        # an answer and not an httpx traceback in a smoke-suite log: it is
        # "the turn never completed", with the upstream excerpt if any
        # arrived. HTTPException from the route is the common case and carries
        # its own already-scrubbed detail.
        raise _fail("the streaming request did not complete", target.provider,
                    model, note=f"{type(exc).__name__}: {exc}") from None
    if status != 200:
        raise _fail("stream request did not return 200", target.provider,
                    model, status=status, body=raw)
    return raw


def _assert_chat_reply(content: str, target: _Target, model: str, *,
                       status=None, body=None, frames=None) -> None:
    """The shared chat invariants, including the truncation guard.

    Every one of these is a *drift* assertion, not a sanity check:
    a known-failure string means an interstitial page was parsed as content;
    a missing tail token means continuation chunks stopped being applied
    (the bug that shipped); a reply that is a prefix of the prompt means the
    model echoed the question and never got to the answer at all.
    """
    evidence = {"status": status, "body": body, "frames": frames}
    text = str(content or "")
    low = text.lower()
    for marker in _KNOWN_FAILURE_MARKERS:
        if marker in low:
            raise _fail(f"reply is a known failure string ({marker!r}), not an "
                        f"answer", target.provider, model, note=text[:200],
                        **evidence)
    if not text.strip():
        raise _fail("reply content is empty", target.provider, model,
                    **evidence)
    if _ANSWER_TOKEN not in text:
        # The load-bearing assertion. A reply cut at any earlier chunk
        # satisfies "non-empty content" and fails here.
        raise _fail(
            f"reply is truncated: the prompt's terminal token "
            f"{_ANSWER_TOKEN!r} is absent (the historical "
            f"'web' instead of 'web2api-ok' bug -- continuation chunks are "
            f"not being applied)",
            target.provider, model, note=f"content={text[:200]!r}", **evidence)
    if _squash(text) == _squash(_CHAT_PROMPT):
        raise _fail("reply is the prompt echoed back, not an answer",
                    target.provider, model, **evidence)


_PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_smoke_probe_value",
        "description": ("Read the fixed live smoke probe value. Read-only, "
                        "takes one key, returns a short string."),
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string",
                                   "description": "the value key to read"}},
            "required": ["key"],
        },
    },
}
_PROBE_TOOL_NAME = "read_smoke_probe_value"

_TOOL_PROMPT = (
    f"You must call the {_PROBE_TOOL_NAME} tool with key \"live-smoke\". "
    f"Do not answer from memory and do not guess the value: call the tool."
)
_TOOL_RESULT = "live-smoke-value-4f2a"


def _probe_nonstream_chat(live: _Live, target: _Target) -> None:
    """Non-stream chat: the shape every OpenAI client asks for first.

    Invariants: HTTP 200, the requested model id echoed back, a
    ``finish_reason`` this bridge actually emits for a plain answer, a usage
    block with positive counts, non-empty content, no known failure string,
    and the prompt's terminal token present (see ``_assert_chat_reply``).
    """
    model = target.chat_model
    data = _nonstream_json(live, target, model, {
        "messages": [{"role": "user", "content": _CHAT_PROMPT}]})

    if data.get("model") != model:
        raise _fail("response echoes a different model than the request",
                    target.provider, model, body=json.dumps(data)[:400])
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise _fail("response has no choices", target.provider, model,
                    body=json.dumps(data)[:400])
    choice = choices[0] or {}
    if choice.get("finish_reason") != "stop":
        # A plain chat turn has no tools and no truncation, so anything but
        # "stop" means the engine's finish mapping drifted.
        raise _fail("finish_reason for a plain chat turn is not 'stop'",
                    target.provider, model,
                    body=json.dumps(data)[:400],
                    note=f"finish_reason={choice.get('finish_reason')!r}")
    message = choice.get("message") or {}
    if message.get("role") != "assistant":
        raise _fail("message.role is not 'assistant'", target.provider, model,
                    body=json.dumps(data)[:400])
    _assert_chat_reply(message.get("content"), target, model,
                       status=200, body=json.dumps(data)[:400])

    usage = data.get("usage")
    if not isinstance(usage, dict) or not usage:
        raise _fail("response carries no usage block", target.provider, model,
                    body=json.dumps(data)[:400])
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if not isinstance(usage.get(key), int) or usage[key] <= 0:
            raise _fail(f"usage.{key} is missing or not positive",
                        target.provider, model, body=json.dumps(data)[:400],
                        note=f"usage={json.dumps(usage)[:200]}")
    if usage["total_tokens"] < usage["completion_tokens"]:
        raise _fail("usage.total_tokens is below completion_tokens",
                    target.provider, model, body=json.dumps(data)[:400])


def _probe_stream_chat(live: _Live, target: _Target) -> None:
    """Streamed chat: SSE framing intact, and the same answer as non-stream.

    Invariants: every frame is a ``data: `` line, exactly one ``data: [DONE]``,
    at least one chunk, every chunk well-formed, one ``id``/``created``/
    ``model`` across the whole stream, no in-band ``error`` frame, exactly one
    ``finish_reason`` (and it is ``stop``), a positive usage chunk when
    ``include_usage`` was asked for, and -- the point of the probe -- the
    concatenated deltas equal the non-stream rendering of the same prompt.

    That last one is what catches both drift modes this project has already
    lived through: DeepSeek's continuation chunks (bare ``{"v": ...}`` with no
    ``p``) silently dropping from the live stream, and Qwen's ``delta.phase``
    mapping (``think`` vs ``answer``) sending text to the wrong field.
    """
    model = target.chat_model
    body = {"messages": [{"role": "user", "content": _CHAT_PROMPT}],
            "stream_options": {"include_usage": True}}
    raw = _post_stream(live, target, model, body)
    seen = _collect_stream(raw)

    if seen["bad_lines"]:
        raise _fail("SSE framing is not intact (non `data: ` or unparseable "
                    "line in the stream)", target.provider, model,
                    frames=raw, note="; ".join(seen["bad_lines"])[:300])
    if seen["dones"] != 1:
        raise _fail("the stream must carry exactly one `data: [DONE]`",
                    target.provider, model, frames=raw,
                    note=f"seen {seen['dones']}")
    if not seen["chunks"]:
        raise _fail("the stream carried no chunk at all", target.provider,
                    model, frames=raw)
    if seen["errors"]:
        raise _fail("the stream delivered an in-band error frame instead of an "
                    "answer", target.provider, model, frames=raw,
                    note=json.dumps(seen["errors"])[:300])
    if len(seen["ids"]) != 1 or len(seen["createds"]) != 1:
        raise _fail("id/created are not consistent across the stream's chunks",
                    target.provider, model, frames=raw,
                    note=f"ids={sorted(seen['ids'])} "
                         f"created={sorted(seen['createds'])}")
    if seen["models"] != {model}:
        raise _fail("a chunk reported a different model than the request",
                    target.provider, model, frames=raw,
                    note=f"models={sorted(seen['models'])}")
    for chunk in seen["chunks"]:
        if chunk.get("object") != "chat.completion.chunk":
            raise _fail("a chunk is not a chat.completion.chunk object",
                        target.provider, model, frames=raw,
                        note=f"object={chunk.get('object')!r}")
    if seen["finish"] != "stop":
        raise _fail("streamed finish_reason for a plain chat turn is not "
                    "'stop'", target.provider, model, frames=raw,
                    note=f"finish_reason={seen['finish']!r}")
    if not isinstance(seen["usage"], dict) or \
            not isinstance(seen["usage"].get("total_tokens"), int) or \
            seen["usage"]["total_tokens"] <= 0:
        raise _fail("stream_options.include_usage produced no positive usage "
                    "chunk before [DONE]", target.provider, model, frames=raw,
                    note=f"usage={json.dumps(seen['usage'])[:200]}")
    if not seen["content"]:
        raise _fail("the stream carried no content delta", target.provider,
                    model, frames=raw,
                    note="every delta was reasoning_content or empty -- the "
                         "upstream's content path is no longer recognised")

    # Same request body as the non-stream probe, so the memo normally serves
    # this from the turn that probe already paid for. It is spelled out rather
    # than shared through a constant only so that a future edit to one probe's
    # prompt cannot silently compare two different questions.
    reference = _nonstream_json(live, target, model, {
        "messages": [{"role": "user", "content": _CHAT_PROMPT}]})
    plain = ((reference.get("choices") or [{}])[0].get("message") or {})
    if _squash(seen["content"]) != _squash(plain.get("content")):
        raise _fail(
            "the streamed deltas do not match the non-stream content",
            target.provider, model, frames=raw,
            note=f"streamed={_squash(seen['content'])[:200]!r} "
                 f"non-stream={_squash(plain.get('content'))[:200]!r}")
    # Same tail-token guarantee on the streaming path, independently of the
    # comparison above: a stream that lost its continuation chunks and a
    # stream that lost its tail both fail here, with a different message.
    _assert_chat_reply(seen["content"], target, model, frames=raw)


def _probe_reasoning(live: _Live, target: _Target) -> None:
    """A thinking model must return ``reasoning_content`` next to content.

    Invariant: for a catalogue model flagged ``thinking``, the request
    succeeds AND ``reasoning_content`` is populated. A reasoner that starts
    answering with no thinking at all is not a style change: it means the
    upstream's thinking channel moved, was renamed, or stopped being emitted
    -- and the bridge would map it to nothing while still answering 200. The
    model's own thinking text is not asserted on, only that it is there.
    """
    model = target.reasoner_model
    if not model:
        pytest.skip(f"{target.provider} serves no thinking-capable model")
    # A prompt that makes thinking unavoidable rather than optional: a model
    # that can answer "6*7" in one token is free to skip the reasoning
    # channel, and then a working upstream would be reported as drift.
    data = _nonstream_json(live, target, model, {
        "messages": [{"role": "user",
                      "content": "Solve 6*7 step by step, keep it very short."}]})
    message = ((data.get("choices") or [{}])[0].get("message") or {})
    if not str(message.get("content") or "").strip():
        raise _fail("the reasoner returned no content", target.provider, model,
                    body=json.dumps(data)[:400])
    if not str(message.get("reasoning_content") or "").strip():
        raise _fail(
            "the reasoner returned NO reasoning_content: the upstream's "
            "thinking channel is no longer being read (this used to be "
            "`thinking_enabled` + a `response/thinking_content` stream path)",
            target.provider, model, body=json.dumps(data)[:400],
            note=f"content={_redact(message.get('content'), 200)!r}")
    reasoning = str(message["reasoning_content"])
    if _squash(reasoning) == _squash(message["content"]):
        raise _fail("reasoning_content duplicates content (the thinking stream "
                    "is being routed to the wrong field)", target.provider,
                    model, body=json.dumps(data)[:400])


def _assert_tool_calls(calls, target: _Target, model: str, *, indexed: bool,
                       status=None, body=None, frames=None) -> None:
    """Shape + allowlist invariants for a tool-call turn.

    ``indexed`` is the difference between the two paths and the reason this
    helper is shared rather than duplicated: OpenAI's streaming contract makes
    ``index`` MANDATORY on ``tool_calls`` deltas, because each delta is
    otherwise a complete standalone call with nothing saying which of N
    parallel calls it belongs to (``core.engine.tool_finish_chunks``). An
    index-less delta raises inside the stream of a real ``openai-python``
    client, mid agent-loop turn, and an index-keyed accumulator cannot merge
    parallel calls at all -- so a missing index is a hard failure, and two
    calls sharing one index is a hard failure too.
    """
    evidence = {"status": status, "body": body, "frames": frames}
    if not calls:
        raise _fail("no tool_calls came back for a prompt that had to call a "
                    "tool", target.provider, model, **evidence)
    allowed = {_PROBE_TOOL_NAME}
    seen_indexes: list = []
    for call in calls:
        if indexed:
            if not isinstance(call.get("index"), int):
                raise _fail(
                    "a streamed tool_calls delta carries no `index` field",
                    target.provider, model, note=f"delta={json.dumps(call)[:200]}",
                    **evidence)
            seen_indexes.append(call["index"])
        else:
            if not call.get("id"):
                raise _fail("a tool_call has no id", target.provider, model,
                            note=f"call={json.dumps(call)[:200]}", **evidence)
            if call.get("type") != "function":
                raise _fail("a tool_call is not type=function", target.provider,
                            model, note=f"call={json.dumps(call)[:200]}",
                            **evidence)
        function = call.get("function") or {}
        name = str(function.get("name") or "")
        if name not in allowed:
            # A name outside the declared allowlist means the parser's
            # name-checking stopped holding, i.e. the harness would be asked to
            # dispatch a tool that does not exist.
            raise _fail("a tool_call named a function that was never declared",
                        target.provider, model,
                        note=f"name={name!r} declared={sorted(allowed)}",
                        **evidence)
        arguments = function.get("arguments")
        if not isinstance(arguments, str) or not arguments.strip():
            raise _fail("a tool_call carries no JSON-string arguments",
                        target.provider, model, note=f"arguments={arguments!r}",
                        **evidence)
        try:
            parsed = json.loads(arguments)
        except ValueError:
            raise _fail("tool_call arguments are not valid JSON", target.provider,
                        model, note=f"arguments={_redact(arguments, 200)!r}",
                        **evidence)
        if not isinstance(parsed, dict):
            raise _fail("tool_call arguments are not a JSON object",
                        target.provider, model, **evidence)
    if indexed and len(set(seen_indexes)) != len(seen_indexes):
        raise _fail("two streamed tool_calls deltas share one `index`, so a "
                    "consumer cannot tell the parallel calls apart",
                    target.provider, model, note=f"indexes={seen_indexes}",
                    **evidence)


def _probe_tool_roundtrip(live: _Live, target: _Target) -> None:
    """The headline feature: a tool call the harness can act on, then a follow-up.

    Three assertions in sequence, each its own drift signal:

    1. **streamed turn** -- ``finish_reason == "tool_calls"``, at least one
       call, every delta carrying ``index`` with distinct values, and every
       name inside the declared allowlist.
    2. **non-stream turn** -- the same call through ``_openai_response``,
       where the calls arrive as one list (so position is implicit and no
       index is expected) but the allowlist still has to hold.
    3. **follow-up** -- the ``role: "tool"`` result is folded back into the
       prompt (``core.tools.prompt.render_prompt``) and the model answers.
       This is the multi-step agent loop; there is no native function calling
       upstream, so the whole thing is the bridge's prompt shaping and its
       parser agreeing with each other.
    """
    model = target.chat_model
    _skip_unless(getattr(live.settings, "enable_tools", True),
                 "enable_tools is off in config.json: the server strips tool "
                 "schemas, so no tool_calls can come back (README: harnesses "
                 "need it on)")

    # 1. streamed tool turn
    raw = _post_stream(live, target, model, {
        "messages": [{"role": "user", "content": _TOOL_PROMPT}],
        "tools": [_PROBE_TOOL], "tool_choice": "auto"})
    seen = _collect_stream(raw)
    if seen["errors"]:
        raise _fail("the tool turn delivered an in-band error frame",
                    target.provider, model, frames=raw,
                    note=json.dumps(seen["errors"])[:300])
    if seen["finish"] != "tool_calls":
        raise _fail("a tool turn must finish with finish_reason='tool_calls'",
                    target.provider, model, frames=raw,
                    note=f"finish_reason={seen['finish']!r}")
    _assert_tool_calls(seen["tool_calls"], target, model, indexed=True,
                       frames=raw)

    # 2. non-stream tool turn
    data = _nonstream_json(live, target, model, {
        "messages": [{"role": "user", "content": _TOOL_PROMPT}],
        "tools": [_PROBE_TOOL], "tool_choice": "auto"})
    choices = data.get("choices") or [{}]
    if (choices[0] or {}).get("finish_reason") != "tool_calls":
        raise _fail("a non-stream tool turn must finish with "
                    "finish_reason='tool_calls'", target.provider, model,
                    status=200, body=json.dumps(data)[:400])
    message = (choices[0] or {}).get("message") or {}
    _assert_tool_calls(message.get("tool_calls"), target, model, indexed=False,
                       body=json.dumps(data)[:400])

    # 3. the follow-up turn: hand the result back and require a real answer.
    #    NOT memoized: the whole point is that this is a second turn over a
    #    different message history (the tool result has to be folded into the
    #    prompt), so it can never share a cache entry with the caller's turn.
    first = (message["tool_calls"])[0]
    status, text = _post_json(live, target, model, {"messages": [
        {"role": "user", "content": _TOOL_PROMPT},
        {"role": "assistant", "content": "",
         "tool_calls": [{
             "id": first["id"], "type": "function",
             "function": {"name": _PROBE_TOOL_NAME,
                          "arguments": first["function"]["arguments"]}}]},
        {"role": "tool", "tool_call_id": first["id"],
         "name": _PROBE_TOOL_NAME, "content": _TOOL_RESULT},
    ]})
    if status != 200:
        raise _fail("the follow-up turn after a tool result did not return 200",
                    target.provider, model, status=status, body=text)
    try:
        final = json.loads(text)
    except ValueError:
        raise _fail("the follow-up turn did not return JSON", target.provider,
                    model, status=status, body=text) from None
    final_message = ((final.get("choices") or [{}])[0].get("message") or {})
    if not str(final_message.get("content") or "").strip():
        raise _fail(
            "the follow-up turn returned no content: the tool result was not "
            "folded back into the prompt, so the agent loop is broken",
            target.provider, model, status=200,
            body=json.dumps(final)[:400])
    if (final.get("choices") or [{}])[0].get("finish_reason") != "stop":
        raise _fail("the follow-up turn did not finish with 'stop'",
                    target.provider, model, status=200,
                    body=json.dumps(final)[:400])


def _probe_catalogue(live: _Live) -> None:
    """``/v1/models`` must list what the active providers claim, and own it.

    Invariant: a well-formed OpenAI model list, and for every provider this
    run probes, its smoke model appears with ``owned_by`` naming that
    provider. A catalogue that has drifted from what the upstream actually
    serves is the failure that makes every other probe here lie -- a harness
    picks a model from this list, and a list that advertises a model the
    upstream no longer has is a 404 at the first real turn.
    """
    try:
        resp = live.client.get("/v1/models",
                               headers=_auth_headers(live.settings))
    except Exception as exc:  # noqa: BLE001 - reported, never raised raw
        raise _fail("/v1/models did not complete", "any", "-",
                    note=f"{type(exc).__name__}: {exc}") from None
    if resp.status_code != 200:
        raise _fail("/v1/models did not return 200", "any", "-",
                    status=resp.status_code, body=resp.text)
    payload = resp.json()
    if payload.get("object") != "list" or not isinstance(payload.get("data"), list):
        raise _fail("/v1/models is not an OpenAI-shaped model list", "any", "-",
                    status=200, body=json.dumps(payload)[:400])
    by_id: dict = {}
    for entry in payload["data"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise _fail("/v1/models contains a malformed entry", "any", "-",
                        status=200, body=json.dumps(payload["data"])[:400])
        if entry.get("object") != "model":
            raise _fail(f"/v1/models entry {entry['id']!r} is not object=model",
                        "any", "-", status=200,
                        body=json.dumps(entry)[:300])
        by_id[entry["id"]] = entry
    for target in live.targets:
        if target.chat_model not in by_id:
            raise _fail("/v1/models does not list a model the active provider "
                        "claims to serve", target.provider, target.chat_model,
                        status=200, body=json.dumps(sorted(by_id))[:400])
        owner = by_id[target.chat_model].get("owned_by")
        if owner != target.provider:
            raise _fail("/v1/models attributes a model to the wrong provider",
                        target.provider, target.chat_model, status=200,
                        body=json.dumps(by_id[target.chat_model])[:300],
                        note=f"owned_by={owner!r}")


# ---------------------------------------------------------------------------
# Per-provider entry points
# ---------------------------------------------------------------------------


def _run_provider_suite(pid: str) -> None:
    """Every probe for one provider, in cheapest-first order."""
    live = _live()
    target = _target(pid)
    probes = _Probes()
    probes.run("non-stream chat", lambda: _probe_nonstream_chat(live, target))
    probes.run("stream chat", lambda: _probe_stream_chat(live, target))
    probes.run("reasoning", lambda: _probe_reasoning(live, target))
    probes.run("tool round trip", lambda: _probe_tool_roundtrip(live, target))
    probes.run("model catalogue", lambda: _probe_catalogue(live))
    probes.report()


@pytest.mark.live
def test_live_smoke_deepseek():
    """DeepSeek (AWS WAF + wasm PoW): every wire-format invariant, in one run.

    Skips with the exact missing prerequisite: no bound account, no token, a
    banned pool, an unresolved ``obscura``, or an unsolved ``aws-waf-token``.
    """
    _run_provider_suite("deepseek")


@pytest.mark.live
def test_live_smoke_qwen():
    """Qwen (Aliyun WAF, JWT cookie, ``delta.phase`` SSE): the same invariants.

    Skips when no Qwen account is configured, when Obscura cannot plant the
    solved ``acw_tc`` jar, or when this run has no OpenAI chat model bound.
    """
    _run_provider_suite("qwen")


@pytest.mark.live
def test_live_smoke_opencode():
    """opencode (local ``opencode serve``, no account, no WAF): the invariants.

    The cheapest and most reliable probe in the file: no account, no browser,
    no WAF -- just a local process. Skips when the binary is absent and no
    ``DEEPSEAPORT_OPENCODE_BASE`` points at a running serve, or when the
    managed instance could not be booted.
    """
    _run_provider_suite("opencode")


# ---------------------------------------------------------------------------
# Leak safety: this file's own diagnostics
# ---------------------------------------------------------------------------


# Three credential shapes, because that is what the scrubber is built for
# (core/engine.py): DeepSeek's opaque 64-char userToken is one long run of
# token characters; Qwen's is a JWT whose dot-framed segments are each too
# short for the run rule and are caught by the framed rule instead. An
# upstream error body is exactly where any of them can appear, because the
# request that failed is the one that carried it.
_FAKE_USER_TOKEN = "sk-" + "A1b2C3d4E5f6G7h8" * 4
_FAKE_JWT = ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
             ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6InF3ZW4ifQ"
             ".dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1gFWFOEjXk")
_FAKE_API_KEY = "sk-ant-api03-" + "Ff09AaBbCcDdEeFf" * 3

# Every secret the leak test asserts is ABSENT from a diagnostic, not just the
# one driving it: a formatter that special-cases the shape under test while
# leaking a different one has moved the problem, not fixed it.
_FAKE_SECRETS = (_FAKE_USER_TOKEN, _FAKE_JWT, _FAKE_API_KEY)


class _SyntheticUpstreamError(Exception):
    """An exception shaped like the ones the engine raises from a real body.

    The realistic sink is not this file's formatting helpers: it is pytest
    rendering an exception that escaped a probe, and any ``str(exc)`` /
    ``exc.detail`` interpolation into a diagnostic. So the test drives the
    exception's own message through the formatters too.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
        self.detail = {"upstream": message}


def _diag_paths(secret: str) -> list[str]:
    """Every message this file can put in front of a human, for one secret.

    Enumerated rather than spot-checked, because the failure mode is
    asymmetric: a formatter that leaks puts a live credential in a log, while
    a formatter nobody exercised is a diagnostic that is merely absent. The
    paths mirror the probes exactly -- a body, an SSE frame dump, an
    exception's ``str()``/``detail`` (pytest renders both), and a note built
    from parsed content -- and every one of them is routed through ``_diag``.
    """
    exc = _SyntheticUpstreamError(
        f"HTTP_502: upstream said Authorization: Bearer {secret} and "
        f"Cookie: token={_FAKE_JWT}")
    body = json.dumps({"detail": {"code": 1002, "trace": secret,
                                  "msg": "generation_timeout: Server busy"}})
    frames = (f'data: {{"p":"response/content","v":"{secret}"}}\n\n'
              f'data: {{"error":{{"message":"{_FAKE_JWT}"}}}}\n\n'
              f"data: [DONE]\n\n")
    seen = _collect_stream(frames)
    return [
        # the full diagnostic, with every evidence field populated at once
        _diag("synthetic drift", "deepseek", "deepseek-flash", status=502,
              body=body, note="checked while probing", frames=frames),
        # a bare upstream body (the non-stream probes' shape)
        _diag("synthetic drift", "qwen", "qwen3.7-plus", body=body),
        # a streamed excerpt (the stream probes' shape)
        _diag("synthetic drift", "qwen", "qwen3.7-plus", frames=frames),
        # an exception that escaped a probe, as pytest would render it
        _diag("synthetic drift", "opencode", "opencode/big-pickle",
              body=exc.message),
        # FastAPI's HTTPException.detail is untyped, so a dict is legal and a
        # probe may well quote the whole thing
        _diag("synthetic drift", "opencode", "opencode/big-pickle",
              body=json.dumps(exc.detail)),
        # a note built from a parsed field (the in-band error frame path)
        _diag("synthetic drift", "deepseek", "deepseek-flash",
              note=json.dumps(seen["errors"])[:300]),
        # the gate's own reasons, which quote exception text
        _redact(_SyntheticUpstreamError(f"boom {secret}"), 200),
        # a raw body that reached a note field unwrapped by a future edit
        _diag("synthetic drift", "deepseek", "deepseek-flash", note=body),
    ]


@pytest.mark.parametrize("secret", _FAKE_SECRETS,
                         ids=["deepseek-user-token", "qwen-jwt",
                              "api-key"])
def test_live_smoke_diagnostics_cannot_leak_a_credential(secret):
    """A live failure must be diagnosable without ever printing a credential.

    The project already pins the scrubber itself
    (``tests/test_secret_and_leak_gaps.py``, and the AST scan at
    ``tests/test_managed_serve_hardening.py:436`` that forbids an unscrubbed
    variant of a message). That leaves one hole those tests cannot see: THIS
    file formats upstream bodies, SSE frames and exception text into
    assertion messages by hand, and a live failure lands in whatever log the
    maintainer's run produced. So the same policy is asserted about this
    file's own formatters, over every path a drift report can take.

    Deliberately NOT marked ``live``: it is a pure string-level test over
    synthetic inputs, needs no credentials and makes no network call. Marking
    it ``live`` for file uniformity would have put the only test guarding THIS
    file's formatters outside CI, so a future edit that dropped ``_diag``'s
    scrubber would pass every ordinary run and only surface for whoever
    happened to run the live tier after a real leak.
    """
    for text in _diag_paths(secret):
        assert isinstance(text, str)
        for other in _FAKE_SECRETS:
            assert other not in text, (
                f"a live diagnostic printed a credential verbatim "
                f"({other[:6]}...): {text[:200]!r}")

    # The other half of the policy: redaction must not eat the evidence. The
    # scrubber is deliberately narrow (that is why it is trustworthy), so the
    # words an operator actually diagnoses drift from have to survive -- the
    # status, the upstream error code and wording, and the SSE path the deltas
    # arrived on. If any of these vanish, the suite still passes but stops
    # being actionable, which is the whole point of the diagnostics.
    full = _diag_paths(secret)[0]
    assert "[redacted]" in full, "the diagnostic did not redact anything"
    for word in ("synthetic drift", "provider=deepseek", "deepseek-flash",
                 "http_status=502", "generation_timeout", "Server busy",
                 "response/content", "data: [DONE]"):
        assert word in full, (
            f"the scrubber ate {word!r}, which is what an operator diagnoses "
            f"drift from: {full[:300]!r}")
    # The same for the exception path, whose status prefix is part of the
    # message the engine builds (`error_text` prefixes every attempt error).
    exc_diag = _diag("synthetic drift", "opencode", "opencode/big-pickle",
                     body=f"HTTP_502: upstream said token={secret}")
    assert "HTTP_502" in exc_diag, (
        f"the scrubber ate the upstream status: {exc_diag!r}")


def test_live_smoke_gate_reasons_are_actionable():
    """Every gate decision must name the provider AND the missing prerequisite.

    A live suite that skips with "no live provider is ready" leaves the
    maintainer guessing which of three upstreams to go fix. The gate's
    decision functions are pure over a ``Settings``, so their wording is
    asserted here against synthetic settings instead of requiring a
    deliberately broken machine.

    Not marked ``live`` for the same reason as the leak-safety test above:
    pure, hermetic, and it must keep running in CI so the skip messages stay
    trustworthy.
    """
    from deepseaport.config import AccountConfig, Settings

    def _settings(accounts) -> Settings:
        return Settings(keys=["sk-live"], accounts=accounts, config_path="",
                        warmup_on_startup=False, log_level="WARNING")

    # The three distinct reasons, in the order the gate can reach them:
    # nothing bound to this provider (an account pinned to another upstream
    # does not count -- provider-scoped pools keep failover inside a provider),
    # bound but without a token, and bound with a token that is banned.
    cases = [
        ([], "no account is bound", "accounts[].provider"),
        ([AccountConfig(email="a@x.com", provider="qwen")],
         "no account is bound", "accounts[].provider"),
        ([AccountConfig(email="a@x.com", provider="deepseek")],
         "no deepseek account has a token", "deepseaport login"),
        ([AccountConfig(email="a@x.com", token="t" * 64, banned=True)],
         "banned", "unblock"),
    ]
    for accounts, fragment, hint in cases:
        usable, why = _ready_accounts(_settings(accounts), "deepseek")
        assert usable == [], f"accounts {accounts!r} were accepted for deepseek"
        assert fragment in why, f"reason does not name the blocker: {why!r}"
        assert hint in why, f"reason carries no fix: {why!r}"

    # A usable account is accepted, and the ban window is respected: a ban
    # whose expiry has passed is exactly what `banned_until` is for, and
    # refusing to serve it would be a second, quieter drift.
    fresh = _settings([AccountConfig(email="a@x.com", token="t" * 64)])
    usable, why = _ready_accounts(fresh, "deepseek")
    assert len(usable) == 1 and not why, why
    expired = _settings([AccountConfig(email="a@x.com", token="t" * 64,
                                       banned=True, banned_until=1.0)])
    usable, why = _ready_accounts(expired, "deepseek")
    assert len(usable) == 1, f"an expired ban still blocks the probe: {why!r}"

    # The binary check: `_discover` returns the bare string "obscura" when it
    # finds nothing, so a plausible command name must not read as a binary.
    assert _binary_ok("") is False
    assert _binary_ok("definitely-not-a-real-binary-xyz") is False
    assert _binary_ok(sys.executable) is True
