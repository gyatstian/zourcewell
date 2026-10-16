"""Opencode local-CLI protocol: base URL, payloads, model splitting.

Talks to a locally running ``opencode serve`` instance (default
``http://127.0.0.1:4096``). The v2 HttpApi (opencode 2.x) serves everything
under ``/api`` and no longer has a blocking message endpoint; the subset the
provider needs is:

  GET    /api/info                                   -> alive
  POST   /api/session  {location?}                   -> {data:{id}}
  POST   /api/session/:id/model  {model}            -> 204 (session-scoped)
  POST   /api/session/:id/agent  {agent}            -> 204 (session-scoped)
  POST   /api/session/:id/prompt {text}             -> {data:{id}} at once
  GET    /api/session/:id/message?limit=&order=     -> {data:[items]}
  POST   /api/session/:id/interrupt {}              -> abort
  DELETE /api/session/:id                           -> 204
  GET    /api/event                                  -> SSE deltas
  GET    /api/model                                  -> dynamic catalogue

Success is enveloped as ``{"data": ...}``; failures are a flat
``{"_tag": "...", "message": "..."}`` object (no envelope). Since opencode
v2 a serve instance always requires HTTP basic auth: it generates a password at
boot and prints it, unless ``OPENCODE_SERVER_PASSWORD`` pins one. The managed
instance sets that env and passes the same value here; for your own serve,
mirror its password in ``DEEPSEAPORT_OPENCODE_PASSWORD``.

Env overrides (no Settings fields):
  DEEPSEAPORT_OPENCODE_BASE      base URL (default: managed isolated serve;
                                 set this to use your own `opencode serve`)
  DEEPSEAPORT_OPENCODE_AGENT     agent name (default deepseaport-passthrough)
  DEEPSEAPORT_OPENCODE_TIMEOUT   per-turn seconds (default 180)
  DEEPSEAPORT_OPENCODE_USERNAME  basic-auth username (default opencode)
  DEEPSEAPORT_OPENCODE_PASSWORD  basic-auth password (default "")
  DEEPSEAPORT_OPENCODE_MANAGED   0/false/no/off disables the managed serve
  DEEPSEAPORT_OPENCODE_CLIENT    client identity header (default cli)
  DEEPSEAPORT_OPENCODE_PROJECT   project identity header (default global)
  DEEPSEAPORT_OPENCODE_USER_AGENT  User-Agent header (default opencode/<cli version>)
  DEEPSEAPORT_OPENCODE_WORKDIR   session working directory for tool execution
                                 (default: deepseaport process cwd; sent as
                                 `location.directory` on session create)
"""

from __future__ import annotations

import os

from .managed_serve import MANAGED_AGENT

DEFAULT_BASE = "http://127.0.0.1:4096"
DEFAULT_AGENT = MANAGED_AGENT
DEFAULT_TIMEOUT = 180
DEFAULT_USERNAME = "opencode"


def base_url() -> str:
    """Serve base URL without a trailing slash (env-overridable)."""
    raw = (os.environ.get("DEEPSEAPORT_OPENCODE_BASE", "") or "").strip()
    return (raw or DEFAULT_BASE).rstrip("/") or DEFAULT_BASE


def agent_name() -> str:
    """Agent to run messages with (env-overridable, default build)."""
    return (os.environ.get("DEEPSEAPORT_OPENCODE_AGENT", "") or "").strip() or DEFAULT_AGENT


def opencode_timeout(default: int = DEFAULT_TIMEOUT) -> float:
    """Per-request timeout in seconds (env-overridable)."""
    try:
        return max(1.0, float(os.environ.get("DEEPSEAPORT_OPENCODE_TIMEOUT", default) or default))
    except (TypeError, ValueError):
        return float(default)


def basic_auth() -> tuple[str, str]:
    """(username, password) for serve basic auth; password "" means none."""
    user = (os.environ.get("DEEPSEAPORT_OPENCODE_USERNAME", "") or "").strip() or DEFAULT_USERNAME
    password = os.environ.get("DEEPSEAPORT_OPENCODE_PASSWORD", "") or ""
    return user, password


_cached_ua = ""


def _detected_user_agent() -> str:
    """`opencode/<cli version>` from the local binary (cached, never raises)."""
    global _cached_ua
    if _cached_ua:
        return _cached_ua
    version = ""
    try:
        import re
        import shutil
        import subprocess

        binary = shutil.which("opencode")
        if binary:
            proc = subprocess.run([binary, "--version"], capture_output=True,
                                  text=True, timeout=10)
            out = ((proc.stdout or "") + " " + (proc.stderr or "")).strip()
            match = re.search(r"(\d+\.\d+\.\d+[\w.\-]*)", out)
            if match:
                version = match.group(1)
    except Exception:
        version = ""
    _cached_ua = f"opencode/{version}" if version else "opencode/unknown"
    return _cached_ua


def client_identity() -> tuple[str, str, str]:
    """(user_agent, client, project) identity for serve requests."""
    ua = ((os.environ.get("DEEPSEAPORT_OPENCODE_USER_AGENT", "") or "").strip()
          or _detected_user_agent())
    client = ((os.environ.get("DEEPSEAPORT_OPENCODE_CLIENT", "") or "").strip()
              or "cli")
    project = ((os.environ.get("DEEPSEAPORT_OPENCODE_PROJECT", "") or "").strip()
               or "global")
    return ua, client, project


def client_headers(session_id: str | None = None) -> dict[str, str]:
    """Identity headers for one serve request.

    The serve binary sets the upstream fingerprint itself, but official
    clients/proxies also identify on the control plane; a bare
    ``Python-urllib`` UA marks the traffic as foreign. ``X-Session-ID``
    carries the serve-assigned session id when known (consistent with the
    serve's own ``x-opencode-session`` upstream value).
    """
    ua, client, project = client_identity()
    headers = {"User-Agent": ua, "x-opencode-client": client,
               "x-opencode-project": project}
    if session_id:
        headers["X-Session-ID"] = str(session_id)
    return headers


def split_model(model: str, default: str = "opencode/muse-spark-1.3-contributor-free") -> tuple[str, str]:
    """Split a deepseaport model id into (providerID, modelID).

    ``provider/model`` splits at the first slash (the model part may itself
    contain slashes, e.g. ``openrouter/anthropic/claude...``). A bare id
    without a slash maps to provider ``opencode``. Empty/blank falls back
    to the default model's split.
    """
    raw = str(model or "").strip() or default
    if "/" in raw:
        provider_id, _, model_id = raw.partition("/")
        provider_id = provider_id.strip() or "opencode"
        model_id = model_id.strip() or raw.strip()
        return provider_id, model_id
    return "opencode", raw


def message_payload(prompt: str) -> dict:
    """POST /api/session/:id/prompt body for one turn: ``{"text": prompt}``.

    v2 moved everything else off the message body. The model and its
    reasoning ``variant`` are SESSION-scoped (``POST /api/session/:id/model``;
    an unusable variant surfaces as a failed turn, not a 4xx on the prompt),
    the agent is session-scoped too (``POST /api/session/:id/agent``), and
    native tools have no per-message map at all: the agent config's permission
    map is the single authority for what the model may call (see
    ``managed_serve.managed_config_content``: bash/edit/webfetch denied, read
    allowed).
    """
    return {"text": str(prompt)}


def serve_hint() -> str:
    """Human hint appended to serve-down errors."""
    return (f"opencode serve is not reachable at {base_url()}. "
            "Deepseaport boots a managed isolated serve automatically "
            "(needs the `opencode` binary on PATH); or run your own "
            "(`opencode serve --port 4096`) and set DEEPSEAPORT_OPENCODE_BASE. "
            "opencode v2 prints a server password at boot: set it (or "
            "OPENCODE_SERVER_PASSWORD) on the serve and mirror it in "
            "DEEPSEAPORT_OPENCODE_PASSWORD. "
            "Free models need no login; `opencode auth login [--provider ...]` "
            "only for providers requiring keys.")


def is_serve_down_error(error: str | None) -> bool:
    """True when the error text means the local serve instance is down."""
    low = (error or "").lower()
    return any(k in low for k in (
        "opencode_serve_down", "serve_down", "connection refused",
        "errno 111", "winerror 10061", "[winerror 10061]",
        "failed to establish a new connection", "no connection could be made",
        "urlopen error", "max retries exceeded",
    ))


def is_free_tier_error(error: str | None) -> bool:
    """True for the upstream free-tier gatekeeper refusal (terminal, no retry)."""
    low = (error or "").lower()
    return ("only be used from within opencode" in low
            or "freetier" in low
            or ("free tier" in low and "403" in low))


def is_session_error(error: str | None) -> bool:
    """True for stale-session races worth one recreate-and-resend.

    v2 answers an unknown session with 404 and
    ``{"_tag": "SessionNotFoundError", "message": "Session not found: ..."}``,
    so both the tag and the prose are matched.
    """
    low = (error or "").lower()
    return any(k in low for k in (
        "invalid_session", "invalid session", "session not found",
        "sessionnotfound", "session_not_found", "no such session",
        "unknown session",
    ))


def session_directory(explicit: str | None = None) -> str:
    """Working directory for new sessions (native tools run here).

    Resolution: explicit arg > DEEPSEAPORT_OPENCODE_WORKDIR env > process
    cwd. Returns "" when nothing usable (the caller omits
    ``location.directory`` and the serve falls back to its own cwd). Never
    raises.
    """
    candidates: list[str] = []
    try:
        if explicit:
            candidates.append(str(explicit))
    except Exception:
        pass
    try:
        candidates.append((os.environ.get("DEEPSEAPORT_OPENCODE_WORKDIR", "") or "").strip())
    except Exception:
        pass
    try:
        candidates.append(os.getcwd())
    except Exception:
        pass
    for cand in candidates:
        try:
            if not cand:
                continue
            import pathlib

            p = pathlib.Path(cand).expanduser()
            if p.is_dir():
                return str(p.resolve())
        except Exception:
            continue
    return ""


def session_query(directory: str | None = None) -> str:
    """Always "" — v2 has no ``?directory=`` query parameter.

    Kept for callers that still thread a directory through: the working
    directory now travels as ``location.directory`` in the ``POST
    /api/session`` body (see :func:`session_directory`) and the session id
    alone addresses every other route, so no query string is built and no
    unknown parameter is sent.
    """
    return ""
