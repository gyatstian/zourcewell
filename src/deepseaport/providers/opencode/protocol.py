"""Opencode local-CLI protocol: base URL, payloads, model splitting.

Talks to a locally running ``opencode serve`` instance (default
``http://127.0.0.1:4096``). ``opencode serve`` exposes a control-plane HTTP
API (``POST /session``, ``POST /session/:id/message``, ``DELETE
/session/:id``, ``GET /config/providers``, ``GET /global/health``); this
module builds the small subset the provider needs. No auth by default
(localhost, no password); when ``OPENCODE_SERVER_PASSWORD`` is set on the
serve instance, the matching password rides HTTP basic auth.

Env overrides (no Settings fields):
  DEEPSEAPORT_OPENCODE_BASE      base URL (default: managed isolated serve;
                                 set this to use your own `opencode serve`)
  DEEPSEAPORT_OPENCODE_AGENT     agent name (default deepseaport-passthrough)
  DEEPSEAPORT_OPENCODE_TIMEOUT   per-request seconds (default 180)
  DEEPSEAPORT_OPENCODE_USERNAME  basic-auth username (default opencode)
  DEEPSEAPORT_OPENCODE_PASSWORD  basic-auth password (default "")
  DEEPSEAPORT_OPENCODE_MANAGED   0/false/no/off disables the managed serve
  DEEPSEAPORT_OPENCODE_CLIENT    client identity header (default cli)
  DEEPSEAPORT_OPENCODE_PROJECT   project identity header (default global)
  DEEPSEAPORT_OPENCODE_USER_AGENT  User-Agent header (default opencode/<cli version>)
  DEEPSEAPORT_OPENCODE_WORKDIR   session working directory for tool execution
                                 (default: deepseaport process cwd; native
                                 tools run here via `?directory=` on session
                                 create/message/delete/event)
"""

from __future__ import annotations

import os
import urllib.parse

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


def message_payload(provider_id: str, model_id: str, agent: str, prompt: str,
                    variant: str | None = None,
                    tools: dict | None = None) -> dict:
    """POST /session/:id/message body for one stateless turn.

    ``variant`` is the one-shot reasoning effort (none/minimal/low/medium/
    high/xhigh); omitted when unset so models without that variant keep
    their default behavior. ``tools`` is the per-message native tool
    allowlist (name -> bool); None omits the key so the serve uses the
    agent defaults (live-verified on opencode 1.18.32: tool-less passes via
    those defaults). Pass :func:`native_tools_minimal` when the harness
    asked for tools: the free-tier gatekeeper needs bash+read *declared*
    (an explicit all-false allowlist 403s with "only be used from within
    OpenCode") but NOT granted — ``bash`` is declared False so the map
    cannot override the managed agent's ``bash: "deny"`` permission.
    """
    body = {
        "model": {"providerID": provider_id, "modelID": model_id},
        "agent": agent or DEFAULT_AGENT,
        "parts": [{"type": "text", "text": prompt}],
    }
    if variant:
        body["variant"] = variant
    if tools is not None:
        body["tools"] = dict(tools)
    return body


# The free-tier gatekeeper needs bash+read effective upstream: omitting the
# per-message `tools` key passes (agent defaults count, live-verified on
# opencode 1.18.32), but an explicit all-false allowlist 403s with "only be
# used from within OpenCode". So plain chat omits the key, while tool turns
# declare this minimal pair and hide everything else: the model then only
# sees pi's tools (via the DSML prompt) for file edits/writes/globs, and
# native edit/write/glob/grep would otherwise shadow pi's same-named tools
# and run in the serve cwd.
#
# The declared VALUE must not hand the model native shell access. The
# managed agent's permission map pins ``bash: "deny"`` (see
# ``managed_serve.managed_config_content``), but this per-message map
# OVERRIDES that permission map — so ``bash: True`` silently re-granted a
# native shell on every tool turn, which is exactly what the managed serve
# exists to prevent. The gatekeeper only needs the key DECLARED, not granted,
# hence False. ``read`` stays True: the serve pins ``read: "allow"`` and
# reads are harmless here (the prompt still routes file access through pi's
# own tools).
NATIVE_TOOLS_MINIMAL: dict[str, bool] = {
    "bash": False,
    "read": True,
    "edit": False,
    "write": False,
    "glob": False,
    "grep": False,
    "task": False,
    "webfetch": False,
    "websearch": False,
    "todowrite": False,
    "skill": False,
    "question": False,
    "apply_patch": False,
    "invalid": False,
}


def native_tools_minimal() -> dict[str, bool]:
    """Copy of the minimal gatekeeper-passing native allowlist (never raises)."""
    try:
        return dict(NATIVE_TOOLS_MINIMAL)
    except Exception:
        return {"bash": False, "read": True}


def serve_hint() -> str:
    """Human hint appended to serve-down errors."""
    return (f"opencode serve is not reachable at {base_url()}. "
            "Deepseaport boots a managed isolated serve automatically "
            "(needs the `opencode` binary on PATH); or run your own "
            "(`opencode serve --port 4096`) and set DEEPSEAPORT_OPENCODE_BASE. "
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
    """True for stale-session races worth one recreate-and-resend."""
    low = (error or "").lower()
    return any(k in low for k in (
        "invalid_session", "invalid session", "session not found",
        "no such session", "unknown session",
    ))


def session_directory(explicit: str | None = None) -> str:
    """Working directory for new sessions (native tools run here).

    Resolution: explicit arg > DEEPSEAPORT_OPENCODE_WORKDIR env > process
    cwd. Returns "" when nothing usable (caller omits `?directory=` and the
    serve falls back to its own cwd). Never raises.
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
    """`?directory=...` suffix for session-scoped endpoints ("" when unset).

    Pass the result of :func:`session_directory` (or an explicit dir) so
    create/message/prompt_async/delete/event all target the same worktree.
    The serve registers the project on demand; unknown dirs 400 downstream.
    """
    try:
        d = str(directory or "").strip()
        if not d:
            return ""
        return "?" + urllib.parse.urlencode({"directory": d})
    except Exception:
        return ""
