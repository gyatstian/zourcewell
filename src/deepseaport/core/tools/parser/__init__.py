"""Parser orchestrator: parse_tool_calls (first match wins)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..common import (
    MAX_ARGS_CHARS,
    MAX_BARE_OBJECTS,
    _looks_like_tool_attempt,
    _recover_loose_openai_call,
    _resolve_forgiving,
    _valid_names,
)
from .bare import try_parse_bare
from .dsml import try_parse_dsml
from .fence import try_parse_fence
from .json_array import try_parse_json_array
from .tags import try_parse_tags

logger = logging.getLogger("deepseaport.tools")


@dataclass
class ParseResult:
    """Explicit parse outcome: calls + what was dropped + per-format trail.

    ``calls``/``rest`` are the historical ``parse_tool_calls`` values.
    ``dropped`` holds the winning format's ``_normalize_calls`` drop
    indexes (unknown names, oversize args); ``diagnostics`` holds one
    ``{"format", "matched", "kept", "dropped", "raw"}`` entry per
    format that produced raw candidates — including formats whose
    candidates were ALL dropped, so a ``(None, text)`` miss still says
    which format almost matched and why nothing survived.
    """

    calls: list[dict] | None
    rest: str
    dropped: list[int] = field(default_factory=list)
    diagnostics: list[dict] = field(default_factory=list)


# Grammar table: the documented first-match-wins order (see
# parse_tool_calls docstring). One row per format keeps the chain explicit
# instead of five copy-pasted if-blocks; keep in sync when adding a format.
PARSE_CHAIN = ("json_array", "tags", "dsml", "fence", "bare")

_CHAIN_FNS = {
    "json_array": try_parse_json_array,
    "tags": try_parse_tags,
    "dsml": try_parse_dsml,
    "fence": try_parse_fence,
    "bare": try_parse_bare,
}


def parse_tool_calls(
    text: str,
    tools: list[dict] | None = None,
    *,
    max_args_chars: int | None = None,
    max_bare_objects: int | None = None,
    forgiving: bool | None = None,
) -> tuple[list[dict] | None, str]:
    """Return (tool_calls or None, remaining_text).

    Accepted formats (first match wins):
    1. {"tool_calls": [...]} — balanced scan, Thought prefix tolerated.
    2. <tool_call>/<function_call>/<invoke> JSON tags.
    3. <||DSML|| invoke/parameter blocks (ASCII + fullwidth variants).
    4. fenced ```json block describing a call.
    5. bare unfenced {"name":..., "arguments":...} (valid-names gated).

    Plain answers return (None, original_text) unchanged.

    Limits are tunable: `max_args_chars` defaults to MAX_ARGS_CHARS
    (env DEEPSEAPORT_MAX_ARGS_CHARS), `max_bare_objects` defaults to
    MAX_BARE_OBJECTS (env DEEPSEAPORT_MAX_BARE_OBJECTS).

    Forgiving bucket (`forgiving=True`, env DEEPSEAPORT_FORGIVING_TOOLCALLS,
    or Settings.forgiving_toolcalls): extra repairs for model slips.
    Strict (default) keeps historical behaviour. Forgiving repairs:
    1. single/zero-pipe, spaced and HTML-escaped DSML markers;
    2. smart quotes / fullwidth ＝ / entities as attribute delimiters
       (parameter values stay verbatim);
    3. missing invoke name when exactly one tool is registered;
    4. nameless parameter assigned to a lone required/single property;
    5. JSON-object invoke body with no parameter tags used as arguments.
    Markup only: values are never invented, truncated calls never executed.
    """
    res = parse_tool_calls_ex(text, tools, max_args_chars=max_args_chars,
                              max_bare_objects=max_bare_objects,
                              forgiving=forgiving)
    return res.calls, res.rest


def parse_tool_calls_ex(
    text: str,
    tools: list[dict] | None = None,
    *,
    max_args_chars: int | None = None,
    max_bare_objects: int | None = None,
    forgiving: bool | None = None,
) -> ParseResult:
    """Explicit variant: same parse, but drops/diagnostics are returned.

    ``parse_tool_calls`` keeps the historical ``(calls, rest)`` seam (all
    existing callers + monkeypatches untouched); this adds the opt-in
    explicit surface on top.
    """
    forgiving_on = _resolve_forgiving(forgiving)
    valid = _valid_names(tools) if tools else set()
    valid_or_none = valid if tools else None
    limit_args = max_args_chars if max_args_chars is not None else MAX_ARGS_CHARS
    limit_bare = max_bare_objects if max_bare_objects is not None else MAX_BARE_OBJECTS

    diagnostics: list[dict] = []
    for name in PARSE_CHAIN:
        _parse = _CHAIN_FNS[name]
        if name == "dsml":
            calls, rest = _parse(text, valid_or_none, limit_args, forgiving_on,
                                 tools, diagnostics)
        elif name == "bare":
            calls, rest = _parse(text, valid, valid_or_none, limit_args,
                                 limit_bare, forgiving_on, diagnostics)
        else:
            calls, rest = _parse(text, valid_or_none, limit_args, forgiving_on,
                                 diagnostics)
        if calls:
            return ParseResult(calls=calls, rest=rest,
                               dropped=list(diagnostics[-1].get("dropped", []))
                               if diagnostics else [],
                               diagnostics=diagnostics)

    recovered = _recover_loose_openai_call(text, valid_or_none, limit_args)
    if recovered:
        logger.info("recovered loose OpenAI tool call (%d call(s))", len(recovered))
        diagnostics.append({"format": "recover_loose", "matched": True,
                            "kept": len(recovered), "dropped": [],
                            "raw": len(recovered)})
        return ParseResult(calls=recovered, rest="", dropped=[],
                           diagnostics=diagnostics)

    if tools and _looks_like_tool_attempt(text, valid):
        logger.debug("tool parse miss preview=%.300s", text[:300])
    return ParseResult(calls=None, rest=text, dropped=[],
                       diagnostics=diagnostics)

