"""Tool support for harnesses: prompt injection + robust call parsing.

The web model has no native function calling, so we describe tools in a
system prompt and parse the model's structured reply back into OpenAI
`tool_calls`. DSML raw-value blocks are preferred for large/file arguments;
JSON forms are accepted for short calls. Parsing tries (first match wins):

1. {"tool_calls": [{"id":..,"type":"function","function":{"name":..,"arguments":..}}]}
2. <tool_call>{"name":..,"arguments":{..}}</tool_call>  (aliases: function_call, invoke)
3. DSML invoke/parameter blocks, ASCII + fullwidth, named or generic close tags.
4. ```json fenced object with tool_calls / name+arguments keys.
5. Bare unfenced {"name":..,"arguments":..} (valid-names gated).
6. Last-resort recovery of unescaped OpenAI-style `arguments`.

Small model-output damage is repaired (raw control chars in strings, bare
keys, trailing commas, Python literals). Individual argument size defaults
to 200k chars (Settings.tool_args_max_chars / DEEPSEAPORT_MAX_ARGS_CHARS).
Opt-in Settings.forgiving_toolcalls / DEEPSEAPORT_FORGIVING_TOOLCALLS enables
the forgiving bucket (markup repairs: pipe/space/HTML-escaped markers, smart
attribute delimiters, missing-name/param inference, JSON invoke bodies);
strict is default.

Split layout (no behavior change):
- :mod:`deepseaport.core.tools.prompt` — render_prompt/tool_system_prompt/tool_reminder_prompt
- :mod:`deepseaport.core.tools.common` — shared base (env, lenient JSON, normalize)
- :mod:`deepseaport.core.tools.parser` — parse_tool_calls + format plugins
  (json_array, tags, dsml, fence, bare); forgiving stays opt-in.
This package re-exports the historical ``core.tools`` surface so existing
``from .core.tools import ...`` and ``tools_support`` alias keep working.
"""

from __future__ import annotations

import logging

from .common import (
    _FENCED_BLOCK_RE,
    _LOOSE_JSON_MISS,
    _OversizeArgs,
    _THOUGHT_LINE_RE,
    MAX_ARGS_CHARS,
    MAX_BARE_OBJECTS,
    _balanced_json,
    _bool_env,
    _clean_rest,
    _escape_control_chars_in_strings,
    _int_env,
    _loads_json_lenient,
    _looks_like_tool_attempt,
    _normalize_args,
    _normalize_calls,
    _normalize_dsml_text,
    _quote_unquoted_keys,
    _recover_loose_openai_call,
    _remove_trailing_commas,
    _replace_python_literals,
    _resolve_forgiving,
    _strip_spans,
    _valid_names,
    looks_truncated_tool_attempt,
)
from .parser import PARSE_CHAIN, ParseResult, parse_tool_calls, parse_tool_calls_ex
from .parser.bare import _parse_bare_name_objects
from .parser.dsml import (
    _DS,
    _GENERIC_CLOSE_FORGIVING_RE,
    _GENERIC_CLOSE_RE,
    _GT,
    _HTML_ATTRS,
    _HTML_VALUES,
    _INVOKE_CLOSE_FORGIVING_RE,
    _INVOKE_CLOSE_RE,
    _INVOKE_OPEN_FORGIVING_RE,
    _INVOKE_OPEN_RE,
    _LT,
    _PARAM_CLOSE_FORGIVING_RE,
    _PARAM_CLOSE_RE,
    _PARAM_OPEN_FORGIVING_RE,
    _PARAM_OPEN_RE,
    _SMART_QUOTES,
    _SP,
    _assign_unnamed_param,
    _coerce_dsml_value,
    _dsml_spans,
    _find_close,
    _invoke_open_re,
    _param_open_re,
    _parse_dsml_attrs,
    _parse_dsml_calls,
    _preprocess_dsml_attrs,
    _scan_dsml_calls,
    _tag_re,
    _tool_param_info,
    _unescape_dsml_value,
)
from .prompt import (
    _example_param_value,
    _message_text,
    render_prompt,
    tool_reminder_prompt,
    tool_system_prompt,
)

logger = logging.getLogger("deepseaport.tools")

__all__ = [
    "MAX_ARGS_CHARS",
    "MAX_BARE_OBJECTS",
    "logger",
    "_int_env",
    "_bool_env",
    "_resolve_forgiving",
    "_valid_names",
    "_example_param_value",
    "tool_system_prompt",
    "tool_reminder_prompt",
    "_message_text",
    "render_prompt",
    "_balanced_json",
    # _LOOSE_JSON_MISS stays importable (parser submodules + compat) but is
    # intentionally absent from __all__: star-imports should not get the
    # miss sentinel; explicit imports keep working.
    "_escape_control_chars_in_strings",
    "_remove_trailing_commas",
    "_quote_unquoted_keys",
    "_replace_python_literals",
    "_loads_json_lenient",
    "_OversizeArgs",
    "_normalize_args",
    "_normalize_calls",
    "_THOUGHT_LINE_RE",
    "_clean_rest",
    "_normalize_dsml_text",
    "_coerce_dsml_value",
    "_tag_re",
    "_invoke_open_re",
    "_param_open_re",
    "_INVOKE_OPEN_RE",
    "_INVOKE_OPEN_FORGIVING_RE",
    "_PARAM_OPEN_RE",
    "_PARAM_OPEN_FORGIVING_RE",
    "_INVOKE_CLOSE_RE",
    "_INVOKE_CLOSE_FORGIVING_RE",
    "_PARAM_CLOSE_RE",
    "_PARAM_CLOSE_FORGIVING_RE",
    "_GENERIC_CLOSE_RE",
    "_GENERIC_CLOSE_FORGIVING_RE",
    "_SMART_QUOTES",
    "_HTML_ATTRS",
    "_HTML_VALUES",
    "_preprocess_dsml_attrs",
    "_unescape_dsml_value",
    "_tool_param_info",
    "_assign_unnamed_param",
    "_parse_dsml_attrs",
    "_find_close",
    "_scan_dsml_calls",
    "_parse_dsml_calls",
    "_dsml_spans",
    "_parse_bare_name_objects",
    "_strip_spans",
    "PARSE_CHAIN",
    "ParseResult",
    "parse_tool_calls_ex",
    "_looks_like_tool_attempt",
    "looks_truncated_tool_attempt",
    "_FENCED_BLOCK_RE",
    "_recover_loose_openai_call",
    "parse_tool_calls",
    "_LT",
    "_GT",
    "_DS",
    "_SP",
]
