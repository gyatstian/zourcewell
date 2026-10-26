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
to 200k chars (Settings.tool_args_max_chars / ZOURCEWELL_MAX_ARGS_CHARS).
Opt-in Settings.forgiving_toolcalls / ZOURCEWELL_FORGIVING_TOOLCALLS enables
the forgiving bucket (markup repairs: pipe/space/HTML-escaped markers, smart
attribute delimiters, missing-name/param inference, JSON invoke bodies);
strict is default.

Split layout (no behavior change):
- :mod:`zourcewell.core.tools.prompt` — render_prompt/tool_system_prompt/tool_reminder_prompt
- :mod:`zourcewell.core.tools.common` — shared base (env, lenient JSON, normalize)
- :mod:`zourcewell.core.tools.parser` — parse_tool_calls + format plugins
  (json_array, tags, dsml, fence, bare); forgiving stays opt-in.
This package re-exports the historical ``core.tools`` names that still have
importers; everything else lives in the module that defines it.
"""

from __future__ import annotations

from .common import (
    MAX_ARGS_CHARS,
    _OversizeArgs,
    _balanced_json,
    _clean_rest,
    _int_env,
    _looks_like_tool_attempt,
    _normalize_args,
    _normalize_calls,
    _normalize_dsml_text,
    _recover_loose_openai_call,
    _strip_spans,
    _valid_names,
    looks_truncated_tool_attempt,
)
from .parser import parse_tool_calls
from .parser.dsml import (
    _coerce_dsml_value,
    _scan_dsml_calls,
)
from .prompt import (
    _message_text,
    render_prompt,
    tool_reminder_prompt,
    tool_system_prompt,
)

__all__ = [
    "MAX_ARGS_CHARS",
    "_int_env",
    "_valid_names",
    "tool_system_prompt",
    "tool_reminder_prompt",
    "_message_text",
    "render_prompt",
    "_balanced_json",
    "_OversizeArgs",
    "_normalize_args",
    "_normalize_calls",
    "_clean_rest",
    "_normalize_dsml_text",
    "_coerce_dsml_value",
    "_scan_dsml_calls",
    "_strip_spans",
    "_looks_like_tool_attempt",
    "looks_truncated_tool_attempt",
    "_recover_loose_openai_call",
    "parse_tool_calls",
]
