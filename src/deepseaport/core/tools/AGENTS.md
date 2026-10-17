# core.tools

## Purpose

Makes a web model with no native function calling behave like one that has it:
tool schemas are rendered into the system prompt, and the model's structured
reply is parsed back into OpenAI `tool_calls`.

## Ownership

- `prompt.py`: renders the tool system prompt and the per-turn reminder.
- `common.py`: env resolution, limits, lenient JSON repair, name validation, `normalize`.
- `parser/__init__.py`: the orchestrator, `parse_tool_calls`, first match wins.
- `parser/json_array.py`: format 1, a balanced scan of `{"tool_calls": [...]}`.
- `parser/tags.py`: format 2, XML-style tag names for calls, parameters and invoke.
- `parser/dsml.py`: format 3, DSML and plain invoke+parameter blocks. Preferred for large or file arguments.
- `parser/fence.py`: format 4, a fenced ```json block describing one call.
- `parser/bare.py`: format 5, bare unfenced name/arguments objects.

## Local Contracts

- Parsing is first match wins, in the order above. Add a new format at the end of the chain; inserting earlier shadows the formats above it and silently changes which replies parse.
- DSML raw-value blocks are preferred for large or file arguments; the JSON forms are accepted for short calls.
- `MAX_ARGS_CHARS` / `DEEPSEAPORT_MAX_ARGS_CHARS` (default 200000) caps the arguments of a single call.
- `MAX_BARE_OBJECTS` / `DEEPSEAPORT_MAX_BARE_OBJECTS` (default 40) bounds how many JSON objects the bare name+arguments scan inspects, and the forgiving bucket's total-call scan reuses the same number.
- `DEEPSEAPORT_FORGIVING_TOOLCALLS` (default off) enables the markup-repair bucket.
- Import limits from `common.py`. Never redefine a bound inside a parser module; two copies drift and one of them stops being enforced.
- The package `__init__` re-exports only the historical `core.tools` names that still have importers; import everything else from `common.py`, `parser/` or `prompt.py`. Adding a name back needs an importer.
- These constants are captured at import time, which is why `tests/conftest.py` has to steer the environment before collection.
- `role: tool` results are folded back into the web prompt. That fold is what makes multi-step agentic loops work against a model with no function-calling API.
- Every parser is a pure function over text: no I/O, no network, no provider imports.

## Work Guidance

- Keep repair local to the format that needs it. The parsers exist because a web model emits the same call in five different shapes; a shared normalizer that guesses harder is how a reply gets truncated.
- Tool call ids must stay stable across the streamed deltas and the final buffered parse.

## Verification

- The hermetic gate is owned by the root AGENTS.md; run it before any change here.
- `tests/test_tools_gaps.py`, `tests/test_tool_and_provider_gaps.py`, `tests/test_tool_role_and_allowlist.py` and `tests/test_stream_protocol_index.py` cover this folder.
- Run `python -m pytest -m live -q` after changing a prompt or parser bound. The live suite asserts a reply contains the end of the prompt, which is the check that catches a parser silently truncating output.