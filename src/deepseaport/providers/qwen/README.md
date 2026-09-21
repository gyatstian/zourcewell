# Qwen provider template

Template only: `models()` is empty, `login`/`complete` raise `NotImplementedError`.

## What to reverse

- Chat home URL + session-create endpoint + request/response shape.
- Completion/SSE envelope (event names, delta paths, finish/error signals).
- Auth token location (e.g. localStorage key; DeepSeek uses `userToken` JSON).
- WAF/PoW presence (headers, challenge endpoints, solver).

## Shared modules to reuse

- `core.tools` — prompt rendering + tool-call parsing.
- `core.transport` — SSE drain helpers.
- `core.openai._openai_response` — OpenAI response shaping.
- `deepseaport.accounts` pool (`core.pools.make_pool`) — per-provider account pools.

## Activate in 4 steps

1. Fill `models.py` (`MODELS_SPEC`, `DEFAULT_MODEL`).
2. Implement `protocol.py`/`client.py` (+ `sse_parser.py`, `pow.py`, `auth.py` as needed).
3. Override `complete_request` / `stream_response` in `provider.py` reusing
   `core.tools`, `core.transport`, `core.openai._openai_response` and
   `core.pools.make_pool` (`deepseaport.accounts` pool).
4. Register: add `"deepseaport.providers.qwen"` to `_BUILTIN_MODULES` in `core/registry.py`, then select it in the TUI provider menu.

## Multi-provider mode

Once registered, multi-provider mode picks it up automatically: models union,
per-provider pools, failover stays within provider.
