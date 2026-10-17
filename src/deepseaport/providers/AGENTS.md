# providers

## Purpose

The contract every upstream implements, plus the three upstreams this project
ships. One provider describes one upstream as data (a model catalogue) plus
delegating methods that hand the real work to its own `completion_service`.

## Ownership

- `deepseek/`: the chat.deepseek.com web upstream. Child doc.
- `qwen/`: the chat.qwen.ai web upstream. Child doc.
- `opencode/`: a locally running `opencode serve` instance. Child doc.
- The ABC they implement lives one level up in `core/provider.py`, which `core/AGENTS.md` owns.

## Local Contracts

- `models()` returns `{id: ModelSpec}`. `model_spec()` is a case-insensitive lookup; `tuple_spec()` flattens the catalogue to the compat `MODELS` shape. `models_hint()` is the MENU view of the same catalogue and must do NO I/O: the default is `models()` (a plain in-process read for deepseek and qwen), opencode overrides it with its last catalogue read else the static fallback, because its `models()` reads the serve and BOOTS the managed instance when nothing answers -- which a menu row must never pay for. A hint may therefore be older than `models()`. There is no `default_model()`/`accepts_model()` on the ABC: a blank id is resolved where it is read (`opencode.protocol.split_model` falls back to `models.DEFAULT_MODEL`; the HTTP body default `deepseek-flash` lives in `server_routes._prepare`).
- `requires_accounts = False` skips slot acquisition and passes `item=None` to the engine; preflight passes when only such providers are active. `needs_bridge = False` skips only the one-shot-chat bridge. The startup Obscura warmup (CHAT_HOME / DeepSeek-host fetch) is gated by `needs_waf_jar` -- ABC default True, qwen and opencode override False -- and every active provider's own `warmup()` still runs.
- `complete()`, `complete_request()` and `stream_response()` all raise `NotImplementedError` in `core/provider.py`. There is NO default engine, and the ABC holds no `_server()` reference to late-bind, so an override is not optional: a provider that skips one ships a request whose first call is a 500. All three providers implement all three.
- `complete_request(app, pool, prep, item, cancel_event)` is the HTTP non-stream route's seam (`server_routes` passes `cancel_event` POSITIONALLY). `complete(app, pool, prep, first_item, cancel_event=None, on_content=None, on_thinking=None)` is `cli.cmd_chat`'s, because one-shot chat has no cancellation source: `cancel_event` defaults to `None` and is forwarded as a KEYWORD to the same `_complete_with_failover_sync`. Both have live callers today; neither is dead.
- The retry table's partial-emit guard belongs to `core/AGENTS.md` (`LiveEmitTracker`): every row of every provider's `drive_retries` table goes through the local `_retryable(kind)` predicate, which ANDs `not tracker.emitted`. It used to sit on the `overload` row alone. Add a row through `_retryable`, never beside it.
- There is deliberately no `stream()`. It once sat beside `stream_response()` with a near-identical signature, was never called, and forced a template author to guess which one to implement. Do not add a second streaming seam; the route layer calls `stream_response`. Add one only with a caller.
- `tuple_spec()` returns the compat `MODELS` shape, `{id: (thinking, search, model_type)}`.
- `probe_ban`, `login` and `warmup` are the account and auth seams. `warmup` defaults to a no-op.
- Every provider method delegates to its own `completion_service`, which is the canonical monkeypatch surface for tests. A provider never imports `deepseaport.server`.
- Each package exports `create_provider()` from `__init__.py` and registers nothing at import time. The registry does the importing.
- The three upstreams are genuinely different, not three dialects: DeepSeek is PoW plus Obscura plus a real browser, Qwen is a JWT cookie plus Aliyun plus no browser, opencode is a localhost process with no accounts at all. Keep the shared surface small enough that those three still fit.

## Work Guidance

- To add an upstream, follow `PROVIDERS.md` "Add a new provider in 4 steps". The registry entry is one line in `core/registry.py`.
- Anything that differs per provider (403 meaning, auth shape, SSE protocol, retry triggers) is passed in as data or an adapter. Never branch on provider id inside the engine.
- `PROVIDERS.md` and each provider's own `README.md` hold the long-form notes; update them with the code.

## Verification

- The hermetic gate is owned by the root AGENTS.md; run it before any change here.
- `tests/test_providers.py`, `tests/test_provider_dispatch.py`, `tests/test_provider_parity_gaps.py`, `tests/test_provider_account_rule.py` and `tests/test_multi_provider.py` cover the contract and the parity between upstreams.
- Run `python -m pytest -m live -q` after any upstream wire change. The opencode provider is the cheapest probe: no account, no browser, no WAF, just a local process.