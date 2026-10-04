# core

## Purpose

Mechanics every provider shares: the retry driver, the cross-account failover
loop, the stream bridge, HTTP transport handling, OpenAI response shaping, the
provider registry, and the browser bridges. Nothing here knows about DeepSeek,
Qwen or opencode.

## Ownership

- `engine.py`: retry driver (`drive_retries`), recovery containment (`_run_recovery`), account-slot release/acquire failover, cancellation checks, producer-thread to consumer bridge, `new_stream_context`, `LiveEmitTracker` / `live_emit_pair` (the partial-emit guard), `sse_error_frame`, `scrub_upstream_text`, `current_cancel_event`.
- `transport.py`: HTTP-level concerns only: content-type checks, `.text` vs body, error drains, caps, `http_error_code`.
- `openai.py`: OpenAI response shaping shared by all web2api providers.
- `pools.py`: pure functions over `Settings` deciding which providers are active and which accounts belong to one.
- `provider.py`: the `Provider` ABC and `ModelSpec`.
- `registry.py`: lazy discovery of built-in provider modules.
- `tokens.py`: `userToken` extraction and normalization.
- `obscura_bridge.py`: WAF-cleared cookies plus browser User-Agent for the hot path.
- `mcp_client.py`: small thread-safe stdio JSON-RPC client for Obscura's MCP server.
- `browser_paths.py`: discovers a real Chromium-family binary for login automation.
- `types.py`: `StreamEvent` and other shared structural types.

`tools/` is a child doc.

## Local Contracts

- `engine.py` imports no provider module and no `deepseaport.server`. Providers hand it small adapters (predicates, recover callbacks, run-once closures); it owns only the generic mechanics.
- Retry order is pow, session, token, waf, overload, one `max_retries`-bounded pass per kind.
- **No retry row may fire after a partial live emit.** `LiveEmitTracker` is wrapped onto the live callbacks once per attempt by `live_emit_pair`, and every provider's `_retryable(kind)` predicate ANDs `not tracker.emitted` -- all three tables, not just the overload row, which is where the guard used to sit. Live mode `_put`s deltas as they arrive, `drain_attempt` makes content-then-error a legal ordering, and the completion POST is NOT idempotent (the upstream already appended a turn): re-POSTing it and re-streaming from delta 0 duplicates text, or interleaves two halves of one `tool_calls` JSON object, inside a 200 stream that still ends in a clean `[DONE]`. A new row that does not go through `_retryable` reintroduces exactly that.
- `_run_recovery` contains any exception a recovery step raises, logs the SCRUBBED reason at warning, and deliberately drops `exc_info`: `traceback.print_exception` reprints the exception message at the bottom, and that message is very often a raw slice of the upstream body -- the traceback was a second unscrubbed copy. Only `HTTPException` and the cancellation sentinel propagate: those are decisions, not malfunctions. A failed recovery must never turn a retryable error into a 500.
- `http_error_code(status_code, forbidden=...)` takes the provider's 403 table as an argument. Omitting it keeps the DeepSeek default, where a 403 is the AWS WAF challenge. Qwen passes an empty table so its 403 stays a real authorization failure. Never hardcode one shared mapping.
- A bare 401 is `INVALID_TOKEN`, never `BAN`. An account is marked banned only on real ban evidence.
- `scrub_upstream_text` runs on every path that can echo an upstream body, and always before any truncation: truncating first can cut a credential mid-token. The sinks to re-check when you add one:
  - `sse_error_frame` -- every outbound SSE error frame, including the locally raised watchdog timeout and generic consumer failures.
  - `transport.log_http_error` -- every non-200, so it fires on the request that is carrying the account's bearer token.
  - `_run_recovery`, and the terminal raisers `raise_rate_limit` / `raise_overload` / `raise_upstream_invalid` / `raise_upstream` (both the `logger.warning` and the `HTTPException` detail).
  - `ban_403_for_item` -- both the primary path and its `except` fallback.
  - The `except Exception:` producer tail inside each of the three `providers/*/completion_service.py::_produce`. This one bypasses the terminal ladder entirely: the client helpers raise with a raw slice of the body in the message, so the tail is the ONLY thing standing between that body and the client. All three carry a 800-cap scrub comment saying so -- keep it when you touch the tail.
  - `server_routes` non-stream catch-all (that file's own doc covers it).
  - Outside the set, and NOT scrubbed today: `providers/deepseek/auth.py` logs a `_collapse(body, N)` login-page slice (three `logger.info`/`warning` lines), and `providers/qwen/auth.py::login_with_expiry` logs a raw `str(exc)` from its `except Exception` fallback. Browser/driver text rather than a completion body, so they are outside the contract above — but they are unscrubbed, so do not widen a claim to cover them without scrubbing them first.
- `_BUFFERED_PRODUCER_READS = 3`. `stream_mode: buffered` sizes its watchdog against the whole producer budget (setup plus one full recovery pass), because the producer emits nothing until the reply lands. Live mode passes `False` and keeps the single-read budget, since every delta resets the clock.
- `current_cancel_event()` publishes the token `drain_attempt` is draining under, thread-locally, for the length of the drain (restored, not cleared). A stream function sits inside a BLOCKING socket read, where returning control to `drain_attempt`'s per-event check is the only way a disconnect is ever noticed. It is thread-local rather than a parameter because every provider's `stream_fn` is `(headers, payload)`: a third positional would have landed in Qwen's `timeout`. DeepSeek reads it (`client._cancellable`); a second provider that needs it must read it the same way, never by adding a parameter its peers do not accept.
- `registry._BUILTIN_MODULES` is the single registration point. Adding a built-in provider is one line there, never a caller-side import.
- `_LOADED` is set after the imports, never before, under `_LOAD_LOCK`, so a reader cannot observe a half-populated registry while the warmup daemon thread is still importing. `_REGISTRY` is updated in place, never rebound; callers and tests hold the dict itself.
- `deepseek` must stay in `_BUILTIN_MODULES`: `default()` and `resolved_ids()` fall back to it by name.
- `resolve_model` guards `provider.model_spec` in a try/except and skips a provider that raises, like its two sibling call sites (`build_models_list`, `catalogue_tuples_for`) already did. It is also order-INDEPENDENT: a provider whose catalogue actually LISTS the model always wins, and a synthesised passthrough spec (opencode's `provider/model` form) is only returned when no provider in the set lists the id. Without that, a `providers` list with opencode first let its slash-passthrough swallow every model id containing a slash. `Provider.accepts_model` cannot express this rule — it delegates to `model_spec`, which the passthrough answers true.
- A provider module that fails to import is logged and skipped, never raised. One broken optional provider cannot take the server down.
- `openai.py` resolves the tool parser through a soft `sys.modules` lookup so a monkeypatched `deepseaport.server.parse_tool_calls` still wins; an explicit `parse_fn` overrides both. Keep the lookup.
- `tokens.extract_token` accepts raw text, JSON, double-encoded JSON, and the literal `{"value": null}` shape, returning the usable value or `""`. `{"value": null}` normalizes to "missing token" instead of causing a later upstream 40003.

## Work Guidance

- Put shared behavior here only once a second provider needs it. Anything one provider uses alone stays in that provider.
- Keep transport free of retry codes, ban classification and string matching. Those are provider decisions; transport moves bytes and classifies HTTP status.
- `pools.py` late-imports `..accounts` inside functions so it never enters an import cycle. `pools.active_provider_ids` does NOT consult the registry — it reports what the config says, and `config.load_settings` is where an unknown id is rejected, so the two must not both second-guess the operator.

## Verification

- The hermetic gate is owned by the root AGENTS.md; run it before any change here.
- `tests/test_engine_retry_gaps.py`, `tests/test_stream_protocol_index.py`, `tests/test_providers.py`, `tests/test_route_params_and_registry.py` and `tests/test_provider_dispatch.py` cover this folder.
- Run `python -m pytest -m live -q` after touching retry order, the stream watchdog, or anything that renders an upstream body.