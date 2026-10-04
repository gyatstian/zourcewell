# providers/deepseek

## Purpose

The chat.deepseek.com upstream: AWS WAF, a wasm proof of work, a real-browser
login, and a `{p,v}` SSE continuation protocol.

## Ownership

- `completion_service.py`: the DeepSeek adapter over `core/engine.py`, plus everything DeepSeek-specific: session create/delete, PoW, SSE, ban probes, Obscura token refresh. The canonical monkeypatch surface for tests.
- `client.py`: synchronous, thread-safe HTTP against the web API.
- `protocol.py`: headers, payloads, error mapping, ban helpers.
- `sse_parser.py`: the stateful `StreamParser` plus a stateless one-line helper.
- `pow.py`: the `DeepSeekHashV1` solver (wasmtime plus the shipped wasm).
- `auth.py`: browser login through the Obscura MCP driver.
- `real_browser.py`: the same login through CDP in a real Helium/Chrome/Edge/Chromium.
- `login_policy.py`: page semantics shared by both login drivers.
- `models.py`: the catalogue as `ModelSpec` entries.
- `provider.py`: the `Provider` implementation, delegating to `completion_service`.

## Local Contracts

- One completion: create session, fetch the PoW challenge, solve it, `POST /api/v0/chat/completion`, then `DELETE /api/v0/chat_session/delete` in a `finally`.
- `client._cancellable(resp)` bounds the SSE read before `T.pump_sse`, because curl_cffi offers NO per-read timeout: `Response.iter_lines(chunk_size, decode_unicode, delimiter)` has no `timeout` parameter and `iter_content` blocks on an untimed `queue.get`. The response's queue is swapped for `_BoundedQueue`, which re-enters `get` every `_READ_SLICE_SECONDS` and raises `_ClientGone` once `engine.current_cancel_event()` is set. This changes the GRANULARITY of cancellation only -- curl still owns the whole `DEFAULT_TIMEOUT` stall budget -- and it is the only place a disconnect is observable while the socket is silent, since `drain_attempt`'s per-event check cannot run until an event is yielded out of the generator.
- `_abandon(resp)` is what runs on that path: it sets `quit_now` and replaces `close` with a no-op. `Response.close()` joins curl's worker thread, and curl only re-checks `quit_now` from inside its WRITE callback, so a silent upstream never calls it and the join blocks for the rest of `DEFAULT_TIMEOUT`. A cancelled reply is discarded, so the transfer is left to end on the next byte or on curl's own stall timer (`Curl.__del__` reclaims the handle). Verified against the installed curl_cffi 0.16.3.
- The session id arrives as `data.biz_data.id` or `data.biz_data.chat_session.id`. Both shapes are real; parse both.
- The PoW answer travels as `X-DS-PoW-Response`, a base64 of a JSON object carrying `algorithm`, `challenge`, `salt`, `answer`, `signature` and `target_path`.
- SSE deltas are `{"p": "response/content", "o": "APPEND", "v": "..."}`, and thinking arrives under `p: response/thinking_content`.
- Continuation chunks omit `p`. A bare `{"v": "api"}` continues the previous event rather than starting one. A stateless parser once shipped that treated this as a fresh event and truncated a reply to `web` instead of `web2api-ok`, and no mock reproduces it, because the mock was the thing that broke.
- A 403 here is the AWS WAF challenge, never a plain authorization failure. Pass no `forbidden` table to `core.transport.http_error_code` so it keeps resolving to `HTTP_403_WAF` and routes to WAF recovery.
- Retry: one `max_retries`-bounded pass per kind through `core.engine.drive_retries`, in order pow, invalid_session, invalid_token, waf, overload -- a fresh PoW, a recreated session, an Obscura token refresh plus a recreated session, a lock-free Obscura WAF refresh plus a fresh PoW, and a fresh PoW on transient overload. Every row goes through the local `_retryable(kind)` predicate, so NO row retries once a live delta has reached the client; the guard used to sit on the overload row alone and the rule is `core/AGENTS.md`'s.
- One in-flight stream per account is an upstream limitation, enforced with a per-account lock, rotation and a 120 s cooldown.
- Direct username/password POSTs are fingerprint-gated (`RISK_DEVICE_DETECTED`). The login runs through a browser. Obscura's no-render engine cannot initialize the Shumei `fp.min.js` SDK, so the sign-in posts `device_id: null` and DeepSeek rejects it; `real_browser.py` drives a real Chromium-family browser over CDP, where the SDK initializes. Obscura stays the WAF and hot-path engine either way.
- `login_policy.py` is the single source of page semantics for both drivers: selectors, phrase lists, visibility and snapshot JS, poll and timeout constants, text collapse, classify and verdict helpers. Transport stays driver-specific (MCP tools versus a CDP websocket).
- `_invalid_token_401_detail` and `_refresh_token_via_obscura` scrub upstream text before slicing.

## Work Guidance

- New page selectors or login phrases go in `login_policy.py`, never forked into a driver.
- Keep `completion_service.py` an adapter. Generic mechanics belong in `core/engine.py`; if the same table would serve a second provider, move it up.
- Hot path is direct HTTP. Do not introduce DOM driving per token.

## Verification

- The hermetic gate is owned by the root AGENTS.md; run it before any change here.
- `python -m deepseaport waf` verifies the Obscura binary and the WAF cookie on a real machine.
- Run `python -m pytest -m live -q` after any wire-format change. `tests/test_live_smoke.py` asserts this provider's continuation chunks, the WAF jar, and the streamed `tool_calls` index, which no stub can assert.