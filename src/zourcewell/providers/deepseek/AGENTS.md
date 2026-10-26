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
- `client._cancellable(resp)` bounds EVERY read on the response -- the SSE pump and the non-SSE 200 `drain_json_text` park -- and runs right after the POST. The machinery now lives in `core.transport` (curl_cffi 0.16.3 rationale there) and this module re-exports the historical names (`_cancellable` / `_BoundedQueue` / `_abandon` / `_ClientGone` / `_READ_SLICE_SECONDS`) because tests read them off `client`. DeepSeek-specific consequence: on that path `_abandon` sets `quit_now` and no-ops `close`, so a cancelled silent read is left to curl's own stall timer (`Curl.__del__` reclaims the handle) rather than joining curl's worker -- the account slot comes back now.
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
- Ban helpers never raise. `_as_float` scales a millisecond `mute_until` (>1e10) to seconds and drops non-finite/out-of-range values to unknown expiry; `format_ban_day_month` / `format_ban_datetime` return "" (and `ban_message` says "(expiry unknown)") when the timestamp is unusable, while `format_ban_label` degrades to a plain "(BANNED)". `parse_ban_until` requires a day ("September 2026" → None), accepts the comma day-first Polish form, and adds +1 year only when the page gave no explicit year.
- [PERM] A login-page ban takes two consecutive ban frames; one sighting is a detail-only hint (`_BAN_HINT_SUFFIX`), never a `banned` verdict. Captcha, credential and no-token exits are banned=False, so `browser_login` still falls back to the Obscura driver; only a produced token or a confirmed ban ends the login attempt.
- `classify_error` checks rate-limit before overload: "429 ... try again" is the terminal 429, not retryable overload.
- The token-refresh singleflight persists a completed login and wakes followers before any cancellation raise; a follower returns a token only when it actually changed from the one it held.

## Work Guidance

- New page selectors or login phrases go in `login_policy.py`, never forked into a driver.
- Keep `completion_service.py` an adapter. Generic mechanics belong in `core/engine.py`; if the same table would serve a second provider, move it up.
- Hot path is direct HTTP. Do not introduce DOM driving per token.

## Verification

- The hermetic gate is owned by the root AGENTS.md; run it before any change here.
- `python -m zourcewell waf` verifies the Obscura binary and the WAF cookie on a real machine.
- Run `python -m pytest -m live -q` after any wire-format change. `tests/test_live_smoke.py` asserts this provider's continuation chunks, the WAF jar, and the streamed `tool_calls` index, which no stub can assert.