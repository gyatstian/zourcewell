# providers/qwen

## Purpose

The chat.qwen.ai upstream: a JWT session cookie, Aliyun anti-bot, and
OpenAI-style SSE. No browser, no captcha, no proof of work.

## Ownership

- `completion_service.py`: the Qwen adapter over `core/engine.py`, plus chat lifecycle, re-signin, WAF recovery and ban handling.
- `client.py`: synchronous, thread-safe HTTP against the web API.
- `protocol.py`: headers, payloads, and Aliyun challenge detection.
- `sse_parser.py`: `QwenStreamParser` plus a one-liner.
- `auth.py`: direct-API signin.
- `models.py`: the catalogue as `ModelSpec` entries.
- `provider.py`: the `Provider` implementation, delegating to `completion_service`.
- `README.md`: the long-form live-verified protocol notes.

## Local Contracts

- One completion: `POST /api/v2/chats/new`, then `POST /api/v2/chat/completions?chat_id=<id>`, then a best-effort `DELETE /api/v2/chats/{id}`. The `?chat_id=` query parameter is required; omitting it changes the response.
- Auth is a direct `POST /api/v1/auths/signin` with a SHA-256-hex password. No browser, no captcha. It returns a flat user object with a JWT plus `expires_at`.
- The JWT rides `Cookie: token=<jwt>` on every hot-path request, and `Authorization: Bearer` is kept too. `auth.py` returns plain tuples: `login` gives `(token, banned, detail)`, `login_with_expiry` adds the expiry, and `unpack_login_result` normalizes a legacy tuple or string. It does not import deepseek.
- The WAF is Aliyun and answers with HTTP 200 plus a `text/html` challenge body (`WAF_CHALLENGE`, detected in `protocol.py`), never with 403. Pass `forbidden={}` to `core.transport.http_error_code` so a real 403 keeps meaning a suspension: 403, cooldown, failover. Sharing DeepSeek's mapping collapsed the two once, and a suspended account was classified WAF, re-primed, answered 502, and never cooled down.
- [PERM] `needs_waf_jar = False` and the shared bridge supplies its User-Agent only: that jar is the DeepSeek-host WAF session (`warmup()` fetches `CHAT_HOME`) and must never ride a qwen request.
- Qwen cookies are its own qwen-host harvest in `client._QWEN_WAF_CACHE` (`acw_tc` plus `x-ap`/`sca`/`cna`/`tfstk`/`isg`), merged into `Cookie` by `_merged_waf_cookies`. The startup warmup is skipped for a qwen-only active set; `provider.warmup()` still primes the qwen host.
- SSE is OpenAI-style JSON per `data:` line with a `delta.phase` split. It is not the DeepSeek `{p,v}` protocol, which is why this provider has its own parser. `client.stream_completion` bounds EVERY read on the response through `core.transport._cancellable` (right after the POST), so a disconnect while the socket is silent -- the SSE pump and the non-SSE `drain_json_text` park alike -- ends cleanly (`_ClientGone`) and the account slot comes back promptly instead of being held for the whole `DEFAULT_TIMEOUT`.
- A 401 probes `GET /api/v2/auths/` first, so a ban masquerading as a 401 becomes 403 plus failover instead of a re-signin loop. `_terminal_ban` honours that probe's BOOLEAN, not only its timestamp: a `(False, ts)` answer must not be adopted as a ban cooldown, and a confirmed ban takes its expiry from the probe. Then one direct re-signin.
- Retry: one `max_retries`-bounded pass per kind through `core.engine.drive_retries`, in order waf, invalid_token, overload -- a solved-jar refresh plus a fresh chat, a direct re-signin plus a fresh chat, and a fresh header build plus a fresh chat. There is no pow or invalid-session row: neither failure exists here. Every row goes through the local `_retryable(kind)` predicate, so NO row retries once a live delta has reached the client; the guard used to sit on the overload row alone and the rule is `core/AGENTS.md`'s.
- The credential is a JWT, and its dot-framed segments are individually too short for a run-length redactor, so an echoed error body is a place a live credential surfaces. Every terminal path scrubs through `core.engine.scrub_upstream_text`, and it scrubs before slicing, never after.
- `_prepare_chat` must delete the superseded chat id, not orphan it. It copies the old id and headers first and assigns `chat_id` only after the replacement exists, so a failed create leaves the old id for the `finally` to clean up. A recovery that created a replacement without this handoff lost the only handle to the old chat, and `auto_delete_session` could never reclaim it.
- `token_expires_at` persists the signin expiry. `0` means unknown and never blocks.
- One proactive direct re-signin when a password is stored and the stored expiry has passed, before the hot path.

## Work Guidance

- Anything still unverified is marked `# TODO(unverified)` and left out rather than invented. That rule is what keeps `protocol.py` trustworthy.
- Keep `README.md` current. It is the answer to "why is this not the DeepSeek path".

## Verification

- The hermetic gate is owned by the root AGENTS.md; run it before any change here.
- `tests/test_qwen_cancel.py` pins the bounded cancel read (hermetic, both providers, the SSE pump and the non-SSE drain).
- Run `python -m pytest -m live -q` after any wire-format change. `tests/test_live_smoke.py` asserts this provider's `delta.phase` split.