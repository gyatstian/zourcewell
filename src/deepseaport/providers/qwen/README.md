# Qwen provider (web2api)

Programmatic client for `chat.qwen.ai` (`/api/v2`). Reverse-engineered from
live probes + the shipped JS bundle (`js/main.js` v0.2.91). Login,
completions payload, and SSE shapes below are live-verified against real
chat.qwen.ai traffic; anything still unverified is called out — nothing was
invented.

## Verified URLs

All on `https://chat.qwen.ai` unless noted. `BASE` overridable via
`DEEPSEAPORT_QWEN_BASE` (host via `DEEPSEAPORT_QWEN_HOST`).

| Endpoint | Method | Auth | Notes |
|---|---|---|---|
| `/` | GET | no | chat home (no redirect observed) |
| `/auth` | GET | no | login route (SPA) |
| `/api/v2/models/` | GET | no | live-verified 200, `{success, request_id, data:{data:[Model]}}`; NOT WAF-gated |
| `/api/v2/auths/` | GET | yes | best-effort probe; unauth → `{success:false, data:{code:"Unauthorized", details:"401 Unauthorized"}}`; NOT WAF-gated |
| `/api/v1/auths/signin` | POST | no | live-verified (see "Login" below); `{email, password:<sha256-hex>}` → flat user object with JWT `token` |
| `/api/v2/chats/new` | POST | yes | `{chatId:"", models:[m], project_id, timestamp, chat_type, chat_mode:"normal"}` → `{success:true, data:{id}}`; WAF-gated |
| `/api/v2/chats/{id}` | DELETE | yes | delete chat (JS-grep); WAF-gated |
| `/api/v2/chat/completions?chat_id=<chat_id>` | POST | yes | `?chat_id=` query param REQUIRED (missing → `RequestValidationError: Field 'query -> chat_id': Field required`); needs `X-Accel-Buffering: no`; WAF-gated |
| `/api/v2/chat/completions?chat_id=..&message_id=..` | GET | yes | stream resume (`responseType:"stream"`) — **NOT implemented**, `# TODO(unverified)` |

## Login (live-verified)

Direct-API login works — no browser, no captcha needed.

- `POST /api/v1/auths/signin` with web headers (`Origin: https://chat.qwen.ai`,
  `Referer: https://chat.qwen.ai/auth`, `source: web`, `Version: 0.2.91`,
  `X-Request-Id: <uuid>`, Chrome UA) and body
  `{"email": "<email>", "password": "<sha256-hex of password>"}`.
- Success: HTTP 200 application/json, FLAT user object (NOT `{success,...}`):
  keys `id, email, name, role, profile_image_url, tier, token, token_type,
  expires_at, permissions`. `token` is a JWT string, `token_type: "Bearer"`,
  `expires_at` unix ts. The same JWT is planted as a cookie named `token`
  on `.qwen.ai` — that cookie is the web session key.
- Wrong password: HTTP 400
  `{"detail":{"code":"INVALID_CRED","details":"The email or password provided is incorrect..."}}`.
  Unknown email: HTTP 400 `{"detail":"email not found"}`.
- 200 + text/html = Aliyun WAF challenge page (see "WAF challenge detection").
- `provider.login()` / `.auth.login()` delegates to this signin flow and
  returns a plain `(token, banned, detail)` tuple, never raising.
  `.auth.login_with_expiry()` additionally returns `expires_at` (JWT `exp`
  fallback) for `AccountConfig.token_expires_at` persistence.

## Auth token

- Stored in `localStorage` under key **`token`** as a **raw JWT string**:
  `localStorage.setItem("token", n.user.token)`. Fallback cookie
  sources: `qwen_token` / `token`.
- Shape: raw JWT string, NOT a `{"value":...}` JSON wrapper (the shared
  `core.tokens.extract_token` still normalizes both fine).
- Completions REQUIRE `Cookie: token=<jwt>` PLUS the solved WAF jar.
  To be explicit: a bare `acw_tc` from curl `GET models/` passes `chats/new`
  but is NOT enough for `completions` (live 2026-09-22: `token+acw_tc`
  still `WAF_CHALLENGE` — that means the jar is incomplete, not that Qwen
  is broken). Solved jar = obscura `fetch https://chat.qwen.ai/ --dump cookies`
  (`acw_tc` 64-char + `x-ap`/`sca`/`cna`/`tfstk`/`isg`), harvested by
  `prime_qwen_bridge()`; `prime_waf()` is bare fallback only.
- Every hot-path request carries `Cookie: token=<jwt>` (account wins) +
  cached jar; `Authorization: Bearer <jwt>` is kept too (helps non-WAF
  endpoints like chats/new).
- **Design decision:** the web app deletes the `Authorization` header for
  `source:web`/`h5` (session rides cookies), but the backend is FastAPI
  with Bearer support. Our programmatic client sends both.

## Request headers

JS-grep interceptor: `Content-Type: application/json`, `Version: 0.2.91`
(`DEEPSEAPORT_QWEN_VERSION` override), `source: web` (or `desktop`),
`X-Request-Id: <uuid per request>`, `Timezone: <tz>`,
`X-Accel-Buffering: no` (completions only), `Origin: https://chat.qwen.ai`,
`Referer: https://chat.qwen.ai/`, Chrome-family `User-Agent`.

## Completion POST body (live-verified, `sendPromptOpenAI`)

`POST /api/v2/chat/completions?chat_id=<chat_id>` — the `?chat_id=` query
param is required.

```json
{ "stream": true, "version": "2.1", "incremental_output": true,
  "chatId": "<chat_id>", "parentId": "", "chat_id": "<chat_id>",
  "chat_mode": "normal", "model": "<model id>", "parent_id": null,
  "messages": [ <one user message item> ],
  "timestamp": <unix sec>,
  "stream_options": {"include_usage": true} }
```

- `version` is the completion version default **"2.1"**
  (`DEEPSEAPORT_QWEN_COMPLETION_VERSION` override), NOT the app header
  version 0.2.91.

User message item (exact, plain text message — NO `files` key):

```json
{ "id": null, "fid": "<uuid4>", "parentId": "", "childrenIds": [],
  "role": "user", "content": "<prompt>", "user_action": "chat",
  "timestamp": <unix sec>, "models": [], "model": "",
  "chat_type": "t2t",
  "feature_config": {"thinking_enabled": <bool>, "output_schema": "phase",
                     "research_mode": "normal",
                     "thinking_format": "summary",
                     "auto_search": <bool>},
  "extra": {"meta": {"subChatType": "t2t"}}, "sub_chat_type": "t2t" }
```

`feature_config` conditionals (live `/api/v2/models/` `info.meta`):

| model | `thinking_format` | `auto_search` |
|---|---|---|
| `qwen3.7-plus` | `"summary"` emitted ONLY when meta has it AND thinking is on | key present → value = request `search` flag (`true` in meta) |
| `qwen3.8-max` | same as above | same as above |
| `qwen3.8-omni-flash` | omit (meta has neither key) | omit |

## SSE event map (live-verified)

OpenAI-style JSON per `data:` line; no DeepSeek `{p,v}` protocol. Real
samples from live completions streams:

```
data: {"response.created":{"chat_id":"...","parent_id":"<user msg fid uuid>","response_id":"<uuid>","response_index":"0"}}
data: {"choices":[{"delta":{"role":"assistant","content":"Hello","phase":"answer","status":"typing"}}],"response_id":"...","usage":{"input_tokens":832,"output_tokens":1,"characters":0,"total_tokens":833,...},"timestamp":...}
data: {"choices":[{"delta":{"content":"","role":"assistant","status":"finished","phase":"answer"}}],"response_id":"..."}
```

| Top-level key | Meaning | Parser action |
|---|---|---|
| `response.created` | start marker (`chat_id`, `parent_id` = user msg fid, `response_id`) | ignore |
| `response.info` (`action:"skip_think"`) | ignore | ignore |
| `response.stopped` | turn finished (JS-grep, not seen live) | ignored here (a final choices/usage line may follow) |
| `choices:[{delta:{content,phase,role,status,extra,lang_code,function_call,function_id,name,mcp_name}}]` | content chunk | map by `delta.phase` (below); FINISH = `delta.status == "finished"` (empty content) → record the finish and still read the chunk's `usage`, then emit usage **before** finished (the consumer stops reading at `finished`); a choice after the finished delta is still mapped, and no empty content is emitted |
| `usage` on the choices line, on the finished chunk, per choice object, or a usage-only line | token counts `input_tokens`/`output_tokens`/`total_tokens` | best-effort usage event carrying the **completion** count: `output_tokens` first, then `completion_tokens`, `total_tokens` only as a last resort — `core.openai.build_usage` treats the emitted number as `completion_tokens`, and Qwen's `total_tokens` is input+output, so preferring it would charge the prompt as output (a live 832/1/833 sample reported 833 completion tokens instead of 1) |
| `error` (`{code,details}` or `{code,message}` or string) | error, e.g. `{"error":{"code":"quota_limit","details":"The service is currently experiencing high demand..."}}` — message key is **`details`** | error event (code from `code` then `type`) |
| `done:true` | finished | finished event |
| plaintext `data: [DONE]` | finished (handled, not observed live — stream simply ends after the finished chunk) | finished event |
| `sources` / `selected_model_id` / `{phase,value}` | misc | ignore |

`delta.phase` semantics: `think`/`think_summary` → **thinking** event;
`answer`/(absent)/`tool_call` → **content** event; `KeepAlive` → ignore.
A `tool_call` phase carries function_call JSON surfaced as content text so
the shared tool parser can recover it. No `[DONE]`, no `response.stopped`,
no `done:true` observed live.

## WAF challenge detection

Mutating endpoints (`/api/v2/chats/new`, `/api/v2/chat/completions`) sit
behind **Aliyun WAF anti-bot**. A bot request returns **HTTP 200 + HTML
challenge** (NOT 403) with `<meta name="aliyun_waf_aa" ...>`/`..._bb` and
references to `AliyunCaptcha.js` (`AliyunCaptcha` → `initAliyunCaptcha`).
Read endpoints (`models/`, `auths/`) are NOT challenge-gated. Edge cookie:
`acw_tc` (httpOnly).

Token-cookie gating (live-verified): `Cookie: token=<jwt>` streams while
`Authorization: Bearer` alone gets the 200 text/html challenge — every
hot-path request merges `token=<bearer>` into `Cookie` (account token wins
over any stale profile `token=`), keeping `Authorization` too. Set
`DEEPSEAPORT_QWEN_NO_BEARER=1` to omit `Authorization` and match strict-web
(`source=web` deletes it, session rides cookies) when WAF persists.

Bare `acw_tc` from curl `GET models/` passes `chats/new` but NOT
`completions` (live 2026-09-22: `token+acw_tc` still `WAF_CHALLENGE`).
Solved jar via obscura `fetch https://chat.qwen.ai/ --dump cookies`
(`acw_tc` 64-char + `x-ap`/`sca`/`cna`/`tfstk`/`isg`) is required;
`prime_qwen_bridge()` harvests it, `prime_waf()` is bare fallback.
`False` from `prime_waf()` with a fake `<JWT>` is envelope-miss only —
the harvest still lands. Completions send `X-Accel-Buffering: no` like the web.

Client detection: content-type `text/html` OR body lines containing
`aliyun_waf`/`AliyunCaptcha`/`nc-container` → emits error event code
**`WAF_CHALLENGE`** (`protocol.is_waf_challenge` for the content-type half,
`protocol.is_waf_body` for the drained-body half). Engine recovery
mirrors DeepSeek's WAF retry minus PoW: release the account slot, qwen-host
solved fetch + `prime_waf`, recreate the chat, retry
(`completion_service._refresh_waf_without_account_lock`). The DeepSeek-host
`bridge.warmup()` never runs for qwen — not at startup (`needs_waf_jar`
is `False`) and not on recovery — so there is no obscura fetch against
chat.deepseek.com and no DeepSeek cookies planted (WinError5 noise on
Windows). The shared bridge contributes its User-Agent only; qwen cookies
stay in the qwen-host cache.

## Models (verified live, unauth catalogue)

| id | thinking | search | notes |
|---|---|---|---|
| `qwen3.7-plus` | true | true | default (`DEFAULT_MODEL`) |
| `qwen3.8-max` | true | true | |
| `qwen3.8-omni-flash` | true | false | no search capability, `is_visitor_active:false` |

## Error mapping

`classify_error` (string → kind), then retry or terminal:

| String family | Kind | Retry | Terminal HTTP |
|---|---|---|---|
| `WAF` / `403` / `challenge` | WAF | qwen-host solved fetch + fresh chat (DeepSeek `warmup` skipped) | 502 |
| `401` / `unauthorized` / `invalid` / `authorization` | INVALID_TOKEN | ban-probe, then one direct re-signin (stored password) + fresh chat | 401 |
| `overload` / `server busy` / `try again` / `timeout` / `502` / `bad gateway` | OVERLOAD | retry (skipped after partial live emit) | 503 |
| `rate_limit` / `quota` (`quota_limit`) / `429` / `too many` | RATE_LIMIT | no | 429 |
| `http_422` / `http_400` / `unprocessable` | UPSTREAM_INVALID | no | 502 |
| generic ban words (`P.is_ban_error_text`) / `restricted` | BAN | no (failover) | 403 + mark banned |
| anything else | UPSTREAM | no | 502 |

**The `403` row is the fallback, not the answer — read the ban row first.**
A Qwen 403 reaches `classify_error` as a plain `HTTP_403`, not
`HTTP_403_WAF` (`core.transport.http_error_code` is called with an empty
`forbidden` table for this provider; see "The 403 meaning is per-provider" in
`PROVIDERS.md` for why). A bare `403` with no ban wording still falls into the
substring WAF branch, but the ban check runs first, so a body that says the
account is suspended routes to **BAN** — 403 + cooldown + failover — not to a
WAF re-prime and a 502:

```
HTTP_403: your account has been suspended   -> RetryKind.BAN
HTTP_403: blocked                            -> RetryKind.WAF   (fallback)
```

This matters because the two providers' WAFs differ: DeepSeek's 403 *is* the
AWS-WAF challenge and must be `HTTP_403_WAF`, while Qwen's WAF answers 200 +
`text/html` and leaves a real 403 to mean what it says.

Ban envelope shape is **UNVERIFIED**: no suspended-account payload has been
captured, so the verdict is best-effort — explicit mute/ban flags plus generic
wording across message/details/code fields (`detect_ban`), with a timestamp
only when a numeric `mute_until`-style key parses (`extract_ban_timestamp`).
`check_ban` claims banned only on such a signal; plain `Unauthorized`/`401`
(invalid token) stays not-banned. Ban timestamps ride `StreamEvent.ban_until`
so 403s cool down precisely instead of the default 120s.
Stream error envelope message key is `details`
(`{"error":{"code":"quota_limit","details":"..."}}`).

## Token lifecycle

- Signin `expires_at` (unix ts, JWT `exp` fallback) persists per account as
  `token_expires_at` in config.json (`0` = unknown: pasted tokens, old
  configs, DeepSeek accounts — never blocks).
- The engine pre-checks expiry before chat-create and re-signs in proactively
  when a password is stored (one cheap direct POST, no browser/captcha,
  max once per request).
- On 401 it probes `GET /api/v2/auths/` first (a ban masquerading as 401
  becomes 403 + cooldown for failover), then re-signs in once with the
  stored password and retries with the fresh token. No password (or failed
  re-signin) → terminal 401 + cross-account failover.

## Ops traps (live 2026-09-22)

- `WinError 5` on `obscura-profile` = transient LevelDB lock from two
  servers racing one `--storage-dir`, not ACL (ACL is FullControl; a manual
  `fetch chat.qwen.ai` succeeds). Kill the stale server before retrying.
- `prime_waf()` returning `False` with a fake `<JWT>` is envelope-miss
  only — the `acw_tc` harvest still lands (`prime_waf` calls
  `_harvest_waf_cookies` *before* parsing the envelope, precisely so it does).
  To test the prime without a JWT, call it with `base_headers()` emptied.
- `prime_qwen_bridge()` is the real fix; `prime_waf()` is bare fallback.

## UNVERIFIED / TODO

- Real Qwen ban envelope (wording + `mute_until`); the current verdict is
  best-effort (flags + generic wording) until a real suspended-account
  payload is captured and the timestamp keys are confirmed.
- Stream resume endpoint (`GET /api/v2/chat/completions?...`) — not implemented.
- Guest/anonymous flow (`enable_anonymous:true`, `/c/guest`,
  `permissions.chat.temporary:true`) — minting a guest token UNVERIFIED.
