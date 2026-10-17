# deepseaport architecture & reverse-engineering notes

How the bridge works and everything non-obvious discovered while building it.
Read this before touching `providers/deepseek/*`, `providers/qwen/*`,
`core/engine.py`, or auth.

## Big picture

```
harness -> FastAPI (/v1/*, OpenAI-compatible) -> curl_cffi (Chrome TLS) -> chat.deepseek.com
                                                        ^
Obscura (h4ckf0r0day/obscura, headless browser) ________|
  - owns persistent profile (~/.deepseaport/obscura-profile)
  - solves AWS WAF JS challenge, holds `aws-waf-token` cookie
  - used only for WAF/hot-path; not for login

real Helium/Chrome/Edge/Chromium (CDP) _______________|
  - optional but preferred for auto-token/login
  - owns a temporary profile
  - runs Shumei `fp.min.js`, supplies `device_id`, captures userToken
  - controlled by `real_browser.py`; selector-driven, not ref-driven
```

Hot path is **direct HTTP, never DOM driving** (every provider).
DeepSeek: Obscura is re-invoked only to refresh the WAF token (startup +
on HTTP 403). Qwen: mutating endpoints sit behind Aliyun WAF anti-bot,
which returns HTTP 200 + HTML challenge (not 403); recovery is a
qwen-host solved-jar fetch + fresh chat (no PoW). Binary resolution
(`obscura_bridge.py:_discover`): `obscura_bin` config / `OBSCURA_BIN` env
first, then PATH, then a repo-local drop-in — `obscura.exe` (or `obscura`
on unix) next to `config.json` / repo root, or under `bin/`, `tools/`,
`vendor/`, `obscura-*/`. v0.2.1 no-render build is enough — no screenshots
needed. Local binaries are gitignored, never committed.

## Request lifecycle (per chat completion)

DeepSeek path:

1. Acquire account (one in-flight stream per account — DeepSeek limitation,
   enforced by per-account `threading.Lock` + rotation + 120 s cooldown).
2. Token: account `token`, else a fast 401 that the failover helper can
   route to another token-backed account (see Auth).
3. `POST /api/v0/chat_session/create` `{"agent":"chat"}` → session id.
   Response shapes vary; parse `data.biz_data.id` or `data.biz_data.chat_session.id`.
4. `POST /api/v0/chat/create_pow_challenge` `{"target_path":"/api/v0/chat/completion"}`.
5. Solve PoW (`DeepSeekHashV1`, wasm, prefix `"{salt}_{expire_at}_"`), send as
   `X-DS-PoW-Response: base64(json({algorithm,challenge,salt,answer,signature,target_path}))`.
6. `POST /api/v0/chat/completion` (SSE `text/event-stream`) with payload:
   `{chat_session_id, parent_message_id: null, model_type: "default", prompt,
   ref_file_ids: [], thinking_enabled, search_enabled, source: "web",
   action: null, preempt: false}` (+ Bearer + WAF cookies + PoW header).
7. Parse SSE → OpenAI chunks/JSON. `DELETE /api/v0/chat_session/delete` in `finally`.

Retry policy: fresh PoW once on PoW errors, recreate session once on
`INVALID_SESSION_ID`, Obscura warmup + retry once on 403/WAF, one
`max_retries`-bounded pass per kind in the order pow → session → token →
waf → overload (`core.engine.drive_retries`).

Five rules keep that table from lying to the client:

- **A 403 means different things per provider, so the mapping is injected, not
  hardcoded.** `core.transport.http_error_code(status, forbidden=...)`
  (`core/transport.py::http_error_code`) takes the provider's `{403: code}` table as an
  argument; omitting it keeps the DeepSeek default. DeepSeek passes nothing and
  gets `HTTP_403_WAF`, because on DeepSeek a 403 *is* the AWS WAF challenge —
  it is never a plain authorization failure, so it must route to WAF recovery
  (Obscura warmup + one retry). Qwen passes an **empty** table, so its 403
  stays a plain `HTTP_403`: Qwen's WAF is Aliyun, and Aliyun does not answer
  403 at all — it returns **HTTP 200 with a `text/html` challenge body**
  (`WAF_CHALLENGE`, detected in `providers/qwen/protocol.py`). That leaves a
  real Qwen 403 to mean what it says: a suspension, which `classify_error`
  routes to a 403 + cooldown + failover. Sharing one mapping collapsed the two:
  a suspended Qwen account was classified WAF, got a WAF re-prime and a 502,
  and was never cooled down.
  `PROVIDERS.md` "The 403 meaning is per-provider" is the long form and is the
  authority `core/transport.py::http_error_code` cites.
- **A bare 401 is an AUTH failure, not a ban.** `classify_error` maps
  `401`/`unauthorized` to `INVALID_TOKEN` (retry + probe, terminal 401),
  never to `BAN` (`providers/deepseek/completion_service.py::classify_error`); Qwen
  already worked that way, so both **web** upstreams now agree. An
   account is marked banned only on real ban evidence — ban/mute wording or an
   explicit mute flag, confirmed by the `users/current` / `GET /api/v2/auths/` /
   `GET /api/v1/auths/` probe (`_probe_ban`). On the login page a single frame
   with ban wording is only a hint appended to the detail: both browser drivers
   require two consecutive ban frames before reporting `banned=True`. An
   invalid token can never prove a ban; only the upstream (or its page) can.
   Before this, one transient 401 wrote `banned: true` into config.json
   permanently and answered 403 with ban wording.
- **A failed recovery step never converts a retryable error into a 500.**
  Every `recover()` runs through `core.engine._run_recovery`, which contains
  the exception, logs the **scrubbed** reason at warning, stops escalating *that*
  kind, and hands the unchanged error to the next handler — the terminal ladder
  then raises the original status (503 overload / 502 WAF / 401 token). Only
  `HTTPException` and the cancellation sentinel propagate: they are
  decisions, not malfunctions. The request still fails, with the upstream
  reason. `_run_recovery` deliberately passes **no `exc_info`**: a recovery
  failure is very often the provider's own HTTP raise, whose message embeds a
  raw slice of the upstream body, and `traceback.print_exception` reprints that
  same message at the bottom — the traceback was a second unscrubbed copy of the
  body, on a path whose whole purpose is de-tokenizing it.
- **No row of that table may fire once a live delta has reached the client.**
  `core.engine.LiveEmitTracker`, wrapped onto the live callbacks by
  `live_emit_pair`, flags exactly that, and every provider's `_retryable(kind)`
  predicate ANDs `not tracker.emitted`. Live mode `_put`s deltas as they
  arrive and `drain_attempt` makes content-then-error a legal ordering, while
  the completion POST is **not idempotent** — the upstream already appended a
  turn. A `pow` / `invalid_session` / `invalid_token` / `waf` / `overload`
  recovery fired there re-POSTs the same payload on the same session and
  re-streams from delta 0 into a stream the client already read: duplicated
  text, or two interleaved halves of one `tool_calls` JSON object, then a clean
  `[DONE]` and no error to stop a harness executing the corrupt call. The guard
  started on the `overload` row alone, on the assumption that the other kinds
  fail before any content. All three tables gate **every** row now (each
  `drive_retries` call in `providers/*/completion_service.py`).
- **`stream_mode: buffered` sizes its watchdog against the whole producer
  budget, not one read.** In live mode `stream_timeout` is one socket read
  (correct: every delta resets the clock). In buffered mode the producer
  emits nothing until the whole reply lands, so the timer covers session
  create + PoW solve + the reply + recovery passes (a browser login refresh
  alone is ~180 s). `new_stream_context(..., buffered=True)` multiplies the
  consumer budget by `_BUFFERED_PRODUCER_READS` (3: the attempt plus
  setup and one full recovery pass of headroom); live passes `False` and
  keeps the single-read budget once a payload has arrived, since every delta
  resets the clock. The FIRST `get` is widened in both modes:
  `drain_stream_queue` waits `consumer_timeout × _BUFFERED_PRODUCER_READS`
  until the first event (no mode has a payload yet, and the producer runs
  the same session + PoW + recovery work either way), so buffered waits up
  to 9 reads for the first event and live up to 3; after that the 3-read and
  1-read budgets above apply. Sizing buffered off one read killed
  healthy-but-slow upstreams with a client-visible timeout and discarded a
  valid answer.

Qwen path (no PoW; `providers/qwen/completion_service.py` over
`core/engine.py`): acquire account → pre-check `token_expires_at` with one
proactive direct re-signin when a password is stored → `POST
/api/v2/chats/new` → `POST /api/v2/chat/completions?chat_id=<id>` (`?chat_id=`
required; `Cookie: token=<jwt>` + solved WAF jar; `Authorization: Bearer`
kept too) → best-effort `DELETE /api/v2/chats/{id}`. Qwen SSE is
OpenAI-style JSON per `data:` line (no `{p,v}` protocol); 401s probe
`GET /api/v2/auths/` first (ban masquerading as 401 → 403 + failover), then
one direct re-signin. Details in `src/deepseaport/providers/qwen/README.md`.

Two Qwen-specific rules follow from its credential being a **JWT** rather than
DeepSeek's opaque 64-char token, and from its chat id being a value the caller
must keep:

- **Upstream bodies are scrubbed before they reach a client or a log.** Qwen's
  session credential is a JWT riding `Cookie: token=` on every hot-path
  request, and its dot-framed segments are individually too short for a
  run-length redactor, so an echoed error body is a place a live credential can
  surface. Qwen scrubs at every point that could echo one: the re-signin
  failure log, the terminal 401 detail, the overload-retry log (all three
  *before* an 80-char slice, because truncating first can cut a JWT mid-token),
  and the `except Exception:` producer tail inside `_produce` — that tail is the
  one path that bypasses the terminal ladder, so the 800-cap scrub there is what
  keeps a client helper's raw body slice out of a live stream. Same helper and
  same reason as DeepSeek's `_invalid_token_401_detail` and
  `_refresh_token_via_obscura` login note. `core/AGENTS.md` holds the full
  cross-provider list of sinks; check it before adding one.
- **The superseded chat id is deleted, not orphaned.** `_prepare_chat`
  rebinds `chat_id` in place, so a WAF/401 recovery that creates a replacement
  without deleting the old one loses the only handle to it and
  `auto_delete_session` can never reclaim it
(`providers/qwen/completion_service.py::_prepare_chat`). The handoff copies the old
  id and headers first, assigns `chat_id` only after the replacement is created
  (so a failed create leaves the old id for the `finally` to clean up rather
  than double-deleting), and deletes the old chat once the new one exists.

## SSE stream shape — DeepSeek (verified against live traffic)

- `event: ready` + `data: {request_message_id, response_message_id, model_type}` opens.
- Deltas: `data: {"p":"response/content","o":"APPEND","v":"..."}` and
  `{"p":"response/thinking_content",...}`.
- **Continuation chunks omit `p`**: bare `data: {"v":"api"}` continues the
  last path. `StreamParser` tracks `last_path`; a stateless parser silently
  truncates replies (this bug actually shipped once — reply came back "web"
  instead of "web2api-ok").
- `{"p":"response/accumulated_token_usage","o":"SET","v":47}` is the **real**
  token usage — prefer it over the `len//4` estimate.
- End: `{"p":"response/status","v":"FINISHED"}`, then `event: finish`,
  `event: title`, `event: close`. Stream close without FINISHED also means done.
- `{"v":"<n>"}` heartbeats between chunks are content when `last_path` is
  content (e.g. "web","2","api","-","ok" spells "web2api-ok").

Qwen SSE differs: OpenAI-style JSON per `data:` line, `delta.phase`
(`think`/`think_summary` → thinking, `answer`/absent/`tool_call` →
content), finish on `delta.status == "finished"` / `done:true` /
`data: [DONE]`, usage from `input_tokens`/`output_tokens`/`total_tokens`.
See `src/deepseaport/providers/qwen/README.md`.

### Outbound SSE error frames (the bridge's own, not an upstream's)

A failure raised mid-stream is delivered **in band**, as one OpenAI-shaped
`data: {"error": {...}}` frame, and every such frame is built in exactly one
place: `core.engine.sse_error_frame` (`core/engine.py::sse_error_frame`). Three call
sites in `drain_stream_queue` (watchdog timeout, propagated `HTTPException`,
any other queued failure) hand-rolled the dict and drifted — some carried no
`type` at all. `message` is coerced with `str()` because `HTTPException.detail`
is untyped in FastAPI: a dict detail serialized as a nested JSON object and
failed strict SDK deserialization of `error.message` (OpenAI types it as `str`),
inside the client that was waiting on the stream.

`type` is a scheme, not a label, chosen so LiteLLM / SDK retry classifiers
bucket these the way they bucket api.openai.com:

- 4xx -> `invalid_request_error`: the caller or the bridge rejected the request
  (auth, unknown model, rate limit) — not worth retrying as sent.
- 5xx, and frames raised locally with no HTTP status at all (the watchdog
  timeout, a generic consumer failure) -> `upstream_error`: the upstream broke
  or stalled, which is a "retry later" bucket. OpenAI's own 5xx slug is
  `server_error`; `upstream_error` is the honest name because every 5xx this
  bridge raises originates upstream.

`code` is the numeric HTTP status when one is known, and `null` when the failure
never had one (OpenAI types it as optional).

## Auth (hard-won; DeepSeek unless noted)

- Token lives in page localStorage as JSON: `userToken = {"value":"<64-char>","__version":"0"}`.
- Direct `POST /api/v0/users/login` (web headers) → `biz_code 11 RISK_DEVICE_DETECTED`.
  Direct login is fingerprinted (fp-1.min.js, fengkongcloud deviceprofile, `did`).
- Old mobile path (`DeepSeek/1.0.13 Android/35`, `x-client-version 1.3.0-auto-resume`)
  → `CLIENT_VERSION_TOO_LOW`. Do not chase client versions.
- Preferred working path: a real Chromium-family browser via CDP
  (`real_browser.py`) when `browser_bin` is `"auto"` or a path. It opens the
  sign-in page, dismisses the cookie banner, fills email/password, clicks the
  login button, waits for Shumei to initialize, and polls
  `localStorage.userToken`. This is what runs on user machines with Helium.
- Obscura MCP (`mcp_client.py`, stdlib JSON-RPC over stdio) remains a fallback
  (`auth.py`): `browser_navigate .../sign_in` (`networkidle0`) → dismiss the
  cookie banner → fill stable selectors → wait for React state → click the
  real "Log in" `role=button` → poll `localStorage.userToken`. It cannot
  initialize Shumei (`device_id: null`), so it is only a fallback.
- Do not use ``browser_fill_form`` with a `submit_selector`: it clicks before
  React commits, so the form can post empty credentials or show a generic
  "Login failed" without a network request. Two-stage fill + click is reliable.
- Refs (`e1`, `e2`, `e8`, ...) are scoped to one MCP connection and renumber
  whenever the SPA changes, so `auth.py` does not use them. Hardcoded refs were
  the cause of clicking cookie/social UI and triggering apparently random
  challenges.
- Captcha detection uses only *visible* text plus an explicit visibility check
  on `#cf-overlay`. The sign-in SPA always ships that overlay with
  `display:none` and the literal words "One more step before you proceed...",
  so scanning raw `document.body.innerText` classified every successful login
  as a captcha.
- Missing-token accounts are not browser-logged-in on the first API request;
  they surface a fast 401 and `_complete_with_failover_sync` can move to the
  next account that already has a token. Expired tokens still get one Obscura
  refresh attempt, with login backoff after captcha/credential failure. A
  real-browser captcha or no-token exit is not a verdict: `browser_login` falls
  back to the Obscura driver on it — only a produced token or a confirmed ban
  ends the login attempt.
- Old/stale tokens fail with `40003 Authorization Failed` at session create.
- `POST /v1/accounts/unblock` (and `deepseaport accounts unblock [identifier]`)
  clears the in-memory cooldown and the persisted ban marker.

## Account routes: what `{identifier}` may be

`{identifier}` is an **account identifier, never a token**. The accepted forms
are exactly what `AccountConfig.identifier` produces and what
`GET /v1/accounts` returns: the account's `email`, its `mobile`, or the
`token[:10] + "..."` short form (e.g. `abcdef0123...`).

Anything that looks like a bearer token — 32+ characters over the bearer/JWT
alphabet `[A-Za-z0-9_.=-]`
(`server_routes.py:73`, `_TOKEN_SHAPED_RE`) — is refused with **400** by
`_reject_token_shaped_identifier` (`server_routes.py:670-686`) on every route
that takes an identifier: `DELETE /v1/accounts/{identifier}`,
`PUT /v1/accounts/{identifier}/token`, `POST /v1/accounts/unblock` (body) and
`POST /v1/accounts/select` (body). The check runs *ahead of* `_check_auth` on
the body routes, so a credential never reaches a code path that could echo,
log or persist it, and the 400 message never quotes the value.

The reason is that a URL is the wrong place for a credential. `accounts._matches`
deliberately accepts a full token so the CLI can delete and select accounts
locally, which means the old 404 detail echoed a live 64-char bearer token
straight back to the caller — and URLs end up in uvicorn access logs, reverse
and forward proxy logs, and `Referer` headers on any redirect. This is also
why the access log is off on every serve path (below).

## Request logging: the access log is off on all three serve paths

| Path | Where | `access_log` | LAN warning |
|---|---|---|---|
| Plain serve (`cli._run_server` → `server_runner._run_server`) | `server_runner.py:427` | `False` | `server_runner.py:436` |
| Server-with-chat / multiple servers (`_start_background_server`, called from `chat_ui.py:197` and `:348`) | `server_runner.py:329` (parameter default) | `False` | `server_runner.py:340` |
| Dry run | `dry_run.py:261` | `False` | `dry_run.py:253` |

Both columns are the same rule applied twice: nothing on a serve path may print
a request target (an identifier can be a full token), and a non-loopback bind
must say so on every path. `lan_exposure_notice`
(`server_runner.py:48-64`) names the bind, what becomes reachable (the whole
API, `/v1/accounts` CRUD, unmetered chat) and how the gate reads — the N
`settings.keys` entries are the only gate and travel in a plain header on an
unencrypted LAN, or `AUTH DISABLED` outright when `keys` is empty. Dry run needs
it as much as serve: it echoes the entire raw body and the formatted prompt
back to the caller.

One setting covers all three because they are the same risk: an account
identifier can be a full token (`accounts._matches` allows a full-token match
so `DELETE /v1/accounts/<token>` can remove a token-only account) and
uvicorn's access line prints the raw request target — so that one request
would write a live bearer token, and the whole query string, into the log.
Chosen over dropping full-token matching from the HTTP layer, because that
would break a documented route and would not actually close the hole: the CLI
keeps the match, and the access log is a plain file on the same machine.

Untouched: uvicorn's **error** log (startup banner, unhandled exceptions,
shutdown) is a different logger and is still configured by `log_level`. Only
the `uvicorn.access` per-request lines go away. Paths in *error* logs are
bounded and redacted separately — `_redact_path` / `_log_target`
(`server_routes.py:110-137`) replace every token-shaped segment with
`<redacted>` and cap the logged path at 200 chars, preferring the route
template once the router has matched.

## PoW details (DeepSeek only; Qwen has no PoW)

- Wasm URL host is **`fe-static.deepseek.com`**, not `chat.deepseek.com`:
  `https://fe-static.deepseek.com/chat/static/sha3_wasm_bg.7b9ca65ddd.wasm`.
  Downloading from the wrong host returns the SPA `index.html` (wasmtime then
  fails with `expected '('`) — validate size/magic, not just HTTP 200.
- `PoW.ensure_wasm()` downloads on demand into `data/`; solving takes ~0.5 s.
- **`solve()` is covered by real wasm execution, not a stub** (`pow.py:199-266`,
  98% line coverage via `tests/test_pow_solve_wasm.py`). It `ctypes.memmove`s
  into the module's linear memory, so the only thing between a malformed
  download and a segfault is the `_check()` bounds guard — which is why the
  suite drives the real shipped wasm with WAT-crafted modules whose
  `add_to_stack` returns out-of-range pointers (`65536`, `-1`, `2147483647`),
  and why it also pins the two things that make that guard sound: `retptr`'s
  16-byte reservation provably covers the `status` (4 B at `+0`) and `value`
  (8 B at `+8`) reads, and memory growth *inside* `wasm_solve` (1 → 9 pages,
  covered) does not stale the re-read `data_ptr`. The shared-`Linker` /
  per-call-`Store` split is asserted under 8 concurrent solves: one `Linker`,
  eight distinct `Store`s, one `Engine`, no raise.
- The `-16` / `+16` `add_to_stack` unwind is **not** balanced on every path
  (a raise from `_check()` or `encode()` at `pow.py:255-257` escapes before
  the `try` opens at `:258`). That is deliberate, not an oversight: the stack
  pointer lives in the per-call `Store`, which is dropped on unwind, so the
  imbalance is unobservable. Balancing it would mean hoisting the `try` above
  the reservation.

## Tool calling (agentic harness)

Web model has no native function calling. `core.tools.render_prompt()`
flattens OpenAI messages (`<User>`/`<Assistant>…<endofsentence>` markers,
`[I called tools: …]` records, `Tool <name> returned: …` results as user
turns); `tool_system_prompt()` injects schemas and now prefers the raw DSML
`tool_calls` / `invoke` / `parameter` block format, because raw parameter
values do not need JSON escaping. `parse_tool_calls()` accepts DSML (ASCII
and fullwidth), `{"tool_calls":[…]}` (balanced-brace scan),
`<tool_call>`/`<function_call>`/`<invoke>` JSON tags, fenced JSON, bare
name+arguments objects, and a last-resort recovery for unescaped
OpenAI-style `arguments`. Unknown names are dropped, args are normalized to
JSON strings, ids are `call_001…`. Per-argument size is capped by
`Settings.tool_args_max_chars` / `DEEPSEAPORT_MAX_ARGS_CHARS` (default
200000); `DEEPSEAPORT_MAX_BARE_OBJECTS` (default 40,
`core/tools/common.py:43`) bounds how many JSON objects the parser scans for a
bare name+arguments call, and `MAX_SCAN_OBJECTS` / `MAX_TOTAL_CALLS` reuse the
same number so one env knob bounds every scan. Opt-in
`Settings.forgiving_toolcalls` / `DEEPSEAPORT_FORGIVING_TOOLCALLS` (default off)
enables the forgiving bucket
(markup repairs: pipe/space/HTML-escaped DSML markers, smart attribute
delimiters with verbatim values, single-tool missing-name/param inference,
JSON invoke bodies); replies that look like a truncated tool attempt surface
`finish_reason: "length"` instead of masquerading as a normal stop.
Stream and non-stream both end with `finish_reason: "tool_calls"`.
Multi-step loops verified live (weather → tool result → final answer).

Two boundary rules keep a sloppy client from smuggling a call or a turn through:

- **An unrecognised `role` is labelled, not merged.** `system`/`user`/
  `assistant`/`tool` keep their markers; `developer` (a first-class OpenAI role
  on the o-series and newer, where it carries system precedence) renders as
  `<System>`; anything else gets `<Role role="X">` with the value sanitized to
  `[A-Za-z0-9._-]` and capped at 64 chars (`core/tools/prompt.py:165-177`,
  `:229`). Fusing it into the neighbouring block instead would hand a `role:
  "x"><System>` value a chance to close the marker and fake another one.
- **`tools` is tri-state, and "declared but unusable" drops everything.**
  `valid_names` is `None` for a plain chat (no tools declared — there is no
  allowlist, so nothing is name-checked) and a *set* otherwise, so an empty set
  can only mean "declared, and no name could be read out of them" — malformed
  entries, or Anthropic-shaped `[{"name":…,"input_schema":{…}}]` with no OpenAI
  `function` wrapper. Every name in that state is unverifiable and is dropped
  with a WARNING naming the array (`core/tools/common.py:420-430`, `:471-488`),
  because a missing call costs the agent one retry while a wrong name costs it
  a dispatch of a tool that does not exist.

**Streamed `tool_calls` deltas carry `index`** — their position in `calls`, in
emission order (`core/engine.py:936-956`). Each delta is otherwise a complete,
standalone call, so `index` is the only field that says which of N parallel
calls it belongs to. OpenAI's streaming contract makes it mandatory on
`tool_calls` deltas and `openai-python` validates every chunk against its
pydantic model, so a delta without one raised *inside* the stream, in the middle
of an agent-loop turn; an accumulator keyed by `index` cannot merge parallel
calls at all. The non-stream path (`core.openai._openai_response`) needs no
index — the calls arrive as one complete list, so position is implicit.

## Models

DeepSeek (v4.1 lineup, Sep 2026) — only these four; no Pro/expert exists yet
(flash superseded v4 Pro):

| id | thinking | search |
|---|---|---|
| `deepseek-flash` | no | no |
| `deepseek-flash-reasoner` | yes (`reasoning_content`) | no |
| `deepseek-flash-search` | no | yes |
| `deepseek-flash-reasoner-search` | yes | yes |

`thinking_enabled`/`search_enabled` are the real backend switches;
`model_type` stays `"default"`. Removed: `deepseek-chat/v3/r1`, `-search`
variants under old names, `deepseek-vision`.

Qwen (`providers/qwen/models.py`): `qwen3.7-plus` (default, thinking +
search), `qwen3.8-max` (thinking + search), `qwen3.8-omni-flash` (thinking,
no search). `/v1/models` serves the union of the active providers.

## TUI / serve modes (`tui.py`, `chat_ui.py`, `cli.py`)

- `main_menu` loop: provider gate first (single/multi select over
  `Settings.provider`/`providers`), then
  `1 Start server / 2 Start server with chat / 3 Run multiple servers /
  4 Settings / 5 Accounts / 6 Dry run / q Quit`.
  Item 3 (`Run multiple servers`) exists only when a chat front-end was
  passed in (`chat_fn is not None`); without one the menu falls back to the
  legacy 1-4 layout `1 Start server / 2 Settings / 3 Accounts / 4 Dry run /
  q Quit` and the on-screen "no accounts" hint renumbers accordingly
  (`tui.py:1449-1503`). CLI
  `--provider`/`--providers` flags do the same non-interactively. Both
  serve paths return to the menu on stop — no app restart to switch
  accounts; API-side account changes are reloaded from disk afterwards
  (`_refresh_from_disk`). `send_dry_run_to_frontend`
  (TUI Settings 6 "Send dry run to frontend", default ON) controls whether the
  reply content carries the printed raw + formatted dump or a short
  terminal-only note.
- Dry run (`dry_run.py`): same port/host/stop-key wiring as serve but no
  bridge, no pool, no preflight. `POST /v1/chat/completions` runs `_prepare`
  (message validation, tool prompts, `append_top`/`append_bottom`), prints
  the raw body + formatted prompt to the terminal, and replies with a normal
  OpenAI-shaped response carrying that dump as the message content
  (`stream: true` gets it as SSE chunks). Nothing is persisted.
- Pre-flight guard (`_preflight_reason`): Start is blocked when the pool is
  empty or no account has a token (first request would 503/401). The menu
  prints why and jumps into `accounts_menu` so the fix is one step away.
- Plain serve (`cli._run_server`, single worker): `uvicorn.Server` in the
  main thread + `_watch_stop_keys` daemon (`msvcrt` on Windows,
  `termios`+`select` on POSIX) mapping `Esc`/`q` to `server.should_exit`;
  `Ctrl+C` arrives via SIGINT. Multi-worker falls back to `uvicorn.run`
  (no Esc listener — it can't reach subprocesses).
- Port auto-fix (`server_runner._ensure_free_port`, thin-wrapped by
  `cli._ensure_free_port`): `cli._port_number` rejects `--port` outside
  1-65535 at parse time; `_ensure_free_port` warns and falls back to scanning
  from 5001 for an out-of-range or non-numeric value and is the only coercion
  point (`_run_server`, chat_ui and dry_run pass the raw value). It then runs
  the socket pre-check, scans `port+1..port+50` when busy, asks `Y/n` on
  a tty (auto-picks headless), and writes the winner to `settings.port` +
  `config.json`. Out-of-range never reaches bind: `socket.bind` raises
  OverflowError (not an OSError), which escaped as a traceback and killed the
  TUI menu.
- Server-with-chat (`chat_ui.py`, stdlib only): same API server in a
  background thread (`log_level=error`, `access_log=False`) + blocking
  `input()` REPL in front posting non-stream completions to
  `127.0.0.1:port` (Bearer = first `settings.keys` entry when set).
  Commands: `/model [name]`, `/clear`, `/quit|/exit`, `/help`.
- Chat log silence: `_silence_chat_logs()` raises `deepseaport.*`,
  `uvicorn*` and root to `ERROR` before the server thread starts, and chat mode
  pins the runtime-only `settings._runtime_log_level = "ERROR"` — the lifespan
  reads `_effective_log_level()` as `_runtime_log_level` > `log_level` > INFO —
  so it stays quiet too (this is what used to spray `deepseaport.obscura`
  warmup INFO across the `You:` line). The saved `log_level` field is never
  touched, so an in-session save (`/model`) cannot persist the override. Levels
  are restored on every exit path, including failed start.
- Chat model memory: `/model` persists to `settings.chat_model`
  (`config.json` + `DEEPSEAPORT_CHAT_MODEL` env); unknown/empty values fall
  back to `deepseek-flash` (`_initial_chat_model`).
- Accounts TUI: `Token (y = auto-token)` — `y` runs provider-specific
  auto-login (`confirmed=True`, no second prompt): DeepSeek via Obscura
  browser login, Qwen via direct-API signin. Anything
  else is a literal token. New accounts become `CURRENT` immediately (TUI,
  CLI `accounts add`, `POST /v1/accounts`, login upsert).

## Environment / files

- `config.json` (gitignored): `{keys, accounts[{email,mobile,password,token,token_expires_at,provider,banned,banned_until}], active_account, obscura_bin, obscura_profile, browser_bin, browser_headless, port, listen, enable_tools, warmup_on_startup, auto_delete_session, max_retries, parallel_challenge_fetch, tool_args_max_chars, forgiving_toolcalls, use_multiple_accounts, log_level, stream_mode, chat_model, enable_append, append_top, append_bottom, send_dry_run_to_frontend, provider, providers}`.
  `enable_append` (default true) gates both appends in `_prepare` (false keeps
  texts stored but sends nothing); `append_top`/`append_bottom` are free-form
  multi-line strings wrapped around
  every outgoing prompt in `_prepare` (top at the absolute top, bottom at the
  absolute bottom); `""` disables. Env: `DEEPSEAPORT_ENABLE_APPEND`, `DEEPSEAPORT_APPEND_TOP/BOTTOM`.
  `AccountConfig.__post_init__` normalizes pasted token JSON (including
  `{"value":null}`) through `tokens.py`, so broken shapes become an empty token
  instead of a mysterious 40003 later. Bans found by the server/login page are
  persisted as `banned`/`banned_until`, so CLI/TUI show `(BANNED: 16 September)`
  — or a plain `(BANNED)` when `banned_until` is missing or unparsable — even
  after a restart and even when the account token is missing/cleared.
  `use_multiple_accounts` (default true): busy/cooldown CURRENT fails over to
  another healthy account (parallel subagents); false pins requests to CURRENT.
  `listen: true` binds `0.0.0.0` (LAN); default false binds `127.0.0.1`; `--host` overrides.
- **An unparsable `config.json` is not an empty one.** `_read_config_json`
  (`config.py:788`) strips a leading UTF-8 BOM (what Windows editors add, and
  what `json.loads` rejects) and repairs it, but genuinely broken input —
  truncated, hand-mangled, or valid JSON of the wrong shape — is latched in
  `_UNPARSABLE_CONFIG` (`config.py:763`) and `save()` **raises** rather than
  writing (`config.py:382`). Loading it as `{}` was how a BOM became "no accounts" and
  the next `save()` — any account action, any TUI toggle, any ban persist —
  rewrote the file with `"accounts": []`, destroying every email, password and
  64-char bearer token in it. The app still *starts* on a broken config (a
  loud `logger.error` says why and that nothing will be written back); it just
  refuses to persist over the file. A successful load clears the latch, so
  fixing the file by hand restores saving. So does a successful `save()`: the
  refusal fires only while the broken file still exists, and without that
  clear the delete-then-save way out would re-arm the refusal the next time a
  save ran against the recreated file. A missing or blank file is a
  legitimate fresh install and is never latched.
- **A present-but-empty env var is treated as absent** (`_env_str`,
  `config.py:740`). `os.environ.get("OBSCURA_BIN", raw.get("obscura_bin", ""))`
  returns `""` for `OBSCURA_BIN=`, so a `.env` copied from the example file
  silently reverted `obscura_bin` to `""` and the WAF path began
  auto-discovering a *different* binary than the operator pinned, with no
  warning. A real value still overrides.
- The load path treats a hand-edited file as hostile input, because every field
  in it can be wrong in a way that is unrecoverable from the outside:
  - **`banned` goes through the string-aware bool** (`_as_bool`,
    `config.py:629-647`, called at `:773`), the same mapping every other
    `Settings` flag gets from the `_bool` closure. `bool("false")` is `True`, so
    a hand-edited `"banned": "false"` used to load as a banned account
    *permanently*: `__post_init__` only expires a ban whose `banned_until` has
    passed, and a false-but-truthy string carries `banned_until: 0.0` — expiry
    unknown, so there is nothing to expire.
  - **`port` is range-checked at load** (`config.py:888-907`). `0` is a legal
    `bind()` argument ("any free port") and never a legal config value: the
    server takes an ephemeral port while every screen keeps printing
    `http://127.0.0.1:0`, so each request fails connection-refused and nothing
    points at the config. `> 65535` reaches `socket.bind()` and raises
    `OverflowError`, which is **not** an `OSError` and so escapes the bind
    path's `except OSError` handlers as a traceback. Both corrections are
    recorded like the `max_retries` clamp, so an unrelated `save()` cannot
    rewrite the operator's own number.
  - **Same-identity accounts are deduped** (`_dedupe_accounts`,
    `config.py:679-714`, over the shared `account_identity` notion). The loader
    builds the list directly and bypasses `AccountPool.add` — the only place
    duplicates used to be rejected — so one login listed twice became two pool
    entries rotating over a single credential, halving the effective pool and
    doubling the load on that account's rate limit. Conservative by
    construction: only entries sharing a provider *and* a non-empty
    email/mobile/token are the same, so a legitimate account is never dropped.
    The surviving identifier is logged at WARNING — silently editing an
    operator's credential list would be worse than the duplicate.
- **`Settings.save`'s degraded path is non-destructive.** The atomic path
  (`mkstemp` + `os.replace`) holds; when `os.replace` is refused — Windows
  rename needs `FILE_SHARE_DELETE` on every open handle, so an editor, AV or
  indexer can block it for a few ms — it falls back to an in-place rewrite that
  opens **without `O_TRUNC`**, copies the previous bytes to `config.json.bak`
  first, truncates only after the new content is written *and* fsynced, and
  rolls the old bytes back on a recoverable write error
  (`config.py:415-546`). `open(path, "w")` truncates at *open*, so before this
  any interruption from there on (Ctrl+C, SIGKILL, power loss, a raise inside
  the write) left `config.json` at **0 bytes**: every email, password, 64-char
  token and API key gone, with no copy anywhere. The residual exposure is a
  crash between the overwrite and the next save leaving `new_prefix +
  old_tail` — invalid JSON, which loads as an empty config, and recoverable
  from the `.bak`. Avoiding even that would mean not writing at all when the
  rename is blocked, silently discarding every ban and token refresh.
- **`Settings` / `AccountConfig` mask secrets in `__repr__`**
  (`config.py:135-157`, `:280-304`): `keys` render as a count (`['<set x2>']`),
  passwords and tokens as `<set>`/`<empty>` — presence, never a length or a
  prefix, because "the token is 64 chars" is itself a credential fingerprint.
  Both objects hold the operator's complete credential set and the dataclass
  repr printed all of it. The concrete sink today is pytest's assertion output:
  any comparison of a `Settings`, or of a list containing one, renders the left
  side with `repr()` and the secret lands in a CI log. `str()` follows
  `__repr__`, so the other sink is one `logger.debug("settings: %s", settings)`
  away. Everything non-secret stays visible — provider, ban markers, expiry,
  `config_path` — because that is what a debugging print is for.
- **The release guard is `scripts/prepare_release.py`.** `dist/` is both the
  PyInstaller output directory and the frozen app's live credential store
  (`config._app_dir()` → `Path(sys.executable).parent` when frozen), so zipping
  it by hand ships `config.json` — gitignored or not. The script reads `dist/`,
  strips `*.bak`/`*.tmp`/`*.log`/profile dirs, refuses to build when a
  `config.json` or any JSON with non-empty `token`/`password`/`keys` is present,
  and never writes inside the source tree. Guarded by
  `tests/test_release_guard.py`; `deepseaport.spec` has no `datas` exclusion,
  so this script is the packaging step, and `.github/workflows/release.yml`
  runs `--check` as a hard gate on a `v*` tag before it publishes.
- `data/` (gitignored): PoW wasm. `~/.deepseaport/obscura-profile`: WAF cookies + login localStorage.
- `src/deepseaport/real_browser.py` drives Helium/Chrome/Edge/Chromium login
  over CDP; `browser_paths.py` discovers the binary. `auth.py` chooses real
  browser first and falls back to Obscura; `mcp_client.py` is the Obscura
  stdio transport. None of these are on the HTTP hot path.
- `GET /v1/waf/status` shows binary/profile/WAF-cookie state and REQUIRES the
  API key (`server_routes.py:151-158`): the payload names the absolute
  obscura binary path, the absolute profile path and the WAF cookie names,
  which is what a LAN scanner wants. Nothing polls it (readiness uses
  `/health`), so the gate costs nothing. `/health` deliberately stays
  unauthenticated — `server_runner._wait_for_server` and uptime checks read
  only the status code, and it reports `current` as a boolean precisely so it
  does not leak the account identity.
- **`tests/test_live_smoke.py` is the upstream-drift detector.** Everything else
  in the suite runs against stubs, and a stub cannot detect that DeepSeek
  renamed an SSE field, that Qwen moved `reasoning` out of `delta.phase`, or
  that a parser now truncates — the stub *is* the thing that broke. The `live`
  tier (`pytest -m live`, opt-in, never in CI) makes one real request per
  provider and asserts the invariants that a stub cannot express: the reply
  contains the *end* of the prompt (the truncation guard — a stateless parser
  once shipped and returned `web` instead of `web2api-ok`), exactly one
  `data: [DONE]`, one consistent `id`/`created` across chunks, concatenated
  stream deltas equal to the non-stream content, a populated
  `reasoning_content` on reasoner models, an `index` on every streamed
  tool_call delta with no hallucinated tool name, and a `role:"tool"`
  follow-up turn that succeeds. Diagnostics go through
  `core.engine.scrub_upstream_text`, and a hermetic meta-test (deliberately
  NOT `live`-marked, so it runs in CI) asserts no diagnostic path can print a
  credential while still preserving the words an operator diagnoses drift
  from. Each provider skips with its own blocker named, so one configured
  upstream is enough. **Run it periodically** — it is the only check in the
  repo that can fail because the internet changed.
- Ad-hoc live verification scripts stay out of the repo; `tests/test_live_smoke.py` supersedes them.

## Provider architecture (multi-provider refactor)

- Folder layout: `core/` holds provider-neutral code (provider ABC, registry,
  pools, tools, transport, `openai` response shim, tokens, `engine.py`
  retry/failover mechanics); each upstream lives in `providers/<name>/`
  (`protocol`, `client`, `sse_parser`, `pow`, `auth`, `models`, `provider`,
  `completion_service.py` adapter). DeepSeek keeps session/PoW/SSE/ban logic
  in `providers/deepseek/completion_service.py`, Qwen keeps chat lifecycle in
  `providers/qwen/completion_service.py`, opencode keeps
  the serve session lifecycle in `providers/opencode/completion_service.py`;
  all
  three delegate generic loops to `core/engine.py`.
- Registry (`core/registry.py:26-27`): lazy discovery via `_BUILTIN_MODULES`
  (`deepseek`, `qwen`, `opencode`) + `create_provider()` factories;
  failures log and never raise; never imports `deepseaport.server` (no cycle).
  The imports run under `_LOAD_LOCK` and the registry plus the `_LOADED` flag
  are published
  as one step (`core/registry.py:45-82`). `_LOADED` used to latch *before* the
  imports, so a concurrent reader could see a half-populated registry:
  `resolve_model()` returned None and a perfectly valid model answered 404.
  That window was not theoretical — `warmup_on_startup` defaults on and the
  lifespan yields while its daemon thread imports, so uvicorn accepts
  connections for the whole load. A module that fails to import is logged and
  skipped, so it costs only that provider and one warning; the load still
  latches, because re-importing a module named one line away in
  `_BUILTIN_MODULES` on every access only bought a log flood.
- Provider-scoped pools (`core/pools.py`): `active_provider_ids`,
  `accounts_for_provider`, `make_pool` build one `AccountPool` per active
  provider; failover stays within provider.
- Account binding: `AccountConfig.provider` pins an account to one upstream
  (default `deepseek`); TUI/CLI `accounts add` binds the selected provider.
- Selection: first-menu provider gate + `--provider`/`--providers` flags over
  `Settings.provider`/`providers` (`config.json`, `DEEPSEAPORT_PROVIDER(S)`).
- Multi mode: 2+ ids in `Settings.providers`; `/v1/models` serves the union
  catalogue; each provider keeps independent auto-failover (no shared CURRENT).
  A **single** entry is ignored rather than honoured
  (`core/pools.py:31-62`): one id would flip `is_multi()` and take the whole
  per-provider pool path — independent failover, union catalogue — for what
  reads like single-provider setup. What was wrong with it was the silence:
  `provider: "deepseek"` + `providers: ["qwen"]` ran deepseek with no hint that
  qwen had been written down and dropped. It now warns once per ignored id (the
  set is process-global because `active_provider_ids` sits on the per-request
  path), naming the entry and the one in effect.
- Shared shim: `core/openai._openai_response` shapes OpenAI replies for all
  providers; tool parsing resolves through `deepseaport.server.parse_tool_calls`
  at call time so test monkeypatches keep working.
- Cross-provider contract that every adapter must keep (see "Request
  lifecycle" for the code citations): a bare 401 is `INVALID_TOKEN` — retry,
  ban-probe, cross-account failover — and only confirmed ban evidence sets
  `accounts[].banned`; a raising `recover()` is contained by
  `core.engine._run_recovery` so the original upstream status survives; and
  every adapter must pass `buffered=not live` into
  `new_producer_context(...)` so the consumer watchdog covers the whole
  producer budget rather than one socket read. All three comply — each
  `providers/*/completion_service.py` passes `buffered=not live` into
  `_engine.new_producer_context(...)` — so `stream_mode` means the
  same thing whichever upstream is serving.
  The `RetryKind` enum is per-provider too, and deliberately so: its members
  are facts about *that* upstream. A member that exists but is never produced
  is fine; a member produced from the wrong evidence is not — and every
  provider's terminal table must cover every member, so a kind can never exist
  in the retry table and not in the terminal one.
- New providers: start from `providers/qwen/` (reference #2) for a web
  upstream or `providers/opencode/` (reference #3) for a local one, then add
  one line to `_BUILTIN_MODULES`. New providers write a thin adapter over
  `core/engine.py` — do NOT copy `completion_service.py`. See `PROVIDERS.md`.
- Back-compat: old module paths (`deepseaport.server`, `completion_service`,
  `client`, ...) remain as alias shims; all monkeypatch seams preserved.

### The opencode provider (a local CLI, not a web upstream)

The third registered provider and the odd one out: it is a **local process**,
so none of the web-provider machinery applies. It speaks opencode v2's HttpApi
(every route under `/api`, live-verified against 2.0.29), so the wire notes
below are v2-specific. Full protocol detail lives in
`src/deepseaport/providers/opencode/README.md`; this is the shape.

- **`requires_accounts = False`** (`providers/opencode/provider.py:29`). No
  tokens, no `AccountPool`, no `AccountConfig`. The route layer reads the flag
  and skips acquisition entirely — `needs_slot` in
  `server_routes.py:334` means `pool` stays `None` and `item` is passed as
  `None` (the opencode engine ignores both). Downstream: `tui.py:1343` and
  `chat_ui.py:73` filter the provider menu so accountless providers are
  selectable with an empty account list, `_preflight_reason` passes with zero
  accounts, and `cli.py:656` bypasses the pool in chat mode. The adapter still
  accepts `pool`/`item` in its signatures so the route contract is unchanged.
- **No PoW, no WAF, no ban, no login.** `probe_ban` is `(False, None)`,
  `login` raises, `warmup` is a log-only `GET /api/info` probe that boots the
  managed serve when nothing answers. The free Zen models need no
  `opencode auth login` (that is only for key-gated providers).
- **Managed isolated serve** (`providers/opencode/managed_serve.py`). The first
  request boots a dedicated `opencode serve` child on a free localhost port
  instead of talking to whatever the operator happens to be running. The child
  gets child-only env: `XDG_CONFIG_HOME` → an empty dir,
  `OPENCODE_CONFIG_DIR` → a generated dir carrying a `strip-system.js` plugin,
  `OPENCODE_CONFIG_CONTENT` → a generated `deepseaport-passthrough` text-only
  agent, plus `DEEPSEAPORT_*`/`DEEPSEEK_*`/`OPENCODE_*` stripped and an
  ephemeral basic-auth password. The env strip runs *before* those keys are
  set (`_child_env`), which is the actual security boundary: without it a
  sourced `.env` would land verbatim in a process the model can be talked into
  reading via prompt injection. `~/.config/opencode`, project files and plain
  `opencode` runs are untouched. Extra workers reuse the instance through the
  `serve.json` port file, and a recorded serve is reaped only after a retried
  probe (`_probe_alive`, 3 attempts) fails — one missed probe no longer kills a
  busy instance. Opt out with `DEEPSEAPORT_OPENCODE_BASE` (use your own serve
  as-is) or `DEEPSEAPORT_OPENCODE_MANAGED=0`.
- **Since v2 the serve always requires HTTP basic auth.** It generates a
  password at boot and prints it unless `OPENCODE_SERVER_PASSWORD` pins one, so
  the managed child sets that env and hands the same value to the client
  (`serve.json` stores it, owner-only; a probed base must be given the password
  too). An operator's own serve is used with
  `DEEPSEAPORT_OPENCODE_USERNAME/PASSWORD`; a 401 is terminal and names both.
- **The turn is prompt-then-poll.** v2 has no blocking message endpoint:
  `POST /api/session/:id/prompt {text}` returns at once with the created user
  item while the agent loop runs in the background, so `send_message` polls
  `GET /api/session/:id/message?limit=&order=desc` until the newest assistant
  item AFTER this turn's user item carries `time.completed`. Model, reasoning
  `variant` and agent are **session-scoped** in v2, so every turn POSTs
  `/api/session/:id/model` and `/agent` before the prompt — without that a
  session born from an empty create body answers from whatever model it happens
  to carry. An unusable model/variant is not a 4xx: it is accepted and fails
  the turn, which is why the serve's own words survive into the error text.
  A disconnect cancels both paths the same way: `send_message` / `_await_turn`
  take `cancel_event`, abort the session and raise the same `_StreamCancelled`
  the live tail raises.
- **Live streaming is real, not buffered** (this is the opencode-specific
  answer to the `stream_mode` discussion above, and the one place a provider
  does *not* use the engine's blocking producer). In `stream_mode: live`
  (`completion_service._stream_completion`) it runs
  `_run_completion_core(..., live=True)`, which calls
  `client.stream_message` — that subscribes to `GET /api/event` **first** (so
  early deltas are not missed), sends the prompt, and forwards
  `session.text.delta` / `session.reasoning.delta` through
  `core.engine.live_emit_pair` as they arrive. `session.execution.succeeded` /
  `.failed` ends the tail, then the authoritative result is read from the same
  session timeline; a partial tail is never returned as the result. Only socket
  timeouts are tolerated in the SSE read loop — any other read failure records
  `SSE_READ`, stops the reader and surfaces terminally unless the timeline read
  shows the turn completed, and the wait loop no longer stalls to its deadline.
  The consumer watchdog is sized from `P.opencode_timeout`, the same env knob
  the producer runs under. Two deliberate rules: thinking always streams, but
  **content streams live only when the client sent no `tools`** — otherwise a
  tool_calls reply would also leak as user-visible text. If the SSE tail cannot
  start, it falls back to the blocking result fanned out as chunks, so
  `stream_mode: buffered` and the fallback are the same code path.
- **`reasoning_effort` → `variant`.** The one request-level knob the serve API
  honors. `server_routes._parse_reasoning_effort` lifts `reasoning_effort`
  (string) or `reasoning.effort` from the OpenAI body and validates it against
  the built-ins `none/minimal/low/medium/high/xhigh`, and the completion path
  forwards it as the `variant` on the session's model select. An unknown value
  is warned-and-dropped (`_extract_passthrough`), then falls back to one
  variant-less resend instead of failing the turn. `temperature` has no field
  upstream on v2, so it stays warn-and-drop like the web providers.
- **Tool access is the managed permission map, and nothing else.** v2 has no
  message-level `tools` map — the prompt body is `{text}` and nothing else — so
  the map in `managed_serve.managed_config_content` (`read`/`glob`/`grep`/
  `subagent` and `edit`/`bash`/`webfetch`/`external_directory`, all `deny`,
  globally and on the agent) is the single layer that bounds what the model
  may call. `read`-like tools are denied because sessions are rooted at the
  operator's cwd, where `config.json` can sit, and `subagent` because a
  spawned agent carries its own permission set. The free-tier
  gatekeeper survives only as a terminal classification
  (`protocol.is_free_tier_error` → `OPENCODE_FREE_TIER_DENIED`); the
  declare-the-native-keys trick that used to ride the message body is history.
- **Control plane and terminal errors.** Every serve request carries
  `User-Agent` (`opencode/<cli version>`), `x-opencode-client` (default `cli`),
  `x-opencode-project` (default `global`) and `X-Session-ID`. Sessions are
  pinned to the caller's working directory via `location.directory` on session
  create (`DEEPSEAPORT_OPENCODE_WORKDIR`) so native tools run where the harness
  expects; v2 has no `?directory=` query. Client failures are `RuntimeError`s
  with a code prefix the classifier reads (`OPENCODE_SERVE_DOWN`,
  `OPENCODE_HTTP_<status>`, `OPENCODE_UPSTREAM`, `OPENCODE_EMPTY`,
  `OPENCODE_INVALID_SESSION`, `OPENCODE_RATE_LIMITED`,
  `OPENCODE_FREE_TIER_DENIED`); free-tier refusals are terminal with an
  upgrade hint and serve-down becomes 502 + an actionable hint. Retry table is
  overload-only (there are no accounts to fail over to) plus one inline
  stale-session recreate outside the budget.
- **Model catalogue.** Two layers in `providers/opencode/models.py`:
  `live_specs()` reads `GET /api/model` (thinking when an entry declares
  `compatibility.reasoningField`; disabled and non-`active` entries dropped)
  and `MODELS_SPEC` is the 8-entry offline fallback used when
  no serve answers — seven `*-free` plus `opencode/big-pickle`, which is *not* a
  `-free` id. One process-wide cache under a lock holds a success for 5 minutes
  and a failure for 30; when nothing answers at the configured base the read
  boots the managed serve and retries, so the catalogue does not depend on which
  process asks. `default_model()` is the explicit
  `opencode/muse-spark-1.3-contributor-free`, never the first catalogue key.
  `protocol.split_model` splits `providerID/modelID` at the FIRST slash, and
  `model_spec` accepts any other id containing a slash as a passthrough
  (user-configured proxy providers on the same serve) so those work in chat.
  Bare catalogue-unknown ids stay unknown, preserving the unknown-model 404
  contract.

## Tests: what the suite does and does not prove

The default run (`pytest -m "not live" -q`, exactly what CI runs) is fully
hermetic: `tests/conftest.py` points the config/data at a temp tree, strips
every `DEEPSEAPORT_*`/`DEEPSEEK_*`/`OBSCURA_*` var, raises on any non-loopback
socket connect, and asserts the imported package is THIS checkout's `src/`.
That guard is why the suite is trustworthy — and also why it has a hard limit.

- **What it proves:** the retry/failover ladder, account-slot accounting and
  cancellation, the tool-call render/parse round trip, the config save
  semantics, the route and auth surface, and the wasmtime PoW path (against the
  real shipped module, not a stub — see "PoW details"). Those are *our* logic,
  and they are heavily covered.
- **What it cannot prove:** that an upstream still speaks the protocol we
  reverse-engineered. Every wire format in this project is empirical, and the
  upstreams change without notice. That is what `pytest -m live` is for, and
  why the live tier must be run periodically — see
  `tests/test_live_smoke.py` in "Environment / files".
- **Timing tests assert ordering, not duration.** A wall-clock budget
  (`assert elapsed < 1.0`) is the classic source of CI flakes, and this suite
  now uses `threading.Event` handshakes instead: the intent ("cancellation was
  observed while the producer was still in flight") is provable without
  measuring the clock. The one wall-clock bound that remains is a watchdog
  *ratio*, and it is written as a derived multiple of the value under test so
  the margin cannot silently decay. A `time.sleep(N)` longer than the watchdog
  it is racing is also a process-level cost even when pytest reports the test
  as fast, because it parks a non-daemon executor worker.
- **Two live smoke meta-tests are deliberately NOT `live`-marked** (the
  credential-leak check and the gate-wording check). They are pure and
  hermetic, so they run in CI. A policy test that only executes when someone
  opts into the live tier is a policy test nobody runs.
- **The live tier covers every registered provider.**
  `_run_provider_suite(pid)` runs the same five probes per provider, so
  adding an upstream is one entry point, not a new suite.
- **The credential-leak meta-test is parameterised over credential *shapes*.**
  It asserts that no diagnostic path can print any
  of the secrets it is given — not just the one driving the run, because a
  formatter that special-cases the shape under test while leaking a different
  one has moved the problem rather than fixed it. That meta-test is
  deliberately NOT `live`-marked, so it runs in CI.
