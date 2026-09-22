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

Hot path is **direct HTTP, never DOM driving** (both providers).
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
`INVALID_SESSION_ID`, Obscura warmup + retry once on 403/WAF. Mark account
bad (cooldown) on ban/restricted/401.

Qwen path (no PoW; `providers/qwen/completion_service.py` over
`core/engine.py`): acquire account → pre-check `token_expires_at` with one
proactive direct re-signin when a password is stored → `POST
/api/v2/chats/new` → `POST /api/v2/chat/completions?chat_id=<id>` (`?chat_id=`
required; `Cookie: token=<jwt>` + solved WAF jar; `Authorization: Bearer`
kept too) → best-effort `DELETE /api/v2/chats/{id}`. Qwen SSE is
OpenAI-style JSON per `data:` line (no `{p,v}` protocol); 401s probe
`GET /api/v2/auths/` first (ban masquerading as 401 → 403 + failover), then
one direct re-signin. Details in `src/deepseaport/providers/qwen/README.md`.

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
  refresh attempt, with login backoff after captcha/credential failure.
- Old/stale tokens fail with `40003 Authorization Failed` at session create.
- `POST /v1/accounts/unblock` (and `deepseaport accounts unblock [identifier]`)
  clears the in-memory cooldown and the persisted ban marker.

## PoW details (DeepSeek only; Qwen has no PoW)

- Wasm URL host is **`fe-static.deepseek.com`**, not `chat.deepseek.com`:
  `https://fe-static.deepseek.com/chat/static/sha3_wasm_bg.7b9ca65ddd.wasm`.
  Downloading from the wrong host returns the SPA `index.html` (wasmtime then
  fails with `expected '('`) — validate size/magic, not just HTTP 200.
- `PoW.ensure_wasm()` downloads on demand into `data/`; solving takes ~0.5 s.

## Tool calling (agentic harness)

Web model has no native function calling. `tools_support.render_prompt()`
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
200000); opt-in `Settings.forgiving_toolcalls` /
`DEEPSEAPORT_FORGIVING_TOOLCALLS` (default off) enables the forgiving bucket
(markup repairs: pipe/space/HTML-escaped DSML markers, smart attribute
delimiters with verbatim values, single-tool missing-name/param inference,
JSON invoke bodies); replies that look like a truncated tool attempt surface
`finish_reason: "length"` instead of masquerading as a normal stop.
Stream and non-stream both end with `finish_reason: "tool_calls"`.
Multi-step loops verified live (weather → tool result → final answer).

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
  `Settings.provider`/`providers`), then `1 Start server`,
  `2 Start server with chat` (chat arg omitted → legacy 1-3 layout),
  `Settings`, `Accounts`, `Dry run` (bottom, before `q Quit`). CLI
  `--provider`/`--providers` flags do the same non-interactively. Both
  serve paths return to the menu on stop — no app restart to switch
  accounts; API-side account changes are reloaded from disk afterwards
  (`_refresh_from_disk`). `send_dry_run_to_frontend`
  (Request handling 6, default ON) controls whether the reply content carries
  the printed raw + formatted dump or a short terminal-only note.
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
  `cli._ensure_free_port`): socket pre-check; busy port scans
  `port+1..port+50`, asks `Y/n` on a tty (auto-picks headless), writes the
  winner to `settings.port` + `config.json`.
- Server-with-chat (`chat_ui.py`, stdlib only): same API server in a
  background thread (`log_level=error`, `access_log=False`) + blocking
  `input()` REPL in front posting non-stream completions to
  `127.0.0.1:port` (Bearer = first `settings.keys` entry when set).
  Commands: `/model [name]`, `/clear`, `/quit|/exit`, `/help`.
- Chat log silence: `_silence_chat_logs()` raises `deepseaport.*`,
  `uvicorn*` and root to `ERROR` before the server thread starts, and
  `settings.log_level` is overridden to `ERROR` in-memory (never saved) so
  the lifespan's `apply_log_level()` stays quiet too (this is what used to
  spray `deepseaport.obscura` warmup INFO across the `You:` line). Levels
  are restored on every exit path, including failed start.
- Chat model memory: `/model` persists to `settings.chat_model`
  (`config.json` + `DEEPSEAPORT_CHAT_MODEL` env); unknown/empty values fall
  back to `deepseek-flash` (`_initial_chat_model`).
- Accounts TUI: `Token (y = auto-token)` — `y` runs provider-specific
  auto-login (`confirmed=True`, no second prompt): DeepSeek via Obscura
  browser login, Qwen via direct-API signin (`provider.login()`); anything
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
  even after a restart and even when the account token is missing/cleared.
  `use_multiple_accounts` (default true): busy/cooldown CURRENT fails over to
  another healthy account (parallel subagents); false pins requests to CURRENT.
  `listen: true` binds `0.0.0.0` (LAN); default false binds `127.0.0.1`; `--host` overrides.
- `data/` (gitignored): PoW wasm. `~/.deepseaport/obscura-profile`: WAF cookies + login localStorage.
- `src/deepseaport/real_browser.py` drives Helium/Chrome/Edge/Chromium login
  over CDP; `browser_paths.py` discovers the binary. `auth.py` chooses real
  browser first and falls back to Obscura; `mcp_client.py` is the Obscura
  stdio transport. None of these are on the HTTP hot path.
- `GET /v1/waf/status` shows binary/profile/WAF-cookie state for debugging.
- Live verification scripts used during development live in the temp dir, not
  the repo: `verify_live2.py` (session→PoW→completion), `verify_tools.py`
  (tool loop), `verify_http.py` (HTTP stream/tools/reasoner), `verify_models.py`.

## Provider architecture (multi-provider refactor)

- Folder layout: `core/` holds provider-neutral code (provider ABC, registry,
  pools, tools, transport, `openai` response shim, tokens, `engine.py`
  retry/failover mechanics); each upstream lives in `providers/<name>/`
  (`protocol`, `client`, `sse_parser`, `pow`, `auth`, `models`, `provider`,
  `completion_service.py` adapter). DeepSeek keeps session/PoW/SSE/ban logic
  in `providers/deepseek/completion_service.py`, Qwen keeps chat lifecycle in
  `providers/qwen/completion_service.py`; both delegate generic loops to
  `core/engine.py`.
- Registry (`core/registry.py`): lazy discovery via `_BUILTIN_MODULES`
  (`deepseek`, `qwen`) + `create_provider()` factories; failures log and never
  raise; never imports `deepseaport.server` (no cycle).
- Provider-scoped pools (`core/pools.py`): `active_provider_ids`,
  `accounts_for_provider`, `make_pool` build one `AccountPool` per active
  provider; failover stays within provider.
- Account binding: `AccountConfig.provider` pins an account to one upstream
  (default `deepseek`); TUI/CLI `accounts add` binds the selected provider.
- Selection: first-menu provider gate + `--provider`/`--providers` flags over
  `Settings.provider`/`providers` (`config.json`, `DEEPSEAPORT_PROVIDER(S)`).
- Multi mode: 2+ ids in `Settings.providers`; `/v1/models` serves the union
  catalogue; each provider keeps independent auto-failover (no shared CURRENT).
- Shared shim: `core/openai._openai_response` shapes OpenAI replies for all
  providers; tool parsing resolves through `deepseaport.server.parse_tool_calls`
  at call time so test monkeypatches keep working.
- Templates: `providers/glm|kimi/` are import-safe stubs (empty models,
  `NotImplementedError` login/complete) activated by filling `models.py`,
  implementing protocol/client + `complete()`, then adding one line to
  `_BUILTIN_MODULES`. Qwen is a full provider (reference #2). New providers
  write a thin adapter over `core/engine.py` — do NOT copy
  `completion_service.py`. See `PROVIDERS.md`.
- Back-compat: old module paths (`deepseaport.server`, `completion_service`,
  `client`, ...) remain as alias shims; all monkeypatch seams preserved.
