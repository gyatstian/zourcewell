# Providers

Goal: adding another web2api upstream is a small, mechanical task — reverse
its web API into one `providers/<name>/` folder and reuse everything else.
**Three** providers are registered and complete: two web upstreams — DeepSeek
(reference) and Qwen (reference #2) — plus opencode (reference #3, a local CLI,
not a web upstream). Generic retry/failover mechanics live in
`core/engine.py`; each provider keeps only its session/chat setup, PoW solve,
WAF warmup body, error classification, and invalid-token terminal in its own
`completion_service.py` adapter.

The one thing that is NOT provider-neutral, and never will be, is what an HTTP
status *means* — see the per-provider 403 split below.

## Layout

```
src/zourcewell/
  core/                  provider-neutral: never imports zourcewell.server at top
    provider.py          Provider ABC + ModelSpec
    registry.py          lazy discovery via _BUILTIN_MODULES + create_provider()
    pools.py             active_provider_ids / accounts_for_provider / make_pool
    engine.py            generic retry/failover/stream mechanics (drive_retries, run_with_failover, ...)
    tools/               PACKAGE, not a module: prompt rendering + tool-call parsing
      __init__.py        re-exports render_prompt / parse_tool_calls / ...
      common.py          MAX_ARGS_CHARS, MAX_BARE_OBJECTS, MAX_PARAMS_PER_INVOKE, _int_env
      prompt.py          render_prompt / tool_system_prompt / tool_reminder_prompt
      parser/            __init__ dispatch + bare.py, dsml.py, fence.py, json_array.py, tags.py
    transport.py         SSE drain helpers + http_error_code(status, forbidden=...)
    openai.py            _openai_response: OpenAI response shaping
    tokens.py            extract_token (localStorage JSON/raw normalization)
    ...                  obscura_bridge, mcp_client, browser_paths, types
  providers/<name>/      one folder per upstream (deepseek = reference)
    __init__.py          `from .provider import <Name>Provider, create_provider`
    provider.py          Provider subclass + create_provider() factory
    models.py            MODELS_SPEC + DEFAULT_MODEL
    protocol.py          headers / payloads / ban-message helpers
    client.py            session / challenge / stream HTTP calls
    sse_parser.py        delta/error/usage event parsing
    pow.py               challenge solver + wasm handling (deepseek only)
    auth.py              browser login flow
    login_policy.py      selectors / phrases / timeouts (deepseek)
    completion_service.py  provider-specific retry/failover ADAPTER over core.engine
    real_browser.py      CDP login driver (deepseek)
    README.md            per-provider reversing notes — see below for which are real
  providers/opencode/    local-CLI reference: no accounts, no pool slot, no
                         login (`requires_accounts = False`; a session per
                         request against a managed isolated serve, or
                         user-managed `opencode serve` via
                         ZOURCEWELL_OPENCODE_BASE). Speaks the opencode v2
                         HttpApi (all routes under `/api`); a turn is
                         prompt-then-poll because v2 has no blocking message
                         endpoint, and the model, its `variant` and the agent
                         are set per session, not per message;
                         `reasoning_effort` forwards as that `variant`;
                         native tools are bounded only by the managed agent's
                         permission map; each matching `permission.asked`
                         event is immediately rejected by the client. Pi tools
                         ride DSML text ->
                          OpenAI `tool_calls` (see README "Pi harness +
                          opencode").
  server.py              thin wiring: create_app + re-exports (historical surface)
  server_routes.py       FastAPI endpoints (dispatch via provider.complete_request / stream_response; shaping via _server()._openai_response)
  accounts.py            AccountPool (per-provider instances via core.pools)
```

App shell (`server.py`, `server_routes.py`, `server_concurrency.py`, `tui.py`,
`cli.py`, `config.py`) is provider-agnostic and resolves providers through
`core.registry` + `core.pools`.

### What the tree above does not say

- **Which `completion_service.py` is real.** DeepSeek and Qwen both exist and
  are both full retry/failover adapters over `core.engine` — Qwen is not a
  DeepSeek-only shape, it just has no PoW/wasm/browser-refresh (see its own
  module docstring, lines 1-9). opencode is the third adapter, with a reduced
  retry table (overload only; there are no accounts to fail over to) and a
  live `/api/event` SSE path. What a NEW provider must not do is copy one of
  those files — write a thin adapter over `core.engine` (see
  "Reuse vs per-provider").
- **Which `README.md` is real.** `providers/qwen/README.md` and
  `providers/opencode/README.md` are substantial, tracked reversing notes for
  finished providers. `providers/deepseek/` has no README at all — its
  reversing notes live in the top-level `ARCHITECTURE.md`.
- **The opencode catalogue is live, with a static fallback.** `models.py` reads
  `GET /api/model` from the serve instance (which also lists the operator's
  other authenticated providers, all exposed as `providerID/modelID`) and falls
  back to the hand-verified fallback list when no serve answers; exact ids and
  current availability live in `providers/opencode/models.py`; the two are
  never merged. Cached process-wide: a success for five minutes, a failure
  for thirty. When nothing answers at the configured base the read boots the
  managed serve and retries, so the ids do not depend on which process asks;
  both opt-outs (an operator-pinned `ZOURCEWELL_OPENCODE_BASE`, and
  `ZOURCEWELL_OPENCODE_MANAGED=0`, which the hermetic tier sets) are honored.
  A blank model id resolves to the explicit
  `opencode/muse-spark-1.3-contributor-free` (`models.DEFAULT_MODEL`), NOT
  the first catalogue key — the live catalogue is in upstream order and
  turns over weekly. A MENU goes through `provider.models_hint()` instead,
  which is `models.cached_specs` — the last read this process did — else the
  static fallback: no I/O, so a row never costs a dead-base timeout and never
  boots the serve. Pin:
  `models.live_specs` (the read, and its cache), `models.cached_specs` (the
  no-I/O menu view), `models.MODELS_SPEC` (the fallback), `models.DEFAULT_MODEL`.

## Provider lifecycle

1. Select at serve: TUI first-menu provider gate, `--provider`/`--providers`
   CLI flags, or `Settings.provider`/`Settings.providers` (`config.json` +
   `ZOURCEWELL_PROVIDER`/`ZOURCEWELL_PROVIDERS`).
2. Scoped pools: `core.pools.make_pool(settings, pid)` builds one
   `AccountPool` per active provider from its bound accounts
   (`AccountConfig.provider`).
3. Model routing: `registry.resolve_model(model)` / `provider.tuple_spec()`
   map a model id to its provider; `/v1/models` serves the union.
4. Warmup: the shared-host Obscura warmup (`CHAT_HOME`, the DeepSeek host) runs
   only when an active provider has `needs_waf_jar` (qwen and opencode set it
   `False`; qwen's jar is its own qwen-host cache), and every active provider's
   `warmup(bridge)` still runs once at startup (DeepSeek prefetches PoW wasm).

## The 403 meaning is per-provider

**This section is the authority `core/transport.py::http_error_code` cites when
it says the 403 mapping is injectable rather than hardcoded.** Read it before
adding a provider: the same status code means different things upstream, and
collapsing them is a real, shipped bug (see the regression below).

`http_error_code(status, forbidden=...)` maps a status to an event code:

| Provider | `forbidden` passed | 403 becomes | Why |
|---|---|---|---|
| DeepSeek | omitted / `None` | `HTTP_403_WAF` | A DeepSeek 403 **is** the AWS WAF challenge — it is never a plain authorization failure, so it must route to WAF recovery (Obscura warmup + one retry). |
| Qwen | `{}` (empty dict) | `HTTP_403` | Qwen sits behind **Aliyun** WAF, which does **not** answer 403: it returns **HTTP 200 with a `text/html` challenge body** (`WAF_CHALLENGE`, detected in `providers/qwen/protocol.py`). So a real Qwen 403 is a genuine upstream status — a suspension body, say — and must stay a plain `HTTP_403` so `classify_error` can match the ban branch. |
| any other | your choice | your choice | The table is a plain `dict[int, str]`; only the 403 key is a mapping, every other status always falls through to `f"HTTP_{status_code}"`. |

The bug this prevents: before the mapping was injectable, both providers
emitted `HTTP_403_WAF` for a 403. A Qwen account suspended upstream was
therefore classified WAF — it got a WAF re-prime and a **502** instead of a
**403 + cooldown + failover** — and the account was never cooled down. When
adding a provider, first find out what its WAF actually answers with
(`403`? `200 + html`? a JSON envelope?) and pass a `forbidden` table that says
so. **An empty table is the correct answer when there is no WAF at all** — it
is not "unconfigured".
`tests/test_provider_parity_gaps.py` pins the injectable halves.

## `Provider` ABC contract (`core/provider.py`)

| Method | Purpose |
|---|---|
| `models()` | Return `{id: ModelSpec}` catalogue. May be a live read that boots a local process (opencode); not for menus. |
| `models_hint()` | The same catalogue for a MENU row: no I/O, nothing started. Default: `models()`; opencode returns the last read, else the static fallback. |
| `model_spec(model)` | Case-insensitive spec lookup, else `None`. |
| `probe_ban(headers, timeout)` | Return `(is_banned, mute_until)`. |
| `login(email, password, settings)` | Authenticate, return login-result object. |
| `warmup(bridge)` | Best-effort startup warmup (default no-op). |
| `complete(app, pool, prep, first_item, cancel_event=None, on_content=None, on_thinking=None)` | Blocking completion. `cancel_event` is optional and is forwarded as a keyword. **Called by `cli.cmd_chat`** (one-shot chat, which has no cancellation source). |
| `complete_request(app, pool, prep, item, cancel_event)` | Blocking completion, `cancel_event` required and forwarded positionally. **Called by the non-stream chat route** (`server_routes`). |
| `stream_response(app, pool, item, prep)` | Async-gen streaming. **Called by the streaming chat route.** |
| `tuple_spec()` | `{id: (thinking, search, model_type)}` compat view. |

All three completion seams **raise `NotImplementedError` on the ABC** —
`core/provider.py` holds no default engine and no `_server()` reference to
late-bind. An override is therefore not optional: a provider that ships without
one answers its first request with a 500, not a fallback. The module docstring
and `src/zourcewell/providers/AGENTS.md` say the same thing; if you find a
fourth seam, give it a caller or do not add it.

## Add a new provider in 4 steps

The three finished providers are all available as references and they are
**not** interchangeable, so pick by upstream shape before copying anything:

| reference | use it when | what it teaches you |
|---|---|---|
| DeepSeek | the protocol is opaque and gated | the full ladder: PoW, WAF re-prime, session recreate, token refresh |
| Qwen (#2) | the protocol is closed but the WAF is not a 403 | injected 403 table, solved-jar warmup, JWT cookie scrubbing, chat-id lifecycle |
| opencode (#3) | the upstream is local, not a web API | `requires_accounts = False`, managed child process, live non-blocked streaming |

- [ ] 1. Fill `providers/<name>/models.py` (`MODELS_SPEC`, `DEFAULT_MODEL`).
      Every row's `owned_by` is **`provider.id`** (e.g. `"qwen"`), never
      a vendor name: `/v1/models` attributes each row to the provider that
      actually serves it, and `tests/test_live_smoke.py::_probe_catalogue`
      asserts `owned_by == provider.id` live. A wrong vendor string there is a
      failing live test, not a cosmetic one — and the same probe is what turns
      **catalogue drift** (the origin serves a model this file never heard of)
      into a failure instead of a 404 at the first real turn.
- [ ] 2. Implement `providers/<name>/protocol.py` + `client.py`
      (+ `sse_parser.py`, `pow.py`, `auth.py` as needed) plus a thin
      `completion_service.py` adapter over `core.engine` — do NOT copy
      `providers/deepseek/completion_service.py`.
- [ ] 3. Override ALL THREE seams — `complete`, `complete_request`,
      `stream_response` — in `providers/<name>/provider.py`, reusing
      `core.engine`, `core.tools`, `core.transport`,
      `core.openai._openai_response` and `core.pools.make_pool`
      (`zourcewell.accounts` pool). None has a default; see the ABC table
      above. Each delegates to its own `completion_service`.
      Every row of the retry table must go through the local
      `_retryable(kind)` predicate, which ANDs `not tracker.emitted` — a
      recovery after a live delta has already reached the client re-POSTs a
      non-idempotent completion and re-streams from delta 0, duplicating text
      or splicing two halves of one `tool_calls` object into a 200 stream that
      still ends in `[DONE]`. The rule and its reasoning are
      `core/AGENTS.md`'s.
- [ ] 4. Register + select: extend the exact line in
       `src/zourcewell/core/registry.py`
        ```python
        _BUILTIN_MODULES = ("zourcewell.providers.deepseek", "zourcewell.providers.qwen",
                            "zourcewell.providers.opencode")
        ```
        to e.g.
         ```python
         _BUILTIN_MODULES = ("zourcewell.providers.deepseek", "zourcewell.providers.qwen",
                             "zourcewell.providers.opencode", "zourcewell.providers.<name>")
         ```
       then pick the provider in the TUI provider menu
       (`Settings.provider` / `--provider`). Local providers
       (`requires_accounts = False`, e.g. opencode) need no accounts:
       routes skip the pool slot and preflight passes with zero accounts.

### Setting `requires_accounts` is a one-liner, not a code change

Every surface that asks "may I tell this user to go add a credential?" reads
that single flag through `core.pools`, so a new provider inherits the correct
behaviour in the TUI, the chat REPL, multi-chat and preflight without any of
them being touched:

| helper | meaning |
|---|---|
| `pools.providers_requiring_accounts(settings)` | active ids that need a pool slot + token |
| `pools.any_provider_needs_accounts(settings)` | ≥1 does → gate on accounts |
| `pools.only_accountless_providers(settings)` | active, and none needs one → serve with an empty pool |

Leave it at the `Provider` default (`True`) for any web upstream. Set it
`False` only for a genuinely local one. Always set a real `bool`: the lookup is
`getattr(provider, "requires_accounts", True)`, so the `True` default applies
only when the attribute is *absent* — a provider that sets it to `None` reads
as accountless, which is the dangerous direction (it unblocks a web
provider's account prompt). `tests/test_provider_account_rule.py` pins this,
including a source-level check that the UI surfaces delegate instead of
re-deriving the rule.

Only finished providers are in `_BUILTIN_MODULES`: a stub must not
auto-register until it is implemented and its line is added.

## Reuse vs per-provider

| Piece | Where | Rule |
|---|---|---|
| protocol / client / SSE parser | `providers/<name>/` | Per-provider: every web API differs. |
| PoW / WAF warmup | `providers/<name>/` | Per-provider: only DeepSeek has `DeepSeekHashV1`. Qwen has a WAF but no PoW. |
| auth / login / ban probe | `providers/<name>/` | Per-provider: token location + login flow differ. Qwen's is a direct-API POST with a SHA-256-hex password; DeepSeek's needs a real Chromium-family CDP session for Shumei. |
| completion engine | `core/engine.py` | Reuse: `drive_retries`, `run_with_failover`, `complete_with_failover_sync`, `drain_attempt`, `slot_releaser` (its give-up is published through `consume_slot_abandoned`, and every release site routes through `run_with_failover`'s `_release_slot`), `ban_403_for_item`, `switch_current_after_ban_failover`, `LiveEmitTracker` / `live_retry_predicate`, the `TRANSIENT_OVERLOAD_MARKERS` / `RATE_LIMIT_MARKERS` / `UPSTREAM_INVALID_MARKERS` tuples, terminal raisers, `drain_stream_queue`, and the whole `_stream_completion` skeleton via `stream_producer` / `new_producer_context` / `consume_stream`. Providers keep only `classify_error` (composing its own extras onto the neutral tuples), session/chat setup, PoW solve, WAF warmup body, the invalid-token terminal, and their own timeout/grace globals. Do NOT copy `completion_service.py` for a new provider — write a thin adapter over `core.engine` and keep the same monkeypatch names (`_attempt`, `_run_completion_core`, `_complete_with_failover_sync`, `_stream_completion`). |
| prompt + tool parsing | `core.tools` | Reuse: `render_prompt`, `parse_tool_calls`. |
| SSE drain helpers | `core.transport` | Reuse. |
| OpenAI response shim | `core.openai._openai_response` | Reuse: tool parsing resolves via `zourcewell.server.parse_tool_calls` at call time. |
| token normalization | `core.tokens.extract_token` | Reuse: generic JSON/raw shapes. |
| account pool | `zourcewell.accounts` via `core.pools.make_pool` | Reuse: one pool instance per provider. |
| concurrency / executors | `server_concurrency.py` | Reuse (shared executors). |
| HTTP routes | `server_routes.py` | Reuse: provider-agnostic endpoints. |

## Multi-provider mode semantics

- `Settings.providers` with 2+ ids selects multi mode; otherwise single
  `Settings.provider` (default `deepseek`).
- `/v1/models` serves the union of each active provider's catalogue.
- Each provider gets its own pool (`make_pool`); accounts bind via
  `AccountConfig.provider`.
- Failover stays within provider: ban/auth cooldowns move to the next account
  of the same provider, never to another upstream.
- Multi mode keeps each provider on independent auto-failover (no shared
  CURRENT) until per-provider current support exists (`core.pools.current_for`).

## Back-compat notes

- Import the moved modules from where they live, not from a shim:
  `providers.deepseek.completion_service`, `providers.deepseek.client`,
  `core.openai`, `core.tools`. The old root paths (`deepseaport.client`,
  `deepseaport.pow`, `deepseaport.tools_support`, ...) are gone.
- `zourcewell.server` is thin wiring only: `create_app`, `MODELS`,
  `active_models`, `configure_logging`. Three names stay reachable as
  attributes of it on purpose — `_pool`, `_openai_response`,
  `parse_tool_calls` — because `server_routes` resolves them through this
  module so patching one covers every pool/response/parse entry point.
- Monkeypatch seams elsewhere: the engine and the completion services resolve
  `_attempt` / `_solve` / `_delete_session_bg` / `_run_completion_core` at
  call time on `providers.deepseek.completion_service`.
- The back-compat seams are covered by tests, but the suite is **not** frozen:
  tests change with the code. Treat any "N test files" figure in
  this file as a snapshot, not a contract, and re-derive it
  (`git ls-files -- tests/` + `Get-ChildItem tests -Filter test_*.py`) instead
  of trusting a remembered number.
- Where the provider contracts are pinned: the registered gate in
  `tests/test_templates.py` (qwen registered, no template providers);
  the cross-provider invariants from this file in
  `tests/test_provider_parity_gaps.py` (403 mapping, usage count) and
  `tests/test_multi_provider.py`; per-provider coverage in
  `tests/test_qwen_provider_gaps.py`,
   `tests/test_opencode_provider.py`,
  `tests/test_opencode_managed.py`, `tests/test_provider_dispatch.py`,
  `tests/test_provider_menu.py`.
