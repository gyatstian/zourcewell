# opencode provider — reversing notes

Local CLI provider: deepseaport talks to a user-managed `opencode serve`
instance on localhost and exposes it as OpenAI-compatible `/v1/*`.

## Serve API (v1, verified via OpenCode-Wrap + opencode docs)

- Base: `http://127.0.0.1:4096` (default serve port; Wrap uses 4100 to
  avoid TUI collision). Env: `DEEPSEAPORT_OPENCODE_BASE`.
- Auth: `OPENCODE_SERVER_PASSWORD` HTTP basic (user `opencode`); default
  localhost has none. Env: `DEEPSEAPORT_OPENCODE_USERNAME/PASSWORD`.
- `GET /global/health` -> `{healthy:true, version}` (used by warmup).
- `POST /session {}` -> `{id,...}` (per-request session, stateless).
- `POST /session/:id/message` waits for the agent loop and returns
  `{info:{modelID,providerID,tokens:{input,output,...},finish}, parts:[...]}`.
  Only `text` parts become content, `reasoning` parts become thinking.
- `DELETE /session/:id` best-effort cleanup (respects
  `auto_delete_session` only indirectly — cleanup always runs; the flag is
  web-provider-scoped).
- Reasoning: the free models emit `reasoning` parts by default (observed
  live, surfaced as OpenAI `reasoning_content`), so all catalogue specs
  are `thinking=True`. Effort control is the per-message `variant`
  (`none/minimal/low/medium/high/xhigh`); `_prepare` takes it from the
  OpenAI body (`reasoning_effort` string or `reasoning.effort`), the
  completion path forwards it, and unknown variants fall back to one
  variant-less resend instead of failing the turn.
- `GET /config/providers` would list provider/model ids, but **nothing in
  `src/` calls it any more** — the only accessor (`client.list_provider_models`)
  was diagnostics-only with zero callers and has been deleted, so there is no
  live consumer of the endpoint. Model ids come from the static catalogue plus
  passthrough (see "Model ids" below), not from this route.
- `GET /event` SSE bus is **used, and it is the live streaming path**.
  `client.stream_message` (`client.py:369-457`) subscribes to
  `GET /event?directory=` *first* — before `prompt_async` — so no early delta
  is missed, then tails it: `message.part.delta` text/reasoning deltas are
  forwarded through callbacks as they arrive, `message.updated` carries the
  `finish`/`error`, and `session.idle` ends the tail.
  `completion_service._stream_completion` picks that up when
  `stream_mode: live`: the deltas go straight out as OpenAI SSE chunks
  (thinking always; content only when the client sent no `tools`, otherwise a
  `tool_calls` reply would also leak as user-visible text). When the tail
  completes, the authoritative result is fetched from
  `GET /session/:id/message` — a partial tail is never returned as the result.
  If the SSE tail cannot start, it falls back to the blocking `POST` whose
  result is fanned out as chunks; that fallback is the same code path as
  `stream_mode: buffered`.

## Model ids

Canonical shape `providerID/modelID` split at the FIRST slash (model part
may contain slashes). Bare ids ride provider `opencode` (see
`protocol.split_model`). The static catalogue holds 8 entries: seven named
`opencode/*-free` plus `opencode/big-pickle`, which is **not** a `-free` id
(it is served and listed all the same — 7 + 1, not "8 free"). All eight were
live-tested session -> message -> `finish: stop`; the set matches
`https://opencode.ai/zen/v1/models`; `model_spec` accepts any
other id containing a slash as passthrough (user-configured proxy
providers on the same serve instance) so those work in chat too, while
`/v1/models` serves the verified set. Bare catalogue-unknown ids stay
unknown (typo guard — the historical unknown-model 404 contract holds).

## Design deltas vs web2api providers

- No accounts/tokens/pool slot: `requires_accounts = False`. Routes skip
  acquisition; `tui._preflight_reason` passes when only local providers
  are active; `cli cmd_chat` bypasses the pool.
- No PoW/WAF/ban/login: `probe_ban` is `(False, None)`, `login` raises,
  `warmup` is a log-only health check.
- No tool-format injection: `_prepare` already renders deepseaport tool
  prompts into `prep["prompt"]`; the single text part carries it, and
  `core.openai`/`core.engine` parse the reply with the shared
  `parse_tool_calls` (no Wrap-style `tool_call` fences).
- Managed isolated serve (see `managed_serve.py`): first use boots a
  dedicated `opencode serve` child on a free localhost port with child-only
  env (`XDG_CONFIG_HOME` -> empty dir, `OPENCODE_CONFIG_DIR` -> generated
  dir with the `strip-system.js` plugin, `OPENCODE_CONFIG_CONTENT` ->
  `deepseaport-passthrough` minimal agent + empty instructions,
   `OPENCODE_DISABLE_PROJECT_CONFIG/CLAUDE_CODE`, ephemeral basic-auth
   password). Before those keys are set, the child's env drops **every**
   `DEEPSEAPORT_*` / `DEEPSEEK_*` / `OPENCODE_*` key (prefix strip in
   `_child_env`), so neither the parent's `OPENCODE_CONFIG` path nor the
   operator's API keys / account tokens can reach a process the model can be
   talked into reading. The agent permission map allows `read` and sets
   `bash`/`edit`/`webfetch`/`external_directory` to `deny`. That map alone is
   not enough: the **per-message `tools` map overrides it**, so
   `protocol.NATIVE_TOOLS_MINIMAL` declares 14 native keys and grants exactly
   one, `read` — `bash` is declared with the value `false`. The free-tier
   gatekeeper needs the key *declared*, not granted, so a declared `bash` costs
   nothing at the gate and must not hand the model shell access. Both layers
   deny `bash`; neither one suffices alone. The user's
   `~/.config/opencode`, project files, and plain `opencode` runs stay
   intact. Extra workers reuse it via the `serve.json` port file.
   Opt-outs: `DEEPSEAPORT_OPENCODE_BASE` (use your own serve as-is),
   `DEEPSEAPORT_OPENCODE_MANAGED=0` (disable the boot).
- Control-plane identity: every serve request carries `User-Agent`
  (`opencode/<cli version>`), `x-opencode-client` (default `cli`),
  `x-opencode-project` (default `global`), and `X-Session-ID` (the
  serve-assigned session id). Free-tier gatekeeper refusals surface as
  terminal `OPENCODE_FREE_TIER_DENIED` (no retry) with an upgrade hint.
- Env: `DEEPSEAPORT_OPENCODE_BASE/AGENT/TIMEOUT/USERNAME/PASSWORD/MANAGED`.
  Free models need no login (`opencode auth login` only for key-gated
  providers). Serve-down errors say so (502 + hint).

## References

- Shim this is based on: https://dev.to/lynkr/how-i-built-a-free-openai-compatible-api-on-top-of-opencode-67a
  (`https://github.com/Fast-Editor/OpenCode-Wrap`, `server.js`).
- Docs: https://opencode.ai/docs/server/, https://opencode.ai/docs/sdk/
- Types: https://github.com/anomalyco/opencode/blob/dev/packages/sdk/js/src/gen/types.gen.ts
- Zen model list: https://opencode.ai/zen/v1/models
