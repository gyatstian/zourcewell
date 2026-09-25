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
- `GET /config/providers` lists provider/model ids (diagnostics only;
  chat passes model ids through, see below).
- `GET /event` SSE bus (`message.part.updated` deltas) exists but is NOT
  used: streaming is buffered (result fanned out as SSE at the end).
  True tailing is a follow-up.

## Model ids

Canonical shape `providerID/modelID` split at the FIRST slash (model part
may contain slashes). Bare ids ride provider `opencode` (see
`protocol.split_model`). Static catalogue holds the 8 verified-free
`opencode/*` models (all live-tested session -> message -> `finish: stop`;
matches `https://opencode.ai/zen/v1/models`); `model_spec` accepts any
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
  `OPENCODE_DISABLE_PROJECT_CONFIG/CLAUDE_CODE`,
  `OPENCODE_CONFIG` dropped, ephemeral basic-auth password). The agent
  allows `read`+`bash` (declared, not used: the free-tier gatekeeper 403s
  tool-less upstream requests) and denies the rest. The user's
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
