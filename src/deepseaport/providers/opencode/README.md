# opencode provider — reversing notes

Local CLI provider: deepseaport talks to an `opencode serve` instance on
localhost (managed or the operator's own) and exposes it as OpenAI-compatible
`/v1/*`.

## Serve API (v2, live-verified against opencode 2.0.29)

Everything is under `/api`; the v1 routes are gone, so a v1-shaped call comes
back as a missing-route 404 (`OPENCODE_HTTP_404`, and the error text names the
`/api` requirement). Success bodies are enveloped `{"data": ...}`, failures a
flat `{"_tag": "...", "message": "..."}`.

- Base: `http://127.0.0.1:4096` (default serve port). Env: `DEEPSEAPORT_OPENCODE_BASE`.
- Auth: HTTP basic, user `opencode` (default), password
  `DEEPSEAPORT_OPENCODE_PASSWORD`. **Since v2 a serve instance always requires
  it**: the server generates a password at boot and prints it, unless
  `OPENCODE_SERVER_PASSWORD` pins one. The managed instance sets that env and
  hands the same value to the client; for your own serve, mirror its password
  in `DEEPSEAPORT_OPENCODE_PASSWORD`. A 401 is terminal and names both vars.
- `GET /api/info` -> alive. `managed_serve._probe` also requires a **JSON
  body**: an unknown GET path on a v2 serve falls back to the web-app index
  and answers 200 with HTML, so a status-only probe would happily call a v1
  instance (or any other local server) healthy.
- `POST /api/session {location:{directory}}` -> `{data:{id}}`. The model is
  **not** accepted here, and v2 can only be born with one, so the turn selects
  it instead. `location.directory` pins the worktree the agent's file tools run
  in; v2 has no `?directory=` query at all.
- `POST /api/session/:id/model {model:{id, providerID, variant?}}` and
  `POST /api/session/:id/agent {agent}` -> 204. The model, its reasoning
  `variant` and the agent are all **session-scoped** in v2, so every turn
  re-sends both before the prompt. An unusable model or variant is ACCEPTED
  here (200) and only fails the turn.
- `POST /api/session/:id/prompt {text}` -> `{data:{id}}` **at once**: v2 runs
  the agent loop asynchronously, so there is no blocking message endpoint any
  more. The returned id is this turn's user item; the reply is polled.
- `GET /api/session/:id/message?limit=&order=desc` -> `{data:[items], cursor}`,
  reversed back into chronological order by the client. It is a MIXED
  timeline: user `{id, time, text, type:"user"}`; assistant
  `{id, time:{created,streamed,completed}, type:"assistant", agent, model,
  content:[{type:"reasoning",text} | {type:"text",text}], finish, rawFinish,
  cost, tokens:{input,output,reasoning,cache}}`; and idle markers
  `{type:"idle", outcome}`. The authoritative final is the newest assistant
  item carrying `time.completed`, counted only among items AFTER this turn's
  user item — that offset is what keeps a resend on the same session from
  reading the previous turn.
- `POST /api/session/:id/interrupt {}` -> abort (used on client disconnect).
- `DELETE /api/session/:id` -> 204, best-effort cleanup.
- `GET /api/event` -> SSE `data: {json}` frames. `session.text.delta` and
  `session.reasoning.delta` both carry `{sessionID, delta}`; the turn ends on
  `session.execution.succeeded` / `session.execution.failed` (a `session.error`
  frame is treated as a failure too).
- `GET /api/model` -> the live catalogue. This is what `models.live_specs`
  calls; there is no other model-discovery route in use.

## Errors

Client failures are `RuntimeError` carrying a short code prefix, which is the
contract the engine's `classify_error` reads instead of parsing prose:
`OPENCODE_SERVE_DOWN` (connection-level, plus a start hint),
`OPENCODE_HTTP_<status>` (401 names the auth vars, 404 splits "unknown session"
into `OPENCODE_INVALID_SESSION` from "route missing" via
`protocol.is_session_error`, 429 is `OPENCODE_RATE_LIMITED`), then turn-level
`OPENCODE_UPSTREAM`, `OPENCODE_EMPTY`, and `OPENCODE_FREE_TIER_DENIED` for the
upstream free-tier gatekeeper.

A bad model or variant is no longer a 4xx. It fails the turn and the timeline
carries `Model/Variant unavailable for opencode/<id>: <why>`, which is why the
serve's own words (including "Variant") are kept in the error text:
`completion_service` keys its one variant-less resend off that word. Unknown
variants therefore cost one resend, not a failed request.

The free-tier gatekeeper itself is only a terminal classification now
(`protocol.is_free_tier_error` -> `OPENCODE_FREE_TIER_DENIED`, no retry, with
an upgrade hint). The message-level tool declaration that used to satisfy it
is gone with the rest of the message body.

## Native tools

v2 has **no message-level tool map at all** — the prompt body carries
`{"text": ...}` and nothing else, and the declare-14-native-keys / grant-one-
`read` machinery that used to ride it is gone with the rest of the body. What
bounds tool access is the managed agent's permission map in
`managed_serve.managed_config_content`: `read` / `glob` / `grep` / `subagent`
/ `edit` / `bash` / `webfetch` / `external_directory` all deny, at both the
global and the agent level. `read` (and `grep`, which prints matching file
content) is denied because sessions are rooted at the caller's directory,
where `config.json` — plaintext passwords and bearer tokens — can sit; on v2
write/patch normalize to `edit` and `task` to `subagent`, so those aliases are
covered by the same keys. That map is now the ONLY layer, so it is where any
future grant must be made deliberately. v2 normalizes the keys we send
(`agent` -> `agents`, `prompt` -> `system`, `bash` -> `shell`); the serve
accepts the un-normalized spellings.

DSML harness tools are unaffected: pi's tools arrive as DSML text in the
prompt and the reply comes back as DSML `tool_calls`, parsed by
`core.tools` and returned as OpenAI `tool_calls`. The managed agent prompt
tells the model that text output is its only channel.

The strip plugin (`managed_serve.STRIP_PLUGIN_JS`) is rewritten for the v2
plugin API: a default-exported `{id, setup(ctx)}` module registering
`ctx.session.hook("context", event => ...)` and mutating `event.system` in
place (reassignment is a silent no-op, and one entry must remain or the
original upstream is restored). The v1 `experimental.chat.system.transform`
hook object no longer exists. Measured live on the managed config: the agent
loop's input is 2122 tokens with the plugin and 3701 without it.

The reap is Windows-shaped, twice over. `shutil.which("opencode")` on an npm
install returns the `opencode.CMD` shim, and CPython hands a batch file to the
OS shell, so the pid recorded in `serve.json` belongs to
`C:\WINDOWS\system32\cmd.exe /c ...\opencode.CMD serve --port N ...` and the
serve itself is that wrapper's child. Two consequences, both fixed and pinned
in `tests/test_managed_serve_hardening.py`: the identity gate reads the
launcher as the third token after `cmd /c` (so `cmd /c start opencode serve`
still fails) and requires `--port` to agree with the base recorded beside the
pid whenever that base carries one; and the kill is `taskkill /T /F`, because
`terminate()` kills the wrapper and leaves `opencode.exe` holding the port.
Verified live on Windows: the recorded pid now passes the gate, the tree kill
frees the port, and two boot/shutdown cycles leave nothing behind.

The pid a reap starts from is stale by default — the atexit kill retires the
process, not `serve.json` — so `_kill_pid` asks the OS whether the pid exists
at all before reading anything about it, and a pid that is gone is skipped
silently. Without that check the command-line reader could not tell "gone"
from "unreadable", and every clean restart spent ~0.5 s on a PowerShell spawn
and then logged "does not look like 'opencode serve'" about a process that no
longer existed. Liveness is ctypes `OpenProcess(SYNCHRONIZE)` plus a zero-wait
on Windows — `os.kill` on a Windows pid is `TerminateProcess`, so the check
would be the kill — and `os.kill(pid, 0)` on POSIX; "cannot tell" falls
through to the identity gate. A pid that is provably gone also narrows the
reuse-probe ladder to one attempt, because the retries exist for a serve that
is alive but too busy to answer.

## Model ids

Two layers. `live_specs()` reads `GET /api/model` and serves ids as
`providerID/modelID` — that is the truth, and it turns over weekly.
`MODELS_SPEC` is the hand-verified fallback list; exact ids and current
availability live in `models.py`. `models()` (and `model_spec()`) serve the
live set whenever it is non-empty; the fallback is a fallback, not a floor, so
the two are never merged. When nothing answers at the
configured base, `live_specs()` boots the managed serve and reads again: the
list must not depend on which process asks, or a fresh `deepseaport models`
answers with the hand-written fallback while the running server answers with
the live catalogue. `ensure()` respects both opt-outs (an operator-pinned
`DEEPSEAPORT_OPENCODE_BASE`, and `DEEPSEAPORT_OPENCODE_MANAGED=0`, which is
what `tests/conftest.py` sets so the hermetic tier never spawns it). Cached
process-wide under one lock: a success for 5 minutes, a failure for 30, so a
serve that is still booting is retried soon rather than in five minutes. The
read that follows a boot gets a 30 s budget because a freshly booted serve
builds its model registry on that first call.

A menu goes through `cached_specs()` (i.e. `Provider.models_hint()`): the last
read this process made, else the fallback, with no I/O and no lock. A menu must
not pay the dead-base timeout, the boot, or a wait on a read that holds the lock
across both -- the lock is held ACROSS the fetch and the boot on purpose, so a
hint that took it would inherit the whole stall.

`thinking` is derived from `compatibility.reasoningField` alone: that field is
where the serve puts a model's thinking. The free tier declared one for every
model except `muse-spark-1.3-contributor-free`, which reports reasoning tokens
and delivers no thinking text (it carries five variants and exposes none), so
variants are not evidence of a thinking channel.

Canonical shape is `providerID/modelID`, split at the FIRST slash (the model
part may contain slashes). Bare ids ride provider `opencode` (see
`protocol.split_model`); a blank model id falls back to `models.DEFAULT_MODEL`
(`opencode/muse-spark-1.3-contributor-free`), which is also the first entry of
the fallback list. `model_spec` accepts
any other id containing a slash as passthrough (user-configured proxy
providers on the same serve instance) so those work in chat too; bare
catalogue-unknown ids stay unknown (typo guard — the unknown-model 404
contract holds).

Disabled and non-`active` live entries are dropped, because a model the
operator cannot pick must not be advertised.

## Design deltas vs web2api providers

- No accounts/tokens/pool slot: `requires_accounts = False`. Routes skip
  acquisition; `tui._preflight_reason` passes when only local providers
  are active; `cli cmd_chat` bypasses the pool.
- No PoW/WAF/ban/login: `probe_ban` is `(False, None)`, `login` raises,
  `warmup` boots the managed serve when nothing answers (`GET /api/info`
  probe) and primes the catalogue.
- No tool-format injection: `_prepare` already renders deepseaport tool
  prompts into `prep["prompt"]`; the single text body carries it, and
  `core.openai`/`core.engine` parse the reply with the shared
  `parse_tool_calls` (no Wrap-style `tool_call` fences).
- Streaming is live token by token off `GET /api/event`, subscribed first, and
  the answer that is returned is the timeline's completed assistant item —
  never the assembled tail, so a partial stream cannot be served as the
  result. `completion_service._stream_completion` picks that up when
  `stream_mode: live`: deltas go straight out as OpenAI SSE chunks (thinking
  always; content only when the client sent no `tools`, otherwise a
  `tool_calls` reply would also leak as user-visible text). If the SSE tail
  cannot start at all, it falls back to the blocking prompt-then-poll turn
  fanned out as chunks — the same code path as `stream_mode: buffered`.
- Managed isolated serve (see `managed_serve.py`): first use boots a
   dedicated `opencode serve` child on a free localhost port with child-only
   env (`XDG_CONFIG_HOME` -> empty dir, `OPENCODE_CONFIG_DIR` -> generated
   dir with the `strip-system.js` plugin, `OPENCODE_CONFIG_CONTENT` ->
   `deepseaport-passthrough` minimal agent + empty instructions,
   `OPENCODE_DISABLE_PROJECT_CONFIG`/`CLAUDE_CODE`, per-boot
   `OPENCODE_SERVER_PASSWORD` shared with the client). Before those keys are
   set, the child's env drops **every** `DEEPSEAPORT_*` / `DEEPSEEK_*` /
   `OPENCODE_*` key plus the well-known third-party credential names
   (prefix strip + explicit deny list in `_child_env`), so neither the parent's
   `OPENCODE_CONFIG` path nor the operator's API keys / account tokens can
   reach a process the model can be talked into reading. The password is
   ephemeral per boot; on POSIX `serve.json` and every managed dir are
   owner-only (0700/0600), and on Windows the password is stored
   DPAPI-wrapped for the current user (`password_dpapi`) instead of
   cleartext, so the one-shot ACL warning fires only while a cleartext
   `password` is still on disk (a real exposure left by an older build or a
   DPAPI-less machine, not a per-boot ritual) and a reuse of such a file
   upgrades it through the current writer.
   The user's `~/.config/opencode`, project files, and plain `opencode` runs
   stay intact. Extra workers reuse it via the `serve.json` port file.
   Opt-outs: `DEEPSEAPORT_OPENCODE_BASE` (use your own serve as-is),
   `DEEPSEAPORT_OPENCODE_MANAGED=0` (disable the boot).
- Control-plane identity: every serve request carries `User-Agent`
  (`opencode/<cli version>`), `x-opencode-client` (default `cli`),
  `x-opencode-project` (default `global`), and `X-Session-ID` (the
  serve-assigned session id).
- Env: `DEEPSEAPORT_OPENCODE_BASE/AGENT/TIMEOUT/USERNAME/PASSWORD/MANAGED`
  (plus `CLIENT/PROJECT/USER_AGENT/WORKDIR`). Free models need no login
  (`opencode auth login` only for key-gated providers). Serve-down errors say
  so (502 + hint).

## References

- Origin of this provider (v1-era shim): https://dev.to/lynkr/how-i-built-a-free-openai-compatible-api-on-top-of-opencode-67a
  (`https://github.com/Fast-Editor/OpenCode-Wrap`, `server.js`).
- Docs: https://opencode.ai/docs/server/, https://opencode.ai/docs/sdk/
- Zen model list (the offline fallback set): https://opencode.ai/zen/v1/models