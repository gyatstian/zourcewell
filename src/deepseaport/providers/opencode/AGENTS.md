# providers/opencode

## Purpose

A locally running `opencode serve` instance exposed through the same API. Not a
web upstream: no accounts, no tokens, no browser, no WAF, no warmup.

## Ownership

- `completion_service.py`: a thin adapter over `core/engine.py`.
- `managed_serve.py`: boots and reaps a dedicated, isolated `opencode serve` subprocess.
- `client.py`: synchronous session lifecycle over plain HTTP.
- `protocol.py`: base URL, payloads, model splitting.
- `models.py`: the catalogue as `ModelSpec` entries.
- `provider.py`: the `Provider` implementation, delegating to `completion_service`.
- `README.md`: the long-form notes for the pi and opencode setup.

## Local Contracts

- `requires_accounts = False` and `needs_bridge = False`. Free models work with zero CLI auth; anything needing keys is authenticated inside the CLI itself.
- The client is stdlib `urllib` against localhost HTTP only, in `health` to `create_session` to `send_message` to best-effort `delete_session` order. `send_message` blocks, because serve waits for the agent loop.
- Transport and upstream failures surface as `RuntimeError` with a short code prefix, `OPENCODE_SERVE_DOWN` or `OPENCODE_HTTP_<status>`, so the engine can classify without parsing prose.
- Retries cover transient overload and empty turns only (`classify_error` maps both onto `RetryKind.OVERLOAD`), with a fresh session each. Serve-down, auth and validation errors are terminal and carry an actionable message. The `drive_retries` table therefore has one row; the stale-session recreate sits OUTSIDE the budget, before the table runs. The row goes through the local `_retryable(kind)` predicate like every other provider's, so it does not retry once a live delta has reached the client (the rule is `core/AGENTS.md`'s).
- Streaming is live token by token via `prompt_async` plus a `GET /event` SSE tail, not buffered until the end.
- `managed_serve.py` boots an isolated child `opencode serve` on first use, tuned for raw API passthrough with a minimal agent, deny-all permissions and a system-strip plugin. The child's environment is its own (empty `XDG_CONFIG_HOME`, `OPENCODE_CONFIG_DIR`), so the operator's `~/.config/opencode` is never touched.
- `DEEPSEAPORT_OPENCODE_BASE` uses the operator's own serve instance as-is instead of booting a managed one.
- `DEEPSEAPORT_OPENCODE_WORKDIR` pins the session directory (`?directory=`) so a harness running elsewhere gets file tools in the project it expects.
- The per-message `tools` map overrides the managed agent's permission map. A free-tier tool turn must declare the native keys and grant exactly one. `bash` is declared with the value `false` on purpose: the free-tier gate needs the key present, not granted. Both layers have to agree, or a granted `bash` hands the model a shell on every tool turn.
- Eight static models, `opencode/muse-spark-1.3-contributor-free` first. Seven carry a `-free` suffix and `opencode/big-pickle` does not; it is still served and still listed. Any `provider/model` configured on the operator's own serve passes through.
- `reasoning_effort` (`none/minimal/low/medium/high/xhigh`, also as `reasoning.effort`) maps to the serve message `variant`. It is the only per-request parameter the serve API supports. `temperature` has no per-message field upstream, so it stays warn-and-drop like the web providers.

## Work Guidance

- Keep subprocess reaping and the free-tier permission map in this folder. Nothing else manages the child process.
- `DEEPSEAPORT_FORGIVING_TOOLCALLS` and DSML arrive as DSML text and come back as DSML `tool_calls`; the bridge returns OpenAI `tool_calls` for the harness to execute.
- Plain chat sends no native `tools` key at all. A tool turn has to send one.

## Verification

- The hermetic gate is owned by the root AGENTS.md; run it before any change here.
- `python -m pytest -m live -q` is the cheap probe for this provider: no account, no browser, no WAF, just a local process. Use it to check the bridge before blaming an upstream.