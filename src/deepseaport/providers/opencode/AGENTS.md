# providers/opencode

## Purpose

A locally running `opencode serve` instance (v2 HttpApi, opencode 2.x) exposed
through the same API. Not a web upstream: no accounts, no tokens, no browser,
no WAF, no warmup.

## Ownership

- `completion_service.py`: a thin adapter over `core/engine.py`.
- `managed_serve.py`: boots and reaps a dedicated, isolated `opencode serve` subprocess.
- `client.py`: prompt-then-poll session lifecycle over plain HTTP.
- `protocol.py`: base URL, payloads, model splitting, error-tag matchers.
- `models.py`: live catalogue from `GET /api/model` plus the offline fallback.
- `provider.py`: the `Provider` implementation, delegating to `completion_service`.
- `README.md`: the long-form notes for the pi and opencode setup.

## Local Contracts

- `requires_accounts = False` and `needs_bridge = False`. Free models work with zero CLI auth; anything needing keys is authenticated inside the CLI itself.
- Since v2 a serve instance ALWAYS requires HTTP basic auth (it generates a password at boot and prints it, unless `OPENCODE_SERVER_PASSWORD` pins one). The managed child sets that env and hands the same value to the client; for an operator's own serve, mirror its password in `DEEPSEAPORT_OPENCODE_PASSWORD`. A 401 is terminal and says so.
- Every route is under `/api`; the v1 paths are gone, so a v1-shaped call comes back as a missing-route 404. The client is stdlib `urllib` against localhost HTTP only, in `health` (`GET /api/info`) to `create_session` (`POST /api/session`) to `send_message` to best-effort `delete_session` order, with `abort_session` (`POST /api/session/:id/interrupt`) on client disconnect.
- A turn is prompt-then-poll, not a blocking POST. `POST /api/session/:id/prompt` carries `{text}` and returns AT ONCE with the created user item while the agent loop runs in the background, so `send_message` polls `GET /api/session/:id/message` until the newest assistant item AFTER this turn's user item carries `time.completed`. Only items after the user id count, so the variant-less resend can never read the previous turn's answer or its error.
- Model, reasoning `variant` and agent are SESSION-SCOPED on v2, so every turn POSTs `/model {model:{id,providerID,variant?}}` and `/agent {agent}` before the prompt. Without that select a session born from an empty create body answers from whatever model it happens to carry, silently.
- The working directory travels as `location.directory` in the create body (`DEEPSEAPORT_OPENCODE_WORKDIR` > process cwd). v2 has no `?directory=` query, so `protocol.session_query()` is kept as a compat shim that always returns "".
- Success is enveloped `{"data": ...}`; failures are a flat `{"_tag", "message"}`. Transport and upstream failures surface as `RuntimeError` with a short code prefix (`OPENCODE_SERVE_DOWN`, `OPENCODE_HTTP_<status>`, `OPENCODE_UPSTREAM`, `OPENCODE_EMPTY`, `OPENCODE_INVALID_SESSION`, `OPENCODE_RATE_LIMITED`, `OPENCODE_FREE_TIER_DENIED`) so the engine classifies without parsing prose. An unusable model/variant is NOT a 4xx on v2: it is accepted and fails the TURN, which is why the word "variant" has to survive into the error text — completion_service keys its variant-less resend off it.
- Streaming is live token by token off a `GET /api/event` SSE tail subscribed BEFORE the prompt: deltas are `session.text.delta` / `session.reasoning.delta`, the wait ends on `session.execution.succeeded` / `session.execution.failed`, and the answer returned is read from the session timeline, never the assembled tail.
- The prompt body carries `{text}` and nothing else: v2 deleted the message-level `tools` map, so the managed agent's permission map (`managed_serve.managed_config_content`: `read` allow, `edit`/`bash`/`webfetch`/`external_directory` deny) is the ONLY thing bounding what the model may call. The free-tier gatekeeper survives only as a terminal classification (`protocol.is_free_tier_error` -> `OPENCODE_FREE_TIER_DENIED`); the declare-the-native-keys trick is history, do not re-add it.
- The catalogue is two layers: `live_specs()` reads `GET /api/model` (thinking only when the entry declares `compatibility.reasoningField` — variants alone do not imply a thinking channel; disabled and non-`active` entries dropped) and `MODELS_SPEC` is the 8-entry offline fallback used when no serve answers. When nothing answers at the configured base it BOOTS the managed serve and reads again, so the list does not depend on which process asks: a fresh `deepseaport models` and a running server report the same ids. `ensure()` is a no-op for an operator-pinned `DEEPSEAPORT_OPENCODE_BASE` and when `DEEPSEAPORT_OPENCODE_MANAGED` is off, which is how the hermetic tier keeps subprocesses out of the suite (`tests/conftest.py` sets it). One process-wide cache holds a success for 5 minutes and a failure for 30. Default model stays `opencode/muse-spark-1.3-contributor-free`; `muse-spark-1.2-contributor-free` and `ling-3.0-flash-fin-free` are no longer served upstream and survive only in the fallback (both answer `Model unavailable` live). Any other `provider/model` on the operator's own serve still passes through.
- `reasoning_effort` (`none/minimal/low/medium/high/xhigh`, also as `reasoning.effort`) maps to the session `variant`. It is the only request-level knob the serve API has: `temperature` has no field upstream, so it stays warn-and-drop like the web providers.

## Work Guidance

- Keep subprocess reaping and the managed agent's permission map in this folder. Nothing else manages the child process, and on v2 that map is the only tool boundary there is.
- The reap has two parts, and Windows needs both. The recorded pid is `cmd.exe`, not the serve: CPython hands a `.cmd` launcher to the OS shell, so the line is `cmd.exe /c <path>\opencode.CMD serve --port N ...` and the serve is that wrapper's child. The identity gate accepts exactly that three-token prefix (never a wrapper that merely mentions opencode later, like `cmd /c start opencode serve`) and still cross-checks `--port` against the base recorded beside the pid. The kill is `taskkill /T /F`, because terminating the wrapper leaves `opencode.exe` running on its port — that leak shipped a serve per restart. POSIX keeps SIGTERM plus the kill escalation.
- The strip plugin is v2's default-exported `{id, setup(ctx)}` module registering `ctx.session.hook("context", ...)`; it mutates `event.system` IN PLACE (reassigning is a silent no-op) and must leave one entry behind. The v1 `experimental.chat.system.transform` hook object does not exist in 2.x.
- v2 normalizes the managed config keys we send: `agent` -> `agents`, `prompt` -> `system`, `bash` -> `shell`. They are accepted as written and normalized by the serve; do not "fix" them here.
- `DEEPSEAPORT_FORGIVING_TOOLCALLS` and DSML arrive as DSML text and come back as DSML `tool_calls`; the bridge returns OpenAI `tool_calls` for the harness to execute. Nothing in the request body can widen the model's tool access any more.

## Verification

- The hermetic gate is owned by the root AGENTS.md; run it before any change here.
- `python -m pytest -m live -q` is the cheap probe for this provider: no account, no browser, no WAF, just a local process. Use it to check the bridge before blaming an upstream.