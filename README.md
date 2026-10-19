## Setup in simple terms
- install [obscura](https://github.com/h4ckf0r0day/obscura/releases) (no-render stealth is the one i use)
- install the newest [release](https://github.com/gyatstian/deepseaport/releases) (TIP: CLICK THE BLUE TEXT)
- place obscura and deepseaport next to eachother or put obscura in PATH
- open deepseaport, add account(s) and then start server.
- *optionally play around with settings or/and config.json!*
- base url: http://127.0.0.1:5001/v1

- app needs https://aka.ms/vs/17/release/vc_redist.x64.exe beacuse of obscura

# deepseaport — DeepSeek + Qwen web2api (multi-provider)

Unofficial, research-only bridge: exposes web chat as an
OpenAI-compatible API (`/v1/chat/completions`, streaming + `tools`).
Supported upstreams: DeepSeek (`chat.deepseek.com`, Obscura WAF +
real-browser login) and Qwen (`chat.qwen.ai`, direct-API login + Aliyun
WAF jar). Provider is selected in the TUI provider gate or via
`--provider`/`--providers` (`Settings.provider`/`providers`).

How it works:

- **Obscura** (`h4ckf0r0day/obscura` headless browser) owns a persistent
  profile that solves the AWS WAF JavaScript challenge and holds the
  `aws-waf-token` cookie. All DeepSeek HTTP calls replay that cookie jar,
  with a browser User-Agent captured from the same engine.
- **Auto-token/login (DeepSeek)** uses a real installed Helium/Chrome/Edge/Chromium via
  CDP (`browser_bin: "auto"`), because DeepSeek's Shumei device-id SDK does
  not initialize in the Obscura no-render engine. Obscura remains the WAF and
  hot-path engine.
- **Login (Qwen)** is a direct `POST /api/v1/auths/signin` (no browser, no
  captcha); the JWT rides the `token` cookie on every hot-path request (see
  `src/deepseaport/providers/qwen/README.md`).
- Hot path is direct HTTP (`curl_cffi`, Chrome TLS impersonation).
  DeepSeek: login → create session → PoW challenge (`DeepSeekHashV1` wasm) →
  `POST /api/v0/chat/completion` (SSE `p`/`v` events) → delete session.
  Qwen: create chat → `POST /api/v2/chat/completions?chat_id=<id>` →
  best-effort delete chat (no PoW; Aliyun 200+HTML challenge vs AWS 403).
- No DOM driving per token: fast enough for agentic loops.

Agentic-harness support:

- Full OpenAI `tools` schema in, `tool_calls` out (stream + non-stream),
  parallel calls, stable `call_xxx` ids, `finish_reason: "tool_calls"`.
- `role: tool` results are folded back into the web prompt, so multi-step
  tool loops work against a model with no native function calling.
- Parser bounds: `tool_args_max_chars` / `DEEPSEAPORT_MAX_ARGS_CHARS` (default
  200000) caps the arguments of one call; `forgiving_toolcalls` /
  `DEEPSEAPORT_FORGIVING_TOOLCALLS` (default off) enables the markup-repair
  bucket; `DEEPSEAPORT_MAX_BARE_OBJECTS` (default 40) bounds how many JSON
  objects the parser scans for a bare name+arguments call, and the
  total-call scan of the forgiving bucket reuses the same number.
- One in-flight stream per account (DeepSeek limitation) enforced with
  per-account locks + rotation; auto session cleanup; PoW/WAF retry once.

## Quickstart
**Setup for both:**
Obscura binary (`h4ckf0r0day/obscura`, i personally use no-render stealth): drop
`obscura.exe` next to `config.json` (or under `bin/`, `tools/`,
`vendor/`, `obscura-*/`), or set `obscura_bin` / `OBSCURA_BIN`. path also works.

**A. exe:**
- just run the .exe from the releases

**B. python:**
```powershell
pip install -e .
python -m deepseaport serve                    # :5001
```



```powershell
curl http://127.0.0.1:5001/v1/models
curl -N http://127.0.0.1:5001/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"model":"deepseek-flash","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

## Showcase
**prompt used (inside opencode with a short but sharp agent):** Design and create a very creative, elaborate, and detailed voxel art scene of a pagoda in a beautiful garden with trees, including some cherry blossoms. Make the scene impressive and varied and use colorful voxels. Use whatever libraries to get this done but make sure I can paste it all into a single HTML file and open it in Chrome.<br>

<div align="center">
  <img src="temple.png" alt="temple" width="500">
</div>

**my opnion:** it looks better than v4 definetly as it has more details but looks much worse than what an actual flash v4.1 can create. it seemed like he just rushed to create it (didn't think about correcting any mistakes after finishing or adding anything further).

## Config

`config.json` (or env: `.env.example` is text to copy into a shell export — nothing reads a `.env` file, and an empty var counts as unset):

```json
{
  "keys": ["sk-local-dev"],
  "accounts": [{ "email": "you@x.com", "mobile": "", "password": "...", "token": "",
                  "token_expires_at": 0.0, "provider": "deepseek",
                  "banned": false, "banned_until": 0.0 }],
  "active_account": "",
  "obscura_bin": "",
  "obscura_profile": "",
  "browser_bin": "auto",
  "browser_headless": false,
  "port": 5001,
  "listen": false,
  "chat_model": "deepseek-flash",
  "provider": "deepseek",
  "providers": []
}
```

`"keys": ["sk-local-dev"]` is a **public example key**. Change it before
enabling `listen` — with `listen: true` the whole API (including
`/v1/accounts` CRUD) is reachable from the network and this key is printed in
this README. See the non-loopback warning below.

Request limits (server-enforced, no config knob for the first three):

- **Body ≤ 8 MB** — a declared `Content-Length` over
  `MAX_REQUEST_BYTES` is rejected with **413** before the payload is read.
- **≤ 128 tools per request** — a longer `tools` array is **400**; counted
  only when tools are actually rendered (`enable_tools` / `tool_choice` on).
- **≤ 64 accounts** — `POST /v1/accounts` past the cap is **409** (each
  accepted call rewrites the whole credential file).
- `"keys": []` (empty or all-blank) means **auth disabled** — every endpoint,
  account CRUD included, is served without a key. That is the documented
  local-dev default; `"keys": [""]` is normalized away at load so it can
  never become this state by accident. A **non-loopback bind** (`listen: true`
  or `--host 0.0.0.0`) therefore prints a startup warning on all three serve
  paths, naming the bind it is warning about (`WARNING: serving on
  <host>:<port> (non-loopback)`), what is reachable (the whole API, `/v1/accounts`
  CRUD and unmetered chat) and how the gate reads — either the N key(s) in
  `keys` are the only gate and travel in a plain header on an unencrypted LAN,
  or `AUTH DISABLED` outright.

### `max_retries` — a knob, but clamped

`max_retries` **is** configurable (`DEEPSEAPORT_MAX_RETRIES`, `config.json`, TUI
Settings) — not one of the fixed limits above — and clamped to **0–5** at load
(a larger value warns and becomes 5), because one request costs up to
`max_retries` session + PoW + stream attempts. With the env var **unset** the
clamp is runtime-only and a hand-edited value survives a save; with it **set**,
the env value is the effective source and the first save persists the clamped
number to `config.json` — keep the two in sync or drop the env var.

`provider` (single mode) / `providers` (multi mode) select the upstream
(`Settings.provider`/`providers`, env
`DEEPSEAPORT_PROVIDER`/`DEEPSEAPORT_PROVIDERS`, or `--provider`/`--providers`;
TUI shows a provider gate first). Multi mode needs **two or more** ids in
`providers`; an empty list means "use `provider` only", and a **single** entry
is ignored — `provider: "deepseek"` + `providers: ["qwen"]` runs deepseek and
logs a warning naming the ignored id, because taking one entry would flip
`is_multi()` and switch the whole per-provider pool path on for what reads like
single-provider setup. `accounts[].provider` pins an account to
one upstream. `accounts[].mobile` is an alternate identifier;
`accounts[].token_expires_at` persists Qwen signin expiry (`0` = unknown,
never blocks).

`active_account` is the `CURRENT` selection (`""` = auto-failover over the
whole pool). `chat_model` remembers the last `/model` pick from server-with-chat
mode (env: `DEEPSEAPORT_CHAT_MODEL`). All remaining behaviour flags and their
defaults live in `config.example.json`; the TUI Settings screen is authoritative
for what is editable there, and any persisted field it does not cover is
`config.json` only. `enable_append` (default `true`, env:
`DEEPSEAPORT_ENABLE_APPEND`, TUI Settings) gates both: when `false`, stored
`append_top` / `append_bottom` texts are kept but nothing is wrapped into
requests. `append_top` / `append_bottom` (env: `DEEPSEAPORT_APPEND_TOP` /
`DEEPSEAPORT_APPEND_BOTTOM`, TUI Settings) wrap every outgoing prompt — top
text goes at the absolute top, bottom text at the absolute bottom. Both are
multi-line (`"append_top": "line1\nline2"` in `config.json`); `""` disables.
`send_dry_run_to_frontend` (default `true`, env:
`DEEPSEAPORT_SEND_DRY_RUN_TO_FRONTEND`, TUI Settings): when `true`, the Dry run
reply is a normal OpenAI-shaped response whose content is what dry run prints
(raw body + formatted prompt); when `false`, the content is a short note while
the terminal still prints everything.

`use_multiple_accounts` (default `true`, env:
`DEEPSEAPORT_USE_MULTIPLE_ACCOUNTS`): when `true` and `CURRENT` is busy
(one in-flight stream per account, enforced by per-account lock), cooling
down (banned), or auth-broken (missing/invalid token), the request fails over
to another healthy account — this is what lets parallel subagents /
multitasking share one instance. When `false`, requests stick to `CURRENT`
and queue on it instead.

`"listen": true` serves on `0.0.0.0` (LAN-visible); default `false` binds
`127.0.0.1` only. `--host` flag overrides both. A non-loopback bind prints a
warning at startup: the full API is network-reachable, the only gate is
`keys` travelling in a plain header on an unencrypted LAN (or auth is
disabled when `keys` is empty). Bind `127.0.0.1` unless you meant this.

You can skip email/password by pasting a `userToken` from a logged-in
browser (localStorage) into `accounts[].token`. Raw values and the full
`{"value":"...","__version":"0"}` JSON are both accepted; a `{"value":null}`
shape is normalized to "missing token" instead of causing a later 40003.

When `use_multiple_accounts` is true, an account with a missing/invalid token
is cooled down briefly and the request fails over to the next account that
already has a working token. Use `deepseaport login` (or the TUI auto-token
flow) to provision password-only accounts; the server does not spawn a login
browser on the request hot path.

Bans detected by the server or login page are persisted in `accounts[].banned`
and `accounts[].banned_until`, so the accounts screen keeps showing
`(BANNED: 16 September)` after restarts and even when the account has no token.
Clear that state with `POST /v1/accounts/unblock` or
`python -m deepseaport accounts unblock [identifier]`.

## Endpoints

- `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`
- `GET /v1/waf/status` — Obscura binary, profile, WAF cookie state.
  **Requires the API key** (same Bearer gate as every other `/v1/*` route):
  the payload names the absolute binary and profile paths plus the WAF cookie
  names, which is exactly the reconnaissance a LAN scanner wants. `/health`
  stays unauthenticated on purpose so existing pollers keep working.
- Accounts: `GET /v1/accounts`, `POST /v1/accounts`,
  `POST /v1/accounts/select`, `PUT /v1/accounts/{identifier}/token`,
  `DELETE /v1/accounts/{identifier}`, `POST /v1/accounts/unblock`
- `{identifier}` is an **account identifier, never a token**: the account's
  `email`, its `mobile`, or the `token[:10] + "..."` short form that
  `GET /v1/accounts` returns. A 32+ character `[A-Za-z0-9_.=-]` segment (the bearer/JWT
  alphabet) is refused with **400** on every identifier route (`{identifier}` on
  `PUT`/`DELETE`, the `identifier` body field on `select`/`unblock`), ahead of
  the API-key check, and the 400 never echoes the value back. Reason: a URL
  ends up in access logs, proxy logs and `Referer` headers, and the old 404
  detail used to hand the caller a live 64-char bearer token. The same reason
  is why the **request access log is off on all three serve paths** (plain
  serve, server-with-chat / multiple servers, dry run); uvicorn's *error* log
  is unaffected, and any path it does print is redacted and bounded.
- **No native tool calling upstream**: the web models have no function-calling
  API, so `tools` schemas are rendered into the prompt and the reply is parsed
  back out. That is why `enable_tools` must stay on for pi/opencode-style
  harnesses.

Models (union of the active providers' catalogues; `/v1/models` serves all):
DeepSeek — `deepseek-flash`, `deepseek-flash-reasoner`,
`deepseek-flash-search`, `deepseek-flash-reasoner-search` (reasoner models
return `reasoning_content` alongside `content`); Qwen — `qwen3.7-plus`
(default), `qwen3.8-max`, `qwen3.8-omni-flash`;
opencode (local CLI, no accounts) — whatever the running `opencode serve`
lists on `GET /api/model` (default `opencode/muse-spark-1.3-contributor-free`),
falling back to the hand-verified fallback list; exact ids and current
availability live in `providers/opencode/models.py`. Any `provider/model`
you configured on your own `opencode serve` passes through.

Note: OpenAI params the web upstreams cannot honor (`temperature`, `top_p`,
`max_tokens`, `stop`, `n`, penalties, `response_format`, `parallel_tool_calls`)
are accepted but ignored upstream (prompt + thinking/search/model only — tools
are serialized into the prompt, so there is nothing to throttle or to switch
off); the server logs a warning per request. `tool_choice` is honored for
`"auto"` and `"none"` (the latter disables tools for that turn), but a
**forcing** value — `"required"` or `{"type":"function",...}` — is warn-and-drop
like the rest, because the web model may answer in prose where the caller is
waiting on a `tool_call`. `stream_options.include_usage` is honored locally.
The opencode provider additionally honors `reasoning_effort`
(`none/minimal/low/medium/high/xhigh`, also as `reasoning.effort`) as the
session's reasoning `variant` — the only request-level parameter the serve API
supports (`temperature` has no field upstream, so it stays warn-and-drop like
the web providers).

## Pi harness + opencode (no accounts, tool_calls out of the box)

1. Install the `opencode` binary (on PATH) and start the bridge on the
   opencode provider: `python -m deepseaport serve --provider opencode`
   (first request auto-boots a managed isolated `opencode serve`; your own
   `~/.config/opencode` is never touched — or set
   `DEEPSEAPORT_OPENCODE_BASE` to use your own serve as-is).
2. In `config.json` keep `"enable_tools": true` (off strips the tool
   schemas pi sends, so pi never sees `tool_calls` — the server now warns
   when it does this) and `"provider": "opencode"`.
3. In pi, pick provider `deepseaport` (base `http://127.0.0.1:5001/v1`)
   and model `opencode/muse-spark-1.3-contributor-free`, then work
   normally: pi's tools arrive as DSML text, the model replies with DSML
   `tool_calls`, and the bridge returns OpenAI `tool_calls` for pi to
   execute. The agent the bridge runs denies `read`/`glob`/`grep`/`subagent`
   and `edit`/`bash`/`webfetch`/`external_directory` — pi's own tools
   ride the prompt text, not the model's native ones.
4. opencode v2 requires a basic-auth password on every serve instance. The
   managed one pins a per-boot `OPENCODE_SERVER_PASSWORD` itself, so nothing
   to do. If you point `DEEPSEAPORT_OPENCODE_BASE` at your own serve, mirror
   the password it printed at boot in `DEEPSEAPORT_OPENCODE_PASSWORD`.
5. If pi runs in a different directory than the bridge, set
   `DEEPSEAPORT_OPENCODE_WORKDIR` to the project dir so sessions pin
   (`location.directory` on session create) where pi expects file tools to run.

See [ARCHITECTURE.md](ARCHITECTURE.md) for protocol details, auth/PoW/SSE
notes, and the tool-calling design, and [PROVIDERS.md](PROVIDERS.md) for
adding an upstream.

## Building

The release `.exe` is a PyInstaller one-file build driven by
`deepseaport.spec` (it sets `version` from `pyproject.toml`, and `upx=False` —
no UPX is installed here, and PyInstaller 6 only warns when it is missing):

```powershell
pip install -e ".[build]"
pyinstaller --clean --noconfirm deepseaport.spec
python scripts/prepare_release.py --check     # audit, writes nothing
python scripts/prepare_release.py             # release/ + release/dist.zip
```

**Package the release with `scripts/prepare_release.py`, not
`Compress-Archive dist\*`.** `dist/` doubles as the frozen app's live
credential store, so a hand-made zip ships `config.json`. The script drops
build noise and secret-shaped leftovers, **refuses to build** when any JSON in
the tree carries a non-empty `email` / `mobile` / `token` / `password` /
`keys` / `api_key` / `secret` value (content scanning is JSON-only), and never
writes to `dist/` — your live `config.json` stays where the exe expects it.
`.github/workflows/release.yml` runs `prepare_release.py --check` as a **hard
gate before every upload step** on a `v*` tag. Exit 1 means the check failed
(a credential was found, or a JSON file it could not read or parse); exit 0
means the artifact is clean. Full rationale:
[ARCHITECTURE.md](ARCHITECTURE.md).

**Version.** `pyproject.toml` `[project] version` is the only place a version
is written. `deepseaport.spec` reads it at build time (so the exe's version
resource and a broken pyproject fail the build loudly instead of silently
stamping `0.0.0`) and bundles it as `_version.txt`, which `deepseaport.__version__`
reads first — then pyproject, then the installed metadata, then `0.0.0`.
`cli --version` and both FastAPI apps report that same value. Bump it there and
nowhere else.

Interpreter: CPython >= 3.11 (CI: 3.11/3.13); `.[build]` asks for
`pyinstaller>=6`, and the exact PyInstaller version is whatever the build used —
record the interpreter and PyInstaller version in the release notes of any build
you produce. One gotcha if you reach for the module instead of the script: the
package is capitalized, so `python -m pyinstaller` fails with
`No module named pyinstaller` while `python -m PyInstaller` works; the
`pyinstaller` console script is the one the command above uses.

The exe needs the **Visual C++ redistributable** on the target machine —
`https://aka.ms/vs/17/release/vc_redist.x64.exe` — which the app needs
anyway for obscura (see top of this file).

## Development

```powershell
pip install -e ".[test]"     # editable install + pytest/httpx/openai SDK
python -m pytest -q          # hermetic suite, no network
```

Two things a new contributor will hit immediately:

1. **Run `pip install -e .` from this checkout**, or the tests import a
   *different* installed copy of deepseaport and you debug code that is not
   the code in front of you. `pyproject.toml` now pins
   `pythonpath = ["src"]` under `[tool.pytest.ini_options]`, so `pytest`
   always resolves this tree — but keep the editable install anyway so
   `python -m deepseaport` runs your checkout too.
2. **`tests/conftest.py` makes the suite hermetic**: it points
   `DEEPSEAPORT_CONFIG`/`DEEPSEAPORT_DATA` at a temp dir, strips
   `DEEPSEAPORT_*`/`DEEPSEEK_*`/`OBSCURA_*` from the ambient environment, and
   raises on any non-loopback socket connect. `live` tests are **skipped**
   unless you ask for them. The network tier is opt-in:

   ```powershell
   python -m pytest -m live -q
   ```

   That tier needs a real account token in `config.json` and the `obscura`
   binary on PATH.

   **Run this periodically** — it is the only thing in the repo that can
   notice an upstream changed. `tests/test_live_smoke.py` is a small suite
   (one run per provider, seconds) that exercises the real wire formats and
   asserts the invariants a stub cannot: DeepSeek's continuation chunks
   (bare `{"v":...}` with no `p`), Qwen's `delta.phase` split, the WAF jar,
   the streamed `tool_calls` `index`, and — most importantly — that a reply
   contains the *end* of the prompt. That last one exists because a stateless
   parser once shipped and truncated a reply to `web` instead of
   `web2api-ok`; no mock reproduces it, because the mock *is* what broke.

   It skips with a precise reason per provider, so you only need one
   upstream configured. opencode is the cheapest probe (no account, no
   browser, no WAF — just a local process). Failures are redacted through
   the same scrubber the server uses, and name the provider, model, status
   and the SSE path the deltas arrived on, so a drift report is diagnosable
   without ever printing a credential.

CI: `.github/workflows/ci.yml` runs exactly
`python -m pytest -m "not live" -q` on Python 3.11 and 3.13 (3.11 is the
`requires-python` floor) against ubuntu-latest and windows-latest, with
`.[test]` installed **editable** so the install and pytest's `pythonpath = ["src"]`
agree — a plain install was shadowed by the checkout and proved nothing about
what was in the wheel. One extra step then installs a real (non-editable) wheel
and imports `deepseaport.core.tools.parser` and `deepseaport.providers.qwen`
from outside the checkout, which is the only check that notices a package
directory missing its `__init__.py`. No obscura, no browser, no account, no
secrets.

## Disclaimer

This is an independent educational/experimental project. 
Use of this project is at your own risk, and users are responsible for complying with all relevant terms and laws. 
It is provided as-is, without warranties.
Unofficial reverse engineering for personal research. Respect DeepSeek's
Terms of Service; web sessions can break or be rate-limited at any time.
Prefer the official `platform.deepseek.com` API for production.
