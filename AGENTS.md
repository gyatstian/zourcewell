# DOX framework

- DOX is highly performant AGENTS.md hierarchy installed here
- Agent must follow DOX instructions across any edits

## Core Contract

- AGENTS.md files are binding work contracts for their subtrees
- Work products, source materials, instructions, records, assets, and durable docs must stay understandable from the nearest applicable AGENTS.md plus every parent AGENTS.md above it

## Read Before Editing

1. Read the root AGENTS.md
2. Identify every file or folder you expect to touch
3. Walk from the repository root to each target path
4. Read every AGENTS.md found along each route
5. If a parent AGENTS.md lists a child AGENTS.md whose scope contains the path, read that child and continue from there
6. Use the nearest AGENTS.md as the local contract and parent docs for repo-wide rules
7. If docs conflict, the closer doc controls local work details, but no child doc may weaken DOX

Do not rely on memory. Re-read the applicable DOX chain in the current session before editing.

## Update After Editing

Every meaningful change requires a DOX pass before the task is done.

Update the closest owning AGENTS.md when a change affects:

- purpose, scope, ownership, or responsibilities
- durable structure, contracts, workflows, or operating rules
- required inputs, outputs, permissions, constraints, side effects, or artifacts
- user preferences about behavior, communication, process, organization, or quality
- AGENTS.md creation, deletion, move, rename, or index contents

Update parent docs when parent-level structure, ownership, workflow, or child index changes. Update child docs when parent changes alter local rules. Remove stale or contradictory text immediately. Small edits that do not change behavior or contracts may leave docs unchanged, but the DOX pass still must happen.

## Hierarchy

- Root AGENTS.md is the DOX rail: project-wide instructions, global preferences, durable workflow rules, and the top-level Child DOX Index
- Child AGENTS.md files own domain-specific instructions and their own Child DOX Index
- Each parent explains what its direct children cover and what stays owned by the parent
- The closer a doc is to the work, the more specific and practical it must be

## Child Doc Shape

- Create a child AGENTS.md when a folder becomes a durable boundary with its own purpose, rules, responsibilities, workflow, materials, or quality standards
- Work Guidance must reflect the current standards of the project or user instructions; if there are no specific standards or instructions yet, leave it empty
- Verification must reflect an existing check; if no verification framework exists yet, leave it empty and update it when one exists

Default section order:
- Purpose
- Ownership
- Local Contracts
- Work Guidance
- Verification
- Child DOX Index

## Style

- Keep docs concise, current, and operational
- Document stable contracts, not diary entries
- Put broad rules in parent docs and concrete details in child docs
- Prefer direct bullets with explicit names
- Do not duplicate rules across many files unless each scope needs a local version
- Delete stale notes instead of explaining history
- Trim obvious statements, repeated rules, misplaced detail, and warnings for risks that no longer exist

## Closeout

1. Re-check changed paths against the DOX chain
2. Update nearest owning docs and any affected parents or children
3. Refresh every affected Child DOX Index
4. Remove stale or contradictory text
5. Run existing verification when relevant
6. Report any docs intentionally left unchanged and why

## User Preferences

When the user requests a durable behavior change, record it here or in the relevant child AGENTS.md

## Project Purpose

zourcewell exposes web and local chat backends as an OpenAI-compatible FastAPI
API. Supported upstreams: DeepSeek (AWS WAF, PoW, real-browser login), Qwen
(Aliyun WAF, JWT cookie), opencode (a local CLI over its own HTTP API). Research
and educational use only.

## Repo-wide Contracts

- Interpreter: CPython 3.13 is the verified target. `pyproject.toml` declares `requires-python = ">=3.11"`; CI runs the matrix on 3.11 and 3.13. 3.11 is the floor because `zourcewell.spec` reads `pyproject.toml` with stdlib `tomllib`.
- `src/` layout. `pythonpath = ["src"]` under `[tool.pytest.ini_options]` makes the suite resolve this checkout instead of another install.
- `config.json` is the live credential store: account emails, plaintext passwords, bearer tokens, API keys. Gitignored, never committed, never copied into an artifact.
- `[project] version` in `pyproject.toml` is the only place a version is written. `zourcewell.spec` stamps it into the frozen exe's bundled `_version.txt`, which `src/zourcewell/__init__.py` reads FIRST (then pyproject, installed metadata, `"0.0.0"`), and `cli --version`, `server.py` / `dry_run.py`'s `FastAPI(version=...)` and `server_routes.py`'s `/health` and `/v1/models` all report that one `zourcewell.__version__`; never add a second literal.
- `scripts/prepare_release.py --check` is a hard gate: it runs before every publish step and no workflow step that publishes may carry `continue-on-error`. `dist/` doubles as the frozen app's credential store; full rationale in `ARCHITECTURE.md`.
- The project name is ONE identity spread over surfaces that must move together: the `src/zourcewell/` package, the `zourcewell` distribution and console script (`pyproject.toml`), the `ZOURCEWELL_*` env prefix, the `zourcewell.*` logger tree, the `zourcewell-passthrough` managed agent and `zourcewell-strip-system` plugin id, the `zourcewell.spec` PyInstaller file and the `zourcewell.exe` it builds, and the `github.com/gyatstian/zourcewell` URLs. A rename changes ALL of them in one change; a partial one strands the rest silently, because each is read by a different consumer.
- `ARCHITECTURE.md` and `PROVIDERS.md` are the authoritative long-form docs and are owned here: request lifecycle, auth, PoW, SSE shapes, tool calling, environment, the `Provider` ABC contract, and the steps for adding an upstream. Update them in the same change as the behavior they describe.
- Runtime output (`build/`, `dist/`, `data/`, `dummy-profile/`, `.pi/`, local `obscura*` binaries) is gitignored and owns no docs.
- No module under `src/zourcewell/core/` or `src/zourcewell/providers/` may import `zourcewell.server` at module scope. Where a seam needs it, late-import inside the function so monkeypatching `zourcewell.server` keeps working.

## Verification

- `python -m pytest -m "not live" -q` is the gate. Hermetic: no account, no obscura binary, no browser, no secrets. CI runs exactly this on ubuntu-latest and windows-latest, Python 3.11 and 3.13.
- `python -m pytest -m live -q` is opt-in and needs a real token plus `obscura` on PATH. Run it when a change touches an upstream wire format. It is the only check in the repo that notices upstream drift.
- `python scripts/prepare_release.py --check` before packaging. Exit 1 means the check failed — a credential found in `dist/`, or a JSON file it could not read or parse. On a `v*` tag the `release` workflow runs it for you, ahead of the PyPI upload.

## Child DOX Index

- `src/zourcewell/AGENTS.md` owns the package itself: config, account pool, server wiring, CLI, and both chat frontends.
- `src/zourcewell/core/AGENTS.md` owns provider-neutral mechanics: retry engine, transport, OpenAI shaping, pool helpers, registry, Obscura bridge, MCP client, token parsing.
- `src/zourcewell/core/tools/AGENTS.md` owns tool prompt rendering and the reply parser formats.
- `src/zourcewell/providers/AGENTS.md` owns the `Provider` contract every upstream implements.
- `src/zourcewell/providers/deepseek/AGENTS.md` owns the DeepSeek upstream: PoW, WAF, browser login, SSE.
- `src/zourcewell/providers/qwen/AGENTS.md` owns the Qwen upstream: JWT auth, Aliyun WAF, chat lifecycle.
- `src/zourcewell/providers/opencode/AGENTS.md` owns the local opencode CLI upstream and its managed serve subprocess.
- `tests/AGENTS.md` owns the hermetic suite and the live tier.
- `scripts/AGENTS.md` owns the release packaging script, `prepare_release.py`.

Deliberately unindexed: `.github/`, whose two workflows (`ci` runs the root command above, `release` publishes on a `v*` tag behind the `prepare_release.py --check` gate), and every gitignored runtime directory.
