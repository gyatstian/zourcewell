# scripts

## Purpose

Release packaging. Nothing here is imported by the package.

## Ownership

- `prepare_release.py`: assembles `release/` plus `dist.zip` from `dist/`, refusing to ship a credential.

## Local Contracts

- `prepare_release.py` is the only supported way to package a release; never zip `dist/` by hand. `dist/` doubles as the frozen app's live credential store, so the artifact rules live in `ARCHITECTURE.md` under "The release guard is `scripts/prepare_release.py`".
- The script never writes to, deletes from or moves anything in the source tree. It reads `dist/` and leaves the operator's live `config.json` exactly where the exe expects it.
- `--check` audits and writes nothing. Exit 1 means the check failed: a credential was found (the file and the JSON path inside it are named), or a JSON file could not be read or parsed (`<unreadable: ...>` / `<unparseable JSON: ...>` — `scan_json` never returns clean for input it could not scan). Exit 0 means the tree is clean.
- `SECRET_KEYS` is `email`, `mobile`, `token`, `password`, `keys`, `api_key`, `secret` — the first two are half an account and were missing once, which let a store carrying only identifiers pass clean. Content scanning is JSON-only; a secret in `settings.ini`, `credentials.txt`, a `.har` or a screenshot ships unless its *name* matches `EXCLUDE_GLOBS`. Never let the docstring promise more than `survey` actually reads.
- Stdlib only, so it runs on the same interpreter that built the exe with nothing else installed.

## Work Guidance

- Keep the script stdlib-only and runnable from a checkout with no install.
- If you add a flag to `prepare_release.py`, keep `--check` read-only.

## Verification

- `python scripts/prepare_release.py --check`, expecting exit 0. This is the check that has to run before any release, and `.github/workflows/release.yml` runs it as a hard gate ahead of the PyPI upload on a `v*` tag.
- `tests/test_release_guard.py` in the suite covers the guard's verdicts and the version derivation in `deepseaport.spec`.