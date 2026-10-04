# scripts

## Purpose

Release packaging and the demo harness. Nothing here is imported by the
package.

## Ownership

- `prepare_release.py`: assembles `release/` plus `dist.zip` from `dist/`, refusing to ship a credential.
- `agentic_demo.py`: a standalone multi-step tool-calling loop against a running local server.

## Local Contracts

- `prepare_release.py` is the only supported way to package a release. `dist/` is two things at once: the PyInstaller output directory and the frozen app's live credential store, because `config._app_dir()` resolves to `Path(sys.executable).parent` when frozen. Zipping `dist/` by hand publishes every account the operator owns, and a credential in a published artifact is published even though `config.json` is gitignored and `deepseaport.spec` has no `datas` exclusion.
- The script never writes to, deletes from or moves anything in the source tree. It reads `dist/` and leaves the operator's live `config.json` exactly where the exe expects it.
- `--check` audits and writes nothing. Exit 1 means a credential was found and names the file and the JSON path inside it. Exit 0 means the tree is clean.
- `SECRET_KEYS` is `email`, `mobile`, `token`, `password`, `keys`, `api_key`, `secret` — the first two are half an account and were missing once, which let a store carrying only identifiers pass clean. Content scanning is JSON-only; a secret in `settings.ini`, `credentials.txt`, a `.har` or a screenshot ships unless its *name* matches `EXCLUDE_GLOBS`. Never let the docstring promise more than `survey` actually reads.
- Stdlib only, so it runs on the same interpreter that built the exe with nothing else installed.
- `agentic_demo.py` is a demo, not a test. It is not collected by pytest, needs a server already listening on `http://127.0.0.1:5001`, and uses the documented public example key.

## Work Guidance

- Keep both scripts stdlib-only and runnable from a checkout with no install.
- If you add a flag to `prepare_release.py`, keep `--check` read-only.

## Verification

- `python scripts/prepare_release.py --check`, expecting exit 0. This is the check that has to run before any release, and `.github/workflows/release.yml` runs it as a hard gate ahead of the PyPI upload on a `v*` tag.
- `tests/test_release_guard.py` in the suite covers the guard's verdicts and the version derivation in `deepseaport.spec`.
- `agentic_demo.py` has no automated check. Run it against a local server when the tool-calling loop changes.