# tests

## Purpose

The hermetic suite that gates every change, plus the opt-in live tier that is
the only thing in the repo able to notice an upstream changed.

## Ownership

Every `conftest.py` and `test_*.py` in this folder. The fixture design in
`conftest.py` is the contract; the individual tests belong to whichever area
they exercise, but they live here.

`support.py` owns the shared test doubles: the `settings()` factory,
`write_config`/`on_disk`, the SSE collectors, and the obscura/uvicorn/thread
stand-ins. It is not collected as a test module.

## Local Contracts

- `conftest.py` runs in `pytest_configure`, before collection, which is the only point at which the import-time constants (`config.DATA_DIR`, `tools.common.MAX_ARGS_CHARS`, the protocol hosts) can still be steered. It points `ZOURCEWELL_CONFIG` and `ZOURCEWELL_DATA` at a temp dir, strips `ZOURCEWELL_*`, `DEEPSEEK_*` and `OBSCURA_*` from the ambient environment, and then sets `ZOURCEWELL_OPENCODE_MANAGED=0`. That last one is load-bearing, not tidiness: the opencode model catalogue BOOTS the managed serve when nothing answers, and the socket guard cannot catch it (it wraps `connect`, not `subprocess`).
- `_assert_local_source_tree` fails at configure time when the imported `zourcewell` is not this checkout's `src/`. An editable install can point at a sibling clone, and then every verdict is about code nobody edited. This check is not lifted in live mode; it has nothing to do with the network.
- Never express the default tier as `addopts = "-m 'not live'"`. A command-line `-m` does not override `addopts`, so `pytest -m live` would silently deselect everything and the documented live command could never run. `-m "not live"` belongs on the CI command line instead.
- Mark anything needing real network, a real browser or a real account `@pytest.mark.live`. Selecting the live tier lifts the hermetic config rewrite and the network guard entirely.
- The SDK-compat tier (`tests/test_openai_sdk.py`) drives the real `openai` SDK against the in-process apps over an ASGI transport (no socket) and `pytest.importorskip`s the SDK plus `httpx2`, so a checkout without the `.[test]` extra skips instead of failing.
- The socket guard wraps `socket.socket.connect`, `socket.socket.connect_ex` and `socket.create_connection` only, and takes the address from a per-callable positional index: `connect`/`connect_ex` are patched onto the class so the address is arg 1, module-level `create_connection` takes it as arg 0 (httpx passes its timeout positionally, which is what makes a uniform index wrong). It is an accident guard, not a sandbox: it does not catch `sendto`, DNS lookups, curl_cffi under the hood, or subprocesses. Do not write a test that depends on it catching those, and do not describe it as complete.
- Live failures are scrubbed through the same helper the server uses and name the provider, model, status and SSE path, so a drift report is diagnosable without printing a credential.

## Work Guidance

- `support.py` is imported directly (`from support import settings, FakeBridge`). These are plain functions, not fixtures: re-exporting them as fixtures would add a second name for the same object. Do not re-add a private `_settings`, `_write_cfg`, `_on_disk` or fake-bridge copy to a test module.
- A file whose defaults differ from `support.settings()` passes them as kwargs on top of it. Where one of those defaults is a list, build it inside a 2-line wrapper (`kw.setdefault("accounts", [...])`) rather than freezing it: `settings()` rebuilds its own lists per call precisely so one test cannot poison the next.
- Prefer `functools.partial` when the differing defaults are all scalars — `partial` keywords are overridable at the call site, a frozen default dict is not.
- `collect_sse` is `async` because `test_qwen_stream_parity.py` awaits it from inside a running loop. Never wrap it in `asyncio.run` inside a sync helper.
- Default to hermetic. If a test needs the real wire, it is a live test, and it skips with a precise reason per provider so only one upstream has to be configured.
- Name a test for the subject it protects: `*_gaps.py`, `*_hardening.py`, `test_<area>_<subject>.py`. A failure should name its own subject.
- Prefer `pythonpath = ["src"]` over relying on an editable install, but keep the editable install too so `python -m zourcewell` runs this checkout.
- Several files carry non-ASCII (a Polish ban-date string, an em dash in prose). PEP 3120 makes `.py` source UTF-8 with no declaration, so keep the characters; do not strip them to make an encoding error go away.

## Verification

- `python -m pytest -m "not live" -q` is the gate, and CI runs exactly that on ubuntu-latest and windows-latest with Python 3.11 and 3.13.
- `python -m pytest -m live -q` periodically, and after any upstream wire change. `tests/test_live_smoke.py` is small (one run per provider, seconds) and asserts what a stub cannot: DeepSeek's bare continuation chunks, Qwen's `delta.phase` split, the WAF jar, the streamed `tool_calls` index, and that a reply contains the end of the prompt.