"""Shared pytest setup: hermetic config, environment and an accident guard on
outbound network.

Without this module the suite's verdict depends on the developer machine:

* ``load_settings()`` resolves ``./config.json`` — the real credential file.
  Three qwen tests reached ``Settings.save()`` through the
  ``engine.app_settings()`` fallback and rewrote it on every run.
* ``DEEPSEAPORT_*`` / ``OBSCURA_*`` in the ambient environment override
  ``config.json`` and are captured at import time by several modules
  (``config.DATA_DIR``, ``tools.common.MAX_ARGS_CHARS``, the protocol hosts),
  so a stray export flipped 44 test verdicts before this file existed.
* ``tests/test_speed.py`` drives the real upstream: it had no marker, so it
  ran by default and failed (not skipped) on any machine whose config selects
  another provider.

What this module does, and why it is safe:

* ``pytest_configure`` runs BEFORE collection, which is the only point where
  the import-time constants above can still be steered.
* Live tests opt in explicitly: mark them ``@pytest.mark.live`` and run
  ``pytest -m live``. Do NOT express the default as ``addopts = "-m 'not
  live'"`` — a command-line ``-m`` does not override ``addopts``, so
  ``pytest -m live`` would silently deselect everything.
* **Non-loopback outbound connections raise instead of leaving the machine.**

  Scope, stated precisely because this is an ACCIDENT GUARD, NOT A SANDBOX.
  It wraps exactly three callables — ``socket.socket.connect``,
  ``socket.socket.connect_ex`` and ``socket.create_connection`` — and
  restores them by assigning the saved native methods back onto
  ``socket.socket``/``socket``. So it catches the mistake where a test
  quietly opens a connection to a real host through the normal API. It does
  NOT catch, and must not be relied on to catch:

  * ``sendto`` / ``sendmsg`` (UDP and raw datagrams — no connect involved),
  * ``socket.getaddrinfo`` / ``gethostbyname`` (DNS leaves the machine even
    when no socket is opened),
  * anything built on a library that bypasses ``socket.socket`` entirely
    (curl_cffi/wasmtime under the hood, an HTTP client with its own
    transport), and
  * subprocesses, which get a pristine ``os.environ``-plus-import state.

  The deliberate design choice is to guard the narrow, high-value path rather
  than pretend to sandbox the interpreter. A guard that claimed to be
  complete would be worse than an honest one: it would stop people from
  looking for the escapes. Anything that legitimately needs the network is a
  ``live`` test, and selecting the live tier lifts the guard entirely.
* The imported ``deepseaport`` package must live in THIS repository's
  ``src/``. An editable install of the project can point at a sibling clone;
  ``pyproject.toml``'s ``pythonpath = ["src"]`` makes ``python -m pytest``
  prefer this tree, but a bare ``pytest`` (wrong rootdir/ini), a stale
  ``sys.path`` entry, or any ad-hoc script silently exercises the OTHER
  source tree and every verdict is about code nobody edited. The guard in
  ``_assert_local_source_tree`` turns that silent trap into a hard configure
  -time failure naming the path that was actually resolved. This one check is
  NOT lifted in live mode — it has nothing to do with the network.
* **Selecting the live tier opts out of the hermetic config.** See
  ``_live_requested``. Rewriting ``DEEPSEAPORT_CONFIG`` to an empty temp
  config under ``-m live`` guaranteed that every live test skipped on its
  readiness check, so the documented live command could never actually run.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import socket
import tempfile
from pathlib import Path

import pytest

# Prefixes whose ambient values must not reach a test. OBSCURA_* is included
# because obscura_bridge resolves the binary from it and would otherwise spawn
# a real browser from a developer's machine.
_STRIPPED_PREFIXES = ("DEEPSEAPORT_", "DEEPSEEK_", "OBSCURA_")
_STRIPPED_EXACT = ("OBSCURA_BIN", "OBSCURA_PROFILE")

_LOOPBACK_HOSTS = ("127.", "::1", "localhost", "0.0.0.0")

_TMP_ROOTS: list[Path] = []

# True when the operator selected the live tier on the command line. Set once
# in pytest_configure and read by the network fixture; see _live_requested.
_LIVE_TIER = False

# This repository, derived from this file (never from cwd): a bare `pytest`
# run from a subdirectory, or `pytest /some/other/path`, must still be judged
# against the tree that owns the tests.
REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_SRC = (REPO_ROOT / "src").resolve()

# Resolved once in pytest_configure and echoed by pytest_report_header, so the
# report states which source tree produced the verdict.
_RESOLVED_PACKAGE_PATH = "<not checked>"

# `-m "not live"` and `-m "not (live)"` DESELECT the live tier; they are the
# opposite of an opt-in. Stripped before looking for a positive selection.
_LIVE_NEGATION_RE = re.compile(r"not\s*\(\s*live\s*\)|not\s+live\b")


def _live_requested(config) -> bool:
    """True when the command line positively SELECTS the live tier.

    This is the fix for a live tier that could only ever skip: the old
    ``pytest_configure`` rewrote ``DEEPSEAPORT_CONFIG`` to a temp config with
    ``accounts: []`` unconditionally, so ``pytest -m live`` always found no
    account and every live test skipped on its readiness check. The operator
    asking for the live tier is asking for the real machine.

    Deliberately narrow, because the failure mode of being too wide is worse
    than the bug being fixed. CI passes ``-m "not live"``; treating that as an
    opt-in would point the whole default suite at a developer's real
    ``config.json`` and its live credentials, silently. So a negated ``live``
    is stripped first and only a surviving positive mention counts:

    ============================  ======  ==============================
    ``-m`` value                  live?   why
    ============================  ======  ==============================
    (none)                        no      default hermetic run
    ``live``                      YES     the documented live command
    ``not live``                  no      what CI passes; a deselection
    ``not (live)``                no      same, parenthesised
    ``live and not slow``         YES     live present, not negated
    ``not live and foo``          no      live is inside the negation
    ============================  ======  ==============================
    """
    expr = str(getattr(getattr(config, "option", None), "markexpr", "") or "").strip()
    if not expr:
        return False
    return "live" in _LIVE_NEGATION_RE.sub("", expr)


def _stripped(name: str) -> bool:
    return name.startswith(_STRIPPED_PREFIXES) or name in _STRIPPED_EXACT


def _installed_deepseaport_path() -> str:
    """Filesystem path the ``deepseaport`` import resolves to right now.

    ``__file__`` covers a normal package; a namespace/extension-less layout
    falls back to the first ``__path__`` entry. Never raises: a failed import
    returns a marker string so the guard can report it instead of exploding
    with an unrelated ImportError.
    """
    try:
        import deepseaport
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return f"<import failed: {type(exc).__name__}: {exc}>"
    try:
        mod = getattr(deepseaport, "__file__", None)
        if mod:
            return str(Path(mod).resolve())
        for entry in getattr(deepseaport, "__path__", []) or []:
            return str(Path(entry).resolve())
    except Exception:  # noqa: BLE001 - defensive
        pass
    return "<unknown location>"


def _is_local_package_path(resolved: str) -> bool:
    """True when ``resolved`` is the deepseaport package inside EXPECTED_SRC."""
    if not resolved or resolved.startswith("<"):
        return False
    try:
        return Path(resolved).resolve().is_relative_to(EXPECTED_SRC)
    except (AttributeError, ValueError, OSError):
        return False


def _assert_local_source_tree(resolved: str | None = None) -> str:
    """Fail loudly unless the imported deepseaport is THIS repo's ``src/``.

    Runs from ``pytest_configure`` (before collection) so a wrong-tree run
    never collects a single test. ``resolved`` is injectable purely so the
    guard's own test can simulate a foreign package location without
    uninstalling anything.
    """
    path = _installed_deepseaport_path() if resolved is None else str(resolved)
    if not _is_local_package_path(path):
        raise pytest.UsageError(
            "wrong deepseaport source tree: `import deepseaport` resolved to\n"
            f"  {path}\n"
            f"but these tests must exercise THIS checkout's src/:\n"
            f"  {EXPECTED_SRC}\n"
            "An editable install of deepseaport points at a different clone, so "
            "any ad-hoc script or a bare `pytest` (no rootdir pyproject.toml, "
            "so no `pythonpath = [\"src\"]`) silently tests code nobody edited.\n"
            "Fix: run the suite as `python -m pytest` from "
            f"{REPO_ROOT} (pyproject.toml sets pythonpath = [\"src\"]), or "
            "reinstall the editable package against this checkout "
            f"(`pip install -e {REPO_ROOT}`), or drop the stale entry from "
            "sys.path / PYTHONPATH."
        )
    return path


def _check_local_source_tree() -> str:
    """Configure-time entry point; remembers the path for the report header."""
    global _RESOLVED_PACKAGE_PATH
    _RESOLVED_PACKAGE_PATH = _assert_local_source_tree()
    return _RESOLVED_PACKAGE_PATH


def pytest_configure(config: pytest.Config) -> None:
    global _LIVE_TIER
    # Always, live or not: this is about WHICH source tree is under test, not
    # about the network or the config file.
    _check_local_source_tree()
    _LIVE_TIER = _live_requested(config)
    if _LIVE_TIER:
        # The operator asked for the real machine. Do NOT rewrite
        # DEEPSEAPORT_CONFIG, do NOT create a temp config, and do NOT strip
        # the ambient DEEPSEAPORT_*/DEEPSEEK_*/OBSCURA_* variables: the live
        # tier needs the operator's real account token, and honouring
        # DEEPSEAPORT_PROVIDER / DEEPSEAPORT_OPENCODE_BASE / an obscura path
        # is the difference between the live tier working and not.
        return
    root = Path(tempfile.mkdtemp(prefix="deepseaport-tests-"))
    _TMP_ROOTS.append(root)
    atexit.register(shutil.rmtree, str(root), ignore_errors=True)

    data_dir = root / "data"
    data_dir.mkdir()
    # A valid, empty config: `load_settings()` with no argument now returns
    # this instead of the developer's credential file, so a stray
    # ``Settings.save()`` can only ever touch the temp tree.
    cfg = root / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "keys": ["sk-hermetic-test"],
                "accounts": [],
                "active_account": "",
                "provider": "deepseek",
                "providers": [],
                "warmup_on_startup": False,
                "log_level": "WARNING",
            }
        ),
        encoding="utf-8",
    )

    for name in [n for n in os.environ if _stripped(n)]:
        del os.environ[name]
    os.environ["DEEPSEAPORT_CONFIG"] = str(cfg)
    os.environ["DEEPSEAPORT_DATA"] = str(data_dir)
    # The opencode model catalogue BOOTS the managed serve when nothing
    # answers at the configured base, so that a fresh process reports the same
    # ids as the running server. A suite must never spawn that process: every
    # catalogue test fakes `client._request` instead. The socket guard cannot
    # catch it -- it wraps connect(), not subprocesses.
    os.environ["DEEPSEAPORT_OPENCODE_MANAGED"] = "0"


def pytest_report_header(config: pytest.Config) -> list[str]:
    if _LIVE_TIER:
        # Do not claim a hermetic config that is not in effect.
        return ["deepseaport tests: LIVE TIER selected - the operator's real "
                "DEEPSEAPORT_CONFIG and ambient DEEPSEAPORT_*/DEEPSEEK_*/"
                "OBSCURA_* environment are in use, NOT a temp config",
                f"deepseaport source tree = {_RESOLVED_PACKAGE_PATH} "
                f"(expected under {EXPECTED_SRC})"]
    env = os.environ.get("DEEPSEAPORT_CONFIG", "")
    return [f"deepseaport tests: hermetic config = {env}",
            f"deepseaport source tree = {_RESOLVED_PACKAGE_PATH} "
            f"(expected under {EXPECTED_SRC})"]


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip `live` tests unless the operator asked for them."""
    if "live" in (getattr(config.option, "markexpr", "") or ""):
        return
    skip = pytest.mark.skip(reason="live test: opt in with `pytest -m live`")
    for item in items:
        # get_closest_marker, not `"live" in item.keywords`: the latter also
        # substring-matches module/function names (e.g. test_live_reasoner_*).
        if item.get_closest_marker("live") is not None:
            item.add_marker(skip)


def _is_loopback(address: object) -> bool:
    host = address[0] if isinstance(address, tuple) else address
    if not isinstance(host, str):
        return False
    host = host.strip("[]")
    return host == "" or any(host.startswith(p) for p in _LOOPBACK_HOSTS)


@pytest.fixture(autouse=True, scope="session")
def _no_external_network():
    """Fail loudly instead of silently hitting the real internet.

    An ACCIDENT GUARD, not a sandbox -- see the module docstring for the exact
    scope (connect / connect_ex / create_connection only; sendto and DNS are
    not covered). Selecting the live tier lifts it, because a live test that
    cannot reach the network is not a live test.
    """
    if _LIVE_TIER:
        yield
        return
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create = socket.create_connection

    def _guard(name, fn, at):
        """Wrap a connect call and allow loopback only.

        ``at`` is the positional index of the address, and it is NOT the same
        for all three: patched onto the class, socket.connect/connect_ex take
        ``(self, address)``, while the module-level create_connection takes
        ``(address, timeout=...)``. httpx passes that timeout POSITIONALLY, so
        reading args[-1] made a plain loopback connect look like a connect to
        20.0 and block it.
        """

        def wrapper(*args, **kwargs):
            target = kwargs.get("address", args[at] if len(args) > at else None)
            if not _is_loopback(target):
                raise RuntimeError(
                    f"hermetic test suite: blocked {name} to {target!r}. "
                    "Mark the test `@pytest.mark.live` if it needs the network."
                )
            return fn(*args, **kwargs)

        return wrapper

    socket.socket.connect = _guard("socket.connect", real_connect, 1)
    socket.socket.connect_ex = _guard("socket.connect_ex", real_connect_ex, 1)
    socket.create_connection = _guard("socket.create_connection", real_create, 0)
    try:
        yield
    finally:
        socket.socket.connect = real_connect
        socket.socket.connect_ex = real_connect_ex
        socket.create_connection = real_create


# --- shared test doubles ---
# The settings factories, config writers, SSE collectors and obscura/uvicorn
# stand-ins live in `tests/support.py` and are imported directly by each test
# file (`from support import settings`). They are plain functions, not
# fixtures, so re-exporting them as fixtures would only add a second name for
# the same object; see `support.py` and `tests/AGENTS.md`.
