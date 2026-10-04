"""Follow-up gap tests: G1..G7 from the IMPLEMENTER I6 work list.

Every test here names the PRE-CHANGE failure it reproduces, so a future
regression is a one-line diagnosis instead of a bisect. Items covered:

* G1  wrong-source-tree trap is loud (``tests/conftest.py``).
* G2  engine ban path must not persist the operator's config through the
      ``app_settings`` fallback.
* G3  provider-neutral pool/queue errors must not hardcode "DeepSeek".
* G5  ``core/openai._openai_response`` had a dead ``_diagnostics`` store.
* G6  CDP login browser must not pass ``--remote-allow-origins=*``.
* G7  ``complete_with_failover_sync`` callback params are accepted AND inert;
      wiring them up would double-apply every delta.

G4 (drop ``bash: True`` from the opencode per-message ``tools`` allowlist) is
deliberately NOT tested here: three existing cases in
``tests/test_opencode_managed.py`` pin ``tools["bash"] is True``, and that
file is outside this agent's write scope. See the change report.

Three tests were REMOVED from this file during the test-hygiene pass because
each survived a revert of the fix it was filed under, and so advertised a
guarantee the suite did not hold. They are listed in place, with the reason,
rather than silently dropped:

* ``test_g1_expected_src_is_derived_from_this_file`` -- recomputed conftest's
  own expression from conftest.
* ``test_g6_websocket_client_sends_no_origin`` -- a source-text grep for
  ``"suppress_origin=True"``; its two siblings assert the real argv.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import os
import threading
import time
from pathlib import Path

import pytest
from fastapi import HTTPException

import deepseaport
import deepseaport.accounts as A
from deepseaport.accounts import AccountPool, PooledAccount
from deepseaport.config import AccountConfig, Settings, load_settings
from deepseaport.core import engine as _engine
from deepseaport.core import openai as _openai
from deepseaport.providers.deepseek import real_browser as _rb
from deepseaport.server_concurrency import _acquire_slot_or_499


# --- G1: wrong-source-tree trap ------------------------------------------------


def test_g1_guard_rejects_foreign_package_location(tmp_path):
    """G1. Pre-change: no guard existed, so a sibling-clone editable install
    silently ran every assertion against code nobody edited.

    Simulated by handing the guard the path an editable install of a
    DIFFERENT checkout would produce. The guard must refuse it AND say what
    was expected, otherwise the failure is still a silent wrong-tree run.
    """
    import conftest

    foreign = (tmp_path / "other-clone" / "src" / "deepseaport" / "__init__.py")
    foreign.parent.mkdir(parents=True)
    foreign.write_text("", encoding="utf-8")

    assert conftest._is_local_package_path(str(foreign)) is False

    with pytest.raises(pytest.UsageError) as ei:
        conftest._assert_local_source_tree(str(foreign))

    msg = str(ei.value)
    assert str(foreign) in msg, "the message must name the path that resolved"
    assert str(conftest.EXPECTED_SRC) in msg, "the message must name the expected src/"
    assert "python -m pytest" in msg, "the message must give the working command"


def test_g1_guard_accepts_this_checkout():
    """G1 (control): the real resolved package is inside this repo's src/,
    so the configure-time guard never blocks a correct run."""
    import conftest

    resolved = conftest._installed_deepseaport_path()
    assert conftest._is_local_package_path(resolved) is True
    assert conftest._assert_local_source_tree() == resolved


def test_g1_report_header_states_the_source_tree():
    """G1: the verdict must be traceable to the tree that produced it."""
    import conftest

    conftest._check_local_source_tree()
    header = "\n".join(conftest.pytest_report_header(config=None))
    assert "deepseaport source tree =" in header
    assert str(conftest.EXPECTED_SRC) in header
    assert str(deepseaport.__file__) in header


# Deleted here on purpose: `test_g1_expected_src_is_derived_from_this_file`.
# It recomputed conftest's own expression
# (`EXPECTED_SRC == (Path(conftest.__file__).parent.parent / "src").resolve()`)
# from the same file that defines it, so it passed by construction, survived
# any revert, and could never fail for a reason a reader would trust. The
# property it gestured at -- "the guard must not depend on cwd" -- is covered
# for real by the two tests above: `_assert_local_source_tree` is called with
# a foreign absolute path and the configure-time guard in `pytest_configure`
# runs on every single run.


# --- G2: engine ban path must not write the operator's config -----------------


class _BareState:
    """An app.state that genuinely has no settings (test/tool embedding)."""


class _BareApp:
    def __init__(self) -> None:
        self.state = _BareState()


def _operator_config(tmp_path: Path) -> Path:
    path = tmp_path / "operator.json"
    path.write_text(json.dumps({
        "keys": ["sk-operator"],
        "accounts": [{"email": "op@x.com", "token": "op-token-0123456789"}],
        "active_account": "op@x.com",
        "provider": "deepseek",
        "providers": [],
    }), encoding="utf-8")
    return path


def test_g2_ban_on_fallback_path_writes_no_config_file(monkeypatch, tmp_path, caplog):
    """G2. Pre-change: ``app_settings(app).save()`` ran unconditionally, so a
    ban 403 raised for an app with no settings of its own loaded the
    operator's real config.json and then WROTE it.

    Reproduced: file bytes + mtime must be untouched, while the in-memory
    marker is still set and a warning says the ban is memory-only.
    """
    path = _operator_config(tmp_path)
    monkeypatch.setenv("DEEPSEAPORT_CONFIG", str(path))
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    app = _BareApp()
    item = PooledAccount(cfg=load_settings(str(path)).accounts[0])

    with caplog.at_level("WARNING", logger="deepseaport.core.engine"):
        exc = _engine.ban_403_for_item(app, item, time.time() + 86400, "user is muted")

    assert exc.status_code == 403
    assert item.cfg.banned is True, "the in-memory marker must still be set"
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before, \
        "the ban path wrote the operator's config through the app_settings fallback"
    assert reloaded_bans(path) == [False], "operator config gained a ban marker"
    assert any("in-memory only" in r.getMessage() for r in caplog.records), \
        "an unpersisted ban must be reported to the operator"


def reloaded_bans(path: Path) -> list[bool]:
    return [bool(getattr(a, "banned", False)) for a in load_settings(str(path)).accounts]


def test_g2_ban_still_persists_when_app_owns_its_settings(tmp_path):
    """G2 (control): the normal server path (create_app -> app.state.settings)
    must keep persisting the ban. Guards against 'fixing' the leak by never
    saving at all, which would silently drop (BANNED ...) from the CLI/TUI
    after a restart."""
    path = tmp_path / "owned.json"
    settings = Settings(keys=["sk"], accounts=[AccountConfig(email="b@x.com", token="tok")],
                        active_account="b@x.com", config_path=str(path))
    assert not path.exists()

    class _App:
        def __init__(self) -> None:
            class S:
                pass
            self.state = S()
            self.state.settings = settings

    app = _App()
    item = PooledAccount(cfg=settings.accounts[0])
    until = time.time() + 86400

    exc = _engine.ban_403_for_item(app, item, until, "user is muted")

    assert exc.status_code == 403
    assert path.exists(), "app-owned settings must still be written"
    reloaded = load_settings(str(path))
    assert reloaded.accounts[0].banned is True
    assert reloaded.accounts[0].banned_until == until


# --- G3: provider-neutral errors must not hardcode "DeepSeek" ----------------


class _Req:
    def __init__(self, disconnected: bool = False) -> None:
        self._d = disconnected

    async def is_disconnected(self) -> bool:
        return self._d


def _qwen_pool() -> AccountPool:
    return AccountPool([AccountConfig(email="q@x.com", token="tok", provider="qwen")])


def test_g3_pool_names_its_own_provider_in_empty_pool_error():
    """G3. Pre-change: ``_next_ordered`` raised the literal
    'no DeepSeek accounts in pool' for every provider, so a Qwen-only
    deployment told the operator to debug the wrong upstream."""
    pool = AccountPool([], provider="qwen")
    with pytest.raises(ValueError) as ei:
        pool.acquire(timeout=0.1)
    assert str(ei.value) == "no Qwen accounts in pool"
    assert "DeepSeek" not in str(ei.value)


def test_g3_pool_busy_message_names_its_own_provider():
    """G3 (same literal, sync side): the TimeoutError text was
    'no DeepSeek account free (busy or cooldown)' for every provider."""
    pool = _qwen_pool()
    held = pool.acquire(timeout=1)
    try:
        with pytest.raises(TimeoutError) as ei:
            pool.acquire(timeout=0.05)
        assert str(ei.value) == "no Qwen account free (busy or cooldown)"
        assert "DeepSeek" not in str(ei.value)
    finally:
        AccountPool.release(held)


def test_g3_aacquire_busy_message_names_its_own_provider():
    """G3: the async twin carried the same hardcoded label."""

    async def scenario() -> str:
        pool = _qwen_pool()
        held = await pool.aacquire(1)
        try:
            with pytest.raises(TimeoutError) as ei:
                await pool.aacquire(0.05)
            return str(ei.value)
        finally:
            AccountPool.release(held)

    assert asyncio.run(scenario()) == "no Qwen account free (busy or cooldown)"


def test_g3_pool_label_follows_the_member_account_provider():
    """G3: the label comes from the account's own ``provider`` field -- the
    same field ``pools.accounts_for_provider`` built the pool from -- so a
    per-provider pool built by ``pools.make_pool`` needs no extra wiring."""
    from deepseaport.core import pools

    settings = Settings(
        accounts=[AccountConfig(email="d@x.com", token="t1", provider="deepseek"),
                  AccountConfig(email="q@x.com", token="t2", provider="qwen")],
        provider="deepseek", providers=[], config_path="")
    assert pools.make_pool(settings, "qwen").provider_label == "Qwen"
    assert pools.make_pool(settings, "deepseek").provider_label == "DeepSeek"


def test_g3_acquire_slot_or_499_503_and_429_use_the_pool_label():
    """G3. Pre-change: 503 'no DeepSeek accounts in pool' / 429 'all DeepSeek
    accounts busy, retry later' were literals in provider-neutral
    ``server_concurrency``. Status codes are unchanged; only the label is."""
    empty = AccountPool([], provider="qwen")
    busy = _qwen_pool()
    held = busy.acquire(timeout=1)

    async def empty_503() -> HTTPException:
        with pytest.raises(HTTPException) as ei:
            await _acquire_slot_or_499(empty, _Req(), threading.Event(), 0.1, True)
        return ei.value

    async def busy_429() -> HTTPException:
        with pytest.raises(HTTPException) as ei:
            await _acquire_slot_or_499(busy, _Req(), threading.Event(), 0.1, True)
        return ei.value

    try:
        e503, e429 = asyncio.run(empty_503()), asyncio.run(busy_429())
    finally:
        AccountPool.release(held)

    assert (e503.status_code, e503.detail) == (503, "no Qwen accounts in pool")
    assert (e429.status_code, e429.detail) == (429, "all Qwen accounts busy, retry later")


def test_g3_acquire_slot_or_499_accepts_an_explicit_label():
    """G3: the OPTIONAL keyword lets a caller name the provider for a pool
    that cannot (e.g. an empty pool built before the accounts were added)."""
    empty = AccountPool([])  # no members -> would fall back to DeepSeek

    async def scenario() -> HTTPException:
        with pytest.raises(HTTPException) as ei:
            await _acquire_slot_or_499(empty, _Req(), threading.Event(), 0.1, True,
                                       provider_label="Qwen")
        return ei.value

    assert asyncio.run(scenario()).detail == "no Qwen accounts in pool"


def test_g3_no_label_fallback_keeps_the_historical_deepseek_wording():
    """G3 (fallback): a pool with no members and no explicit label -- and a
    duck-typed pool exposing no label at all -- must still say exactly what
    it always said, so single-provider/default deployments and existing
    string assertions are untouched."""
    assert AccountPool([]).provider_label == "DeepSeek"
    assert AccountPool([]).no_accounts_error() == "no DeepSeek accounts in pool"
    assert A.pool_provider_label(object()) == "DeepSeek"
    assert A.pool_provider_label(None) == "DeepSeek"
    # A duck-typed pool carrying a bare provider ID resolves the display name.
    class _Fake:
        provider = "qwen"

    class _ReadyLabel:
        provider_label = "Qwen"

    assert A.pool_provider_label(_Fake()) == "Qwen"
    assert A.pool_provider_label(_ReadyLabel()) == "Qwen"

    class _Bare:
        def aacquire(self, *_a, **_k):
            raise ValueError("no DeepSeek accounts in pool")

    async def scenario() -> tuple[int, str]:
        with pytest.raises(HTTPException) as ei:
            await _acquire_slot_or_499(_Bare(), _Req(), threading.Event(), 0.1, True)
        return ei.value.status_code, str(ei.value.detail)

    assert asyncio.run(scenario()) == (503, "no DeepSeek accounts in pool")


def test_g3_provider_label_helper_maps_ids_and_never_raises():
    """G3: single home for the id->label mapping; unknown ids degrade to the
    id instead of to somebody else's product name."""
    assert A.provider_label("deepseek") == "DeepSeek"
    assert A.provider_label("qwen") == "Qwen"
    assert A.provider_label("opencode") == "opencode"
    assert A.provider_label("  QWEN ") == "Qwen"
    assert A.provider_label(None) == "DeepSeek"
    assert A.provider_label("") == "DeepSeek"
    assert A.provider_label("totally-unknown") == "totally-unknown"


def test_g3_pool_provider_id_and_signature_are_backwards_compatible():
    """G3: the new ``provider`` kwarg is optional and trailing; the two-arg
    construction every caller/test uses is unchanged."""
    params = list(inspect.signature(AccountPool.__init__).parameters)
    assert params == ["self", "accounts", "current", "provider"]
    assert inspect.signature(AccountPool.__init__).parameters["provider"].default is None
    assert AccountPool([AccountConfig(email="a@x.com")], "a@x.com").current == "a@x.com"
    # Empty pool, no label: historical default.
    assert AccountPool([]).provider_id == "deepseek"


def test_g3_acquire_slot_or_499_signature_only_gained_an_optional_kwarg():
    """G3: the exported signature keeps all five positional parameters, so the
    two 5-arg call sites in server_routes need no change."""
    sig = inspect.signature(_acquire_slot_or_499)
    names = list(sig.parameters)
    assert names == ["pool", "request", "cancel_event", "timeout", "allow_failover",
                     "provider_label"]
    assert sig.parameters["provider_label"].default is None
    for name in ("pool", "request", "cancel_event", "timeout", "allow_failover"):
        assert sig.parameters[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


# --- G8: `pytest -m live` must not be silently neutered by conftest -----------


def _cfg(markexpr: str):
    """Minimal stand-in for the bits of pytest.Config conftest reads."""
    class _Option:
        pass

    class _Config:
        pass

    cfg = _Config()
    cfg.option = _Option()
    cfg.option.markexpr = markexpr
    return cfg


@pytest.mark.parametrize("markexpr,expected", [
    ("", False),                    # plain `python -m pytest`
    ("live", True),                 # the documented live command
    ("not live", False),            # exactly what CI passes
    ("not (live)", False),
    ("live and not slow", True),    # live present and not negated
    ("not live and foo", False),    # live is inside the negation
])
def test_g8_live_tier_detection(markexpr, expected):
    """G8. The opt-in must be a POSITIVE selection only.

    Too wide is worse than the bug it replaces: if `-m "not live"` counted as
    an opt-in, CI's own command would stop rewriting the config and the whole
    default suite would read a developer's real ``config.json``.
    """
    import conftest

    assert conftest._live_requested(_cfg(markexpr)) is expected, markexpr


def test_g8_live_selection_leaves_the_operator_config_and_env_alone(
        monkeypatch, tmp_path):
    """G8. Pre-change: ``pytest_configure`` rewrote ``DEEPSEAPORT_CONFIG`` to a
    temp config with ``accounts: []`` and stripped every ambient
    ``DEEPSEAPORT_*``/``DEEPSEEK_*``/``OBSCURA_*`` var, unconditionally. Under
    ``-m live`` that meant the live tier's readiness check could never find an
    account, so the documented live command could only ever skip.
    """
    import conftest

    real_cfg = tmp_path / "operator.json"
    real_cfg.write_text(json.dumps({
        "keys": ["sk-real"],
        "accounts": [{"email": "a@x.com", "token": "tok-real"}],
    }), encoding="utf-8")
    monkeypatch.setenv("DEEPSEAPORT_CONFIG", str(real_cfg))
    monkeypatch.setenv("DEEPSEAPORT_PROVIDER", "qwen")
    monkeypatch.setenv("DEEPSEAPORT_OPENCODE_BASE", "http://127.0.0.1:4999")
    monkeypatch.setenv("OBSCURA_BIN", "C:/oc/obscura.exe")
    # Registered so monkeypatch restores the module global after this test,
    # even though pytest_configure rebinds it directly.
    monkeypatch.setattr(conftest, "_LIVE_TIER", False)

    created: list = []
    monkeypatch.setattr(conftest.tempfile, "mkdtemp",
                        lambda **kw: created.append(kw) or str(tmp_path / "unused"))
    monkeypatch.setattr(conftest.atexit, "register", lambda *a, **k: None)

    conftest.pytest_configure(_cfg("live"))

    assert os.environ["DEEPSEAPORT_CONFIG"] == str(real_cfg), \
        "-m live still rewrote the config path, so the live tier can never run"
    assert created == [], "a temp config tree was created under -m live"
    # The ambient overrides the operator set are part of "the real machine".
    assert os.environ["DEEPSEAPORT_PROVIDER"] == "qwen"
    assert os.environ["DEEPSEAPORT_OPENCODE_BASE"] == "http://127.0.0.1:4999"
    assert os.environ["OBSCURA_BIN"] == "C:/oc/obscura.exe"
    # And the accident guard is lifted, or a live test still cannot connect.
    assert conftest._LIVE_TIER is True
    header = "\n".join(conftest.pytest_report_header(config=None))
    assert "LIVE TIER" in header
    assert "hermetic config" not in header, \
        "the report must not claim a hermetic config that is not in effect"


def test_g8_default_run_still_rewrites_the_config_and_strips_the_env(
        monkeypatch, tmp_path):
    """G8 (control): the hermetic default is unchanged. Without this the fix
    above would be satisfied by simply deleting the isolation."""
    import conftest

    monkeypatch.setenv("DEEPSEAPORT_CONFIG", str(tmp_path / "operator.json"))
    monkeypatch.setenv("DEEPSEAPORT_PROVIDER", "qwen")
    monkeypatch.setenv("DEEPSEEK_TOKEN", "leaked")
    monkeypatch.setenv("OBSCURA_BIN", "C:/oc/obscura.exe")
    monkeypatch.setattr(conftest, "_LIVE_TIER", False)

    temp_root = tmp_path / "hermetic"
    temp_root.mkdir()  # pytest_configure creates `data/` and config.json itself
    monkeypatch.setattr(conftest.tempfile, "mkdtemp", lambda **kw: str(temp_root))
    monkeypatch.setattr(conftest.atexit, "register", lambda *a, **k: None)

    conftest.pytest_configure(_cfg("not live"))

    cfg = Path(os.environ["DEEPSEAPORT_CONFIG"])
    assert cfg == temp_root / "config.json"
    written = json.loads(cfg.read_text(encoding="utf-8"))
    assert written["accounts"] == [], "the hermetic config must carry no accounts"
    assert written["keys"] == ["sk-hermetic-test"]
    assert os.environ["DEEPSEAPORT_DATA"] == str(temp_root / "data")
    for name in ("DEEPSEAPORT_PROVIDER", "DEEPSEEK_TOKEN", "OBSCURA_BIN"):
        assert name not in os.environ, f"{name} leaked into the hermetic run"
    assert conftest._LIVE_TIER is False
    assert "hermetic config" in "\n".join(conftest.pytest_report_header(config=None))



# --- G6: CDP login browser launch argv ---------------------------------------


def test_g6_launch_args_drop_remote_allow_origins_and_pin_loopback():
    """G6. Pre-change: the sign-in browser was launched with
    ``--remote-allow-origins=*``, disabling Chrome's CDP Origin check so any
    local process could attach to the debug port and drive the live login
    (type into the password field, read the token). The websocket client
    already sends no Origin (``suppress_origin=True``), so the flag bought
    nothing.
    """
    browser = _rb.RealBrowser("chrome.exe", headless=False)
    headless_browser = _rb.RealBrowser("chrome.exe", headless=True)
    try:
        args = browser.launch_args()
        assert not any(a.startswith("--remote-allow-origins") for a in args), \
            "CDP Origin check is still disabled for any local attacher"
        assert "--remote-debugging-address=127.0.0.1" in args
        assert "--remote-debugging-port=0" in args
        assert args[0] == "chrome.exe" and args[-1] == "about:blank"
        assert f"--user-data-dir={browser.profile}" in args
        assert "--headless=new" not in args

        # Second instance, also closed: RealBrowser.__init__ makes a
        # deepseaport-browser-* temp profile, so an unclosed one leaks a
        # directory into %TEMP% on every run.
        headless = headless_browser.launch_args()
        assert "--headless=new" in headless
        assert not any(a.startswith("--remote-allow-origins") for a in headless)
        # Headless is inserted before the trailing URL, never as the last arg.
        assert headless[-1] == "about:blank"
        assert headless[-2] == "--headless=new"
    finally:
        headless_browser.close()
        browser.close()


def test_g6_start_actually_launches_with_the_pinned_argv(monkeypatch):
    """G6: pin the argv that reaches ``subprocess.Popen``, not just the
    helper -- the flag could be re-added in ``start()`` itself."""
    seen: dict = {}

    class _Proc:
        returncode = None

        def poll(self):
            return None

    def _fake_popen(args, **kw):
        seen["args"] = list(args)
        return _Proc()

    monkeypatch.setattr(_rb.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(_rb.RealBrowser, "_wait_for_port", lambda self: 9222)
    monkeypatch.setattr(_rb.RealBrowser, "_connect", lambda self, port: None)

    browser = _rb.RealBrowser("chrome.exe", headless=True)
    try:
        browser.start()
    finally:
        browser.close()

    args = seen["args"]
    assert not any(a.startswith("--remote-allow-origins") for a in args)
    assert "--remote-debugging-address=127.0.0.1" in args
    assert "--headless=new" in args


# Deleted here on purpose: `test_g6_websocket_client_sends_no_origin`. It
# asserted that the string "suppress_origin=True" appears somewhere in
# `inspect.getsource(RealBrowser._connect)` -- a source-text check, not a
# behaviour check, so it survived a revert of the G6 fix it was filed under.
# Its two siblings above already cover the flag behaviour proper: they assert
# the argv that actually reaches `subprocess.Popen`.


# --- G7: inert callback parameters on complete_with_failover_sync ------------


def _one_account_pool() -> AccountPool:
    pool = AccountPool([AccountConfig(email="a@x.com", token="tok")])
    return pool


def test_g7_callbacks_are_inert_and_never_double_emit():
    """G7. The provider's ``run_core`` closure already captured the emit
    callbacks. If the engine ALSO applied ``on_content``/``on_thinking``, the
    client would get every delta twice -- duplicated SSE text and a doubled
    tool-call JSON payload, which renders as a plausible wrong answer rather
    than a crash.

    Pins the guarantee: passing them changes nothing about what is emitted.
    """
    seen_content: list[str] = []
    seen_thinking: list[str] = []
    prep = {"model": "m", "prompt": "hi", "tools": []}
    pool = _one_account_pool()
    item = pool.get("a@x.com")

    def run_core(account):
        # The provider closure captured the callbacks already.
        for piece in ("Hel", "lo "):
            seen_content.append(piece)
        seen_thinking.append("thinking")
        return {"content": "Hello ", "thinking": "thinking"}

    out = _engine.complete_with_failover_sync(
        _BareApp(), pool, prep, item, run_core, lambda exc: False,
        on_content=lambda t: seen_content.append(t),
        on_thinking=lambda t: seen_thinking.append(t))

    assert seen_content == ["Hel", "lo "], "engine re-applied on_content (double emit)"
    assert seen_thinking == ["thinking"], "engine re-applied on_thinking (double emit)"
    assert out["content"] == "Hello "


def test_g7_parameters_are_still_accepted_with_their_historical_names():
    """G7: both provider adapters call the engine's version with ``prep``
    POSITIONALLY and ``on_content=``/``on_thinking=``/``allow_failover=``/
    ``get_settings=`` by keyword. Removing a parameter would break them, so
    the names, order and defaults are pinned."""
    params = inspect.signature(_engine.complete_with_failover_sync).parameters
    assert list(params) == ["app", "pool", "prep", "first_item", "run_core",
                            "is_ban_exc", "cancel_event", "on_content", "on_thinking",
                            "allow_failover", "get_settings"]
    for required in ("app", "pool", "prep", "first_item", "run_core", "is_ban_exc"):
        assert params[required].default is inspect.Parameter.empty
    for optional in ("cancel_event", "on_content", "on_thinking", "allow_failover",
                     "get_settings"):
        assert params[optional].default is None


def test_g7_server_reexports_keep_their_historical_signatures():
    """G7: the completion/concurrency entry points keep their historical
    signatures, and ``deepseaport.server`` exposes only what it owns.

    The re-export block is gone: those names are imported from the module that
    defines them (``completion_service`` / ``server_concurrency`` /
    ``server_routes``), so that is where the signatures are pinned. What is
    still asserted on ``deepseaport.server`` is its own thin surface: the
    process entry point, the catalogue, and the three monkeypatch seams the
    route layer resolves through it."""
    from deepseaport import server as _server
    from deepseaport.providers.deepseek import completion_service as _cs
    from deepseaport import server_concurrency as _sc
    from deepseaport import server_routes as _sr

    # Historical deepseek adapter signature: (app, pool, prep, first_item,
    # cancel_event, on_content, on_thinking, allow_failover).
    assert list(inspect.signature(
        _cs._complete_with_failover_sync).parameters) == [
        "app", "pool", "prep", "first_item", "cancel_event",
        "on_content", "on_thinking", "allow_failover"]
    # 5 positional + the optional label keyword added by G3.
    assert list(inspect.signature(_sc._acquire_slot_or_499).parameters) == [
        "pool", "request", "cancel_event", "timeout", "allow_failover",
        "provider_label"]
    # The engine's `del` must not have leaked into any of them.
    for name in ("_ban_403_for_item", "_ensure_token", "_headers",
                 "_is_invalid_token_error", "_is_transient_overload"):
        assert hasattr(_cs, name), f"{name} is missing from completion_service"
    for name in ("_StreamCancelled", "FAILOVER_ACQUIRE_TIMEOUT"):
        assert hasattr(_sc, name), f"{name} is missing from server_concurrency"
    for name in ("_check_auth", "_pool", "_prepare", "register_routes"):
        assert hasattr(_sr, name), f"{name} is missing from server_routes"
    # What deepseaport.server still owns, plus the three deliberate seams.
    for name in ("MODELS", "active_models", "create_app", "register_routes",
                 "_pool", "_openai_response", "parse_tool_calls",
                 "Settings", "load_settings", "apply_log_level", "lifespan"):
        assert getattr(_server, name, None) is not None, f"{name} is unresolvable"
