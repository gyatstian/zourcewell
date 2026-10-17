"""Shared test doubles and factories, collected out of 51 test files that each
carried a private copy of the same handful of helpers.

Import directly (``from support import settings, FakeBridge``); these are plain
functions, not fixtures. See ``tests/AGENTS.md`` for what the suite guarantees.
"""

from __future__ import annotations

import json

from deepseaport.accounts import AccountConfig
from deepseaport.config import Settings
from deepseaport.server import create_app

__all__ = [
    "AccountConfig",
    "AliveThread",
    "DeadThread",
    "DeadThreadNoStart",
    "FakeBridge",
    "FakeUvicornConfig",
    "FakeUvicornServer",
    "JoinCountingThread",
    "Settings",
    "app_with_bridge",
    "cached_wasm",
    "collect_sse",
    "on_disk",
    "settings",
    "sse_deltas",
    "sse_payloads",
    "sse_text",
    "write_config",
]


def settings(**kw) -> Settings:
    """Hermetic ``Settings``: no keys, no accounts, no config file, no warmup.

    The base dict is a literal INSIDE this function on purpose. A module-level
    ``BASE = {...,"accounts": []}`` hands every caller the same list object, so
    one test's ``settings().accounts.append(...)`` poisons the next one -- and
    ``Settings`` uses ``field(default_factory=list)``, which an explicit
    ``accounts=[]`` silently overrides. Fresh values per call, always.

    Anything else a file needs (a provider, an account, ``max_retries=0``)
    belongs in ``kw``, not in another private ``_settings``.
    """
    base = {
        "keys": [],
        "accounts": [],
        "config_path": "",
        "log_level": "WARNING",
        "warmup_on_startup": False,
    }
    base.update(kw)
    return Settings(**base)


def write_config(tmp_path, payload: dict, name: str = "config.json") -> str:
    """Write a config file and return its path as a str."""
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def on_disk(path) -> dict:
    """The parsed JSON of a config file on disk."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --- SSE ---------------------------------------------------------------------
# One collector for all three providers: pass the module that owns
# ``_stream_completion`` (deepseek / qwen / opencode ``completion_service``).
# It stays ``async`` on purpose -- opencode's copy wrapped ``asyncio.run``
# internally, which cannot be called from inside a running loop, and
# test_qwen_stream_parity.py awaits this from one.


async def collect_sse(mod, app, pool, item, prep) -> list[bytes]:
    """Drain one ``_stream_completion`` run into its raw SSE frames."""
    return [f async for f in mod._stream_completion(app, pool, item, prep)]


def sse_text(frames) -> str:
    """Decode and join SSE frames into a body string."""
    return "".join(f.decode("utf-8", "replace") for f in frames)


def sse_payloads(body: str) -> list[dict]:
    """The JSON payloads of an SSE body, ``[DONE]`` sentinel excluded."""
    return [json.loads(line[len("data: "):]) for line in body.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"]


def sse_deltas(body: str) -> list[dict]:
    """The non-empty ``choices[].delta`` objects of an SSE body, in order.

    ``sse_payloads`` returns the whole payload; this flattens to the deltas.
    Which one a test wants is the real difference between the qwen and
    opencode helpers that collapsed into these two.
    """
    out: list[dict] = []
    for payload in sse_payloads(body):
        for choice in payload.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta:
                out.append(delta)
    return out


# --- obscura bridge ----------------------------------------------------------


class FakeWafState:
    """Browser-state stand-in: the UA and cookie jar the request headers read.

    ``cookies`` is per-instance for the same reason ``settings()`` rebuilds its
    lists: a class-level ``{}`` is one dict shared by every test that ever
    writes to it.
    """

    def __init__(self, user_agent: str = "test-ua") -> None:
        self.user_agent = user_agent
        self.cookies: dict = {}


class FakeBridge:
    """``obscura_bridge`` stand-in for the ``app.state.bridge`` seam.

    ``cookies`` is a load-bearing parameter, not decoration: the DeepSeek
    header path is only exercised when a test hands a real cookie string in.
    ``has_token`` flips ``has_waf_token()`` for the paths that skip the warmup
    challenge fetch when a token is already cached.
    """

    def __init__(self, cookies: str = "", has_token: bool = False) -> None:
        self.state = FakeWafState()
        self._cookies = cookies
        self._has_token = has_token

    def cookie_header(self) -> str:
        return self._cookies

    def has_waf_token(self) -> bool:
        return self._has_token

    def warmup(self) -> dict:
        return {}


def app_with_bridge(settings: Settings, bridge=None):
    """An app with a fake bridge already on ``app.state``."""
    app = create_app(settings)
    app.state.bridge = bridge if bridge is not None else FakeBridge()
    return app


def cached_wasm(monkeypatch, tmp_path):
    """Point PoW at an existing >1000 byte file so ``ensure_wasm`` is skipped."""
    wp = tmp_path / "wasm.bin"
    wp.write_bytes(b"x" * 2000)
    monkeypatch.setattr("deepseaport.providers.deepseek.pow.wasm_path", lambda: wp)
    return wp


# --- uvicorn + threading stand-ins -------------------------------------------


class FakeUvicornConfig:
    # ponytail: a stand-in, not a fake uvicorn -- it swallows construction
    # entirely and implements nothing else. Grow it if the code under test ever
    # reads a Config attribute; until then, a real uvicorn.Config in a test is
    # cheaper than keeping this in sync.
    def __init__(self, *a, **kw):
        pass


class FakeUvicornServer:
    def __init__(self, cfg=None):
        self.should_exit = False
        self.config = cfg

    def run(self):
        pass


class AliveThread:
    """``threading.Thread`` stand-in: alive until ``join()``, flags ``started``.

    Strategy "alive by flag" (test_chatui_gaps.py): the code under test
    starts the thread, reads ``is_alive()`` to decide whether the server came
    up, and ``join()`` on the way out.
    """

    def __init__(self, target=None, daemon=None, **_kw):
        self._target = target
        self.daemon = daemon
        self.started = False
        self._alive = True

    def start(self):
        self.started = True

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        self._alive = False


class DeadThread(AliveThread):
    """Strategy "dead by ``is_alive()``" (test_chatui_gaps.py).

    ``is_alive`` is hard-wired False while ``_alive`` stays True. Not
    interchangeable with ``DeadThreadNoStart``: this one models a thread that
    ran and exited, so anything reading ``_alive`` directly sees a live object.
    """

    def is_alive(self):
        return False


class DeadThreadNoStart(AliveThread):
    """Strategy "dead by ``start``" (test_cli_ux_hardening.py).

    The watcher/serve call never came up, so ``start()`` is a no-op that must
    NOT revive the thread: ``is_alive()`` has to read False before *and* after
    the start, which a base-class ``start()`` would let through if it ever grew
    one.
    """

    def __init__(self, target=None, daemon=None, **_kw):
        super().__init__(target=target, daemon=daemon)
        self._alive = False

    def start(self):
        pass


class JoinCountingThread(AliveThread):
    """Strategy "join-counting spy" (test_interactive_robustness.py).

    An assertion spy, not a fake: the test asserts ``joined == 1`` to pin that
    a Ctrl+C cleanup joined the thread EXACTLY once. ``join()`` deliberately
    does not call ``super()`` and so never clears ``_alive`` -- liveness is not
    what these tests are about, the count is.
    """

    def __init__(self, target=None, daemon=None, **_kw):
        super().__init__(target=target, daemon=daemon)
        self.joined = 0

    def join(self, timeout=None):
        self.joined += 1