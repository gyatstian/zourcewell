"""Real Chromium-family browser login driver over CDP.

The Obscura no-render engine is excellent for WAF cookies, but DeepSeek's
sign-in page loads Shumei ``fp.min.js`` and only posts a real ``device_id``
when that SDK initializes. In Obscura the SDK never becomes ready, so the
login request carries ``device_id: null`` and DeepSeek rejects it. This module
drives a normal Helium/Chrome/Edge/Chromium browser instead, where the SDK
works exactly like it does in the user's main browser.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from ...core.browser_paths import resolve_browser
from .login_policy import (
    CDP_CLICK_LOGIN_JS,
    CDP_DISMISS_COOKIE_JS,
    CDP_SNAPSHOT_JS,
    COOKIE_DISMISS_DELAY_CDP,
    DEFAULT_LOGIN_TIMEOUT,
    NAVIGATE_RENDER_TIMEOUT,
    REACT_COMMIT_DELAY,
    REAL_POLL_INTERVAL,
    SIGN_IN_URL,
    clean_text as _clean,
    snapshot_verdict,
)

logger = logging.getLogger("deepseaport.real_browser")

# A single ban phrase in one frame is a HINT, not a verdict (see the login
# loop); the exits that cannot report a verdict surface it in the DETAIL.
_BAN_HINT_SUFFIX = " [ban wording seen in an earlier frame]"

# Page policy (selectors, phrase lists, visibility/snapshot JS, poll/timeout
# constants, verdict helpers) lives in ``login_policy``; the names imported
# above are the ones this driver actually uses. The login loop classifies each
# snapshot through ``snapshot_verdict`` (the shared helper the Obscura driver
# also uses) instead of re-deriving phrase matches locally.


class RealBrowser:
    """Minimal CDP client: enough to drive the DeepSeek sign-in form."""

    def __init__(self, binary: str, *, headless: bool = False,
                 timeout: float = DEFAULT_LOGIN_TIMEOUT):
        self.binary = binary
        self.headless = headless
        self.timeout = timeout
        self.profile = tempfile.mkdtemp(prefix="deepseaport-browser-")
        self.proc: subprocess.Popen | None = None
        self.ws = None
        self._id = 0

    def launch_args(self) -> list[str]:
        """Argv for the sign-in browser.

        The DevTools endpoint is pinned to loopback
        (``--remote-debugging-address=127.0.0.1``) so no other host on the LAN
        can attach. ``--remote-allow-origins=*`` is deliberately ABSENT: it
        turns Chrome's CDP Origin check off, which would let any local process
        attach to this debug port and drive the live sign-in session (typing
        into the password field, reading the resulting token). The websocket
        client already sends no ``Origin`` header at all
        (``suppress_origin=True`` in :meth:`_connect`), so the check the flag
        disables is not even hit by the legitimate client.

        Kept as a method (not an inline list in ``start``) so the security
        shape of the launch is unit-testable without spawning a browser.
        """
        args = [
            self.binary,
            "--remote-debugging-port=0",
            "--remote-debugging-address=127.0.0.1",
            f"--user-data-dir={self.profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-extensions",
            "--disable-background-networking",
            "--disable-sync",
            "--disable-blink-features=AutomationControlled",
            "--window-size=1200,900",
            "--window-position=50,50",
            "about:blank",
        ]
        if self.headless:
            args.insert(-1, "--headless=new")
        return args

    def start(self) -> None:
        try:
            import websocket  # noqa: F401  (dependency checked here)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "real-browser login needs websocket-client; "
                "install it with `pip install websocket-client`") from exc

        args = self.launch_args()
        creationflags = 0
        try:
            creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        except Exception:
            creationflags = 0
        logger.info("launching real browser login: %s", Path(self.binary).name)
        self.proc = subprocess.Popen(
            args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        port = self._wait_for_port()
        self._connect(port)

    def _wait_for_port(self) -> int:
        port_file = Path(self.profile) / "DevToolsActivePort"
        deadline = time.monotonic() + 25.0
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(f"browser exited with code {self.proc.returncode}")
            try:
                lines = port_file.read_text(encoding="utf-8").splitlines()
                if lines:
                    return int(lines[0].strip())
            except Exception:
                pass
            time.sleep(0.1)
        raise TimeoutError("browser did not expose a DevTools port")

    def _connect(self, port: int) -> None:
        import websocket

        deadline = time.monotonic() + 10.0
        targets = []
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/json/list", timeout=2) as resp:
                    targets = json.loads(resp.read().decode("utf-8"))
                if targets:
                    break
            except Exception:
                time.sleep(0.15)
        target = next((t for t in targets if t.get("type") == "page"), None)
        if not target:
            raise RuntimeError("browser exposed no page target")
        self.ws = websocket.create_connection(
            target["webSocketDebuggerUrl"], timeout=5, suppress_origin=True)
        self.ws.settimeout(0.2)

    def close(self) -> None:
        try:
            if self.ws is not None:
                self.ws.close()
        except Exception:
            pass
        self.ws = None
        try:
            if self.proc is not None and self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=3)
        except Exception:
            pass
        self.proc = None
        shutil.rmtree(self.profile, ignore_errors=True)

    def __enter__(self) -> "RealBrowser":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- CDP -----------------------------------------------------------
    def call(self, method: str, params: dict | None = None,
             timeout: float = 20.0):
        import websocket

        if self.ws is None:
            raise RuntimeError("browser CDP connection is not open")
        self._id += 1
        mid = self._id
        self.ws.send(json.dumps(
            {"id": mid, "method": method, "params": params or {}}))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                raw = self.ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"CDP {method}: {msg['error']}")
                return msg.get("result")
        raise TimeoutError(f"CDP {method} timed out")

    def evaluate(self, expression: str, timeout: float = 20.0):
        result = self.call("Runtime.evaluate", {
            "expression": expression,
            "returnByValue": True,
            "awaitPromise": False,
        }, timeout=timeout)
        if result.get("exceptionDetails"):
            raise RuntimeError(str(result["exceptionDetails"].get("text", "JS exception")))
        return result.get("result", {}).get("value")

    def navigate(self, url: str) -> None:
        self.call("Page.enable")
        self.call("Runtime.enable")
        self.call("Page.navigate", {"url": url}, timeout=NAVIGATE_RENDER_TIMEOUT)
        deadline = time.monotonic() + NAVIGATE_RENDER_TIMEOUT
        while time.monotonic() < deadline:
            try:
                ready = self.evaluate("document.readyState")
                inputs = self.evaluate("document.querySelectorAll('input').length")
                if str(ready) == "complete" and int(inputs or 0) >= 2:
                    return
            except Exception:
                pass
            time.sleep(0.4)
        raise TimeoutError("sign-in form did not render")

    # -- page actions --------------------------------------------------
    def dismiss_cookie_banner(self) -> None:
        try:
            self.evaluate(CDP_DISMISS_COOKIE_JS)
            time.sleep(COOKIE_DISMISS_DELAY_CDP)
        except Exception:
            pass

    def fill_credentials(self, email: str, password: str) -> None:
        js = (
            "(function(email,password){"
            "function setv(el,v){var d=Object.getOwnPropertyDescriptor("
            "window.HTMLInputElement.prototype,'value');"
            "d.set.call(el,v);"
            "el.dispatchEvent(new Event('input',{bubbles:true}));"
            "el.dispatchEvent(new Event('change',{bubbles:true}));}"
            "var inputs=[].slice.call(document.querySelectorAll('input'));"
            "var e=inputs.find(function(i){return i.type==='text'||i.type==='email';});"
            "var p=inputs.find(function(i){return i.type==='password';});"
            "if(!e||!p) return 'NOFIELDS';"
            "setv(e,email); setv(p,password); return 'FILLED';"
            "})(%s,%s)" % (json.dumps(email), json.dumps(password))
        )
        value = self.evaluate(js)
        if value != "FILLED":
            raise RuntimeError("could not find email/password inputs")

    def click_login(self) -> None:
        value = self.evaluate(CDP_CLICK_LOGIN_JS)
        if value != "CLICKED":
            raise RuntimeError("could not find the login button")

    def snapshot(self) -> dict:
        raw = self.evaluate(CDP_SNAPSHOT_JS)
        try:
            return json.loads(raw)
        except Exception:
            return {}

    # -- login flow ----------------------------------------------------
    def login(self, email: str, password: str) -> tuple[str | None, bool, str]:
        self.navigate(SIGN_IN_URL)
        self.dismiss_cookie_banner()
        try:
            self.evaluate("localStorage.removeItem('userToken'); 'cleared'")
        except Exception:
            pass
        self.fill_credentials(email, password)
        time.sleep(REACT_COMMIT_DELAY)
        self.click_login()

        deadline = time.monotonic() + self.timeout
        detail = ""
        ban_hint = False
        ban_streak = 0
        while time.monotonic() < deadline:
            time.sleep(REAL_POLL_INTERVAL)
            try:
                snap = self.snapshot()
            except Exception as exc:
                detail = str(exc)
                continue
            if not snap:
                continue
            body = _clean(snap.get("body"))
            detail = body[:800]
            # Page semantics come from login_policy (single source of truth for
            # both drivers) via snapshot_verdict: the captcha_visible overlay
            # flag AND the LOGIN_CAPTCHA_PHRASES list. Checking only the flag
            # meant a text-only challenge ("Slide to verify", no overlay
            # element) was never detected: the driver burned the full 75s login
            # timeout and returned "no token", which sets the wrong (generic)
            # login backoff instead of a captcha one.
            kind, phrase = snapshot_verdict(snap)
            token = str(snap.get("token") or "").strip()
            if token == "null":
                token = ""
            if kind == "captcha":
                return None, False, (
                    f"real browser login captcha (phrase={phrase!r}): "
                    f"{body[:300]}" + (_BAN_HINT_SUFFIX if ban_hint else ""))
            if kind == "ban":
                # One sighting is a HINT, not a verdict -- same rule as the MCP
                # driver (auth.py:264-270, ban_streak >= 2). LOGIN_BAN_PHRASES
                # holds bare tokens ("muted", "banned", "suspend", "violation")
                # and this loop takes ~125 snapshots over 75s, so latching on a
                # single frame (a terms block, a footer, a toast, a route
                # transition) permanently bans a healthy account:
                # completion_service.py:353-362 throws the freshly obtained
                # token away and _refresh_token_once
                # persists `banned: true` to config.json, with a cooldown of up
                # to a year when parse_ban_until finds a date in the body. A
                # persisted false ban is near-unrecoverable from the user's
                # side, so confirmation is required.
                ban_streak += 1
                # Sticky hint for the exits below that cannot report a verdict
                # of their own (captcha/credential/no token): they append it to
                # the DETAIL, never to the banned slot -- the user still
                # deserves to hear that ban wording was seen.
                ban_hint = True
                if ban_streak >= 2:
                    # Confirmed. Still hand back a token the page is serving: a
                    # suspended account can log in (see parse_ban_until) and the
                    # caller decides what to do with the pair.
                    return (token or None), True, (
                        f"real browser login suspension (phrase={phrase!r}): "
                        f"{body[:400]}")
                # One sighting: keep polling rather than returning. A transient
                # frame clears on the next poll; a real one confirms above.
                continue
            ban_streak = 0
            if token:
                # A page serving a real userToken is positive evidence the
                # account is NOT banned, so this path never reports banned=True:
                # an ambiguous earlier frame must not veto a working credential.
                return token, False, body[:800]
            if kind == "credential":
                return None, False, (
                    f"real browser login rejected (phrase={phrase!r}): "
                    f"{body[:400]}" + (_BAN_HINT_SUFFIX if ban_hint else ""))
        return None, False, (
            f"real browser login no token: {detail[:600]}"
            + (_BAN_HINT_SUFFIX if ban_hint else ""))


def login_with_real_browser(email: str, password: str,
                            settings) -> tuple[str | None, bool, str] | None:
    """Try the configured real browser. Returns None when no browser is set."""
    binary = resolve_browser(getattr(settings, "browser_bin", "") or "")
    if not binary:
        return None
    headless = bool(getattr(settings, "browser_headless", False))
    browser = RealBrowser(binary, headless=headless)
    try:
        browser.start()
        return browser.login(email, password)
    except Exception as exc:
        logger.warning("real browser login failed for %s: %s", email, exc)
        return None, False, f"real browser login failed: {exc}"
    finally:
        browser.close()
