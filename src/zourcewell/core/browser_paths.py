"""Locate a real Chromium-family browser for login automation.

Obscura's no-render engine cannot initialize Shumei's `fp.min.js` SDK, so the
sign-in page posts ``device_id: null`` and DeepSeek rejects the login.  A real
Chromium/Helium/Chrome/Edge browser can generate the fingerprint normally.
This module only discovers a binary; the CDP driver lives in real_browser.py.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def _anchored(path: str) -> bool:
    """True when ``path`` resolves independently of the working directory.

    Deliberately not ``Path.is_absolute()``: on Windows that is False for a
    rooted, driveless path (``/usr/bin/google-chrome``, one of the candidates
    below), while the question here is only whether the working directory can
    redirect it. ``Path("") == Path(".")`` has an empty anchor -- the dangerous
    shape, produced by an unset LOCALAPPDATA/PROGRAMFILES.

    A bare Windows drive (``c:chrome.exe``, anchor ``"c:"``) is still
    drive-RELATIVE: it resolves against the working directory on that drive,
    so it is rejected too.
    """
    anchor = Path(path).anchor
    if not anchor:
        return False
    return not (len(anchor) == 2 and anchor[1] == ":")


def _candidates() -> list[str]:
    out: list[str] = []
    home = Path.home()
    local = Path(os.environ.get("LOCALAPPDATA", ""))
    pf = Path(os.environ.get("PROGRAMFILES", ""))
    pf86 = Path(os.environ.get("PROGRAMFILES(X86)", ""))
    # Helium (imput build) is what many DeepSeek users run as their main browser.
    out += [
        str(local / "imput" / "Helium" / "Application" / "chrome.exe"),
        str(local / "Helium" / "Application" / "chrome.exe"),
    ]
    # Google Chrome.
    out += [
        str(local / "Google" / "Chrome" / "Application" / "chrome.exe"),
        str(pf / "Google" / "Chrome" / "Application" / "chrome.exe"),
        str(pf86 / "Google" / "Chrome" / "Application" / "chrome.exe"),
    ]
    # Microsoft Edge.
    out += [
        str(local / "Microsoft" / "Edge" / "Application" / "msedge.exe"),
        str(pf / "Microsoft" / "Edge" / "Application" / "msedge.exe"),
        str(pf86 / "Microsoft" / "Edge" / "Application" / "msedge.exe"),
    ]
    # Chromium / Brave.
    out += [
        str(local / "Chromium" / "Application" / "chrome.exe"),
        str(local / "BraveSoftware" / "Brave-Browser" / "Application" / "brave.exe"),
        str(home / ".local" / "share" / "helium" / "chrome"),
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/usr/bin/microsoft-edge",
    ]
    # An unset LOCALAPPDATA/PROGRAMFILES makes Path("") == Path("."), so the
    # candidates built from it are resolved against the CWD -- and
    # discover_browser would then launch a binary planted in the working
    # directory, which the login types the account's credentials into.
    return [p for p in out if p and _anchored(p)]


def discover_browser() -> str:
    """Return a usable Chromium-family browser path, or an empty string."""
    for command in ("helium", "chrome", "google-chrome", "chromium",
                    "chromium-browser", "msedge", "brave"):
        found = shutil.which(command)
        # which() returns a CWD-relative path when PATH itself holds one (a
        # bare "." entry); same planted-binary hazard as _candidates().
        if found and _anchored(found):
            return found
    for raw in _candidates():
        try:
            if raw and _anchored(raw) and Path(raw).is_file():
                return raw
        except Exception:
            continue
    return ""


def resolve_browser(value: str) -> str:
    """Resolve a configured browser path / ``auto`` / empty value."""
    value = (value or "").strip()
    if not value:
        return ""
    if value.lower() in ("auto", "default"):
        return discover_browser()
    try:
        # A bare name (no separator) must not win through the CWD either:
        # `chrome.exe` planted in the working directory is the same hazard as
        # a relative PATH hit, and the owner passes "chrome.exe" meaning a
        # PATH lookup. Absolute paths (and rooted /usr/bin-style ones) are the
        # operator's own choice and still win.
        if _anchored(value) and Path(value).is_file():
            return value
    except Exception:
        pass
    # A configured NAME goes through PATH, which may hold a relative entry.
    found = shutil.which(value)
    return found if found and _anchored(found) else ""
