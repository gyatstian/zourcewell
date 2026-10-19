"""Browser discovery must never hand back a CWD-relative binary.

``Path(os.environ.get("LOCALAPPDATA", ""))`` is ``Path("")`` == ``Path(".")``
when the variable is unset, so every Windows candidate was built relative to
the working directory. Reproduced by planting
``Google/Chrome/Application/chrome.exe`` in the cwd with LOCALAPPDATA /
PROGRAMFILES unset: ``discover_browser()`` returned exactly that relative path,
and the real-browser login then types the account's email and password into it.

The fix is ``_anchored()`` (the path does not resolve against the CWD) in
``_candidates()``, in ``discover_browser()`` -- for the candidate list and for
a ``shutil.which()`` hit (``PATH=.`` makes which() answer a relative path too)
-- and in ``resolve_browser()``, whose configured NAME also goes through PATH.
Windows candidates stay as they were, and the POSIX + home-install entries are
unaffected (``Path.is_absolute`` would have dropped the rooted ``/usr/bin/*``
entries on Windows).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import deepseaport.core.browser_paths as bp


def _windows_env_unset(monkeypatch) -> None:
    for var in ("LOCALAPPDATA", "PROGRAMFILES", "PROGRAMFILES(X86)"):
        monkeypatch.delenv(var, raising=False)


def _plant_relative_candidate(tmp_path: Path) -> Path:
    """A binary at ``./Google/Chrome/Application/chrome.exe``: the pre-fix hit."""
    exe = tmp_path / "Google" / "Chrome" / "Application" / "chrome.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"MZ")  # is_file() is all discovery checks
    return exe


def test_no_candidate_is_relative_when_the_env_base_is_unset(monkeypatch, tmp_path):
    _windows_env_unset(monkeypatch)
    monkeypatch.chdir(tmp_path)
    _plant_relative_candidate(tmp_path)

    candidates = bp._candidates()

    assert all(bp._anchored(c) for c in candidates), candidates
    # None of the env-derived (Windows) entries survives an unset base.
    for name in ("imput", "Helium", "Google", "Microsoft", "Chromium",
                 "BraveSoftware"):
        assert not any(c.startswith(name) for c in candidates), candidates


def test_planted_cwd_binary_is_not_discovered(monkeypatch, tmp_path):
    """The end-to-end shape of the reported hole: env emptied, which() finding
    nothing, a browser planted in the working directory."""
    _windows_env_unset(monkeypatch)
    monkeypatch.chdir(tmp_path)
    planted = _plant_relative_candidate(tmp_path)
    monkeypatch.setattr(shutil, "which", lambda name: None)

    found = bp.discover_browser()

    # A real browser installed on this machine is a legitimate hit (CI images
    # ship /usr/bin/google-chrome), so the property is "never CWD-relative,
    # never the planted file" -- not "always empty".
    if found:
        assert bp._anchored(found), found
        assert Path(found).resolve() != planted.resolve()
    else:
        assert found == ""


def test_relative_candidate_is_skipped(monkeypatch, tmp_path):
    monkeypatch.setattr(bp.shutil, "which", lambda name: None)
    monkeypatch.setattr(bp, "_candidates",
                        lambda: ["Google/Chrome/Application/chrome.exe"])
    monkeypatch.chdir(tmp_path)
    _plant_relative_candidate(tmp_path)

    assert bp.discover_browser() == ""


def test_which_hit_must_be_absolute(monkeypatch):
    """``PATH=.`` makes shutil.which() answer a bare relative name."""
    monkeypatch.setattr(bp.shutil, "which",
                        lambda name: "chrome" if name == "chrome" else None)

    found = bp.discover_browser()

    assert found != "chrome"
    if found:
        assert bp._anchored(found), found
    else:
        assert found == ""


@pytest.mark.parametrize("relative", ["c:chrome.exe", "C:chrome.exe", "c:"])
def test_bare_drive_is_not_anchored(relative):
    """On Windows ``Path("c:x").anchor`` is ``"c:"``, but the path is still
    drive-RELATIVE: it resolves against the working directory on that drive."""
    assert not bp._anchored(relative)
    assert bp._anchored("/usr/bin/chrome")


def test_which_hit_that_is_drive_relative_is_rejected(monkeypatch):
    """PATH can hold drive-relative entries (``c:bin``); which() then answers
    ``c:bin\\chrome.exe``, which anchor alone used to accept."""
    monkeypatch.setattr(bp.shutil, "which",
                        lambda name: "c:chrome.exe" if name == "chrome" else None)

    assert bp.resolve_browser("chrome") == ""
    found = bp.discover_browser()
    assert found != "c:chrome.exe"
    if found:
        assert bp._anchored(found), found


def test_configured_bare_name_does_not_resolve_through_the_cwd(
        monkeypatch, tmp_path):
    """``browser_bin: chrome.exe`` means a PATH lookup, but the is_file()
    shortcut used to return a planted ``./chrome.exe`` -- the same
    planted-binary login hazard the PATH guard already rejects."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "chrome.exe").write_bytes(b"MZ")
    monkeypatch.setattr(bp.shutil, "which", lambda name: None)

    assert bp.resolve_browser("chrome.exe") == ""


def test_resolve_browser_rejects_a_relative_path_hit(monkeypatch, tmp_path):
    """A configured NAME is looked up in PATH, which may hold a relative entry."""
    monkeypatch.chdir(tmp_path)
    _plant_relative_candidate(tmp_path)
    monkeypatch.setattr(bp.shutil, "which", lambda name: "Google/Chrome/Application/chrome.exe")

    assert bp.resolve_browser("chrome") == ""

    monkeypatch.setattr(bp.shutil, "which", lambda name: "/usr/bin/chrome")
    assert bp.resolve_browser("chrome") == "/usr/bin/chrome"


def test_windows_and_posix_candidates_still_work(monkeypatch, tmp_path):
    local = tmp_path / "AppData" / "Local"
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    monkeypatch.delenv("PROGRAMFILES", raising=False)
    monkeypatch.delenv("PROGRAMFILES(X86)", raising=False)

    candidates = bp._candidates()

    assert str(local / "Google" / "Chrome" / "Application" / "chrome.exe") in candidates
    assert str(local / "imput" / "Helium" / "Application" / "chrome.exe") in candidates
    assert str(local / "BraveSoftware" / "Brave-Browser" / "Application" / "brave.exe") in candidates
    assert "/usr/bin/google-chrome" in candidates
    assert "/usr/bin/microsoft-edge" in candidates
    assert str(Path.home() / ".local" / "share" / "helium" / "chrome") in candidates
    assert all(bp._anchored(c) for c in candidates)


def test_empty_env_base_is_not_a_relative_base(monkeypatch, tmp_path):
    """A blank (not just absent) LOCALAPPDATA is the same trap."""
    monkeypatch.setenv("LOCALAPPDATA", "")
    monkeypatch.setenv("PROGRAMFILES", "   ")
    monkeypatch.chdir(tmp_path)
    _plant_relative_candidate(tmp_path)

    assert all(bp._anchored(c) for c in bp._candidates())
