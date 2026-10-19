"""A non-frozen install with no source checkout must not write into ``Lib/``.

``_app_dir()`` was ``PACKAGE_DIR.parent.parent`` unconditionally: the repo root
in a checkout, but the interpreter's ``Lib/`` for a pip-installed wheel. So
``pip install deepseaport`` plus ``login`` from a config-less cwd wrote the
live credential store (account emails, plaintext passwords, 64-char bearer
tokens) into a stdlib directory -- admin-only on a system Python, and wiped by
a venv rebuild. ``_default_config_path()`` and the import-time ``DATA_DIR``
default both resolve through it.

The fallback is a per-user dir (%APPDATA%\\deepseaport, else
$XDG_CONFIG_HOME/deepseaport, else ~/.config/deepseaport) and a released exe is
unchanged (config.json next to the binary).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from deepseaport import config as config_mod
from deepseaport.config import _app_dir, _default_config_path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _wheel_package_dir(monkeypatch, tmp_path: Path) -> Path:
    """PACKAGE_DIR as a wheel install sees it: <Lib>/site-packages/deepseaport.

    The per-user env bases are pointed at tmp_path too, so the expected value
    never depends on the machine or the developer's real profile.
    """
    pkg = tmp_path / "Lib" / "site-packages" / "deepseaport"
    pkg.mkdir(parents=True)
    monkeypatch.setattr(config_mod, "PACKAGE_DIR", pkg)
    monkeypatch.setenv("APPDATA", str(tmp_path / "user"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "user"))
    monkeypatch.delenv("DEEPSEAPORT_CONFIG", raising=False)
    # No ./config.json in cwd, so _default_config_path() reaches its fallback.
    monkeypatch.chdir(tmp_path)
    return pkg


def test_wheel_install_uses_a_per_user_dir(monkeypatch, tmp_path):
    pkg = _wheel_package_dir(monkeypatch, tmp_path)

    app_dir = _app_dir()

    assert app_dir == tmp_path / "user" / "deepseaport"
    # The bug: PACKAGE_DIR.parent.parent == <Lib>/site-packages/..
    assert app_dir != pkg.parent.parent, "must not be the interpreter's Lib/"
    assert _default_config_path() == app_dir / "config.json"


def test_source_checkout_still_wins(monkeypatch, tmp_path):
    """``pyproject.toml`` next to it is what makes a tree a checkout."""
    root = tmp_path / "checkout"
    pkg = root / "src" / "deepseaport"
    pkg.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'deepseaport'\n",
                                         encoding="utf-8")
    monkeypatch.setattr(config_mod, "PACKAGE_DIR", pkg)
    monkeypatch.setenv("APPDATA", str(tmp_path / "user"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "user"))

    assert _app_dir() == root


def test_this_checkout_resolves_to_its_own_root():
    """The tree this suite runs from is a checkout, so nothing above changes
    where it reads config.json from (conftest already asserts the source tree)."""
    assert _app_dir() == REPO_ROOT
    assert config_mod.REPO_DIR == REPO_ROOT


def test_frozen_exe_is_unchanged(monkeypatch, tmp_path):
    """A shipped exe keeps config.json next to the binary -- dist/ doubles as
    the frozen app's credential store."""
    exe = tmp_path / "bin" / "deepseaport.exe"
    exe.parent.mkdir(parents=True)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    monkeypatch.setenv("APPDATA", str(tmp_path / "user"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "user"))

    assert _app_dir() == exe.parent


def test_obscura_repo_root_follows_the_app_dir(monkeypatch, tmp_path):
    """obscura_bridge probes for a dropped-in binary next to config.json, so its
    base must be ``_app_dir()``: a wheel used to leave it at the interpreter's
    ``Lib/`` (``Path(__file__).parents[3]``), where no config.json can be."""
    from deepseaport.core import obscura_bridge as ob

    _wheel_package_dir(monkeypatch, tmp_path)

    assert ob._repo_root() == tmp_path / "user" / "deepseaport"
    assert ob._repo_root() == _app_dir()


def test_obscura_repo_root_is_the_checkout_root_here():
    """The live value (a module-level constant) follows the same rule."""
    from deepseaport.core import obscura_bridge as ob

    assert ob.REPO_ROOT == REPO_ROOT == _app_dir()


@pytest.mark.parametrize("base_env", ["APPDATA", "XDG_CONFIG_HOME"])
def test_per_user_fallback_needs_no_env_base(base_env, monkeypatch, tmp_path):
    """A missing %APPDATA% / XDG_CONFIG_HOME is not a reason to fall back to a
    relative path (or to Lib/)."""
    _wheel_package_dir(monkeypatch, tmp_path)
    monkeypatch.delenv(base_env, raising=False)

    assert _app_dir().is_absolute()
