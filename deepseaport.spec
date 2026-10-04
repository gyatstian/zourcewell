# -*- mode: python ; coding: utf-8 -*-
import tomllib
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

# Single source of truth for the frozen version: [project] version in
# pyproject.toml, read at build time with stdlib tomllib (3.11+;
# requires-python is >= 3.11, so tomllib is always there). This used to be
# wrapped in `except Exception: VERSION = "0.0.0"`, which meant a build against a
# moved or half-written pyproject.toml shipped an exe whose version resource
# silently read 0.0.0 -- no build failure, and the exe name carries no version to
# notice it by. A build input that cannot be read is now a build failure.
VERSION = tomllib.loads(
    (Path(SPECPATH) / "pyproject.toml").read_text(encoding="utf-8")
)["project"]["version"]

datas = []
binaries = []
hiddenimports = []
# Order matters: it is the order PyInstaller sees these in the Analysis input.
for _pkg in ('curl_cffi', 'wasmtime', 'deepseaport', 'websocket'):
    _tmp_ret = collect_all(_pkg)
    datas += _tmp_ret[0]
    binaries += _tmp_ret[1]
    hiddenimports += _tmp_ret[2]


a = Analysis(
    ['entry.py'],
    pathex=['src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    # optimize=0 pins the bytecode level to 0. PyInstaller's own default is -1,
    # which means "inherit the build process's -O", so dropping this would make
    # the shipped exe's contents depend on how the builder invoked python.
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='deepseaport',
    version=VERSION,
    # No UPX: it is not installed on this machine (and GitHub runners do not
    # ship it), and PyInstaller 6 only warns ("UPX not available") instead of
    # failing. Compressing the bundled curl_cffi/wasmtime binaries has also
    # been known to break signed/AV-scanned loads, so the exe ships uncompressed.
    upx=False,
    # This is a terminal app: the TUI and the chat REPL both need a console.
    # Spelled out rather than left to PyInstaller's default so a tool bump can
    # never quietly turn the shipped exe into a windowed GUI binary.
    console=True,
)
