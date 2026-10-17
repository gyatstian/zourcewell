# -*- mode: python ; coding: utf-8 -*-
import tomllib
from pathlib import Path

from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.win32 import versioninfo

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


def _file_version(version: str) -> tuple:
    """VERSION as the 4-int tuple a Windows file-version resource needs.

    Zero-padded like WIN32 itself ("0.28" -> (0, 28, 0, 0)). A value that is
    not 1-4 dot-separated 16-bit numbers is a build failure, same rule as a
    missing pyproject.toml above: shipping an exe whose version resource
    silently disagrees with [project] version is worse than not shipping.
    """
    try:
        nums = [int(part) for part in version.split(".")]
    except ValueError:
        nums = []
    if not 1 <= len(nums) <= 4 or any(n < 0 or n > 0xFFFF for n in nums):
        raise SystemExit(
            f"deepseaport.spec: [project] version {version!r} in pyproject.toml "
            f"is not a Windows file version (1-4 dot-separated integers, "
            f"0-65535 each)")
    return tuple((nums + [0, 0, 0, 0])[:4])


def _version_resource(version: str):
    """The exe's version resource, built from the pyproject version string.

    EXE(version=...) takes a PATH to a version-resource TEXT FILE, not a
    version string: PyInstaller treats a str as a filename, so the old
    ``version=VERSION`` made ``pyinstaller deepseaport.spec`` raise
    FileNotFoundError: '0.28' at spec execution. A real VSVersionInfo object
    is accepted as-is, and constructing it here keeps the exe's FileVersion
    readable in Explorer / Get-Item and tied to the single VERSION above.
    """
    filevers = _file_version(version)
    return versioninfo.VSVersionInfo(
        ffi=versioninfo.FixedFileInfo(
            filevers=filevers,
            prodvers=filevers,
            # Standard defaults, spelled out: VOS_NT_WINDOWS32 | VFT_APP.
            mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0,
            date=(0, 0),
        ),
        kids=[
            versioninfo.StringFileInfo([
                # 040904B0 = US English, Unicode (matches the Translation
                # VarStruct below; the pair is what Windows shows).
                versioninfo.StringTable('040904B0', [
                    versioninfo.StringStruct('CompanyName', 'deepseaport'),
                    versioninfo.StringStruct('FileVersion', version),
                    versioninfo.StringStruct('OriginalFilename',
                                             'deepseaport.exe'),
                    versioninfo.StringStruct('ProductName', 'deepseaport'),
                    versioninfo.StringStruct('ProductVersion', version),
                ]),
            ]),
            versioninfo.VarFileInfo([
                versioninfo.VarStruct('Translation', [1033, 1200]),
            ]),
        ],
    )

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='deepseaport',
    # A VSVersionInfo object, never the VERSION string: PyInstaller reads a
    # str here as a path (version=VERSION raised FileNotFoundError, see
    # _version_resource above).
    version=_version_resource(VERSION),
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
