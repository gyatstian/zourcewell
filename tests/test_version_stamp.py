"""The frozen exe's version must come from the spec, not from a stale dist-info.

``collect_all('zourcewell')`` in ``zourcewell.spec`` bundles the build
machine's dist-info as well, so inside the exe ``importlib.metadata`` reports
whatever the last ``pip install -e .`` wrote -- /health, /v1/models and the
OpenAPI banner then disagree with the exe's own file version. The spec writes
VERSION to a bundled ``_version.txt`` and ``__init__`` reads it first; these
tests pin both halves of that contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

import zourcewell

_ROOT = Path(__file__).resolve().parents[1]


def _write_stamp(tmp_path: Path, text: str) -> Path:
    """A bundled ``_version.txt`` under the datas destination ('zourcewell')."""
    root = tmp_path / "zourcewell"
    root.mkdir()
    (root / "_version.txt").write_text(text, encoding="utf-8")
    return root


def test_bundled_version_reads_the_meipass_layout(monkeypatch, tmp_path):
    """One-file and one-dir builds both set ``sys._MEIPASS``, and the datas
    destination is ``zourcewell``, so the file sits at
    ``<_MEIPASS>/zourcewell/_version.txt``."""
    _write_stamp(tmp_path, "9.9.9")
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

    assert zourcewell._bundled_version() == "9.9.9"


def test_bundled_version_is_stripped(monkeypatch, tmp_path):
    """A trailing newline from an editor must not reach /health's version."""
    _write_stamp(tmp_path, "8.8.8\n")
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

    assert zourcewell._bundled_version() == "8.8.8"


def test_bundled_version_absent_is_none(monkeypatch):
    """No stamp next to the source package (a checkout, a wheel, CI): the
    pyproject -> metadata -> "0.0.0" order has to stay in charge."""
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)

    assert zourcewell._bundled_version() is None


def test_bundled_version_blank_content_is_none(monkeypatch, tmp_path):
    _write_stamp(tmp_path, "  \n")
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

    assert zourcewell._bundled_version() is None


def test_spec_writes_and_bundles_the_version_stamp(tmp_path):
    """Exec the spec's stamp wiring out of the spec source rather than
    reimplementing it here, so a spec that stops writing the file -- or stops
    bundling it under ``zourcewell`` -- fails this test instead of silently
    shipping an exe that reports the build machine's dist-info version."""
    src = (_ROOT / "zourcewell.spec").read_text(encoding="utf-8")
    start = src.index("_version_file = Path(SPECPATH)")
    end = src.index("datas.append((str(_version_file)", start)
    block = src[start:src.index("\n", end) + 1]
    ns = {"Path": Path, "SPECPATH": str(tmp_path), "datas": [], "VERSION": "8.8.8"}

    exec(block, ns)

    assert (tmp_path / "build" / "_version.txt").read_text(encoding="utf-8") == "8.8.8"
    # The dest must match the layout __init__ reads (see the _MEIPASS test above).
    assert ns["datas"] == [(str(tmp_path / "build" / "_version.txt"), "zourcewell")]
