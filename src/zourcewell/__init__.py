"""zourcewell: unofficial OpenAI-compatible web2api for DeepSeek, Qwen and opencode. Research/educational use only."""

import importlib.metadata
import sys
from pathlib import Path

__all__ = ["__version__"]


def _bundled_version():
    """The version ``zourcewell.spec`` stamped into a frozen exe, else None.

    ``collect_all('zourcewell')`` bundles the BUILD machine's dist-info too, so
    inside the exe ``importlib.metadata`` reports whatever the last
    ``pip install -e .`` wrote -- a number that drifts from the exe's own file
    version. The spec writes VERSION to a bundled ``_version.txt``; this is the
    only source that cannot disagree with the exe it ships in, so it wins.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    root = (Path(meipass) / "zourcewell" if meipass
            else Path(__file__).resolve().parent)
    try:
        text = (root / "_version.txt").read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return text or None


# pyproject.toml [project] version is the single source of truth: a frozen exe's
# bundled stamp first, then the file read directly. An editable install's
# dist-info is a PEP 660 snapshot taken at install time: it keeps reporting
# whatever the version was at the last `pip install -e .`, which on this checkout
# still said 0.1.0 long after pyproject moved on, so metadata-first would outvote
# the file it was derived from.
__version__ = _bundled_version()
if not __version__:
    try:
        import tomllib
        __version__ = tomllib.loads(
            (Path(__file__).resolve().parents[2] / "pyproject.toml")
            .read_text(encoding="utf-8")
        )["project"]["version"]
    except Exception:
        # No pyproject.toml here: an installed wheel, or the frozen exe, where
        # the file is not bundled at all. The installed metadata is the same
        # number for a wheel; inside the exe neither source exists, so it
        # settles on 0.0.0 rather than failing to import. requires-python is
        # >= 3.11, so a missing tomllib is impossible and the fallback is about
        # the FILE, not the module.
        try:
            __version__ = importlib.metadata.version("zourcewell")
        except Exception:
            __version__ = "0.0.0"