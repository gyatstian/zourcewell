"""deepseaport: Obscura-backed DeepSeek web2api with OpenAI-compatible API."""

import importlib.metadata
from pathlib import Path

__all__ = ["__version__"]

# pyproject.toml [project] version is the single source of truth, so it is read
# first and directly. An editable install's dist-info is a PEP 660 snapshot
# taken at install time: it keeps reporting whatever the version was at the last
# `pip install -e .`, which on this checkout still said 0.1.0 long after pyproject
# moved on, so metadata-first would outvote the file it was derived from.
try:
    import tomllib
    __version__ = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "pyproject.toml")
        .read_text(encoding="utf-8")
    )["project"]["version"]
except Exception:
    # No pyproject.toml here: an installed wheel, or the frozen exe, where the
    # file is not bundled at all. The installed metadata is the same number for
    # a wheel; inside the exe neither source exists, so it settles on 0.0.0
    # rather than failing to import. requires-python is >= 3.11, so a missing
    # tomllib is impossible and the fallback is about the FILE, not the module.
    try:
        __version__ = importlib.metadata.version("deepseaport")
    except Exception:
        __version__ = "0.0.0"