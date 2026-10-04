"""DeepSeekHashV1 proof-of-work solver (wasmtime + shipped wasm)."""

from __future__ import annotations

import base64
import ctypes
import json
import logging
import os
import struct
import tempfile
import threading
import time
from pathlib import Path

from ...config import DATA_DIR

logger = logging.getLogger("deepseaport.pow")

# Env overrides for rotation survival without code change. Defaults preserve
# current pinned behaviour; setting DEEPSEAPORT_WASM_URL lets ops point at a
# rotated hash without waiting for a release. The filename is joined onto
# DATA_DIR, so it is validated (no separators, no "..") before use, and an
# overridden URL is logged loudly: whatever is fetched from it is validated
# only by size + "\0asm" magic and then INSTANTIATED and executed by wasmtime.
WASM_NAME = os.environ.get(
    "DEEPSEAPORT_WASM_NAME", "sha3_wasm_bg.7b9ca65ddd.wasm")
WASM_URL = os.environ.get(
    "DEEPSEAPORT_WASM_URL",
    "https://fe-static.deepseek.com/chat/static/" + WASM_NAME)
ALGORITHM = "DeepSeekHashV1"
# WebAssembly modules start with the magic bytes b"\0asm" (then version 1).
_WASM_MAGIC = b"\x00asm"
# Warn once per process about the remote-code override, not on every solve.
_warned_wasm_url = False

# Cached compiled wasm: Engine + Module are thread-safe/shareable, while
# Store/instance stay per-call. Recompiling per request cost 100-300ms.
_ENGINE = None
_MODULE = None
_MODULE_KEY: tuple | None = None
_MODULE_LOCK = threading.Lock()
_LINKER = None
_LINKER_LOCK = threading.Lock()
_WASM_PATH: Path | None = None


def validated_wasm_name(name: str | None = None) -> str:
    """Return WASM_NAME, rejecting anything that escapes DATA_DIR.

    The upstream asset is content-hash-named (``sha3_wasm_bg.<hash>.wasm``) and
    is joined onto DATA_DIR, so a plain filename is the only acceptable value.
    A separator or ``..`` in DEEPSEAPORT_WASM_NAME would otherwise let an env
    var point the download/write outside DATA_DIR (and the module is
    instantiated + executed afterwards, so a planted file runs code).
    """
    raw = str(WASM_NAME if name is None else name).strip()
    if (not raw or raw in (".", "..") or "/" in raw or "\\" in raw
            or ".." in raw or ":" in raw or "\0" in raw):
        raise RuntimeError(
            f"refusing unsafe DEEPSEAPORT_WASM_NAME {raw!r}: expected a plain "
            "filename (no path separator, no '..') resolved inside the data dir")
    return raw


# DATA_DIR is resolved once at import from the environment; it never changes
# for a process. Cache the joined path to avoid a mkdir syscall per request.
def wasm_path() -> Path:
    global _WASM_PATH
    if _WASM_PATH is None:
        name = validated_wasm_name()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _WASM_PATH = DATA_DIR / name
    return _WASM_PATH


def _warn_wasm_url_override() -> None:
    """Loud, once-per-process warning when the wasm URL is env-overridden.

    The download is only checked for size + ``\\0asm`` magic, so a pinned digest
    is the real control and an operator override is an explicit trust decision.
    """
    global _warned_wasm_url
    if _warned_wasm_url:
        return
    try:
        if not os.environ.get("DEEPSEAPORT_WASM_URL", "").strip():
            return
    except Exception:
        return
    _warned_wasm_url = True
    try:
        override = str(os.environ.get("DEEPSEAPORT_WASM_URL", "") or "").strip()
        logger.warning(
            "DEEPSEAPORT_WASM_URL is overridden (%s): the PoW wasm is fetched "
            "from that URL instead of the pinned upstream host. The module is "
            "only checked for size and \\0asm magic, then executed by wasmtime "
            "— only do this for a hash you trust.", override)
    except Exception:
        pass


def ensure_wasm(cookies: str = "", user_agent: str = "") -> Path:
    """Return local wasm path, downloading it (WAF cookies help) if missing."""
    _warn_wasm_url_override()
    path = wasm_path()
    if path.exists() and path.stat().st_size > 1000:
        return path
    from curl_cffi import requests as crequests

    headers = {"User-Agent": user_agent or "Mozilla/5.0", "Referer": "https://chat.deepseek.com/"}
    if cookies:
        headers["Cookie"] = cookies
    resp = crequests.get(WASM_URL, headers=headers, impersonate="chrome", timeout=60)
    resp.raise_for_status()
    content = resp.content or b""
    # Wrong host / WAF interstitial returns index.html with a 200. Reject
    # anything that is not a real wasm module before it reaches disk, so a
    # bad download fails here instead of as a confusing Module parse later.
    if len(content) <= 1000:
        raise RuntimeError(f"PoW wasm download too small ({len(content)} bytes)")
    if not content.startswith(_WASM_MAGIC):
        raise RuntimeError(
            f"PoW wasm download is not a wasm module "
            f"(magic={content[:4]!r}, {len(content)} bytes)")
    # Unique temp name per call, same technique as Settings.save: two callers
    # can both pass the exists() check above, and a shared "<name>.tmp" made the
    # loser's os.replace fail (FileNotFoundError once the file was already
    # moved, PermissionError on Windows while the winner still held it open) and
    # left a stale .tmp behind. Each caller owns its own file; the loser's
    # failure is harmless because the winner left a valid wasm in place.
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                    dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(content)
        replaced = False
        last_err: OSError | None = None
        for attempt in range(4):
            try:
                os.replace(tmp_name, path)
                replaced = True
                break
            except PermissionError as exc:
                # Transient AV/indexer locks block the rename for a few ms.
                last_err = exc
                time.sleep(0.05 * (attempt + 1))
            except OSError as exc:
                last_err = exc
                break
        if not replaced:
            raise RuntimeError(f"PoW wasm install failed ({last_err}): {path}")
    finally:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
    logger.info("downloaded PoW wasm (%d bytes)", len(content))
    return path


def _get_engine_module():
    """Return shared (engine, module), compiling once per wasm file version."""
    global _ENGINE, _MODULE, _MODULE_KEY
    path = wasm_path()
    try:
        stat = path.stat()
    except FileNotFoundError:
        raise
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if _ENGINE is not None and _MODULE is not None and _MODULE_KEY == key:
        return _ENGINE, _MODULE
    with _MODULE_LOCK:
        if _ENGINE is not None and _MODULE is not None and _MODULE_KEY == key:
            return _ENGINE, _MODULE
        from wasmtime import Engine, Module

        wasm_bytes = path.read_bytes()
        engine = _ENGINE if _ENGINE is not None else Engine()
        module = Module(engine, wasm_bytes)
        _ENGINE, _MODULE, _MODULE_KEY = engine, module, key
        return engine, module


def prewarm() -> bool:
    """Compile Engine/Module now so first request skips 100-300ms compile.

    Best-effort: missing wasm or wasmtime errors return False, never raise.
    Store/instance stay per-solve (Store is not thread-safe to share).
    """
    try:
        _get_engine_module()
        return True
    except Exception as exc:
        logger.debug("PoW prewarm skipped: %s", exc)
        return False


def solve(algorithm: str, challenge: str, salt: str, difficulty: int | float, expire_at: int) -> int | None:
    """Search the answer int via the site's own wasm. Returns None on failure."""
    if algorithm != ALGORITHM:
        raise ValueError(f"unsupported PoW algorithm: {algorithm}")
    from wasmtime import Linker, Store

    prefix = f"{salt}_{expire_at}_"
    engine, module = _get_engine_module()
    store = Store(engine)
    # Linker is thread-safe and bound to the (cached) engine: share it instead
    # of rebuilding per solve. Store stays per-call (not thread-safe to share).
    global _LINKER
    if _LINKER is None:
        with _LINKER_LOCK:
            if _LINKER is None:
                _LINKER = Linker(engine)
    linker = _LINKER
    instance = linker.instantiate(store, module)
    exports = instance.exports(store)
    try:
        memory = exports["memory"]
        add_to_stack = exports["__wbindgen_add_to_stack_pointer"]
        alloc = exports["__wbindgen_export_0"]
        wasm_solve = exports["wasm_solve"]
    except KeyError as exc:
        raise RuntimeError(f"PoW wasm missing expected export: {exc}")

    def _base() -> int:
        # Re-read per access: alloc/wasm_solve may grow memory, which can
        # relocate the buffer and stale a cached pointer.
        return ctypes.cast(memory.data_ptr(store), ctypes.c_void_p).value

    def _check(offset: int, size: int) -> None:
        if offset < 0 or size < 0 or offset + size > memory.data_len(store):
            raise RuntimeError(
                f"PoW wasm memory access out of bounds "
                f"(offset={offset}, size={size}, mem={memory.data_len(store)})")

    def write(offset: int, data: bytes) -> None:
        _check(offset, len(data))
        ctypes.memmove(_base() + offset, data, len(data))

    def read(offset: int, size: int) -> bytes:
        _check(offset, size)
        return ctypes.string_at(_base() + offset, size)

    def _to_int(value) -> int:
        return int(value.value) if hasattr(value, "value") else int(value)

    def encode(text: str) -> tuple[int, int]:
        data = text.encode("utf-8")
        ptr = _to_int(alloc(store, len(data), 1))
        write(ptr, data)
        return ptr, len(data)

    retptr = _to_int(add_to_stack(store, -16))
    _check(retptr, 16)
    ptr_c, len_c = encode(challenge)
    ptr_p, len_p = encode(prefix)
    try:
        wasm_solve(store, retptr, ptr_c, len_c, ptr_p, len_p, float(difficulty))
        status = struct.unpack("<i", read(retptr, 4))[0]
        value = struct.unpack("<d", read(retptr + 8, 8))[0]
    finally:
        add_to_stack(store, 16)
    if status == 0:
        return None
    return int(value)


def build_header(challenge: dict) -> str:
    """Encode a solved challenge as the X-DS-PoW-Response header value."""
    payload = {
        "algorithm": challenge["algorithm"],
        "challenge": challenge["challenge"],
        "salt": challenge["salt"],
        "answer": challenge["answer"],
        "signature": challenge["signature"],
        "target_path": challenge["target_path"],
    }
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return base64.b64encode(raw).decode("utf-8")
