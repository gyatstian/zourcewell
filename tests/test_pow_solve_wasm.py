"""Real-wasm coverage for the DeepSeek PoW solver + its download validation.

Why this file exists: ``solve()`` runs on EVERY DeepSeek chat completion
(ARCHITECTURE.md, "PoW details") and until now had never executed in CI. Its
body between ``instance.exports(store)`` and ``add_to_stack(store, 16)``
(pow.py:216-263) reaches into wasm linear memory with raw
``ctypes.memmove``/``string_at``, so the only honest way to cover it is to run a
module through it. Everything here is local: no socket, no credential, no
upstream request. ``tests/conftest.py``'s outbound guard stays armed throughout.

Two kinds of module are used, and the split is deliberate:

* The SHIPPED asset (``data/sha3_wasm_bg.<hash>.wasm``) for the end-to-end,
  determinism and concurrency tests. It reports "no answer at this difficulty"
  by leaving ``status == 0``, which ``solve()`` maps to ``None`` (pow.py:264-265)
  -- a legitimate module status, not a failure. Those tests skip with a precise
  reason when the asset is not in the checkout, rather than substituting a
  stand-in that would only exercise the plumbing.
* Hand-written WAT modules, compiled with ``wasmtime.wat2wasm``, for failure
  modes the real module will not produce on demand: a missing export, a
  ``retptr`` outside linear memory, an allocator handing back a bogus pointer,
  a trap. Those are what make the ``_check()`` OOB guard and the
  ``KeyError`` -> ``RuntimeError`` translation load-bearing rather than
  decorative.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import pytest
from curl_cffi.requests.errors import CurlError

from deepseaport.providers.deepseek import pow as PoW

wasmtime = pytest.importorskip(
    "wasmtime",
    reason="wasmtime is a hard dependency of pow.py; none of this file is "
           "testable without the real engine")

_REPO_DATA = Path(__file__).resolve().parent.parent / "data"


def _shipped_wasm() -> Path:
    """Where the asset ``ensure_wasm()`` would download lives in THIS checkout.

    Derived from this file, never from cwd or from ``PoW.DATA_DIR`` -- the
    hermetic conftest redirects the latter to a temp dir precisely so the repo's
    own data/ is never the thing under test.
    """
    try:
        name = PoW.validated_wasm_name()
    except RuntimeError:
        # DEEPSEAPORT_WASM_NAME is not a plain filename; nothing to locate and
        # the skip reason below explains it.
        return _REPO_DATA / "<rejected wasm name>"
    return _REPO_DATA / name


SHIPPED_WASM = _shipped_wasm()

requires_shipped_wasm = pytest.mark.skipif(
    not (SHIPPED_WASM.is_file() and SHIPPED_WASM.stat().st_size > 1000),
    reason=f"shipped PoW wasm asset absent at {SHIPPED_WASM}: the end-to-end "
           "solve() tests execute the upstream module itself and a "
           "hand-written stand-in would prove nothing about it")


# --- solve()'s answer contract ------------------------------------------------

def _assert_solved(result: object) -> None:
    """Assert solve()'s documented union of outcomes, not a specific answer.

    Whether a given (salt, challenge, difficulty) is actually solvable depends
    on the live site's parameters, so pinning a literal here would encode one
    machine's luck. What IS the contract is pow.py:264-266: ``None`` when the
    module leaves ``status == 0``, otherwise the decoded value as a positive
    int.
    """
    assert result is None or (
        isinstance(result, int) and not isinstance(result, bool) and result > 0
    ), f"unexpected solve() result: {result!r}"


# --- fixtures ----------------------------------------------------------------

@pytest.fixture(autouse=True)
def _restore_pow_caches():
    """Undo process-global cache mutation so one test cannot skew the next.

    ``_ENGINE`` / ``_MODULE`` / ``_MODULE_KEY`` / ``_LINKER`` (pow.py:39-44) are
    module-level singletons. A test that installs a hand-written module, or
    touches the wasm file's mtime, would otherwise leave the *next* test
    compiling whatever it left behind -- and would leave ``_LINKER`` populated,
    so the double-checked init at pow.py:211-214 would never re-run and the
    concurrency test would be vacuous.
    """
    saved = (PoW._ENGINE, PoW._MODULE, PoW._MODULE_KEY, PoW._LINKER)
    try:
        yield
    finally:
        PoW._ENGINE, PoW._MODULE, PoW._MODULE_KEY, PoW._LINKER = saved


@pytest.fixture
def pow_env():
    """Install the shipped wasm where ``pow.wasm_path()`` looks, and return it.

    ``tests/conftest.py`` points ``DEEPSEAPORT_DATA`` at a throwaway temp dir, so
    the checkout's ``data/`` asset is not on the solver's path. Copying the
    26 kB file in keeps the run hermetic -- nothing writes to the repo's own
    data dir -- while still executing the module the site would serve. The copy
    is removed on teardown so a later test cannot inherit an asset it did not
    install (a leftover file would make ``ensure_wasm``'s exists() fast path
    short-circuit and hide a download it meant to exercise).
    """
    path = PoW.wasm_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    pre_existing = path.exists()
    shutil.copyfile(SHIPPED_WASM, path)
    try:
        yield path
    finally:
        if not pre_existing:
            path.unlink(missing_ok=True)


def _wat_module(*, retptr: int = 1024, status: int = 1, value: str = "42.0",
                alloc: int = 2048, grow: int = 8) -> str:
    """A stand-in exporting exactly the four names pow.py looks up.

    pow.py:219-222 indexes ``instance.exports(store)`` for ``memory``,
    ``__wbindgen_add_to_stack_pointer``, ``__wbindgen_export_0`` (the wbindgen
    allocator: ``(size, align) -> ptr``) and ``wasm_solve``. This module exports
    all four with pow.py's real signatures and nothing else, so each test can
    turn exactly one knob.

    ``wasm_solve`` grows linear memory by ``grow`` pages BEFORE writing the
    result pair. That is the load-bearing detail: a host base pointer cached
    before the call would decode whatever the runtime left in the region it
    vacated, so this fixture is what proves the per-access
    ``memory.data_ptr`` re-read at pow.py:226-229 is needed.
    """
    return f"""(module
  (memory (export "memory") 1)
  (func (export "__wbindgen_add_to_stack_pointer") (param i32) (result i32)
    i32.const {retptr})
  (func (export "__wbindgen_export_0") (param i32 i32) (result i32)
    i32.const {alloc})
  (func $grow (memory.grow (i32.const {grow})) (drop))
  (func (export "wasm_solve")
        (param $ret i32) (param i32) (param i32) (param i32) (param i32) (param f64)
    call $grow
    (i32.store (local.get $ret) (i32.const {status}))
    (f64.store offset=8 (local.get $ret) (f64.const {value}))))
"""


# Same shape, but wasm_solve traps instead of writing a result: the `finally`
# unwind at pow.py:262-263 has to run on this path too.
_TRAPPING_WAT = """(module
  (memory (export "memory") 1)
  (func (export "__wbindgen_add_to_stack_pointer") (param i32) (result i32)
    i32.const 1024)
  (func (export "__wbindgen_export_0") (param i32 i32) (result i32)
    i32.const 2048)
  (func (export "wasm_solve")
        (param i32) (param i32) (param i32) (param i32) (param i32) (param f64)
    unreachable))
"""


@pytest.fixture
def crafted(tmp_path, monkeypatch):
    """Make pow.py compile and solve a hand-written module instead of the asset.

    ``wasm_path`` is redirected into ``tmp_path`` and the compiled-module cache
    dropped, so the next ``_get_engine_module()`` reads these bytes. The ENGINE
    is deliberately left alone: ``solve()`` binds its Linker to the engine and
    pow.py:179 never replaces a live one, so rebuilding it per test would throw
    away exactly the reuse the concurrency test is about.
    """
    path = tmp_path / "crafted.wasm"

    def _install(wat_text: str) -> Path:
        # Convert to binary here instead of letting Module() sniff WAT text:
        # Module() only accepts WAT when wasmtime was built with the optional
        # `wat` feature, and not at all when it loads by path -- which is why
        # pow.py:178 reads bytes. This keeps the fixture a genuine ``\\0asm``
        # file, exactly what ensure_wasm() would have installed.
        path.write_bytes(bytes(wasmtime.wat2wasm(wat_text.encode("utf-8"))))
        PoW._MODULE, PoW._MODULE_KEY = None, None
        return path

    monkeypatch.setattr(PoW, "wasm_path", lambda: path)
    return _install


@pytest.fixture
def linker_spy(monkeypatch):
    """Tally ``solve()``'s ``add_to_stack`` calls while the wasm still runs.

    Nothing is stubbed: ``instantiate`` returns the real Instance, and the
    exports mapping still hands back the real ``Func`` objects, so the module
    executes exactly as in production. Only the two ``add_to_stack`` calls are
    recorded.

    This is the only way to observe them. The Stack lives in the module's own
    mutable global, the Store is per-call (pow.py:207), and the Store dies with
    the call -- so once ``solve()`` returns there is nothing left to read the
    pointer from, on the success path or after a trap.
    """
    calls: list[int] = []
    real_linker = wasmtime.Linker

    class _Exports:
        def __init__(self, real):
            self._real = real

        def __getitem__(self, name):
            found = self._real[name]
            if name != "__wbindgen_add_to_stack_pointer":
                return found

            def _tally(store, delta):
                calls.append(delta)
                return found(store, delta)

            return _tally

    class _Instance:
        def __init__(self, real):
            self._real = real

        def exports(self, store):
            return _Exports(self._real.exports(store))

    class _Linker:
        def __init__(self, engine):
            self._real = real_linker(engine)

        def instantiate(self, store, module):
            return _Instance(self._real.instantiate(store, module))

    monkeypatch.setattr(wasmtime, "Linker", _Linker)
    monkeypatch.setattr(PoW, "_LINKER", None)
    return calls


# --- real-module end-to-end --------------------------------------------------

@requires_shipped_wasm
def test_solve_runs_the_real_module_without_raising(pow_env):
    """End to end: no raise, no network, no credential, valid answer or None."""
    _assert_solved(
        PoW.solve(PoW.ALGORITHM, "probe-challenge", "salt123", 10000, 1700000000))


@requires_shipped_wasm
@pytest.mark.parametrize("difficulty", [0, 1, 1000, 10000, 20000])
@pytest.mark.parametrize("challenge", ["c", "", "ünïcödé-challenge", "x" * 4096])
def test_solve_holds_its_contract_across_inputs(pow_env, difficulty, challenge):
    """Every (challenge bytes, difficulty) shape must come back answered or None.

    The challenge is written through ``encode()`` (pow.py:248-252), so the
    non-ASCII and long cases are what cover its utf-8 encode plus the
    bounds-checked ``memmove`` into linear memory.
    """
    _assert_solved(
        PoW.solve(PoW.ALGORITHM, challenge, "salt123", difficulty, 1700000000))


@requires_shipped_wasm
def test_solve_is_fast_once_the_module_is_compiled(pow_env):
    """Bound the per-solve cost, after prewarm() has taken the compile out.

    prewarm() exists for exactly this (pow.py:185-196); this asserts the split
    actually holds, i.e. a solve is not silently recompiling 100-300 ms of wasm
    on the hot path. The bound is deliberately loose -- an order of magnitude
    over the measured sub-millisecond solve -- because it is there to catch a
    regression to the compile, not to time anything.
    """
    assert PoW.prewarm() is True
    started = time.perf_counter()
    for _ in range(5):
        _assert_solved(
            PoW.solve(PoW.ALGORITHM, "probe-challenge", "salt123", 10000,
                      1700000000))
    elapsed = time.perf_counter() - started
    assert elapsed < 5.0, f"5 solves took {elapsed:.3f}s -- module recompiling?"


@requires_shipped_wasm
def test_solve_is_deterministic_and_idempotent(pow_env):
    """Same inputs, same answer -- and repeating them must not drift.

    A per-call Store (pow.py:207) means a repeat solve starts from a fresh
    instance, so any state leaking across calls through the shared Engine or
    Linker would show up here as a changed result.
    """
    args = (PoW.ALGORITHM, "probe-challenge", "salt123", 10000, 1700000000)
    first = PoW.solve(*args)
    second = PoW.solve(*args)
    third = PoW.solve(*args)
    _assert_solved(first)
    assert second == first
    assert third == first


@requires_shipped_wasm
def test_solve_is_thread_safe_with_a_shared_linker(pow_env, monkeypatch):
    """The real concurrency risk: one Linker, one Engine, N per-call Stores.

    pow.py:208-215 shares a Linker across calls behind a double-checked init and
    keeps the Store per-call. If the init were not actually safe, N threads
    released together would build N Linkers or worse; if the Store were shared,
    two threads would interleave inside one instance's linear memory and decode
    each other's answers. Both are asserted, not assumed.

    The Barrier releases all threads INTO solve() at the same instant, so the
    race is actually contended; no sleeps are involved. The switch interval is
    dropped for the duration to widen the window between the ``_LINKER is None``
    test and the assignment -- this makes a broken lock far more likely to be
    caught, but a pass does not depend on it. The spies deliberately append
    without a lock: ``list.append`` is atomic under the GIL, and serialising the
    workers inside the spy would blunt the very race being tested.
    """
    real_store = wasmtime.Store
    real_linker = wasmtime.Linker
    linkers: list[object] = []
    stores: list[object] = []
    engines: list[object] = []

    def _spy_linker(engine):
        linker = real_linker(engine)
        linkers.append(linker)
        return linker

    class _SpyStore(real_store):
        def __init__(self, engine=None, *args, **kwargs):
            super().__init__(engine, *args, **kwargs)
            stores.append(self)
            engines.append(engine)

    monkeypatch.setattr(wasmtime, "Linker", _spy_linker)
    monkeypatch.setattr(wasmtime, "Store", _SpyStore)
    monkeypatch.setattr(PoW, "_LINKER", None)

    workers = 8
    barrier = threading.Barrier(workers, timeout=60)
    results: dict[int, object] = {}
    errors: list[BaseException] = []

    def _run(index: int) -> None:
        try:
            barrier.wait()
            results[index] = PoW.solve(PoW.ALGORITHM, f"challenge-{index}",
                                       "salt123", 10000, 1700000000)
        except BaseException as exc:  # noqa: BLE001 - reported to the main thread
            errors.append(exc)

    threads = [threading.Thread(target=_run, args=(i,), name=f"pow-{i}")
               for i in range(workers)]
    previous_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(120)
    finally:
        sys.setswitchinterval(previous_interval)
    stuck = [t.name for t in threads if t.is_alive()]
    assert not stuck, f"threads did not finish: {stuck}"

    assert errors == [], f"concurrent solve raised: {errors!r}"
    assert sorted(results) == list(range(workers))
    for value in results.values():
        _assert_solved(value)
    # One Linker for the whole burst: the double-checked init really is shared.
    assert len(linkers) == 1, f"expected 1 Linker, built {len(linkers)}"
    # ...and a distinct Store per call, all on that one cached Engine.
    assert len(stores) == workers
    assert len({id(store) for store in stores}) == workers
    assert len({id(engine) for engine in engines}) == 1


@requires_shipped_wasm
def test_add_to_stack_is_balanced_around_a_real_solve(pow_env, linker_spy):
    """Exactly one -16 in, one +16 out, on the ordinary path.

    pow.py:254 reserves a 16-byte return slot and pow.py:262-263 gives it back in
    a ``finally``. A missing or doubled unwind would be invisible to every other
    assertion here, because the Store that holds the pointer is discarded when
    ``solve()`` returns.
    """
    _assert_solved(
        PoW.solve(PoW.ALGORITHM, "probe-challenge", "salt123", 10000, 1700000000))
    assert linker_spy == [-16, 16]


# --- failure paths the real module will not produce on demand ----------------

def test_add_to_stack_unwinds_when_wasm_solve_traps(linker_spy, crafted):
    """A trap inside wasm_solve must still unwind, and must not be swallowed.

    pow.py:259 sits inside the ``try`` whose ``finally`` holds the +16, so the
    unwind is the whole point of that construct on this path. Needs only the
    stand-in, so it runs even without the shipped asset.
    """
    crafted(_TRAPPING_WAT)
    with pytest.raises(wasmtime.Trap):
        PoW.solve(PoW.ALGORITHM, "c", "s", 1, 1)
    assert linker_spy == [-16, 16]


@pytest.mark.parametrize("missing", [
    "memory", "__wbindgen_add_to_stack_pointer", "__wbindgen_export_0",
    "wasm_solve",
])
def test_solve_reports_a_missing_export_as_runtime_error(crafted, missing):
    """Every name pow.py requires must fail as a clear RuntimeError.

    pow.py:219-222 indexes all four exports before using any of them, and
    wasmtime's ``InstanceExports`` raises ``KeyError`` for an unknown name. If
    that were not caught, a module missing one export would escape as a
    ``KeyError`` from inside the request path with no hint about the wasm at all.
    ``KeyError`` is not a ``RuntimeError`` subclass, so ``pytest.raises`` here
    already proves the translation happened; the message pins which export.

    Each case is a real, valid module that simply omits one name and keeps the
    earlier ones -- so the failing lookup is genuinely ``missing``, not a module
    that fails to compile.
    """
    memory = '(memory (export "memory") 1)'
    add_to_stack = ('(func (export "__wbindgen_add_to_stack_pointer") '
                    '(param i32) (result i32) i32.const 1024)')
    alloc = ('(func (export "__wbindgen_export_0") (param i32 i32) '
             '(result i32) i32.const 2048)')
    wasm_solve = '(func (export "wasm_solve"))'
    # Built in pow.py's own lookup order so the first absent name is the one
    # under test in each case.
    parts = {
        "memory": [wasm_solve],
        "__wbindgen_add_to_stack_pointer": [memory],
        "__wbindgen_export_0": [memory, add_to_stack],
        "wasm_solve": [memory, add_to_stack, alloc],
    }[missing]
    crafted(f"(module {' '.join(parts)})")

    with pytest.raises(RuntimeError, match=re.escape(
            f"PoW wasm missing expected export: '{missing}'")):
        PoW.solve(PoW.ALGORITHM, "c", "s", 1, 1)


@pytest.mark.parametrize("retptr", [65536, 65521, -1, -4096, 2147483647])
def test_check_rejects_an_out_of_range_retptr(crafted, retptr):
    """The retptr handed back by the module is checked before it is used.

    ``_check(retptr, 16)`` at pow.py:255 is the single guard between a module
    that hands back a nonsense pointer and a raw ``ctypes.string_at`` into
    unmapped host memory. 65536 is the first byte past a 1-page memory and
    65521 is ``data_len - 15``, the smallest offset whose 16-byte read would
    spill over the end.
    """
    crafted(_wat_module(retptr=retptr))
    with pytest.raises(RuntimeError, match="out of bounds") as caught:
        PoW.solve(PoW.ALGORITHM, "c", "s", 1, 1)
    assert f"offset={retptr}" in str(caught.value)


def test_check_admits_the_last_exact_fit(crafted):
    """``retptr + 16 == data_len`` is in bounds -- the boundary is exact.

    This is the positive half of the guard above, and it is what makes the
    16-byte check sufficient for BOTH reads pow.py performs: the status int32 at
    ``retptr`` and the f64 at ``retptr + 8`` together occupy exactly
    ``[retptr, retptr + 16)``, so a single 16-byte check provably covers the
    pair. Off-by-one in the other direction (retptr+16 > data_len) is rejected.
    """
    crafted(_wat_module(retptr=65536 - 16, status=1, value="9.0"))
    assert PoW.solve(PoW.ALGORITHM, "c", "s", 1, 1) == 9


@pytest.mark.parametrize("alloc_ptr", [-8, 65536, 2147483647])
def test_check_rejects_an_out_of_range_allocator_pointer(crafted, alloc_ptr):
    """The write side of the guard: a bogus alloc ptr must not reach memmove.

    ``encode()`` (pow.py:248-252) writes the challenge and prefix through
    ``write()``, which bounds-checks at pow.py:237-239 immediately before
    ``ctypes.memmove``. Without that check this is a straight write into
    whatever host address the pointer names.
    """
    crafted(_wat_module(alloc=alloc_ptr))
    with pytest.raises(RuntimeError, match="out of bounds") as caught:
        PoW.solve(PoW.ALGORITHM, "c", "s", 1, 1)
    assert f"offset={alloc_ptr}" in str(caught.value)


def test_solve_returns_none_when_the_module_reports_no_answer(crafted):
    """``status == 0`` short-circuits to None and never reaches the value.

    pow.py:264-265. Asserting this on the stand-in matters because the shipped
    module happens to report no answer for every input a test can construct, so
    the ``status != 0`` branch and this one are the two halves of the return
    contract and each needs its own case.
    """
    crafted(_wat_module(status=0, value="7.0"))
    assert PoW.solve(PoW.ALGORITHM, "c", "s", 1, 1) is None


def test_solve_returns_the_decoded_value_when_the_module_reports_success(crafted):
    """The success half: unpack the i32/f64 pair and return the int.

    Uses the stand-in because the shipped module's answer depends on salt,
    challenge and difficulty from the live site -- asserting a literal would
    encode one machine's luck, not the code. The wasm still runs, memory still
    grows mid-solve, and the ``struct.unpack`` pair at pow.py:260-261 is what is
    under test.
    """
    crafted(_wat_module(status=1, value="1234.5"))
    # int() truncates, exactly as pow.py:266 does.
    assert PoW.solve(PoW.ALGORITHM, "c", "s", 1, 1) == 1234


def test_solve_survives_memory_growth_between_the_pointer_read_and_the_result(
        crafted):
    """The base pointer really is re-read per access, not cached.

    The stand-in grows linear memory by 8 pages inside ``wasm_solve``, i.e.
    between ``encode()`` writing the challenge and pow.py:260-261 unpacking the
    result. wasmtime reserves its address range up front so the host pointer may
    or may not actually move here; what this pins down is that ``solve()`` does
    not depend on it either way -- it re-reads ``memory.data_ptr`` for every
    access (pow.py:226-229), so a runtime that does relocate is handled.
    """
    crafted(_wat_module(retptr=1024, status=1, value="5.0"))
    assert PoW.solve(PoW.ALGORITHM, "c", "s", 1, 1) == 5


# --- prewarm() ----------------------------------------------------------------

@requires_shipped_wasm
def test_prewarm_true_when_the_wasm_is_present(pow_env):
    assert PoW.prewarm() is True
    # Compiling is the whole job; the cached module is now warm.
    assert PoW._MODULE is not None
    assert PoW._MODULE_KEY == (str(pow_env), pow_env.stat().st_size,
                               pow_env.stat().st_mtime_ns)


def test_prewarm_false_when_the_wasm_is_absent(tmp_path, monkeypatch):
    """Best-effort by contract (pow.py:188): a missing file is False, not a raise."""
    monkeypatch.setattr(PoW, "wasm_path", lambda: tmp_path / "absent.wasm")
    assert PoW.prewarm() is False


def test_prewarm_false_on_a_non_wasm_file(tmp_path, monkeypatch):
    """A garbage file must be a False, not a wasmtime parse error.

    This is the shape the real-world bad download takes (ARCHITECTURE.md, "PoW
    details"): HTTP 200, SPA ``index.html``, confusing parse error much later.
    prewarm() is the one caller that is allowed to swallow that; solve()'s
    callers are not.
    """
    bad = tmp_path / "sha3.wasm"
    bad.write_bytes(b"<!doctype html><html><body>not a module</body></html>" * 40)
    monkeypatch.setattr(PoW, "wasm_path", lambda: bad)
    assert PoW.prewarm() is False


def test_get_engine_module_recompiles_only_when_the_file_version_changes(
        tmp_path, monkeypatch):
    """The cache key is (path, size, mtime_ns) -- exercise all three limbs.

    A Module compiled from stale bytes would solve every request against a wasm
    the site no longer serves, and the only thing standing between that and a
    silent wrong answer is this key (pow.py:170).

    The same-size/different-bytes case is why the mtime limb exists at all: size
    alone would miss a hash rotation to a file of identical length.
    """
    path = tmp_path / "cache.wasm"
    monkeypatch.setattr(PoW, "wasm_path", lambda: path)
    PoW._ENGINE = PoW._MODULE = PoW._MODULE_KEY = None

    def _binary(**kwargs) -> bytes:
        return bytes(wasmtime.wat2wasm(_wat_module(**kwargs).encode("utf-8")))

    first = _binary(value="1.0")
    path.write_bytes(first)
    engine_a, module_a = PoW._get_engine_module()
    stamp = path.stat()
    assert PoW._MODULE_KEY == (str(path), stamp.st_size, stamp.st_mtime_ns)

    # Unchanged file -> the identical pair, and no recompile in between.
    assert PoW._get_engine_module() == (engine_a, module_a)

    # Same size, different bytes, newer mtime -> new Module, SAME Engine.
    # pow.py:179 keeps a live Engine on purpose: the shared Linker is bound to
    # it, so replacing it would strand that Linker on the old engine.
    second = _binary(value="2.0")
    assert len(second) == len(first), "fixture must keep the size limb unchanged"
    path.write_bytes(second)
    assert path.stat().st_size == stamp.st_size
    os.utime(path, ns=(stamp.st_mtime_ns + 2_000_000_000,) * 2)
    assert path.stat().st_mtime_ns != stamp.st_mtime_ns, "mtime limb did not move"
    engine_b, module_b = PoW._get_engine_module()
    assert module_b is not module_a
    assert engine_b is engine_a

    # Different size -> recompile again on the size limb alone. 64 is chosen over
    # a neighbouring single-digit constant because LEB128 encodes 8 and 9 in the
    # same number of bytes, which would leave the size limb unchanged.
    third = _binary(value="2.0", grow=64)
    assert len(third) != len(second)
    path.write_bytes(third)
    _, module_c = PoW._get_engine_module()
    assert module_c is not module_b
    assert PoW._MODULE_KEY[1] == path.stat().st_size

    # The path is a limb too: moving to identical bytes elsewhere recompiles,
    # rather than serving the module cached for the old location.
    other = tmp_path / "other.wasm"
    other.write_bytes(third)
    monkeypatch.setattr(PoW, "wasm_path", lambda: other)
    engine_d, module_d = PoW._get_engine_module()
    assert module_d is not module_c
    assert PoW._MODULE_KEY[0] == str(other)
    assert engine_d is engine_a, "the Engine is reused across every recompile"


def test_get_engine_module_key_is_exactly_path_size_mtime(tmp_path, monkeypatch):
    """Pin the key's shape, and state its one documented limit.

    With size and mtime_ns held constant the cache is reused even if the bytes
    change. That is sound for this deployment -- the asset is content-hash-named
    and installed with an atomic ``os.replace``, so a real swap always moves the
    mtime -- and the test names the limit rather than hiding it.
    """
    path = tmp_path / "key.wasm"
    monkeypatch.setattr(PoW, "wasm_path", lambda: path)
    PoW._ENGINE = PoW._MODULE = PoW._MODULE_KEY = None

    payload = bytes(wasmtime.wat2wasm(_wat_module(value="1.0").encode("utf-8")))
    path.write_bytes(payload)
    _, module_a = PoW._get_engine_module()
    stamp = path.stat()

    other = bytes(wasmtime.wat2wasm(_wat_module(value="2.0").encode("utf-8")))
    assert len(other) == len(payload)
    path.write_bytes(other)
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    _, module_b = PoW._get_engine_module()
    assert module_b is module_a, "unchanged (size, mtime) must reuse the cache"
    assert PoW._MODULE_KEY == (str(path), stamp.st_size, stamp.st_mtime_ns)


# --- ensure_wasm() download validation ---------------------------------------

_WASM_HEADER = b"\x00asm\x01\x00\x00\x00"


def _payload(size: int) -> bytes:
    """A body of exactly ``size`` bytes that starts with the wasm magic.

    ``ensure_wasm()`` validates size + magic and nothing else (pow.py:120-125),
    so padding past the header is enough to produce a body it accepts. Building
    the length exactly is what lets the boundary tests pin the ``> 1000`` /
    ``<= 1000`` pair from both sides.
    """
    return (_WASM_HEADER + b"x" * size)[:size]


class _FakeResponse:
    """Minimal stand-in for curl_cffi's Response.

    ``ensure_wasm()`` touches exactly two members -- ``raise_for_status()``
    (pow.py:115) and ``content`` (pow.py:116) -- so a full transport fake would
    be noise. Raising ``CurlError`` is what the real client does for a non-2xx,
    and pow.py lets it propagate, which is the behaviour worth asserting.
    """

    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise CurlError(f"HTTP {self.status_code}")


def _fake_get(response: _FakeResponse, seen: dict | None = None):
    """Build a ``curl_cffi.requests.get`` replacement returning ``response``."""

    def _get(url, **kwargs):
        if seen is not None:
            seen["url"] = url
            seen["kwargs"] = kwargs
        return response

    return _get


def _no_tmp_residue(path: Path) -> None:
    assert not list(path.parent.glob(f"{path.name}.*.tmp")), \
        f"stale .tmp left next to {path}"


def test_wasm_url_is_pinned_to_the_documented_host():
    """The host is the documented one, and only the host is configurable.

    ARCHITECTURE.md ("PoW details"): the asset lives on ``fe-static.deepseek.com``.
    Fetching from anywhere else answers HTTP 200 with the SPA ``index.html``,
    which is the historical failure the size + magic checks exist to catch -- so
    the constant itself is worth an assertion, not just the checks.
    """
    if os.environ.get("DEEPSEAPORT_WASM_URL"):
        pytest.skip("DEEPSEAPORT_WASM_URL is set: WASM_URL is intentionally "
                    "rotated here, the pinned-host assertion does not apply")
    parsed = urlparse(PoW.WASM_URL)
    assert parsed.scheme == "https"
    assert parsed.netloc == "fe-static.deepseek.com", \
        f"wrong wasm host: {PoW.WASM_URL}"
    assert parsed.path == f"/chat/static/{PoW.validated_wasm_name()}"


def test_ensure_wasm_installs_a_valid_response(tmp_path, monkeypatch):
    """A good response lands byte-for-byte on disk and is returned."""
    target = tmp_path / "good.wasm"
    payload = _payload(2048)
    monkeypatch.setattr(PoW, "wasm_path", lambda: target)
    monkeypatch.setattr("curl_cffi.requests.get",
                        _fake_get(_FakeResponse(payload)))
    assert PoW.ensure_wasm() == target
    assert target.read_bytes() == payload
    _no_tmp_residue(target)


@requires_shipped_wasm
def test_ensure_wasm_installs_bytes_that_actually_solve(tmp_path, monkeypatch):
    """The strongest form of "a good response": the result is executable.

    Anything that merely starts with ``\\0asm`` passes the download checks, so
    this installs the real shipped asset through the fake transport and then
    runs ``prewarm()`` and a solve against it -- a truncated body, or one whose
    magic is right but whose body is not, would fail here.
    """
    target = tmp_path / "real.wasm"
    monkeypatch.setattr(PoW, "wasm_path", lambda: target)
    monkeypatch.setattr("curl_cffi.requests.get",
                        _fake_get(_FakeResponse(SHIPPED_WASM.read_bytes())))
    assert PoW.ensure_wasm() == target
    assert target.read_bytes() == SHIPPED_WASM.read_bytes()
    PoW._ENGINE = PoW._MODULE = PoW._MODULE_KEY = None
    assert PoW.prewarm() is True
    _assert_solved(PoW.solve(PoW.ALGORITHM, "probe-challenge", "salt123",
                             10000, 1700000000))


@pytest.mark.parametrize("size", [0, 1, 8, 100, 999, 1000])
def test_ensure_wasm_rejects_a_too_small_response(tmp_path, monkeypatch, size):
    """``<= 1000`` bytes is refused, and nothing is written.

    The threshold is shared with the "already have it" fast path
    (pow.py:107, ``> 1000``), so 1000 is pinned as the boundary from both sides
    by this test and the 1001-byte one below: a body that slipped through here
    would be accepted by one check and refused by the other, i.e. the file would
    be re-downloaded on every single solve.
    """
    target = tmp_path / "small.wasm"
    monkeypatch.setattr(PoW, "wasm_path", lambda: target)
    monkeypatch.setattr("curl_cffi.requests.get",
                        _fake_get(_FakeResponse(_payload(size))))
    with pytest.raises(RuntimeError, match="PoW wasm download too small"):
        PoW.ensure_wasm()
    assert not target.exists()
    _no_tmp_residue(target)


def test_ensure_wasm_rejects_an_html_response(tmp_path, monkeypatch):
    """THE documented failure: HTTP 200 plus the SPA index.html.

    Big enough to clear the size check, so only the magic check can catch it.
    Without this the SPA page would be written to disk and the error would
    surface much later as wasmtime's "expected '('" parse failure, with nothing
    pointing at the download.
    """
    target = tmp_path / "html.wasm"
    html = (b"<!doctype html><html><head><title>DeepSeek</title></head>"
            b"<body><div id=root></div>" + b"<!-- padding -->" * 200 + b"</body></html>")
    assert len(html) > 1000, "fixture must clear the size check"
    monkeypatch.setattr(PoW, "wasm_path", lambda: target)
    monkeypatch.setattr("curl_cffi.requests.get", _fake_get(_FakeResponse(html)))
    with pytest.raises(RuntimeError, match="PoW wasm download is not a wasm module"):
        PoW.ensure_wasm()
    assert not target.exists()
    _no_tmp_residue(target)


def test_ensure_wasm_accepts_the_first_valid_size(tmp_path, monkeypatch):
    """1001 bytes with the magic header is the smallest body that installs.

    The other half of the boundary pair: proving the check is ``> 1000`` and not
    something stricter or looser.
    """
    target = tmp_path / "min.wasm"
    payload = _payload(1001)
    assert len(payload) == 1001
    monkeypatch.setattr(PoW, "wasm_path", lambda: target)
    monkeypatch.setattr("curl_cffi.requests.get",
                        _fake_get(_FakeResponse(payload)))
    assert PoW.ensure_wasm() == target
    assert target.read_bytes() == payload
    _no_tmp_residue(target)


def test_ensure_wasm_propagates_an_http_error_and_writes_nothing(
        tmp_path, monkeypatch):
    """``raise_for_status`` is not swallowed: no file, and the error surfaces."""
    target = tmp_path / "err.wasm"
    monkeypatch.setattr(PoW, "wasm_path", lambda: target)
    monkeypatch.setattr("curl_cffi.requests.get",
                        _fake_get(_FakeResponse(_payload(2000), status_code=403)))
    with pytest.raises(CurlError):
        PoW.ensure_wasm()
    assert not target.exists()
    _no_tmp_residue(target)


def test_ensure_wasm_leaves_no_tmp_when_the_install_itself_fails(
        tmp_path, monkeypatch):
    """The ``finally`` unlink (pow.py:153-157) covers the retry loop too.

    This is the only failure path where a temp file genuinely exists: the
    payload passed validation and a real file was created, so if the ``finally``
    were dropped an ``os.replace`` failure would leave a multi-kilobyte ``.tmp``
    behind on every attempt. A directory is used as the destination because
    ``os.replace(file, dir)`` fails for free on both POSIX and Windows, with no
    monkeypatching of ``os`` to arrange it.
    """
    target = tmp_path / "busy.wasm"
    target.mkdir()
    monkeypatch.setattr(PoW, "wasm_path", lambda: target)
    monkeypatch.setattr("curl_cffi.requests.get",
                        _fake_get(_FakeResponse(_payload(2000))))
    with pytest.raises(RuntimeError, match="PoW wasm install failed"):
        PoW.ensure_wasm()
    _no_tmp_residue(target)


def test_overridden_wasm_url_is_fetched_and_still_validated(tmp_path, monkeypatch):
    """An override changes WHERE from, never WHAT is accepted.

    ``DEEPSEAPORT_WASM_URL`` is a rotation escape hatch, not a bypass: whatever
    comes back from it is still gated by the same size + magic checks, because
    the result is INSTANTIATED and EXECUTED by wasmtime (pow.py:20-30).

    The seam patched is ``WASM_URL`` itself, not the environment variable:
    ``WASM_URL`` is captured from the environment at import (pow.py:28-30), which
    is how a real deployment supplies it -- ops set it before the process starts.
    Re-reading the environment per download would swap the module underneath an
    already-compiled Module and cached Linker, so the import-time capture is the
    correct design and the test targets the constant that is actually used.
    """
    target = tmp_path / "rotated.wasm"
    seen: dict = {}
    monkeypatch.setattr(PoW, "wasm_path", lambda: target)
    monkeypatch.setattr(PoW, "WASM_URL", "https://elsewhere.example/sha3.wasm")
    monkeypatch.setattr("curl_cffi.requests.get", _fake_get(
        _FakeResponse(b"<html>" + b"<!-- pad -->" * 400), seen=seen))
    with pytest.raises(RuntimeError, match="not a wasm module"):
        PoW.ensure_wasm()
    assert seen["url"] == "https://elsewhere.example/sha3.wasm"
    assert not target.exists()
    _no_tmp_residue(target)


def test_ensure_wasm_retries_a_transient_lock_then_gives_up(tmp_path, monkeypatch):
    """PermissionError is retried with a growing backoff; the temp file is freed.

    pow.py:139-152 distinguishes a transient AV/indexer lock (PermissionError,
    retried 4x with a growing sleep) from any other OSError (fatal, break at
    once). Both branches must end in the same install-failed RuntimeError with
    no ``.tmp`` residue -- the retry is what stops a one-off AV lock from
    failing a request, and the ``finally`` is what stops the retry from littering
    ``data/``. ``time.sleep`` is stubbed so the test costs no wall-clock time;
    the backoff values themselves are what is asserted.
    """
    target = tmp_path / "locked.wasm"
    monkeypatch.setattr(PoW, "wasm_path", lambda: target)
    monkeypatch.setattr("curl_cffi.requests.get",
                        _fake_get(_FakeResponse(_payload(2000))))
    attempts: list[str] = []
    slept: list[float] = []

    def _locked(src, dst):
        attempts.append(str(src))
        raise PermissionError(13, "used by another process")

    monkeypatch.setattr(PoW.os, "replace", _locked)
    monkeypatch.setattr(PoW.time, "sleep", slept.append)

    with pytest.raises(RuntimeError, match="PoW wasm install failed"):
        PoW.ensure_wasm()

    assert len(attempts) == 4, "the retry loop must make exactly 4 attempts"
    assert slept == [0.05 * (i + 1) for i in range(4)], \
        "backoff must grow per attempt"
    assert not target.exists()
    _no_tmp_residue(target)


def test_ensure_wasm_does_not_retry_a_permanent_rename_error(tmp_path, monkeypatch):
    """Any other OSError is fatal on the first attempt.

    The counterpart to the test above: retrying a genuinely permanent failure
    would just add four sleeps to every request until the error clears.
    """
    target = tmp_path / "permanent.wasm"
    monkeypatch.setattr(PoW, "wasm_path", lambda: target)
    monkeypatch.setattr("curl_cffi.requests.get",
                        _fake_get(_FakeResponse(_payload(2000))))
    attempts: list[str] = []

    def _broken(src, dst):
        attempts.append(str(src))
        raise OSError(28, "no space left on device")

    monkeypatch.setattr(PoW.os, "replace", _broken)

    with pytest.raises(RuntimeError, match="PoW wasm install failed"):
        PoW.ensure_wasm()

    assert len(attempts) == 1, "a non-PermissionError must not be retried"
    assert not target.exists()
    _no_tmp_residue(target)


def test_wasm_url_override_warns_once_per_process(tmp_path, monkeypatch, caplog):
    """The override warning is emitted on the first download only.

    pow.py:77-101 warns loudly but once, because ``ensure_wasm()`` sits on the
    request path -- a per-solve WARNING would drown the log. The ``_warned``
    latch is process-global, so this asserts both that the first call warns and
    that a second call is silent; without the latch every DeepSeek request would
    re-log the same line.
    """
    target = tmp_path / "warn.wasm"
    monkeypatch.setattr(PoW, "wasm_path", lambda: target)
    monkeypatch.setattr(PoW, "WASM_URL", "https://elsewhere.example/sha3.wasm")
    monkeypatch.setenv("DEEPSEAPORT_WASM_URL", "https://elsewhere.example/sha3.wasm")
    monkeypatch.setattr(PoW, "_warned_wasm_url", False)
    monkeypatch.setattr("curl_cffi.requests.get",
                        _fake_get(_FakeResponse(_payload(2000))))

    with caplog.at_level(logging.WARNING, logger="deepseaport.pow"):
        PoW.ensure_wasm()
        first = [r for r in caplog.records if "DEEPSEAPORT_WASM_URL" in r.getMessage()]
        caplog.clear()
        PoW.ensure_wasm()
        second = [r for r in caplog.records if "DEEPSEAPORT_WASM_URL" in r.getMessage()]

    assert len(first) == 1
    assert "https://elsewhere.example/sha3.wasm" in first[0].getMessage()
    assert second == [], "the override warning must not repeat per request"


def test_ensure_wasm_forwards_cookies_and_user_agent(tmp_path, monkeypatch):
    """The download carries the WAF cookies and a browser impersonation.

    The site sits behind the same WAF as chat, which is why this fetch is not a
    plain GET: without the cookie the WAF answers with the interstitial instead
    of the asset. Asserted because the failure it prevents is indistinguishable,
    once it reaches the checks, from a hostile response.
    """
    target = tmp_path / "waf.wasm"
    seen: dict = {}
    monkeypatch.setattr(PoW, "wasm_path", lambda: target)
    monkeypatch.setattr("curl_cffi.requests.get",
                        _fake_get(_FakeResponse(_payload(2048)), seen=seen))
    PoW.ensure_wasm(cookies="aws-waf-token=abc", user_agent="UA/1.0")
    assert seen["kwargs"]["headers"]["Cookie"] == "aws-waf-token=abc"
    assert seen["kwargs"]["headers"]["User-Agent"] == "UA/1.0"
    assert seen["kwargs"]["impersonate"] == "chrome"
    assert target.is_file()
    _no_tmp_residue(target)

