"""Opencode model catalogue: live from the serve, static as the fallback.

Two layers, and which one answers depends on whether a local ``opencode
serve`` is up:

* :func:`live_specs` reads ``GET /api/model`` on the serve instance. That is
  the truth: the upstream catalogue turns over weekly and also lists the
  operator's other authenticated providers, all of which are exposed here as
  ``providerID/modelID``. One process-wide cache (:data:`_CACHE_TTL`, five
  minutes) holds BOTH the catalogue and a failure, because ``/v1/models`` is
  hit on every request and a serve that is not running must not cost one
  timeout per call.
* :data:`MODELS_SPEC` below is the OFFLINE FALLBACK: the hand-verified list,
  and where the exact ids and their current upstream availability are written
  down. It was verified by hand against a serve (session -> message ->
  ``finish: stop`` with real text, no auth) and matches the Zen catalog
  (``https://opencode.ai/zen/v1/models``). An entry upstream no longer serves
  stays listed so a serve-less start still answers; it fails live with
  ``Model unavailable``. All are flagged thinking-capable: ``reasoning`` parts
  were observed by default (muse-spark live: 203 reasoning tokens) and the rest
  are adapter-attested thinking models, so ``reasoning_effort`` is accepted as
  the serve message ``variant`` for all.

``provider.models()`` serves the live set whenever it is non-empty and
:data:`MODELS_SPEC` otherwise, so ``/v1/models`` and the CLI list are as
current as the serve. ``provider.models_hint()``, behind :func:`cached_specs`,
is the menus' variant of the same two layers: answered from what this process
already read, never touching the network and never booting anything.
``provider.model_spec`` additionally accepts
any catalogue-unknown ``provider/model`` id as a passthrough (user-configured
proxy providers on the same serve instance), so ``/v1/chat/completions``
works with those too.
"""

from __future__ import annotations

import logging
import threading
import time

from ...core.provider import ModelSpec

logger = logging.getLogger("zourcewell.providers")

MODELS_SPEC: dict[str, ModelSpec] = {
    "opencode/muse-spark-1.3-contributor-free": ModelSpec(
        id="opencode/muse-spark-1.3-contributor-free", thinking=True, search=False,
        model_type="default", owned_by="opencode"),
    "opencode/muse-spark-1.2-contributor-free": ModelSpec(
        id="opencode/muse-spark-1.2-contributor-free", thinking=True, search=False,
        model_type="default", owned_by="opencode"),
    "opencode/ling-3.0-flash-fin-free": ModelSpec(
        id="opencode/ling-3.0-flash-fin-free", thinking=True, search=False,
        model_type="default", owned_by="opencode"),
    "opencode/space-bunny-free": ModelSpec(
        id="opencode/space-bunny-free", thinking=True, search=False,
        model_type="default", owned_by="opencode"),
    "opencode/mimo-v2.6-flash-free": ModelSpec(
        id="opencode/mimo-v2.6-flash-free", thinking=True, search=False,
        model_type="default", owned_by="opencode"),
    "opencode/nemotron-3-ultra-free": ModelSpec(
        id="opencode/nemotron-3-ultra-free", thinking=True, search=False,
        model_type="default", owned_by="opencode"),
    "opencode/nemotron-3.5-lightning-free": ModelSpec(
        id="opencode/nemotron-3.5-lightning-free", thinking=True, search=False,
        model_type="default", owned_by="opencode"),
    "opencode/big-pickle": ModelSpec(
        id="opencode/big-pickle", thinking=True, search=False,
        model_type="default", owned_by="opencode"),
}

DEFAULT_MODEL = "opencode/muse-spark-1.3-contributor-free"

# How long one catalogue read (success OR failure) is reused process-wide.
_CACHE_TTL = 300.0
# A failure expires far sooner than a success: a serve that is booting answers
# /api/model slowly (it builds its registry on the first read), and a failed
# read cached for the full TTL would leave /v1/models on the static set for
# five minutes right after the managed instance came up.
_FAIL_TTL = 30.0
# Cold read against localhost: a closed port refuses instantly, so this only
# bounds a hung/blackholed one. Short on purpose -- it is on the /v1/models
# path; the slow first read of a booting serve is primed off-path by
# ``provider.warmup``.
_LIVE_TIMEOUT = 2.0
# The read that follows a boot: a serve builds its model registry on the first
# /api/model call, which takes longer than a route can afford but is a one-off.
_BOOT_READ_TIMEOUT = 30.0

_cache_lock = threading.Lock()
_cached: dict[str, ModelSpec] | None = None
_cached_at = 0.0


def _spec_from_entry(entry: object) -> ModelSpec | None:
    """One ``GET /api/model`` entry -> ModelSpec, or None when unusable.

    Disabled (``enabled: false``) and non-``active`` entries are dropped: a
    model the operator cannot pick must not be advertised.

    ``thinking`` is True only when the entry declares a
    ``compatibility.reasoningField`` -- that field is where the serve puts the
    model's thinking, so its absence means there is no channel to read. The
    free tier declared one for every model except
    ``muse-spark-1.3-contributor-free``, which reports reasoning TOKENS and
    delivers no thinking text at all (live-verified on opencode 2.0.29).
    Variants alone do not imply thinking: that model carries five of them and
    exposes none.
    """
    if not isinstance(entry, dict):
        return None
    provider_id = str(entry.get("providerID") or "").strip()
    model_id = str(entry.get("modelID") or entry.get("id") or "").strip()
    if not provider_id or not model_id:
        return None
    if entry.get("enabled") is False:
        return None
    status = str(entry.get("status") or "").strip().lower()
    if status and status != "active":
        return None
    compatibility = entry.get("compatibility")
    thinking = bool(isinstance(compatibility, dict)
                    and compatibility.get("reasoningField"))
    return ModelSpec(id=f"{provider_id}/{model_id}", thinking=thinking,
                     search=False, model_type="default", owned_by=provider_id)


def _read_live(timeout: float) -> dict[str, ModelSpec]:
    """One ``GET /api/model`` -> specs; ``{}`` on any failure (never raises)."""
    try:
        # Late import: client -> protocol -> managed_serve, and a module
        # import must not boot anything.
        from . import client as OC

        # _request is client.py's shared HTTP seam; it carries the serve
        # basic auth and the identity headers a password-protected
        # instance demands.
        status, data = OC._request("GET", "/api/model", timeout=timeout)
        items = data.get("data")
        if status == 200 and isinstance(items, list):
            specs: dict[str, ModelSpec] = {}
            for entry in items:
                spec = _spec_from_entry(entry)
                if spec is not None:
                    specs[spec.id] = spec
            return specs
    except Exception as exc:  # noqa: BLE001 -- a catalogue miss is not an error
        logger.debug("opencode live catalogue unavailable: %s", exc)
    return {}


def live_specs(timeout: float = _LIVE_TIMEOUT) -> dict[str, ModelSpec]:
    """The serve's live catalogue as ``{providerID/modelID: ModelSpec}``.

    Never raises and never returns a partial read: serve down, timeout, a v1
    route, a junk payload or an empty ``data`` all yield ``{}``, which is the
    caller's signal to use :data:`MODELS_SPEC` instead.

    When nothing answers at the configured base it BOOTS the managed serve and
    reads again. That is what makes the list the same in every process: a fresh
    ``zourcewell models`` and a server that booted before this change report
    the same ids, instead of the fresh ones silently falling back to a
    hand-written list that is stale within days. ``ensure()`` is a no-op when
    the operator pinned their own ``ZOURCEWELL_OPENCODE_BASE`` (it is not ours
    to replace) and when ``ZOURCEWELL_OPENCODE_MANAGED`` is off, which is how
    the hermetic tier keeps subprocesses out of the suite. A MENU row must not
    call this: use :func:`cached_specs`.

    Cached process-wide: :data:`_CACHE_TTL` for a catalogue read,
    :data:`_FAIL_TTL` for a failed one (so a serve that is still booting is
    retried soon, not in five minutes). The lock is held ACROSS the fetch and
    the boot, so N concurrent callers on a cold cache cost one serve, not N.
    """
    global _cached, _cached_at
    now = time.monotonic()
    with _cache_lock:
        fresh = now - _cached_at
        if _cached is not None and fresh < (_CACHE_TTL if _cached
                                            else _FAIL_TTL):
            return dict(_cached)
        specs = _read_live(timeout)
        if not specs:
            try:
                # Late import again: managed_serve imports config.
                from . import managed_serve as MS

                if MS.ensure():
                    # A serve that just booted builds its model registry on the
                    # first read, so this one gets the generous budget.
                    specs = _read_live(_BOOT_READ_TIMEOUT)
            except Exception as exc:  # noqa: BLE001 -- never fails the listing
                logger.debug("opencode managed serve unavailable: %s", exc)
        if not specs:
            logger.debug("opencode live catalogue empty; using the static set")
        _cached, _cached_at = specs, now
        return dict(specs)


def cached_specs() -> dict[str, ModelSpec]:
    """The catalogue this process already knows, without touching anything.

    ``{}`` when nothing has been read yet (or the last read failed), which is
    the caller's signal to use :data:`MODELS_SPEC`. No I/O of any kind and no
    lock: this is what a MENU row may call. A live read against a dead base
    costs a full timeout where loopback connects hang (this codebase has seen
    hosts where a *refused* connect takes seconds), a read that boots the
    managed serve costs the boot on top, and ``_cache_lock`` is held ACROSS
    both -- so waiting for it would inherit exactly the stall this exists to
    avoid.
    """
    # Deliberately lock-free: the only writer REBINDS _cached (never mutates
    # it in place), so a racing read yields an older catalogue or {}, which is
    # all a hint promises.
    cached = _cached
    return dict(cached) if cached else {}
