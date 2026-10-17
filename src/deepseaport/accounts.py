"""Account pool: rotation, cooldowns, and one in-flight stream per account.

Thread-safe. Supports runtime add/remove. Cooldown accounts auto-skipped,
acquire() fails over to next healthy account.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field

from .config import AccountConfig, account_identity
from .core.tokens import extract_token

logger = logging.getLogger("deepseaport.accounts")

COOLDOWN_SECONDS = 120
_POLL_INTERVAL = 0.05

# Historical wording of the pool errors, kept as the fallback so a pool with
# no accounts and no label still says exactly what it always said:
#   "no DeepSeek accounts in pool"
#   "no DeepSeek account free (busy or cooldown)"
DEFAULT_PROVIDER_LABEL = "DeepSeek"
# Provider id assumed when nothing else is known: the migration target for
# accounts with no provider in config, and the fallback when the registry
# cannot answer. Must match the value AccountConfig normalizes an empty
# provider to.
DEFAULT_PROVIDER_ID = "deepseek"


def provider_label(provider_id: object = None) -> str:
    """Human label for a provider id, for user-facing pool errors.

    Registry ``display_name`` when the provider is registered ("DeepSeek",
    "Qwen", "opencode"), else the normalized id. Empty/None resolves to
    "DeepSeek" so the default deployment -- and any pool that cannot name its
    provider -- keeps the historical message byte-for-byte.

    Single home for this mapping: ``server_concurrency._acquire_slot_or_499``
    and the pool itself both route through here instead of hardcoding one
    upstream's name into provider-neutral code.
    """
    key = str(provider_id or "").strip().lower() or "deepseek"
    try:
        from .core import registry as _reg

        name = str(getattr(_reg.get(key), "display_name", "") or "").strip()
        if name:
            return name
    except Exception:
        pass
    return DEFAULT_PROVIDER_LABEL if key == "deepseek" else key


def pool_provider_label(pool) -> str:
    """Provider label for a pool-like object (never raises, never empty).

    Tolerates duck-typed pools and plain fakes. ``provider_label`` (already a
    human label) wins as-is; a bare ``provider``/``provider_id`` is treated as
    an id and resolved through :func:`provider_label`, so a fake carrying
    ``provider="qwen"`` yields "Qwen", not "qwen". Anything else falls back to
    the historical DeepSeek default.
    """
    try:
        ready = str(getattr(pool, "provider_label", None) or "").strip()
    except Exception:
        ready = ""
    if ready:
        return ready
    for attr in ("provider", "provider_id"):
        try:
            raw = getattr(pool, attr, None)
        except Exception:
            continue
        if callable(raw):
            continue
        pid = str(raw or "").strip()
        if pid:
            return provider_label(pid)
    return DEFAULT_PROVIDER_LABEL


def _acquire_remaining(deadline: float) -> float:
    """Remaining wait for one acquire poll, clamped to (_POLL_INTERVAL].

    Single home for the deadline math shared by acquire()/aacquire().
    The wait mechanism stays split on purpose: sync uses the _SLOT_COND
    event wake (release() notifies), async uses asyncio.sleep (a threading
    Condition cannot park the event loop without a real bridge).
    """
    return min(_POLL_INTERVAL, max(0.001, deadline - time.monotonic()))
# Event-driven wake for acquire(): release() notifies so waiters wake
# immediately instead of sleeping the full 50ms poll. Wait timeout stays
# _POLL_INTERVAL so cooldown/ban expiry behaviour is unchanged.
_SLOT_COND = threading.Condition()
# Ban check skips obviously-fake tokens (unit tests use "t1", "tok-...").
# Real userToken values are 64-char; short ones would only waste a request.
MIN_TOKEN_LEN_FOR_BAN_CHECK = 20
BAN_CHECK_TIMEOUT = 12

TOKEN_HELP = (
    "userToken needed per account. Two ways:\n"
    "  AUTO (recommended): python -m deepseaport login --email you@x.com\n"
    "    Obscura browser logs in, captures token, appends to pool.\n"
    "  MANUAL: login at https://chat.deepseek.com in Chrome, then F12 >\n"
    "    Application > Local Storage > https://chat.deepseek.com > userToken.\n"
    "    Copy the `value` field (64-char). Or Console:\n"
    "      JSON.parse(localStorage.getItem('userToken')).value\n"
    "    Then: python -m deepseaport accounts add --email you@x.com --token <value>\n"
    "    Paste raw value OR full JSON; both accepted. "
    "Password-only accounts need browser login via CLI/TUI; direct POST is "
    "fingerprint-gated (RISK_DEVICE_DETECTED)."
)


def token_help_for(provider: str = "deepseek") -> str:
    """Per-provider token help. DeepSeek keeps TOKEN_HELP; others get generic."""
    pid = (provider or "deepseek").strip().lower() or "deepseek"
    if pid == "deepseek":
        return TOKEN_HELP
    return (
        f"{pid} token needed per account. Two ways:\n"
        f"  AUTO: python -m deepseaport login --provider {pid} --email you@x.com\n"
        "    Browser login captures token, appends to pool.\n"
        "  MANUAL: paste raw value OR full JSON; both accepted.\n"
        "    Then: python -m deepseaport accounts add "
        "--email you@x.com --token <value> --provider "
        f"{pid}\n"
    )



def _norm(identifier: str) -> str:
    return (identifier or "").strip().lower()


def _provider_of(cfg: AccountConfig) -> str:
    """Normalized provider id for an account (defaults to deepseek).

    Same rule as ``core.pools._provider_of`` (which partitions the pools) and
    ``config.account_identity`` (which dedupes): one normalization, so a
    scoped match here cannot disagree with the pool an account lands in.
    """
    return (getattr(cfg, "provider", "deepseek") or "deepseek").strip().lower()


def _matches(cfg: AccountConfig, identifier: str) -> bool:
    key = _norm(identifier)
    if not key:
        return False
    if _norm(cfg.email) and _norm(cfg.email) == key:
        return True
    if cfg.mobile and cfg.mobile.strip() == identifier.strip():
        return True
    if _norm(cfg.identifier) == key:
        return True
    # Allow full token match (not prefix) for delete by token.
    if cfg.token and cfg.token.strip() == identifier.strip():
        return True
    return False


def _is_duplicate(a: AccountConfig, b: AccountConfig) -> bool:
    # Same email/mobile/token in different providers is a distinct account.
    # The identity rule itself lives in config.account_identity: load_settings
    # dedupes with it (see _dedupe_accounts), and two definitions of "same
    # account" is how a duplicate gets through one gate and fails the other.
    a_provider, a_keys = account_identity(a)
    b_provider, b_keys = account_identity(b)
    if a_provider != b_provider:
        return False
    return bool(a_keys & b_keys)


@dataclass
class AccountHealth:
    """In-memory health for one pooled account: cooldown + login backoff.

    Persisted ban state stays on ``cfg.banned``/``cfg.banned_until`` so it
    survives restarts; this object is the single in-memory home for the two
    transient timers (cooldown and browser-login backoff). Every mutation
    path (server routes, completion engine, TUI working copy) touches the
    same state instead of parallel dicts/labels.
    """

    bad_until: float = 0.0
    login_until: float = 0.0
    login_reason: str = ""

    def clear(self) -> None:
        self.bad_until = 0.0
        self.login_until = 0.0
        self.login_reason = ""


@dataclass
class PooledAccount:
    cfg: AccountConfig
    lock: threading.Lock = field(default_factory=threading.Lock)
    health: AccountHealth = field(default_factory=AccountHealth)
    uses: int = 0

    def __post_init__(self) -> None:
        # Carry a persisted ban into the in-memory cooldown so a restarted
        # server does not immediately retry a known-banned account.
        try:
            until = float(getattr(self.cfg, "banned_until", 0.0) or 0.0)
        except Exception:
            until = 0.0
        if until > time.time():
            self.health.bad_until = max(self.health.bad_until, until)
        elif getattr(self.cfg, "banned", False):
            self.health.bad_until = max(self.health.bad_until,
                                        time.time() + COOLDOWN_SECONDS)

    @property
    def bad_until(self) -> float:
        """Cooldown deadline. Stored on ``health`` (kept as a property alias)."""
        return self.health.bad_until

    @bad_until.setter
    def bad_until(self, value: float) -> None:
        self.health.bad_until = value

    @property
    def identifier(self) -> str:
        return self.cfg.identifier

    @property
    def busy(self) -> bool:
        return self.lock.locked()

    def login_backoff_skip(self) -> str | None:
        """Remaining-skipping reason when a recent login failure backs off."""
        if self.health.login_until <= 0:
            return None
        now = time.time()
        if now >= self.health.login_until:
            self.health.login_until = 0.0
            self.health.login_reason = ""
            return None
        return (f"recent {self.health.login_reason} "
                f"{int(self.health.login_until - now)}s ago")

    def set_login_backoff(self, reason: str, seconds: float) -> None:
        self.health.login_reason = str(reason or "")
        self.health.login_until = time.time() + max(0.0, float(seconds))

    def clear_login_backoff(self) -> None:
        self.health.login_until = 0.0
        self.health.login_reason = ""

    @property
    def banned(self) -> bool:
        """True while the persisted ban marker is active.

        Expired known-expiry bans are cleared lazily so a restarted process
        doesn't keep an account disabled forever after the suspension ended.
        Unknown-expiry bans stay active until the user clears them.
        """
        if not getattr(self.cfg, "banned", False):
            return False
        try:
            until = float(getattr(self.cfg, "banned_until", 0.0) or 0.0)
        except (TypeError, ValueError):
            until = 0.0
        if until and until <= time.time():
            self.cfg.banned = False
            self.cfg.banned_until = 0.0
            self.bad_until = 0.0
            return False
        return True

    @property
    def cooldown_remaining(self) -> float:
        return max(0.0, self.bad_until - time.time())

    def status(self) -> dict:
        return {
            "identifier": self.identifier,
            "email": self.cfg.email,
            "busy": self.busy,
            "cooldown_remaining": round(self.cooldown_remaining, 1),
            "uses": self.uses,
            "banned": self.banned,
            "banned_until": float(getattr(self.cfg, "banned_until", 0.0) or 0.0),
        }


def sync_current_from_settings(pool: AccountPool, settings) -> bool:
    """Copy settings.active_account into pool. Clears stale selection.

    Returns True when selection changed (caller should settings.save()).
    """
    want = (getattr(settings, "active_account", "") or "").strip()
    if not want:
        if pool.current:
            pool.set_current(None)
            return True
        return False
    cur_item = pool.get_current()
    if cur_item is not None and _matches(cur_item.cfg, want):
        return False
    if pool.set_current(want):
        # Normalize stored id to canonical identifier.
        settings.active_account = pool.current
        return True
    # Stale id (deleted from config): clear.
    settings.active_account = ""
    pool.set_current(None)
    return True


def account_has_token(settings, identifier: str) -> bool:
    """True when settings holds a matching account that carries a token."""
    for a in (getattr(settings, "accounts", None) or []):
        if _matches(a, identifier):
            return bool(a.token)
    return False


def configured_provider_id(settings) -> str:
    """The configured selected provider id, lowercased and never empty.

    Thin lazy wrapper over ``core.pools.selected_provider`` so the callers that
    bind a new account to the configured provider (the route and the CLI) share
    one fallback chain. The import stays inside the function because
    ``core.pools`` reaches back into this module (``pools.make_pool`` imports
    ``..accounts``).
    """
    from .core import pools

    return pools.selected_provider(settings)


def usable_provider(pid: object) -> str:
    """A registered provider id, coercing an unregistered CONFIGURED one.

    For the fallback binding only (the selected provider, when the caller
    expressed no choice of its own). Same contract as
    ``server_routes._usable_provider``, kept here so the HTTP route and the
    CLI/TUI add paths cannot drift again. ``settings.provider`` is not
    registry-validated on load, so a junk id (hand-edited config.json or
    DEEPSEAPORT_PROVIDER) would otherwise be persisted onto every account
    added without an explicit provider. Coerce to the registry default rather
    than refusing the add - the caller is trying to store a credential, and
    the misconfiguration is named in the log instead of blocking the recovery.
    A *registered* id is returned untouched, so a legitimate non-default
    provider selection still wins.

    Never applied to an explicitly chosen provider: the CLI and the TUI accept
    an id the registry does not know (registered later by a plugin-style
    deployment), so that call belongs to the operator.

    ``core.registry`` is imported lazily, the way the route does it: it must
    not be pulled in at ``accounts`` import time (cycle risk with
    ``deepseaport.server``), and a registry that cannot be imported must not
    break account storage.
    """
    candidate = str(pid or "").strip().lower()
    try:
        from .core import registry as _reg
    except Exception:
        # Cannot answer "is it registered": keep the configured id rather than
        # blocking the store (and fall back when there was none).
        return candidate or DEFAULT_PROVIDER_ID
    if candidate and _reg.get(candidate) is not None:
        return candidate
    try:
        fallback = str(_reg.default().id)
    except Exception:
        fallback = DEFAULT_PROVIDER_ID
    if candidate:
        logger.warning("configured provider '%s' is not registered; "
                       "storing the new account as '%s'", candidate, fallback)
    return fallback


def add_account(settings, pool: "AccountPool", cfg: AccountConfig) -> PooledAccount:
    """Add cfg to pool + settings and make it CURRENT. Raises ValueError on dup."""
    item = pool.add(cfg)
    settings.accounts.append(cfg)
    # New account becomes CURRENT (default) immediately.
    settings.active_account = cfg.identifier
    try:
        pool.set_current(cfg.identifier)
    except Exception:
        pass
    return item


def set_account_token(settings, identifier: str, token: str,
                      provider: str | None = None) -> AccountConfig | None:
    """Set normalized token on first matching settings account; return it or None.

    ``provider`` (optional) scopes the match to that provider's accounts: the
    same email is a distinct account per provider (config.account_identity),
    and the previous settings-wide first match wrote the token onto a
    DIFFERENT provider's record -- which that provider's pool then never
    serves, while the other pool serves a token it was not given.
    """
    token = extract_token(token)
    want = (str(provider or "")).strip().lower()
    for a in (getattr(settings, "accounts", None) or []):
        if want and _provider_of(a) != want:
            continue
        if _matches(a, identifier):
            a.token = token
            return a
    return None


def remove_account(settings, identifier: str,
                   provider: str | None = None) -> bool:
    """Drop the first matching account from settings; clear stale CURRENT.

    Mirrors AccountPool.remove() ordering. Returns True when one was removed.
    ``provider`` (optional) scopes the match like ``set_account_token``: the
    same email on two providers is two accounts, and a settings-wide first
    match deleted the other provider's record while the target's survived.
    """
    want = (str(provider or "")).strip().lower()
    kept, dropped = [], False
    for a in (getattr(settings, "accounts", None) or []):
        if (not dropped and (not want or _provider_of(a) == want)
                and _matches(a, identifier)):
            dropped = True
            continue
        kept.append(a)
    if not dropped:
        return False
    settings.accounts = kept
    if settings.active_account and not any(
            _matches(a, settings.active_account) for a in settings.accounts):
        settings.active_account = ""
    return True


def ban_label_for(ban_map: dict, identifier: str) -> tuple[bool, float | None]:
    """Look up a ban expiry: exact key, then case-insensitive fallback.

    Returns (found, mute_until). found is True when the identifier is banned
    even if the expiry is unknown (None).
    """
    if not ban_map:
        return False, None
    if identifier in ban_map:
        return True, ban_map[identifier]
    key = str(identifier).strip().lower()
    for bid, until in ban_map.items():
        if str(bid).strip().lower() == key:
            return True, until
    return False, None


class AccountPool:
    def __init__(self, accounts: list[AccountConfig] | None = None,
                 current: str | None = None,
                 provider: str | None = None) -> None:
        self._items: list[PooledAccount] = [PooledAccount(cfg=a) for a in (accounts or [])]
        self._cursor = 0
        self._guard = threading.Lock()
        self._current: str = (current or "").strip()
        # Explicit label. Provider-scoped pools always pass one
        # (``pools.make_pool``, the account routes); the unscoped management
        # pools (cli/tui) omit it and the id is then derived from the member
        # accounts' own ``provider`` field -- the same field
        # ``pools._provider_of`` / ``accounts_for_provider`` build the pool
        # from. The explicit value is what names the provider of a pool that
        # is EMPTY, where no member can answer the question.
        self._provider_id: str = (provider or "").strip().lower()

    @property
    def provider_id(self) -> str:
        """Provider id this pool serves (explicit, else a member's, else deepseek).

        Read-only view; ``threading.Lock`` is not reentrant, so callers that
        already hold ``_guard`` must not use this (see ``_next_ordered``).
        """
        if self._provider_id:
            return self._provider_id
        with self._guard:
            for item in self._items:
                pid = str(getattr(item.cfg, "provider", "") or "").strip().lower()
                if pid:
                    return pid
        return "deepseek"

    @property
    def provider_label(self) -> str:
        """Human label for :attr:`provider_id` ("DeepSeek", "Qwen", ...)."""
        return provider_label(self.provider_id)

    def no_accounts_error(self) -> str:
        """"no <label> accounts in pool" -- 503 text, same shape as before."""
        return f"no {self.provider_label} accounts in pool"

    def no_free_error(self) -> str:
        """"no <label> account free (busy or cooldown)" -- 429/timeout text."""
        return f"no {self.provider_label} account free (busy or cooldown)"

    @property
    def current(self) -> str:
        with self._guard:
            return self._current

    def set_current(self, identifier: str | None) -> bool:
        """Select CURRENT account. None/empty clears to auto-failover mode.

        Returns False when identifier not found (selection unchanged).
        """
        key = (identifier or "").strip()
        with self._guard:
            if not key:
                self._current = ""
                return True
            for item in self._items:
                if _matches(item.cfg, key):
                    self._current = item.identifier
                    logger.info("current account -> %s", item.identifier)
                    return True
            return False

    def get_current(self) -> PooledAccount | None:
        with self._guard:
            cur = self._current
            items = list(self._items)
        if not cur:
            return None
        for item in items:
            if _matches(item.cfg, cur):
                return item
        return None

    def __len__(self) -> int:
        with self._guard:
            return len(self._items)

    def add(self, cfg: AccountConfig) -> PooledAccount:
        """Add account. Raises ValueError on empty/duplicate."""
        if not (cfg.email or cfg.mobile or cfg.token):
            raise ValueError("account needs email, mobile, or token")
        with self._guard:
            for item in self._items:
                if _is_duplicate(item.cfg, cfg):
                    # NEVER item.identifier: for a token-only account that is
                    # ``token[:10] + "..."`` (config.AccountConfig.identifier),
                    # and this ValueError is surfaced verbatim as the HTTP 409
                    # detail on POST /v1/accounts. Same rule as
                    # server_routes._reject_token_shaped_identifier -- nothing
                    # derived from the token reaches a response. The email or
                    # mobile number is the identity the operator typed and is
                    # already echoed in URLs by design.
                    raise ValueError("account already in pool: "
                                     + (item.cfg.email or item.cfg.mobile
                                        or "a token-only account with no identifier"))
            item = PooledAccount(cfg=cfg)
            self._items.append(item)
            logger.info("account added: %s (pool=%d)", item.identifier, len(self._items))
            return item

    def remove(self, identifier: str) -> bool:
        """Delete account by email/mobile/identifier/token. True if removed.

        Safe while account in-flight: holder keeps its ref, release() still works.
        Clears CURRENT selection when deleted account was selected.
        """
        with self._guard:
            for i, item in enumerate(self._items):
                if _matches(item.cfg, identifier):
                    del self._items[i]
                    if self._current and _matches(item.cfg, self._current):
                        self._current = ""
                    logger.info("account removed: %s (pool=%d)",
                                item.identifier, len(self._items))
                    return True
        return False

    def get(self, identifier: str) -> PooledAccount | None:
        with self._guard:
            for item in self._items:
                if _matches(item.cfg, identifier):
                    return item
        return None

    def status(self) -> list[dict]:
        with self._guard:
            items = list(self._items)
            cur = self._current
        rows = [item.status() for item in items]
        for row, item in zip(rows, items):
            row["current"] = bool(cur) and _matches(item.cfg, cur)
        return rows

    def clear_cooldown(self, identifier: str | None = None) -> int:
        """Clear cooldown, persisted ban, AND login backoff (one/all).

        The single unblock entry point: every part of ``AccountHealth`` is
        reset here so the REST unblock/set-token paths cannot leave a stale
        login backoff behind."""
        n = 0
        with self._guard:
            for item in self._items:
                if identifier is None or _matches(item.cfg, identifier):
                    if (item.health.bad_until > 0
                            or getattr(item.cfg, "banned", False)
                            or item.health.login_until > 0):
                        n += 1
                    item.health.clear()
                    try:
                        item.cfg.banned = False
                        item.cfg.banned_until = 0.0
                    except Exception:
                        pass
        if n:
            logger.info("cooldown/ban/login state cleared for %d account(s)", n)
        return n

    def _next_ordered(self, allow_failover: bool
                      ) -> tuple[list[PooledAccount], float]:
        """Snapshot + CURRENT-first ordering; advances the round-robin cursor.

        Single source for acquire()/aacquire() rotation, CURRENT pinning, and
        failover ordering. Raises ValueError when the pool is empty.

        The empty check and the snapshot are two separate guarded reads on
        purpose: :attr:`provider_id` needs ``_guard`` to look at the members,
        and ``_guard`` is a plain (non-reentrant) Lock, so resolving the label
        inside the ``with`` would self-deadlock on the empty-pool error path.
        An account added between the two reads simply takes the normal path.
        """
        with self._guard:
            if not self._items:
                empty = True
            else:
                empty = False
        if empty:
            raise ValueError(self.no_accounts_error())
        with self._guard:
            snapshot = list(self._items)
            cur = self._current
            start = self._cursor % len(snapshot)
            self._cursor += 1
        now = time.time()
        ordered = snapshot[start:] + snapshot[:start]
        if cur:
            preferred = [it for it in snapshot if _matches(it.cfg, cur)]
            if preferred and not allow_failover:
                ordered = preferred
            else:
                rest = [it for it in ordered if not _matches(it.cfg, cur)]
                ordered = preferred + rest
        return ordered, now

    def _try_acquire_once(self, ordered: list[PooledAccount], now: float,
                          ) -> PooledAccount | None:
        """One nonblocking pass over ordered; holder or None.

        Skips cooldown/banned, takes the first free lock, then rechecks pool
        membership + bad_until under race (releasing on either) so a removed
        or newly-cooled account never leaks a held lock to the caller.
        """
        for item in ordered:
            with self._guard:
                if item not in self._items:
                    continue  # removed concurrently
            if item.bad_until > now or item.banned:
                continue  # on cool/ban -> fail over to next
            if item.lock.acquire(blocking=False):
                with self._guard:
                    if item not in self._items:
                        item.lock.release()
                        continue
                if item.bad_until > time.time():
                    item.lock.release()  # marked bad while racing
                    continue
                return item
        return None

    def acquire(self, timeout: float = 60, allow_failover: bool = True) -> PooledAccount:
        """Return account with lock held. Skips busy + cooldown, round-robin.

        CURRENT account (when set) tried first, others fail over after it.
        When allow_failover=False and CURRENT is in the pool, only CURRENT
        is considered (sticky: concurrent requests queue on it instead of
        spreading to other accounts). Busy is per-account lock state.
        Raises ValueError when pool empty, TimeoutError when all busy/cooldown
        for longer than timeout. Both messages name the pool's provider
        (:attr:`provider_label`); only the wording is provider-aware, the
        exception TYPES are the frozen contract callers branch on.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            got = self._try_acquire_once(*self._next_ordered(allow_failover))
            if got is not None:
                return got
            if time.monotonic() >= deadline:
                raise TimeoutError(self.no_free_error())
            # Event-driven: wake immediately on release(), else re-poll after
            # _POLL_INTERVAL (same timeout/behavior as the old time.sleep).
            remaining = _acquire_remaining(deadline)
            with _SLOT_COND:
                _SLOT_COND.wait(timeout=remaining)

    async def aacquire(self, timeout: float = 60, allow_failover: bool = True) -> PooledAccount:
        """Async acquire: same rotation/skip semantics, no worker thread held.

        Polls with asyncio.sleep so the default executor stays free for real
        blocking work (challenge fetch, parse, status, save). Used by the
        async FastAPI endpoint; sync worker threads keep using acquire().
        allow_failover=False pins to CURRENT (see acquire()).
        """
        deadline = time.monotonic() + max(0.0, timeout)
        # ponytail: async side keeps plain asyncio.sleep polling; only the
        # sync acquire() path uses the _SLOT_COND wake (mixing a threading
        # Condition into the event loop needs a real bridge, deferred).
        while True:
            got = self._try_acquire_once(*self._next_ordered(allow_failover))
            if got is not None:
                return got
            if time.monotonic() >= deadline:
                raise TimeoutError(self.no_free_error())
            await asyncio.sleep(_acquire_remaining(deadline))

    @staticmethod
    def release(item: PooledAccount) -> None:
        # ponytail: uses increment is intentionally unlocked (stats counter
        # only, read in status()); taking item.lock here would deadlock the
        # normal held-lock path (Lock is not reentrant).
        item.uses += 1
        try:
            item.lock.release()
        except RuntimeError:
            pass  # already released / never acquired
        finally:
            try:
                with _SLOT_COND:
                    _SLOT_COND.notify_all()
            except Exception:
                pass

    @staticmethod
    def mark_bad(item: PooledAccount, seconds: float = COOLDOWN_SECONDS) -> None:
        item.bad_until = time.time() + max(0.0, seconds)
        logger.warning("account %s cooling down for %ds", item.cfg.identifier, int(seconds))

    @staticmethod
    def mark_banned(item: PooledAccount, until: float | None = None,
                    default_seconds: float = COOLDOWN_SECONDS) -> None:
        """Mark an account banned and remember its expiry when known.

        ``until=None`` means the page/API proved a ban but did not provide an
        expiry; the account is still marked (CLI/TUI show ``(BANNED)``) and
        cooled down for ``default_seconds`` so the pool skips it.
        """
        now = time.time()
        try:
            deadline = float(until) if until is not None else 0.0
        except (TypeError, ValueError):
            deadline = 0.0
        if deadline > now:
            cooldown = deadline - now
            item.cfg.banned_until = deadline
        else:
            cooldown = max(0.0, float(default_seconds))
            item.cfg.banned_until = 0.0
        item.cfg.banned = True
        item.bad_until = now + cooldown
        logger.warning("account %s marked banned%s", item.cfg.identifier,
                       f" until {int(deadline)}" if deadline > now else "")


def check_ban_for_token(token: str, timeout: int = BAN_CHECK_TIMEOUT,
                        provider: str = "deepseek") -> tuple[bool, float | None]:
    """Best-effort ban probe for one token. Never raises; (False, None) on skip/fail.

    Routes through ``Provider.probe_ban`` (registry) so non-DeepSeek accounts
    use their own probe instead of the DeepSeek users/current shape. Defaults
    preserve the historical DeepSeek behaviour (bridge headers + users/current)
    so TUI/tests are unchanged.

    Skips empty/short (test) tokens without network.
    """
    tok = (token or "").strip()
    if len(tok) < MIN_TOKEN_LEN_FOR_BAN_CHECK:
        return False, None
    try:
        # Local import: registry/client pull network sessions, not needed at import.
        # Delegates to Provider.probe_ban; DeepSeek header shape is lazy so
        # this module has no top-level providers.deepseek import.
        from .core import registry as _reg

        pid = (provider or "deepseek").strip().lower() or "deepseek"
        prov = _reg.get(pid)
        if prov is None:
            prov = _reg.get("deepseek")
            pid = "deepseek" if prov is not None else pid
        headers: dict
        if pid == "deepseek":
            try:
                from .providers.deepseek.protocol import bridge_headers as _bh
                headers = _bh(bearer=tok)
            except Exception:
                headers = {"Authorization": f"Bearer {tok}"}
        else:
            headers = {"Authorization": f"Bearer {tok}"}
        if prov is None:
            # Registry unavailable: narrow legacy direct DeepSeek path.
            from .providers.deepseek import client as _DS

            return _DS.check_ban(headers, timeout=timeout)
        return prov.probe_ban(headers, timeout=timeout)
    except Exception as exc:
        logger.debug("ban check ignored: %s", exc)
        return False, None


# TTL cache for ban labels: the TUI menu redraws on every keypress, and a
# network probe per redraw blocks the menu (2 accounts ~1s, offline up to
# 12s). Cache for BAN_LABEL_TTL seconds; TUI reads the cache instantly and
# refreshes in background so menu render never blocks on network.
BAN_LABEL_TTL = 60.0
# Bound on parallel ban-probe threads: one short-lived thread per account,
# capped so a 50-account pool never piles 50 concurrent probes/threads.
_BAN_PROBE_MAX_WORKERS = 8
_BAN_CACHE_LOCK = threading.Lock()
_BAN_CACHE: dict = {"ts": 0.0, "key": None, "result": {}}
_BAN_REFRESH_INFLIGHT = False


def _persisted_ban_labels(accounts: list[AccountConfig] | None) -> dict[str, float | None]:
    """Ban labels already persisted in config, no network required."""
    now = time.time()
    out: dict[str, float | None] = {}
    for a in (accounts or []):
        if not getattr(a, "banned", False):
            continue
        try:
            until = float(getattr(a, "banned_until", 0.0) or 0.0)
        except (TypeError, ValueError):
            until = 0.0
        if until and until <= now:
            # Expired persisted ban: clear it on first access.
            a.banned = False
            a.banned_until = 0.0
            continue
        out[a.identifier] = until if until > now else None
    return out


def _ban_targets(accounts: list[AccountConfig] | None) -> tuple[tuple[str, str], ...]:
    """Sorted (identifier, token) pairs worth live-probing.

    Persisted bans are skipped; their state is already known locally and the
    CLI/TUI should show it instantly even when the account has no token.
    """
    try:
        targets = []
        for a in (accounts or []):
            if getattr(a, "banned", False):
                continue
            token = (a.token or "").strip()
            if len(token) >= MIN_TOKEN_LEN_FOR_BAN_CHECK:
                targets.append((a.identifier or "", token))
        return tuple(sorted(targets))
    except Exception:
        return ()


def _ban_cache_key(accounts: list[AccountConfig] | None) -> tuple:
    """Cache key covering persisted ban state + probeable tokens."""
    try:
        return tuple(sorted(
            (a.identifier or "", bool(getattr(a, "banned", False)),
             float(getattr(a, "banned_until", 0.0) or 0.0),
             (a.token or "").strip())
            for a in (accounts or [])
        ))
    except Exception:
        return ()


def get_cached_ban_labels(accounts: list[AccountConfig] | None) -> dict[str, float | None]:
    """Instant, never-network ban labels (persisted + TTL-cached live probes)."""
    try:
        out = _persisted_ban_labels(accounts)
        key = _ban_cache_key(accounts)
        if not key:
            return out
        with _BAN_CACHE_LOCK:
            if _BAN_CACHE.get("key") == key:
                out.update(_BAN_CACHE.get("result") or {})
        return out
    except Exception:
        return {}


def refresh_ban_labels_background(accounts: list[AccountConfig] | None,
                                  timeout: int = BAN_CHECK_TIMEOUT) -> None:
    """Refresh live ban labels in a daemon thread unless fresh/refreshing."""
    global _BAN_REFRESH_INFLIGHT
    try:
        key = _ban_cache_key(accounts)
        if not key:
            return
        with _BAN_CACHE_LOCK:
            if _BAN_REFRESH_INFLIGHT:
                return
            try:
                fresh = ((time.time() - float(_BAN_CACHE.get("ts") or 0)) < BAN_LABEL_TTL
                         and _BAN_CACHE.get("key") == key)
            except Exception:
                fresh = False
            if fresh:
                return
            _BAN_REFRESH_INFLIGHT = True
        snapshot = list(accounts or [])

        def _run() -> None:
            global _BAN_REFRESH_INFLIGHT
            try:
                collect_ban_labels(snapshot, timeout=timeout, force_refresh=True)
            except Exception:
                pass
            finally:
                with _BAN_CACHE_LOCK:
                    _BAN_REFRESH_INFLIGHT = False

        threading.Thread(target=_run, daemon=True,
                         name="ds-ban-refresh").start()
    except Exception:
        pass


def collect_ban_labels(accounts: list[AccountConfig],
                       timeout: int = BAN_CHECK_TIMEOUT,
                       ttl: float = BAN_LABEL_TTL,
                       force_refresh: bool = False) -> dict[str, float | None]:
    """Map identifier -> mute_until for banned accounts only (parallel, best-effort).

    Empty dict when none banned, offline, or tokens missing/short. Never raises.
    Used by TUI/CLI account managers to render red '(BANNED: D Month)'.
    Results are TTL-cached (default 60s) so menu redraws don't probe network.
    """
    import concurrent.futures as _fut

    persisted = _persisted_ban_labels(accounts)
    targets = list(_ban_targets(accounts))
    key = _ban_cache_key(accounts)
    if not targets:
        return persisted
    if not force_refresh and ttl and ttl > 0:
        try:
            with _BAN_CACHE_LOCK:
                if (_BAN_CACHE.get("key") == key
                        and (time.time() - float(_BAN_CACHE.get("ts") or 0)) < float(ttl)):
                    out = dict(persisted)
                    out.update(_BAN_CACHE.get("result") or {})
                    return out
        except Exception:
            pass
    out: dict[str, float | None] = dict(persisted)
    # Per-account provider so each token probes through its own
    # Provider.probe_ban instead of the DeepSeek users/current shape.
    provider_by_pair: dict[tuple[str, str], str] = {}
    try:
        for _a in (accounts or []):
            provider_by_pair[((_a.identifier or ""),
                              (_a.token or "").strip())] = (
                getattr(_a, "provider", "deepseek") or "deepseek")
    except Exception:
        provider_by_pair = {}

    def _one(pair: tuple[str, str]) -> tuple[str, bool, float | None]:
        ident, tok = pair[0], pair[1]
        prov = provider_by_pair.get((ident, (tok or "").strip()), "deepseek")
        try:
            banned, until = check_ban_for_token(tok, timeout=timeout,
                                               provider=prov)
            return ident, banned, until
        except Exception:
            return ident, False, None

    try:
        with _fut.ThreadPoolExecutor(
                max_workers=min(_BAN_PROBE_MAX_WORKERS, len(targets)),
                thread_name_prefix="ds-ban-probe") as ex:
            for ident, banned, until in ex.map(_one, targets):
                if banned:
                    out[ident] = until
    except Exception as exc:
        logger.debug("collect_ban_labels ignored: %s", exc)
    try:
        with _BAN_CACHE_LOCK:
            _BAN_CACHE["key"] = key
            _BAN_CACHE["ts"] = time.time()
            _BAN_CACHE["result"] = dict(out)
    except Exception:
        pass
    return out
