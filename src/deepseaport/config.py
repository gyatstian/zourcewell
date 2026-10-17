"""Config: config.json + environment, no secrets committed."""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .core.tokens import extract_token

logger = logging.getLogger("deepseaport.config")

DEFAULT_PORT = 5001
PACKAGE_DIR = Path(__file__).resolve().parent

# Documented ceiling for Settings.max_retries (the TUI asks for 0-5). Enforced
# at load time because the counter multiplies upstream work: one request costs
# up to max_retries session+PoW+stream attempts, each holding an account slot,
# so an unclamped value ("max_retries": 1000000) turns one call into millions
# of upstream attempts. A hand-edited config.json or a stray env var is the
# only way to reach it now, so the clamp is the last line of defence.
#
# The clamp is RUNTIME-ONLY and must stay that way: save() runs on every account
# add/select/unblock/token-refresh and every TUI toggle, so persisting the
# clamped number would silently rewrite the operator's config.json (this
# checkout holds "max_retries": 7) on the first unrelated action. load_settings
# records the pre-clamp value in _RAW_PERSISTED_VALUES and save() writes that
# back instead of the clamp.
MAX_RETRIES_LIMIT = 5

# Upper bound for a TCP port, applied at load time for the same "last line of
# defence" reason as MAX_RETRIES_LIMIT above: config.json and DEEPSEAPORT_PORT
# are hand-edited and unvalidated (the TUI checks 1-65535, nothing else does),
# and neither end of the legal range is survivable.
#   * port 0 binds an OS-assigned ephemeral port while the UI keeps printing
#     "http://127.0.0.1:0", so every request fails with [WinError 10061]
#     connection-refused and nothing points at the config that caused it.
#   * port > 65535 reaches socket.bind() and raises OverflowError, which is
#     NOT an OSError, so it slips past the "except OSError" handlers around
#     the bind path and kills the TUI with a traceback.
# So: clamp above to 65535, and treat 0 as "no port chosen" -> DEFAULT_PORT.
# 0 is a legitimate value for bind(0) ("give me any free port") but never in a
# config file, where it silently desynchronizes the port actually bound from
# the one displayed.
PORT_LIMIT = 65535

# Mirror of server_routes.MAX_ACCOUNTS (server_routes.py:55), which is the
# enforced cap on POST /v1/accounts. config cannot import server_routes (that
# module imports config), so the number is repeated here with a comment; keep
# the two in sync if either moves. Load-time only enforces NOTHING -- see the
# comment in load_settings for why dropping accounts at load would destroy
# credentials on the next save().
MAX_ACCOUNTS_LIMIT = 64

# Serializes the config.json read-modify-write within the process. Callers span
# the event loop (endpoint saves) and worker threads (_ensure_token / account
# pool). It must cover the accounts SNAPSHOT as well as the write, not just the
# bytes: a lock on the write alone lets two mutators each build a payload from a
# stale list and the second one silently overwrites the other's account. The
# atomic temp-file + os.replace below is the other half -- this lock keeps the
# snapshot and the write together, the replace keeps readers off a half-written
# file.
# NOTE: process-local only. Separate OS processes (uvicorn --workers N) each
# have their own lock; atomic os.replace keeps the file valid JSON but the
# last writer wins (a ban/token saved in one worker can be overwritten by a
# concurrent save from another). No file lock: it would add a Windows
# dependency + stale-lock risk without fixing last-writer-wins.
# Plain Lock, NOT an RLock: save() acquires it exactly once, so a second
# `with _CONFIG_SAVE_LOCK` inside save() self-deadlocks rather than nesting.
_CONFIG_SAVE_LOCK = threading.Lock()


def _app_dir() -> Path:
    # Frozen exe: exe location; source: repo root.
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return PACKAGE_DIR.parent.parent


REPO_DIR = _app_dir()
DATA_DIR = Path(os.environ.get("DEEPSEAPORT_DATA", str(REPO_DIR / "data")))


@dataclass(repr=False)
class AccountConfig:
    email: str = ""
    mobile: str = ""
    password: str = ""
    token: str = ""
    # Qwen signin returns ``expires_at`` (unix ts) alongside the JWT. Persist
    # it so the engine can pre-check expiry and auto re-login via the cheap
    # direct POST instead of surfacing 401s. 0.0 = unknown (e.g. pasted
    # tokens, DeepSeek accounts); never blocks, only enables proactivity.
    token_expires_at: float = 0.0
    # Persisted ban marker so CLI/TUI labels survive process restarts and
    # accounts whose token is missing. ``banned_until`` is 0 when unknown.
    banned: bool = False
    banned_until: float = 0.0
    # Owning provider id. Accounts with no provider in config migrate to
    # "deepseek" so existing deployments keep their single-provider behaviour.
    provider: str = "deepseek"

    def __setattr__(self, name, value):
        # Normalize at every assignment, not just construction. Runtime token
        # refresh, TUI set-token, and API endpoints all mutate this field and
        # must never leak the full JSON wrapper into an Authorization header.
        if name == "token":
            value = extract_token(value)
        if name == "provider":
            # Hand-edited config.json can hold a non-str provider (e.g. 123);
            # without this, pools._provider_of crashes on .strip().
            value = str(value or "deepseek").strip().lower() or "deepseek"
        super().__setattr__(name, value)

    def __post_init__(self) -> None:
        # Normalize pasted JSON shapes (including {"value":null}) at the
        # boundary so the rest of the app only ever sees a raw token or "".
        self.token = extract_token(self.token)
        try:
            self.token_expires_at = max(0.0, float(self.token_expires_at or 0.0))
        except (TypeError, ValueError):
            self.token_expires_at = 0.0
        try:
            self.banned_until = max(0.0, float(self.banned_until or 0.0))
        except (TypeError, ValueError):
            self.banned_until = 0.0
        # Expired persisted bans shouldn't hide a usable account forever.
        if self.banned and self.banned_until and self.banned_until <= time.time():
            self.banned = False
            self.banned_until = 0.0

    @property
    def identifier(self) -> str:
        return (self.email or self.mobile or (self.token[:10] + "..." if self.token else "?")).strip()

    def __repr__(self) -> str:
        """Redacted repr: this object holds a plaintext password and a 64-char
        bearer token.

        The dataclass-generated repr printed both verbatim, and there is a real
        sink for it today even though no production site calls repr() on an
        account: pytest's assertion output. ``assert loaded.accounts == [...]``
        or any Settings comparison renders the left side with repr(), so the
        secrets of whichever config the suite happens to run against land in
        the failure diff - and CI logs are exactly where a credential should
        not be. The other sink is one careless
        ``logger.debug("settings: %s", settings)`` away, so the guard is here
        rather than at the (nonexistent) call site.

        Everything non-secret stays visible: the fields that decide routing and
        pool behaviour (provider, ban markers, expiry) are what a debugging
        print is actually for.
        """
        return (f"AccountConfig(email={self.email!r}, mobile={self.mobile!r}, "
                f"password={_masked(self.password)}, token={_masked(self.token)}, "
                f"token_expires_at={self.token_expires_at!r}, "
                f"banned={self.banned!r}, banned_until={self.banned_until!r}, "
                f"provider={self.provider!r})")


def _masked_keys(keys) -> str:
    """``['<set x2>']`` for the API key list - count only, never a value."""
    try:
        n = len(keys or [])
    except TypeError:  # pragma: no cover - defensive
        return "<unknown>"
    return f"['<set x{n}>']" if n else "[]"


def _masked(value: str) -> str:
    """Presence-only rendering of a secret for __repr__ (``<set>``/``<empty>``).

    Presence, not a length or a prefix: even "token is 64 chars" is a
    fingerprint of the credential, and the existing prefix helper
    (``AccountConfig.identifier`` -> ``token[:10] + "..."``) is fine for a
    user-facing label but wrong for an assertion diff that may be pasted
    somewhere.
    """
    return "<set>" if value else "<empty>"


VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
VALID_STREAM_MODES = ("buffered", "live")

# Single source of truth for Settings.save(): every persistable scalar/list
# field. ``accounts`` (custom shape) and ``config_path`` (never persisted)
# are excluded. Adding a dataclass field without adding it here means it
# silently stops persisting, so keep this beside the dataclass.
_SETTINGS_SAVE_FIELDS = (
    "keys", "active_account", "obscura_bin", "obscura_profile",
    "browser_bin", "browser_headless", "port", "listen",
    "enable_tools", "warmup_on_startup", "auto_delete_session",
    "max_retries", "parallel_challenge_fetch", "tool_args_max_chars",
    "forgiving_toolcalls", "use_multiple_accounts", "log_level",
    "stream_mode", "chat_model", "enable_append", "append_top",
    "append_bottom",     "send_dry_run_to_frontend", "provider", "providers",
)

# Name of the plain instance attribute holding the clamp bookkeeping for the
# fields load_settings had to bound: ``{"max_retries": (9, 5)}`` = the
# operator's own value 9, the effective clamped value 5. save() writes 9 back
# while the runtime value is still 5, and the new value once the operator
# changes it deliberately (TUI "Retry count" -> 2 must persist 2, not 9).
#
# Deliberately NOT a dataclass field:
#   * asdict()/replace()/eq/repr stay about the effective settings, and the
#     bookkeeping can never be serialised by accident;
#   * ``__dataclass_fields__`` remains exactly the set of *persistable
#     candidate* fields, which is the contract the mirror guard in save()
#     checks - adding a real field without a _SETTINGS_SAVE_FIELDS entry must
#     still warn, so the guard is left untouched rather than widened.
# A module-level WeakKeyDictionary keyed by Settings was the alternative and
# was rejected: it adds an import plus a GC/lifetime dependency to a
# dataclass that already has a __dict__, and it hides the value from any
# future debug/repr tooling. The attribute is private and read in exactly one
# place (save()).
_RAW_PERSISTED_VALUES = "_raw_persisted_values"


@dataclass(repr=False)
class Settings:
    keys: list[str] = field(default_factory=list)
    accounts: list[AccountConfig] = field(default_factory=list)
    active_account: str = ""
    obscura_bin: str = ""
    obscura_profile: str = ""
    # Real-browser login: "" disables it; load_settings() defaults a fresh
    # config to "auto" so Chrome/Helium discovery works out of the box.
    # The browser is used only for interactive login/token refresh, never the
    # hot path; Obscura still owns WAF cookies.
    browser_bin: str = ""
    browser_headless: bool = False
    port: int = DEFAULT_PORT
    # true -> bind 0.0.0.0 (all interfaces); false -> 127.0.0.1 only.
    # --host CLI flag overrides both.
    listen: bool = False
    config_path: str = ""
    # TUI-tunable behaviour. Defaults preserve the historical behaviour.
    enable_tools: bool = True
    warmup_on_startup: bool = True
    auto_delete_session: bool = True
    max_retries: int = 1
    parallel_challenge_fetch: bool = True
    # Reject tool-call arguments above this many characters before the
    # harness sees them. 200k keeps large single-file writes possible while
    # still bounding memory; the old 8k default dropped ordinary file writes.
    tool_args_max_chars: int = 200_000
    # More forgiving toolcalls: opt-in repair bucket for model markup slips
    # (pipe/space/HTML-escaped markers, smart attribute delimiters, single-
    # tool missing-name/param inference, JSON invoke bodies).
    # False (default) preserves strict historical parsing; True enables the
    # bucket. Future repairs land here too, so one toggle covers them all.
    forgiving_toolcalls: bool = False
    # True -> busy CURRENT fails over to another healthy account (parallel
    # subagents / multitasking). False -> stick to CURRENT, queue on it.
    use_multiple_accounts: bool = True
    log_level: str = "INFO"
    stream_mode: str = "live"
    # Last /model picked in server-with-chat mode. Validated in chat_ui
    # (MODELS lives in server.py; importing it here would cycle).
    chat_model: str = "deepseek-flash"
    # Master switch for the custom wrapper below. False -> _prepare skips
    # both append_top and append_bottom (texts preserved, just not sent).
    enable_append: bool = True
    # Custom text wrapped around every outgoing prompt. append_top goes at
    # the very top (before everything, incl. tool instructions), append_bottom
    # at the very bottom (after everything, incl. tool reminder). Free-form
    # strings; internal newlines preserved so multi-line blocks work.
    # Edit directly in config.json ("append_top": "line1\nline2") — "" disables.
    append_top: str = ""
    append_bottom: str = ""
    # Dry run: True -> the echo response carries what dry run prints
    # (raw body + formatted prompt). False -> slim metadata-only response;
    # the full dump still prints on the server terminal.
    send_dry_run_to_frontend: bool = True
    # Selected provider id (single-provider mode) and optional multi-provider
    # list (empty = use ``provider`` only).
    provider: str = "deepseek"
    providers: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        """Redacted repr; see AccountConfig.__repr__ for the threat model.

        ``keys`` are API keys and every account carries a password + token, so
        the dataclass-generated repr of a Settings printed the operator's
        complete credential set. Concrete sink today: pytest assertion output
        (any comparison of Settings or of a list containing one), which CI
        records. ``str(s)`` follows __repr__ - there is no __str__ - so
        ``logger.info("loaded %s", settings)`` would leak identically.

        accounts keep their own redacted repr, so account emails/providers stay
        visible while the secrets do not. ``config_path`` is included because
        which file was loaded is the first question when two configs disagree.
        """
        return (f"Settings(keys={_masked_keys(self.keys)}, "
                f"accounts={self.accounts!r}, "
                f"active_account={self.active_account!r}, "
                f"port={self.port!r}, listen={self.listen!r}, "
                f"config_path={self.config_path!r}, "
                f"enable_tools={self.enable_tools!r}, "
                f"max_retries={self.max_retries!r}, "
                f"use_multiple_accounts={self.use_multiple_accounts!r}, "
                f"log_level={self.log_level!r}, stream_mode={self.stream_mode!r}, "
                f"chat_model={self.chat_model!r}, "
                f"provider={self.provider!r}, providers={self.providers!r})")

    def save(self) -> None:
        if not self.config_path:
            return
        # Mirror guard: a new dataclass field without a _SETTINGS_SAVE_FIELDS
        # entry silently stops persisting. Loud warning, not an exception
        # (endpoints must not 500 on a field addition). Intentionally NOT
        # widened for the private _raw_persisted_values bookkeeping attribute
        # (see _RAW_PERSISTED_VALUES): it is not a field, and keeping it out
        # preserves the guard's meaning.
        try:
            _known = set(_SETTINGS_SAVE_FIELDS) | {"accounts", "config_path"}
            _missing = [n for n in self.__dataclass_fields__ if n not in _known]
            if _missing:
                logger.warning("Settings field(s) not persisted (add to "
                               "_SETTINGS_SAVE_FIELDS): %s", sorted(_missing))
        except Exception:
            pass
        # Clamp bookkeeping for the fields the loader bounded, if any. Read
        # once, never mutated here, so concurrent saves stay consistent.
        _raw_persisted = getattr(self, _RAW_PERSISTED_VALUES, None) or {}
        # ONE critical section for the whole read-modify-write. The accounts
        # snapshot below must be taken under the same lock as the write, or two
        # concurrent mutators interleave and the second write persists a payload
        # built from a stale list: a valid, atomically replaced config.json that
        # silently lost an account (POST /v1/accounts racing DELETE
        # /v1/accounts/{id}). This is a plain Lock, so exactly one acquisition
        # may span the block -- never nest a second `with _CONFIG_SAVE_LOCK`.
        with _CONFIG_SAVE_LOCK:
            # Payload order mirrors the historical hand-built dict (keys,
            # accounts, then scalars) so on-disk JSON stays diff-stable.
            payload: dict = {"keys": self.keys}
            payload["accounts"] = [
                {"email": a.email, "mobile": a.mobile, "password": a.password,
                 "token": extract_token(a.token),
                 "token_expires_at": float(getattr(a, "token_expires_at", 0.0) or 0.0),
                 "banned": bool(getattr(a, "banned", False)),
                 "banned_until": float(getattr(a, "banned_until", 0.0) or 0.0),
                 "provider": getattr(a, "provider", "deepseek") or "deepseek"}
                for a in self.accounts
            ]
            for _key in _SETTINGS_SAVE_FIELDS:
                if _key == "keys":
                    continue  # already first, keeps historical order
                # Non-destructive clamp: while the runtime value is still the
                # clamp, persist the operator's own number so an unrelated save
                # cannot rewrite what they wrote. Once it differs (TUI toggle,
                # endpoint), the new value is the intentional one - persist it.
                _val = getattr(self, _key)
                _entry = _raw_persisted.get(_key)
                if _entry is not None and _val == _entry[1]:
                    _val = _entry[0]
                payload[_key] = list(_val) if isinstance(_val, list) else _val
            data = json.dumps(payload, ensure_ascii=False, indent=2)
            target = Path(self.config_path)
            # Atomic write: unique temp file in the same dir, then os.replace.
            # Concurrent writers (event loop + worker threads) can't interleave
            # or observe a half-written config. If the rename is blocked (see
            # below) we degrade to an in-place rewrite.
            # Never overwrite a file whose last load found it unparsable AND
            # that is still on disk: the in-memory Settings would be the empty
            # fallback, so this write would replace the operator's accounts
            # (and the .bak would then hold the good copy, but only if it had
            # ever been saved). Raise rather than destroy -- the caller can
            # surface it, and the file on disk is still recoverable. The
            # exists() half matters because a running process never reloads:
            # after the operator DELETES the file (the documented way out)
            # nothing would ever clear the latch, so every later save -- an
            # account add that already mutated memory -- kept 500ing and the
            # file was never recreated.
            if str(target) in _UNPARSABLE_CONFIG and target.exists():
                raise RuntimeError(
                    f"refusing to save over {target}: its last load found it "
                    f"unparsable, so the in-memory settings are an empty "
                    f"fallback and saving would destroy the accounts in it. "
                    f"Delete the file and retry, or fix it and restart so the "
                    f"fixed file is loaded before anything writes.")
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(data)
                    fh.flush()
                    os.fsync(fh.fileno())
                try:
                    # Tokens + passwords live here: owner-only on POSIX.
                    os.chmod(tmp_name, 0o600)
                except OSError:
                    pass
                # Transient AV/indexer locks block rename for ~ms: retry so the
                # atomic path still wins instead of degrading to in-place.
                replaced = False
                last_err: OSError | None = None
                for attempt in range(4):
                    try:
                        os.replace(tmp_name, target)
                        tmp_name = None
                        replaced = True
                        break
                    except PermissionError as exc:
                        last_err = exc
                        time.sleep(0.05 * (attempt + 1))
                    except OSError as exc:
                        last_err = exc
                        break
                if not replaced:
                    # Windows: an external handle (editor, antivirus, another
                    # process) can block the rename even though in-place writes
                    # are allowed, because rename needs FILE_SHARE_DELETE on
                    # every open handle. Fall back to a NON-DESTRUCTIVE in-place
                    # rewrite (_overwrite_in_place); the save lock still
                    # serializes our own writers. Loud warning, no raise: the
                    # in-memory state already succeeded and 500-ing an endpoint
                    # over a file-handle race would be the worse outcome.
                    logger.warning(
                        "config replace blocked (%s), overwriting in place: %s",
                        last_err, target)
                    _overwrite_in_place(target, data)
            finally:
                if tmp_name:
                    try:
                        os.unlink(tmp_name)
                    except OSError:
                        pass
            _warn_inherited_acl_once(target)
            # This save just wrote the settings to disk, so an earlier load's
            # "the file is unparsable and memory is an empty fallback" is
            # superseded by a file that matches memory. Keeping the latch would
            # re-arm the refusal the moment the recreated file exists (the
            # delete-then-save recovery would work exactly once).
            _UNPARSABLE_CONFIG.discard(str(target))


# Set after the first successful Windows save, never reset. The ACL situation
# is a property of the file/directory, not of any one save, and every account
# add/select/ban/token refresh persists -- a warning per save would bury the
# log it lives in.
_ACL_WARNING_EMITTED = False


def _write_bytes_nondestructive(target: Path, payload: bytes,
                                previous: bytes) -> None:
    """Write ``payload`` over ``target`` from offset 0, truncate only at the end.

    Two properties, both required by the save-crash guarantee:

    1. The open is ``O_WRONLY | O_CREAT`` with NO ``O_TRUNC``, so the file keeps
       its previous length for the whole overwrite. A crash mid-write leaves at
       worst a partial prefix, never an empty credential store.
    2. ``ftruncate`` happens only after the payload is fully written AND
       fsynced. Shrinking first is what would turn "new content is shorter than
       the old" into a torn file; doing it last means a reader can only ever
       see the old length or the new length.

    On failure the previous bytes are written back before the exception
    propagates, so a *recoverable* error (full disk on some filesystems, a
    transient write error) does not leave a half-written credential file
    either.

    ``O_BINARY`` matters on Windows: without it the CRT translates ``\\n`` to
    ``\\r\\n`` on write, so the bytes on disk differ from the bytes written and
    a length computed from ``payload`` would be wrong.
    """
    flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_BINARY", 0)
    fd = os.open(str(target), flags, 0o600)
    try:
        try:
            _write_all(fd, payload)
            os.fsync(fd)
            os.ftruncate(fd, len(payload))
        except BaseException:
            # Best-effort rollback to the last good content. A failure HERE is
            # strictly better than leaving a torn file, and it must not mask
            # the original error.
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                _write_all(fd, previous)
                os.ftruncate(fd, len(previous))
                os.fsync(fd)
            except OSError as exc:  # pragma: no cover - needs a broken FS
                logger.error("config rollback failed (%s); %s now holds a "
                             "PARTIAL write - restore it from %s",
                             exc, target, target.name + ".bak")
            raise
    finally:
        os.close(fd)


def _write_all(fd: int, payload: bytes) -> None:
    """os.write until the whole buffer is out (short writes are normal)."""
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:  # pragma: no cover - defensive
            raise OSError("config write made no progress")
        view = view[written:]


def _backup_previous(target: Path, previous: bytes) -> None:
    """Copy the last good config next to it before an in-place overwrite.

    Only the degraded path needs this (the atomic path cannot lose the old
    content: the rename either happens or does not). ``*.bak`` is already in
    .gitignore, so this adds no tracked file and nothing to clean up, and it is
    the difference between "the file is unusable" and "copy this one file
    back" after a power cut during a save.

    The truncation here is safe (unlike the one this replaced) because it runs
    while the target still holds a complete, good copy: a torn .bak can only
    happen when the target is about to be overwritten, and the target is
    itself intact at that moment.
    """
    if not previous:
        return  # nothing to preserve: first save, or the file was already empty
    bak = target.with_name(target.name + ".bak")
    try:
        fd = os.open(str(bak), os.O_WRONLY | os.O_CREAT | os.O_TRUNC
                     | getattr(os, "O_BINARY", 0), 0o600)
        try:
            _write_all(fd, previous)
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError as exc:
        # Losing the backup must not lose the save.
        logger.warning("could not write config backup %s: %s", bak, exc)


def _overwrite_in_place(target: Path, data: str) -> None:
    """Non-destructive in-place rewrite, used only when os.replace is blocked.

    Guarantees, and why each holds:

    * **A crash during the save never reduces config.json to zero bytes.**
      The file is opened without O_TRUNC and truncated only after the new
      content is written and fsynced, so the on-disk length is monotonically
      the old length until the new content is durable.
    * **A crash during the save never loses the previous good state.** Two
      independent copies: the previous bytes are copied to config.json.bak
      first, and the target itself still holds the previous content until the
      overwrite is complete.
    * An exception *inside* the write is not a crash: the previous bytes are
      written back before it propagates.

    The remaining exposure is a crash in the gap between the overwrite and
    the next save, with the file holding ``new_prefix + old_tail`` - invalid
    JSON, which load_settings treats as an empty config. That is strictly
    better than an empty file and recoverable from the .bak.
    """
    previous = b""
    try:
        previous = target.read_bytes()
    except OSError as exc:
        logger.debug("no readable config to back up (%s): %s", exc, target)
    _backup_previous(target, previous)
    # Byte-exact, unlike the text-mode atomic path above: on Windows that one
    # writes CRLF, so a config alternates between the two after degraded saves.
    # Cosmetic only - both parse identically - and staying byte-exact is what
    # makes the length maths in _write_bytes_nondestructive correct.
    _write_bytes_nondestructive(target, data.encode("utf-8"), previous)
    try:
        os.chmod(target, 0o600)
    except OSError:
        pass


def _warn_inherited_acl_once(target: Path,
                             platform: str | None = None) -> None:
    """On Windows, warn ONCE that config.json inherits its directory's ACL.

    ``save()`` chmods the file 0600, which on win32 only toggles the
    read-only attribute and never touches the DACL -- the file keeps whatever
    the containing directory grants, and on a default install that includes
    read access for ``BUILTIN\\Users``. The file holds account emails,
    passwords and 64-char bearer tokens, so on a shared machine any other
    local user can read them.

    Deliberately a warning rather than a fix. Rewriting the ACL behind the
    operator's back (``icacls``, a deny ACE) can lock them out of their own
    credential store, and it needs the file to exist first, so a config this
    process is about to create could not be secured the same way anyway. The
    actionable part is where the file lives, which is the operator's call.
    No-op on POSIX, where the chmod above is the whole story.

    ``platform`` exists only so the branch is testable on both platforms:
    faking ``os.name`` process-wide breaks ``pathlib`` (it picks PosixPath,
    which cannot be instantiated on Windows). It defaults to the real one.
    """
    global _ACL_WARNING_EMITTED
    if (os.name if platform is None else platform) != "nt":
        return
    if _ACL_WARNING_EMITTED:
        return
    _ACL_WARNING_EMITTED = True
    try:
        logger.warning(
            "config file %s inherits the ACL of its directory: the account "
            "emails, passwords and 64-char bearer tokens it holds are "
            "readable by every other local user that directory grants access "
            "to. POSIX 0600 does NOT apply on Windows -- os.chmod only "
            "toggles the read-only attribute and never touches the DACL. "
            "Mitigation: keep config.json in a directory only your own user "
            "can read, or point DEEPSEAPORT_CONFIG at one (a per-user data "
            "dir, not a shared repo checkout).", target)
    except Exception:
        pass


def _default_config_path() -> Path:
    # Resolution order is intentional: explicit DEEPSEAPORT_CONFIG env wins,
    # then a config.json in the current working directory (per-deployment
    # override), then the app/exe directory. Keeping cwd support (not a single
    # path) preserves portable multi-instance setups; callers log the chosen
    # file so a different-cwd silent 401/503 is diagnosable.
    env = os.environ.get("DEEPSEAPORT_CONFIG")
    if env:
        return Path(env)
    here = Path.cwd() / "config.json"
    if here.exists():
        return here
    return _app_dir() / "config.json"


def _normalize_keys(raw) -> list[str]:
    """Coerce a config/env key list into usable, non-blank API keys.

    A blank entry is an auth bypass, not a weaker key: ``_check_auth`` skips
    only an *empty* list, and a request with no Authorization header presents
    "" — so ``"keys": [""]`` made ``"" in settings.keys`` true and served
    every endpoint (account CRUD included) unauthenticated. Non-str entries
    (hand-edited config.json) are coerced instead of raising.
    """
    if isinstance(raw, str):
        # Bare string: a comma list, same contract as DEEPSEAPORT_KEYS.
        raw = raw.split(",")
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[str] = []
    for entry in raw:
        key = str(entry if entry is not None else "").strip()
        if key:
            out.append(key)
    return out


def _account_field(value) -> str:
    """One account string field, coerced at the boundary.

    ``{"mobile": 13800138000}`` in a hand-edited config loaded as an int, and
    ``accounts._matches``/``_norm`` call ``.strip()`` on it: GET /v1/accounts
    answered 500 and the account could not be selected or deleted from any UI.
    None (a present null) reads as absent, not as ``"None"``. Not stripped:
    the sibling fields load verbatim elsewhere in this loader.
    """
    return "" if value is None else str(value)


def _as_bool(value, default: bool = False) -> bool:
    """String-aware truthiness for a single config value (no env layer).

    ``bool("false")`` is ``True``. Every boolean in Settings goes through the
    ``_bool`` closure in load_settings for exactly this reason; ``banned`` was
    the one field that bypassed it because it is parsed per account, before
    that closure exists. This is that same mapping, lifted to module scope so
    both callers share one notion of "true".

    Truthy: True, 1, "1", "true", "yes", "on" (case- and whitespace-insensitive).
    Anything else is False, including unrecognised strings - an unreadable flag
    must not read as "banned" on a credential the pool would then refuse to use.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value) if value is not None else default


def account_identity(cfg) -> tuple[str, frozenset]:
    """``(provider, identity keys)`` for one account - the dedupe notion of "same".

    Single source of truth for what "the same account" means, shared by
    ``accounts._is_duplicate`` (the runtime pool check) and ``load_settings``
    (the load-time dedupe), so the two can never disagree and let a duplicate
    through the gap between them.

    Identity is per PROVIDER plus any shared non-empty email / mobile / token:
    the same login on two providers is two accounts (they are billed, banned and
    rate-limited separately), while one login listed twice is one credential.
    Keys are only produced for fields that actually hold a value, so a
    half-filled entry (email only, token only) can only collide on what it has -
    the conservative direction: ambiguous identity is not treated as duplicate.
    """
    provider = str(getattr(cfg, "provider", "deepseek") or "deepseek").strip().lower()
    keys: set[tuple[str, str]] = set()
    email = str(getattr(cfg, "email", "") or "").strip().lower()
    if email:
        keys.add(("email", email))
    mobile = str(getattr(cfg, "mobile", "") or "").strip()
    if mobile:
        keys.add(("mobile", mobile))
    token = str(getattr(cfg, "token", "") or "").strip()
    if token:
        keys.add(("token", token))
    return provider, frozenset(keys)


def _dedupe_accounts(accounts: list[AccountConfig]) -> list[AccountConfig]:
    """Drop repeats of an already-seen identity, keeping the first occurrence.

    ``load_settings`` builds the account list directly, bypassing
    ``AccountPool.add``, so it needs its own duplicate check. A repeated entry
    is not cosmetic: the pool holds one PooledAccount per list entry, so
    round-robin failover rotates across two entries that share one token,
    halving the effective pool and doubling the load on that single account's
    rate limit.

    Conservative by construction (see account_identity): only entries that
    share a provider AND a non-empty email/mobile/token are considered the same,
    so a legitimate account is never dropped. When a drop does happen the
    surviving identifier is logged at WARNING with the config path - silently
    editing an operator's credential list would be worse than the duplicate.
    """
    if len(accounts) < 2:
        return list(accounts)
    kept: list[AccountConfig] = []
    seen: list[tuple[str, frozenset]] = []
    dropped: list[str] = []
    for acct in accounts:
        identity = account_identity(acct)
        if any(p == identity[0] and (k & identity[1]) for p, k in seen):
            dropped.append(str(getattr(acct, "identifier", "") or "?"))
            continue
        seen.append(identity)
        kept.append(acct)
    if dropped:
        logger.warning(
            "config has %d duplicate account entr(ies) (same provider + "
            "email/mobile/token as an earlier entry); keeping the first and "
            "dropping %s - a repeated entry makes the pool rotate across one "
            "credential and doubles the load on its rate limit",
            len(dropped), dropped)
    return kept


# Config paths whose last load found the file unparsable. save() consults this
# so a broken file is never overwritten with an empty one (see load_settings
# and _read_config_json). It is cleared by a successful load, and by a
# successful save over a file the operator deleted -- a running process never
# reloads, so deleting is the other documented way out, and the latch must not
# survive that save (the file on disk then matches memory again).
_UNPARSABLE_CONFIG: set[str] = set()

# A UTF-8 BOM is written by many Windows editors (and by anything that saves
# UTF-8 with BOM). json.loads rejects it, which is the single most likely way
# a hand-edited config stops parsing.
_UTF8_BOM = "\ufeff"


def _env_str(env_key: str, raw_key: str, raw: dict | None = None) -> str:
    """A *present but empty* env var must not blank the config value.

    ``os.environ.get(K, raw.get(...))`` returns "" for ``K=``, and an empty
    string is truthy as far as the rest of the loader is concerned -- so
    ``OBSCURA_BIN=`` in a .env silently reverts obscura_bin to "" and the WAF
    path starts auto-discovering a *different* binary than the operator
    pinned, with no warning. An empty override is indistinguishable from an
    accidental one, so treat it as absent and fall back to config.json.
    """
    val = str(os.environ.get(env_key, "") or "").strip()
    if val:
        return val
    src = raw if isinstance(raw, dict) else {}
    return str(src.get(raw_key, "") or "").strip()


def _read_config_json(cfg_path: Path) -> tuple[dict, bool]:
    """Return (parsed_object, was_unparsable).

    Never raises. A missing/blank file is a legitimate "fresh install" and
    yields ({}, False). Anything present but unreadable yields ({}, True) so
    the caller can refuse to persist over it -- see _UNPARSABLE_CONFIG.
    """
    try:
        text = cfg_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.error(
            "config %s could not be read (%s: %s); treating it as EMPTY and "
            "refusing to save over it -- fix or remove the file",
            cfg_path, type(exc).__name__, exc)
        return {}, True

    if not text.strip():
        return {}, False

    if text.startswith(_UTF8_BOM):
        # Not an error the user can act on by eye, so fix it for them.
        logger.warning(
            "config %s starts with a UTF-8 BOM; stripping it (this is what "
            "Windows editors add, and json.loads rejects it)", cfg_path)
        text = text[len(_UTF8_BOM):]

    try:
        parsed = json.loads(text)
    except ValueError as exc:
        logger.error(
            "config %s is not valid JSON (%s). It is being treated as EMPTY so "
            "the app still starts, but NOTHING will be written back to it until "
            "this is fixed -- otherwise the next save would overwrite it with an "
            "empty config and destroy every account in it. Fix the file by hand, "
            "or delete it to start fresh.", cfg_path, exc)
        return {}, True

    if not isinstance(parsed, dict):
        logger.error(
            "config %s is valid JSON but a %s, not an object; treating it as "
            "EMPTY and refusing to save over it", cfg_path, type(parsed).__name__)
        return {}, True

    return parsed, False


def load_settings(path: str | None = None) -> Settings:
    cfg_path = Path(path) if path else _default_config_path()
    try:
        logger.info("using config: %s (cwd=%s)", cfg_path, Path.cwd())
    except Exception:
        pass
    raw: dict = {}
    if cfg_path.exists():
        raw, unparsable = _read_config_json(cfg_path)
        if unparsable:
            # An unparsable config is NOT an empty config. Treating it as one
            # is how a UTF-8 BOM, a half-finished hand edit or a truncated
            # write silently becomes "no accounts" -- and the next save()
            # then rewrites the file with "accounts": [], destroying every
            # email, password and 64-char bearer token in it. Refuse to load
            # as empty, shout, and keep the file untouched on the next save.
            _UNPARSABLE_CONFIG.add(str(cfg_path))
        else:
            # A good load means the operator fixed it (or deleted it): drop
            # the latch so saving works again.
            _UNPARSABLE_CONFIG.discard(str(cfg_path))
    # Both key sources go through _normalize_keys: a blank entry is an auth
    # bypass (see its docstring), not a weaker key.
    raw_keys = raw.get("keys", [])
    keys = _normalize_keys(raw_keys)
    if raw_keys and not keys:
        # `"keys": [""]` / `["   "]` is the "I want to require *something*"
        # hand edit. Dropping the blanks makes it identical to `"keys": []`,
        # i.e. the server is open - so the intent is not honoured and the
        # operator must be told, or they will believe the port is protected.
        logger.warning(
            "config %s has %s API key entr(y/ies) but none of them is usable "
            "(blank): the server is AUTH DISABLED (no API key required on any "
            "endpoint). Set a real key, or remove \"keys\" if you want it open.",
            cfg_path, len(raw_keys) if hasattr(raw_keys, "__len__") else 1)
    env_keys = os.environ.get("DEEPSEAPORT_KEYS")
    if env_keys:
        parsed = _normalize_keys(env_keys.split(","))
        if parsed:
            keys = parsed
        else:
            # DEEPSEAPORT_KEYS="," (or only whitespace) is truthy but parses to
            # zero keys. Applying it would silently replace the config's keys
            # with an empty list, i.e. leave the server AUTH DISABLED with no
            # error anywhere. Keep the file's keys and shout about the trap.
            logger.warning(
                "DEEPSEAPORT_KEYS=%r contains no usable key: keeping the %d "
                "key(s) from %s. Applying it literally would leave the server "
                "AUTH DISABLED (no API key required on any endpoint).",
                env_keys, len(keys), cfg_path)

    raw_accounts = raw.get("accounts", [])
    if not isinstance(raw_accounts, list):
        # A present "accounts": null (or a junk dict) used to raise TypeError
        # while iterating, so the app would not start at all; read it as no
        # accounts, the same silent coercion _normalize_keys/_providers_list
        # apply to their own fields.
        raw_accounts = []
    accounts: list[AccountConfig] = []
    for a in raw_accounts:
        if isinstance(a, dict) and (a.get("email") or a.get("mobile") or a.get("token")):
            accounts.append(AccountConfig(
                email=_account_field(a.get("email")),
                mobile=_account_field(a.get("mobile")),
                password=_account_field(a.get("password")),
                token=_account_field(a.get("token")),
                token_expires_at=a.get("token_expires_at", 0.0),
                # NOT bool(): "banned": "false" is a banned account forever.
                # AccountConfig.__post_init__ only clears a ban whose
                # banned_until has passed, and a false-but-truthy string flag
                # carries banned_until=0.0 - "expiry unknown" - so there is
                # nothing to expire and the account stays (BANNED) with no date
                # until someone POSTs /v1/accounts/unblock. _as_bool is the
                # same mapping every other Settings flag gets from _bool.
                banned=_as_bool(a.get("banned", False)),
                banned_until=a.get("banned_until", 0.0),
                provider=a.get("provider", "deepseek") or "deepseek",
            ))
    # Single-account env: replaces placeholder config, otherwise appends as an
    # extra failover account (DEEPSEAPORT_KEYS replaces keys; TOKEN appends).
    env_token = os.environ.get("DEEPSEAPORT_TOKEN") or os.environ.get("DEEPSEEK_TOKEN")
    env_email = os.environ.get("DEEPSEAPORT_EMAIL") or os.environ.get("DEEPSEEK_EMAIL")
    env_password = os.environ.get("DEEPSEAPORT_PASSWORD") or os.environ.get("DEEPSEEK_PASSWORD")
    if env_token or (env_email and env_password):
        env_provider = (os.environ.get("DEEPSEAPORT_PROVIDER")
                        or raw.get("provider", "deepseek") or "deepseek")
        env_acc = AccountConfig(email=env_email or "", password=env_password or "",
                                token=env_token or "", provider=env_provider)
        if not accounts or not (accounts[0].token or accounts[0].email):
            accounts = [env_acc]
        else:
            accounts.append(env_acc)
    # After the env append, so an env token that repeats a config account is
    # caught too - it is the same credential on the same provider by any other
    # measure.
    accounts = _dedupe_accounts(accounts)
    if len(accounts) > MAX_ACCOUNTS_LIMIT:
        # Warn, never truncate. MAX_ACCOUNTS (server_routes.py:55) is an
        # admission cap on POST /v1/accounts, meant to stop that endpoint from
        # being a disk-write amplifier; a config file over the limit is the
        # operator's own account list, and dropping entries here would delete
        # credentials on the very next save() (save() rewrites the file from
        # settings.accounts). Reporting it is the only safe response.
        logger.warning(
            "config %s holds %d accounts, above the POST /v1/accounts limit of "
            "%d; loading all of them (load must not drop credentials - the next "
            "save() would persist the loss)",
            cfg_path, len(accounts), MAX_ACCOUNTS_LIMIT)

    def _bool(env_key: str, raw_key: str, default: bool) -> bool:
        if env_key in os.environ:
            return _as_bool(os.environ[env_key])
        return _as_bool(raw.get(raw_key, default), default)

    # Clamp bookkeeping per field, keyed by the config.json key:
    # {config key: (operator's own value, effective clamped value)}.
    # Only the ceiling clamp is recorded: a negative counter has never been
    # honoured (the 0 floor predates the ceiling) and 0 is its only sane
    # reading, while an over-large counter is a plausible number the operator
    # deliberately wrote.
    clamped_from_file: dict[str, tuple[int, int]] = {}

    def _int(env_key: str, raw_key: str, default: int,
             hi: int | None = None) -> int:
        # ``hi`` is a documented upper bound applied to env and config alike
        # (only counters that multiply work should pass one; a floor of 0 is
        # all the other integer settings need). Clamping rather than falling
        # back to the default keeps a too-large value from looking like a
        # config error, and the warning names the field so the operator can
        # see why the number they wrote is not the number in effect.
        def _bounded(val: int) -> int:
            if hi is not None and val > hi:
                logger.warning(
                    "%s=%d clamped to %d (documented range 0..%d)", raw_key,
                    val, hi, hi)
                return hi
            return val

        if env_key in os.environ:
            try:
                return _bounded(max(0, int(os.environ[env_key])))
            except ValueError:
                return default
        try:
            file_val = max(0, int(raw.get(raw_key, default)))
        except (ValueError, TypeError):
            return default
        bounded = _bounded(file_val)
        if bounded != file_val:
            # Remember the operator's number so Settings.save() writes it back
            # unchanged (save() runs on unrelated actions all the time).
            # Config-file values only: an env override is transient and is not
            # owned by config.json, so there is no on-disk value to protect -
            # and baking it into the file is pre-existing behaviour, unchanged
            # here.
            clamped_from_file[raw_key] = (file_val, bounded)
        return bounded

    def _str(env_key: str, raw_key: str, default: str,
             valid: tuple[str, ...] | None = None) -> str:
        raw_val = raw.get(raw_key, default)
        val = str(os.environ.get(env_key, raw_val)).strip()
        if valid and val not in valid:
            # Case-insensitive match for convenience, else fall back.
            lowered = {v.lower(): v for v in valid}
            return lowered.get(val.lower(), default)
        return val or default

    def _text(env_key: str, raw_key: str) -> str:
        # Free-form multi-line text: internal newlines preserved verbatim.
        # No .strip(): leading spaces (code indent) and blank edge lines are
        # the user's formatting; _prepare trims only edge blank lines when
        # joining. \r\n normalized so Windows-edited config/env joins cleanly.
        if env_key in os.environ:
            return str(os.environ[env_key]).replace("\r\n", "\n")
        val = raw.get(raw_key, "")
        if val is None:
            return ""
        if not isinstance(val, str):
            val = str(val)
        return val.replace("\r\n", "\n")

    # Range-checked, same "last line of defence" position as max_retries above
    # (see MAX_RETRIES_LIMIT / PORT_LIMIT): the TUI validates 1-65535 but
    # config.json and DEEPSEAPORT_PORT do not, and an out-of-range port is
    # unrecoverable-looking from the outside (0 -> every request fails
    # connection-refused against the port the UI is printing; >65535 ->
    # OverflowError out of socket.bind, which is not an OSError and so escapes
    # the bind path's handlers as a traceback).
    port = _int("DEEPSEAPORT_PORT", "port", DEFAULT_PORT, hi=PORT_LIMIT)
    if port == 0:
        # _int only floors at 0. 0 is a legal bind() argument ("any free
        # port") and never a legal config value: the server would take an
        # ephemeral port while every screen keeps showing http://127.0.0.1:0,
        # so each request fails with [WinError 10061] and nothing anywhere
        # points at config.json. Treat it as "no port chosen" (and a negative
        # value lands here through _int's floor of 0).
        logger.warning(
            "port=%r is not a usable listen port (0 means \"bind any free "
            "port\", which nothing can dial): using the default %d instead",
            os.environ.get("DEEPSEAPORT_PORT", raw.get("port")), DEFAULT_PORT)
        port = DEFAULT_PORT
        if ("DEEPSEAPORT_PORT" not in os.environ and raw.get("port") == 0):
            # Recorded like any other clamp so an unrelated save() cannot
            # rewrite the operator's own number (see _RAW_PERSISTED_VALUES);
            # the next load re-warns, which is right for a value that was
            # never what they meant. An env override is transient and has no
            # on-disk value to protect.
            clamped_from_file["port"] = (0, DEFAULT_PORT)
    settings = Settings(
        keys=keys,
        accounts=accounts,
        active_account=str(os.environ.get("DEEPSEAPORT_ACTIVE_ACCOUNT",
                                           raw.get("active_account", ""))).strip(),
        obscura_bin=_env_str("OBSCURA_BIN", "obscura_bin", raw),
        obscura_profile=_env_str("OBSCURA_PROFILE", "obscura_profile", raw),
        browser_bin=os.environ.get(
            "DEEPSEAPORT_BROWSER_BIN", raw.get("browser_bin", "auto")),
        browser_headless=_bool("DEEPSEAPORT_BROWSER_HEADLESS", "browser_headless", False),
        port=port,
        listen=_bool("DEEPSEAPORT_LISTEN", "listen", False),
        config_path=str(cfg_path),
        enable_tools=_bool("DEEPSEAPORT_ENABLE_TOOLS", "enable_tools", True),
        warmup_on_startup=_bool("DEEPSEAPORT_WARMUP_ON_STARTUP", "warmup_on_startup", True),
        auto_delete_session=_bool("DEEPSEAPORT_AUTO_DELETE_SESSION", "auto_delete_session", True),
        max_retries=_int("DEEPSEAPORT_MAX_RETRIES", "max_retries", 1,
                         hi=MAX_RETRIES_LIMIT),
        parallel_challenge_fetch=_bool(
            "DEEPSEAPORT_PARALLEL_FETCH", "parallel_challenge_fetch", True),
        tool_args_max_chars=(
            _int("DEEPSEAPORT_MAX_ARGS_CHARS", "tool_args_max_chars", 200_000)
            or 200_000
        ),
        forgiving_toolcalls=_bool(
            "DEEPSEAPORT_FORGIVING_TOOLCALLS", "forgiving_toolcalls", False),
        use_multiple_accounts=_bool(
            "DEEPSEAPORT_USE_MULTIPLE_ACCOUNTS", "use_multiple_accounts", True),
        log_level=_str("DEEPSEAPORT_LOG_LEVEL", "log_level", "INFO", VALID_LOG_LEVELS),
        stream_mode=_str("DEEPSEAPORT_STREAM_MODE", "stream_mode", "live", VALID_STREAM_MODES),
        chat_model=_str("DEEPSEAPORT_CHAT_MODEL", "chat_model", "deepseek-flash"),
        enable_append=_bool("DEEPSEAPORT_ENABLE_APPEND", "enable_append", True),
        append_top=_text("DEEPSEAPORT_APPEND_TOP", "append_top"),
        append_bottom=_text("DEEPSEAPORT_APPEND_BOTTOM", "append_bottom"),
        send_dry_run_to_frontend=_bool(
            "DEEPSEAPORT_SEND_DRY_RUN_TO_FRONTEND", "send_dry_run_to_frontend", True),
        provider=_validate_provider_id(
            _str("DEEPSEAPORT_PROVIDER", "provider", "deepseek"), cfg_path),
        providers=_validated_provider_ids(_providers_list(raw), cfg_path),
    )
    if clamped_from_file:
        # Plain instance attribute, NOT a dataclass field: the runtime bound
        # above is what the engine obeys, this is only what save() writes back.
        setattr(settings, _RAW_PERSISTED_VALUES, dict(clamped_from_file))
    return settings


def _providers_list(raw: dict) -> list[str]:
    """Parse the multi-provider selection (env comma-list, else config list)."""
    if "DEEPSEAPORT_PROVIDERS" in os.environ:
        raw_val = os.environ["DEEPSEAPORT_PROVIDERS"]
    else:
        raw_val = raw.get("providers", [])
    if isinstance(raw_val, str):
        parts = raw_val.split(",")
    elif isinstance(raw_val, (list, tuple)):
        parts = [str(x) for x in raw_val]
    else:
        parts = []
    return [p.strip() for p in parts if p and str(p).strip()]


def _registered_provider_ids() -> set[str] | None:
    """Lowercased ids the registry knows, or None when it cannot be asked.

    The late import is load-bearing, not a style choice: ``core.registry``
    imports the provider packages and each of those imports this module, so a
    module-scope ``from .core import registry`` here would close an import
    cycle. ``load_settings`` only ever runs at runtime (never while this module
    is still executing), so by then every part of the cycle is importable.

    None means "could not answer", not "nothing is registered": a registry that
    fails to import must leave the configured ids alone rather than flatten a
    working deployment to the default.
    """
    try:
        from .core import registry as _registry

        return {str(pid).strip().lower() for pid in _registry.registered()}
    except Exception:
        return None


def _validate_provider_id(pid: str, cfg_path) -> str:
    """One configured provider id, checked against the registry.

    ``provider`` used to load with no ``valid=`` tuple (unlike ``log_level`` and
    ``stream_mode``, which both have one), so a typo -- ``"provider": "glm"``, a
    stale ``DEEPSEAPORT_PROVIDER=quwn`` -- produced a server that reported
    ready, answered /v1/models with DeepSeek's catalogue and routed every chat
    request to DeepSeek: no error, no log line, no 400.

    A bad id is refused LOUDLY rather than refused outright: it falls back to
    the documented ``deepseek`` default and a warning names the rejected id, the
    file it came from and the ids that do exist. Not a raise, because
    ``load_settings`` is the shared entry point of serve, dry-run, the TUI and
    the CLI, and turning one mistyped letter into an unstartable server would
    block the operator from fixing anything but that one field. The same
    "misconfiguration is named in the log instead of blocking recovery" rule
    ``accounts.usable_provider`` documents for the add path.
    """
    key = str(pid or "").strip().lower()
    known = _registered_provider_ids()
    if not key or key in (known or {key}):
        return key or "deepseek"
    logger.warning(
        "provider %r in %s is not a registered provider; using 'deepseek' "
        "instead. Known providers: %s. Set provider to one of those (or add "
        "the module to core.registry._BUILTIN_MODULES).",
        pid, cfg_path, ", ".join(sorted(known)) or "none")
    return "deepseek"


def _validated_provider_ids(ids: list[str], cfg_path) -> list[str]:
    """``providers`` entries the registry knows, each rejected one named.

    Applied to the multi-provider list for the reason ``_validate_provider_id``
    gives the scalar field, plus one more: an unknown entry there also flips
    ``is_multi()`` (multi mode is "2+ ids"), so it would build a pool and a
    model list for a provider that cannot exist. Dropping it keeps the list
    describing real upstreams; if nothing survives, the empty list means "use
    ``provider`` only", which is the documented meaning of an empty list.
    """
    known = _registered_provider_ids()
    if known is None:
        return list(ids)
    out: list[str] = []
    for raw in ids:
        key = str(raw or "").strip().lower()
        if key in known and key not in [o.lower() for o in out]:
            out.append(raw)
            continue
        logger.warning(
            "providers entry %r in %s is not a registered provider; ignoring "
            "it. Known providers: %s.", raw, cfg_path, ", ".join(sorted(known)))
    return out
