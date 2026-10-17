"""CLI: serve | waf | models | chat."""

from __future__ import annotations

import argparse
import json
import re
import sys

from . import server_runner as _runner
from .providers.deepseek.auth import unpack_login_result as _unpack_login_result
from .server_runner import _resolve_bind_host, _watch_stop_keys

# Shared server bootstrap lives in deepseaport.server_runner (single source of
# truth for cli + chat_ui). Thin wrappers below keep the historical cli-level
# names importable and monkeypatchable (tests/tui import these from cli).


def _is_port_free(host: str, port: int) -> bool:
    return _runner._is_port_free(host, port)


def _scan_free_ports(host: str, start: int, count: int,
                     limit: int | None = None, check=None) -> list[int]:
    return _runner._scan_free_ports(host, start, count, limit=limit, check=check)


def _next_free_port(host: str, port: int, limit: int = 50) -> int | None:
    return _runner._next_free_port(host, port, limit=limit, check=_is_port_free)


def _ensure_free_port(settings, host: str, port: int) -> int:
    return _runner._ensure_free_port(settings, host, port,
                                     is_free=_is_port_free,
                                     next_free=_next_free_port)


def _run_server(settings, args) -> int:
    return _runner._run_server(settings, args, ensure_free=_ensure_free_port)


def _valid_provider_id(raw) -> str:
    """Permissive provider-id check for account binding (lowercase alnum/dash)."""
    s = str(raw or "").strip().lower()
    if not s:
        return ""
    if re.match(r"^[a-z0-9-]+$", s):
        return s
    return ""


def _port_number(raw: str) -> int:
    """argparse ``type=`` for ``--port``: an int inside the bindable range.

    A raw ``type=int`` accepted 70000 and only failed later inside
    ``socket.bind`` as an uncaught OverflowError ("port must be 0-65535").
    """
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            f"port must be an integer 1-65535, got {raw!r}")
    if not 1 <= value <= 65535:
        raise argparse.ArgumentTypeError(f"port must be 1-65535, got {value}")
    return value


def provider_is_registered(pid: str) -> bool:
    """True when ``pid`` names a provider the registry actually has.

    The shape check above is NOT membership: `_valid_provider_id` accepts any
    ``^[a-z0-9-]+$``, so ``login --provider glm`` resolved to no provider at all
    and fell through to the DeepSeek Obscura driver -- a ~30s browser window
    against chat.deepseek.com, then `_upsert_account(provider="glm")` storing
    DeepSeek's userToken under a label no UI lists and no request can use.

    Shared with `tui._offer_auto_login` (which cannot exit) so both frontends
    resolve a login provider the same way.
    """
    key = _valid_provider_id(pid)
    if not key:
        return False
    try:
        from .core import registry as _reg

        return _reg.get(key) is not None
    except Exception:
        return False


def _unknown_provider_exit(raw) -> int:
    """Print the one refusal every unknown-provider path shares; return 2.

    `serve --provider`, `serve --providers`, `chat --provider` and now
    `login --provider` all spelled this out, and the wording is the only thing a
    user sees after a typo -- three copies is how they drift.
    """
    try:
        from .core import registry as _reg

        available = ", ".join(sorted(_reg.registered().keys()))
    except Exception:
        available = ""
    print(f"unknown provider: {raw}")
    print("available: " + available)
    return 2


def _provider_display_name(pid: str) -> str:
    """Human name for a provider id, from the registry (falls back to the id)."""
    key = str(pid or "").strip().lower() or "deepseek"
    try:
        from .core import registry as _reg

        name = str(getattr(_reg.get(key), "display_name", "") or "").strip()
        if name:
            return name
    except Exception:
        pass
    return key


def _stdin_is_tty() -> bool:
    """True only when there is a console to prompt on. Never raises.

    getattr-style because sys.stdin is None under pythonw.exe and some frozen
    launchers; a bare sys.stdin.isatty() raised AttributeError there.
    """
    _stdin = getattr(sys, "stdin", None)
    try:
        return bool(getattr(_stdin, "isatty", None) and _stdin.isatty())
    except Exception:
        return False


def _warn_secret_on_cmdline(*flags: str) -> None:
    """One line: a secret passed as a flag is visible to other processes."""
    print("  ! " + "/".join(flags) + " was given on the command line: the value "
          "is visible in the process list and in your shell history. Omit the "
          "flag to be prompted without echo.")


def _read_secret_line() -> str:
    """One line from a non-tty stdin, or "" when there is nothing to read.

    The piped-secret path: ``getpass.getpass`` falls back to
    ``getpass.fallback_getpass`` when it cannot open /dev/tty, which reads a
    single line from stdin - that is how ``deepseaport login --email a@x.com
    < password.txt``, service wrappers and CI steps pass a secret. Replacing
    that with "" made every scripted login authenticate with an empty string.

    EOF, a closed/NULL stdin (``sys.stdin is None`` under pythonw.exe) and a
    read-blocked stream (pytest's captured stdin raises OSError) all mean "no
    secret here", not a crash, and the caller's existing "no password" path
    takes over. Only the line terminator is stripped, exactly as
    ``fallback_getpass`` does: trailing spaces can be part of the secret.
    ``fallback_getpass``'s "input may be echoed" warning is deliberately not
    reproduced - it is printed *to stdin*, which for ``< password.txt``
    overwrites the beginning of the very file holding the secret.

    Like ``fallback_getpass``, this blocks on an open-but-silent stdin (e.g.
    ``docker run -i`` with no input): a secret the operator meant to pipe in
    has not arrived yet, and silently logging in with "" is the worse failure.
    """
    try:
        stdin = getattr(sys, "stdin", None)
        if stdin is None:
            return ""
        line = stdin.readline()
    except (EOFError, KeyboardInterrupt, OSError, ValueError, AttributeError):
        return ""
    if not line:
        return ""  # EOF (or a stream that yields nothing)
    return str(line).rstrip("\r\n")


def _secret_from_arg(flag_value: str, flag: str, prompt: str) -> str:
    """Flag value if given (with a warning), else a no-echo prompt on a tty.

    Non-interactive callers (pipes, CI, service wrappers) get one line read
    from stdin, i.e. the pre-existing ``getpass`` fallback_getpass behaviour:
    a scripted ``... login --email a@x.com < password.txt`` authenticates
    with the piped secret. No flag, no tty and no data on stdin yields "",
    which the callers already handle as "no password / no token".
    """
    if flag_value:
        _warn_secret_on_cmdline(flag)
        return flag_value
    if not _stdin_is_tty():
        return _read_secret_line()
    import getpass as _gp

    try:
        return _gp.getpass(f"{prompt}: ")
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


# Canonical short aliases shared by CLI flags and the TUI interactive gate.
# Single home so `ds` means `deepseek` in both. TUI keeps thin wrappers
# (_PROVIDER_ALIASES / _normalize_provider_choice) delegating here for
# backward compatibility (tests import those names from tui).
PROVIDER_ALIASES = {"ds": "deepseek"}


def normalize_provider_id(raw: str, valid_ids: list[str]) -> str:
    """Normalize typed provider to a valid id, resolving aliases.

    Returns "" when the choice does not match. Empty input returns "" here;
    callers apply the deepseek default. Case-insensitive exact match after
    alias resolution (e.g. ds -> deepseek).
    """
    key = (raw or "").strip().lower()
    if not key:
        return ""
    key = PROVIDER_ALIASES.get(key, key)
    for pid in valid_ids:
        if str(pid or "").strip().lower() == key:
            return str(pid or "").strip().lower()
    return ""


def resolve_login_provider_id(settings, provider: str | None = None) -> str:
    """Login pid shared by CLI login and TUI auto-login.

    Explicit provider wins; else the selected (single-mode) provider; else
    deepseek. Always lowercased, never empty. Thin adapter so both frontends
    resolve the same way without duplicating the fallback chain.
    """
    pid = (str(provider or "").strip().lower()
           or (str(getattr(settings, "provider", "") or "").strip().lower())
           or "deepseek")
    return pid or "deepseek"


def _save(settings) -> bool:
    """``settings.save()`` with a one-line failure report instead of a traceback.

    ``Settings.save()`` raises by design (an unparsable config.json, an
    unwritable location). A traceback here is the whole UX of a one-shot CLI,
    and in ``cmd_login`` it also discarded the token a ~30s browser
    authorization had just produced. Returns True on success.
    """
    try:
        settings.save()
        return True
    except Exception as exc:
        print(f"could not write {getattr(settings, 'config_path', '')}: {exc}")
        return False


def _apply_serve_provider_flags(settings, args) -> int | None:
    """Apply --provider/--providers to settings. None = ok, int = exit code."""
    from .core import registry as _reg

    single = getattr(args, "provider", None)
    multi = getattr(args, "providers", None)
    if single:
        pid = str(single).strip().lower()
        if _reg.get(pid) is None:
            return _unknown_provider_exit(single)
        settings.provider = pid
        settings.providers = []
        if not _save(settings):
            return 1
        return None
    if multi:
        ids: list[str] = []
        for part in str(multi).split(","):
            key = part.strip().lower()
            if key and key not in ids:
                ids.append(key)
        if not ids:
            return None
        for pid in ids:
            if _reg.get(pid) is None:
                return _unknown_provider_exit(pid)
        if len(ids) == 1:
            settings.provider = ids[0]
            settings.providers = []
        else:
            settings.providers = ids
            settings.provider = ids[0]
        if not _save(settings):
            return 1
        return None
    return None


def cmd_serve(args) -> int:
    from .config import load_settings

    settings = load_settings(args.config)
    if args.port:
        settings.port = args.port

    flagged = _apply_serve_provider_flags(settings, args)
    if flagged is not None:
        return flagged
    explicit = bool(getattr(args, "provider", None) or getattr(args, "providers", None))

    no_tui = bool(getattr(args, "no_tui", False))
    # Headless serve is the right fallback for a windowed process that has no
    # console to draw a TUI on -- see _stdin_is_tty.
    if not no_tui and _stdin_is_tty():
        from .tui import main_menu, provider_menu

        if not explicit:
            provider_menu(settings)

        def serve_fn(current_settings) -> int:
            return _run_server(current_settings, args)

        def chat_fn(current_settings) -> int:
            from .chat_ui import run_chat_server
            return run_chat_server(current_settings, args)

        def multi_fn(current_settings) -> int:
            from .chat_ui import run_multi_chat_servers
            return run_multi_chat_servers(current_settings, args)

        def dry_fn(current_settings) -> int:
            from .dry_run import run_dry_run
            return run_dry_run(current_settings, args)

        return main_menu(settings, serve_fn, chat_fn, multi_fn, dry_fn)
    return _run_server(settings, args)


def cmd_waf(args) -> int:
    from .config import load_settings
    from .core.obscura_bridge import ObscuraBridge

    settings = load_settings(args.config)
    bridge = ObscuraBridge(settings.obscura_bin, settings.obscura_profile)
    print("binary:", bridge.binary)
    print("version:", bridge.version())
    cookies = bridge.warmup()
    print("cookies:", sorted(cookies))
    print("has_waf_token:", bridge.has_waf_token())
    print("ua:", bridge.state.user_agent)
    return 0 if bridge.has_waf_token() else 2


def cmd_models(args) -> int:
    from .server import MODELS, active_models

    flag = getattr(args, "provider", None)
    if flag:
        from .core import registry as _reg

        prov = _reg.get(str(flag).strip().lower())
        if prov is None:
            print(f"unknown provider: {flag}")
            print("available: " + ", ".join(sorted(_reg.registered().keys())))
            return 2
        try:
            ids = sorted(prov.models().keys())
        except Exception:
            ids = []
        print(json.dumps(ids, indent=2, ensure_ascii=False))
        return 0
    try:
        from .config import load_settings as _load_settings

        _settings = _load_settings(getattr(args, "config", None))
        catalogue = active_models(_settings) or dict(MODELS)
    except Exception:
        catalogue = dict(MODELS)
    print(json.dumps(sorted(catalogue), indent=2, ensure_ascii=False))
    return 0


def cmd_accounts(args) -> int:
    from .accounts import TOKEN_HELP, AccountPool, extract_token, sync_current_from_settings
    from .config import AccountConfig, load_settings

    settings = load_settings(args.config)
    # Sync stale CURRENT selection (e.g. manually edited config).
    _tmp = AccountPool(settings.accounts, current=settings.active_account)
    if sync_current_from_settings(_tmp, settings):
        settings.active_account = _tmp.current
        if not _save(settings):
            return 1
    action = getattr(args, "accounts_action", "list")

    if action == "list":
        pool = AccountPool(settings.accounts, current=settings.active_account)
        rows = pool.status()
        print(f"CURRENT: [{pool.current or 'ALL (auto-failover)'}]")
        if not rows:
            print("pool empty")
            print(TOKEN_HELP)
            return 0
        try:
            from .accounts import ban_label_for as _ban_lookup, collect_ban_labels as _ban_map
            from .providers.deepseek.protocol import format_ban_label as _ban_label
            ban_map = _ban_map(settings.accounts) or {}
        except Exception:
            ban_map = {}
            _ban_lookup = None  # type: ignore
            _ban_label = None  # type: ignore
        for i, r in enumerate(rows, 1):
            mark = " *CURRENT*" if r.get("current") else ""
            token_flag = "yes" if _account_has_token(settings, str(r["identifier"])) else "MISSING"
            ban_flag = ""
            try:
                found, until = (_ban_lookup(ban_map, str(r["identifier"]))
                                if _ban_lookup else (False, None))
                if found:
                    ban_flag = f"  {_ban_label(until)}" if _ban_label else "  (BANNED)"
            except Exception:
                ban_flag = ""
            print(f"{i}. {r['identifier']}{mark}  uses={r['uses']}  "
                  f"busy={r['busy']}  cooldown={r['cooldown_remaining']}s  token={token_flag}{ban_flag}")
        missing = [r["identifier"] for r in rows
                   if not _account_has_token(settings, str(r["identifier"]))
                   and not r.get("banned")]
        if missing:
            print(f"missing token: {', '.join(missing)}")
            print(TOKEN_HELP)
        return 0

    if action == "select":
        ident = (getattr(args, "identifier", "") or "").strip()
        if getattr(args, "clear", False) or not ident or ident.lower() in ("none", "all", "auto"):
            settings.active_account = ""
            if not _save(settings):
                return 1
            print("Selection cleared -> ALL (auto-failover)")
            return 0
        pool = AccountPool(settings.accounts, current=settings.active_account)
        if not pool.set_current(ident):
            print(f"not found: {ident}")
            return 1
        settings.active_account = pool.current
        if not _save(settings):
            return 1
        print(f"CURRENT -> [{pool.current}] (saved)")
        return 0

    if action == "add":
        # Secrets default to a no-echo prompt so they never have to be typed
        # into a command line (visible to every same-user process and kept in
        # the shell history). Non-interactive callers read one line from stdin
        # (see _secret_from_arg).
        raw_token = _secret_from_arg(getattr(args, "token", "") or "", "--token",
                                     "Token (Enter=skip)")
        raw_password = _secret_from_arg(getattr(args, "password", "") or "",
                                        "--password", "Password (Enter=skip)")
        cfg = AccountConfig(email=args.email or "", mobile=args.mobile or "",
                            password=raw_password, token=extract_token(raw_token))
        if not (cfg.email or cfg.mobile or cfg.token):
            print("need --email, --mobile, or --token")
            print(TOKEN_HELP)
            return 2
        _flag = _valid_provider_id(getattr(args, "provider", None))
        if _flag:
            cfg.provider = _flag
        else:
            # No --provider: bind to the configured selection here, because
            # AccountConfig's own default ("deepseek") is indistinguishable
            # from an explicit choice once constructed. Same coercion, and the
            # same reason, as accounts.add_account -- see that comment.
            try:
                from .accounts import configured_provider_id as _configured
                from .accounts import usable_provider as _usable

                cfg.provider = _usable(_configured(settings))
            except Exception:
                pass
        pool = AccountPool(settings.accounts, current=settings.active_account)
        try:
            from .accounts import add_account as _add
            _add(settings, pool, cfg)
        except ValueError as exc:
            print(f"add failed: {exc}")
            return 1
        if not _save(settings):
            return 1
        print(f"added {cfg.identifier} -> {settings.config_path}")
        if not cfg.token:
            print("WARNING: no token. Password-only fails (RISK_DEVICE_DETECTED).")
            print("TIP: python -m deepseaport login --email "
                  f"{cfg.email or cfg.mobile or ''} (auto-capture)")
            print(TOKEN_HELP)
        return 0

    if action == "set-token":
        from .accounts import set_account_token as _set
        token = extract_token(args.token or "")
        if not args.identifier or not token:
            print("usage: accounts set-token <email> --token <userToken value>")
            print(TOKEN_HELP)
            return 2
        matched = _set(settings, args.identifier, token)
        if matched is not None:
            if not _save(settings):
                return 1
            print(f"token updated for {matched.identifier}")
            return 0
        print(f"not found: {args.identifier}")
        return 1

    if action == "remove":
        identifier = args.identifier or ""
        if not identifier:
            print("need identifier (email/mobile/token)")
            return 2
        pool = AccountPool(settings.accounts, current=settings.active_account)
        if not pool.remove(identifier):
            print(f"not found: {identifier}")
            return 1
        # Keep config file in sync with pool.
        from .accounts import remove_account as _remove
        _remove(settings, identifier)
        if not _save(settings):
            return 1
        print(f"removed {identifier}")
        return 0

    if action == "unblock":
        # Clear persisted ban markers in config. A running server also needs
        # POST /v1/accounts/unblock to drop its in-memory cooldown singleton.
        from .accounts import _matches as _m
        ident = (getattr(args, "identifier", "") or "").strip()
        n = 0
        matched = 0
        for a in settings.accounts:
            if ident and not _m(a, ident):
                continue
            matched += 1
            if getattr(a, "banned", False) or getattr(a, "banned_until", 0.0):
                n += 1
            a.banned = False
            a.banned_until = 0.0
        # A typed identifier that matched nothing exits 1 like `select`,
        # `remove` and `set-token` do: "cleared for 0 account(s)" plus exit 0
        # read as "done", so a typo'd email left the account banned and the
        # operator restarted nothing. No identifier still means "all
        # accounts", which legitimately clears 0 of 0.
        if ident and not matched:
            print(f"not found: {ident}")
            print("accounts unblock <email|mobile|identifier> -- or run it "
                  "with no identifier to clear every account.")
            return 1
        if not _save(settings):
            return 1
        print(f"persisted ban state cleared for {n} account(s)")
        print("If a server is running, also POST /v1/accounts/unblock "
              "(or restart it) to clear its in-memory cooldown.")
        return 0

    print(f"unknown accounts action: {action}")
    return 2


def _account_has_token(settings, identifier: str) -> bool:
    from .accounts import account_has_token
    return account_has_token(settings, identifier)


def _upsert_account(settings, email: str, password: str, token: str,
                    provider: str | None = None,
                    token_expires_at: float = 0.0):
    """Append or update account by email. Preserves pool, fixes wipe bug."""
    from .accounts import _matches as _m, extract_token
    from .config import AccountConfig

    token = extract_token(token)
    try:
        token_expires_at = max(0.0, float(token_expires_at or 0.0))
    except (TypeError, ValueError):
        token_expires_at = 0.0
    if provider:
        # Explicit --provider: verbatim, same contract as `accounts add`
        # (an id the registry does not know may still be the operator's).
        effective = str(provider).strip().lower()
    else:
        # Inherited selection, coerced exactly as accounts.add_account does.
        from .accounts import configured_provider_id as _configured
        from .accounts import usable_provider as _usable

        effective = _usable(_configured(settings))
    if not effective:
        effective = "deepseek"
    for a in settings.accounts:
        # Scoped to the effective provider: the same email is a distinct
        # account per provider, and the unscoped match stamped a Qwen JWT onto
        # the provider:deepseek record (or the reverse).
        if (a.email and _m(a, email)
                and ((getattr(a, "provider", "deepseek") or "deepseek"
                      ).strip().lower() == effective)):
            a.token = token
            if password:
                a.password = password
            if token_expires_at:
                a.token_expires_at = token_expires_at
            return a
    cfg = AccountConfig(email=email, password=password, token=token)
    cfg.provider = effective
    settings.accounts.append(cfg)
    # New account becomes CURRENT (default) immediately.
    settings.active_account = cfg.identifier
    return cfg


# Login implementation lives in providers.deepseek.auth (selector-driven, visible
# text only). These wrappers keep the historical CLI/TUI/server import points
# stable and are intentionally tiny.

def _unpack_login_token_result(res) -> tuple[str | None, bool, str]:
    return _unpack_login_result(res)


def _obscura_login_token(email: str, password: str, settings,
                         provider: str | None = None) -> tuple[str | None, bool, str]:
    """Login via registry.get(pid).login for non-DeepSeek, else historical shim.

    DeepSeek default keeps the canonical ``providers.deepseek.auth`` path so
    existing monkeypatch seams (tests) and behavior are unchanged.
    """
    pid = resolve_login_provider_id(settings, provider)
    if pid != "deepseek":
        try:
            from .core import registry as _reg
            prov = _reg.get(pid)
            if prov is not None:
                return _unpack_login_token_result(prov.login(email, password, settings))
        except Exception:
            pass
    from .providers.deepseek.auth import obscura_login_token as _impl
    return _impl(email, password, settings)


# Identity of the login driver as shipped. cmd_login compares against it to
# tell "the real browser path" from "a caller injected its own driver" (tests
# and embedders), which is the only thing that decides whether the
# empty-credential guard below applies. See the comment there.
_CANONICAL_LOGIN_DRIVER = _obscura_login_token


def _missing_credential(email: str, password: str) -> str:
    """One actionable block for a login that has no email and/or no password.

    Risk control is the reason this is a hard stop: with an empty password
    ``providers/deepseek/auth.py:310`` calls ``login_with_real_browser``
    BEFORE the empty check at ``auth.py:152``, so an Enter at the password
    prompt (which carries no "Enter=skip" hint) opened a real Helium/Chrome
    window, filled the email with an empty password and clicked Log in. A
    failed sign-in from the user's own IP is exactly what puts the account
    behind risk control / captcha. The TUI path already refuses this
    ("Auto-login needs password. Cancelled."); the CLI did not.
    """
    who = (email or "").strip() or "<you@example.com>"
    lines = ["login needs an account email AND a password; nothing was "
             "opened and nothing was sent.",
             f"  interactive:  deepseaport login --email {who}",
             f"  scripted:     deepseaport login --email {who} < password.txt"
             "   (stdin is the password, one line, no --email prompt)"]
    if not (email or "").strip():
        lines.insert(1, "  ! no email: pass --email, or type it at the prompt."
                        " On a pipe there is no prompt, so a piped secret is "
                        "never read as the email.")
    if not password:
        lines.insert(1, "  ! no password: type it at the no-echo prompt, pass "
                        "--password (visible in the process list), or pipe one "
                        "line in.")
    return "\n".join(lines)


def _token_expires_at_for(provider_id: str, token: str) -> float:
    """Best-effort JWT ``exp`` for a token from ``provider_id``. 0.0 when unknown.

    Qwen's ``jwt_exp`` is the decoder: a generic base64url JWT-``exp`` read that
    is right for Qwen and for every provider whose token is a plain JWT. It is
    also the only ``jwt_exp`` in the tree, so per-provider dispatch resolved to
    this same function and only added an ``importlib`` round-trip. 0.0 is always
    a valid answer: the expiry is a pre-check, so "unknown" must never be fatal.
    """
    try:
        from .providers.qwen.auth import jwt_exp as _qwen_jwt_exp

        return float(_qwen_jwt_exp(token) or 0.0)
    except Exception:
        return 0.0


def cmd_login(args) -> int:
    """Interactive login for the resolved provider -> saves the account token."""

    from .config import load_settings

    settings = load_settings(args.config)
    # Name the provider actually being logged into: "DeepSeek email" while
    # driving a Qwen signin pointed the user at the wrong service. DeepSeek's
    # display_name is "DeepSeek", so its prompts stay byte-identical.
    _pid = _valid_provider_id(getattr(args, "provider", None)) or \
        _valid_provider_id(getattr(settings, "provider", None)) or "deepseek"
    # Shape is not membership, and only the FLAG is checked here (a
    # hand-edited settings.provider is normalized elsewhere; see
    # _upsert_account). deepseek is exempt because its login is the direct
    # Obscura browser path, not a registry entry point -- so the default and
    # every `--provider deepseek` keep working unchanged.
    _flag_pid = _valid_provider_id(getattr(args, "provider", None)) or None
    if _flag_pid and _flag_pid != "deepseek" and not provider_is_registered(_flag_pid):
        return _unknown_provider_exit(_flag_pid)
    _label = _provider_display_name(_pid)
    _email_arg = getattr(args, "email", "") or ""
    if _email_arg:
        email = _email_arg
    elif _stdin_is_tty():
        email = input(f"{_label} email: ").strip()
    else:
        # No --email and no console: do NOT prompt. A non-tty stdin here is
        # the secret stream (`login < password.txt`, a service wrapper, CI),
        # and a bare input() reads its first line â€” the PASSWORD â€” as the
        # email address, so the login then ran as
        # (email=<the password>, password=""). Same tty probe _secret_from_arg
        # uses, so both prompts agree about who owns stdin.
        email = ""
    password = _secret_from_arg(getattr(args, "password", "") or "",
                                "--password", f"{_label} password")
    # Skipped when the driver is injected: an injected driver owns its own
    # empty-credential handling and never reaches the real browser, and
    # test_cli_config_hardening.py::test_piped_secret_eof_is_the_no_password_case
    # pins that a piped EOF still reaches the driver with password="". The
    # shipped driver is the one the risk-control guard is for.
    if (_obscura_login_token is _CANONICAL_LOGIN_DRIVER
            and (not (email or "").strip() or not password)):
        print(_missing_credential(email, password))
        return 2
    _login_prov = _valid_provider_id(getattr(args, "provider", None)) or None
    if _login_prov and _login_prov != "deepseek":
        # Non-DeepSeek: registry login (keeps 3-arg mock seam for default).
        try:
            from .core import registry as _reg_login
            _prov_obj = _reg_login.get(_login_prov)
            if _prov_obj is not None:
                token, login_banned, ban_detail = _unpack_login_token_result(
                    _prov_obj.login(email, password, settings))
            else:
                token, login_banned, ban_detail = _unpack_login_token_result(
                    _obscura_login_token(email, password, settings))
        except Exception as exc:
            print(f"Auto-login failed: {exc}")
            return 3
    elif _login_prov == "deepseek":
        # --provider deepseek must PIN the driver: without the hint
        # _obscura_login_token resolves settings.provider, so a config set to
        # qwen sent the credentials to the QWEN signin and stored its JWT.
        token, login_banned, ban_detail = _unpack_login_token_result(
            _obscura_login_token(email, password, settings, "deepseek"))
    else:
        token, login_banned, ban_detail = _unpack_login_token_result(
            _obscura_login_token(email, password, settings))
    if not token:
        if login_banned:
            print(f"login refused for {email}: account BANNED per login page.")
            if ban_detail:
                print(f"Evidence: {ban_detail[:300]}")
        elif ban_detail:
            # A single ban-wording sighting does not set login_banned, but the
            # driver still reports it here; the user deserves to hear it.
            print(f"Evidence: {ban_detail[:300]}")
        return 3

    _prov = _valid_provider_id(getattr(args, "provider", None)) or None
    _login_expires_at = 0.0
    if _prov and _prov != "deepseek" and token:
        # providers.qwen.auth.login_with_expiry captures the signin expiry, but
        # the registry login() seam returns the legacy 3-tuple and discards it,
        # so the expiry has to be recovered from the token afterwards.
        _login_expires_at = _token_expires_at_for(_prov, token)
    acc = _upsert_account(settings, email, password, token, provider=_prov,
                          token_expires_at=_login_expires_at)
    if login_banned:
        # A suspended account can still complete login and expose a token.
        # Keep that token (so the Accounts list can probe/display the ban) and
        # persist the best human-readable expiry we can recover.
        try:
            from .providers.deepseek.auth import parse_ban_until
            until = parse_ban_until(ban_detail)
        except Exception:
            until = None
        try:
            from .accounts import check_ban_for_token
            try:
                if _prov and _prov != "deepseek":
                    _banned, _until = check_ban_for_token(token, provider=_prov)
                else:
                    _banned, _until = check_ban_for_token(token)
            except TypeError:
                # Legacy monkeypatch without the provider kwarg (tests).
                _banned, _until = check_ban_for_token(token)
            until = _until or until
        except Exception:
            pass
        acc.banned = True
        acc.banned_until = float(until or 0.0)
        saved = _save(settings)
        if saved:
            print(f"login ok, token saved for {email}, but account is BANNED.")
        else:
            # The browser authorization is not repeatable on demand and a
            # retry is another sign-in from this IP: show the token.
            print(f"login ok for {email}, but account is BANNED; "
                  f"token value: {token}")
        if acc.banned_until:
            from .providers.deepseek.protocol import format_ban_datetime
            _expires = format_ban_datetime(acc.banned_until)
            if _expires:
                print("Expires:", _expires)
        if ban_detail:
            print(f"Evidence: {ban_detail[:300]}")
        if saved:
            print(f"Token -> {settings.config_path} (accounts list will show BANNED)")
        return 3

    if not _save(settings):
        # Same reason as the banned branch: never throw away the token the
        # ~30s browser run just produced.
        print(f"login ok, token not saved for {email}; token value: {token}")
        return 3
    print(f"login ok, token saved for {email} -> {settings.config_path}")
    return 0


def cmd_chat(args) -> int:
    """Single non-stream chat against the local server logic (needs account)."""
    from fastapi import HTTPException

    from .accounts import AccountPool, token_help_for
    from .config import load_settings
    from .core.obscura_bridge import ObscuraBridge
    from .server import create_app
    from .server_routes import _prepare

    def _chat_fail(exc: BaseException) -> int:
        """One short actionable line + a non-zero code instead of a traceback.

        `chat` is a one-shot CLI command, so a raw traceback IS the whole UX,
        and the frozen exe then appends "Press Enter to close..." to it. The
        three escapees: ValueError from pool.acquire ("no DeepSeek accounts in
        pool"), HTTPException from _prepare (404 unknown model, 400 bad
        messages), and TimeoutError when every slot stays busy.
        """
        if isinstance(exc, HTTPException):
            print(f"  ! chat failed: HTTP {exc.status_code}: {exc.detail}")
        elif isinstance(exc, TimeoutError):
            print("  ! chat failed: timed out waiting for a free account slot "
                  "(all accounts busy or cooling down). Retry, or add an account.")
        else:
            print(f"  ! chat failed: {exc}")
        print()
        print(token_help_for(getattr(settings, "provider", "deepseek")))
        return 1

    settings = load_settings(args.config)
    from .core import registry as _reg

    _chat_prov = getattr(args, "provider", None)
    if _chat_prov:
        _pid = str(_chat_prov).strip().lower()
        if _reg.get(_pid) is None:
            print(f"unknown provider: {_chat_prov}")
            print("available: " + ", ".join(sorted(_reg.registered().keys())))
            return 2
        settings.provider = _pid
        settings.providers = []
    app = create_app(settings)
    try:
        prep = _prepare({"model": args.model, "messages": [{"role": "user", "content": args.prompt}]}, settings)
    except (ValueError, HTTPException, TimeoutError) as exc:
        return _chat_fail(exc)
    try:
        _chat_provider = _reg.get(prep.get("provider", ""))
    except Exception:
        _chat_provider = None
    if _chat_provider is None:
        # Unresolvable provider: refuse HERE, because the account-holding
        # branch below has no provider to dispatch to. _chat_fail rather than a
        # raise -- `chat` is a one-shot CLI, where a raw traceback is the whole
        # UX (see its docstring).
        return _chat_fail(ValueError(f"no provider registered for {prep.get('provider')!r}"))
    if getattr(_chat_provider, "needs_bridge", True):
        from .core.obscura_bridge import ObscuraBridge, register_default_bridge

        app.state.bridge = ObscuraBridge(settings.obscura_bin, settings.obscura_profile)
        # Publish for protocol.bridge_headers ban probes; without this the probe
        # falls back to bare headers (no WAF cookie/UA) and can false-negative.
        register_default_bridge(app.state.bridge)
        # Only a jar consumer needs the CHAT_HOME (DeepSeek-host) fetch. Qwen is
        # not one: this bridge is its engine/UA source, while its cookies come
        # from the qwen-host cache -- warming would boot obscura against
        # chat.deepseek.com for a qwen-only request.
        if getattr(_chat_provider, "needs_waf_jar", True):
            app.state.bridge.warmup()
    else:
        # Local providers (opencode CLI) need no WAF bridge: skip obscura.
        app.state.bridge = None
    if not getattr(_chat_provider, "requires_accounts", True):
        # Local (accountless) provider (opencode CLI): no pool, no slot.
        result = _chat_provider.complete(app, None, prep, None)
        print(("THINKING:\n" + result["thinking"] + "\n\n" if result["thinking"] else "") + result["content"])
        return 0
    try:
        from .core import pools as _pools

        _prov_accts = _pools.accounts_for_provider(settings, prep.get("provider", ""))
        try:
            _current = _pools.current_for(settings, prep.get("provider", ""))
        except Exception:
            _current = getattr(settings, "active_account", "")
    except Exception:
        _prov_accts = settings.accounts
        _current = getattr(settings, "active_account", "")
    pool = AccountPool(_prov_accts, current=_current)
    try:
        item = pool.acquire(timeout=30,
                            allow_failover=bool(getattr(settings, "use_multiple_accounts", True)))
        # Through the PROVIDER, never DeepSeek's engine directly. `chat` serves
        # --provider qwen / opencode too, and calling DeepSeek's
        # _complete_with_failover_sync here ran DeepSeek's whole code path for
        # every provider: it closes over its own module globals and never reads
        # prep["provider"], so `chat --provider qwen` transmitted the Qwen ES256
        # JWT to chat.deepseek.com as a userToken.
        #
        # `complete` and not `complete_request` because it is the seam whose
        # cancel_event defaults to None -- `chat` has no cancellation source --
        # and because every provider forwards it to its OWN
        # _complete_with_failover_sync as a KEYWORD, so the same-request ban
        # failover this line existed for is unchanged and the existing
        # 4-positional doubles of the engine seam still fit.
        result = _chat_provider.complete(app, pool, prep, item)
    except (ValueError, HTTPException, TimeoutError) as exc:
        return _chat_fail(exc)
    print(("THINKING:\n" + result["thinking"] + "\n\n" if result["thinking"] else "") + result["content"])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="deepseaport")
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="cmd")
    serve = sub.add_parser("serve",
                           help="run the OpenAI-compatible API server "
                                "(main menu; TUI when a console is available)")
    serve.add_argument("--host", default="",
                       help="bind address; overrides config listen=true/false")
    serve.add_argument("--port", type=_port_number, default=0)
    serve.add_argument("--workers", type=int, default=1)
    serve.add_argument("--no-tui", action="store_true",
                       help="skip the main-page TUI and start the server directly")
    serve.add_argument("--provider", default=None,
                       help="serve a single provider (id from registry)")
    serve.add_argument("--providers", default=None,
                       help="comma-separated provider ids for multi-provider mode")
    sub.add_parser("waf", help="warm the Obscura browser and print WAF/PoW state")
    models_p = sub.add_parser("models", help="print the model ids this config serves")
    models_p.add_argument("--provider", default=None,
                          help="print model ids for one provider only")
    acc = sub.add_parser("accounts", help="list/add/remove pooled accounts")
    acc_sub = acc.add_subparsers(dest="accounts_action", required=True)
    acc_sub.add_parser("list", help="show pooled accounts + token status")
    acc_sel = acc_sub.add_parser("select", help="set CURRENT account (preferred first)")
    acc_sel.add_argument("identifier", nargs="?", default="",
                         help="email to select; empty/ALL/NONE clears")
    acc_sel.add_argument("--clear", action="store_true",
                         help="clear selection -> ALL (auto-failover)")
    acc_add = acc_sub.add_parser("add", help="add account (needs --token, see TOKEN_HELP)")
    acc_add.add_argument("--email", default="", help="account email")
    acc_add.add_argument("--mobile", default="", help="account mobile")
    acc_add.add_argument("--password", default="",
                         help="required for browser auto-refresh on 40003 (direct API login fails); "
                              "SECRET: the value is visible in the process list and shell history -- "
                              "omit the flag to be prompted without echo")
    acc_add.add_argument("--token", default="",
                         help="userToken value from chat.deepseek.com localStorage (raw or JSON); "
                              "SECRET: the value is visible in the process list and shell history -- "
                              "omit the flag to be prompted without echo")
    acc_add.add_argument("--provider", default=None,
                         help="bind the new account to a provider id (default: selected provider)")
    acc_st = acc_sub.add_parser("set-token",
                                help="paste/refresh the userToken of an existing "
                                     "account (--token is REQUIRED)")
    acc_st.add_argument("identifier", help="email/mobile/identifier")
    acc_st.add_argument("--token", default="", required=True,
                        help="userToken value (raw 64-char or full JSON). "
                             "REQUIRED: set-token has no prompt, and a "
                             "tokenless account cannot serve requests")
    acc_rm = acc_sub.add_parser("remove", help="delete account from pool")
    acc_rm.add_argument("identifier", help="email/mobile/identifier/token")
    acc_unblock = acc_sub.add_parser("unblock", help="clear persisted + in-memory ban/cooldown")
    acc_unblock.add_argument("identifier", nargs="?", default="",
                             help="email/mobile/identifier; empty = all accounts "
                                  "(an unknown identifier exits 1, not 'cleared 0')")
    login = sub.add_parser("login",
                           help="browser login for one account; saves its token "
                                "(both --email and a password are required)")
    login.add_argument("--email", default="",
                       help="account email; required when stdin is a pipe "
                            "(a piped stdin is the password, never the email)")
    login.add_argument("--password", default="",
                       help="account password; SECRET: the value is visible in the "
                            "process list and shell history -- omit the flag to be "
                            "prompted without echo")
    login.add_argument("--provider", default=None,
                       help="bind the logged-in account to a provider id")
    chat = sub.add_parser("chat", help="one non-stream chat against the local server logic")
    chat.add_argument("--prompt", required=True)
    chat.add_argument("--model", default="deepseek-flash")
    chat.add_argument("--provider", default=None,
                      help="chat against one provider's accounts")
    args = parser.parse_args(argv)
    if args.cmd is None:
        # No subcommand (e.g. double-clicked exe) -> start the server.
        args = argparse.Namespace(config=args.config, host="", port=0,
                                  workers=1, no_tui=False, cmd="serve",
                                  provider=None, providers=None)
    if args.cmd == "serve":
        return cmd_serve(args)
    if args.cmd == "waf":
        return cmd_waf(args)
    if args.cmd == "models":
        return cmd_models(args)
    if args.cmd == "accounts":
        return cmd_accounts(args)
    if args.cmd == "login":
        return cmd_login(args)
    if args.cmd == "chat":
        return cmd_chat(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
