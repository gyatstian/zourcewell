"""CLI: serve | waf | models | chat."""

from __future__ import annotations

import argparse
import json
import re
import sys

from . import server_runner as _runner
from .auth import browser_login as _browser_login, unpack_login_result as _unpack_login_result
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


def _apply_serve_provider_flags(settings, args) -> int | None:
    """Apply --provider/--providers to settings. None = ok, int = exit code."""
    from .core import registry as _reg

    single = getattr(args, "provider", None)
    multi = getattr(args, "providers", None)
    if single:
        pid = str(single).strip().lower()
        if _reg.get(pid) is None:
            print(f"unknown provider: {single}")
            print("available: " + ", ".join(sorted(_reg.registered().keys())))
            return 2
        settings.provider = pid
        settings.providers = []
        settings.save()
        return None
    if multi:
        ids: list[str] = []
        for part in str(multi).split(","):
            key = part.strip().lower()
            if key and key not in ids:
                ids.append(key)
        if not ids:
            return None
        available = _reg.registered()
        for pid in ids:
            if _reg.get(pid) is None:
                print(f"unknown provider: {pid}")
                print("available: " + ", ".join(sorted(available.keys())))
                return 2
        if len(ids) == 1:
            settings.provider = ids[0]
            settings.providers = []
        else:
            settings.providers = ids
            settings.provider = ids[0]
        settings.save()
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
    if not no_tui and sys.stdin.isatty():
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
    from .obscura_bridge import ObscuraBridge

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
        settings.save()
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
            from .protocol import format_ban_label as _ban_label
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
            settings.save()
            print("Selection cleared -> ALL (auto-failover)")
            return 0
        pool = AccountPool(settings.accounts, current=settings.active_account)
        if not pool.set_current(ident):
            print(f"not found: {ident}")
            return 1
        settings.active_account = pool.current
        settings.save()
        print(f"CURRENT -> [{pool.current}] (saved)")
        return 0

    if action == "add":
        raw_token = args.token or ""
        cfg = AccountConfig(email=args.email or "", mobile=args.mobile or "",
                            password=args.password or "", token=extract_token(raw_token))
        if not (cfg.email or cfg.mobile or cfg.token):
            print("need --email, --mobile, or --token")
            print(TOKEN_HELP)
            return 2
        _flag = _valid_provider_id(getattr(args, "provider", None))
        if _flag:
            cfg.provider = _flag
        else:
            try:
                from .core import pools as _pools

                cfg.provider = _pools.selected_provider(settings)
            except Exception:
                pass
        pool = AccountPool(settings.accounts, current=settings.active_account)
        try:
            from .accounts import add_account as _add
            _add(settings, pool, cfg)
        except ValueError as exc:
            print(f"add failed: {exc}")
            return 1
        settings.save()
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
            settings.save()
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
        settings.save()
        print(f"removed {identifier}")
        return 0

    if action == "unblock":
        # Clear persisted ban markers in config. A running server also needs
        # POST /v1/accounts/unblock to drop its in-memory cooldown singleton.
        from .accounts import _matches as _m
        ident = (getattr(args, "identifier", "") or "").strip()
        n = 0
        for a in settings.accounts:
            if ident and not _m(a, ident):
                continue
            if getattr(a, "banned", False) or getattr(a, "banned_until", 0.0):
                n += 1
            a.banned = False
            a.banned_until = 0.0
        settings.save()
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
    for a in settings.accounts:
        if a.email and _m(a, email):
            a.token = token
            if password:
                a.password = password
            if token_expires_at:
                a.token_expires_at = token_expires_at
            return a
    cfg = AccountConfig(email=email, password=password, token=token)
    try:
        from .core import pools as _pools

        cfg.provider = (str(provider).strip().lower() if provider
                        else _pools.selected_provider(settings))
    except Exception:
        if provider:
            cfg.provider = str(provider).strip().lower()
    if not (cfg.provider or "").strip():
        cfg.provider = "deepseek"
    settings.accounts.append(cfg)
    # New account becomes CURRENT (default) immediately.
    settings.active_account = cfg.identifier
    return cfg


# Login implementation lives in deepseaport.auth (selector-driven, visible
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
    pid = (str(provider or "").strip().lower()
           or (str(getattr(settings, "provider", "") or "").strip().lower())
           or "deepseek")
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


def cmd_login(args) -> int:
    """Interactive login inside Obscura (real device fingerprint) -> saves userToken."""
    import getpass

    from .config import load_settings

    settings = load_settings(args.config)
    email = args.email or input("DeepSeek email: ").strip()
    password = args.password or getpass.getpass("DeepSeek password: ")
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
    else:
        token, login_banned, ban_detail = _unpack_login_token_result(
            _obscura_login_token(email, password, settings))
    if not token:
        if login_banned:
            print(f"login refused for {email}: account BANNED per login page.")
            if ban_detail:
                print(f"Evidence: {ban_detail[:300]}")
        return 3

    _prov = _valid_provider_id(getattr(args, "provider", None)) or None
    _login_expires_at = 0.0
    if _prov and _prov != "deepseek" and token:
        # Qwen signin expiry is captured by providers.qwen.auth.login_with_expiry,
        # but the registry login seam returns the legacy 3-tuple. Recover the
        # JWT exp claim best-effort so the engine can pre-check expiry.
        try:
            from .providers.qwen.auth import jwt_exp as _qwen_jwt_exp
            _login_expires_at = float(_qwen_jwt_exp(token) or 0.0)
        except Exception:
            _login_expires_at = 0.0
    acc = _upsert_account(settings, email, password, token, provider=_prov,
                          token_expires_at=_login_expires_at)
    if login_banned:
        # A suspended account can still complete login and expose a token.
        # Keep that token (so the Accounts list can probe/display the ban) and
        # persist the best human-readable expiry we can recover.
        try:
            from .auth import parse_ban_until
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
        settings.save()
        print(f"login ok, token saved for {email}, but account is BANNED.")
        if acc.banned_until:
            from .protocol import format_ban_datetime
            print("Expires:", format_ban_datetime(acc.banned_until))
        if ban_detail:
            print(f"Evidence: {ban_detail[:300]}")
        print(f"Token -> {settings.config_path} (accounts list will show BANNED)")
        return 3

    settings.save()
    print(f"login ok, token saved for {email} -> {settings.config_path}")
    return 0


def cmd_chat(args) -> int:
    """Single non-stream chat against the local server logic (needs account)."""
    from .accounts import AccountPool
    from .config import load_settings
    from .obscura_bridge import ObscuraBridge
    from .server import _complete_with_failover_sync, _prepare, create_app

    settings = load_settings(args.config)
    _chat_prov = getattr(args, "provider", None)
    if _chat_prov:
        from .core import registry as _reg

        _pid = str(_chat_prov).strip().lower()
        if _reg.get(_pid) is None:
            print(f"unknown provider: {_chat_prov}")
            print("available: " + ", ".join(sorted(_reg.registered().keys())))
            return 2
        settings.provider = _pid
        settings.providers = []
    app = create_app(settings)
    from .obscura_bridge import register_default_bridge

    app.state.bridge = ObscuraBridge(settings.obscura_bin, settings.obscura_profile)
    # Publish for protocol.bridge_headers ban probes; without this the probe
    # falls back to bare headers (no WAF cookie/UA) and can false-negative.
    register_default_bridge(app.state.bridge)
    app.state.bridge.warmup()
    prep = _prepare({"model": args.model, "messages": [{"role": "user", "content": args.prompt}]}, settings)
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
    item = pool.acquire(timeout=30,
                        allow_failover=bool(getattr(settings, "use_multiple_accounts", True)))
    # Same-request ban failover as the server endpoint: a banned CURRENT
    # transparently retries on the next unbanned account.
    result = _complete_with_failover_sync(app, pool, prep, item)
    print(("THINKING:\n" + result["thinking"] + "\n\n" if result["thinking"] else "") + result["content"])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="deepseaport")
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="cmd")
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="",
                       help="bind address; overrides config listen=true/false")
    serve.add_argument("--port", type=int, default=0)
    serve.add_argument("--workers", type=int, default=1)
    serve.add_argument("--no-tui", action="store_true",
                       help="skip the main-page TUI and start the server directly")
    serve.add_argument("--provider", default=None,
                       help="serve a single provider (id from registry)")
    serve.add_argument("--providers", default=None,
                       help="comma-separated provider ids for multi-provider mode")
    sub.add_parser("waf")
    models_p = sub.add_parser("models")
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
    acc_add.add_argument("--password", default="", help="required for browser auto-refresh on 40003 (direct API login fails)")
    acc_add.add_argument("--token", default="",
                         help="userToken value from chat.deepseek.com localStorage (raw or JSON)")
    acc_add.add_argument("--provider", default=None,
                         help="bind the new account to a provider id (default: selected provider)")
    acc_st = acc_sub.add_parser("set-token", help="paste/refresh userToken for existing account")
    acc_st.add_argument("identifier", help="email/mobile/identifier")
    acc_st.add_argument("--token", default="", required=True,
                        help="userToken value (raw 64-char or full JSON)")
    acc_rm = acc_sub.add_parser("remove", help="delete account from pool")
    acc_rm.add_argument("identifier", help="email/mobile/identifier/token")
    acc_unblock = acc_sub.add_parser("unblock", help="clear persisted + in-memory ban/cooldown")
    acc_unblock.add_argument("identifier", nargs="?", default="",
                             help="email/mobile/identifier; empty = all accounts")
    login = sub.add_parser("login")
    login.add_argument("--email", default="")
    login.add_argument("--password", default="")
    login.add_argument("--provider", default=None,
                       help="bind the logged-in account to a provider id")
    chat = sub.add_parser("chat")
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
