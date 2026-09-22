"""Stdlib-only TUI: main page (Start server / Settings) + settings editor.

No third-party deps (Windows-safe: plain input(), no curses) so `serve`
works everywhere. All edits mutate the passed Settings in place and persist
via Settings.save().
"""

from __future__ import annotations

import os
import sys
from typing import Callable

from .config import VALID_LOG_LEVELS, Settings
# Re-exported for backward compat (tests do tui._read_interactive).
from .tui_editor import (
    _WIN_SPECIAL_MAP,
    _apply_cursor_move,
    _char_cell,
    _decode_posix_special,
    _decode_windows_special,
    _editor_width,
    _ensure_vt,
    _line_start,
    _pos_line_col,
    _read_interactive,
    _read_interactive_posix,
    _read_interactive_windows,
    _read_keys_react,
    _redraw_editor,
    _unescape_newlines,
    _visual_pos,
)


def _on_off(value: bool) -> str:
    return "ON" if value else "OFF"


def _use_color() -> bool:
    """False when NO_COLOR is set or stdout is not a tty (logs/tests stay clean)."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def _c(text: str, code: str) -> str:
    if not _use_color():
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def _clear_screen() -> None:
    """Clear screen on menu enter. No-op when not a tty (tests/pipes)."""
    try:
        if not sys.stdout.isatty():
            return
    except Exception:
        return
    try:
        if os.name == "nt":
            # Enable ANSI VT processing on Windows 10+, best-effort.
            try:
                import ctypes

                kernel32 = ctypes.windll.kernel32
                handle = kernel32.GetStdHandle(-11)
                mode = ctypes.c_ulong()
                if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                    kernel32.SetConsoleMode(handle, mode.value | 0x0004)
            except Exception:
                pass
        print("\x1b[2J\x1b[H", end="")
    except Exception:
        pass


def _header(title: str) -> None:
    """Blank line + title + blank line. Single visual anchor per screen."""
    print()
    print(_c(f"=== {title} ===", "1;36"))
    print()


def _section(title: str) -> None:
    """Group related settings without adding a nested menu."""
    print()
    print(_c(f"  {title}", "1"))


def _setting_line(number: int, label: str, value: str,
                  width: int = 28) -> None:
    """One settings row: number, label, current value."""
    print(f"  {number:>2}. {label:<{width}} {value}")


def _ok(msg: str) -> None:
    print(_c(f"  ok: {msg}", "32"))


def _info(msg: str) -> None:
    print(f"  {msg}")


def _err(msg: str) -> None:
    print(_c(f"  ! {msg}", "31"))


def _ask(prompt: str, default: str = "") -> str:
    try:
        suffix = f" [{default}]" if default else ""
        return input(f"{prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise


def _ask_password(prompt: str = "Password (Enter=skip)") -> str:
    """Hidden password prompt (no echo). Preserves inner/edge spaces.

    Falls back to visible _ask when getpass cannot control the terminal
    (e.g. no tty in tests) — getpass itself falls back to input() there,
    so mocked builtins.input still works.
    """
    try:
        import getpass as _gp

        try:
            # getpass strips only the trailing newline, keeps spaces intact.
            return _gp.getpass(f"{prompt}: ")
        except (EOFError, KeyboardInterrupt):
            print()
            raise
        except Exception:
            return _ask(prompt)
    except (EOFError, KeyboardInterrupt):
        print()
        raise


def _has_token(settings: Settings, identifier: str) -> bool:
    from .accounts import account_has_token
    return account_has_token(settings, identifier)


def _toggle_bool(settings: Settings, attr: str, label: str) -> None:
    """Instant ON<->OFF flip, saved immediately (no y/n prompt)."""
    setattr(settings, attr, not bool(getattr(settings, attr, True)))
    settings.save()
    _ok(f"{label} -> {_on_off(getattr(settings, attr))} (saved)")


def _toggle_stream(settings: Settings) -> None:
    """Instant buffered<->live flip, saved immediately."""
    settings.stream_mode = "live" if settings.stream_mode != "live" else "buffered"
    settings.save()
    _ok(f"Reply streaming -> {settings.stream_mode} (saved)")


def _edit_choice(settings: Settings, attr: str, label: str, options: tuple[str, ...]) -> None:
    current = str(getattr(settings, attr))
    print()
    _info(f"{label} (current: {current})")
    for i, opt in enumerate(options, 1):
        mark = " *" if opt == current else ""
        print(f"    {i}. {opt}{mark}")
    print()
    ans = _ask("Number or value (Enter=cancel)", default="")
    if not ans:
        _info("Cancelled.")
        return
    try:
        idx = int(ans) - 1
    except ValueError:
        # Allow typing the value directly.
        lowered = {o.lower(): o for o in options}
        if ans.lower() in lowered:
            setattr(settings, attr, lowered[ans.lower()])
            settings.save()
            _ok(f"{label} -> {getattr(settings, attr)} (saved)")
        else:
            _err("Invalid choice, cancelled.")
        return
    if 0 <= idx < len(options):
        setattr(settings, attr, options[idx])
        settings.save()
        _ok(f"{label} -> {getattr(settings, attr)} (saved)")
    else:
        _err("Invalid choice, cancelled.")


def _edit_retries(settings: Settings) -> None:
    print()
    _info(f"Retries per failed request (current: {settings.max_retries}).")
    _info("0 disables retries; the original default is 1.")
    print()
    ans = _ask("Enter 0-5 (Enter=cancel)", default="")
    if not ans:
        _info("Cancelled.")
        return
    try:
        value = int(ans)
    except ValueError:
        _err("Not a number, cancelled.")
        return
    if 0 <= value <= 5:
        settings.max_retries = value
        settings.save()
        _ok(f"Retry count -> {value} (saved)")
    else:
        _err("Out of range 0-5, cancelled.")


def _mask_key(key: str) -> str:
    k = (key or "").strip()
    if len(k) <= 8:
        return "***"
    return f"{k[:4]}***{k[-2:]}"


def _edit_port(settings: Settings) -> None:
    """Prompt for port, free-check via cli helpers, save."""
    from .cli import _ensure_free_port, _resolve_bind_host

    print()
    _info(f"Port (current: {settings.port}). 1-65535.")
    print()
    ans = _ask("Enter port (Enter=cancel)", default="")
    if not ans:
        _info("Cancelled.")
        return
    try:
        candidate = int(ans)
    except ValueError:
        _err("Not a number, cancelled.")
        return
    if not 1 <= candidate <= 65535:
        _err("Out of range 1-65535, cancelled.")
        return
    host = _resolve_bind_host(settings)
    try:
        picked = _ensure_free_port(settings, host, candidate)
    except (EOFError, KeyboardInterrupt):
        print()
        return
    # _ensure_free_port saves the winner when it auto-picks; when the
    # candidate was free it returns it unsaved — persist either way.
    try:
        settings.port = int(picked)
        settings.save()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    _ok(f"Port -> {settings.port} (saved)")


def _edit_keys(settings: Settings) -> None:
    """Add/remove API keys. Values masked, never printed in full."""
    while True:
        print()
        keys = list(getattr(settings, "keys", []) or [])
        if keys:
            _info(f"API keys ({len(keys)}):")
            for i, k in enumerate(keys, 1):
                print(f"    {i}. {_mask_key(k)}")
        else:
            _info("API keys (0): empty = local API needs no Bearer.")
        print()
        print("    a. Add key")
        print("    d. Delete key by number")
        print("    b. Back")
        print()
        try:
            ans = input("Keys (a/d <n>/b): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not ans or ans in ("b", "back", "q", "quit"):
            return
        if ans in ("a", "add"):
            try:
                import getpass as _gp

                try:
                    raw = _gp.getpass("New API key (Enter=cancel): ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return
                except Exception:
                    raw = _ask("New API key (Enter=cancel)", default="")
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not raw:
                _info("Cancelled.")
                continue
            if raw in keys:
                _err("Key already present, cancelled.")
                continue
            keys.append(raw)
            settings.keys = keys
            settings.save()
            _ok(f"Key added ({len(keys)} total, saved)")
            continue
        if ans.startswith("d"):
            parts = ans.split()
            num = parts[1] if len(parts) > 1 else ""
            if not num:
                try:
                    num = _ask("Number to DELETE", default="")
                except (EOFError, KeyboardInterrupt):
                    print()
                    return
            try:
                idx = int(num) - 1
            except ValueError:
                _err("Not a number, cancelled.")
                continue
            if 0 <= idx < len(keys):
                try:
                    confirm = input(f"Delete key {idx + 1} ({_mask_key(keys[idx])})? Type y: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print()
                    continue
                if confirm not in ("y", "yes"):
                    _info("Cancelled (nothing deleted).")
                    continue
                del keys[idx]
                settings.keys = keys
                settings.save()
                _ok(f"Key removed ({len(keys)} left, saved)")
            else:
                _err("Out of range, nothing deleted.")
            continue
        _err("Unknown option. Use a, d <n>, or b.")


def _edit_obscura_bin(settings: Settings) -> None:
    """Set obscura binary path. Empty/'auto'/'clear' resets to auto-discover."""
    from pathlib import Path

    print()
    cur = (getattr(settings, "obscura_bin", "") or "").strip()
    _info(f"Obscura binary (current: {cur or '(auto-discover)'}).")
    _info("Drop obscura.exe next to config.json, or set explicit path.")
    print()
    ans = _ask("Enter path (Enter=cancel, 'clear'=auto)", default="")
    if not ans:
        _info("Cancelled.")
        return
    low = ans.strip().lower()
    if low in ("clear", "auto", "none"):
        settings.obscura_bin = ""
        settings.save()
        _ok("Obscura binary -> auto-discover (saved)")
        return
    p = Path(ans.strip()).expanduser()
    if not p.exists():
        _err(f"Path not found: {p}. Not saved.")
        _info("TIP: place obscura.exe next to config.json, bin/, tools/, vendor/.")
        return
    settings.obscura_bin = str(p)
    settings.save()
    _ok(f"Obscura binary -> {settings.obscura_bin} (saved)")


def _edit_chat_model(settings: Settings) -> None:
    """Pick chat_model from the active providers' models (fallback to known four)."""
    options: tuple[str, ...] = ()
    try:
        from .core import pools as _pools
        from .core import registry as _reg

        ids = _pools.active_provider_ids(settings)
        found: list[str] = []
        for _pid in ids:
            _p = _reg.get(_pid)
            if _p is None:
                continue
            try:
                found.extend(list(_p.models().keys()))
            except Exception:
                continue
        if found:
            options = tuple(sorted(set(found)))
    except Exception:
        options = ()
    if not options:
        try:
            from .server import MODELS as _MODELS

            options = tuple(sorted(_MODELS)) if isinstance(_MODELS, dict) else ()
        except Exception:
            options = ()
    if not options:
        options = ("deepseek-flash", "deepseek-flash-reasoner",
                   "deepseek-flash-search", "deepseek-flash-reasoner-search")
    else:
        # Registry unavailable (or active provider unknown): keep the current
        # value selectable alongside the known four.
        try:
            cur = str(getattr(settings, "chat_model", "") or "").strip()
        except Exception:
            cur = ""
        if cur and cur not in options:
            options = tuple(sorted(set(options) | {cur}))
    _edit_choice(settings, "chat_model", "Chat model", options)


def _select_provider(settings: Settings) -> None:
    """Shared provider-selection loop used by the serve gate and settings item 19.

    Never raises on bad input; EOF/KeyboardInterrupt keeps the current
    selection and returns.
    """
    from .core import registry as _reg

    try:
        from .core import pools as _pools
    except Exception:
        _pools = None  # type: ignore
    while True:
        try:
            providers = _reg.registered()
        except Exception:
            providers = {}
        ids = list(providers.keys())
        _header("Provider")
        try:
            active = list(_pools.active_provider_ids(settings)) if _pools else [
                (getattr(settings, "provider", "") or "deepseek").strip().lower()]
            multi = len(active) > 1
        except Exception:
            active = [(getattr(settings, "provider", "") or "deepseek").strip().lower()]
            multi = False
        if multi:
            _info("Mode: multi (one server will expose all listed providers' models)")
        else:
            _info(f"Mode: single (current: {active[0] if active else 'deepseek'})")
        print()
        if not ids:
            _err("No providers registered.")
            print()
            print("  q. Back")
            print()
            try:
                choice = input("Select: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if choice in ("q", "quit", "b", "back", ""):
                return
            _err("Unknown option.")
            continue
        for i, pid in enumerate(ids, 1):
            try:
                prov = providers[pid]
                name = str(getattr(prov, "display_name", "") or pid)
                nmodels = len(prov.models())
            except Exception:
                name, nmodels = pid, 0
            try:
                accts = _pools.accounts_for_provider(settings, pid) if _pools else []
            except Exception:
                accts = []
            ntok = sum(1 for a in accts if (getattr(a, "token", "") or "").strip())
            mark = " * CURRENT" if (pid in active) else ""
            print(f"  {i}. {name} ({pid}) — {nmodels} models, "
                  f"{len(accts)} accounts ({ntok} with token){mark}")
        print()
        print("  m. Multi (all providers, one server exposes every model)")
        print("  q. Back (keep current)")
        print()
        try:
            choice = input("Select: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if choice in ("q", "quit", "b", "back", ""):
            return
        if choice in ("m", "multi"):
            settings.providers = list(ids)
            settings.provider = ids[0] if ids else "deepseek"
            try:
                settings.save()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            _ok(f"Provider mode -> multi ({', '.join(ids)}) (saved)")
            _info("One server will expose all listed providers' models.")
            continue
        picked = ""
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(ids):
                picked = ids[idx]
        elif choice:
            for pid in ids:
                if pid.strip().lower() == choice:
                    picked = pid
                    break
        if not picked:
            _err("Unknown option. Type a number, a provider id, m, or q.")
            continue
        settings.provider = picked
        settings.providers = []
        try:
            settings.save()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        _ok(f"Provider -> {picked} (saved)")


def provider_menu(settings: Settings) -> None:
    """First-run provider gate for `serve`. Always returns (never exits)."""
    _clear_screen()
    _select_provider(settings)


def _preview_append(value: str, width: int = 30) -> str:
    """One-line preview so multi-line text never breaks the menu layout."""
    text = str(value or "")
    if not text.strip():
        return "(empty)"
    lines = text.strip("\r\n").splitlines() or [text]
    first = lines[0].strip()
    if len(first) > width:
        first = first[:width - 1] + "…"
    if len(lines) > 1:
        chars = len(text)
        return f"{first or '…'} ({len(lines)} lines, {chars} chars)"
    return first or "(empty)"


def _edit_append_text(settings: Settings, attr: str, label: str) -> None:
    """Interactive editor: current text preloaded in the field, Enter saves.

    Arrow Left/Right move within text, Up/Down change lines, Backspace/Delete
    edit at the cursor, Ctrl+U clears the field. Typed ``\\n`` opens a new
    line at the cursor (never shown literally). Unchanged Enter just saves
    as-is, empty field + Enter cancels (keeps old), 'clear' empties the
    setting, Esc aborts. Stored value holds real newlines (``\\\\`` types
    one backslash).
    """
    current = str(getattr(settings, attr, "") or "")
    try:
        is_tty = bool(sys.stdin.isatty())
    except Exception:
        is_tty = False
    print()
    if is_tty:
        if current.strip():
            _info(f"{label} ({len(current.splitlines())} lines, {len(current)} chars — loaded below, edit away).")
        else:
            _info(f"{label} (current: empty).")
    elif current.strip():
        _info(f"{label} (current, {len(current.splitlines())} lines, {len(current)} chars):")
        for line in current.splitlines():
            print(f"    | {line}")
    else:
        _info(f"{label} (current: empty).")
    print()
    _info("Arrows move, \\n = new line, paste OK, Ctrl+U clears, Enter = save, Esc = cancel.")
    print()
    try:
        text = _read_interactive("> ", initial=current if is_tty else "")
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if text is None or not text.strip():
        _info("Cancelled.")
        return
    if text.strip().lower() == "clear":
        setattr(settings, attr, "")
        settings.save()
        _ok(f"{label} -> empty (saved)")
        return
    setattr(settings, attr, text)
    settings.save()
    saved = str(getattr(settings, attr))
    _ok(f"{label} -> {len(saved.splitlines())} lines, {len(saved)} chars (saved)")


def settings_menu(settings: Settings) -> None:
    """Blocking settings editor; returns on Back.

    Settings are grouped by purpose instead of shown as one flat list. Items
    1-5 predate the dry-run toggle (6); toggles save immediately and
    multi-value options prompt for a new value.
    """
    # Descriptor table: menu number -> zero-arg action. Numbers are stable
    # API (tests drive "1".."19"); edit actions here, not the chain below.
    def _actions() -> dict[str, Callable[[], None]]:
        return {
            "1": lambda: _toggle_bool(settings, "enable_tools", "Tool calling"),
            "2": lambda: _toggle_bool(settings, "warmup_on_startup", "WAF warmup at startup"),
            "3": lambda: _toggle_bool(settings, "auto_delete_session", "Delete session after reply"),
            "4": lambda: _edit_retries(settings),
            "5": lambda: _toggle_bool(settings, "parallel_challenge_fetch", "Parallel setup requests"),
            "6": lambda: _toggle_bool(settings, "send_dry_run_to_frontend", "Send dry run to frontend"),
            "7": lambda: _edit_choice(settings, "log_level", "Log level", VALID_LOG_LEVELS),
            "8": lambda: _toggle_stream(settings),
            "9": lambda: _toggle_bool(settings, "listen", "Allow LAN access"),
            "10": lambda: _edit_port(settings),
            "11": lambda: _edit_keys(settings),
            "12": lambda: _edit_obscura_bin(settings),
            "13": lambda: _edit_chat_model(settings),
            "14": lambda: _toggle_bool(settings, "use_multiple_accounts", "Account failover"),
            "15": lambda: _toggle_bool(settings, "enable_append", "Enable append"),
            "16": lambda: _edit_append_text(settings, "append_top", "Append at top"),
            "17": lambda: _edit_append_text(settings, "append_bottom", "Append at bottom"),
            "18": lambda: _toggle_bool(settings, "forgiving_toolcalls", "More forgiving toolcalls"),
            "19": lambda: _select_provider(settings),
        }
    _clear_screen()
    while True:
        _header(f"Settings ({settings.config_path or 'unsaved'})")

        _section("Request handling")
        _setting_line(1, "Tool calling", _on_off(settings.enable_tools))
        _setting_line(2, "Warm up WAF at startup",
                      _on_off(settings.warmup_on_startup))
        _setting_line(3, "Delete session after reply",
                      _on_off(settings.auto_delete_session))
        _setting_line(4, "Retries per failed request", str(settings.max_retries))
        _setting_line(5, "Parallel setup requests",
                      _on_off(settings.parallel_challenge_fetch))
        _setting_line(6, "Send dry run to frontend",
                      _on_off(getattr(settings, "send_dry_run_to_frontend", True)))

        _section("Chat & logs")
        _setting_line(7, "Log level", str(settings.log_level))
        _setting_line(8, "Reply streaming", str(settings.stream_mode))

        _section("Server access")
        _setting_line(9, "Allow LAN access", _on_off(settings.listen))
        _setting_line(10, "Port", str(settings.port))
        nkeys = len(getattr(settings, "keys", []) or [])
        keys_disp = f"{nkeys} set" if nkeys else "none (open)"
        _setting_line(11, "API keys", keys_disp)

        _section("Setup & defaults")
        obsc = (getattr(settings, "obscura_bin", "") or "").strip() or "(auto)"
        if len(obsc) > 30:
            obsc = "..." + obsc[-27:]
        _setting_line(12, "Obscura binary", obsc)
        _setting_line(13, "Chat model", str(settings.chat_model))

        _section("Accounts")
        _setting_line(14, "Account failover",
                      _on_off(getattr(settings, "use_multiple_accounts", True)))

        _section("Prompt")
        _setting_line(15, "Enable append",
                      _on_off(getattr(settings, "enable_append", True)))
        _setting_line(16, "Append at top",
                      _preview_append(getattr(settings, "append_top", "")))
        _setting_line(17, "Append at bottom",
                      _preview_append(getattr(settings, "append_bottom", "")))

        _section("Tools")
        _setting_line(18, "More forgiving toolcalls",
                      _on_off(getattr(settings, "forgiving_toolcalls", False)))

        _section("Provider")
        _setting_line(19, "Provider...", str(getattr(settings, "provider", "deepseek") or "deepseek"))

        print()
        print("  20. Back (q)")
        print()
        try:
            choice = input("Select: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if choice in ("20", "b", "back", "q", "quit", ""):
            return
        if choice not in {str(i) for i in range(1, 20)}:
            _err("Unknown option: type 1-19, or q.")
            continue
        # Clear the menu before prompting so repeated edits do not stack
        # multiple copies of the settings screen in the terminal.
        _clear_screen()
        try:
            _actions()[choice]()
        except (EOFError, KeyboardInterrupt):
            print()
            return


def accounts_menu(settings: Settings) -> None:
    """List/select/add/remove pooled accounts. Mutates settings + saves.

    Number NEVER deletes: it SELECTS current account. Delete is explicit
    via `d` + confirmation. Missing token offers in-TUI auto-login.
    """
    from .accounts import (TOKEN_HELP, AccountPool, extract_token,
                           sync_current_from_settings)
    from .config import AccountConfig

    def _pool() -> AccountPool:
        # Throwaway working copy, not the server's app.state.pools: the TUI
        # and the server never mutate at once (serve_fn blocks), and a
        # cross-process/thread shared pool would be unsound. Settings (saved
        # to disk) is the single hand-off; the server rebuilds/syncs from it.
        pool = AccountPool(settings.accounts, current=settings.active_account)
        if sync_current_from_settings(pool, settings):
            settings.save()
        return pool

    def _offer_auto_login(email: str, password: str, confirmed: bool = False) -> str:
        """Run Obscura login inside TUI. confirmed=True skips the Y/n prompt."""
        from .accounts import token_help_for as _token_help_for
        try:
            from .core import pools as _pools
            _pid = _pools.selected_provider(settings)
        except Exception:
            _pid = "deepseek"
        _pid = (str(_pid or "deepseek").strip().lower()) or "deepseek"
        _help = _token_help_for(_pid)
        if not email:
            return ""
        if not confirmed:
            try:
                ans = input("No token. Run auto-login via Obscura now? [Y/n]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return ""
            if ans not in ("", "y", "yes"):
                print()
                print(_help)
                return ""
        if not password:
            try:
                import getpass
                password = getpass.getpass("DeepSeek password (Enter=skip): ")
            except (EOFError, KeyboardInterrupt):
                print()
                return ""
        if not password:
            _err("Auto-login needs password. Cancelled.")
            print()
            print(_help)
            return ""

        def _registry_login():
            """Non-DeepSeek login via registry.get(pid).login; None = use shim."""
            if _pid == "deepseek":
                return None  # historical shim preserves seams/behavior
            try:
                from .core import registry as _reg
                prov = _reg.get(_pid)
                if prov is None:
                    return None
                raw = prov.login(email, password, settings)
                try:
                    from .providers.deepseek.auth import unpack_login_result as _unpack
                except Exception:
                    def _unpack(r):
                        if isinstance(r, tuple) and len(r) == 3:
                            return r
                        return (getattr(r, "token", None),
                                bool(getattr(r, "banned", False)),
                                str(getattr(r, "detail", "") or ""))
                return _unpack(raw)
            except NotImplementedError as exc:
                _err(f"Auto-login failed: {exc}")
                print()
                print(_help)
                return (None, False, "__handled__")
            except Exception as exc:
                _err(f"Auto-login failed: {exc}")
                print()
                print(_help)
                return (None, False, "__handled__")

        _info("Running Obscura login (browser, ~30s)...")
        try:
            _reg_result = _registry_login()
            if _reg_result is not None:
                if len(_reg_result) == 3 and _reg_result[2] == "__handled__":
                    return ""
                token, login_banned, ban_detail = _reg_result
            else:
                from .auth import obscura_login_token as _login_token
                from .auth import unpack_login_result as _unpack_login
                token, login_banned, ban_detail = _unpack_login(
                    _login_token(email, password, settings))
        except Exception as exc:
            _err(f"Auto-login failed: {exc}")
            print()
            print(_help)
            return ""
        if login_banned:
            if token:
                try:
                    from .auth import parse_ban_until
                    from .accounts import _matches as _m
                    until = parse_ban_until(ban_detail)
                    for a in settings.accounts:
                        if _m(a, email):
                            a.banned = True
                            a.banned_until = float(until or 0.0)
                except Exception:
                    pass
                _err(f"Auto-login captured a token for {email}, "
                     "but the account is BANNED. Token saved; Account "
                     "list will show the ban.")
                if ban_detail:
                    print(f"Evidence: {ban_detail[:300]}")
                return token
            _err(f"Auto-login refused for {email}: account BANNED per login page.")
            if ban_detail:
                print(f"Evidence: {ban_detail[:300]}")
            return ""
        if not token:
            return ""
        _ok(f"Auto-login ok for {email}")
        return token

    _clear_screen()
    while True:
        pool = _pool()
        rows = pool.status()
        current = pool.current or ""
        # Best-effort ban labels without blocking menu render: instant TTL
        # cache read + background refresh (network never on the draw path).
        # Skips missing/short (test) tokens without network; never raises.
        try:
            from .accounts import (ban_label_for as _ban_lookup,
                                   get_cached_ban_labels as _ban_cached,
                                   refresh_ban_labels_background as _ban_refresh)
            from . import protocol as _P
            ban_map = _ban_cached(settings.accounts) or {}
            try:
                _ban_refresh(settings.accounts)
            except Exception:
                pass
        except Exception:
            ban_map = {}
            _ban_lookup = None  # type: ignore
            _P = None  # type: ignore
        _header(f"Accounts ({len(settings.accounts)})")
        _info(f"Current: [{current or 'ALL (auto-failover)'}]")
        try:
            from .core import pools as _pools

            _scope = _pools.selected_provider(settings)
        except Exception:
            _scope = (getattr(settings, "provider", "") or "deepseek")
        _info(f"Provider scope: {_scope}")
        print()
        if rows:
            id_width = max([len("Identifier")] + [len(r["identifier"]) for r in rows])
            id_width = min(id_width, 32)
            print(f"    {'#':<2} {'Identifier':<{id_width}}  {'Uses':<4} {'Token':<7} {'Busy':<4} {'Cool':<6} {'Cur'}")
            print(f"    {'--':<2} {'-' * id_width}  {'----':<4} {'-----':<7} {'----':<4} {'----':<6} {'---'}")
            try:
                from .core.pools import _provider_of as _prov_of

                _row_provs = [_prov_of(a) for a in settings.accounts]
            except Exception:
                _row_provs = []
            for i, row in enumerate(rows, 1):
                mark = "*" if row.get("current") else ""
                has = _has_token(settings, row["identifier"])
                ident = row["identifier"]
                if len(ident) > 32:
                    ident = ident[:29] + "..."
                busy = "busy" if row.get("busy") else "-"
                try:
                    cool = float(row.get("cooldown_remaining", 0) or 0)
                except (TypeError, ValueError):
                    cool = 0
                cool_s = f"{cool:.0f}s" if cool > 0 else "-"
                # Red ban suffix: "(BANNED: 16 September)" from mute_until.
                ban_suffix = ""
                try:
                    _found, _until = (_ban_lookup(ban_map, row["identifier"])
                                      if _ban_lookup else (False, None))
                    if _found and _P is not None:
                        ban_suffix = " " + _c(_P.format_ban_label(_until), "31")
                    elif _found:
                        ban_suffix = " (BANNED)"
                except Exception:
                    ban_suffix = ""
                _prov = _row_provs[i - 1] if 0 < i <= len(_row_provs) else "deepseek"
                print(f"    {i:<2} {ident:<{id_width}}  {row['uses']:<4} "
                      f"{'yes' if has else 'missing':<7} {busy:<4} {cool_s:<6} {mark}{ban_suffix} [{_prov}]")
        else:
            _info("(no accounts yet -- use 'a' to add one)")
        print()
        _info("number = select account (never deletes)")
        print()
        print("  a. Add account")
        print("  t. Set/refresh token")
        print("  d. Delete account (asks confirmation)")
        print("  c. Clear selection (use ALL)")
        print("  h. How to get token")
        print()
        print("  b. Back (q)")
        print()
        try:
            choice = input("Select: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if choice in ("b", "back", "q", "quit", ""):
            return
        if choice == "h":
            _clear_screen()
            print()
            try:
                from .accounts import token_help_for as _th_for
                from .core import pools as _pools_h
                print(_th_for(_pools_h.selected_provider(settings)))
            except Exception:
                print(TOKEN_HELP)
            continue
        if choice == "c":
            _clear_screen()
            settings.active_account = ""
            settings.save()
            _ok("Selection cleared -> ALL (auto-failover)")
            continue
        if choice == "a":
            _clear_screen()
            print()
            try:
                email = _ask("Email (Enter=skip)", default="")
                password = _ask_password("Password (Enter=skip)")
                raw_token = _ask("Token (y = auto-token)", default="")
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if raw_token.strip().lower() in ("y", "yes"):
                token = _offer_auto_login(email, password, confirmed=True)
            else:
                token = extract_token(raw_token)
            cfg = AccountConfig(email=email, password=password, token=token)
            try:
                from .core import pools as _pools

                cfg.provider = _pools.selected_provider(settings)
            except Exception:
                pass
            try:
                from .accounts import add_account as _add
                _add(settings, pool, cfg)
            except ValueError as exc:
                _err(f"Add failed: {exc}")
                continue
            settings.save()
            _ok(f"Added {cfg.identifier} (saved)")
            if not cfg.token:
                _err("No token. Password-only fails.")
                print()
                print(TOKEN_HELP)
            continue
        if choice == "t":
            _clear_screen()
            print()
            try:
                ident = _ask("Account email", default="")
                token = _ask("New token value (Enter=skip = auto-login offer)", default="")
            except (EOFError, KeyboardInterrupt):
                print()
                return
            token = extract_token(token)
            if not ident:
                _info("Cancelled.")
                continue
            if not token:
                from .accounts import _matches as _m
                existing = next((a for a in settings.accounts if _m(a, ident)), None)
                if existing is None:
                    _err(f"Not found: {ident}")
                    continue
                token = _offer_auto_login(existing.email, existing.password)
                if not token:
                    continue
            from .accounts import set_account_token as _set
            matched = _set(settings, ident, token)
            if matched is not None:
                settings.save()
                _ok(f"Token updated for {matched.identifier} (saved)")
            else:
                _err(f"Not found: {ident}")
            continue
        if choice == "d":
            _clear_screen()
            print()
            try:
                ident = _ask("Identifier to DELETE (email or number)", default="")
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not ident:
                _info("Cancelled (nothing deleted).")
                continue
            # Allow number for delete, but ONLY inside explicit d flow.
            rows_now = pool.status()
            if ident.isdigit():
                idx = int(ident) - 1
                if 0 <= idx < len(rows_now):
                    ident = rows_now[idx]["identifier"]
                else:
                    _err("Out of range, nothing deleted.")
                    continue
            try:
                confirm = input(f"Delete '{ident}'? Type y to confirm: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            if confirm not in ("y", "yes"):
                _info("Cancelled (nothing deleted).")
                continue
            if not pool.remove(ident):
                _err(f"Not found: {ident} (nothing deleted)")
                continue
            from .accounts import remove_account as _remove
            _remove(settings, ident)
            settings.save()
            _ok(f"Removed {ident} (saved)")
            continue
        # Number = SELECT, never delete.
        try:
            idx = int(choice) - 1
        except ValueError:
            _err("Unknown option. Number selects, d deletes.")
            continue
        _clear_screen()
        rows_now = pool.status()
        if 0 <= idx < len(rows_now):
            ident = rows_now[idx]["identifier"]
            if pool.set_current(ident):
                settings.active_account = pool.current
                settings.save()
                _ok(f"Current -> [{ident}] (saved). Failover to others on cooldown.")
            else:
                _err(f"Select failed: {ident}")
        else:
            _err("Out of range.")


def _preflight_reason(settings: Settings) -> str:
    """Block Start when serving would fail fast. "" means ready.

    Empty pool -> 503 on first request; accounts without any token ->
    401/failover on first request. Caller jumps to the Accounts menu so
    the fix is one step away. Scoped to the active provider(s)' accounts.
    """
    try:
        from .core import pools as _pools

        _ids = _pools.active_provider_ids(settings)
        _accts = [a for _pid in _ids for a in _pools.accounts_for_provider(settings, _pid)]
    except Exception:
        _accts = settings.accounts
    if not _accts:
        return "No accounts yet. Add one first."
    for a in _accts:
        if (a.token or "").strip():
            return ""
    return "Accounts have no token. Add one (y = auto-token) first."


def _refresh_from_disk(settings: Settings) -> None:
    """Pick up account/config changes made via API while serving."""
    try:
        from .config import load_settings as _reload
        if settings.config_path:
            _fresh = _reload(settings.config_path)
            settings.accounts = _fresh.accounts
            settings.active_account = _fresh.active_account
    except Exception:
        pass


def main_menu(settings: Settings, serve_fn: Callable[[Settings], int],
              chat_fn: Callable[[Settings], int] | None = None,
              multi_fn: Callable[[Settings], int] | None = None,
              dry_fn: Callable[[Settings], int] | None = None) -> int:
    """Blocking main page. Server stop returns to menu (0 = quit)."""
    _clear_screen()
    accounts_num = "5" if chat_fn is not None else "3"
    dry_num = "6" if chat_fn is not None else "4"
    while True:
        _header("deepseaport")
        _info(f"Config   : {settings.config_path or '(unsaved)'}")
        _info(f"Host     : {'0.0.0.0 (all interfaces)' if settings.listen else '127.0.0.1 (localhost only)'}")
        _info(f"Port     : {settings.port}")
        _info(f"Reply    : {settings.stream_mode}  |  Tools: {_on_off(settings.enable_tools)}"
              f"  |  Failover: {_on_off(getattr(settings, 'use_multiple_accounts', True))}")
        cur = (settings.active_account or "").strip() or "ALL"
        _info(f"Accounts : {len(settings.accounts)}  |  Current: [{cur}]")
        try:
            from .core import pools as _pools

            _active = list(_pools.active_provider_ids(settings))
            _display = (" + ".join(_active) + " (multi)") if len(_active) > 1 else (
                _active[0] if _active else "deepseek")
        except Exception:
            _display = (getattr(settings, "provider", "") or "deepseek")
        _info(f"Provider  : {_display}")
        has_token = any((getattr(a, "token", "") or "").strip()
                        for a in settings.accounts)
        if not settings.accounts:
            _info(f"Setup    : no accounts yet -> Accounts ({accounts_num}) to add one")
        elif not has_token:
            _info(f"Setup    : no account token -> Accounts ({accounts_num}) to refresh one")
        print()
        print("  1. Start server")
        if chat_fn is not None:
            print("  2. Start server with chat")
            print("  3. Run multiple servers")
            print("  4. Settings")
            print("  5. Accounts")
            print("  6. Dry run (echo prompt, no DeepSeek)")
        else:
            print("  2. Settings")
            print("  3. Accounts")
            print("  4. Dry run (echo prompt, no DeepSeek)")
        print("  q. Quit")
        print()
        try:
            choice = input("Select: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if choice == "1":
            reason = _preflight_reason(settings)
            if reason:
                print()
                _err(reason)
                if "token" in reason.lower():
                    from .accounts import TOKEN_HELP as _TH
                    print()
                    print(_TH)
                accounts_menu(settings)
                continue
            print()
            _info("Server running. Press Ctrl+C or Esc to stop and return to menu.")
            print()
            try:
                serve_fn(settings)
            except (EOFError, KeyboardInterrupt):
                print()
                _info("Server stopped, back to menu.")
            _refresh_from_disk(settings)
            continue
        if chat_fn is not None and choice == "2":
            reason = _preflight_reason(settings)
            if reason:
                print()
                _err(reason)
                if "token" in reason.lower():
                    from .accounts import TOKEN_HELP as _TH
                    print()
                    print(_TH)
                accounts_menu(settings)
                continue
            try:
                chat_fn(settings)
            except (EOFError, KeyboardInterrupt):
                print()
                _info("Chat stopped, back to menu.")
            _refresh_from_disk(settings)
            continue
        if chat_fn is not None and choice == "3":
            reason = _preflight_reason(settings)
            if reason:
                print()
                _err(reason)
                if "token" in reason.lower():
                    from .accounts import TOKEN_HELP as _TH
                    print()
                    print(_TH)
                accounts_menu(settings)
                continue
            try:
                if multi_fn is not None:
                    multi_fn(settings)
                else:
                    from .chat_ui import run_multi_chat_servers as _multi
                    _multi(settings)
            except (EOFError, KeyboardInterrupt):
                print()
                _info("Multi-server stopped, back to menu.")
            _refresh_from_disk(settings)
            continue
        if choice == ("4" if chat_fn is not None else "2"):
            settings_menu(settings)
            _clear_screen()
            continue
        if choice == ("5" if chat_fn is not None else "3"):
            accounts_menu(settings)
            _clear_screen()
            continue
        if choice == dry_num or choice in ("dry", "dry-run", "dryrun"):
            # No preflight: dry run needs no accounts, launches no browser,
            # and never calls DeepSeek — it echoes the formatted prompt.
            print()
            _info("Dry run. Press Ctrl+C or Esc to stop and return to menu.")
            print()
            try:
                if dry_fn is not None:
                    dry_fn(settings)
                else:
                    from .dry_run import run_dry_run as _dry
                    _dry(settings)
            except (EOFError, KeyboardInterrupt):
                print()
                _info("Dry run stopped, back to menu.")
            continue
        if choice in ("q", "quit", "0", "exit"):
            return 0
        _err("Unknown option: type 1, 2, 3, 4, 5, 6, or q." if chat_fn is not None
             else "Unknown option: type 1, 2, 3, 4, or q.")
