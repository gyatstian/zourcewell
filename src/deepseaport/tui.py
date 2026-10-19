"""Stdlib-only TUI: main page (Start server / Settings) + settings editor.

No third-party deps (Windows-safe: plain input(), no curses) so `serve`
works everywhere. All edits mutate the passed Settings in place and persist
via Settings.save().
"""

from __future__ import annotations

import json
import os
import sys
from typing import Callable

from .config import VALID_LOG_LEVELS, Settings, _normalize_keys
# Only _read_interactive is imported at module scope; the rest of tui_editor is
# an implementation detail of the editor itself and re-exporting it only
# invites callers to depend on internals (tests pin tui._read_interactive).
from . import tui_editor as _tui_editor
from .tui_editor import _read_interactive


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


def _stdout_is_tty() -> bool:
    """False for pipes/logs/tests; the guard every screen draw uses."""
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def _clear_screen() -> None:
    """Clear screen on menu enter. No-op when not a tty (tests/pipes)."""
    if not _stdout_is_tty():
        return
    try:
        # Local import on purpose: tui_editor is imported at module scope for
        # `_read_interactive` only (see the top of this file), so _ensure_vt
        # must stay an in-function import rather than a tui attribute.
        from .tui_editor import _ensure_vt

        _ensure_vt()
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


def _term_size() -> tuple[int, int]:
    """(columns, rows) of the console, (80, 24) when it cannot be asked.

    ``shutil.get_terminal_size`` follows COLUMNS/LINES, then the real console,
    then its own (80, 24) fallback -- the same source tui_editor._editor_width
    reads, and 80/24 are the Windows console defaults the table widths and the
    menu compaction below are budgeted against. Piped output (tests, logs) also
    lands on the fallback, so those draws stay deterministic. No try/except is
    needed around this: get_terminal_size substitutes its own fallback and
    always returns int fields.
    """
    import shutil

    size = shutil.get_terminal_size()
    return (int(size.columns) or 80, int(size.lines) or 24)


def _longest_shared_prefix(idents: list[str]) -> int:
    """Longest prefix any two identifiers share; 0 with fewer than two rows."""
    worst = 0
    for i, a in enumerate(idents):
        for b in idents[i + 1:]:
            k = 0
            while k < min(len(a), len(b)) and a[k] == b[k]:
                k += 1
            worst = max(worst, k)
    return worst


def _elide_ids(idents: list[str], width: int) -> list[str]:
    """Shorten each identifier to ``width`` columns, head-anchored on its
    first difference from the others.

    A fixed head/tail cut makes two long addresses that differ in the middle
    render identically, and two identical rows are worse than a wrapped one:
    nothing left on the row can tell the accounts apart. So every row keeps at
    least the first column where it differs from every other row, and spends
    its tail on the domain only once that head is long enough. The caller sizes
    ``width`` so this head fits (see the table's ``need``).

    The marker is ASCII "..." -- what this table used before -- and unlike
    U+2026 it cannot blow up on a legacy console codepage mid-draw.
    """
    out: list[str] = []
    for i, ident in enumerate(idents):
        ident = str(ident or "")
        if len(ident) <= width:
            out.append(ident)
            continue
        need = 1
        for j, other in enumerate(idents):
            if i == j:
                continue
            other = str(other or "")
            k = 0
            while k < min(len(ident), len(other)) and ident[k] == other[k]:
                k += 1
            need = max(need, k + 1)
        keep = width - 3                      # room for the marker
        head = min(keep, max(keep // 2 + keep % 2, need))
        out.append(ident[:head] + "..." + ident[len(ident) - (keep - head):])
    return out


# Measured height of the grouped settings screen in settings_menu: header(3)
# + 7 sections x 2 + 17 items + 2 blanks + Back + the Select: prompt row = 38.
# It is the threshold below which the screen is redrawn without its section
# headers and blank lines, so the prompt stays on a short console. Re-measure
# when an item or a section is added or removed.
_GROUPED_SETTINGS_ROWS = 38


def _run_submenu(label: str, fn: Callable[[Settings], None],
                 settings: Settings) -> None:
    """Run a submenu; an unexpected failure must not end the process.

    Same guard style the serve branches use. EOFError/KeyboardInterrupt (how
    the menus are left) and SystemExit are re-raised on purpose -- swallowing
    them would turn Ctrl+C into "back to the main menu" and a serve-side
    SystemExit into silence.
    """
    try:
        fn(settings)
    except (EOFError, KeyboardInterrupt):
        raise
    except Exception as exc:
        _err(f"{label} failed: {type(exc).__name__}: {exc}")
        _info("Back to the main menu.")


def _ask_raw(prompt: str) -> str:
    """input() with no strip() and no default suffix.

    Secret-input companion to _ask: stripping " pass word " to "pass word"
    silently corrupts the value that gets saved and re-sent to the login API.
    """
    try:
        return input(f"{prompt}: ")
    except (EOFError, KeyboardInterrupt):
        print()
        raise


_LOGIN_BANNERS = {
    # Byte-identical to the pre-provider-split wording.
    "deepseek": "Running Obscura login (browser, ~30s)...",
}


def _login_banner(pid: str) -> str:
    """One-line "what is about to happen" for a provider login.

    A mapping rather than an if-chain, because the browser-based providers are
    not an exception to the rule any more: DeepSeek drives a real Obscura
    session, so "one browser-based provider" was never a category with a
    single member.
    A new one is one line here instead of another branch.

    The generic fallback stays "direct API signin, ~5s" and no browser: that
    was the failure being fixed. Every non-browser provider's login is a POST
    that returns in seconds, so promising a window sent the user watching the
    terminal for ~30s for a window that never opened.
    """
    key = (str(pid or "").strip().lower() or "deepseek")
    banner = _LOGIN_BANNERS.get(key)
    if banner is not None:
        return banner
    from .accounts import provider_label

    return f"Running {provider_label(key)} login (direct API signin, ~5s)..."


def _login_token_expiry(token: str, provider: str | None) -> float:
    """JWT ``exp`` of a login token for a non-DeepSeek provider, else 0.0.

    Mirrors cli.cmd_login: providers.qwen.auth.login_with_expiry captures the
    signin expiry, but the registry ``login()`` seam returns the legacy 3-tuple
    and discards it, so the expiry is recovered from the token afterwards.
    jwt_exp is the tree's only JWT ``exp`` reader; 0.0 = unknown, never fatal.
    Without this a Qwen account added by the TUI kept ``token_expires_at`` 0.0,
    so the documented proactive re-signin could never fire for it.
    """
    key = str(provider or "").strip().lower() or "deepseek"
    if not token or key == "deepseek":
        return 0.0
    try:
        from .providers.qwen.auth import jwt_exp

        return float(jwt_exp(token) or 0.0)
    except Exception:
        return 0.0


def _provider_scope(settings: Settings) -> tuple[list[str], bool, str]:
    """The one provider-scope lookup: (active_ids, is_multi, single).

    Five screens need "which providers is this session scoped to", and each
    used to spell it out with its own copy of the same import, the same
    len() > 1 test and the same inline "deepseek" fallback. They drifted. Every
    caller goes through here now.

    `single` is always normalized (str/strip/lower, never empty), so callers do
    not have to repeat that either. `core.pools` is the source of truth; when it
    cannot be imported the configured selection stands in for it.
    """
    try:
        from .core import pools as _pools

        active = [str(p) for p in _pools.active_provider_ids(settings)]
        single = _pools.selected_provider(settings)
    except Exception:
        single = None
    single = (str(single or getattr(settings, "provider", "") or "deepseek")
              .strip().lower() or "deepseek")
    if not active:
        active = [single]
    return active, len(active) > 1, single


def _provider_display(settings: Settings) -> str:
    """The active provider(s) as ONE string: the main menu's header.

    It used to back the main-menu "Provider:" line and settings item 19 at
    once, and each spelled the fact its own way -- with
    providers=["deepseek", "qwen"] one screen read single-provider while
    another read "deepseek + qwen (multi)". _provider_scope is the only source
    now, and this is its only reader.
    """
    active, is_multi, _single = _provider_scope(settings)
    if is_multi:
        return " + ".join(active) + " (multi)"
    return active[0]


def _active_providers_need_accounts(settings: Settings) -> bool:
    """True when some active provider really needs an account/token.

    A thin alias over ``core.pools.any_provider_needs_accounts`` so every hint
    in this file asks one question of one implementation. Kept as a named local
    because it is read at five call sites and the intent ("may I tell this user
    to go add a credential?") is clearer than the pool-module name at each of
    them.
    """
    from .core import pools as _pools

    return _pools.any_provider_needs_accounts(settings)


def _ask_password(prompt: str = "Password (Enter=skip)") -> str:
    """Hidden no-echo prompt, kept for ``_ask_secret`` (a bearer token).

    Account passwords go through plain visible ``_ask_raw`` instead: this is a
    local app and hiding the password only made it unverifiable while typing.
    Falls back to visible _ask_raw when getpass cannot control the terminal
    (e.g. no tty in tests) — getpass itself falls back to input() there,
    so mocked builtins.input still works. On that path the secret IS echoed
    (say so, instead of silently typing a token into a shared screen or
    log) and spaces are preserved; the old _ask fallback stripped them, so
    " pass word " was silently saved as "pass word".
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
            _info("(no echo control available: the value below IS echoed)")
            return _ask_raw(prompt)
    except (EOFError, KeyboardInterrupt):
        print()
        raise


def _ask_secret(prompt: str) -> str:
    """Hidden prompt for a bearer token.

    A pasted userToken is a 64-char credential: through ``_ask`` (plain
    input()) it lands in the console scrollback, the conhost buffer, any
    screen share and Windows Terminal's scrollback, while the password two
    fields earlier was typed blind. Same secret, same treatment.

    The echo-control decision is made BEFORE calling getpass, which is the one
    difference from _ask_password: with no console on stdin there is nothing to
    turn the echo off, and getpass gets there via win_getpass ->
    fallback_getpass -> sys.stdin.readline() (an exception here, or a read
    from whatever stdin happens to be). Deciding up front keeps the visible
    fallback on ``input()`` -- the path piped/automated callers drive -- and
    announces the echo exactly as _ask_password does when it falls back.
    """
    interactive = _tui_editor._stdin_is_tty()
    if interactive:
        return _ask_password(prompt)
    _info("(no echo control available: the value below IS echoed)")
    return _ask_raw(prompt)


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
    # Was the port the user typed actually free? _ensure_free_port returns the
    # port unchanged both when it is free and when the busy-port offer was
    # DECLINED (server_runner._ensure_free_port:238), so the return value alone
    # cannot tell "saved a good port" from "saved a port we just proved is
    # taken". Ask the socket, the same question _ensure_free_port asks.
    try:
        from .cli import _is_port_free

        busy = not _is_port_free(host, candidate)
    except Exception:
        busy = False
    try:
        if busy:
            picked = _ensure_free_port(settings, host, candidate)
        else:
            picked = candidate
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if int(picked) == candidate and busy:
        # Declined ("I'll handle it"), or no free port anywhere nearby. The
        # old "persist either way" wrote the busy port to config.json and
        # printed "ok: Port -> 54937 (saved)" -- the next Start then failed to
        # bind on a port the app had just certified as occupied.
        _info(f"Port {candidate} is in use; kept the previous port "
              f"({settings.port}). Free it, or pick another port.")
        return
    # _ensure_free_port saves the winner when it auto-picks; when the candidate
    # was free it returns it unsaved -- persist that case here.
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
            # Same normalizer load_settings applies, applied BEFORE the
            # emptiness/dupe checks: a stored key must equal the key
            # _check_auth matches. Untrimmed it 401s every client for the rest
            # of the session with no diagnosable cause, and the next start
            # strips it - silently changing the key and invalidating clients
            # already configured with it. A blank result is a cancel, not a
            # key: stored, it is an auth-bypass entry once reloaded.
            candidate = _normalize_keys([raw])
            if not candidate:
                _err("Blank key (whitespace only) - cancelled.")
                continue
            key = candidate[0]
            if key in keys:
                _err("Key already present, cancelled.")
                continue
            keys.append(key)
            settings.keys = _normalize_keys(keys)
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
                # Same writer as the add path: the assignment is the one that
                # persists, so it normalizes even though a delete cannot
                # introduce an untrimmed key.
                settings.keys = _normalize_keys(keys)
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
    try:
        p = Path(ans.strip()).expanduser()
    except (RuntimeError, OSError) as exc:
        # expanduser() raises RuntimeError("Could not determine home
        # directory.") when USERPROFILE and HOMEDRIVE+HOMEPATH are all
        # missing (stripped service account, some CI runners). That escaped
        # Settings 12 and killed the TUI with a traceback; check the literal
        # path instead so the operator still gets a real answer.
        p = Path(ans.strip())
        _err(f"Could not expand '~' ({exc}); checking the path as typed.")
    if not p.exists():
        _err(f"Path not found: {p}. Not saved.")
        _info("TIP: place obscura.exe next to config.json, bin/, tools/, vendor/.")
        return
    if p.is_dir():
        # A pasted release folder: resolve to the exe inside it instead of
        # saving a directory (which fails at runtime as WinError 5).
        hit = None
        for name in ("obscura.exe", "obscura"):
            cand = p / name
            try:
                if cand.is_file():
                    hit = cand
                    break
            except Exception:
                continue
        if hit is None:
            _err(f"Directory has no obscura.exe in it: {p}. Not saved.")
            return
        p = hit
        _info(f"Resolved directory -> {p}.")
    settings.obscura_bin = str(p)
    settings.save()
    _ok(f"Obscura binary -> {settings.obscura_bin} (saved)")


def _select_provider(settings: Settings) -> bool:
    """Shared provider-selection: the serve startup gate and the main menu's
    "Go back" entry.

    Nothing is pre-chosen and no current mode is shown: the user answers this
    question explicitly every time. A successful pick saves and returns
    immediately (the caller redraws its menu). Only invalid input re-prompts.
    Never raises on bad input; EOF/KeyboardInterrupt cancels the screen.
    Returns False ONLY when the user chose to quit the serve.
    """
    from .accounts import provider_label
    from .core import registry as _reg

    # Provider alias/match is shared (cli.normalize_provider_id via
    # _normalize_provider_choice below); no local alias map needed.
    _clear_screen()
    while True:
        try:
            providers = _reg.registered()
        except Exception:
            providers = {}
        ids = list(providers.keys())
        _header("Choose provider")
        if not ids:
            _err("No providers registered.")
            print()
            print("  q. Quit")
            print()
            try:
                choice = input("Select: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return True
            if choice in ("q", "quit"):
                return False
            _err("Unknown option.")
            continue
        for i, pid in enumerate(ids, 1):
            # Label only: no model count (a catalogue read per row --
            # opencode's models() BOOTS its managed serve when nothing
            # answers) and no account/token tally (the start preflight names a
            # missing token when it actually blocks a start).
            print(f"  {i}. {provider_label(pid)}")
        print()
        print("  m. Use all")
        print("  q. Quit")
        print()
        try:
            choice = input("Select: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return True
        if choice in ("q", "quit"):
            return False
        if choice in ("m", "multi"):
            settings.providers = list(ids)
            settings.provider = ids[0] if ids else "deepseek"
            try:
                settings.save()
            except (EOFError, KeyboardInterrupt):
                print()
                return True
            except (RuntimeError, OSError) as exc:
                # Same treatment as _run_submenu / cli._save: save() refuses an
                # unparsable config and fails on an unwritable one, and this is
                # also the serve startup gate -- a traceback here killed the TUI.
                _err(f"Provider not saved: {type(exc).__name__}: {exc}")
                return True
            _ok(f"Provider mode -> multi ({', '.join(ids)}) (saved)")
            return True
        picked = ""
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(ids):
                picked = ids[idx]
        elif choice:
            # Shared alias+match (cli.normalize_provider_id via thin wrapper).
            picked = _normalize_provider_choice(choice, ids)
        if not picked:
            _err("Unknown option. Type a number, a provider id, m, or q.")
            continue
        settings.provider = picked
        settings.providers = []
        try:
            settings.save()
        except (EOFError, KeyboardInterrupt):
            print()
            return True
        except (RuntimeError, OSError) as exc:
            _err(f"Provider not saved: {type(exc).__name__}: {exc}")
            return True
        _ok(f"Provider -> {picked} (saved)")
        return True


def provider_menu(settings: Settings) -> bool:
    """First-run provider gate for `serve`. False = the user quit the serve."""
    return _select_provider(settings)


def _normalize_provider_choice(raw: str, valid_ids: list[str]) -> str:
    """Normalize typed provider to a valid id, resolving aliases.

    Thin adapter over cli.normalize_provider_id (single implementation).
    Returns "" when the choice does not match. Empty input returns "" here;
    callers apply the deepseek default.
    """
    # In-function import on purpose: cli and tui import each other lazily.
    from .cli import normalize_provider_id

    return normalize_provider_id(raw, valid_ids)


def _ask_account_provider(settings: Settings) -> str:
    """Provider id for a new account. Single mode returns it directly.

    Multi mode prompts with text choices (e.g. deepseek [ds] / qwen), defaulting
    to deepseek when it is active, else to the first active provider. Never
    raises; EOF defaults the same way.
    """
    active, multi, single = _provider_scope(settings)
    if not multi:
        # Single mode: no prompt, the id is INHERITED from the selection, which
        # load_settings does not registry-check - coerce it like the CLI add
        # path, or a junk DEEPSEAPORT_PROVIDER lands on every new account and
        # no pool, login or ban check can ever match it. An explicitly picked id
        # (multi mode below) is the operator's choice and stays verbatim.
        try:
            from .accounts import usable_provider as _usable

            return _usable(single)
        except Exception:
            return (single or "deepseek").strip().lower() or "deepseek"
    # The empty/EOF answer must land INSIDE the active set. It used to return
    # the literal "deepseek", so with providers=["qwen","opencode"] an Enter
    # filed the account under a provider that is not active -- no pool builds
    # from it and no request can serve it (the `login --provider glm` class).
    _default = "deepseek" if "deepseek" in active else (active[0] if active else "deepseek")
    # Multi: text choices only, deepseek shows its ds alias in brackets.
    parts: list[str] = []
    for pid in active:
        low = str(pid or "").strip().lower()
        if low == "deepseek":
            parts.append("deepseek [ds]")
        else:
            parts.append(low)
    prompt = f"Provider ({' / '.join(parts)})"
    while True:
        try:
            ans = _ask(prompt, default=_default)
        except (EOFError, KeyboardInterrupt):
            print()
            return _default
        if not ans.strip():
            return _default
        picked = _normalize_provider_choice(ans, active)
        if picked:
            return picked
        _err(f"Unknown provider '{ans}'. Choose: {', '.join(active)} (ds = deepseek).")


def _preview_append(value: str, width: int = 30) -> str:
    """One-line preview so multi-line text never breaks the menu layout.

    Elides with ASCII "..." like ``_elide_ids``: this used to carry a
    double-encoded U+2026 (three mojibake chars, C3 A2 E2 82 AC C2 A6) that
    rendered as garbage on screen, and ASCII is also what keeps the result
    inside a legacy console codepage mid-draw.
    """
    text = str(value or "")
    if not text.strip():
        return "(empty)"
    lines = text.strip("\r\n").splitlines() or [text]
    first = lines[0].strip()
    if len(first) > width:
        first = first[:width - 3] + "..."
    if len(lines) > 1:
        chars = len(text)
        return f"{first or '...'} ({len(lines)} lines, {chars} chars)"
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
    is_tty = _tui_editor._stdin_is_tty()
    print()
    if is_tty:
        if current.strip():
            _info(f"{label} ({len(current.splitlines())} lines, {len(current)} chars - loaded below, edit away).")
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
    # The 17 settings, declared ONCE. Both the draw below and the dispatch at
    # the bottom of the loop read this one table, so a row cannot drift away
    # from the key that runs it -- which is exactly what happened before, when
    # setting 2 read "WAF warmup at startup" to the toggle and "Warm up WAF at
    # startup" on screen. Numbers are stable API (tests drive "1".."17"), so an
    # edit here is an edit to both at once; never renumber to insert one.
    def _table() -> tuple[tuple[int, str, str, str, Callable[[], None]], ...]:
        obsc = (getattr(settings, "obscura_bin", "") or "").strip() or "(auto)"
        if len(obsc) > 30:
            obsc = "..." + obsc[-27:]
        nkeys = len(getattr(settings, "keys", []) or [])
        return (
            (1, "Request handling", "Tool calling",
             _on_off(settings.enable_tools),
             lambda: _toggle_bool(settings, "enable_tools", "Tool calling")),
            (2, "Request handling", "Warm up WAF at startup",
             _on_off(settings.warmup_on_startup),
             lambda: _toggle_bool(settings, "warmup_on_startup", "Warm up WAF at startup")),
            (3, "Request handling", "Delete session after reply",
             _on_off(settings.auto_delete_session),
             lambda: _toggle_bool(settings, "auto_delete_session", "Delete session after reply")),
            (4, "Request handling", "Retries per failed request",
             str(settings.max_retries),
             lambda: _edit_retries(settings)),
            (5, "Request handling", "Parallel setup requests",
             _on_off(settings.parallel_challenge_fetch),
             lambda: _toggle_bool(settings, "parallel_challenge_fetch", "Parallel setup requests")),
            (6, "Request handling", "Send dry run to frontend",
             _on_off(getattr(settings, "send_dry_run_to_frontend", True)),
             lambda: _toggle_bool(settings, "send_dry_run_to_frontend", "Send dry run to frontend")),
            (7, "Chat & logs", "Log level", str(settings.log_level),
             lambda: _edit_choice(settings, "log_level", "Log level", VALID_LOG_LEVELS)),
            (8, "Chat & logs", "Reply streaming", str(settings.stream_mode),
             lambda: _toggle_stream(settings)),
            (9, "Server access", "Allow LAN access",
             _on_off(settings.listen),
             lambda: _toggle_bool(settings, "listen", "Allow LAN access")),
            (10, "Server access", "Port", str(settings.port),
             lambda: _edit_port(settings)),
            (11, "Server access", "API keys",
             f"{nkeys} set" if nkeys else "none (open)",
             lambda: _edit_keys(settings)),
            (12, "Setup & defaults", "Obscura binary", obsc,
             lambda: _edit_obscura_bin(settings)),
            (13, "Accounts", "Account failover",
             _on_off(getattr(settings, "use_multiple_accounts", True)),
             lambda: _toggle_bool(settings, "use_multiple_accounts", "Account failover")),
            (14, "Prompt", "Enable append",
             _on_off(getattr(settings, "enable_append", True)),
             lambda: _toggle_bool(settings, "enable_append", "Enable append")),
            (15, "Prompt", "Append at top",
             _preview_append(getattr(settings, "append_top", "")),
             lambda: _edit_append_text(settings, "append_top", "Append at top")),
            (16, "Prompt", "Append at bottom",
             _preview_append(getattr(settings, "append_bottom", "")),
             lambda: _edit_append_text(settings, "append_bottom", "Append at bottom")),
            (17, "Tools", "More forgiving toolcalls",
             _on_off(getattr(settings, "forgiving_toolcalls", False)),
             lambda: _toggle_bool(settings, "forgiving_toolcalls", "More forgiving toolcalls")),
        )

    _by_number = {str(n): act for n, _sec, _lbl, _val, act in _table()}

    # Message from the previous rejected digit, redrawn above Select: -- the
    # loop clears before every redraw, so without this the reason a mistyped
    # digit was refused would be wiped by the very redraw that asks again.
    _pending = ""

    def _group(title: str) -> None:
        """Section header, dropped in the compact redraw (see compact below)."""
        if not compact:
            _section(title)

    while True:
        # The grouped screen below is 38 rows (see _GROUPED_SETTINGS_ROWS). On
        # a 24-row console that puts Select: below the fold on the FIRST draw,
        # and with no clear every rejected digit stacks another full copy
        # underneath. Compact form (title + the 17 items + Back) fits 24 rows
        # with the prompt visible. Only on a real console: piped output keeps
        # the grouped screen.
        _clear_screen()
        compact = _stdout_is_tty() and _term_size()[1] < _GROUPED_SETTINGS_ROWS
        if compact:
            _info(f"=== Settings ({settings.config_path or 'unsaved'}) ===")
        else:
            _header(f"Settings ({settings.config_path or 'unsaved'})")

        _section_name = None
        for num, section, label, value, _action in _table():
            if section != _section_name:
                _group(section)
                _section_name = section
            _setting_line(num, label, value)

        if not compact:
            print()
        print("  18. Back (q)")
        if _pending:
            _err(_pending)
            _pending = ""
        print()
        try:
            choice = input("Select: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if choice in ("18", "b", "back", "q", "quit", ""):
            return
        if choice not in _by_number:
            _pending = "Unknown option: type 1-17, or q."
            continue
        # Clear the menu before prompting so repeated edits do not stack
        # multiple copies of the settings screen in the terminal.
        _clear_screen()
        try:
            _by_number[choice]()
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

    def _row_cfg(row_index: int):
        """settings.accounts entry for a visible 1-based row number, else None.

        The row list is provider-scoped (_vis_idx), so the screen number is
        NOT a settings index. Two providers may also hold the same email, and
        every accounts helper (_matches, pool.remove, set_account_token)
        resolves by identifier and takes the FIRST hit -- so row 2 of
        [a@x.com deepseek, a@x.com qwen] addressed by email would delete the
        DeepSeek account while the user confirmed the Qwen one.
        """
        real = _vis_idx[row_index]
        if 0 <= real < len(settings.accounts):
            return settings.accounts[real]
        return None

    def _drop_cfg(cfg) -> bool:
        """Delete exactly this AccountConfig (identity, not identifier).

        Falls back to the identifier APIs only when no row was resolved, so
        typing a literal email/mobile/token keeps working. CURRENT is cleared
        only when no remaining account can satisfy it -- with a shared email
        the surviving provider must keep the selection.
        """
        from .accounts import _matches as _m

        dropped = False
        with pool._guard:
            for i, item in enumerate(pool._items):
                if item.cfg is cfg:
                    del pool._items[i]
                    dropped = True
                    break
        kept = [a for a in settings.accounts if a is not cfg]
        if len(kept) != len(settings.accounts):
            settings.accounts = kept
            dropped = True
        if not dropped:
            return False
        if settings.active_account and not any(
                _m(a, settings.active_account) for a in settings.accounts):
            settings.active_account = ""
        return True

    def _label_of(cfg) -> str:
        return f"{cfg.identifier} [{getattr(cfg, 'provider', '') or 'deepseek'}]"

    def _offer_auto_login(email: str, password: str, confirmed: bool = False,
                          provider: str | None = None
                          ) -> tuple[str, bool, float]:
        """Run the provider login inside the TUI. confirmed=True skips the Y/n prompt.

        Returns ``(token, banned, banned_until)`` and deliberately does NOT
        touch settings.accounts: the caller owns the concrete AccountConfig
        it is about to write (a fresh one on the add path, an existing one on
        the set-token path) and is the only place that can apply the ban
        state unambiguously. The old version re-scanned settings.accounts by
        email to set banned/banned_until, which matched NOTHING on the add
        path (the config is not appended yet) and matched the WRONG provider
        when two providers share an email.
        """
        from .accounts import token_help_for as _token_help_for
        from .accounts import provider_label
        # Shared membership rule with cli.cmd_login's --provider, imported here
        # because tui.py must not gain cli module attributes (tests/test_tui_gaps).
        from .cli import _unknown_provider_exit
        from .cli import provider_is_registered as _login_provider_known
        # Shared login-pid resolution (cli.resolve_login_provider_id): explicit
        # provider wins, else the selected single-mode provider, else deepseek.
        if provider:
            try:
                from .cli import resolve_login_provider_id as _resolve_pid
                _pid = _resolve_pid(settings, provider)
            except Exception:
                _pid = (str(provider or "deepseek").strip().lower()) or "deepseek"
        else:
            try:
                from .core import pools as _pools
                _pid = _pools.selected_provider(settings)
            except Exception:
                _pid = "deepseek"
            _pid = (str(_pid or "deepseek").strip().lower()) or "deepseek"
        _help = _token_help_for(_pid)
        # Prompt/login wording follows the resolved provider: "DeepSeek
        # password" while driving a Qwen signin is a copy-paste bug, and
        # "Running Obscura login (browser, ~30s)" is a lie for the providers
        # whose login is a direct API POST with no browser at all.
        _label = provider_label(_pid)
        # Same membership rule as cli.cmd_login's --provider, same reason: an id
        # the registry does not know used to fall through to the DeepSeek
        # Obscura driver, so the banner below promised a Qwen-shaped "direct API
        # signin, ~5s" while a ~30s DeepSeek browser opened, and the resulting
        # userToken was then stored under the unknown label -- an account no UI
        # lists and no request can use. The TUI cannot exit 2, so this is the
        # in-app equivalent already used everywhere else in this function: the
        # same refusal text, the token help, and the empty-token return the
        # caller reads as "no login happened". The menu stays alive.
        # deepseek is exempt: its login is the direct browser path, not a
        # registry entry point.
        if _pid != "deepseek" and not _login_provider_known(_pid):
            _unknown_provider_exit(_pid)
            print(_help)
            return "", False, 0.0
        if not email:
            return "", False, 0.0
        if not confirmed:
            try:
                ans = input(f"No token. Run auto-login via {_label} now? [Y/n]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return "", False, 0.0
            if ans not in ("", "y", "yes"):
                print()
                print(_help)
                return "", False, 0.0
        if not password:
            # Visible, like the add form's password field: hiding a local-app
            # password only made it unverifiable while typing, and the old
            # getpass had no fallback for a stdin it cannot read -- its
            # exception escaped the whole accounts menu instead of offering
            # TOKEN_HELP.
            password = _ask_raw(f"{_label} password (Enter=skip)")
        if not password:
            _err("Auto-login needs password. Cancelled.")
            print()
            print(_help)
            return "", False, 0.0

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

        _info(_login_banner(_pid))
        try:
            _reg_result = _registry_login()
            if _reg_result is not None:
                if len(_reg_result) == 3 and _reg_result[2] == "__handled__":
                    return "", False, 0.0
                token, login_banned, ban_detail = _reg_result
            else:
                from .providers.deepseek.auth import obscura_login_token as _login_token
                from .providers.deepseek.auth import unpack_login_result as _unpack_login
                token, login_banned, ban_detail = _unpack_login(
                    _login_token(email, password, settings))
        except Exception as exc:
            _err(f"Auto-login failed: {exc}")
            print()
            print(_help)
            return "", False, 0.0
        _until = 0.0
        if login_banned:
            try:
                from .providers.deepseek.auth import parse_ban_until
                _until = float(parse_ban_until(ban_detail) or 0.0)
            except Exception:
                _until = 0.0
            if token:
                # cli.cmd_login's refinement, same reason: the login page's
                # wording is one source for the expiry, the provider's own ban
                # probe (Provider.probe_ban, so a Qwen token is probed as Qwen)
                # is the better one when it answers. Best-effort, never fatal.
                try:
                    from .accounts import check_ban_for_token
                    if _pid != "deepseek":
                        _probed, _probe_until = check_ban_for_token(token, provider=_pid)
                    else:
                        _probed, _probe_until = check_ban_for_token(token)
                    _until = float(_probe_until or 0.0) or _until
                except Exception:
                    pass
                _err(f"Auto-login captured a token for {email}, "
                     "but the account is BANNED. Token saved; Account "
                     "list will show the ban.")
                if ban_detail:
                    print(f"Evidence: {ban_detail[:300]}")
                return token, True, _until
            _err(f"Auto-login refused for {email}: account BANNED per login page.")
            if ban_detail:
                print(f"Evidence: {ban_detail[:300]}")
            return "", False, 0.0
        if not token:
            if ban_detail:
                # A single ban-wording sighting does not set login_banned, but
                # the driver reports it here; surface it rather than returning
                # with nothing said.
                print(f"Evidence: {ban_detail[:300]}")
            return "", False, 0.0
        _ok(f"Auto-login ok for {email}")
        return token, False, 0.0

    _clear_screen()
    # Message from the last rejected action, redrawn above Select: (see the
    # settings loop): the screen is redrawn from scratch every iteration, so a
    # reason printed under the old copy is gone by the time the user could
    # read it.
    _pending = ""
    while True:
        pool = _pool()
        # Provider-scoped view: single-provider shows only its own accounts,
        # multi shows everything. Numbers below always refer to this view.
        _active_ids, _is_multi_v, _single_v = _provider_scope(settings)
        if _is_multi_v:
            _single_v = ""
        try:
            from .core.pools import _provider_of as _prov_of_v
            _all_provs = [_prov_of_v(a) for a in settings.accounts]
        except Exception:
            _all_provs = ["deepseek"] * len(getattr(settings, "accounts", []) or [])
        full_rows = pool.status()
        if _is_multi_v:
            _vis_idx = list(range(len(full_rows)))
            _visible_accounts = list(settings.accounts)
            _scope_label = "multi (" + " + ".join(_active_ids) + ")"
        else:
            _vis_idx = [i for i, _p in enumerate(_all_provs) if _p == _single_v]
            _visible_accounts = [settings.accounts[i] for i in _vis_idx
                                 if 0 <= i < len(settings.accounts)]
            _scope_label = _single_v
        rows = [full_rows[i] for i in _vis_idx if 0 <= i < len(full_rows)]
        _row_provs = [_all_provs[i] if 0 <= i < len(_all_provs) else "deepseek"
                      for i in _vis_idx]
        # Hide a CURRENT that belongs to another provider (stale after switch).
        try:
            from .accounts import _matches as _m_cur
            _cur_raw = pool.current or ""
            if (not _is_multi_v and _cur_raw
                    and not any(_m_cur(a, _cur_raw) for a in _visible_accounts)):
                current = ""
            else:
                current = _cur_raw
        except Exception:
            current = pool.current or ""
        # Best-effort ban labels without blocking menu render: instant TTL
        # cache read + background refresh (network never on the draw path).
        # Skips missing/short (test) tokens without network; never raises.
        # Scoped to the visible accounts so a qwen view never probes deepseek.
        try:
            from .accounts import (ban_label_for as _ban_lookup,
                                   get_cached_ban_labels as _ban_cached,
                                   refresh_ban_labels_background as _ban_refresh)
            from .providers.deepseek import protocol as _P
            ban_map = _ban_cached(_visible_accounts) or {}
            try:
                _ban_refresh(_visible_accounts)
            except Exception:
                pass
        except Exception:
            ban_map = {}
            _ban_lookup = None  # type: ignore
            _P = None  # type: ignore
        _header(f"Accounts ({len(rows)})")
        _info(f"Current: [{current or 'ALL (auto-failover)'}]")
        _info(f"Provider scope: {_scope_label}")
        print()
        if rows:
            id_width = max([len("Identifier")] + [len(r["identifier"]) for r in rows])
            id_width = min(id_width, 32)
            # Render every row first, then size the one elastic column. 80 is
            # the Windows console default and the widest thing on screen is
            # fixed (mark + ban label + (NO TOKEN) + provider tag), so the
            # identifier column is what has to give -- but not below 16,
            # because two accounts that render the same are worse than one
            # long row that wraps.
            #
            # The [provider] tag is dropped in single-provider mode, where the
            # "Provider scope:" line above already names it and the tag is
            # pure repetition. In multi mode it is the only thing telling two
            # same-email accounts apart, so it is never dropped.
            prepared = []
            for i, row in enumerate(rows, 1):
                mark = "*" if row.get("current") else ""
                has = _has_token(settings, row["identifier"])
                ident = row["identifier"]
                busy = "busy" if row.get("busy") else "-"
                # Red ban suffix: "(BANNED: 16 September)" from mute_until.
                # Kept uncolored too: the column budget below is measured in
                # VISIBLE columns, and the ANSI escapes take none of them.
                ban_plain, ban_suffix = "", ""
                try:
                    _found, _until = (_ban_lookup(ban_map, row["identifier"])
                                      if _ban_lookup else (False, None))
                    if _found and _P is not None:
                        ban_plain = " " + _P.format_ban_label(_until)
                        ban_suffix = " " + _c(_P.format_ban_label(_until), "31")
                    elif _found:
                        ban_plain = ban_suffix = " (BANNED)"
                except Exception:
                    ban_plain = ban_suffix = ""
                # "(NO TOKEN)" shares the tail with "(BANNED)"; both marks can
                # be on one account at once, so they append, never replace.
                if has:
                    tok_plain = tok_suffix = ""
                else:
                    tok_plain = " (NO TOKEN)"
                    tok_suffix = " " + _c("(NO TOKEN)", "31")
                _prov = _row_provs[i - 1] if 0 < i <= len(_row_provs) else "deepseek"
                tag = f" [{_prov}]" if _is_multi_v else ""
                prepared.append({
                    "n": i, "ident": ident, "busy": busy,
                    "tag": tag, "tail": mark + ban_suffix + tok_suffix,
                    # Worst-case row width outside the identifier column.
                    "w": len(mark) + len(ban_plain) + len(tok_plain) + len(tag),
                })
            # Everything the row format prints except the identifier and the
            # tail: 4 indent + 2-wide row number + the gap and the 4-wide Busy
            # cell around it.
            _fixed = 4 + 2 + 1 + 2 + 4 + 1
            widest = max(p["w"] for p in prepared)
            budget = _term_size()[0] - _fixed - widest
            # Wide enough to tell every row from every other: one column past
            # the longest prefix two identifiers share, plus the marker. Below
            # that two rows are identical and the table says nothing. It only
            # bites for long AND similar addresses (a plus-addressed pair on
            # one domain), and it is capped by the 32-column preference.
            need = _longest_shared_prefix([p["ident"] for p in prepared]) + 4
            # The floor is 16: below that even distinct short addresses get
            # mangled, and a wrapped row is recoverable while that is not. A
            # 20-column ban date plus a multi-provider tag can still push such
            # a row a few columns past the width on purpose -- the date and
            # the tag are the information here, the width is only the budget.
            id_width = max(16, min(id_width, budget), min(need, 32))
            idents = _elide_ids([p["ident"] for p in prepared], id_width)
            print(f"    {'#':<2} {'Identifier':<{id_width}}  {'Busy':<4} {'Cur'}")
            print(f"    {'--':<2} {'-' * id_width}  {'----':<4} {'---'}")
            for p, ident in zip(prepared, idents):
                print(f"    {p['n']:<2} {ident:<{id_width}}  {p['busy']:<4} "
                      f"{p['tail']}{p['tag']}")
        elif _active_providers_need_accounts(settings):
            _info("(no accounts yet -- use 'a' to add one)")
        else:
            # Accountless provider (opencode CLI): there is no pool to fill and
            # its login() raises, so "add one" is advice that cannot be taken.
            _info("(no accounts -- this provider needs none; Start server works)")
        if _pending:
            _err(_pending)
            _pending = ""
        print()
        print("  a. Add account")
        print("  t. Set/refresh token")
        print("  d. Delete mode")
        print("  c. Clear selection (use ALL)")
        print("  h. How to get token")
        print()
        print("  q. Back")
        print()
        try:
            choice = input("Select: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if choice in ("q", "quit", ""):
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
            print()
            # Without the wait, the loop redraws the menu straight over the
            # help: on a 24-row console only the redrawn menu stayed visible.
            try:
                input("Press Enter to return: ")
            except (EOFError, KeyboardInterrupt):
                print()
                return
            _clear_screen()
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
            # Multi mode: ask provider first (text only, ds = deepseek).
            # Single mode: no prompt, binds to the selected provider.
            try:
                new_provider = _ask_account_provider(settings)
            except (EOFError, KeyboardInterrupt):
                print()
                return
            try:
                email = _ask("Email (Enter=skip)", default="")
                # Visible on purpose: a local app, there is nobody to hide the
                # account password from.
                password = _ask_raw("Password (Enter=skip)")
                # Hidden unlike the password above: a pasted userToken is a
                # bearer credential and would sit in the scrollback.
                raw_token = _ask_secret("Token (y = auto-token)")
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if raw_token.strip().lower() in ("y", "yes"):
                token, login_banned, banned_until = _offer_auto_login(
                    email, password, confirmed=True, provider=new_provider)
            else:
                token, login_banned, banned_until = extract_token(raw_token), False, 0.0
            # The ban state has to be built into the config that is actually
            # appended: _offer_auto_login used to set banned/banned_until by
            # scanning settings.accounts for the email, which on this path
            # matched nothing (the account does not exist yet), so the list
            # said "Token saved; Account list will show the ban" while the
            # saved account kept banned=False.
            cfg = AccountConfig(email=email, password=password, token=token,
                                provider=new_provider, banned=login_banned,
                                banned_until=banned_until,
                                token_expires_at=_login_token_expiry(token, new_provider))
            try:
                from .accounts import add_account as _add
                _add(settings, pool, cfg)
            except ValueError as exc:
                _err(f"Add failed: {exc}")
                continue
            settings.save()
            _ok(f"Added {cfg.identifier} [{cfg.provider}] (saved)")
            if not cfg.token:
                _err("No token. Password-only fails.")
                print()
                print(TOKEN_HELP)
            continue
        if choice == "t":
            _clear_screen()
            print()
            try:
                ident = _ask("Account email or row number", default="")
                # Hidden for the same reason as the add form's token field.
                token = _ask_secret("New token value (Enter=skip = auto-login offer)")
            except (EOFError, KeyboardInterrupt):
                print()
                return
            token = extract_token(token)
            if not ident:
                _info("Cancelled.")
                continue
            # A row number addresses the concrete account; a typed identifier
            # keeps the by-identifier path (and its first-match caveat).
            target = None
            if ident.isdigit():
                idx = int(ident) - 1
                if not 0 <= idx < len(rows):
                    _pending = "Out of range, nothing changed."
                    continue
                target = _row_cfg(idx)
                ident = rows[idx]["identifier"]
            if not token:
                from .accounts import _matches as _m
                # Scoped like the `d` branch: the visible rows are the scope,
                # and the unscoped first match offered the OTHER provider's
                # credentials for auto-login -- then wrote the token it
                # produced onto the scoped record.
                existing = (target if target is not None else
                            next((a for a in _visible_accounts
                                  if _m(a, ident)), None))
                if existing is None:
                    _pending = f"Not found: {ident}"
                    continue
                try:
                    _exist_prov = (getattr(existing, "provider", "deepseek")
                                   or "deepseek")
                except Exception:
                    _exist_prov = "deepseek"
                token, login_banned, banned_until = _offer_auto_login(
                    existing.email, existing.password, provider=_exist_prov)
                if not token:
                    continue
                if login_banned:
                    # Apply to THIS object: a by-email rescan would flag the
                    # same address under the other provider.
                    existing.banned = True
                    existing.banned_until = banned_until
            if target is not None:
                target.token = token  # __setattr__ re-normalizes the value
                matched = target
            else:
                from .accounts import set_account_token as _set
                # Provider-SCOPED like the routes / the delete path: the same
                # email is one record per provider, and the unscoped first
                # match wrote the token onto the OTHER provider's record.
                # Multi scope ("") keeps the historical first match.
                matched = _set(settings, ident, token, provider=_single_v)
            if matched is not None:
                _exp = _login_token_expiry(token, getattr(matched, "provider", ""))
                if _exp:
                    # cli parity (_upsert_account writes a known expiry only):
                    # a token that says nothing about its expiry must not erase
                    # one already persisted.
                    matched.token_expires_at = _exp
                settings.save()
                _ok(f"Token updated for {_label_of(matched)} (saved)")
                if getattr(matched, "banned", False):
                    # CLI parity (cmd_login's banned branch): name the persisted
                    # ban instead of a bare "(saved)", and say what clears it.
                    # Never auto-clear the flag here -- the REST/CLI unblock owns
                    # that decision.
                    _err("Account is BANNED; token kept. Clear with: "
                         "accounts unblock")
            else:
                _pending = f"Not found: {ident}"
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
            # Numbers refer to the provider-scoped view above.
            target = None
            if ident.isdigit():
                idx = int(ident) - 1
                if not 0 <= idx < len(rows):
                    _pending = "Out of range, nothing deleted."
                    continue
                target = _row_cfg(idx)
                ident = rows[idx]["identifier"]
            # Name the provider in the prompt, outside the quotes and exactly
            # as the row renders it: with a shared email the identifier alone
            # cannot tell the user which account is about to be destroyed.
            if target is not None:
                name = target.identifier
                prov = getattr(target, "provider", "") or "deepseek"
            else:
                name, prov = ident, ""
            try:
                confirm = input(f"Delete '{name}'"
                                + (f" [{prov}]" if prov else "")
                                + "? Type y to confirm: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            if confirm not in ("y", "yes"):
                _info("Cancelled (nothing deleted).")
                continue
            shown = f"{name} [{prov}]" if prov else name
            if target is not None:
                if not _drop_cfg(target):
                    _pending = f"Not found: {shown} (nothing deleted)"
                    continue
            else:
                # Typed identifier: legacy by-identifier path, kept working --
                # but provider-SCOPED like the routes (server_routes
                # .delete_account): one email is one record per provider, and
                # the unscoped first match removed the OTHER provider's record
                # while the one this screen shows survived. Multi scope ("")
                # keeps the historical first match; a row number disambiguates.
                from .accounts import remove_account as _remove
                if not _remove(settings, ident, provider=_single_v):
                    _pending = f"Not found: {ident} (nothing deleted)"
                    continue
            settings.save()
            _ok(f"Removed {shown} (saved)")
            continue
        # Number = SELECT, never delete.
        try:
            idx = int(choice) - 1
        except ValueError:
            _pending = "Unknown option. Number selects, d deletes."
            continue
        _clear_screen()
        rows_now = rows
        if 0 <= idx < len(rows_now):
            ident = rows_now[idx]["identifier"]
            if pool.set_current(ident):
                settings.active_account = pool.current
                settings.save()
                _ok(f"Current -> [{ident}] (saved). Failover to others on cooldown.")
            else:
                _pending = f"Select failed: {ident}"
        else:
            _pending = "Out of range."


def _preflight_reason(settings: Settings) -> str:
    """Block Start when serving would fail fast. "" means ready.

    Empty pool -> 503 on first request; accounts without any token ->
    401/failover on first request. Caller jumps to the Accounts menu so
    the fix is one step away. Scoped to the active provider(s)' accounts.
    """
    try:
        from .core import pools as _pools

        if _pools.only_accountless_providers(settings):
            # Only local (accountless) providers active (opencode CLI):
            # nothing to provision, serve can start.
            return ""
        # Scope to the account-needing providers only. `providers` is empty
        # when the registry could not answer, in which case every id counts.
        _accts = [a for _pid in (_pools.providers_requiring_accounts(settings)
                                 or _pools.active_provider_ids(settings))
                  for a in _pools.accounts_for_provider(settings, _pid)]
    except Exception:
        _accts = settings.accounts
    if not _accts:
        return "No accounts yet. Add one first."
    for a in _accts:
        if (a.token or "").strip():
            return ""
    return "Accounts have no token. Add one (y = auto-token) first."


def _refresh_from_disk(settings: Settings) -> None:
    """Pick up account/config changes made via API while serving. Non-destructive.

    load_settings() swallows a JSON error and returns accounts=[], so trusting
    it blindly is data loss: a transiently truncated config.json (the Windows
    non-atomic save fallback, an editor mid-write, a crashed writer) would
    replace the in-memory accounts with [] and the very next settings.save()
    from any toggle / key add / port change would persist "accounts": [],
    destroying every email, password and token. So validate the file parses as
    a JSON *object* BEFORE handing it to load_settings(); a missing file is a
    no-op and a broken one keeps memory and says so loudly.
    """
    path = getattr(settings, "config_path", "")
    if not path:
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        return  # Never saved yet (or deleted): nothing to pick up.
    except (OSError, ValueError) as exc:
        # ValueError covers json.JSONDecodeError (truncated/half-written file).
        _err(f"Config not valid JSON ({exc}); keeping in-memory settings "
             f"and NOT overwriting {path}. Fix or delete the file by hand.")
        return
    if not isinstance(raw, dict):
        _err(f"Config is {type(raw).__name__}, not a JSON object; keeping "
             f"in-memory settings and NOT overwriting {path}.")
        return
    try:
        from .config import load_settings as _reload

        fresh = _reload(path)
    except Exception as exc:
        _err(f"Config reload failed ({exc}); keeping in-memory settings.")
        return
    settings.accounts = fresh.accounts
    settings.active_account = fresh.active_account


def _preflight_gate(settings: Settings) -> bool:
    """True when serving may start; False when the user was sent to Accounts.

    Single home for the gate that was copy-pasted into three main_menu
    branches: print the reason, print TOKEN_HELP when it is about tokens,
    then jump straight into accounts_menu so the fix is one step away.
    accounts_menu swallows EOF/KeyboardInterrupt itself (it returns), so the
    menu always continues afterwards.
    """
    reason = _preflight_reason(settings)
    if not reason:
        return True
    print()
    _err(reason)
    if "token" in reason.lower():
        from .accounts import TOKEN_HELP as _TH

        print()
        print(_TH)
    accounts_menu(settings)
    return False


def _report_serve_exit(label: str, exc: SystemExit) -> None:
    """Report a SystemExit from a serve path and hand control back to the menu.

    uvicorn calls sys.exit(STARTUP_FAILURE) when bind() fails, and SystemExit
    is a BaseException: the (EOFError, KeyboardInterrupt) handlers never saw
    it, so a taken port (the port pre-check is TOCTOU) or any other startup
    exit killed the whole TUI with no menu and no message. A clean
    SystemExit(0) is a normal stop, not an error.
    """
    code = exc.code
    if code in (0, None):
        print()
        _info(f"{label}, back to menu.")
        return
    print()
    _err(f"{label} exited with code {code}. Port already in use, or the "
         "provider failed to start -- fix it in Settings/Accounts, or pick "
         "another port, then Start again.")


def _run_mode(settings: Settings, runner: Callable[[Settings], object],
              label: str, stopped: str, refresh: bool) -> None:
    """Run one serve-style mode under the shared exit guard, then return.

    Serve, chat, multi-server and dry run all end the same way: EOF or Ctrl+C
    returns to the menu, a SystemExit is reported as a serve failure, and the
    settings are re-read from disk so an edit made elsewhere shows up. That
    guard was copied four times; it lives here once now.

    `refresh=False` is deliberate for dry run: it writes nothing, so re-reading
    the config would only discard in-memory edits.
    """
    try:
        runner(settings)
    except (EOFError, KeyboardInterrupt):
        print()
        _info(stopped)
    except SystemExit as exc:
        _report_serve_exit(label, exc)
    if refresh:
        _refresh_from_disk(settings)


def _multi_default(settings: Settings) -> object:
    """`multi_fn` fallback: the caller wired no multi-server entry point."""
    from .chat_ui import run_multi_chat_servers

    return run_multi_chat_servers(settings)


def _dry_default(settings: Settings) -> object:
    """`dry_fn` fallback: the caller wired no dry-run entry point."""
    from .dry_run import run_dry_run

    return run_dry_run(settings)


def main_menu(settings: Settings, serve_fn: Callable[[Settings], int],
              chat_fn: Callable[[Settings], int] | None = None,
              multi_fn: Callable[[Settings], int] | None = None,
              dry_fn: Callable[[Settings], int] | None = None) -> int:
    """Blocking main page. Server stop returns to menu (0 = quit).

    Quit lives in the provider chooser ("q. Go back", then its "q. Quit");
    Ctrl+C/EOF on this prompt still quit directly.
    """
    _clear_screen()
    while True:
        _header(_provider_display(settings))
        _info(f"Config   : {settings.config_path or '(unsaved)'}")
        _info(f"Host     : http://{'0.0.0.0' if settings.listen else '127.0.0.1'}"
              f":{settings.port} ({'all interfaces' if settings.listen else 'localhost only'})")
        _active, _is_multi_mm, _single_mm = _provider_scope(settings)
        if _is_multi_mm:
            _scoped_accts = list(settings.accounts)
            _cur_mm = (settings.active_account or "").strip() or "ALL"
        else:
            try:
                from .core import pools as _pools

                _scoped_accts = _pools.accounts_for_provider(settings, _single_mm)
            except Exception:
                _scoped_accts = list(getattr(settings, "accounts", []) or [])
            try:
                from .accounts import _matches as _m_mm

                _raw_mm = (settings.active_account or "").strip()
                if _raw_mm and any(_m_mm(a, _raw_mm) for a in _scoped_accts):
                    _cur_mm = _raw_mm
                else:
                    _cur_mm = "ALL"
            except Exception:
                _cur_mm = (settings.active_account or "").strip() or "ALL"
        # An accountless provider (opencode CLI) has no accounts to manage:
        # its one placeholder screen was a dead end, so the info line and the
        # menu row are both dropped for it.
        show_accounts = _active_providers_need_accounts(settings)
        if show_accounts:
            _info(f"Accounts : {len(_scoped_accts)}  |  Current: [{_cur_mm}]")
        # Menu rows in order; the numbering is derived from this one list, so a
        # dropped row cannot leave a numbered gap on screen.
        entries: list[tuple[str, str]] = [("Start server", "serve")]
        if chat_fn is not None:
            entries.append(("Start server with chat", "chat"))
        if show_accounts:
            entries.append(("Account manager", "accounts"))
        entries.append(("Settings", "settings"))
        if chat_fn is not None:
            entries.append(("Run multiple servers", "multi"))
        entries.append(("Dry run", "dry"))
        accounts_num = str(next((i for i, (_label, _key) in enumerate(entries, 1)
                                 if _key == "accounts"), 0))
        has_token = any((getattr(a, "token", "") or "").strip()
                        for a in _scoped_accts)
        if not _scoped_accts:
            if show_accounts:
                _info(f"Setup    : no accounts yet -> Account manager ({accounts_num}) to add one")
            else:
                # Accountless provider (opencode CLI): _preflight_reason returns
                # "" and Start server works, so pointing at Accounts sent the
                # user to provision a credential nothing will ever read.
                _info("Setup    : ready - this provider needs no accounts")
        elif not has_token and show_accounts:
            _info(f"Setup    : no account token -> Account manager ({accounts_num}) to refresh one")
        print()
        for i, (_label, _key) in enumerate(entries, 1):
            print(f"  {i}. {_label}")
        print("  q. Go back")
        print()
        try:
            choice = input("Select: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if choice == "q":
            # "q. Go back" opens the chooser; that is where the serve is quit
            # from ("q. Quit" in there).
            if _select_provider(settings) is False:
                return 0
            _clear_screen()
            continue
        if choice == "dry":
            action = "dry"
        elif choice.isdigit():
            idx = int(choice) - 1
            action = entries[idx][1] if 0 <= idx < len(entries) else ""
        else:
            action = ""
        if action == "serve":
            if not _preflight_gate(settings):
                continue
            print()
            _info("Server running. Press Ctrl+C or Esc to stop and return to menu.")
            print()
            _run_mode(settings, serve_fn, "Server",
                     "Server stopped, back to menu.", True)
            continue
        if action == "chat":
            if not _preflight_gate(settings):
                continue
            _run_mode(settings, chat_fn, "Chat server",
                      "Chat stopped, back to menu.", True)
            continue
        if action == "multi":
            if not _preflight_gate(settings):
                continue
            _run_mode(settings, multi_fn or _multi_default, "Multi-server",
                      "Multi-server stopped, back to menu.", True)
            continue
        if action == "settings":
            # Guarded like the serve branches: an unexpected failure inside a
            # submenu used to end the process with a traceback, taking the
            # whole TUI down over one bad edit.
            _run_submenu("Settings", settings_menu, settings)
            _clear_screen()
            continue
        if action == "accounts":
            _run_submenu("Account manager", accounts_menu, settings)
            _clear_screen()
            continue
        if action == "dry":
            # No preflight: dry run needs no accounts, launches no browser,
            # and never calls DeepSeek — it echoes the formatted prompt.
            print()
            _info("Dry run. Press Ctrl+C or Esc to stop and return to menu.")
            print()
            # refresh=False: dry run writes nothing (see _run_mode).
            _run_mode(settings, dry_fn or _dry_default, "Dry run",
                      "Dry run stopped, back to menu.", False)
            continue
        _err(f"Unknown option: type 1-{len(entries)} or q.")
