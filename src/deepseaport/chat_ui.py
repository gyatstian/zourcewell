"""Server-with-chat mode: live API server + simple local chat REPL.

Stdlib only, Windows-safe (plain input(), no curses). The uvicorn server
runs in a background thread with quiet logs so the screen stays a chat
window; external clients can still use the OpenAI-compatible API at the
printed URL while you chat. Exit with /quit (or Ctrl+C) to stop the
server and return to the main menu.

Shared helpers live in deepseaport.chat_repl (REPL/commands) and
deepseaport.server_runner (ports/bootstrap); the names are re-exported here
so existing importers (tests, tui, cli) keep working.
"""

from __future__ import annotations

import random

from . import chat_repl
from .chat_repl import (
    DEFAULT_MODEL,
    _build_instance_settings,
    _build_payload,
    _first_api_key,
    _initial_chat_model,
    _model_list,
    _post_completion,
    _print_help,
    _print_multi_help,
    _prompt_instance_count,
    _remember_chat_model,
    _restore_chat_logs,
    _select_usable_accounts_live,
    _silence_chat_logs,
    resolve_chat_models,
)
from .server_runner import (
    _allocate_sequential_ports,
    _resolve_bind_host,
    _start_background_server,
    _stop_all_servers,
    _wait_for_server,
    lan_exposure_notice,
)

__all__ = [
    "DEFAULT_MODEL",
    "run_chat_server",
    "run_multi_chat_servers",
]


def _announce_exposure(settings, host: str, port: int, what: str = "") -> str:
    """Print the non-loopback exposure warning; "" when the bind is loopback.

    Printed rather than logged, on purpose: the chat modes pin every
    ``deepseaport.*`` logger to ERROR before the server starts (see
    ``_silence_chat_logs`` and ARCHITECTURE.md "Chat log silence"), so the
    notice ``_start_background_server`` logs is discarded — the one path that
    had a warning and showed nothing. Request-log silence is deliberate and
    stays; a startup line that is printed once is not a request log.

    The chat server here is a normal OpenAI-compatible API on whatever host
    the operator configured, including the /v1/accounts CRUD routes, so it
    needs the same warning serve prints rather than a chat-specific exemption.
    """
    notice = lan_exposure_notice(settings, host, int(port))
    if notice:
        print(f"  {what + ' ' if what else ''}{notice}")
    return notice


def _client_host(bind_host: str) -> str:
    """Host the chat client dials for a server bound to ``bind_host``.

    A wildcard/empty bind has no address of its own to dial, so it maps to
    loopback; a specific bind address is reached AT that address (binding
    ``127.0.0.2`` or a LAN address leaves 127.0.0.1 with nothing listening,
    which made every message "connection refused"). IPv6 literals come back
    bracketed so they can ride an ``http://host:port`` URL.
    """
    host = str(bind_host or "").strip().strip("[]")
    if not host or host in ("0.0.0.0", "::", "::0"):
        return "127.0.0.1"
    return f"[{host}]" if ":" in host else host


def _active_providers_need_accounts(settings) -> bool:
    """True when an active provider actually requires accounts/tokens.

    Delegates to ``core.pools.any_provider_needs_accounts`` -- the single
    implementation of the ``requires_accounts`` rule. It used to be a third
    copy of it, written here with lazy imports because ``chat_ui`` is itself
    imported from inside ``tui.main_menu`` and a module-level tui import would
    make the graph circular. The cycle argument does not apply to
    ``core.pools``, which imports neither.
    """
    from .core import pools as _pools

    return _pools.any_provider_needs_accounts(settings)


def run_multi_chat_servers(settings, args=None, count: int | None = None) -> int:
    """Start N chat servers (distinct ports/accounts), one round-robin REPL.

    Each instance gets a random distinct usable (tokened, live-checked
    non-banned) account and a distinct free port scanning up from
    settings.port. The original settings object is never mutated
    (no port save, no account rewrite). Returns 0 on clean stop/cancel,
    1 when startup is blocked or a server never became ready.
    """
    # Account gating only applies when a provider that needs accounts is
    # active; an accountless local provider (opencode) has nothing to
    # provision, so an empty pool must not block the start.
    if _active_providers_need_accounts(settings):
        if not getattr(settings, "accounts", None):
            print("  ! No accounts yet. Add one first.")
            return 1
        if not any((getattr(a, "token", "") or "").strip()
                   for a in (settings.accounts or [])):
            print("  ! Accounts have no token. Add one (y = auto-token) first.")
            return 1

    print("  Checking account bans (live, parallel)...")
    usable, banned, no_token = _select_usable_accounts_live(settings)
    cap = len(usable)
    if cap == 0:
        print("  ! No usable accounts (all banned or missing token). "
              "Add/unban accounts first.")
        return 1
    if count is None:
        print(f"  Usable accounts: {cap} ({banned} banned, "
              f"{no_token} without token skipped)")
        n = _prompt_instance_count(cap)
        if n is None:
            print("  Cancelled.")
            return 0
    else:
        try:
            n = int(count)
        except (TypeError, ValueError):
            print(f"  ! Bad instance count: {count!r}.")
            return 1
        if not 1 <= n <= cap:
            print(f"  ! Need 1-{cap} instance(s) for {cap} usable account(s). "
                  "Add/unban accounts first.")
            return 1

    picked = random.sample(usable, n)
    bind_host = _resolve_bind_host(settings, args)
    try:
        base = int(getattr(settings, "port", 5001) or 5001)
    except (TypeError, ValueError):
        base = 5001
    ports = _allocate_sequential_ports(bind_host, base, n)
    if len(ports) < n:
        print(f"  ! Only {len(ports)} free port(s) from {base}; need {n}. "
              "Free some ports and retry.")
        return 1

    spares = [a for a in usable if not any(a is p for p in picked)]
    instances = [_build_instance_settings(settings, acc, spares, port)
                 for acc, port in zip(picked, ports)]

    from .server import create_app

    saved_logs = _silence_chat_logs()
    servers: list = []
    threads: list = []
    state = {"repl": False}

    def _cleanup() -> None:
        """Stop every server and restore log levels, exactly once.

        Widened to BaseException callers: Ctrl+C while _wait_for_server() is
        polling (up to 20s, the likeliest Ctrl+C window) raises
        KeyboardInterrupt, which `except Exception` never sees. That leaked
        every uvicorn thread and left the whole process at ERROR for the rest
        of the TUI session. The guard makes the REPL's finally idempotent.
        """
        if state.get("done"):
            return
        state["done"] = True
        _stop_all_servers(servers, threads)
        _restore_chat_logs(saved_logs)

    try:
        # Inside the try (i.e. after _silence_chat_logs, whose only restore
        # path is _cleanup) and printed, not logged: the log-level raise is
        # deliberate and permanent for the whole session, so a notice emitted
        # at WARNING would be discarded exactly where the chat server is
        # exposed. `inst` carries the instance's own keys, so the warning
        # names the gate these requests actually face.
        _first = instances[0]
        _notice = lan_exposure_notice(_first, bind_host, int(_first.port))
        if _notice:
            if n == 1:
                print(f"  [1/1 instance] {_notice}")
            else:
                # One warning for the set, naming the range: the instances are
                # N adjacent ports on one host and the text is ~400 chars, so
                # N copies would scroll the REPL help off the screen for an
                # 8-instance run.
                print(f"  [{n} instances: {bind_host}:{ports[0]}-{ports[-1]}] "
                      f"{_notice}")
        for inst in instances:
            app = create_app(inst)
            srv, th = _start_background_server(app, bind_host, int(inst.port))
            servers.append(srv)
            threads.append(th)

        bases = [f"http://{_client_host(bind_host)}:{p}" for p in ports]
        for i, base_url in enumerate(bases):
            if not _wait_for_server(base_url):
                alive = threads[i].is_alive() if i < len(threads) else False
                if not alive:
                    print(f"  ! server {i + 1}/{n} failed to start "
                          f"on {bind_host}:{ports[i]}")
                    return 1
                print(f"  ! server {i + 1}/{n} slow to respond, "
                      f"continuing anyway ({base_url}).")

        try:
            api_key = _first_api_key(settings)
        except Exception:
            api_key = ""
        # Central catalogue: union across instances, fallback to MODELS.
        models, model = resolve_chat_models(settings, instances)
        history: list[dict] = []

        print()
        for i, (port, acc) in enumerate(zip(ports, picked), 1):
            print(f"  [{i}/{n}] API http://{_client_host(bind_host)}:{port} "
                  f"(bind {bind_host}:{port}) account {acc.identifier}")
        print(f"  Chat round-robin across {n} server(s). Model: {model}")
        _print_multi_help()
        print()

        state["repl"] = True
        rstate = {"idx": 0}

        def _next_target():
            target = rstate["idx"] % n
            rstate["idx"] += 1
            tag = (f"[{target + 1}/{n} :{ports[target]} "
                   f"{picked[target].identifier}]")
            return bases[target], tag

        def _servers_command():
            for i, (port, acc) in enumerate(zip(ports, picked), 1):
                alive = (threads[i - 1].is_alive()
                         if i - 1 < len(threads) else False)
                print(f"  [{i}/{n}] :{port} {acc.identifier} "
                      f"({'up' if alive else 'down'})")

        def _alive(i: int) -> bool:
            return i < len(threads) and threads[i].is_alive()

        def _dead_note() -> str:
            """Name the instances whose thread is gone, or "" when all are up.

            A uvicorn thread that dies from an unhandled exception never sets
            `should_exit`, so without the liveness term the REPL kept
            prompting and every message came back "Connection refused" —
            a client-side error for a server-side death, with nothing saying
            the server was gone.
            """
            dead = [f"[{i + 1}/{n} :{ports[i]}]" for i in range(n) if not _alive(i)]
            if not dead:
                return ""
            return (f"chat server {'s' if len(dead) > 1 else ''} no longer "
                    f"running ({', '.join(dead)}): the server thread exited or "
                    f"crashed. /quit back to the menu, then start it again; "
                    f"the /servers list shows the rest as up/down.")

        chat_repl.run_repl(
            model=model,
            models=models,
            api_key=api_key,
            history=history,
            settings=settings,
            post=_post_completion,
            can_continue=lambda: (
                not any(getattr(s, "should_exit", False) for s in servers)
                and all(_alive(i) for i in range(n))),
            print_help=_print_multi_help,
            next_target=_next_target,
            servers_command=_servers_command,
            stop_note=_dead_note,
        )
    finally:
        if state["repl"]:
            print()
            print("  Stopping servers, back to menu...")
        _cleanup()
    return 0


def run_chat_server(settings, args=None) -> int:
    """Start server in background thread, run chat REPL in foreground.

    Returns 0 on clean stop (back to menu), 1 when the server never
    became ready. Ctrl+C also stops and returns.
    """
    from .cli import _ensure_free_port
    from .server import create_app

    bind_host = _resolve_bind_host(settings, args)
    # No int() here: _ensure_free_port owns the coercion (garbage -> warn +
    # scan from 5001), and a cast first is the traceback it exists to prevent.
    port = _ensure_free_port(settings, bind_host, getattr(settings, "port", 5001))
    client_base = f"http://{_client_host(bind_host)}:{port}"
    # Before the silencing below: printed, not logged, because
    # _silence_chat_logs() pins deepseaport.* to ERROR and would discard it.
    _announce_exposure(settings, bind_host, port, "chat + API:")

    # Silence BEFORE the server starts: lifespan warmup (obscura cookies,
    # PoW wasm) logs INFO straight onto the chat input line otherwise.
    # The override lives in the runtime-only _runtime_log_level attribute
    # (the server lifespan reads it before settings.log_level), so the saved
    # log_level field is never touched and a /model pick's settings.save()
    # cannot write ERROR into config.json. Restored on exit.
    orig_runtime = getattr(settings, "_runtime_log_level", None)
    settings._runtime_log_level = "ERROR"
    saved_logs = _silence_chat_logs()
    servers: list = []
    threads: list = []
    state = {"repl": False}

    def _cleanup() -> None:
        """Signal every server to exit, join, restore log levels -- once.

        Ctrl+C here is the common case, not the exotic one: _wait_for_server()
        polls for up to 20s, so that window is where a user hits
        Ctrl+C, and KeyboardInterrupt is a BaseException that the previous
        `except Exception` handlers skipped. The result was a live uvicorn
        thread plus every logger left pinned at ERROR for the rest of the TUI
        session. The once-guard keeps the REPL's finally idempotent, so a
        normal exit still stops the servers exactly one time.
        """
        if state.get("done"):
            return
        state["done"] = True
        for srv in servers:
            try:
                srv.should_exit = True
            except Exception:
                pass
        for th in threads:
            try:
                th.join(timeout=10)
            except Exception:
                pass
        try:
            settings._runtime_log_level = orig_runtime
        except Exception:
            pass
        _restore_chat_logs(saved_logs)

    try:
        app = create_app(settings)
        server, thread = _start_background_server(app, bind_host, port)
        servers.append(server)
        threads.append(thread)

        if not _wait_for_server(client_base):
            if not thread.is_alive():
                print(f"  ! server failed to start on {bind_host}:{port}")
                return 1
            print(f"  ! server slow to respond, continuing anyway ({client_base}).")

        api_key = _first_api_key(settings)
        # Central catalogue: shared with multi-chat via chat_repl.
        models, model = resolve_chat_models(settings)
        history: list[dict] = []

        print()
        print(f"  Chat + API live at {client_base} (bind {bind_host}:{port})")
        print(f"  Model: {model}  |  Account: {(settings.active_account or 'ALL').strip() or 'ALL'}")
        _print_help()
        print()

        state["repl"] = True
        chat_repl.run_repl(
            model=model,
            models=models,
            api_key=api_key,
            history=history,
            settings=settings,
            post=_post_completion,
            # Liveness of the background thread, not just should_exit: a
            # uvicorn thread killed by an unhandled exception never sets it,
            # so the REPL would keep prompting into a port nobody is
            # listening on ("Connection refused" for every message).
            can_continue=lambda: (thread.is_alive()
                                  and not getattr(server, "should_exit", False)),
            print_help=_print_help,
            next_target=lambda: (client_base, ""),
            stop_note=lambda: (
                "" if thread.is_alive() else
                "chat server is no longer running: the server thread exited "
                "or crashed, so the last request failed with a connection "
                "error. /quit back to the menu, then start chat again."),
        )
    finally:
        if state["repl"]:
            print()
            print("  Stopping server, back to menu...")
        _cleanup()
    return 0
