"""Shared server bootstrap: bind-host resolution, port scanning/auto-fix,
uvicorn Config/Server wiring, readiness polling, and shutdown helpers.

Single source of truth for the serve (cli) and chat (chat_ui) entry points;
previously these helpers were duplicated across deepseaport.cli and
deepseaport.chat_ui. cli.py / chat_ui.py keep thin re-exports for backward
compatibility (tests and tui import the historical names from there).
"""

from __future__ import annotations

import ipaddress
import logging
import time
import urllib.request

import uvicorn

logger = logging.getLogger("deepseaport.server_runner")

# Server limits applied to every uvicorn.Config this module builds (serve,
# chat background server, multi-instance chat). uvicorn's defaults are
# unlimited concurrency and a 5s keep-alive: an unbounded in-flight pool
# behind a handful of account slots is how one retry storm turns into socket
# exhaustion, and a 5s keep-alive re-uses sockets of clients that already
# left. 256 concurrent requests is far above what a few agent clients need.
UVICORN_LIMIT_CONCURRENCY = 256
UVICORN_TIMEOUT_KEEP_ALIVE = 15


def _is_loopback_host(host: str) -> bool:
    """True when ``host`` binds only this machine (127.0.0.0/8, ::1, localhost).

    Empty is treated as non-loopback because an empty bind host means "every
    interface" to the socket layer, which is exactly the case to warn about.
    """
    text = str(host or "").strip().lower().strip("[]")
    if not text:
        return False
    if text == "localhost":
        return True
    try:
        return ipaddress.ip_address(text).is_loopback
    except ValueError:
        return False


def lan_exposure_notice(settings, host: str, port: int) -> str:
    """Return a warning line for a non-loopback bind, or "" when loopback."""
    if _is_loopback_host(host):
        return ""
    keys = list(getattr(settings, "keys", None) or [])
    if keys:
        gate = (f"the {len(keys)} API key(s) in settings.keys are the only "
                f"gate, and they travel in a plain header on an unencrypted LAN")
    else:
        gate = ("AUTH DISABLED: settings.keys is empty, so every request is "
                "served without a key")
    return (f"WARNING: serving on {host}:{port} (non-loopback). The whole API "
            f"is reachable from the network, including /v1/accounts CRUD "
            f"(read, overwrite and delete account emails, passwords and "
            f"tokens) and unmetered chat; {gate}. Bind 127.0.0.1 "
            f"(--host 127.0.0.1, or listen: false in config.json) unless you "
            f"meant this.")


def _watch_stop_keys(server, stop=None) -> None:
    """Daemon: Esc/q stops a running server so TUI returns to menu.

    Best-effort, never raises. Ctrl+C is handled by uvicorn/SIGINT;
    this adds Esc (and q) without Enter on Windows (msvcrt) and POSIX.

    `stop` is the caller's own ``threading.Event``, set in a ``finally``
    around the serve call. It exists because ``server.should_exit`` is NOT
    a reliable loop exit: uvicorn's ``Server.startup()`` does
    ``sys.exit(STARTUP_FAILURE)`` when the bind fails (a port that was taken
    between the pre-flight check and the bind) and never touches
    ``should_exit``. The thread then outlived the serve call, kept draining
    CONIN$ every 100 ms and silently swallowed every main-menu keystroke that
    was not Esc/q/Q - the menu looked frozen, and the one line that explained
    it ("Stopping server (key pressed), returning to menu...") was a lie
    because the user was already in the menu. With `stop` the watcher always
    terminates with the serve call, and waits on the event instead of
    busy-polling so it wakes immediately rather than up to 100 ms later.
    """
    def _stopped() -> bool:
        if stop is not None and stop.is_set():
            return True
        return bool(getattr(server, "should_exit", False))

    try:
        try:
            import msvcrt  # type: ignore
        except ImportError:
            msvcrt = None  # type: ignore
        if msvcrt is not None:
            while not _stopped():
                try:
                    if msvcrt.kbhit():
                        ch = msvcrt.getch()
                        if ch in (b"\x1b", b"q", b"Q", b"\x03"):
                            print("\nStopping server (key pressed), returning to menu...")
                            server.should_exit = True
                            break
                    elif stop is not None:
                        stop.wait(0.1)
                    else:
                        import time as _time

                        _time.sleep(0.1)
                except Exception:
                    return
            return
        # POSIX fallback: cbreak stdin + select, restore on exit.
        import select as _select
        import sys as _sys

        try:
            import termios as _termios
            import tty as _tty
        except ImportError:
            return
        try:
            _fd = _sys.stdin.fileno()
        except Exception:
            return
        try:
            _old = _termios.tcgetattr(_fd)
        except Exception:
            return
        try:
            _tty.setcbreak(_fd)
            while not _stopped():
                try:
                    _r, _, _ = _select.select([_sys.stdin], [], [], 0.2)
                except Exception:
                    return
                if _r:
                    try:
                        _ch = _sys.stdin.read(1)
                    except Exception:
                        return
                    if _ch in ("\x1b", "q", "Q"):
                        print("\nStopping server (key pressed), returning to menu...")
                        server.should_exit = True
                        break
        except Exception:
            pass
        finally:
            try:
                _termios.tcsetattr(_fd, _termios.TCSADRAIN, _old)
            except Exception:
                pass
    except Exception:
        return


def _is_port_free(host: str, port: int) -> bool:
    """True when host:port can be bound right now (pre-flight check)."""
    import os
    import socket

    family = socket.AF_INET6 if ":" in (host or "") else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        # SO_REUSEADDR only where uvicorn's own bind will set it. asyncio
        # (uvicorn's transport) sets it when `os.name == "posix"` and NOT on
        # Windows, and the two platforms disagree about what it means: on
        # POSIX it never lets a bind steal a live listener, on Windows it does
        # (both sockets must set it, which is exactly what node,
        # `python -m http.server` and waitress do). Setting it here on Windows
        # therefore made this pre-check report "free" for a port uvicorn would
        # then fail to bind - verified on this box - and a wrong answer in that
        # direction is the one that costs the user a "port busy" dialog
        # instead of a wrong auto-pick.
        if os.name == "posix":
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            except OSError:
                pass
        sock.bind((host, int(port)))
        return True
    except (OSError, OverflowError, ValueError):
        # OverflowError/ValueError: a port outside the legal range or
        # unparsable. config.load_settings already clamps the loaded value
        # (config.py PORT_LIMIT); this keeps a hand-built Settings or a future
        # caller from turning a bad port into a traceback out of the
        # pre-flight check.
        return False
    finally:
        try:
            sock.close()
        except OSError:
            pass


def _resolve_bind_host(settings, args=None) -> str:
    """--host flag wins; else listen=true -> 0.0.0.0, false -> 127.0.0.1."""
    override = ""
    if args is not None:
        override = (getattr(args, "host", "") or "").strip()
    if override:
        return override
    return "0.0.0.0" if bool(getattr(settings, "listen", False)) else "127.0.0.1"


def _scan_free_ports(host: str, start: int, count: int,
                     limit: int | None = None, check=None) -> list[int]:
    """Collect up to `count` free ports scanning upward from `start`.

    Stops at `start+limit-1` when limit is given, else 65535. `check` defaults
    to _is_port_free. Returns fewer than `count` when the range is exhausted.
    """
    free_fn = check if check is not None else _is_port_free
    try:
        candidate = int(start)
    except (TypeError, ValueError):
        candidate = 5001
    if candidate < 1:
        candidate = 1
    end = 65535 if limit is None else min(65535, candidate + int(limit) - 1)
    ports: list[int] = []
    while len(ports) < count and candidate <= end:
        try:
            free = bool(free_fn(host, candidate))
        except Exception:
            free = False
        if free:
            ports.append(candidate)
        candidate += 1
    return ports


def _next_free_port(host: str, port: int, limit: int = 50,
                    check=None) -> int | None:
    """Scan port+1..port+limit for a free port. None when all busy."""
    found = _scan_free_ports(host, int(port) + 1, 1, limit=limit, check=check)
    return found[0] if found else None


def _ensure_free_port(settings, host: str, port: int,
                      is_free=None, next_free=None) -> int:
    """Port auto-fix: busy port offers next free one and saves it.

    Interactive (tty stdin): asks Y/n. Non-interactive (headless): auto-picks
    the next free port when available. Returns the port to use (original when
    user declines or no free port found; 5001-or-next when `port` itself is
    out of range or not a number).

    `is_free` / `next_free` are injection seams used by deepseaport.cli so
    tests can monkeypatch the historical cli-level names; default to the
    canonical helpers in this module.
    """
    import sys as _sys

    free = is_free if is_free is not None else _is_port_free
    nxt_fn = next_free if next_free is not None else _next_free_port

    raw_port = port
    try:
        port = int(port)
    except (TypeError, ValueError):
        port = 0
    if not 1 <= port <= 65535:
        # Out of range must never reach uvicorn: bind() raises OverflowError,
        # which is not an OSError and escapes _run_server's KeyboardInterrupt
        # handler as a traceback (in the TUI it killed the menu).
        logger.warning("port=%r is not a usable listen port: scanning from "
                       "5001 instead", raw_port)
        port = 5001
    if free(host, port):
        return port
    nxt = nxt_fn(host, port)
    if nxt is None:
        print(f"Port {port} busy and no free port found nearby.")
        return port
    try:
        interactive = bool(_sys.stdin.isatty())
    except Exception:
        interactive = False
    if interactive:
        try:
            ans = input(f"Port {port} busy. Use {nxt} instead? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return port
        if ans not in ("", "y", "yes"):
            return port
    else:
        print(f"Port {port} busy, auto-using {nxt}.")
    settings.port = int(nxt)
    try:
        settings.save()
        print(f"Port -> {nxt} (saved)")
    except Exception:
        print(f"Port -> {nxt} (unsaved)")
    return int(nxt)


def _allocate_sequential_ports(host: str, base: int, count: int,
                               is_free=None) -> list[int]:
    """Allocate `count` distinct free ports scanning upward from `base`.

    Mirrors the single-server auto-fix: each candidate is socket-checked;
    busy ones are skipped (5001 taken -> 5002 -> 5003 ...). Never mutates
    settings. Returns fewer than `count` when the range is exhausted.
    """
    return _scan_free_ports(host, base, count, check=is_free)


def _wait_for_server(base_url: str, timeout: int = 20) -> bool:
    """Poll GET /health until 200 or timeout. True when ready."""
    deadline = time.monotonic() + max(1, int(timeout))
    url = base_url.rstrip("/") + "/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _stop_all_servers(servers, threads, timeout: float = 5.0) -> None:
    for srv in servers:
        try:
            srv.should_exit = True
        except Exception:
            pass
    for th in threads:
        try:
            th.join(timeout=timeout)
        except Exception:
            pass


def _start_background_server(app, host: str, port: int,
                             log_level: str = "error",
                             access_log: bool = False):
    """Build a uvicorn Config/Server and run it in a daemon thread.

    Shared by `serve + chat` (one server) and multi-chat (N servers).
    Returns (server, thread) so callers can join/stop them.
    """
    import threading

    # app.state is a plain attribute bag; missing settings only costs the
    # warning (a mock app in tests, or an embedder that never set it).
    state = getattr(app, "state", None)
    notice = lan_exposure_notice(getattr(state, "settings", None), host, int(port))
    if notice:
        # Logged, not printed: the chat/multi-instance entry points own the
        # terminal and an interleaved warning there would corrupt the REPL.
        try:
            logger.warning(notice)
        except Exception:
            pass
    config = uvicorn.Config(app, host=host, port=int(port),
                            log_level=log_level, access_log=access_log,
                            limit_concurrency=UVICORN_LIMIT_CONCURRENCY,
                            timeout_keep_alive=UVICORN_TIMEOUT_KEEP_ALIVE)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server, thread


def _multiworker_rejection(workers: int) -> str:
    """Return the failure message for an unsupported --workers N, else "".

    Kept separate from the call site so the guard is testable without building
    an app or touching uvicorn.
    """
    if int(workers or 1) == 1:
        return ""
    return (
        f"--workers {int(workers)} is not supported. deepseaport serves an "
        "in-process app object and uvicorn needs an import string for "
        "workers>1: it logs \"You must pass the application as an import "
        "string to enable 'reload' or 'workers'\" and exits 3 before the "
        "menu opens. Run one worker (--workers 1), or one server per port / "
        "per config file; process-local state (token-refresh singleflight, "
        "login backoff, the config-save lock) would not cross workers "
        "anyway, so concurrent saves last-writer-win.")


def _run_server(settings, args, ensure_free=None) -> int:
    """Blocking serve loop (single worker). ``--workers N>1`` is rejected.

    Multi-worker was never actually reachable: `uvicorn.run(app, ...)` with an
    app *instance* and workers>1 is a startup failure in uvicorn (it logs the
    import-string requirement and calls sys.exit(3)), so the old branch fell
    through to a hard exit with a process-local-state warning as the only
    explanation. Fail loudly and specifically instead
    (see `_multiworker_rejection`).

    `ensure_free` is an injection seam for deepseaport.cli so the historical
    cli-level `_ensure_free_port` monkeypatch keeps working.
    """
    from .server import apply_log_level, create_app

    import threading

    apply_log_level(getattr(settings, "log_level", "INFO"))
    app = create_app(settings)
    # --host flag overrides config; else listen=true -> 0.0.0.0, false -> 127.0.0.1.
    host = _resolve_bind_host(settings, args)
    # Deliberately not int() here: ensure_free below owns the coercion, and a
    # hand-built Settings with a non-numeric port must not raise ValueError
    # out of the pre-flight line as a traceback.
    port = getattr(settings, "port", 5001)
    log_level = str(getattr(settings, "log_level", "info")).lower()
    workers = int(getattr(args, "workers", 1) or 1)
    # Before the port fix-up: no config save, no prompt, no bind for a mode
    # that cannot start.
    if workers != 1:
        raise SystemExit(_multiworker_rejection(workers))
    if ensure_free is None:
        ensure_free = _ensure_free_port
    port = ensure_free(settings, host, port)
    config = uvicorn.Config(app, host=host, port=port, log_level=log_level,
                            # Access log OFF, same as dry_run and the
                            # background/chat server. Reason: an account
                            # identifier may be a FULL token (accounts._matches
                            # allows a full-token match so
                            # `DELETE /v1/accounts/<token>` can remove a
                            # token-only account) and uvicorn's access line
                            # prints the raw request target, so that one
                            # request writes a live 64-char bearer token --
                            # and the whole query string -- into the log.
                            # Chosen over dropping full-token matching from the
                            # HTTP route layer because that would break a
                            # documented route and would not actually close the
                            # hole: the CLI keeps the match, and the access log
                            # is a plain file on the same machine.
                            # Untouched: uvicorn's error log (startup banner,
                            # unhandled exceptions, shutdown) is a different
                            # logger and still configured by `log_level`; only
                            # the `uvicorn.access` per-request lines go away.
                            access_log=False,
                            limit_concurrency=UVICORN_LIMIT_CONCURRENCY,
                            timeout_keep_alive=UVICORN_TIMEOUT_KEEP_ALIVE)
    server = uvicorn.Server(config)
    keys = list(getattr(settings, "keys", None) or [])
    print(f"deepseaport: serving on {host}:{port} - "
          + (f"auth: {len(keys)} API key(s) required on every endpoint"
             if keys else
             "auth: DISABLED (no keys configured, local dev default)"))
    notice = lan_exposure_notice(settings, host, port)
    if notice:
        print(notice)
        try:
            logger.warning(notice)
        except Exception:
            pass
    stop_keys = threading.Event()
    watcher = threading.Thread(target=_watch_stop_keys, args=(server, stop_keys),
                               daemon=True)
    watcher.start()
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nServer stopped (Ctrl+C), back to menu.")
    finally:
        # Own stop flag, set on every exit path (including the SystemExit from
        # uvicorn's failed-bind startup, which never sets should_exit) so the
        # watcher thread cannot outlive this call and eat menu keystrokes. See
        # _watch_stop_keys. The join makes the guarantee synchronous instead of
        # "within one 100ms poll": the watcher is either in kbhit() (returns at
        # once) or in stop.wait() (returns on set), so this cannot block for
        # long, and it is bounded either way.
        stop_keys.set()
        watcher.join(timeout=1.0)
    return 0
