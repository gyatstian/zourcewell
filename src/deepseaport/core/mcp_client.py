"""Small, thread-safe MCP client for Obscura's stdio JSON-RPC server.

Obscura is used as an interactive browser only for login and OAuth-style
flows. The client intentionally supports only what those flows need:
initialize and tools/call. Responses are routed by request id to
``queue.Queue`` objects, so callers wait on an event instead of polling a
shared list. Stderr is drained into a small tail for diagnostics, which makes
a broken browser/subprocess far easier to debug.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading

DEFAULT_TIMEOUT = 120.0


class McpClient:
    """Line-delimited JSON-RPC client over a child process stdio pair."""

    def __init__(self, binary: str, profile: str, timeout: float = DEFAULT_TIMEOUT):
        self.binary = binary
        self.profile = profile
        self.timeout = timeout
        self._id = 0
        self._closed = False
        self._state_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._pending: dict[int, queue.Queue] = {}
        self._stderr_tail: list[str] = []
        # Set to None first so close() is safe on a half-built object (a Popen
        # that raised must not turn cleanup into a second AttributeError).
        self.proc = None
        self._reader = None
        self._stderr_reader = None
        try:
            self.proc = subprocess.Popen(
                [binary, "--storage-dir", profile, "mcp"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self._reader = threading.Thread(target=self._read_loop, daemon=True)
            self._reader.start()
            self._stderr_reader = threading.Thread(target=self._stderr_loop, daemon=True)
            self._stderr_reader.start()
            self._init()
        except BaseException:
            # A handshake that never completes (child alive, no `initialize`
            # reply within self.timeout) used to leave the child AND both
            # readers running: the threads hold bound methods -> self, so the
            # object is rooted in threading._active and __del__/close() never
            # runs. The child keeps --storage-dir open, and the caller
            # (providers/deepseek/auth.py:154-177) swallows the error and
            # RELEASES the profile RLock in its finally -- so the next login
            # spawns a second obscura on the same profile, which is exactly
            # the cookie-jar corruption profile_lock() exists to prevent, and
            # each attempt leaked 2 threads. Fail closed instead.
            self.close()
            raise

    # -- stdio readers -------------------------------------------------
    def _read_loop(self) -> None:
        try:
            for line in self.proc.stdout:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                mid = msg.get("id")
                if mid is None:
                    continue
                with self._state_lock:
                    box = self._pending.get(mid)
                if box is not None:
                    box.put(msg)
        except Exception:
            pass
        finally:
            # Wake every waiter so none hangs on process death.
            with self._state_lock:
                pending = list(self._pending.values())
                self._pending.clear()
            for box in pending:
                box.put_nowait(None)

    def _stderr_loop(self) -> None:
        try:
            for line in self.proc.stderr:
                text = line.rstrip()
                if not text:
                    continue
                self._stderr_tail.append(text)
                del self._stderr_tail[:-20]
        except Exception:
            pass

    # -- request/response ---------------------------------------------
    def _send(self, method: str, params=None):
        if self._closed:
            raise RuntimeError("MCP client is closed")
        if self.proc.poll() is not None:
            raise RuntimeError(f"MCP process exited ({self.proc.returncode}): "
                               f"{self._stderr_tail[-1] if self._stderr_tail else ''}")
        timeout = self.timeout
        with self._state_lock:
            self._id += 1
            mid = self._id
            box = queue.Queue()
            self._pending[mid] = box
        payload = json.dumps(
            {"jsonrpc": "2.0", "id": mid, "method": method,
             "params": params or {}},
            ensure_ascii=False,
        )
        try:
            with self._write_lock:
                self.proc.stdin.write(payload + "\n")
                self.proc.stdin.flush()
            msg = box.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"mcp {method} timed out after {timeout:g}s") from None
        except Exception:
            raise
        finally:
            with self._state_lock:
                self._pending.pop(mid, None)
        if msg is None:
            tail = self._stderr_tail[-1] if self._stderr_tail else "process closed stdout"
            raise RuntimeError(f"mcp {method} failed: {tail}")
        if "error" in msg:
            raise RuntimeError(f"mcp {method}: {msg['error']}")
        return msg.get("result")

    def _init(self) -> None:
        self._send(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "deepseaport", "version": "0.1"},
            },
        )
        with self._write_lock:
            self.proc.stdin.write(
                json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
            )
            self.proc.stdin.flush()

    def call(self, name: str, arguments: dict):
        return self._send("tools/call", {"name": name, "arguments": arguments})

    # -- lifecycle -----------------------------------------------------
    def close(self) -> None:
        """Stop the child and both reader threads. Idempotent.

        Tolerates a partially constructed object (``proc is None``, no reader
        threads yet), because the constructor's failure path calls it after
        anything from ``Popen`` onwards may have raised.
        """
        if getattr(self, "_closed", False):
            return
        self._closed = True
        proc = getattr(self, "proc", None)
        try:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=3)
        except Exception:
            pass
        # Join before closing the pipes: _read_loop/_stderr_loop leave their
        # loop on EOF, i.e. once the child is gone, so a terminated child
        # releases both readers. Bounded so a wedged child cannot hang close();
        # the pipe close below is the backstop that breaks the read.
        for name in ("_reader", "_stderr_reader"):
            thread = getattr(self, name, None)
            try:
                if thread is not None and thread is not threading.current_thread():
                    thread.join(timeout=2)
            except Exception:
                pass
        if proc is None:
            return
        for pipe in (proc.stdin, proc.stdout, proc.stderr):
            try:
                pipe.close()
            except Exception:
                pass

    def __enter__(self) -> "McpClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
