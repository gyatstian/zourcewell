"""Cursor-aware multiline editor for the TUI (extracted from tui.py).

Stdlib-only, no Settings/config imports so provider/settings menus stay
readable in tui.py. tui.py re-exports _read_interactive for backward
compatibility (tests do tui._read_interactive).
"""

from __future__ import annotations

import os
import shutil
import sys
import unicodedata


def _unescape_newlines(raw: str) -> str:
    """Turn typed ``\\n`` into real newlines (``\\\\`` stays a backslash).

    Lets one input line carry multi-line text; ``\\\\n`` types a literal
    backslash-n for the rare case the text itself needs one.
    """
    sentinel = "\x00"
    return (raw.replace("\\\\", sentinel)
               .replace("\\n", "\n")
               .replace(sentinel, "\\"))


def _stdin_is_tty() -> bool:
    """True only when there is a console to prompt on. Never raises.

    The lowest-level terminal helper in the package (this module imports
    nothing from deepseaport), so tui and cli reuse it instead of each spelling
    their own probe; cli imports it from here. getattr-style because
    sys.stdin is None under pythonw.exe and some frozen launchers, where a
    bare sys.stdin.isatty() raises.
    """
    _stdin = getattr(sys, "stdin", None)
    try:
        return bool(getattr(_stdin, "isatty", None) and _stdin.isatty())
    except Exception:
        return False


def _ensure_vt() -> None:
    """Enable ANSI VT processing on Windows consoles, best-effort."""
    try:
        if os.name != "nt":
            return
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


def _read_interactive(prompt: str, initial: str = "") -> str | None:
    """Cursor-aware keystroke editor: ``\\n`` opens a new line at the cursor.

    Left/Right move within text, Up/Down move across lines, Home/End jump
    to line bounds, Backspace/Delete edit at the cursor, Ctrl+U clears the
    whole buffer. ``initial`` preloads the field with cursor at the end.
    Multiline paste works: bracketed paste lands literally, otherwise a CR
    with queued input folds to LF (lone CR = Enter). The prompt is rewritten on every redraw so it cannot be edited or
    deleted. Enter or Ctrl+D saves, Backspace edits (across newlines too),
    Esc / Ctrl+C / Ctrl+Z cancel. Returns None on cancel. Falls back to
    plain ``input()`` + unescape when stdin is not a tty (pipes, tests).
    """
    if not _stdin_is_tty():
        try:
            return _unescape_newlines(input(prompt))
        except (EOFError, KeyboardInterrupt):
            print()
            return None
    try:
        import msvcrt  # type: ignore
    except ImportError:
        msvcrt = None  # type: ignore
    if msvcrt is not None:
        return _read_interactive_windows(prompt, initial)
    try:
        import select as _select
        import termios as _termios
        import tty as _tty
    except ImportError:
        try:
            return _unescape_newlines(input(prompt))
        except (EOFError, KeyboardInterrupt):
            print()
            return None
    return _read_interactive_posix(prompt, _select, _termios, _tty, initial)


_WIN_SPECIAL_MAP = {
    "H": "up",
    "P": "down",
    "K": "left",
    "M": "right",
    "G": "home",
    "O": "end",
    "S": "delete",
    "R": "unknown",
}


def _decode_windows_special(second: str) -> str:
    """Map msvcrt arrow/home/end/delete tail to an action name."""
    return _WIN_SPECIAL_MAP.get(second, "unknown")


def _pos_line_col(text: str, pos: int) -> tuple[int, int]:
    before = text[:max(0, min(pos, len(text)))]
    return before.count("\n"), len(before.split("\n")[-1])


def _line_start(text: str, line_idx: int) -> int:
    lines = text.split("\n")
    line_idx = max(0, min(line_idx, len(lines) - 1))
    return sum(len(l) + 1 for l in lines[:line_idx])


def _apply_cursor_move(text: str, pos: int, action: str) -> int:
    pos = max(0, min(pos, len(text)))
    if action == "left":
        return max(0, pos - 1)
    if action == "right":
        return min(len(text), pos + 1)
    if action == "home":
        line_idx, _ = _pos_line_col(text, pos)
        return _line_start(text, line_idx)
    if action == "end":
        line_idx, _ = _pos_line_col(text, pos)
        return _line_start(text, line_idx) + len(text.split("\n")[line_idx])
    if action in ("up", "down"):
        lines = text.split("\n")
        line_idx, col = _pos_line_col(text, pos)
        target = line_idx - 1 if action == "up" else line_idx + 1
        if target < 0 or target >= len(lines):
            return pos
        return _line_start(text, target) + min(col, len(lines[target]))
    return pos


def _editor_width() -> int:
    """Live terminal width for wrap-aware cursor math (fallback 80)."""
    try:
        return shutil.get_terminal_size().columns or 80
    except Exception:
        return 80


def _char_cell(ch: str, col: int) -> int:
    """Display columns ``ch`` occupies when written at column ``col``."""
    if ch == "\t":
        return 8 - (col % 8)
    if unicodedata.combining(ch):
        return 0  # combining marks stack on the previous cell
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def _visual_pos(text: str, width: int) -> tuple[int, int]:
    """(row, col) of the cursor after rendering ``text``, wrap-aware.

    Mirrors terminal wrapping: a char that would pass the right margin
    starts a new row first (tabs jump to multiples of 8, CJK takes 2).
    """
    if width < 2:
        width = 80
    row, col = 0, 0
    for ch in text:
        if ch == "\n":
            row += 1
            col = 0
            continue
        w = _char_cell(ch, col)
        if col + w > width:
            row += 1
            col = 0
            w = _char_cell(ch, 0)
        col += w
    return row, col


def _redraw_editor(out, prompt: str, buf: list[str], old_text: str, old_pos: int, new_pos: int) -> None:
    """Full redraw of the cursor-aware editor.

    Moves from the old cursor back to the input start, clears everything
    below, rewrites ``prompt + text``, then parks the cursor at ``new_pos``.
    Row math is wrap-aware (live terminal width) so long pasted lines stay
    in sync; the prompt is rewritten every time so it can never be edited
    or deleted.
    """
    old_pos = max(0, min(old_pos, len(old_text)))
    new_text = "".join(buf)
    new_pos = max(0, min(new_pos, len(new_text)))
    width = _editor_width()
    old_row, _ = _visual_pos(prompt + old_text[:old_pos], width)
    if old_row > 0:
        out.write(f"\x1b[{old_row}A")
    out.write("\r")
    # Clear first: a shortened middle line would otherwise leave ghost
    # chars behind (its newline lands on the stale column without erasing).
    out.write("\x1b[J")
    out.write(prompt + new_text)
    new_row, new_col = _visual_pos(prompt + new_text[:new_pos], width)
    end_row, end_col = _visual_pos(prompt + new_text, width)
    delta = end_row - new_row
    if delta > 0:
        out.write(f"\x1b[{delta}A")
        out.write("\r")
        if new_col > 0:
            out.write(f"\x1b[{new_col}C")
    else:
        back = end_col - new_col
        if back > 0:
            out.write(f"\x1b[{back}D")
    out.flush()


def _read_interactive_windows(prompt: str, initial: str = "") -> str | None:
    """Cursor-aware editor: Left/Right move, Up/Down change lines.

    ``initial`` preloads the field; cursor starts at the end.
    """
    import sys

    import msvcrt  # type: ignore

    _ensure_vt()
    out = sys.stdout
    out.write(prompt + initial)
    out.flush()
    buf: list[str] = list(initial)
    pos = len(buf)
    pending_bs = False
    prefetch: str | None = None  # char already consumed (paste peek), runs next

    def _insert(s: str) -> None:
        nonlocal pos
        old_text = "".join(buf)
        old_pos = pos
        for c in s:
            buf.insert(pos, c)
            pos += 1
        _redraw_editor(out, prompt, buf, old_text, old_pos, pos)

    def _move(action: str) -> None:
        nonlocal pos
        text = "".join(buf)
        new_pos = _apply_cursor_move(text, pos, action)
        if new_pos != pos:
            old_pos = pos
            pos = new_pos
            _redraw_editor(out, prompt, buf, text, old_pos, pos)

    def _paste_chunk() -> str | None:
        """Read a bracketed paste (``ESC[200~`` already eaten) to ``ESC[201~``.

        Raw chars, no key handling; CRLF/CR fold to LF. None = cancelled.
        """
        acc: list[str] = []
        while True:
            try:
                p = msvcrt.getwch()
            except KeyboardInterrupt:
                out.write("\n")
                out.flush()
                return None
            acc.append(p)
            if len(acc) >= 6 and "".join(acc[-6:]) == "\x1b[201~":
                break
        return "".join(acc[:-6]).replace("\r\n", "\n").replace("\r", "\n")

    while True:
        if prefetch is not None:
            ch = prefetch
            prefetch = None
        else:
            try:
                ch = msvcrt.getwch()
            except KeyboardInterrupt:
                out.write("\n")
                out.flush()
                return None
        if ch == "\x1b":  # Esc — or bracketed-paste open marker
            if msvcrt.kbhit():
                import time as _time

                ahead = ""
                deadline = _time.monotonic() + 0.05
                while len(ahead) < 5:
                    if msvcrt.kbhit():
                        ahead += msvcrt.getwch()
                        if ahead == "[200~":
                            break
                        if not "[200~".startswith(ahead):
                            ahead = ""
                            break
                    elif not ahead:
                        break  # nothing queued: lone Esc, instant
                    elif _time.monotonic() >= deadline:
                        ahead = ""
                        break  # stalled mid-marker: treat as Esc
                    else:
                        _time.sleep(0.005)
                if ahead == "[200~":
                    if pending_bs:
                        _insert("\\")
                        pending_bs = False
                    chunk = _paste_chunk()
                    if chunk is None:
                        return None
                    if chunk:
                        _insert(chunk)
                    continue
            out.write("\n")
            out.flush()
            return None
        if ch == "\r":
            if pending_bs:  # lone "\" then Enter/newline: keep it
                _insert("\\")
                pending_bs = False
            if msvcrt.kbhit():
                # Queued input right after CR = pasted line break (CRLF or
                # CR), not Enter: fold to LF, keep any char after it.
                nxt = msvcrt.getwch()
                _insert("\n")
                if nxt != "\n":
                    prefetch = nxt
                continue
            out.write("\n")
            out.flush()
            return "".join(buf)
        if ch in ("\x03", "\x1a"):  # Ctrl+C / Ctrl+Z
            out.write("\n")
            out.flush()
            return None
        if ch == "\x04":  # Ctrl+D: save
            if pending_bs:
                _insert("\\")
                pending_bs = False
            out.write("\n")
            out.flush()
            return "".join(buf)
        if ch in ("\x00", "\xe0"):
            try:
                second = msvcrt.getwch()
            except KeyboardInterrupt:
                out.write("\n")
                out.flush()
                return None
            if pending_bs:  # "\" then arrow: keep "\", then move
                _insert("\\")
                pending_bs = False
            action = _decode_windows_special(second)
            if action in ("left", "right", "up", "down", "home", "end"):
                _move(action)
            elif action == "delete":
                if pos < len(buf):
                    old_text = "".join(buf)
                    old_pos = pos
                    del buf[pos]
                    _redraw_editor(out, prompt, buf, old_text, old_pos, pos)
            continue
        if ch in ("\x08", "\x7f"):  # Backspace at cursor
            if pending_bs:
                pending_bs = False
                continue
            if pos > 0:
                old_text = "".join(buf)
                old_pos = pos
                del buf[pos - 1]
                pos -= 1
                _redraw_editor(out, prompt, buf, old_text, old_pos, pos)
            continue
        if ch in ("\x01",):  # Ctrl+A: line start
            if pending_bs:
                _insert("\\")
                pending_bs = False
            _move("home")
            continue
        if ch in ("\x05",):  # Ctrl+E: line end
            if pending_bs:
                _insert("\\")
                pending_bs = False
            _move("end")
            continue
        if ch in ("\x15",):  # Ctrl+U: clear whole buffer
            pending_bs = False
            if buf:
                old_text = "".join(buf)
                old_pos = pos
                del buf[:]
                pos = 0
                _redraw_editor(out, prompt, buf, old_text, old_pos, pos)
            continue
        if pending_bs:
            pending_bs = False
            if ch == "n":  # the magic: insert real newline at cursor
                _insert("\n")
                continue
            if ch == "\\":
                _insert("\\")
                continue
            # Anything else keeps both characters literally at cursor.
            _insert("\\" + ch)
            continue
        if ch == "\\":
            pending_bs = True  # hold back: echo only once resolved
            continue
        if ch == "\n":  # pasted LF (Enter arrives as \r): newline, not submit
            if pending_bs:
                _insert("\\")
                pending_bs = False
            _insert("\n")
            continue
        if len(ch) == 1 and (ch.isprintable() or ch == "\t"):
            _insert(ch)
            continue
        # Unmapped control char: ignore.
    # Unreachable: every path above continues or returns via Enter/Esc.


def _decode_posix_special(seq: str) -> str:
    """Map a POSIX escape tail (after ESC) to left/right/up/down/home/end/delete."""
    if not seq:
        return "unknown"
    if seq.startswith("["):
        body = seq[1:]
        if not body:
            return "unknown"
        final = body[-1]
        if final == "A":
            return "up"
        if final == "B":
            return "down"
        if final == "C":
            return "right"
        if final == "D":
            return "left"
        if final in ("H",):
            return "home"
        if final in ("F",):
            return "end"
        if final == "~":
            first = (body[:-1].split(";")[0] or "")
            if first in ("3",):
                return "delete"
            if first in ("1", "7"):
                return "home"
            if first in ("4", "8"):
                return "end"
            return "unknown"
        return "unknown"
    if seq.startswith("O"):
        if len(seq) < 2:
            return "unknown"
        final = seq[1]
        return {"A": "up", "B": "down", "C": "right", "D": "left",
                "H": "home", "F": "end"}.get(final, "unknown")
    return "unknown"


def _read_interactive_posix(prompt: str, _select, _termios, _tty, initial: str = "") -> str | None:
    """Cursor-aware editor (POSIX): Left/Right move, Up/Down change lines.

    ``initial`` preloads the field; cursor starts at the end.
    """
    import sys

    stdin, out = sys.stdin, sys.stdout
    try:
        fd = stdin.fileno()
    except Exception:
        try:
            return _unescape_newlines(input(prompt))
        except (EOFError, KeyboardInterrupt):
            print()
            return None
    try:
        old = _termios.tcgetattr(fd)
    except Exception:
        try:
            return _unescape_newlines(input(prompt))
        except (EOFError, KeyboardInterrupt):
            print()
            return None
    out.write(prompt + initial)
    out.flush()
    buf: list[str] = list(initial)
    pos = len(buf)
    pending_bs = False
    prefetch: str | None = None  # char already consumed (paste peek), runs next

    def _insert(s: str) -> None:
        nonlocal pos
        old_text = "".join(buf)
        old_pos = pos
        for c in s:
            buf.insert(pos, c)
            pos += 1
        _redraw_editor(out, prompt, buf, old_text, old_pos, pos)

    def _move(action: str) -> None:
        nonlocal pos
        text = "".join(buf)
        new_pos = _apply_cursor_move(text, pos, action)
        if new_pos != pos:
            old_pos = pos
            pos = new_pos
            _redraw_editor(out, prompt, buf, text, old_pos, pos)

    def _read_raw_seq() -> str:
        # stdin is in cbreak; read the rest of the escape sequence.
        seq = ""
        try:
            c1 = stdin.read(1)
        except Exception:
            return ""
        if not c1:
            return ""
        seq += c1
        if c1 == "[":
            while True:
                try:
                    c2 = stdin.read(1)
                except Exception:
                    break
                if not c2:
                    break
                seq += c2
                if c2.isalpha() or c2 == "~":
                    break
        elif c1 == "O":
            try:
                c2 = stdin.read(1)
            except Exception:
                c2 = ""
            if c2:
                seq += c2
        return seq

    def _read_special() -> str:
        seq = _read_raw_seq()
        return _decode_posix_special(seq) if seq else "unknown"

    def _paste_chunk() -> str | None:
        """Read a bracketed paste (``ESC[200~`` already eaten) to ``ESC[201~``.

        Raw chars, no key handling; CRLF/CR fold to LF. None = cancelled.
        """
        acc: list[str] = []
        while True:
            try:
                p = stdin.read(1)
            except KeyboardInterrupt:
                out.write("\n")
                out.flush()
                return None
            if not p:
                break  # EOF mid-paste: keep what arrived
            if p == "\x03":
                out.write("\n")
                out.flush()
                return None
            acc.append(p)
            if len(acc) >= 6 and "".join(acc[-6:]) == "\x1b[201~":
                acc = acc[:-6]
                break
        return "".join(acc).replace("\r\n", "\n").replace("\r", "\n")

    try:
        _tty.setcbreak(fd)
        out.write("\x1b[?2004h")  # bracketed paste: pastes arrive wrapped
        out.flush()
        while True:
            if prefetch is not None:
                ch = prefetch
                prefetch = None
            else:
                try:
                    ch = stdin.read(1)
                except KeyboardInterrupt:
                    out.write("\n")
                    out.flush()
                    return None
            if not ch:
                break  # EOF pipe-closed: save what there is
            if ch == "\x04":  # Ctrl+D: done
                if pending_bs:
                    _insert("\\")
                    pending_bs = False
                break
            if ch in ("\x03", "\x1a"):  # Ctrl+C / Ctrl+Z: cancel
                out.write("\n")
                out.flush()
                return None
            if ch == "\x1b":
                # Lone Esc cancels; an escape *sequence* (arrows) moves;
                # a bracketed-paste marker drops into literal paste mode.
                r, _, _ = _select.select([stdin], [], [], 0.05)
                if r:
                    if pending_bs:
                        _insert("\\")
                        pending_bs = False
                    seq = _read_raw_seq()
                    if seq == "[200~":
                        chunk = _paste_chunk()
                        if chunk is None:
                            return None
                        if chunk:
                            _insert(chunk)
                        continue
                    action = _decode_posix_special(seq)
                    if action in ("left", "right", "up", "down", "home", "end"):
                        _move(action)
                    elif action == "delete":
                        if pos < len(buf):
                            old_text = "".join(buf)
                            old_pos = pos
                            del buf[pos]
                            _redraw_editor(out, prompt, buf, old_text, old_pos, pos)
                    continue
                out.write("\n")
                out.flush()
                return None
            if ch == "\r":
                if pending_bs:
                    _insert("\\")
                    pending_bs = False
                r, _, _ = _select.select([stdin], [], [], 0)
                if r:
                    # Queued input right after CR = pasted CRLF/CR line
                    # break, not Enter: fold to LF, keep any char after it.
                    nxt = stdin.read(1)
                    _insert("\n")
                    if nxt and nxt != "\n":
                        prefetch = nxt
                    continue
                out.write("\n")
                out.flush()
                break
            if ch == "\n":  # pasted LF (Enter arrives as \r): newline
                if pending_bs:
                    _insert("\\")
                    pending_bs = False
                _insert("\n")
                continue
            if ch in ("\x08", "\x7f"):  # Backspace at cursor
                if pending_bs:
                    pending_bs = False
                    continue
                if pos > 0:
                    old_text = "".join(buf)
                    old_pos = pos
                    del buf[pos - 1]
                    pos -= 1
                    _redraw_editor(out, prompt, buf, old_text, old_pos, pos)
                continue
            if ch == "\x01":  # Ctrl+A: line start
                if pending_bs:
                    _insert("\\")
                    pending_bs = False
                _move("home")
                continue
            if ch == "\x05":  # Ctrl+E: line end
                if pending_bs:
                    _insert("\\")
                    pending_bs = False
                _move("end")
                continue
            if ch == "\x15":  # Ctrl+U: clear whole buffer
                pending_bs = False
                if buf:
                    old_text = "".join(buf)
                    old_pos = pos
                    del buf[:]
                    pos = 0
                    _redraw_editor(out, prompt, buf, old_text, old_pos, pos)
                continue
            if pending_bs:
                pending_bs = False
                if ch == "n":
                    _insert("\n")
                    continue
                if ch == "\\":
                    _insert("\\")
                    continue
                _insert("\\" + ch)
                continue
            if ch == "\\":
                pending_bs = True
                continue
            if len(ch) == 1 and (ch.isprintable() or ch == "\t"):
                _insert(ch)
                continue
            # Unmapped control char: ignore.
    finally:
        try:
            out.write("\x1b[?2004l")  # bracketed paste off
            out.flush()
        except Exception:
            pass
        try:
            _termios.tcsetattr(fd, _termios.TCSADRAIN, old)
        except Exception:
            pass
    out.write("\n")
    out.flush()
    return "".join(buf)
