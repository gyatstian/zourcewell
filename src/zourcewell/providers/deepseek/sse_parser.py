"""SSE event parsing: Stateful StreamParser + stateless one-line helper.

Owns SSE line/event parsing: StreamParser.feed, the batched v-list,
usage/finished/error envelopes, continuation chunks, and the
fragments-based stream shape (response/fragments/-1/content). Ban
classification, retry codes, string matching and the header builders stay
in protocol.py, the authoritative source (imported here only via a deferred
local import, to avoid the module cycle); HTTP status/body/error-drain
handling lives in core/transport.py.
"""

from __future__ import annotations

import json
import logging
from typing import Iterator

from ...core.types import StreamEvent

# Preserve the original log channel so existing filters see the same records.
logger = logging.getLogger("zourcewell.protocol")


class StreamParser:
    """Stateful SSE parser: continuation chunks arrive as bare {"v": "..."}
    with no "p" path, so the last content path is remembered.

    Upstream streams in two shapes (both handled):
    - legacy: {"p": "response/content" | "response/thinking_content", "v": ...}
    - fragments (current): the reply is a list of typed fragments. The first
      fragment arrives inside {"v": {"response": {"fragments": [...]}}}, later
      fragments via {"p": "response/fragments", "o": "APPEND", "v": [...]},
      and every text delta via {"p": "response/fragments/-1/content", ...}
      plus bare {"v": "..."} continuations. Fragment {"type": "THINK"} routes
      to thinking, {"type": "RESPONSE"} to content; any other fragment type
      is UI chrome (e.g. the TIP disclaimer) and is dropped. The active
      fragment type is remembered so continuations land on the right side.
    """

    CONTENT = "response/content"
    THINKING = "response/thinking_content"
    FRAG_CONTENT = "response/fragments/-1/content"
    FRAGMENTS = "response/fragments"
    # Finished markers: legacy response/status plus the BATCH-carried
    # quasi_status (both qualified and parent-relative bare forms).
    FINISHED_PATHS = ("response/status", "status",
                      "response/quasi_status", "quasi_status")
    USAGE_PATHS = ("response/accumulated_token_usage",
                   "accumulated_token_usage")

    def __init__(self) -> None:
        self.last_path = ""
        # True when the current (last) fragment is a THINK fragment, so
        # fragment-content deltas and continuations route to thinking.
        self.frag_thinking = False

    @staticmethod
    def _is_frag_content_path(path: str) -> bool:
        """True for fragment text-delta paths (any fragment index)."""
        return (path == StreamParser.FRAG_CONTENT
                or (path.startswith("response/fragments/")
                    and path.endswith("/content")))

    def _route_text(self, path: str, text: str) -> StreamEvent | None:
        """Route one non-empty text delta to thinking/content, else None.

        Fragment deltas follow the active fragment type; legacy paths follow
        the exact path. Sets last_path only when an event is produced (same
        as the historical branches did).
        """
        if not isinstance(text, str) or not text:
            return None
        if self._is_frag_content_path(path):
            kind = "thinking" if self.frag_thinking else "content"
        elif path == self.THINKING:
            kind = "thinking"
        elif path == self.CONTENT:
            kind = "content"
        else:
            return None
        self.last_path = path
        return StreamEvent(kind=kind, text=text)

    def _frag_events(self, frag: dict) -> Iterator[StreamEvent]:
        """Track one answer fragment, yielding its initial content (if any).

        Only THINK (thinking) and RESPONSE (content) fragments carry answer
        text. UI chrome such as {"type": "TIP", "style": "WARNING",
        "hide_on_wip": true} ("This response is AI-generated, for reference
        only.") is dropped entirely — and must not touch frag_thinking /
        last_path, so a chrome fragment can never reroute later deltas.
        """
        try:
            kind = str(frag.get("type", "") or "").upper()
        except Exception:
            kind = ""
        if "THINK" in kind:
            self.frag_thinking = True
        elif "RESPONSE" in kind:
            self.frag_thinking = False
        else:
            return
        self.last_path = self.FRAG_CONTENT
        try:
            text = frag.get("content", "")
        except Exception:
            text = ""
        event = self._route_text(self.last_path, text)
        if event is not None:
            yield event

    def feed(self, line: str) -> Iterator[StreamEvent]:
        # Deferred import: protocol.py re-exports this module at top level,
        # so a top-level import would cycle. Resolved at call time when both
        # modules are fully loaded; behavior identical to direct globals.
        from .protocol import ban_message, detect_ban, is_ban_payload, with_ban_expiry

        line = line.strip()
        if not line or not line.startswith("data:"):
            return
        data = line[5:].strip()
        if data == "[DONE]":
            yield StreamEvent(kind="finished")
            return
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            logger.debug("skip non-JSON SSE line: %.80s", data)
            return
        if not isinstance(chunk, dict):
            return
        if "v" not in chunk:
            # Toast errors arrive as data: {"type":"error","content":"...","finish_reason":"..."}
            # with no "v"/"code" keys (e.g. rate_limit_reached). Surface them.
            if chunk.get("type") == "error" or "content" in chunk and "finish_reason" in chunk:
                code = str(chunk.get("finish_reason") or chunk.get("type") or "UPSTREAM_ERROR")
                yield StreamEvent(kind="error", code=code,
                                  message=str(chunk.get("content") or chunk.get("msg") or ""))
                return
            biz = chunk.get("data") if isinstance(chunk.get("data"), dict) else {}
            code = chunk.get("code", (biz or {}).get("biz_code", ""))
            # Top-level code 0 can still hide an inner ban (biz_code 5):
            # completion ban arrives as {code:0, data:{biz_code:5,...}}.
            inner_code = (biz or {}).get("biz_code", "")
            if code in (0, "0", None, "") and inner_code not in (0, "0", None, ""):
                code = inner_code
            if code not in (0, "0", None, "") or is_ban_payload(chunk):
                msg = str(chunk.get("msg") or (biz or {}).get("biz_msg") or "")
                ban_ts: float | None = None
                try:
                    banned, ts = detect_ban(chunk)
                    if banned:
                        ban_ts = ts
                        msg = with_ban_expiry(msg, chunk)
                        if not msg:
                            msg = ban_message(ts)
                        code = str(code) if code not in (0, "0", None, "") else "5"
                except Exception:
                    pass
                yield StreamEvent(kind="error", code=str(code), message=msg,
                                  ban_until=ban_ts)
            return
        # Fragments snapshot: {"v": {"response": {"fragments": [...]}}}.
        # Carries the first fragment's initial content (THINK -> thinking,
        # RESPONSE -> content). Unknown v-dicts keep the historical no-op.
        if isinstance(chunk.get("v"), dict):
            resp = chunk["v"].get("response")
            if isinstance(resp, dict):
                for frag in resp.get("fragments", []) or []:
                    if isinstance(frag, dict):
                        yield from self._frag_events(frag)
            return
        # Later fragments: {"p": "response/fragments", "o": "APPEND", "v": [...]}
        # (e.g. the THINK -> RESPONSE switch, carrying its initial content).
        path = chunk.get("p", "") or self.last_path
        value = chunk.get("v")
        if path == self.FRAGMENTS and isinstance(value, (list, dict)):
            frags = value if isinstance(value, list) else [value]
            for frag in frags:
                if isinstance(frag, dict):
                    yield from self._frag_events(frag)
            return
        if value == "FINISHED" and path in self.FINISHED_PATHS:
            self.last_path = ""
            yield StreamEvent(kind="finished")
            return
        if path == "response/search_status" or path.endswith("/elapsed_secs"):
            return
        if isinstance(value, int) and path in self.USAGE_PATHS:
            yield StreamEvent(kind="usage", text=str(value))
            return
        if isinstance(value, list):
            # Batched deltas: upstream can pack several {p,v} pairs (or raw
            # strings) in one "v" list. The old code only looked for FINISHED
            # and dropped any batched content, truncating the reply.
            # Items under {"p": "response", "o": "BATCH"} carry parent-relative
            # paths ("accumulated_token_usage", "quasi_status",
            # "fragments/-1/content"), resolved here -- but ONLY when the item
            # actually carried its own "p". An item without one is a
            # continuation: it inherits the last live text path, exactly like
            # the top-level bare {"v": "..."} shape. Prefixing an inherited
            # path produced "response/response" (no content path at all) and
            # routing a raw string to the parent "response" dropped it, so
            # batched continuations silently truncated replies.
            for item in value:
                if isinstance(item, dict):
                    own = str(item.get("p", "") or "")
                    if own:
                        ip = own
                        if "/" not in own and path == "response":
                            ip = "response/" + own
                    else:
                        ip = self.last_path or path
                    iv = item.get("v", "")
                    if iv == "FINISHED" and ip in self.FINISHED_PATHS + ("status",):
                        self.last_path = ""
                        yield StreamEvent(kind="finished")
                        return
                    if ip == "response/search_status":
                        continue
                    if (ip in self.USAGE_PATHS
                            and isinstance(iv, int)):
                        yield StreamEvent(kind="usage", text=str(iv))
                        continue
                    if isinstance(iv, list):
                        for sub in iv:
                            event = self._route_text(ip, sub) if isinstance(sub, str) else None
                            if event is not None:
                                yield event
                        continue
                    event = self._route_text(ip, iv) if isinstance(iv, str) else None
                    if event is not None:
                        yield event
                elif isinstance(item, str) and item:
                    event = self._route_text(self.last_path or path, item)
                    if event is not None:
                        yield event
            return
        event = self._route_text(path, value) if isinstance(value, str) else None
        if event is not None:
            yield event


def parse_sse_line(line: str) -> Iterator[StreamEvent]:
    """Stateless one-line parse (no continuation tracking)."""
    yield from StreamParser().feed(line)
