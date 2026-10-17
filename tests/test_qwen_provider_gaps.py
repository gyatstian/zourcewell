"""Gap tests for the qwen provider: protocol.py / sse_parser.py / client.py /
auth.py / completion_service.py / provider.py — offline only, no network.
Mirrors tests/test_client_protocol_gaps.py."""

from __future__ import annotations

import json

import pytest

from deepseaport.providers.qwen import protocol as QP
from deepseaport.providers.qwen import sse_parser as QSP
from deepseaport.providers.qwen import client as QC
from deepseaport.providers.qwen import auth as QA
from deepseaport.providers.qwen import provider as QProv
from deepseaport.providers.qwen import completion_service as QCS


_UNSET = object()


class _FakeResp:
    """Mirrors tests/test_client_protocol_gaps.py._FakeResp."""

    def __init__(self, status_code=200, text="", content_type="application/json",
                 lines=None, json_data=_UNSET, json_raises=False):
        self.status_code = status_code
        self._text = text
        self.headers = {"content-type": content_type}
        self._lines = list(lines) if lines is not None else []
        self._json_data = json_data
        self._json_raises = json_raises
        self.closed = False

    @property
    def text(self) -> str:
        return self._text

    def json(self):
        if self._json_raises:
            raise ValueError("bad json")
        if self._json_data is not _UNSET:
            return self._json_data
        return json.loads(self._text)

    def iter_lines(self):
        return iter(self._lines)

    def close(self) -> None:
        self.closed = True
        return None


class _DrainedResp(_FakeResp):
    """A 200 non-SSE double that RAISES if its body is iterated a second time.

    curl_cffi does not yield EOF for a second iteration of a drained, closed
    response: against 0.16.3 a 200 + empty body parked in the second
    ``iter_lines`` forever, with no exception to unwind on (a local server
    repro sat there for 3s+). A test cannot model "blocks forever", so the
    second call raises instead -- the point under test is that the fixed code
    never reaches it.
    """

    def __init__(self, status_code=200, **kw):
        super().__init__(status_code, **kw)
        self.iter_lines_calls = 0

    def iter_lines(self):
        self.iter_lines_calls += 1
        if self.iter_lines_calls > 1:
            raise AssertionError(
                "pump_sse re-iterated a drained+closed 200 response")
        return super().iter_lines()


def _sess(monkeypatch, *, get=None, get_exc=None, post=None, post_exc=None,
          delete=None, delete_exc=None):
    attrs = {}
    if get_exc is not None:
        def _raise_get(*a, **k):
            raise get_exc
        attrs["get"] = staticmethod(_raise_get)
    elif get is not None:
        attrs["get"] = staticmethod(lambda *a, **k: get)
    if post_exc is not None:
        def _raise_post(*a, **k):
            raise post_exc
        attrs["post"] = staticmethod(_raise_post)
    elif post is not None:
        attrs["post"] = staticmethod(lambda *a, **k: post)
    if delete_exc is not None:
        def _raise_delete(*a, **k):
            raise delete_exc
        attrs["delete"] = staticmethod(_raise_delete)
    elif delete is not None:
        attrs["delete"] = staticmethod(lambda *a, **k: delete)
    monkeypatch.setattr(QC, "_SESSION", type("_S", (), attrs)())


def _recording_post(monkeypatch, resp):
    """Fake _SESSION whose post() records the URL and returns resp."""
    calls = []

    class _S:
        def post(self, url, *a, **k):
            calls.append(url)
            return resp

    monkeypatch.setattr(QC, "_SESSION", _S())
    return calls


# protocol -------------------------------------------------------------------


def test_base_headers_defaults():
    h = QP.base_headers()
    assert h["User-Agent"] == QP.CHROME_UA
    assert "Version" in h and h["Version"]
    assert h["source"] == "web"
    assert h["X-Request-Id"]
    assert h["Timezone"]
    assert "Cookie" not in h
    assert "Authorization" not in h


def test_base_headers_with_auth():
    h = QP.base_headers(user_agent="UA-X", waf_cookies="a=b", bearer="tok")
    assert h["User-Agent"] == "UA-X"
    assert h["Cookie"] == "a=b; token=tok"
    assert h["Authorization"] == "Bearer tok"


def test_base_headers_bearer_only_token_cookie():
    h = QP.base_headers(bearer="tok")
    assert h["Cookie"] == "token=tok"
    assert h["Authorization"] == "Bearer tok"


def test_base_headers_account_token_wins():
    h = QP.base_headers(waf_cookies="acw_tc=a; token=OLD", bearer="x")
    assert h["Cookie"] == "acw_tc=a; token=x"
    assert "OLD" not in h["Cookie"]
    assert h["Authorization"] == "Bearer x"


def test_qwen_headers_latin1_safe():
    for h in (QP.base_headers(),
              QP.base_headers(user_agent="ua", waf_cookies="c=d", bearer="tok")):
        for k, v in h.items():
            v.encode("latin-1")  # raises if a value has non-latin-1 chars (curl_cffi hard constraint)


def test_bridge_headers_fallback(monkeypatch):
    monkeypatch.setattr("deepseaport.core.obscura_bridge.get_default_bridge",
                        lambda: None)
    bh = QP.bridge_headers(bearer="t")
    base = QP.base_headers(bearer="t")
    # X-Request-Id is regenerated per call, so compare the rest of the dict.
    bh.pop("X-Request-Id", None)
    base.pop("X-Request-Id", None)
    assert bh == base


def test_chat_new_payload():
    p = QP.chat_new_payload("qwen3.7-plus")
    assert p["chatId"] == ""
    assert p["models"] == ["qwen3.7-plus"]
    assert isinstance(p["timestamp"], int)
    assert p["chat_type"] == "t2t"
    assert p["chat_mode"] == "normal"
    assert p["project_id"] is None


def test_completion_payload_shape():
    p = QP.completion_payload("chat-1", "hello", "qwen3.7-plus",
                              thinking=True, search=False)
    assert p["stream"] is True
    assert p["incremental_output"] is True
    assert p["version"] == QP.COMPLETION_VERSION
    assert p["chatId"] == "chat-1"
    assert p["chat_id"] == "chat-1"
    assert p["parent_id"] is None
    assert p["chat_mode"] == "normal"
    assert p["model"] == "qwen3.7-plus"
    assert len(p["messages"]) == 1
    assert p["stream_options"] == {"include_usage": True}
    msg = p["messages"][0]
    assert msg["role"] == "user"
    assert msg["content"] == "hello"
    assert msg["user_action"] == "chat"
    assert "files" not in msg
    assert msg["models"] == []
    assert msg["model"] == ""
    assert msg["sub_chat_type"] == "t2t"
    assert msg["extra"] == {"meta": {"subChatType": "t2t"}}
    assert msg["chat_type"] == "t2t"
    fc = msg["feature_config"]
    assert fc["thinking_enabled"] is True
    assert fc["thinking_format"] == "summary"
    assert fc["auto_search"] is False


def test_completion_payload_thinking_flag():
    p = QP.completion_payload("c", "hi", "m", thinking=False, search=True)
    assert p["messages"][0]["feature_config"]["thinking_enabled"] is False


def test_completion_payload_thinking_off_drops_thinking_format():
    p = QP.completion_payload("c", "hi", "qwen3.7-plus",
                              thinking=False, search=True)
    fc = p["messages"][0]["feature_config"]
    assert "thinking_format" not in fc
    assert fc["auto_search"] is True


def test_completion_payload_omni_flash_no_optional_feature_keys():
    p = QP.completion_payload("c", "hi", "qwen3.8-omni-flash",
                              thinking=True, search=True)
    fc = p["messages"][0]["feature_config"]
    assert fc["thinking_enabled"] is True
    assert "thinking_format" not in fc
    assert "auto_search" not in fc


def test_is_waf_challenge():
    assert QP.is_waf_challenge(_FakeResp(200, content_type="text/html")) is True
    assert QP.is_waf_challenge(_FakeResp(200, content_type="application/json")) is False


def test_is_waf_body():
    assert QP.is_waf_body("aliyun_waf_aa block") is True
    assert QP.is_waf_body("AliyunCaptcha here") is True
    assert QP.is_waf_body("nc-container") is True
    assert QP.is_waf_body("plain json") is False


def test_extract_ban_timestamp_best_effort():
    assert QP.extract_ban_timestamp({}) is None
    assert QP.extract_ban_timestamp(None) is None
    # Numeric mute_until-style keys parse (s or ms); unknown shapes stay None.
    assert QP.extract_ban_timestamp({"mute_until": 123, "is_muted": 1}) == 123.0
    assert QP.extract_ban_timestamp({"data": {"ban_until": 1789555791}}) == 1789555791.0
    assert QP.extract_ban_timestamp({"data": {"mute_until": 1789555791000}}) == 1789555791.0
    assert QP.extract_ban_timestamp({"expires_at": 123}) is None  # token expiry, not a ban
    assert QP.extract_ban_timestamp({"mute_until": "soon"}) is None
    assert QP.extract_ban_timestamp({"mute_until": 0}) is None


def test_ban_error_text():
    assert QP.is_ban_error_text("user is muted") is True
    assert QP.is_ban_error_text("commute") is False
    assert QP.is_ban_error_text("transmute") is False


def test_detect_ban():
    assert QP.detect_ban({}) == (False, None)
    assert QP.detect_ban({"message": "ok"}) == (False, None)
    assert QP.detect_ban({"message": "user is muted"}) == (True, None)
    # Explicit flags count even without wording; Unauthorized stays clean.
    assert QP.detect_ban({"is_muted": 1}) == (True, None)
    assert QP.detect_ban({"data": {"code": "Unauthorized",
                                   "details": "401 Unauthorized"}}) == (False, None)
    assert QP.detect_ban({"message": "suspended",
                          "data": {"mute_until": 999}}) == (True, 999.0)
    assert QP.detect_ban(None) == (False, None)


def test_parse_json_ok():
    assert QP.parse_json(_FakeResp(200, json_data={"a": 1})) == {"a": 1}


def test_parse_json_exception_safe():
    class _Bad:
        def json(self):
            raise ValueError("nope")

    assert QP.parse_json(_Bad()) == {}


# sse_parser -----------------------------------------------------------------


def _evs(line: str):
    return list(QSP.parse_sse_line(line))


def test_parse_created_and_info_no_events():
    assert _evs('data: {"response.created":{"response_id":"r1"}}') == []
    assert _evs('data: {"response.info":{"response_id":"r1","action":"skip_think"}}') == []


def test_parse_stopped_no_events():
    assert _evs('data: {"response.stopped":{"response_id":"r1"}}') == []


def test_parse_content():
    line = 'data: {"choices":[{"delta":{"phase":"answer","content":"Hello"}}]}'
    assert [(e.kind, e.text) for e in _evs(line)] == [("content", "Hello")]


def test_parse_thinking_phases():
    line = 'data: {"choices":[{"delta":{"phase":"think","content":"hmm"}}]}'
    assert [(e.kind, e.text) for e in _evs(line)] == [("thinking", "hmm")]
    line = 'data: {"choices":[{"delta":{"phase":"think_summary","content":"sum"}}]}'
    assert [(e.kind, e.text) for e in _evs(line)] == [("thinking", "sum")]


def test_parse_keepalive_ignored():
    line = 'data: {"choices":[{"delta":{"phase":"KeepAlive","content":"x"}}]}'
    assert _evs(line) == []


def test_parse_tool_call_surfaced_as_content():
    line = 'data: {"choices":[{"delta":{"phase":"tool_call","content":"{\\"name\\":\\"f\\"}"}}]}'
    assert [(e.kind, e.text) for e in _evs(line)] == [("content", '{"name":"f"}')]


def test_parse_usage_event():
    # completion_tokens is preferred over total_tokens: build_usage feeds this
    # number into completion_tokens, so preferring the (input+output) total
    # would charge the prompt as output.
    line = ('data: {"choices":[{"delta":{"content":"a"}}],'
            ' "usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}')
    assert [(e.kind, e.text) for e in _evs(line)] == [("content", "a"), ("usage", "2")]


def test_parse_status_finished_no_content():
    line = 'data: {"choices":[{"delta":{"status":"finished","content":""}}]}'
    assert [(e.kind, e.text) for e in _evs(line)] == [("finished", "")]


def test_parse_finished_chunk_still_emits_chunk_usage():
    # Pre-change: the finished branch yielded and RETURNED, aborting the whole
    # parse and skipping the chunk-level usage block -> the run reported no
    # usage at all for a stream whose only usage rode on the finished chunk.
    line = ('data: {"choices":[{"delta":{"status":"finished","content":""}}],'
            '"usage":{"input_tokens":10,"output_tokens":20,"total_tokens":30}}')
    assert [(e.kind, e.text) for e in _evs(line)] == [("usage", "20"), ("finished", "")]
    # Usage must come FIRST: core.engine.drain_attempt / transport.pump_sse
    # stop reading at the finished event, so a trailing usage event is lost.


def test_parse_finished_chunk_emits_per_choice_usage():
    # usage documented inside the choice object (module docstring); the parser
    # only ever read the chunk-level block.
    line = ('data: {"choices":[{"delta":{"status":"finished","content":""},'
            '"usage":{"input_tokens":10,"output_tokens":20,"total_tokens":30}}]}')
    assert [(e.kind, e.text) for e in _evs(line)] == [("usage", "20"), ("finished", "")]


def test_parse_multi_choice_usage_is_summed():
    line = ('data: {"choices":[{"delta":{"content":"a"},'
            '"usage":{"output_tokens":5,"total_tokens":5}},'
            '{"delta":{"content":"b"},"usage":{"output_tokens":7,"total_tokens":7}}]}')
    assert [(e.kind, e.text) for e in _evs(line)] == [
        ("content", "a"), ("content", "b"), ("usage", "12")]


def test_parse_chunk_usage_wins_over_per_choice():
    line = ('data: {"choices":[{"delta":{"content":"a"},'
            '"usage":{"output_tokens":5}}],'
            '"usage":{"output_tokens":9,"total_tokens":11}}')
    assert [(e.kind, e.text) for e in _evs(line)] == [("content", "a"), ("usage", "9")]


def test_parse_tail_content_after_finished_delta_not_lost():
    # Same-chunk tail choice: the finished delta must not abort the loop, or
    # the trailing content choice disappears from the assembled answer.
    line = ('data: {"choices":[{"delta":{"status":"finished","content":""}},'
            '{"delta":{"phase":"answer","content":"tail"}}]}')
    assert [(e.kind, e.text) for e in _evs(line)] == [
        ("content", "tail"), ("finished", "")]
    # Next line after the finished line is still surfaced (stateless parser).
    parser = QP.QwenStreamParser()
    assert [e.kind for e in parser.feed(
        'data: {"choices":[{"delta":{"status":"finished"}}]}')] == ["finished"]
    assert [(e.kind, e.text) for e in parser.feed(
        'data: {"choices":[{"delta":{"phase":"answer","content":"tail"}}]}')] == [
        ("content", "tail")]


def test_parse_thinking_after_finished_delta_not_lost():
    line = ('data: {"choices":[{"delta":{"status":"finished"}},'
            '{"delta":{"phase":"think","content":"hmm"}}]}')
    assert [(e.kind, e.text) for e in _evs(line)] == [
        ("thinking", "hmm"), ("finished", "")]


def test_parse_error_envelope():
    evs = _evs('data: {"error":{"message":"boom","code":"x1"}}')
    assert len(evs) == 1 and evs[0].kind == "error"
    assert evs[0].code == "x1"
    assert evs[0].message == "boom"


def test_parse_error_details_key():
    evs = _evs('data: {"error":{"code":"quota_limit","details":"busy"}}')
    assert len(evs) == 1 and evs[0].kind == "error"
    assert evs[0].code == "quota_limit"
    assert evs[0].message == "busy"


def test_parse_usage_live_keys():
    # output_tokens is the completion count -> preferred over the (input+output)
    # total_tokens. build_usage counts this as completion_tokens.
    line = 'data: {"usage":{"input_tokens":5,"output_tokens":7,"total_tokens":12}}'
    assert [(e.kind, e.text) for e in _evs(line)] == [("usage", "7")]


def test_parse_usage_output_tokens_fallback():
    line = 'data: {"usage":{"input_tokens":5,"output_tokens":7}}'
    assert [(e.kind, e.text) for e in _evs(line)] == [("usage", "7")]


def test_parse_usage_total_tokens_last_resort():
    # No output/completion key: total_tokens is better than nothing.
    line = 'data: {"usage":{"input_tokens":5,"total_tokens":12}}'
    assert [(e.kind, e.text) for e in _evs(line)] == [("usage", "12")]


def test_parse_usage_unusable_output_falls_through():
    # A null/garbage output_tokens must not swallow the usable total.
    line = 'data: {"usage":{"input_tokens":5,"output_tokens":null,"total_tokens":12}}'
    assert [(e.kind, e.text) for e in _evs(line)] == [("usage", "12")]
    line = 'data: {"usage":{"output_tokens":"n/a","total_tokens":"12"}}'
    assert [(e.kind, e.text) for e in _evs(line)] == [("usage", "12")]


def test_parse_usage_empty_object_no_event():
    assert _evs('data: {"usage":{}}') == []
    assert _evs('data: {"usage":"nope"}') == []


def test_parse_done_and_done_marker():
    assert [e.kind for e in _evs('data: {"done":true}')] == ["finished"]
    assert [e.kind for e in _evs('data: [DONE]')] == ["finished"]


def test_parse_garbage_no_events():
    assert _evs("data: not-json") == []
    assert _evs('data: {"unknown_key":1}') == []


def test_stream_parser_feed_matches_parse():
    parser = QP.QwenStreamParser()
    evs = list(parser.feed('data: {"choices":[{"delta":{"phase":"answer","content":"Hi"}}]}'))
    assert [(e.kind, e.text) for e in evs] == [("content", "Hi")]
    assert [e.kind for e in parser.feed('data: {"done":true}')] == ["finished"]
    assert list(parser.feed('data: {"response.created":{"response_id":"r1"}}')) == []


# client ---------------------------------------------------------------------


def test_create_chat_success(monkeypatch):
    resp = _FakeResp(200, json_data={"success": True, "data": {"id": "chat-1"}})
    _sess(monkeypatch, post=resp)
    assert QC.create_chat({"h": "1"}, "qwen3.7-plus") == "chat-1"
    assert resp.closed is True


def test_create_chat_waf_challenge(monkeypatch):
    resp = _FakeResp(200, content_type="text/html")
    _sess(monkeypatch, post=resp)
    with pytest.raises(RuntimeError, match="WAF.*challenge"):
        QC.create_chat({"h": "1"}, "qwen3.7-plus")


def test_create_chat_biz_error(monkeypatch):
    data = {"success": False,
            "data": {"code": "Unauthorized", "details": "401 Unauthorized"}}
    _sess(monkeypatch, post=_FakeResp(200, json_data=data))
    with pytest.raises(RuntimeError, match="create_chat"):
        QC.create_chat({"h": "1"}, "qwen3.7-plus")


def test_create_chat_no_id(monkeypatch):
    _sess(monkeypatch, post=_FakeResp(200, json_data={"success": True, "data": {}}))
    with pytest.raises(RuntimeError, match="no data.id"):
        QC.create_chat({"h": "1"}, "qwen3.7-plus")


def test_delete_chat_ok(monkeypatch):
    resp = _FakeResp(204, text="", content_type="text/plain")
    _sess(monkeypatch, delete=resp)
    QC.delete_chat({"h": "1"}, "chat-1")  # must not raise


def test_delete_chat_swallows_exception(monkeypatch):
    _sess(monkeypatch, delete_exc=RuntimeError("net down"))
    QC.delete_chat({"h": "1"}, "chat-1")  # must not raise


def test_stream_completion_waf_html(monkeypatch):
    resp = _FakeResp(200, content_type="text/html", lines=[])
    _sess(monkeypatch, post=resp)
    events = list(QC.stream_completion({"h": "1"}, {"prompt": "hi"}))
    assert len(events) == 1 and events[0].kind == "error"
    assert events[0].code == "WAF_CHALLENGE"
    assert resp.closed is True


def test_stream_completion_sse_content_done(monkeypatch):
    resp = _FakeResp(200, content_type="text/event-stream", lines=[
        'data: {"choices":[{"delta":{"phase":"answer","content":"Hello"}}]}',
        'data: {"done":true}',
    ])
    _sess(monkeypatch, post=resp)
    events = list(QC.stream_completion({"h": "1"}, {"prompt": "hi"}))
    assert [(e.kind, e.text) for e in events] == [("content", "Hello"),
                                                  ("finished", "")]
    assert resp.closed is True


def test_stream_completion_sse_error_stops(monkeypatch):
    resp = _FakeResp(200, content_type="text/event-stream", lines=[
        'data: {"error":{"message":"boom","code":"x1"}}',
        'data: {"choices":[{"delta":{"phase":"answer","content":"late"}}]}',
    ])
    _sess(monkeypatch, post=resp)
    events = list(QC.stream_completion({"h": "1"}, {"prompt": "hi"}))
    assert len(events) == 1 and events[0].kind == "error"
    assert events[0].code == "x1"
    assert resp.closed is True


def test_stream_completion_422_detail(monkeypatch):
    body = '{"detail":[{"loc":["body","prompt"],"msg":"field required"}]}'
    resp = _FakeResp(422, text=body, content_type="application/json", lines=[])
    _sess(monkeypatch, post=resp)
    events = list(QC.stream_completion({"h": "1"}, {"prompt": "hi"}))
    assert events[0].code == "HTTP_422"
    assert "detail" in events[0].message
    assert "prompt" in events[0].message
    assert resp.closed is True


def test_stream_completion_200_json_success_false(monkeypatch):
    line = json.dumps({"success": False,
                       "data": {"code": "Unauthorized", "details": "401"}})
    resp = _FakeResp(200, content_type="application/json", lines=[line])
    _sess(monkeypatch, post=resp)
    events = list(QC.stream_completion({"h": "1"}, {"prompt": "hi"}))
    assert len(events) == 1 and events[0].kind == "error"
    assert events[0].code == "Unauthorized"
    assert "401" in events[0].message
    assert resp.closed is True


def test_stream_completion_empty_200_body_is_upstream_error(monkeypatch):
    # Pre-change: with nothing to yield (no WAF body, no envelope, empty
    # body) the code closed the response and fell through to
    # T.pump_sse, which iterated the drained stream a SECOND time -> parked
    # forever, holding the worker and the account slot (reproduced against
    # curl_cffi 0.16.3; _DrainedResp raises where curl_cffi blocks).
    resp = _DrainedResp(200, text="", content_type="application/json",
                        lines=[])
    _sess(monkeypatch, post=resp)
    events = list(QC.stream_completion({"h": "1"}, {"prompt": "hi"}))
    assert len(events) == 1
    assert events[0].kind == "error"
    assert events[0].code == "UPSTREAM_ERROR"
    assert "empty" in events[0].message
    assert resp.iter_lines_calls == 1, "the SSE pump re-entered the drained body"
    assert resp.closed is True


# 403 mapping / ban classification (Qwen's WAF is 200 + text/html, never 403) ---


_SUSPENDED_BODY = ("Due to violation of user policies, your account has been "
                   "suspended")


def test_stream_completion_403_is_not_labelled_waf(monkeypatch):
    # Pre-change: the shared transport mapped every 403 to HTTP_403_WAF, so a
    # real Qwen 403 (a suspension body) was classified as WAF -> re-primes and
    # a pointless 502, no cooldown, no failover.
    resp = _FakeResp(403, text=_SUSPENDED_BODY, content_type="application/json",
                     lines=[])
    _sess(monkeypatch, post=resp)
    events = list(QC.stream_completion({"h": "1"}, {"prompt": "hi"}))
    assert len(events) == 1
    assert events[0].kind == "error"
    assert events[0].code == "HTTP_403"
    assert events[0].code != "HTTP_403_WAF"
    assert "suspended" in events[0].message
    assert resp.closed is True


def test_qwen_403_suspension_body_classifies_as_ban():
    # End to end through the string the retry/terminal tables see.
    err = f"HTTP_403: {_SUSPENDED_BODY}"
    assert QCS.classify_error(err) is QCS.RetryKind.BAN
    # ... and the WAF retry branch must not claim it.
    assert QCS.classify_error(err) is not QCS.RetryKind.WAF
    # Terminal ladder: BAN is registered, so the account is failed over.
    assert QCS._TERMINAL_HANDLERS[QCS.RetryKind.BAN] is QCS._terminal_ban


def test_qwen_ban_checked_before_waf_families():
    assert QCS.classify_error("HTTP_403: user is muted") is QCS.RetryKind.BAN
    assert QCS.classify_error("HTTP_401: account suspended") is QCS.RetryKind.BAN
    assert QCS.classify_error("restricted account") is QCS.RetryKind.BAN
    # No ban wording -> the WAF / 401 / rate-limit families are unchanged.
    assert QCS.classify_error(
        "HTTP_403_WAF: blocked by WAF") is QCS.RetryKind.WAF
    assert QCS.classify_error(
        "WAF_CHALLENGE: Qwen completions returned the Aliyun WAF challenge page "
        "(200 text/html) — warm up the bridge and retry") is QCS.RetryKind.WAF
    assert QCS.classify_error("aliyun_waf_aa challenge") is QCS.RetryKind.WAF
    assert QCS.classify_error("HTTP_401: Unauthorized") is QCS.RetryKind.INVALID_TOKEN
    assert QCS.classify_error(
        "quota_limit: high demand") is QCS.RetryKind.RATE_LIMIT
    assert QCS.classify_error("HTTP_422: bad field") is QCS.RetryKind.UPSTREAM_INVALID
    assert QCS.classify_error("upstream timed out") is QCS.RetryKind.OVERLOAD
    assert QCS.classify_error("weird boom") is QCS.RetryKind.UPSTREAM


def test_qwen_200_text_html_waf_challenge_still_waf(monkeypatch):
    # The real Qwen WAF shape (200 + HTML) must keep its own detection path.
    resp = _FakeResp(200, content_type="text/html", lines=[])
    _sess(monkeypatch, post=resp)
    events = list(QC.stream_completion({"h": "1"}, {"prompt": "hi"}))
    assert [e.code for e in events] == ["WAF_CHALLENGE"]
    assert QCS.classify_error(events[0].code + ": " + events[0].message) is (
        QCS.RetryKind.WAF)
    # Body markers on a drained (non-HTML-content-type) 200 line.
    resp2 = _FakeResp(200, content_type="text/plain", lines=[
        "<html>aliyun_waf_aa</html>", 'data: {"choices":[]}'])
    _sess(monkeypatch, post=resp2)
    events2 = list(QC.stream_completion({"h": "1"}, {"prompt": "hi"}))
    assert [e.code for e in events2] == ["WAF_CHALLENGE"]


def test_with_chat_id_unit():
    base = QP.CHAT_COMPLETIONS_URL
    assert QC._with_chat_id(base, {"chat_id": "a b/c"}) == f"{base}?chat_id=a%20b%2Fc"
    assert QC._with_chat_id(base, {"chatId": "x"}) == f"{base}?chat_id=x"
    assert QC._with_chat_id(base, {"prompt": "x"}) == base
    assert QC._with_chat_id(base, None) == base


def test_stream_completion_chat_id_query(monkeypatch):
    resp = _FakeResp(200, content_type="text/event-stream", lines=[
        'data: {"choices":[{"delta":{"status":"finished"}}]}',
    ])
    calls = _recording_post(monkeypatch, resp)
    events = list(QC.stream_completion({"h": "1"}, {"prompt": "hi", "chat_id": "chat-9"}))
    assert calls == [f"{QP.CHAT_COMPLETIONS_URL}?chat_id=chat-9"]
    assert [e.kind for e in events] == ["finished"]
    assert resp.closed is True


def test_stream_completion_bare_url_without_chat_id(monkeypatch):
    resp = _FakeResp(200, content_type="text/event-stream",
                     lines=['data: {"done":true}'])
    calls = _recording_post(monkeypatch, resp)
    events = list(QC.stream_completion({"h": "1"}, {"prompt": "hi"}))
    assert calls == [QP.CHAT_COMPLETIONS_URL]
    assert [e.kind for e in events] == ["finished"]


def test_signin_success(monkeypatch):
    user = {"id": "u1", "email": "e@x.com", "name": "n", "role": "user",
            "token": "jwt", "token_type": "Bearer", "expires_at": 123}
    resp = _FakeResp(200, json_data=dict(user))
    _sess(monkeypatch, post=resp)
    assert QC.signin("e@x.com", "pw") == user
    assert resp.closed is True


def test_signin_invalid_cred(monkeypatch):
    data = {"detail": {"code": "INVALID_CRED", "details": "bad"}}
    _sess(monkeypatch, post=_FakeResp(400, json_data=data))
    with pytest.raises(RuntimeError, match="INVALID_CRED"):
        QC.signin("e@x.com", "pw")


def test_signin_waf_challenge(monkeypatch):
    resp = _FakeResp(200, content_type="text/html")
    _sess(monkeypatch, post=resp)
    with pytest.raises(RuntimeError, match="WAF"):
        QC.signin("e@x.com", "pw")


def test_check_ban_success_true(monkeypatch):
    _sess(monkeypatch, get=_FakeResp(200, json_data={"success": True}))
    assert QC.check_ban({"h": "1"}) == (False, None)


def test_check_ban_success_false(monkeypatch):
    _sess(monkeypatch, get=_FakeResp(200, json_data={
        "success": False, "data": {"code": "Unauthorized"}}))
    assert QC.check_ban({"h": "1"}) == (False, None)


def test_check_ban_403_html(monkeypatch):
    _sess(monkeypatch, get=_FakeResp(403, content_type="text/html"))
    assert QC.check_ban({"h": "1"}) == (False, None)


def test_check_ban_exception(monkeypatch):
    _sess(monkeypatch, get_exc=RuntimeError("net down"))
    assert QC.check_ban({"h": "1"}) == (False, None)


def test_fetch_models_list(monkeypatch):
    data = {"success": True,
            "data": {"data": [{"id": "qwen3.7-plus"}, {"id": "qwen3.8-max"}]}}
    _sess(monkeypatch, get=_FakeResp(200, json_data=data))
    assert [m["id"] for m in QC.fetch_models({"h": "1"})] == \
        ["qwen3.7-plus", "qwen3.8-max"]


def test_fetch_models_error(monkeypatch):
    _sess(monkeypatch, get=_FakeResp(200, json_data={
        "success": False, "data": {"code": "Unauthorized"}}))
    assert QC.fetch_models({"h": "1"}) == []


def test_fetch_models_exception(monkeypatch):
    _sess(monkeypatch, get_exc=RuntimeError("net down"))
    assert QC.fetch_models({"h": "1"}) == []


# provider -------------------------------------------------------------------


def test_provider_models_and_default():
    p = QProv.QwenProvider()
    models = p.models()
    assert set(models) == {"qwen3.7-plus", "qwen3.8-max", "qwen3.8-omni-flash"}
    assert models["qwen3.7-plus"].thinking is True
    assert models["qwen3.7-plus"].search is True
    assert models["qwen3.8-max"].thinking is True
    assert models["qwen3.8-max"].search is True
    assert models["qwen3.8-omni-flash"].thinking is True
    assert models["qwen3.8-omni-flash"].search is False
    assert p.default_model() == "qwen3.7-plus"


def test_provider_normalize_token():
    p = QProv.QwenProvider()
    assert p.normalize_token('{"value":"tok64"}') == "tok64"
    assert p.normalize_token("tok64") == "tok64"


def test_provider_probe_ban_never_raises(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("net down")

    monkeypatch.setattr(QC, "check_ban", _boom)
    assert QProv.QwenProvider().probe_ban({}, timeout=5) == (False, None)


def test_provider_probe_ban_delegates(monkeypatch):
    monkeypatch.setattr(QC, "check_ban", lambda *a, **k: (True, 5.0))
    assert QProv.QwenProvider().probe_ban({}, timeout=5) == (True, 5.0)


def test_provider_login_delegates(monkeypatch):
    monkeypatch.setattr(QA, "login", lambda *a, **k: ("tok-x", False, ""))
    assert QProv.QwenProvider().login("e", "p", None) == ("tok-x", False, "")


def test_provider_warmup_noop():
    assert QProv.QwenProvider().warmup(None) is None


def test_provider_complete_delegates(monkeypatch):
    result = {"content": "ok", "thinking": "", "usage_total": 1}
    monkeypatch.setattr(QCS, "_complete_with_failover_sync",
                        lambda *a, **k: result)
    p = QProv.QwenProvider()
    assert p.complete(None, None, {}, None) == result
    assert p.complete_request(None, None, {}, None, None) == result


def test_create_provider_factory():
    assert isinstance(QProv.create_provider(), QProv.QwenProvider)


# auth -----------------------------------------------------------------------


def test_auth_login_success(monkeypatch):
    monkeypatch.setattr(QC, "signin",
                        lambda *a, **k: {"id": "u", "token": "jwt",
                                         "token_type": "Bearer"})
    assert QA.login("e@x.com", "pw") == ("jwt", False, "")


def test_auth_login_invalid_cred(monkeypatch):
    def _bad(*a, **k):
        raise RuntimeError("Qwen signin failed INVALID_CRED: bad")

    monkeypatch.setattr(QC, "signin", _bad)
    token, banned, detail = QA.login("e@x.com", "pw")
    assert token is None
    assert banned is False
    assert "INVALID_CRED" in detail


def test_auth_login_waf(monkeypatch):
    msg = "WAF_CHALLENGE: signin returned the Aliyun challenge page"

    def _waf(*a, **k):
        raise RuntimeError(msg)

    monkeypatch.setattr(QC, "signin", _waf)
    assert QA.login("e@x.com", "pw") == (None, False, msg)


def test_auth_login_missing_credentials():
    # No network: returns before touching QC.signin.
    assert QA.login("", "") == (None, False, "email and password are required")


# engine ---------------------------------------------------------------------


def test_classify_error_quota_limit():
    assert QCS.classify_error("quota_limit: high demand") == QCS.RetryKind.RATE_LIMIT


# waf_fix (offline, monkeypatch _SESSION; no network) ------------------------


def test_cookie_names_only_no_values():
    assert QP.cookie_names("acw_tc=SECRET; token=JWT; x=1") == ["acw_tc", "token", "x"]
    assert QP.cookie_names("") == []
    desc = QP.describe_cookies({"Cookie": "acw_tc=SECRET; token=JWT"})
    assert desc["names"] == ["acw_tc", "token"]
    assert desc["has_token"] is True
    assert desc["has_acw_tc"] is True
    assert "SECRET" not in str(desc["names"])
    assert "JWT" not in str(desc["names"])
    desc2 = QP.describe_cookies({"Cookie": "a=1"})
    assert desc2["has_token"] is False
    assert desc2["has_acw_tc"] is False


def test_bridge_headers_explicit_bridge_wins_over_default(monkeypatch):
    """The explicit bridge supplies the UA; NO bridge supplies cookies.

    This test used to pin the leak: the app bridge's jar (the DeepSeek-host
    WAF jar) was merged into the Cookie header and shipped to chat.qwen.ai.
    """
    class _State:
        def __init__(self, ua):
            self.user_agent = ua

    class _B:
        def __init__(self, ua, cookies):
            self.state = _State(ua)
            self._c = cookies

        def cookie_header(self):
            return self._c

    monkeypatch.setattr("deepseaport.core.obscura_bridge.get_default_bridge",
                        lambda: _B("DEFAULT-UA", "acw_tc=default; token=OLDDEFAULT"))
    app_bridge = _B("APP-UA", "aws-waf-token=DSJAR")
    h = QP.bridge_headers(bearer="jwt123", bridge=app_bridge)
    assert h["User-Agent"] == "APP-UA"
    assert h["Cookie"] == "token=jwt123"
    assert "aws-waf-token" not in h["Cookie"]
    assert "DEFAULT" not in str(h)
    # explicit None forces fallback (no bridge UA, token still sent)
    h2 = QP.bridge_headers(bearer="jwt123", bridge=None)
    assert h2["Cookie"] == "token=jwt123"


def test_headers_prefers_app_bridge(monkeypatch):
    """The app bridge's UA wins over the default; its jar never ships."""
    QC._clear_waf_cache()
    try:
        class _State:
            def __init__(self, ua):
                self.user_agent = ua

        class _B:
            def __init__(self, ua, cookies):
                self.state = _State(ua)
                self._c = cookies

            def cookie_header(self):
                return self._c

        class _St:
            bridge = _B("APP-UA", "aws-waf-token=ds-jar")

        class _App:
            state = _St()

        monkeypatch.setattr("deepseaport.core.obscura_bridge.get_default_bridge",
                            lambda: _B("DEFAULT-UA", "aws-waf-token=default-jar"))
        h = QCS._headers(_App(), "jwt999")
        assert h["User-Agent"] == "APP-UA"
        assert "aws-waf-token" not in str(h)
        assert h["Cookie"] == "token=jwt999"
        assert "DEFAULT" not in str(h)
    finally:
        QC._clear_waf_cache()


def test_ensure_token_cookie_merges_and_wins():
    h = {"Authorization": "Bearer NEWJWT", "Cookie": "acw_tc=a; token=OLD"}
    QC._ensure_token_cookie(h)
    assert h["Cookie"] == "acw_tc=a; token=NEWJWT"
    assert "OLD" not in h["Cookie"]
    h2 = {"Authorization": "Bearer ONLYJWT"}
    QC._ensure_token_cookie(h2)
    assert h2["Cookie"] == "token=ONLYJWT"


def test_create_chat_ensures_token_cookie(monkeypatch):
    seen = {}

    class _S:
        def post(self, url, *a, **k):
            seen.update(k.get("headers") or {})
            return _FakeResp(200, json_data={"success": True, "data": {"id": "c1"}})

    monkeypatch.setattr(QC, "_SESSION", _S())
    headers = {"Authorization": "Bearer jwtABC", "Cookie": "acw_tc=a"}
    assert QC.create_chat(headers, "qwen3.7-plus") == "c1"
    assert "token=jwtABC" in headers["Cookie"]
    assert "token=jwtABC" in seen["Cookie"]


def test_stream_completion_ensures_token_cookie(monkeypatch):
    seen = {}

    class _S:
        def post(self, url, *a, **k):
            seen.update(k.get("headers") or {})
            return _FakeResp(200, content_type="text/event-stream",
                             lines=['data: {"done":true}'])

    monkeypatch.setattr(QC, "_SESSION", _S())
    headers = {"Authorization": "Bearer jwtXYZ"}
    events = list(QC.stream_completion(headers, {"prompt": "hi"}))
    assert [e.kind for e in events] == ["finished"]
    assert headers.get("Cookie") == "token=jwtXYZ"
    assert seen.get("Cookie") == "token=jwtXYZ"


def test_prime_waf_gets_models_url(monkeypatch):
    calls = []

    class _S:
        def get(self, url, *a, **k):
            calls.append((url, dict(k.get("headers") or {})))
            return _FakeResp(200, json_data={"success": True,
                                             "data": {"data": [{"id": "qwen3.7-plus"}]}})

    monkeypatch.setattr(QC, "_SESSION", _S())
    assert QC.prime_waf({"Cookie": "token=t"}) is True
    assert calls and calls[0][0] == QP.MODELS_URL


def test_prime_waf_false_on_envelope_miss(monkeypatch):
    _sess(monkeypatch, get=_FakeResp(200, json_data={"success": False}))
    assert QC.prime_waf({"h": "1"}) is False


def test_provider_warmup_primes_models_url(monkeypatch):
    calls = []

    class _S:
        def get(self, url, *a, **k):
            calls.append(url)
            return _FakeResp(200, json_data={"success": True,
                                             "data": {"data": [{"id": "qwen3.7-plus"}]}})

    monkeypatch.setattr(QC, "_SESSION", _S())

    class _State:
        user_agent = "WARM-UA"

    class _Bridge:
        state = _State()

        def cookie_header(self):
            return "acw_tc=warm"

    assert QProv.QwenProvider().warmup(None) is None
    assert QProv.QwenProvider().warmup(_Bridge()) is None
    assert calls == [QP.MODELS_URL]


def test_waf_log_names_only_no_values(monkeypatch, caplog):
    import logging
    resp = _FakeResp(200, content_type="text/html")
    _sess(monkeypatch, post=resp)
    headers = {"Authorization": "Bearer SECRETJWT",
               "Cookie": "acw_tc=SECRETACW; token=SECRETJWT"}
    with caplog.at_level(logging.WARNING, logger="deepseaport.client"):
        with pytest.raises(RuntimeError, match="WAF"):
            QC.create_chat(dict(headers), "qwen3.7-plus")
    names_hit = any("acw_tc" in r.message and "token" in r.message
                    for r in caplog.records)
    assert names_hit
    assert "SECRETJWT" not in caplog.text
    assert "SECRETACW" not in caplog.text


def test_overload_rebuilds_headers_no_stale(monkeypatch):
    # First attempt overloads, retry must use fresh (not stale) headers.
    from deepseaport.config import AccountConfig
    from deepseaport.accounts import PooledAccount
    from deepseaport.providers.qwen.sse_parser import StreamEvent

    calls = {"n_headers": 0}

    def _fake_headers(app, token):
        calls["n_headers"] += 1
        return {"Authorization": f"Bearer {token}",
                "Cookie": f"token={token}; gen={calls['n_headers']}"}

    monkeypatch.setattr(QCS, "_headers", _fake_headers)
    monkeypatch.setattr(QC, "create_chat", lambda h, m, **k: "chat-1")
    seen_cookies = []

    def _fake_stream(h, p, **k):
        seen_cookies.append(h.get("Cookie"))
        if len(seen_cookies) == 1:
            yield StreamEvent(kind="error", code="overload",
                              message="server busy")
            return
        yield StreamEvent(kind="content", text="ok")
        yield StreamEvent(kind="finished", text="")

    monkeypatch.setattr(QC, "stream_completion", _fake_stream)
    monkeypatch.setattr(QCS, "_delete_chat_bg", lambda *a, **k: None)

    class _FakeApp:
        class state:
            bridge = None
            settings = None

    cfg = AccountConfig(email="e@x.com", password="p", token="tok123")
    item = PooledAccount(cfg=cfg)
    prep = {"model": "qwen3.7-plus", "prompt": "hi",
            "thinking": False, "search": False}
    out = QCS._run_completion_core(_FakeApp(), item, prep)
    assert out["content"] == "ok"
    assert len(seen_cookies) == 2
    assert seen_cookies[0] != seen_cookies[1]
    assert "gen=2" in seen_cookies[1]


# qwen WAF fix (logs: token without acw_tc, WinError5 deepseek host) --------


def _winerror5(msg="Access is denied"):
    exc = OSError(13, msg)
    try:
        exc.winerror = 5  # type: ignore[attr-defined]
    except Exception:
        pass
    return exc


def test_prime_targets_qwen_host_not_deepseek(monkeypatch):
    QC._clear_waf_cache()
    try:
        calls = []

        class _S:
            def get(self, url, *a, **k):
                calls.append(url)
                return _FakeResp(200, json_data={"success": True,
                                                 "data": {"data": []}})

        monkeypatch.setattr(QC, "_SESSION", _S())
        assert QC.prime_waf({"Cookie": "token=t"}) is True
        assert calls == [QP.MODELS_URL]
        assert "chat.qwen.ai" in calls[0]
        assert "deepseek" not in calls[0].lower()
    finally:
        QC._clear_waf_cache()


def test_prime_harvests_acw_tc_despite_discard(monkeypatch):
    QC._clear_waf_cache()
    try:
        resp = _FakeResp(200, json_data={"success": True,
                                         "data": {"data": [{"id": "qwen3.7-plus"}]}})
        resp.headers["Set-Cookie"] = "acw_tc=HARVEST123; Path=/; HttpOnly"
        _sess(monkeypatch, get=resp)
        assert QC.qwen_waf_cookie_header() == ""
        assert QC.prime_waf({"Cookie": "token=t"}) is True
        assert QC.qwen_waf_cookie_header() == "acw_tc=HARVEST123"
        assert QC._merged_waf_cookies("") == "acw_tc=HARVEST123"
        assert QC._merged_waf_cookies("x=1") == "x=1; acw_tc=HARVEST123"
        # No dup when already present.
        assert QC._merged_waf_cookies("acw_tc=OLD; x=1") == "acw_tc=OLD; x=1"
    finally:
        QC._clear_waf_cache()


def test_fetch_models_harvests_acw_tc(monkeypatch):
    QC._clear_waf_cache()
    try:
        resp = _FakeResp(200, json_data={"success": True,
                                         "data": {"data": [{"id": "qwen3.7-plus"}]}})
        resp.headers["set-cookie"] = "acw_tc=FETCH456; Path=/; Secure"
        _sess(monkeypatch, get=resp)
        models = QC.fetch_models({"h": "1"})
        assert [m["id"] for m in models] == ["qwen3.7-plus"]
        assert QC.qwen_waf_cookie_header() == "acw_tc=FETCH456"
    finally:
        QC._clear_waf_cache()


def test_is_winerror5_helper():
    assert QC._is_winerror5(_winerror5()) is True
    assert QC._is_winerror5(PermissionError("denied")) is True
    assert QC._is_winerror5(RuntimeError("net down")) is False
    assert QC._is_winerror5(ValueError("WinError 5 whatever")) is True


def test_prime_winerror5_graceful_names_only(monkeypatch, caplog):
    import logging
    QC._clear_waf_cache()
    try:
        def _raise_get(*a, **k):
            raise _winerror5("obscura profile dir Access is denied")

        class _S:
            get = staticmethod(_raise_get)

        monkeypatch.setattr(QC, "_SESSION", _S())
        with caplog.at_level(logging.WARNING, logger="deepseaport.client"):
            assert QC.prime_waf({"Cookie": "token=SECRETJWT"}) is False
        assert "WinError5" in caplog.text or "winerror5" in caplog.text.lower()
        assert "SECRETJWT" not in caplog.text
    finally:
        QC._clear_waf_cache()


def test_headers_include_cached_acw_tc_plus_token(monkeypatch):
    QC._clear_waf_cache()
    try:
        resp = _FakeResp(200, json_data={"success": True,
                                         "data": {"data": []}})
        resp.headers["Set-Cookie"] = "acw_tc=CACHED789; Path=/"
        QC._harvest_waf_cookies(resp)

        class _State:
            user_agent = "UA-X"

        class _Bridge:
            state = _State()

            def cookie_header(self):
                return "x=1"

        class _App:
            class state:
                bridge = _Bridge()

        h = QCS._headers(_App(), "jwtKEEP")
        assert h["Authorization"] == "Bearer jwtKEEP"
        assert "token=jwtKEEP" in h["Cookie"]
        assert "acw_tc=CACHED789" in h["Cookie"]
        assert "CACHED789" in h["Cookie"]  # value present in header itself
        # Bearer+token merge kept: stale token replaced.
        h2 = {"Authorization": "Bearer NEWJWT",
              "Cookie": "acw_tc=CACHED789; token=OLD"}
        QC._ensure_token_cookie(h2)
        assert h2["Cookie"] == "acw_tc=CACHED789; token=NEWJWT"
    finally:
        QC._clear_waf_cache()


def test_refresh_winerror5_primes_qwen_host(monkeypatch, caplog):
    import logging
    import threading
    from deepseaport.config import AccountConfig
    from deepseaport.accounts import PooledAccount
    QC._clear_waf_cache()
    try:
        prime_urls = []

        class _S:
            def get(self, url, *a, **k):
                prime_urls.append(url)
                r = _FakeResp(200, json_data={"success": True,
                                              "data": {"data": []}})
                r.headers["Set-Cookie"] = "acw_tc=REFRESH999; Path=/"
                return r

        monkeypatch.setattr(QC, "_SESSION", _S())

        class _State:
            user_agent = "UA-R"

        class _WinBridge:
            state = _State()

            def cookie_header(self):
                return "x=1"

            def warmup(self):
                raise _winerror5("profile dir Access is denied")

        class _App:
            class state:
                bridge = _WinBridge()
                settings = None

        cfg = AccountConfig(email="e@x.com", password="p", token="tokREFRESH")
        item = PooledAccount(cfg=cfg)
        with caplog.at_level(logging.WARNING):
            QCS._refresh_waf_without_account_lock(_App(), item)  # never raises
        assert prime_urls == [QP.MODELS_URL]
        assert "chat.qwen.ai" in prime_urls[0]
        assert QC.qwen_waf_cookie_header() == "acw_tc=REFRESH999"
        assert "tokREFRESH" not in caplog.text
        assert "REFRESH999" not in caplog.text
    finally:
        QC._clear_waf_cache()


def test_provider_warmup_harvests_and_targets_qwen(monkeypatch):
    QC._clear_waf_cache()
    try:
        calls = []

        class _S:
            def get(self, url, *a, **k):
                calls.append(url)
                r = _FakeResp(200, json_data={"success": True,
                                              "data": {"data": [{"id": "qwen3.7-plus"}]}})
                r.headers["Set-Cookie"] = "acw_tc=WARM111; Path=/"
                return r

        monkeypatch.setattr(QC, "_SESSION", _S())

        class _State:
            user_agent = "WARM-UA"

        class _Bridge:
            state = _State()

            def cookie_header(self):
                return "x=warm"

        assert QProv.QwenProvider().warmup(_Bridge()) is None
        assert calls == [QP.MODELS_URL]
        assert QC.qwen_waf_cookie_header() == "acw_tc=WARM111"
    finally:
        QC._clear_waf_cache()


def test_merged_full_jar_no_dup_no_token_overwrite():
    QC._clear_waf_cache()
    try:
        QC._store_waf_cookies({"acw_tc": "A", "x-ap": "eu", "cna": "C"})
        assert QC._merged_waf_cookies("") == "acw_tc=A; x-ap=eu; cna=C"
        assert QC._merged_waf_cookies("x=1") == "x=1; acw_tc=A; x-ap=eu; cna=C"
        # Existing names kept, token untouched, missing jar added.
        base = "acw_tc=OLD; token=JWT; x=1"
        merged = QC._merged_waf_cookies(base)
        assert merged == "acw_tc=OLD; token=JWT; x=1; x-ap=eu; cna=C"
        assert "token=JWT" in merged
    finally:
        QC._clear_waf_cache()


def test_prime_qwen_bridge_stores_solved_jar(monkeypatch):
    QC._clear_waf_cache()
    try:
        dump = ('[{"name":"acw_tc","value":"SOLVED64"},'
                '{"name":"x-ap","value":"eu-central-1"},'
                '{"name":"cna","value":"C1"}]')

        class _B:
            profile = "p"
            def _run(self, *a, **k):
                class _P:
                    stdout = dump
                return _P()
            def _parse_cookie_dump(self, payload):
                import json as _j
                return {c["name"]: c["value"] for c in _j.loads(payload)}

        monkeypatch.setattr("deepseaport.core.obscura_bridge.profile_lock",
                            lambda prof: __import__("contextlib").nullcontext())
        assert QC.prime_qwen_bridge(_B()) is True
        d = QC.qwen_waf_cookie_dict()
        assert d.get("acw_tc") == "SOLVED64"
        assert d.get("x-ap") == "eu-central-1"
        assert "token" not in d
    finally:
        QC._clear_waf_cache()


def test_prime_qwen_bridge_winerror5_graceful(monkeypatch):
    QC._clear_waf_cache()
    try:
        def _raise(*a, **k):
            e = OSError(13, "Access is denied")
            try:
                e.winerror = 5  # type: ignore[attr-defined]
            except Exception:
                pass
            raise e

        class _B:
            profile = "p"
            def _run(self, *a, **k):
                return _raise()

        monkeypatch.setattr("deepseaport.core.obscura_bridge.profile_lock",
                            lambda prof: __import__("contextlib").nullcontext())
        assert QC.prime_qwen_bridge(_B()) is False
    finally:
        QC._clear_waf_cache()


def test_no_bearer_env_omits_authorization(monkeypatch):
    monkeypatch.setenv("DEEPSEAPORT_QWEN_NO_BEARER", "1")
    h = QP.base_headers(bearer="jwtX", waf_cookies="acw_tc=a")
    assert h["Cookie"] == "acw_tc=a; token=jwtX"
    assert "Authorization" not in h
    monkeypatch.delenv("DEEPSEAPORT_QWEN_NO_BEARER", raising=False)
    h2 = QP.base_headers(bearer="jwtX", waf_cookies="acw_tc=a")
    assert h2["Authorization"] == "Bearer jwtX"


def test_stream_headers_carry_x_accel(monkeypatch):
    seen = {}

    class _S:
        def post(self, url, *a, **k):
            seen.update(k.get("headers") or {})
            return _FakeResp(200, content_type="text/event-stream",
                             lines=['data: {"done":true}'])

    monkeypatch.setattr(QC, "_SESSION", _S())
    headers = {"Authorization": "Bearer j", "Cookie": "token=j"}
    list(QC.stream_completion(headers, {"prompt": "hi"}))
    assert seen.get("X-Accel-Buffering") == "no"


# auth lifecycle: expires_at capture + expiry pre-check -----------------------


def _jwt(exp: int) -> str:
    import base64
    import json as _j

    def _b64(obj: dict) -> str:
        return base64.urlsafe_b64encode(
            _j.dumps(obj).encode()).decode().rstrip("=")

    return f"{_b64({'alg': 'none'})}.{_b64({'exp': exp})}.sig"


def test_login_with_expiry_captures_expires_at(monkeypatch):
    monkeypatch.setattr(QC, "signin",
                        lambda *a, **k: {"id": "u", "token": "jwt",
                                         "token_type": "Bearer",
                                         "expires_at": 1789555791})
    token, exp, banned, detail = QA.login_with_expiry("e@x.com", "pw")
    assert (token, exp, banned, detail) == ("jwt", 1789555791.0, False, "")


def test_login_with_expiry_jwt_fallback(monkeypatch):
    tok = _jwt(1789555791)
    monkeypatch.setattr(QC, "signin",
                        lambda *a, **k: {"id": "u", "token": tok})
    token, exp, banned, detail = QA.login_with_expiry("e@x.com", "pw")
    assert token == tok
    assert exp == 1789555791.0
    assert (banned, detail) == (False, "")


def test_login_stays_3_tuple(monkeypatch):
    monkeypatch.setattr(QC, "signin",
                        lambda *a, **k: {"id": "u", "token": "jwt",
                                         "expires_at": 123})
    assert QA.login("e@x.com", "pw") == ("jwt", False, "")


def test_token_expired_persisted_and_jwt():
    import time as _t
    from deepseaport.config import AccountConfig

    now = _t.time()
    assert QA.token_expired(AccountConfig(token="x", token_expires_at=now - 10)) is True
    assert QA.token_expired(AccountConfig(token="x",
                                          token_expires_at=now + 3600)) is False
    # Unknown persisted expiry falls back to the JWT exp claim.
    assert QA.token_expired(AccountConfig(token=_jwt(int(now - 10)))) is True
    assert QA.token_expired(AccountConfig(token=_jwt(int(now + 3600)))) is False
    assert QA.token_expired(AccountConfig(token="short")) is False
    # Leeway: an expiry inside the window already counts as expired.
    assert QA.is_expired(now + 60, now=now) is True
    assert QA.is_expired(now + 3600, now=now) is False


def test_unpack_login_result_ex_roundtrip():
    assert QA.unpack_login_result_ex(("tok", 999.0, False, "")) == ("tok", 999.0, False, "")
    assert QA.unpack_login_result(("tok", 999.0, False, "")) == ("tok", False, "")
    assert QA.unpack_login_result_ex(("tok", False, "")) == ("tok", None, False, "")


# ban probe: check_ban verdict -------------------------------------------------


def test_check_ban_claims_ban_with_timestamp(monkeypatch):
    data = {"success": False, "data": {"is_muted": 1, "mute_until": 1789555791,
                                       "details": "user is muted"}}
    _sess(monkeypatch, get=_FakeResp(200, json_data=data))
    assert QC.check_ban({"h": "1"}) == (True, 1789555791.0)


def test_check_ban_unauthorized_not_banned(monkeypatch):
    data = {"success": False, "data": {"code": "Unauthorized",
                                       "details": "401 Unauthorized"}}
    _sess(monkeypatch, get=_FakeResp(200, json_data=data))
    assert QC.check_ban({"h": "1"}) == (False, None)


def test_stream_error_event_carries_ban_until():
    evs = list(QSP.parse_sse_line(
        'data: {"error":{"code":"muted","details":"user is muted"}}'))
    assert len(evs) == 1 and evs[0].kind == "error"
    assert evs[0].ban_until is None  # no timestamp key in the envelope
    evs = list(QSP.parse_sse_line(
        'data: {"error":{"code":"quota_limit","details":"busy"}}'))
    assert evs[0].ban_until is None


# completion auto re-login -----------------------------------------------------


def _qwen_app():
    class _FakeApp:
        class state:
            bridge = None
            settings = None

    return _FakeApp()


def _qwen_prep():
    return {"model": "qwen3.7-plus", "prompt": "hi",
            "thinking": False, "search": False}


def test_expired_token_proactive_relogin(monkeypatch):
    import time as _t
    from deepseaport.config import AccountConfig
    from deepseaport.accounts import PooledAccount
    from deepseaport.providers.qwen.sse_parser import StreamEvent

    monkeypatch.setattr(QCS, "_headers",
                        lambda app, token: {"Authorization": f"Bearer {token}",
                                            "Cookie": f"token={token}"})
    monkeypatch.setattr(QC, "create_chat", lambda h, m, **k: "chat-1")

    def _fake_stream(h, p, **k):
        assert "NEWJWT" in h.get("Cookie", "")
        yield StreamEvent(kind="content", text="ok")
        yield StreamEvent(kind="finished", text="")

    monkeypatch.setattr(QC, "stream_completion", _fake_stream)
    monkeypatch.setattr(QCS, "_delete_chat_bg", lambda *a, **k: None)
    monkeypatch.setattr(QA, "login_with_expiry",
                        lambda e, p, s=None: ("NEWJWT", _t.time() + 3600,
                                             False, ""))

    cfg = AccountConfig(email="e@x.com", password="pw", token="OLDJWT",
                        token_expires_at=_t.time() - 10)
    item = PooledAccount(cfg=cfg)
    out = QCS._run_completion_core(_qwen_app(), item, _qwen_prep())
    assert out["content"] == "ok"
    assert item.cfg.token == "NEWJWT"
    assert item.cfg.token_expires_at > _t.time()


def test_401_recovers_via_relogin_once(monkeypatch):
    import time as _t
    from deepseaport.config import AccountConfig
    from deepseaport.accounts import PooledAccount
    from deepseaport.providers.qwen.sse_parser import StreamEvent

    monkeypatch.setattr(QCS, "_headers",
                        lambda app, token: {"Authorization": f"Bearer {token}",
                                            "Cookie": f"token={token}"})
    monkeypatch.setattr(QC, "create_chat", lambda h, m, **k: "chat-1")

    seen = []

    def _fake_stream(h, p, **k):
        seen.append(h.get("Cookie", ""))
        if len(seen) == 1:
            yield StreamEvent(kind="error", code="Unauthorized",
                              message="401 Unauthorized")
            return
        yield StreamEvent(kind="content", text="recovered")
        yield StreamEvent(kind="finished", text="")

    monkeypatch.setattr(QC, "stream_completion", _fake_stream)
    monkeypatch.setattr(QCS, "_delete_chat_bg", lambda *a, **k: None)
    monkeypatch.setattr(QC, "check_ban", lambda h, timeout=20: (False, None))
    logins = []
    def _fake_login(e, p, s=None):
        logins.append((e, p))
        return "FRESH", None, False, ""
    monkeypatch.setattr(QA, "login_with_expiry", _fake_login)

    cfg = AccountConfig(email="e@x.com", password="pw", token="STALE")
    item = PooledAccount(cfg=cfg)
    out = QCS._run_completion_core(_qwen_app(), item, _qwen_prep())
    assert out["content"] == "recovered"
    assert item.cfg.token == "FRESH"
    assert "FRESH" in seen[-1]
    assert logins == [("e@x.com", "pw")]


def test_401_without_password_stays_401(monkeypatch):
    from fastapi import HTTPException

    from deepseaport.config import AccountConfig
    from deepseaport.accounts import PooledAccount
    from deepseaport.providers.qwen.sse_parser import StreamEvent

    monkeypatch.setattr(QCS, "_headers",
                        lambda app, token: {"Authorization": f"Bearer {token}",
                                            "Cookie": f"token={token}"})
    monkeypatch.setattr(QC, "create_chat", lambda h, m, **k: "chat-1")

    def _fake_stream(h, p, **k):
        yield StreamEvent(kind="error", code="Unauthorized",
                          message="401 Unauthorized")

    monkeypatch.setattr(QC, "stream_completion", _fake_stream)
    monkeypatch.setattr(QCS, "_delete_chat_bg", lambda *a, **k: None)
    monkeypatch.setattr(QC, "check_ban", lambda h, timeout=20: (False, None))

    cfg = AccountConfig(email="e@x.com", password="", token="STALE")
    item = PooledAccount(cfg=cfg)
    with pytest.raises(HTTPException) as exc:
        QCS._run_completion_core(_qwen_app(), item, _qwen_prep())
    assert exc.value.status_code == 401
    assert "No password stored" in str(exc.value.detail)


def test_401_ban_probe_fails_over_with_403(monkeypatch):
    from fastapi import HTTPException

    from deepseaport.config import AccountConfig
    from deepseaport.accounts import PooledAccount
    from deepseaport.providers.qwen.sse_parser import StreamEvent

    monkeypatch.setattr(QCS, "_headers",
                        lambda app, token: {"Authorization": f"Bearer {token}",
                                            "Cookie": f"token={token}"})
    monkeypatch.setattr(QC, "create_chat", lambda h, m, **k: "chat-1")

    def _fake_stream(h, p, **k):
        yield StreamEvent(kind="error", code="Unauthorized",
                          message="401 Unauthorized")

    monkeypatch.setattr(QC, "stream_completion", _fake_stream)
    monkeypatch.setattr(QCS, "_delete_chat_bg", lambda *a, **k: None)
    # Banned account masquerading as 401: probe says banned.
    monkeypatch.setattr(QC, "check_ban", lambda h, timeout=20: (True, None))

    cfg = AccountConfig(email="e@x.com", password="pw", token="STALETOKEN1234567890")
    item = PooledAccount(cfg=cfg)
    with pytest.raises(HTTPException) as exc:
        QCS._run_completion_core(_qwen_app(), item, _qwen_prep())
    assert exc.value.status_code == 403
    assert item.cfg.banned is True
