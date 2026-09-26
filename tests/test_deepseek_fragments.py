"""Regression tests for the fragments-based DeepSeek SSE shape.

Upstream moved from {"p": "response/content", ...} deltas to typed
fragments: initial content inside {"v": {"response": {"fragments": [...]}}},
later fragments via {"p": "response/fragments", "o": "APPEND", ...}, text via
{"p": "response/fragments/-1/content", ...} plus bare {"v": "..."}
continuations, and usage/finish inside {"p": "response", "o": "BATCH", ...}
with parent-relative paths. The old parser dropped all of it, so every
completion surfaced as 502 "empty upstream response". Offline only.
"""

from __future__ import annotations

from deepseaport.providers.deepseek.sse_parser import StreamParser


def _drain(lines):
    parser = StreamParser()
    content, thinking, usage = [], [], []
    finished, errors = 0, []
    for line in lines:
        for ev in parser.feed(line):
            if ev.kind == "content":
                content.append(ev.text)
            elif ev.kind == "thinking":
                thinking.append(ev.text)
            elif ev.kind == "usage":
                usage.append(ev.text)
            elif ev.kind == "finished":
                finished += 1
            elif ev.kind == "error":
                errors.append(ev)
    return "".join(content), "".join(thinking), usage, finished, errors


def test_fragments_plain_response():
    lines = [
        'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}',
        'data: {"v":{"response":{"message_id":2,"status":"WIP","accumulated_token_usage":0,'
        '"fragments":[{"id":2,"type":"RESPONSE","content":"Hello","references":[],"stage_id":1}]}}}',
        'data: {"p":"response/fragments/-1/content","o":"APPEND","v":" there"}',
        'data: {"v":","}',
        'data: {"v":" how"}',
        'data: {"v":" are"}',
        'data: {"v":" you"}',
        'data: {"v":"?"}',
        'data: {"p":"response","o":"BATCH","v":[{"p":"accumulated_token_usage","v":47},'
        '{"p":"quasi_status","v":"FINISHED"}]}',
        'data: {"p":"response/status","o":"SET","v":"FINISHED"}',
        'data: {"content":"Hi in 5 words"}',
    ]
    content, thinking, usage, finished, errors = _drain(lines)
    assert content == "Hello there, how are you?"
    assert thinking == ""
    assert usage == ["47"]
    assert finished >= 1
    assert errors == []


def test_fragments_think_then_response_split():
    lines = [
        'data: {"v":{"response":{"thinking_enabled":true,"status":"WIP",'
        '"fragments":[{"id":2,"type":"THINK","content":"We","references":[],"stage_id":1}]}}}',
        'data: {"p":"response/fragments/-1/content","o":"APPEND","v":" think"}',
        'data: {"v":" 2+2=4"}',
        'data: {"p":"response/fragments/-1/elapsed_secs","o":"SET","v":1.02}',
        'data: {"p":"response/fragments","o":"APPEND","v":[{"id":3,"type":"RESPONSE",'
        '"content":"2","references":[],"stage_id":1}]}',
        'data: {"p":"response/fragments/-1/content","v":" +"}',
        'data: {"v":" 2 = 4."}',
        'data: {"p":"response/status","o":"SET","v":"FINISHED"}',
    ]
    content, thinking, usage, finished, errors = _drain(lines)
    assert thinking == "We think 2+2=4"
    assert content == "2 + 2 = 4."
    assert finished == 1
    assert errors == []


def test_fragments_unknown_type_routes_to_content():
    lines = [
        'data: {"v":{"response":{"fragments":[{"id":9,"type":"WEIRD","content":"x"}]}}}',
        'data: {"p":"response/status","o":"SET","v":"FINISHED"}',
    ]
    content, thinking, _, finished, _ = _drain(lines)
    assert content == "x"
    assert thinking == ""
    assert finished == 1


def test_batch_relative_usage_and_quasi_finish():
    lines = [
        'data: {"p":"response","o":"BATCH","v":[{"p":"accumulated_token_usage","v":78},'
        '{"p":"quasi_status","v":"FINISHED"}]}',
    ]
    content, thinking, usage, finished, _ = _drain(lines)
    assert usage == ["78"]
    assert finished == 1
    assert content == "" and thinking == ""


def test_legacy_paths_still_work():
    lines = [
        'data: {"p":"response/thinking_content","o":"APPEND","v":"hmm"}',
        'data: {"v":"..."}',
        'data: {"p":"response/content","o":"APPEND","v":"hi"}',
        'data: {"v":"!"}',
        'data: {"p":"response/accumulated_token_usage","v":10}',
        'data: {"p":"response/status","o":"SET","v":"FINISHED"}',
    ]
    content, thinking, usage, finished, errors = _drain(lines)
    assert thinking == "hmm..."
    assert content == "hi!"
    assert usage == ["10"]
    assert finished == 1
    assert errors == []


def test_obscura_bridge_resolves_directory_to_exe(tmp_path):
    from deepseaport.core.obscura_bridge import ObscuraBridge

    exe = tmp_path / "obscura.exe"
    exe.write_bytes(b"x")
    bridge = ObscuraBridge(str(tmp_path), profile=str(tmp_path / "profile"))
    assert bridge.binary == str(exe)
