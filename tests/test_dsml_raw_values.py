"""DSML parameter values: what survives, byte for byte.

Protects three defects:

- F4  a value holding a ``</>`` (a JSX fragment) was cut at the generic close
      marker even though the block's own ``</parameter>`` came later, so a
      file write was silently truncated;
- F7  the declared schema was ignored when coercing a value, so a
      ``string`` parameter lost its padding, its trailing newline, or a
      zero-padded id (``007`` -> 7, ``None`` -> null);
- F29 the lowercased-name map behind case-fold canonicalization followed set
      iteration order, so one declared spelling won in one process and the
      other in the next (PYTHONHASHSEED).

Two known limits are pinned as xfail (see their reasons): a value that
contains the *specific* close text (``</parameter>``, in any of the spellings
the close pattern accepts) or a quoted ``</invoke>`` / ``<parameter ...>``
marker, and the generic-close dialect (``</>`` as the parameter closer), still
cut there.
"""

import asyncio
import json
import os
import subprocess
import sys
import textwrap

import pytest

from deepseaport.core import tools as T

_LT = chr(60)
_GT = chr(62)
_DS = "||DSML||"
_GENERIC_CLOSE = _LT + "/" + _GT


def _op(tag: str, attrs: str = "") -> str:
    return f"{_LT}{_DS}{tag}{attrs}{_GT}"


def _cl(tag: str) -> str:
    return f"{_LT}/{_DS}{tag}{_GT}"


def _write_tools():
    return [{"type": "function",
             "function": {"name": "write",
                          "description": "write a file",
                          "parameters": {
                              "type": "object",
                              "properties": {
                                  "filePath": {"type": "string"},
                                  "content": {"type": "string"},
                                  "count": {"type": "integer"},
                                  "ratio": {"type": "number"},
                                  "flag": {"type": "boolean"},
                              },
                              "required": ["filePath", "content"]}}}]


def _block(params, param_close=None, invoke_close=None) -> str:
    """DSML tool_calls block; params = [(name, attr_text, value)]."""
    parts = [_op("tool_calls"), _op("invoke", ' name="write"')]
    for name, attrs, value in params:
        parts.append(_op("parameter", f' name="{name}"{attrs}') + value
                     + (param_close or _cl("parameter")))
    parts.append(invoke_close or _cl("invoke"))
    parts.append(_cl("tool_calls"))
    return "".join(parts)


def _args(text: str, tools=None, forgiving=None) -> dict:
    calls, _ = T.parse_tool_calls(text, tools or _write_tools(),
                                  forgiving=forgiving)
    assert calls is not None, text
    return json.loads(calls[0]["function"]["arguments"])


_JSX = "const App = () => (<><div>hi</div></>);\n"


# --- F4: the value ends at its own close, not at a quoted marker ----------

@pytest.mark.parametrize("forgiving", [None, True])
def test_jsx_fragment_is_value_text_not_the_close(forgiving):
    """``</>`` inside a write payload must not end the content there."""
    got = _args(_block([("filePath", ' string="true"', "App.jsx"),
                        ("content", ' string="true"', _JSX)]),
                forgiving=forgiving)
    assert got == {"filePath": "App.jsx", "content": _JSX}


def test_jsx_fragment_survives_without_the_string_attr():
    """Same payload with no string attr: the schema declares it a string."""
    assert _args(_block([("content", "", _JSX)]))["content"] == _JSX


def test_jsx_fragment_survives_a_second_parameter_after_it():
    """The next ``<parameter>`` open bounds the range, the marker inside does
    not end it."""
    got = _args(_block([("content", ' string="true"', _JSX),
                        ("filePath", ' string="true"', "App.jsx")]))
    assert got == {"filePath": "App.jsx", "content": _JSX}


@pytest.mark.xfail(reason="a value quoting the literal close text is still "
                          "cut at it: fixing that needs 'last specific close "
                          "in range', which breaks the MAX_PARAMS_PER_INVOKE "
                          "value ranges (test_param_cap_* in "
                          "test_tool_and_provider_gaps.py)",
                   strict=False)
@pytest.mark.parametrize("doc", [
    "close a parameter with </parameter>, then a fragment </>\n",
    # Same limit, the other spellings the close pattern accepts: case
    # (IGNORECASE), a space before ">" ([ ]* in the tag pattern), and the
    # fullwidth form (normalized 1:1 before parsing).
    "close a parameter with </PARAMETER>, then a fragment </>\n",
    "close a parameter with </parameter >, then a fragment </>\n",
    "close a parameter with \uFF1C/parameter\uFF1E, then a fragment </>\n",
])
def test_value_quoting_the_literal_close_text_is_still_cut(doc):
    assert _args(_block([("content", ' string="true"', doc)]))["content"] == doc


@pytest.mark.xfail(reason="a value quoting the HTML-escaped close text is "
                          "still cut at it in the forgiving dialect: the "
                          "escaped marker is a legitimate close there, same "
                          "'last specific close in range' limit as the ASCII "
                          "form",
                   strict=False)
def test_value_quoting_the_escaped_close_text_is_still_cut():
    doc = "close with &lt;/parameter&gt; in text\n"
    assert _args(_block([("content", ' string="true"', doc)]),
                 forgiving=True)["content"] == doc


@pytest.mark.xfail(reason="a value quoting the invoke close </invoke> is "
                          "still cut at it: the first invoke close bounds "
                          "every parameter value range, and choosing the "
                          "close after the last parameter close needs the "
                          "closed-parameter structure the capped scan does "
                          "not build",
                   strict=False)
def test_value_quoting_the_invoke_close_text_is_still_cut():
    doc = "close the block with </invoke> in prose\n"
    assert _args(_block([("filePath", ' string="true"', "a.md"),
                         ("content", ' string="true"', doc)]))["content"] == doc


@pytest.mark.xfail(reason="a value quoting a <parameter ...> open is cut "
                          "there AND read as another argument: an open tag "
                          "inside a raw value is indistinguishable from a "
                          "real one without a closed-parameter scan, and the "
                          "next-open boundary wins",
                   strict=False)
def test_value_quoting_a_parameter_open_is_still_cut():
    doc = 'example: <parameter name="x"> here\n'
    args = _args(_block([("filePath", ' string="true"', "a.md"),
                         ("content", ' string="true"', doc)]))
    assert args["content"] == doc
    assert "x" not in args


@pytest.mark.xfail(reason="the generic-close dialect (</> as the parameter "
                          "closer) cannot tell a JSX </> from its own closer; "
                          "'last generic in range' regresses "
                          "test_parse_dsml_generic_close_tags",
                   strict=False)
def test_jsx_fragment_in_the_generic_close_dialect_is_still_cut():
    text = _block([("content", ' string="true"', _JSX)],
                  param_close=_GENERIC_CLOSE)
    assert _args(text)["content"] == _JSX


# --- F7: a declared string keeps its value verbatim -----------------------

@pytest.mark.parametrize("forgiving", [None, True])
@pytest.mark.parametrize("value", ["007", "12345", "None", "  padded  ",
                                   "1.50", "0"])
def test_declared_string_value_is_never_coerced(forgiving, value):
    """Zero padding, padding and null-looking text are the file's content."""
    assert _args(_block([("filePath", "", value)]),
                 forgiving=forgiving)["filePath"] == value


@pytest.mark.parametrize("string_attr", [' string="true"', ""])
def test_declared_string_keeps_a_trailing_newline(string_attr):
    body = "line1\nline2\n"
    assert _args(_block([("content", string_attr, body)]))["content"] == body


@pytest.mark.parametrize("forgiving", [None, True])
def test_other_declared_types_keep_the_coercion(forgiving):
    got = _args(_block([("count", "", "42"), ("ratio", "", "3.14"),
                        ("flag", "", "true"), ("filePath", "", "x.md")]),
                forgiving=forgiving)
    assert got == {"count": 42, "ratio": 3.14, "flag": True,
                   "filePath": "x.md"}


def test_no_schema_keeps_the_coercion():
    """Nothing declared: the historical int/float/bool read and its strip."""
    tools = [{"type": "function", "function": {"name": "write"}}]
    assert _args(_block([("count", "", "42"), ("content", "", "  padded  ")]),
                 tools) == {"count": 42, "content": "padded"}


def test_string_attr_still_beats_a_non_string_declared_type():
    """The model's own ``string="true"`` keeps today's precedence order."""
    assert _args(_block([("count", ' string="true"', " 7")]))["count"] == " 7"


def test_both_dsml_shapes_agree_on_declared_strings():
    """A JSON-object invoke body keeps values verbatim; the tag shape must too.

    The JSON body is a forgiving repair, so both sides run forgiving.
    """
    body_args = {"filePath": "007", "content": "line1\n"}
    json_body = (_op("tool_calls") + _op("invoke", ' name="write"')
                 + json.dumps(body_args) + _cl("invoke") + _cl("tool_calls"))
    tagged = _block([("filePath", "", body_args["filePath"]),
                     ("content", "", body_args["content"])])
    assert _args(tagged, forgiving=True) == body_args
    assert _args(json_body, forgiving=True) == body_args


def test_streamed_path_agrees_with_the_buffered_parse():
    """The stream tail (parse_stream_calls) reads values like parse_tool_calls.

    A JSX fragment plus a zero-padded string, both directions of the toggle:
    the streamed path is where a live harness first sees the arguments, so a
    value healed only in the buffered parser would still be cut in practice.
    """
    from deepseaport.core.engine import parse_stream_calls

    text = _block([("filePath", ' string="true"', "007"),
                   ("content", ' string="true"', _JSX)])
    for forgiving in (False, True):
        prep = {"tools": _write_tools(), "forgiving_toolcalls": forgiving,
                "model": "test"}
        calls, rest, truncated = asyncio.run(parse_stream_calls(prep, text))
        assert calls is not None
        assert json.loads(calls[0]["function"]["arguments"]) == {
            "filePath": "007", "content": _JSX}
        assert rest == "" and truncated is False


# --- F29: canonical spelling must not depend on the process --------------

def _canonical_for_hash_seed(seed: int):
    """Canonical name for a ``{"read", "Read"}`` declaration, fresh process."""
    import deepseaport

    # <repo>/src, so the child imports THIS checkout, not an installed copy.
    src_root = os.path.dirname(os.path.dirname(
        os.path.abspath(deepseaport.__file__)))
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(seed)
    env["PYTHONPATH"] = src_root + os.pathsep + env.get("PYTHONPATH", "")
    code = textwrap.dedent(
        """
        import json
        from deepseaport.core.tools import common as TC
        calls, _ = TC._normalize_calls(
            [{"function": {"name": "READ", "arguments": "{}"}}],
            {"read", "Read"})
        print(json.dumps(calls[0]["function"]["name"] if calls else None))
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=120, env=env)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_case_variant_tool_names_canonicalize_the_same_in_every_process():
    """One declared spelling wins everywhere, not per hash seed.

    Before the map was built in sorted order these seeds disagreed
    (``PYTHONHASHSEED=0`` -> "Read", ``1`` -> "read"), so the same reply
    dispatched under a different tool name in a different process.
    """
    canonicals = {seed: _canonical_for_hash_seed(seed) for seed in (0, 1)}
    assert len(set(canonicals.values())) == 1, canonicals
    assert canonicals[0] in {"read", "Read"}, canonicals
