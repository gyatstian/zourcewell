"""Message-shape validation at the HTTP edge (F12).

``_validate_messages`` checked that each entry is an object with a ``role``,
but not the SHAPE of its ``tool_calls``: ``{"role": "assistant", "tool_calls":
5}`` reached ``core.tools.prompt.render_prompt``, which iterates the field --
TypeError -> an opaque 500 on the live route AND on the dry app (dry_run reuses
``_prepare``). A non-list ``tool_calls`` is now a 400 naming the field.

Only the container is validated. Elements stay the renderer's job (it skips
non-dicts), and a falsy value -- missing, ``null``, ``[]`` -- is an ordinary
assistant turn that must keep rendering.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from deepseaport.dry_run import create_dry_app
from deepseaport.server import create_app
from deepseaport.server_routes import _prepare
from support import settings as _settings

MSGS = [{"role": "user", "content": "hi"}]
_BAD_DETAIL = "messages[1].tool_calls must be a list"


def _body(tool_calls):
    return {"model": "deepseek-flash",
            "messages": [*MSGS, {"role": "assistant", "tool_calls": tool_calls}]}


@pytest.mark.parametrize("tool_calls", [5, 1.5, "5", {"function": {"name": "f"}}])
def test_non_list_tool_calls_is_a_400_naming_the_field(tool_calls):
    with pytest.raises(HTTPException) as ei:
        _prepare(_body(tool_calls), _settings())

    assert ei.value.status_code == 400
    assert str(ei.value.detail) == _BAD_DETAIL


@pytest.mark.parametrize("tool_calls", [5, "5", {"function": {"name": "f"}}])
def test_live_route_400s_instead_of_500(tool_calls):
    resp = TestClient(create_app(_settings()),
                      raise_server_exceptions=False).post(
        "/v1/chat/completions", json=_body(tool_calls))

    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == _BAD_DETAIL


def test_dry_app_400s_instead_of_500():
    """Same 500 ran on the dry surface: create_dry_app calls the same _prepare."""
    resp = TestClient(create_dry_app(_settings()),
                      raise_server_exceptions=False).post(
        "/v1/chat/completions", json=_body(5))

    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == _BAD_DETAIL


@pytest.mark.parametrize("tool_calls", [None, [], [{"function": {"name": "f"}}], [5]])
def test_container_ok_still_renders(tool_calls):
    """Control: the guard is about the container, never the elements or a
    falsy value -- these are all ordinary shapes and must not become a 400."""
    prep = _prepare(_body(tool_calls), _settings())

    assert prep["prompt"].strip()


def test_a_real_assistant_tool_call_still_renders():
    prep = _prepare(_body([{"function": {"name": "read",
                                         "arguments": '{"path": "x"}'}}]),
                    _settings())

    assert "read(" in prep["prompt"]
