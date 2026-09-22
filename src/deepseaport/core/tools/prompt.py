"""Prompt rendering for tool harnesses."""

from __future__ import annotations

import logging

from .common import MAX_ARGS_CHARS, _valid_names

logger = logging.getLogger("deepseaport.tools")


def _example_param_value(pinfo) -> str:
    if not isinstance(pinfo, dict):
        return "value"
    ptype = str(pinfo.get("type", "string") or "string").lower()
    if ptype in ("integer", "number"):
        return "0"
    if ptype == "boolean":
        return "true"
    if ptype == "array":
        return "[]"
    if ptype == "object":
        return "{}"
    return "value"


def tool_system_prompt(tools: list[dict], max_args_chars: int | None = None) -> str:
    lt = chr(60)
    gt = chr(62)
    ds = "||DSML||"
    op = lt + ds
    cl = lt + "/" + ds
    lines = ["You have access to the following tools. Call one when it helps.",
             ""]
    first_fn: dict = {}
    for tool in tools:
        fn = tool.get("function", {}) if isinstance(tool, dict) else {}
        if not first_fn and isinstance(fn, dict) and fn.get("name"):
            first_fn = fn
        name = fn.get("name", "unknown")
        desc = fn.get("description", "")
        params = fn.get("parameters", {}) or {}
        lines.append(f"Tool: {name}")
        if desc:
            lines.append(f"Description: {desc}")
        props = params.get("properties", {}) or {}
        required = set(params.get("required", []) or [])
        if props:
            lines.append("Parameters:")
            for pname, pinfo in props.items():
                ptype = pinfo.get("type", "string") if isinstance(pinfo, dict) else "string"
                pdesc = pinfo.get("description", "") if isinstance(pinfo, dict) else ""
                mark = " (required)" if pname in required else ""
                lines.append(f"  - {pname}: {ptype}{mark} - {pdesc}")
        lines.append("")

    names = sorted(_valid_names(tools))
    example_name = first_fn.get("name", "tool_name")
    example_props = (first_fn.get("parameters", {}) or {}).get("properties", {}) or {}
    example_required = list((first_fn.get("parameters", {}) or {}).get("required", []) or [])
    example_params = example_required or list(example_props)[:2] or ["param"]
    example_values = {
        pname: _example_param_value(example_props.get(pname, {}))
        for pname in example_params
    }
    call_lines = [op + "tool_calls" + gt,
                  op + f'invoke name="{example_name}"' + gt]
    for pname in example_params:
        call_lines.append(
            op + f'parameter name="{pname}" string="true"' + gt
            + example_values[pname] + cl + "parameter" + gt)
    call_lines += [cl + "invoke" + gt, cl + "tool_calls" + gt]

    arg_limit = max_args_chars if (isinstance(max_args_chars, int)
                                 and max_args_chars > 0) else MAX_ARGS_CHARS
    lines += [
        "PREFERRED call format (raw values; quotes, backslashes, newlines and HTML do NOT need escaping):",
        *call_lines,
        "",
        "Rules:",
        "- Prefer the DSML block format above. Use it for any call with file content, code or a shell command.",
        "- Use the exact tool name and parameter names shown in the schema above; never rename parameters.",
        "- Include every parameter marked (required).",
        "- Output ONLY the tool call. No prose, no Thought:, no Markdown fences, no extra wrapper tags.",
        "- Put parallel calls in one DSML tool_calls block (multiple invoke blocks).",
        "- Keep each call's arguments below " + f"{arg_limit:,}" + " characters. If a file or command is larger, split it into sequential smaller calls: write one chunk, then append the next chunk in later calls. The upstream model output is truncated on very long replies, so one giant call will fail.",
        "- If no tool is needed, answer normally without any tool markup.",
        "- After tool results arrive, use them and answer the user.",
        "",
        "A short JSON form is also accepted for simple calls (arguments may be an object or a JSON string):",
        '{"tool_calls": [{"id": "call_001", "type": "function", "function": {"name": "tool_name", "arguments": {"param": "value"}}}]}',
    ]
    if names:
        lines.append(f"Valid tool names: {', '.join(names)}. Do not invent other names.")
    return "\n".join(lines)


def tool_reminder_prompt(tools: list[dict] | None = None) -> str:
    """Short recency reminder appended after the user prompt (tool calls only).

    The main schema lives in tool_system_prompt() at the start; long histories
    bury it, so this repeats only the output contract at the end.
    """
    names = sorted(_valid_names(tools))
    base = ("Reminder: to call a tool, output ONLY a DSML tool_calls block "
            "(preferred; JSON tool_calls also accepted). Use the exact parameter "
            "names from the schema. Raw DSML parameter values need no escaping. "
            "If file content or a command is large, send several smaller calls "
            "instead of one oversized call. If no tool is needed, answer normally.")
    if names:
        base += f" Valid names: {', '.join(names)}."
    return base


def _message_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        dropped_images = 0
        for p in content:
            if not isinstance(p, dict):
                # Historical behaviour: bare strings/None in a content list
                # are skipped (content lists hold typed dicts).
                continue
            ptype = str(p.get("type", "") or "")
            if ptype in ("text", "input_text", "output_text"):
                parts.append(str(p.get("text", "") or ""))
            elif ptype in ("image_url", "image", "input_image"):
                dropped_images += 1
                url = ""
                try:
                    img = p.get("image_url", {})
                    if isinstance(img, dict):
                        url = str(img.get("url", "") or "")
                    elif isinstance(img, str):
                        url = img
                    if not url:
                        url = str(p.get("url", "") or "")
                except Exception:
                    url = ""
                # Web upstreams accept text prompts only: keep a visible
                # marker (with URL when short) instead of silently dropping
                # the image so the model knows an image was intended.
                if url and len(url) <= 300:
                    parts.append(f"[image: {url}]")
                elif url:
                    parts.append(f"[image: {url[:300]}... ({len(url)} chars total, omitted)]")
                else:
                    parts.append("[image input omitted: upstream accepts text only]")
            elif ptype:
                # Unknown typed blocks stay skipped (historical); images
                # above are the only lossy case that now warns + marks.
                continue
            elif p.get("text"):
                parts.append(str(p.get("text") or ""))
        if dropped_images:
            logger.warning(
                "image content blocks (%d) reduced to text placeholders "
                "(web upstream accepts text prompts only)", dropped_images)
        return "\n".join(s for s in parts if s)
    return str(content or "")


def render_prompt(messages: list[dict]) -> str:
    """Flatten OpenAI messages into the single web `prompt` string.

    - assistant tool_call messages become explicit "I called ..." records,
    - tool messages become "<Tool>Tool <name> returned: ..." records (kept
      distinct from user turns so tool boundaries survive),
    - same-role neighbours merge with a labeled separator so no boundary is
      lost; roles map to <System>/<User>/<Assistant>/<Tool> markers.
    """
    blocks: list[tuple[str, str]] = []
    for msg in messages:
        role = msg.get("role", "user")
        text = _message_text(msg.get("content", ""))
        if role == "assistant" and msg.get("tool_calls"):
            calls = "; ".join(
                f"{c.get('function', {}).get('name', '?')}({c.get('function', {}).get('arguments', '{}')})"
                for c in msg["tool_calls"] if isinstance(c, dict)
            )
            text = (text + "\n" if text else "") + f"[I called tools: {calls}]"
            role = "assistant"
        elif role == "tool":
            name = msg.get("name", "tool")
            text = f"Tool {name} returned: {text}"
            role = "tool"  # distinct role: tool output is new info, not a user turn
        blocks.append((role, text))
    # Accumulate chunks per role instead of concatenating the growing string
    # (O(n^2) on long same-role runs produced by tool loops).
    merged: list[tuple[str, list[str]]] = []
    for role, text in blocks:
        if merged and merged[-1][0] == role:
            merged[-1][1].append(text)
        else:
            merged.append((role, [text]))
    parts = []
    for idx, (role, chunks) in enumerate(merged):
        # Labeled separator preserves tool/message boundaries when same-role
        # turns merge (previously plain "\n\n" lost them).
        if len(chunks) > 1:
            text = f"\n\n--- {role} ---\n\n".join(chunks)
        else:
            text = chunks[0]
        if role == "assistant":
            parts.append(f"<Assistant>{text}<endofsentence>")
        elif role == "system":
            parts.append(f"<System>{text}")
        elif role == "tool":
            parts.append(f"<Tool>{text}")
        elif role == "user":
            parts.append(text if idx == 0 else f"<User>{text}")
        else:
            parts.append(text)
    return "".join(parts)
