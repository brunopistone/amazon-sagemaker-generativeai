"""Helpers shared by the reward functions in this package.

Modules whose name starts with an underscore are skipped by the package's
auto-discovery, so this file can hold plain utilities without them being
mistaken for reward modules.
"""

import hashlib
import json
import logging
from typing import Any, List

logger = logging.getLogger(__name__)

# Roles that may appear in a conversational prompt. A prompt must END on one of
# ``TURN_YIELDING_ROLES`` so the model starts a new assistant turn instead of
# continuing one that is already open.
CHAT_ROLES = ("system", "developer", "user", "assistant", "tool")
TURN_YIELDING_ROLES = ("user", "tool")


def _flatten_content(content: Any) -> str:
    """Reduce a message's ``content`` field to text.

    ``content`` is a plain string in the common case and a list of typed blocks
    for multimodal messages; only the text blocks survive.
    """
    if content is None:
        return ""
    if isinstance(content, list):
        text_parts = [
            block["text"]
            for block in content
            if isinstance(block, dict)
            and isinstance(block.get("text"), str)
            and block["text"]
        ]
        return " ".join(text_parts).strip()
    return str(content)


def _normalize_tool_call(call: Any):
    if not isinstance(call, dict):
        return None

    source = call
    for key in ("function", "functionCall", "toolUse", "tool_use"):
        nested = call.get(key)
        if isinstance(nested, dict):
            source = nested
            break

    name = source.get("name")
    if not name:
        return None
    for key in ("arguments", "args", "input"):
        if key in source:
            arguments = source[key]
            break
    else:
        arguments = {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments, strict=False)
        except (json.JSONDecodeError, ValueError):
            pass
    return name, arguments


# Tool-call syntaxes emitted by the chat templates in use. Rewards re-render
# historical tool calls, so the wrong syntax shows the judge an action in a form
# the policy never produces.
FORMAT_JSON_TOOLCALL = "json_toolcall"   # Qwen3:            <tool_call>{"name","arguments"}</tool_call>
FORMAT_TAGS = "tags"                     # Qwen3.5/3.8, Nemotron 3: <function=>/<parameter=>
FORMAT_GLM = "glm"                       # GLM-4.5:          <tool_call>name/<arg_key>/<arg_value>
FORMAT_LLAMA = "llama"                   # Llama 3.x:        bare {"name","parameters"}
FORMAT_MISTRAL = "mistral"               # Mistral:          [TOOL_CALLS] [{...,"id"}]
FORMAT_DEEPSEEK = "deepseek"             # DeepSeek:         tool_sep + ```json fence
FORMAT_NEMOTRON_TOOLCALL = "nemotron_toolcall"  # Nemotron Nano: <TOOLCALL>[{...}]</TOOLCALL>
FORMAT_HARMONY = "harmony"               # gpt-oss:          to=functions.name commentary channel

_DS_CALLS_BEGIN = "<｜tool▁calls▁begin｜>"
_DS_CALL_BEGIN = "<｜tool▁call▁begin｜>"
_DS_SEP = "<｜tool▁sep｜>"
_DS_CALL_END = "<｜tool▁call▁end｜>"

# Ordered most-specific first: the tag templates also contain "<tool_call>", and
# Llama is identified only by its bare {"name","parameters"} payload, so it is
# checked after the wrapper-based conventions.
_FORMAT_MARKERS = (
    ("<arg_key>", FORMAT_GLM),
    ("[TOOL_CALLS]", FORMAT_MISTRAL),
    ("tool▁sep", FORMAT_DEEPSEEK),
    ("<TOOLCALL>", FORMAT_NEMOTRON_TOOLCALL),
    ("<|start|>assistant to=", FORMAT_HARMONY),
    ("<function=", FORMAT_TAGS),
    ("<tool_call>", FORMAT_JSON_TOOLCALL),
    ('"parameters": ', FORMAT_LLAMA),
)

_TOOL_CALL_FORMAT = FORMAT_JSON_TOOLCALL
_SCALARS_AS_JSON = True

_PROBE_ARGS = {"s": "x", "b": True}


def set_tool_call_format(chat_template: Any = None, tokenizer: Any = None) -> str:
    """Match the tool-call syntax this run's chat template actually emits.

    Detection is by marker string, so a model whose template is not recognised
    falls back to the JSON convention with a warning rather than silently
    emitting a format nothing produces.

    The tag templates further disagree on non-string scalars: some emit Jinja's
    ``| string`` (``True``), others ``| tojson`` (``true``). Passing ``tokenizer``
    calibrates that by rendering one probe call, which stays correct if a template
    changes.
    """
    global _TOOL_CALL_FORMAT, _SCALARS_AS_JSON

    if chat_template is None and tokenizer is not None:
        chat_template = getattr(tokenizer, "chat_template", None)

    _TOOL_CALL_FORMAT = FORMAT_JSON_TOOLCALL
    _SCALARS_AS_JSON = True

    if not isinstance(chat_template, str):
        logger.warning(
            "No chat template available; assuming %s tool-call rendering.",
            FORMAT_JSON_TOOLCALL,
        )
        return _TOOL_CALL_FORMAT

    for marker, fmt in _FORMAT_MARKERS:
        if marker in chat_template:
            _TOOL_CALL_FORMAT = fmt
            break
    else:
        logger.warning(
            "Chat template has no recognised tool-call syntax%s; falling back to "
            "%s. Tool calls in prompt history may not match what the model emits.",
            "" if "tool_calls" in chat_template else " (and no tool_calls support)",
            FORMAT_JSON_TOOLCALL,
        )

    if _TOOL_CALL_FORMAT == FORMAT_TAGS and tokenizer is not None:
        probe = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "p", "arguments": _PROBE_ARGS}}
            ],
        }
        try:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": "go"}, probe],
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:  # a probe failure must not abort training
            rendered = ""
        if "<parameter=b>\nTrue\n</parameter>" in rendered:
            _SCALARS_AS_JSON = False

    return _TOOL_CALL_FORMAT


def _arg_value(value: Any) -> str:
    """Encode one argument value the way the active template would."""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)) or _SCALARS_AS_JSON:
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _as_dict(arguments: Any) -> dict:
    return arguments if isinstance(arguments, dict) else {"arguments": arguments}


def _mistral_id(call: Any, name: str, arguments: Any) -> str:
    """Mistral rejects tool call ids that are not 9 alphanumeric characters."""
    for key in ("id", "tool_call_id", "toolUseId"):
        candidate = (call or {}).get(key) if isinstance(call, dict) else None
        if isinstance(candidate, str) and len(candidate) == 9 and candidate.isalnum():
            return candidate
    seed = json.dumps([name, arguments], ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return digest[:9]


def _render_one(name: str, arguments: Any) -> str:
    fmt = _TOOL_CALL_FORMAT
    if fmt == FORMAT_TAGS:
        lines = ["<tool_call>", f"<function={name}>"]
        for key, value in _as_dict(arguments).items():
            lines.extend([f"<parameter={key}>", _arg_value(value), "</parameter>"])
        lines.extend(["</function>", "</tool_call>"])
        return "\n".join(lines)

    if fmt == FORMAT_GLM:
        lines = [f"<tool_call>{name}"]
        for key, value in _as_dict(arguments).items():
            lines.append(f"<arg_key>{key}</arg_key>")
            lines.append(f"<arg_value>{_arg_value(value)}</arg_value>")
        lines.append("</tool_call>")
        return "\n".join(lines)

    if fmt == FORMAT_LLAMA:
        return json.dumps({"name": name, "parameters": arguments}, ensure_ascii=False)

    if fmt == FORMAT_HARMONY:
        return (
            f"<|start|>assistant to=functions.{name}<|channel|>commentary json"
            f"<|message|>{json.dumps(arguments, ensure_ascii=False)}<|call|>"
        )

    payload = {"name": name, "arguments": arguments}
    return "<tool_call>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_call>"


def _render_tool_calls(tool_calls: Any) -> List[str]:
    """Render structured tool calls in the syntax the active model emits."""
    if isinstance(tool_calls, dict):
        tool_calls = [tool_calls]

    calls = []
    for call in tool_calls or []:
        normalized = _normalize_tool_call(call)
        if normalized is None:
            continue
        name, arguments = normalized
        calls.append((name, arguments, call))
    if not calls:
        return []

    # These templates wrap every call of a turn in ONE envelope.
    if _TOOL_CALL_FORMAT == FORMAT_MISTRAL:
        payload = [
            {
                "name": name,
                "arguments": arguments,
                "id": _mistral_id(call, name, arguments),
            }
            for name, arguments, call in calls
        ]
        return ["[TOOL_CALLS] " + json.dumps(payload, ensure_ascii=False)]

    if _TOOL_CALL_FORMAT == FORMAT_DEEPSEEK:
        blocks = [
            f"{_DS_CALL_BEGIN}function{_DS_SEP}{name}\n```json\n"
            f"{json.dumps(arguments, ensure_ascii=False)}\n```{_DS_CALL_END}"
            for name, arguments, _ in calls
        ]
        return [_DS_CALLS_BEGIN + "\n".join(blocks)]

    if _TOOL_CALL_FORMAT == FORMAT_NEMOTRON_TOOLCALL:
        payload = [
            {"name": name, "arguments": arguments} for name, arguments, _ in calls
        ]
        return ["<TOOLCALL>" + json.dumps(payload, ensure_ascii=False) + "</TOOLCALL>"]

    return [_render_one(name, arguments) for name, arguments, _ in calls]


def _content_tool_calls(content: Any) -> List[Any]:
    if not isinstance(content, list):
        return []
    return [
        block
        for block in content
        if isinstance(block, dict)
        and (
            block.get("type") in ("tool_use", "function_call")
            or "functionCall" in block
            or "toolUse" in block
        )
    ]


def render_message_content(message: Any) -> str:
    """Flatten ONE chat message to the text the model itself would emit.

    A message carries its payload across up to three channels, and in an agentic
    dataset any single one of them may be the only one populated:

      - ``content``            plain text, or a list of multimodal blocks
      - ``reasoning_content``  the ``<think>`` channel (Qwen3.5 and similar)
      - ``tool_calls``         the action, when ``content`` is empty

    Reading only ``content`` blanks every assistant turn that just calls a tool,
    which in a tool-use trajectory is *all* of them - so a judge shown such a
    prompt sees the tool output but not the action that produced it, and cannot
    tell a sensible next step from a redundant one.

    When there is nothing to compose (no reasoning, no tool calls) ``content`` is
    returned **byte for byte**: reward functions regex over this text and measure
    its length, so trailing whitespace is load-bearing and must survive.
    """
    if isinstance(message, str):
        return message
    if not isinstance(message, dict):
        return str(message)

    content = _flatten_content(message.get("content"))

    extras = []
    reasoning = message.get("reasoning_content")
    if reasoning and str(reasoning).strip():
        extras.append(f"<think>\n{str(reasoning).strip()}\n</think>")
    tool_calls = message.get("tool_calls") or message.get("function_call")
    rendered = _render_tool_calls(tool_calls)
    rendered += [
        block
        for block in _render_tool_calls(_content_tool_calls(message.get("content")))
        if block not in rendered
    ]
    # Consecutive tool calls of one turn are adjacent in every template, so they
    # join with a single newline; only the channels are separated by a blank line.
    if rendered:
        extras.append("\n".join(rendered))

    if not extras:
        return content

    head = content.strip()
    return "\n\n".join(([head] if head else []) + extras)


def extract_completion_text(completion: Any) -> str:
    """Normalize a single GRPO completion to plain text.

    GRPOTrainer passes completions in one of two shapes depending on the prompt
    format:
      - conversational prompt -> completion is a list of message dicts, e.g.
        ``[{"role": "assistant", "content": "..."}]``
      - standard (plain string) prompt -> completion is a plain ``str``

    This helper handles both (plus a bare dict, defensively) so reward functions
    never crash with ``'str' object has no attribute 'get'`` on standard-format
    datasets.

    Every message is rendered and joined rather than only ``completion[0]``: TRL
    happens to wrap a generation as a single-element list today, but a
    multi-message completion would otherwise be silently truncated to its first
    turn.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        if not completion:
            return ""
        if len(completion) == 1:
            return render_message_content(completion[0])
        return "\n\n".join(render_message_content(m) for m in completion)
    if isinstance(completion, dict):
        return render_message_content(completion)
    return str(completion)
