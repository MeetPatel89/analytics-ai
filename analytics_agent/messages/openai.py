"""OpenAI Responses API adapters for canonical chat messages."""

from __future__ import annotations

from typing import Any

from analytics_agent.messages.factories import function_call_message
from analytics_agent.messages.types import (
    AssistantMessage,
    ChatMessage,
    ContentPart,
    FunctionCallMessage,
    FunctionCallOutputMessage,
    RefusalPart,
    SystemMessage,
    TextPart,
    UserMessage,
)


def to_openai_input_item(message: ChatMessage) -> dict[str, Any]:
    """Serialize a canonical message to an OpenAI Responses API input item."""
    if isinstance(message, (SystemMessage, UserMessage)):
        return {
            "role": message.role,
            "content": _require_plain_text(message.content),
        }
    if isinstance(message, AssistantMessage):
        item: dict[str, Any] = {
            "type": "message",
            "role": message.role,
            "content": [_assistant_part_to_openai(part) for part in message.content],
        }
        if message.id:
            item["id"] = message.id
        if message.status:
            item["status"] = message.status
        if message.phase:
            item["phase"] = message.phase
        return item
    if isinstance(message, FunctionCallMessage):
        item = {
            "type": "function_call",
            "call_id": message.call_id,
            "name": message.name,
            "arguments": message.arguments_raw,
        }
        if message.id:
            item["id"] = message.id
        if message.status:
            item["status"] = message.status
        return item
    if isinstance(message, FunctionCallOutputMessage):
        return {
            "type": "function_call_output",
            "call_id": message.call_id,
            "output": message.output,
        }
    raise TypeError(f"Unsupported message type: {type(message)!r}")


def from_openai_output_item(item: object) -> ChatMessage | None:
    """Deserialize an OpenAI Responses API output item."""
    item_type = getattr(item, "type", None)
    if item_type == "function_call":
        return function_call_message(
            call_id=item.call_id,
            name=item.name,
            arguments_raw=item.arguments,
            id=getattr(item, "id", None),
            status=getattr(item, "status", None),
            raw_payload=_model_to_dict(item),
        )
    if item_type == "message":
        role = getattr(item, "role", None)
        content = tuple(_part_from_openai(part) for part in item.content)
        if role == "assistant":
            return AssistantMessage(
                content=content,
                id=getattr(item, "id", None),
                status=getattr(item, "status", None),
                phase=getattr(item, "phase", None),
                raw_payload=_model_to_dict(item),
            )
        if role == "user":
            return UserMessage(
                content=_text_parts_only(content),
                raw_payload=_model_to_dict(item),
            )
        if role == "system":
            return SystemMessage(
                content=_text_parts_only(content),
                raw_payload=_model_to_dict(item),
            )
    return None


def _require_plain_text(parts: tuple[TextPart, ...]) -> str:
    return "".join(part.text for part in parts)


def _assistant_part_to_openai(part: ContentPart) -> dict[str, Any]:
    if isinstance(part, TextPart):
        return {
            "type": "output_text",
            "text": part.text,
            "annotations": list(part.annotations),
        }
    if isinstance(part, RefusalPart):
        return {
            "type": "refusal",
            "refusal": part.refusal,
        }
    raise ValueError(f"Unsupported content part: {type(part)!r}")


def _part_from_openai(part: object) -> ContentPart:
    part_type = getattr(part, "type", None)
    raw_payload = _model_to_dict(part)
    if part_type in {"text", "output_text", "input_text"}:
        return TextPart(
            text=part.text,
            annotations=tuple(getattr(part, "annotations", ()) or ()),
            raw_payload=raw_payload,
        )
    if part_type == "refusal":
        return RefusalPart(
            refusal=part.refusal,
            raw_payload=raw_payload,
        )
    raise ValueError(f"Unsupported OpenAI content part: {part_type!r}")


def _text_parts_only(parts: tuple[ContentPart, ...]) -> tuple[TextPart, ...]:
    text_parts = tuple(part for part in parts if isinstance(part, TextPart))
    if len(text_parts) != len(parts):
        raise ValueError("System and user messages only support text parts.")
    return text_parts


def _model_to_dict(item: object) -> dict[str, Any] | None:
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if isinstance(item, dict):
        return item
    if hasattr(item, "__dict__"):
        return dict(vars(item))
    return None
