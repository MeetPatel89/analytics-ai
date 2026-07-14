"""Canonical chat message types and OpenAI serialization helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

Provider = Literal["openai"]
Role = Literal["user", "assistant", "system"]
AssistantStatus = Literal["in_progress", "completed", "incomplete"]
AssistantPhase = Literal["commentary", "final_answer"]


@dataclass(frozen=True)
class TextPart:
    """Normalized text content."""

    text: str
    annotations: tuple[dict[str, Any], ...] = ()
    raw_payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class RefusalPart:
    """Normalized refusal content."""

    refusal: str
    raw_payload: dict[str, Any] | None = None


ContentPart = TextPart | RefusalPart


@dataclass(frozen=True)
class BaseMessage:
    """Shared message metadata."""

    raw_payload: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    def to_provider_message(self, provider: Provider) -> dict[str, Any]:
        """Convert this message to a provider-specific API payload."""
        match provider:
            case "openai":
                return to_openai_input_item(self)


@dataclass(frozen=True)
class SystemMessage(BaseMessage):
    """System instruction message."""

    content: tuple[TextPart, ...] = field(default_factory=tuple)
    role: Literal["system"] = "system"


@dataclass(frozen=True)
class UserMessage(BaseMessage):
    """User input message."""

    content: tuple[TextPart, ...] = field(default_factory=tuple)
    role: Literal["user"] = "user"


@dataclass(frozen=True)
class AssistantMessage(BaseMessage):
    """Assistant output message."""

    content: tuple[ContentPart, ...] = field(default_factory=tuple)
    id: str | None = None
    status: AssistantStatus | None = None
    phase: AssistantPhase | None = None
    role: Literal["assistant"] = "assistant"


@dataclass(frozen=True)
class FunctionCallMessage(BaseMessage):
    """Assistant tool/function call."""

    call_id: str = ""
    name: str = ""
    arguments_raw: str = ""
    arguments_parsed: dict[str, Any] | None = None
    id: str | None = None
    status: str | None = None
    type: Literal["function_call"] = "function_call"


@dataclass(frozen=True)
class FunctionCallOutputMessage(BaseMessage):
    """Tool/function output returned to the model."""

    call_id: str = ""
    output: str = ""
    type: Literal["function_call_output"] = "function_call_output"


ChatMessage = (
    SystemMessage
    | UserMessage
    | AssistantMessage
    | FunctionCallMessage
    | FunctionCallOutputMessage
)


def text_part(text: str) -> TextPart:
    """Create a text content part."""
    return TextPart(text=text)


def refusal_part(refusal: str) -> RefusalPart:
    """Create a refusal content part."""
    return RefusalPart(refusal=refusal)


def system_message(prompt: str) -> SystemMessage:
    """Create a system message with the given prompt."""
    return SystemMessage(content=(text_part(prompt),))


def user_message(prompt: str) -> UserMessage:
    """Create a user message with the given prompt."""
    return UserMessage(content=(text_part(prompt),))


def assistant_message(prompt: str) -> AssistantMessage:
    """Create an assistant message with the given prompt."""
    return AssistantMessage(content=(text_part(prompt),))


def function_call_message(
    call_id: str,
    name: str,
    arguments_raw: str,
    *,
    id: str | None = None,
    status: str | None = None,
    metadata: dict[str, Any] | None = None,
    raw_payload: dict[str, Any] | None = None,
) -> FunctionCallMessage:
    """Create a function call message with parsed arguments when possible."""
    return FunctionCallMessage(
        call_id=call_id,
        name=name,
        arguments_raw=arguments_raw,
        arguments_parsed=_parse_arguments(arguments_raw),
        id=id,
        status=status,
        metadata=metadata,
        raw_payload=raw_payload,
    )


def function_call_output_message(
    call_id: str,
    output: str,
    *,
    metadata: dict[str, Any] | None = None,
    raw_payload: dict[str, Any] | None = None,
) -> FunctionCallOutputMessage:
    """Create a function call output message."""
    return FunctionCallOutputMessage(
        call_id=call_id,
        output=output,
        metadata=metadata,
        raw_payload=raw_payload,
    )


def generate_initial_messages(
    system_prompt: str, user_prompt: str
) -> list[ChatMessage]:
    """Generate the initial messages for the conversation."""
    return [
        system_message(system_prompt),
        user_message(user_prompt),
    ]


def get_message_text(message: ChatMessage) -> str | None:
    """Extract concatenated text content from a message."""
    if not isinstance(message, (SystemMessage, UserMessage, AssistantMessage)):
        return None

    text_segments = [
        part.text for part in message.content if isinstance(part, TextPart)
    ]
    if not text_segments:
        return None
    return "".join(text_segments)


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


def _parse_arguments(arguments_raw: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(arguments_raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _require_plain_text(parts: tuple[TextPart, ...]) -> str:
    return "".join(part.text for part in parts)


def _assistant_part_to_openai(part: ContentPart) -> dict[str, Any]:
    if isinstance(part, TextPart):
        return {
            "type": "output_text",
            "text": part.text,
            "annotations": list(part.annotations),
        }
    return {
        "type": "refusal",
        "refusal": part.refusal,
    }


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
