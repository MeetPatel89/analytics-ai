"""Constructors and helpers for canonical chat messages."""

from __future__ import annotations

import json
from typing import Any

from analytics_agent.messages.types import (
    AssistantMessage,
    ChatMessage,
    FunctionCallMessage,
    FunctionCallOutputMessage,
    RefusalPart,
    SystemMessage,
    TextPart,
    UserMessage,
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


def _parse_arguments(arguments_raw: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(arguments_raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
