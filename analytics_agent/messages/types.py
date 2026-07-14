"""Canonical typed message models."""

from __future__ import annotations

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
                from analytics_agent.messages.openai import to_openai_input_item

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
