"""Provider interfaces and shared history handling."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal, Protocol, TypeVar

from analytics_agent.messages import ChatMessage, FunctionCallMessage

Provider = Literal["openai"]


class ToolLoopResponse(Protocol):
    """Response behavior required by the shared tool loop."""

    @property
    def output(self) -> Sequence[object]:
        """Provider output items for the current turn."""
        ...

    def model_dump_json(self, *, indent: int | None = None) -> str:
        """Serialize provider diagnostics for verbose output."""
        ...


ProviderResponseT = TypeVar("ProviderResponseT", bound=ToolLoopResponse)


class ToolLoopProvider(Protocol[ProviderResponseT]):
    """Focused provider boundary consumed by the shared tool loop."""

    def generate(self) -> ProviderResponseT:
        """Issue a model request and return the provider response."""
        ...

    def add_response_output(
        self, response: ProviderResponseT
    ) -> list[FunctionCallMessage]:
        """Add response items to history and return any function calls."""
        ...

    def add_tool_output(self, call_id: str, output: str) -> None:
        """Add one tool result to conversation history."""
        ...

    def serialized_history(self, provider: Provider) -> Sequence[object]:
        """Serialize provider history for verbose diagnostics."""
        ...


class BaseProvider(ABC):
    """Store provider-neutral conversation history."""

    def __init__(self) -> None:
        self._history: list[ChatMessage] = []

    @property
    def history(self) -> list[ChatMessage]:
        """Canonical conversation history."""
        return list(self._history)

    def add_items(self, *items: ChatMessage) -> None:
        """Append canonical messages to provider history."""
        self._history.extend(items)

    def serialized_history(self, provider: Provider) -> list[dict[str, Any]]:
        """Serialize history for a concrete provider request."""
        return [message.to_provider_message(provider) for message in self._history]

    def add_message(self, message: ChatMessage) -> None:
        """Append a canonical message to history."""
        self.add_items(message)

    @abstractmethod
    def generate(self) -> object:
        """Issue a model request and return the provider response."""
