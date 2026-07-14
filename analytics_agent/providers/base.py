"""Provider base interfaces and shared history handling."""

from abc import ABC, abstractmethod
from typing import Any, Literal

from analytics_agent.messages import ChatMessage

Provider = Literal["openai"]


class BaseProvider(ABC):
    """Base class for all providers."""

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

    @abstractmethod
    def add_message(self, message: ChatMessage) -> None:
        """Append a canonical message to history."""

    @abstractmethod
    def generate(self) -> object:
        """Issue a model request and return the provider response."""
