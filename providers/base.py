from abc import ABC, abstractmethod
from typing import Any, Literal

from messages import ChatMessage

Provider = Literal["openai"]


class BaseProvider(ABC):
    """Base class for all providers."""

    def __init__(self) -> None:
        self._history: list[dict[str, Any]] = []

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def add_items(self, *items: dict[str, Any]) -> None:
        self._history.extend(items)

    @abstractmethod
    def add_message(self, message: ChatMessage) -> None: ...

    @abstractmethod
    def generate(self): ...
