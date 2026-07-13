"""Chat message types and provider serialization helpers."""

from dataclasses import dataclass
from typing import Any, Literal

Provider = Literal["openai"]
Role = Literal["user", "assistant", "system"]


@dataclass(frozen=True)
class ChatMessage:
    """A message in a chat conversation."""

    role: Role
    content: str

    def to_provider_message(self, provider: Provider) -> dict[str, Any]:
        """Convert this message to a provider-specific API payload.

        Parameters
        ----------
        provider : Provider
            The LLM provider to serialize for.

        Returns
        -------
        dict[str, Any]
            Provider-ready message dictionary.

        Raises
        ------
        ValueError
            If ``provider`` is not supported.
        """
        if provider == "openai":
            return {
                "role": self.role,
                "content": self.content,
            }
        raise ValueError(f"Unsupported provider: {provider}")
