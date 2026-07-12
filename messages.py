from dataclasses import dataclass
from typing import Literal, Any

Provider = Literal["openai"]
Role = Literal["user", "assistant", "system"]

@dataclass(frozen=True)
class ChatMessage:
    """A message in a chat conversation."""

    role: Role
    content: str

    def to_provider_message(self, provider: Provider) -> dict[str, Any]:
        if provider == "openai":
            return {
                "role": self.role,
                "content": self.content,
            }
        raise ValueError(f"Unsupported provider: {provider}")
