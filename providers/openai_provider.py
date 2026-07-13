from dataclasses import dataclass
from typing import Any, Literal

from openai import OpenAI

from messages import ChatMessage
from providers.base import BaseProvider

AssistantStatus = Literal["in_progress", "completed", "incomplete"]
AssistantPhase = Literal["commentary", "final_answer"]


@dataclass(frozen=True)
class FunctionCallHistory:
    call_id: str
    name: str
    arguments: str

    @classmethod
    def from_response(cls, item: Any) -> FunctionCallHistory:
        return cls(
            call_id=item.call_id,
            name=item.name,
            arguments=item.arguments,
        )

    def to_provider_item(self) -> dict[str, Any]:
        return {
            "type": "function_call",
            "call_id": self.call_id,
            "name": self.name,
            "arguments": self.arguments,
        }


@dataclass(frozen=True)
class FunctionCallOutputHistory:
    call_id: str
    output: str

    def to_provider_item(self) -> dict[str, Any]:
        return {
            "type": "function_call_output",
            "call_id": self.call_id,
            "output": self.output,
        }


@dataclass(frozen=True)
class OutputTextContent:
    text: str

    def to_provider_item(self) -> dict[str, Any]:
        return {
            "type": "output_text",
            "text": self.text,
            "annotations": [],
        }


@dataclass(frozen=True)
class RefusalContent:
    refusal: str

    def to_provider_item(self) -> dict[str, Any]:
        return {
            "type": "refusal",
            "refusal": self.refusal,
        }


@dataclass(frozen=True)
class AssistantMessageHistory:
    content: tuple[OutputTextContent | RefusalContent, ...]
    id: str | None = None
    status: AssistantStatus | None = None
    phase: AssistantPhase | None = None

    @classmethod
    def from_response(cls, item: Any) -> AssistantMessageHistory:
        content: list[OutputTextContent | RefusalContent] = []
        for part in item.content:
            if part.type in ("text", "output_text"):
                content.append(OutputTextContent(text=part.text))
            elif part.type == "refusal":
                content.append(RefusalContent(refusal=part.refusal))

        return cls(
            content=tuple(content),
            id=item.id,
            status=item.status,
            phase=item.phase,
        )

    def to_provider_item(self) -> dict[str, Any]:
        history_item: dict[str, Any] = {
            "type": "message",
            "role": "assistant",
            "content": [part.to_provider_item() for part in self.content],
        }
        if self.id:
            history_item["id"] = self.id
        if self.status:
            history_item["status"] = self.status
        if self.phase:
            history_item["phase"] = self.phase
        return history_item


HistoryItem = FunctionCallHistory | FunctionCallOutputHistory | AssistantMessageHistory


def history_item_from_response(item: Any) -> HistoryItem | None:
    if item.type == "function_call":
        return FunctionCallHistory.from_response(item)
    if item.type == "message":
        return AssistantMessageHistory.from_response(item)
    return None


class OpenAIProvider(BaseProvider):
    """OpenAI Provider."""

    def __init__(
        self,
        api_key: str,
        model: str,
        tools: list[dict] | None = None,
        messages: list[ChatMessage] | None = None,
    ):
        super().__init__()
        if not api_key:
            raise ValueError("API key is required")
        if not model:
            raise ValueError("Model is required")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.tools = tools
        for message in messages or []:
            self.add_message(message)

    def add_message(self, message: ChatMessage) -> None:
        self.add_items(message.to_provider_message("openai"))

    def add_response_output(self, response: Any) -> list[FunctionCallHistory]:
        """Append model output items to history and return any function calls."""
        function_calls: list[FunctionCallHistory] = []
        for item in response.output:
            history_item = history_item_from_response(item)
            if history_item is None:
                continue
            self.add_items(history_item.to_provider_item())
            if isinstance(history_item, FunctionCallHistory):
                function_calls.append(history_item)
        return function_calls

    def add_tool_output(self, call_id: str, output: str) -> None:
        self.add_items(
            FunctionCallOutputHistory(call_id=call_id, output=output).to_provider_item()
        )

    def generate(self):
        return self.client.responses.create(
            model=self.model,
            tools=self.tools,
            input=self._history,
        )
