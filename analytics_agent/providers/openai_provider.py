"""OpenAI provider implementation backed by the Responses API."""

from types import SimpleNamespace

from openai import OpenAI
from openai.types.responses import Response

from analytics_agent.messages import (
    ChatMessage,
    FunctionCallMessage,
    from_openai_output_item,
    function_call_output_message,
)
from analytics_agent.providers.base import BaseProvider


def list_available_models(api_key: str) -> list[str]:
    """Return the OpenAI model IDs available to the supplied API key."""
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required to list available models.")

    try:
        models = OpenAI(api_key=api_key).models.list()
    except Exception as exc:
        raise RuntimeError(f"Unable to list OpenAI models: {exc}") from exc

    return sorted({model.id for model in models.data})


class OpenAIProvider(BaseProvider):
    """OpenAI Provider."""

    def __init__(
        self,
        api_key: str,
        model: str,
        tools: list[dict] | None = None,
        messages: list[ChatMessage] | None = None,
    ) -> None:
        super().__init__()
        if not api_key:
            raise ValueError("API key is required")
        if not model:
            raise ValueError("Model is required")
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._tools = list(tools) if tools is not None else None
        for message in messages or []:
            self.add_message(message)

    def add_response_output(
        self, response: Response | SimpleNamespace
    ) -> list[FunctionCallMessage]:
        """Append model output items to history and return any function calls."""
        function_calls: list[FunctionCallMessage] = []
        for item in response.output:
            message = from_openai_output_item(item)
            if message is None:
                continue
            self.add_items(message)
            if isinstance(message, FunctionCallMessage):
                function_calls.append(message)
        return function_calls

    def add_tool_output(self, call_id: str, output: str) -> None:
        """Append tool execution output to history."""
        self.add_items(function_call_output_message(call_id=call_id, output=output))

    def generate(self) -> Response:
        """Send the current conversation history to OpenAI."""
        return self._client.responses.create(
            model=self._model,
            tools=self._tools,
            input=self.serialized_history("openai"),
        )
