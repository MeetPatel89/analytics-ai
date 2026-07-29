"""OpenAI provider implementation backed by the Responses API."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

from openai import OpenAI
from openai.types.responses import Response
from opentelemetry import trace

from analytics_agent.messages import (
    ChatMessage,
    FunctionCallMessage,
    from_openai_output_item,
    function_call_output_message,
)
from analytics_agent.providers.base import BaseProvider
from analytics_agent.tools.provider_factories import OpenAIToolSchema

_TRACER = trace.get_tracer(__name__)


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
        tools: Sequence[OpenAIToolSchema] | None = None,
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
        input_items = self.serialized_history("openai")
        with _TRACER.start_as_current_span(
            "llm.generate",
            attributes={
                "llm.model": self._model,
                "llm.request.input_item_count": len(input_items),
                "llm.request.tool_count": len(self._tools or ()),
            },
        ) as span:
            response = self._client.responses.create(
                model=self._model,
                tools=self._tools,
                input=input_items,
            )
            output_items = getattr(response, "output", ()) or ()
            span.set_attribute(
                "llm.response.output_item_count",
                len(output_items),
            )
            response_id = getattr(response, "id", None)
            if response_id:
                span.set_attribute("llm.response.id", response_id)
            response_model = getattr(response, "model", None)
            if response_model:
                span.set_attribute("llm.response.model", response_model)
            return response
