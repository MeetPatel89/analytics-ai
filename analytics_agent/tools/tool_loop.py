"""Shared OpenAI tool-calling loop."""

import json

from analytics_agent.messages import (
    FunctionCallMessage,
    from_openai_output_item,
    get_message_text,
)
from analytics_agent.providers.base import (
    ProviderResponseT,
    ToolLoopProvider,
    ToolLoopResponse,
)
from analytics_agent.tools.registry import ToolRegistry

DEFAULT_MAX_TURNS = 10


def run_tool_loop(
    provider: ToolLoopProvider[ProviderResponseT],
    tool_registry: ToolRegistry,
    max_turns: int = DEFAULT_MAX_TURNS,
    *,
    verbose: bool = False,
) -> None:
    """Run model and tool turns until the model returns a final answer."""
    for _ in range(max_turns):
        response = provider.generate()

        if verbose:
            print("--------------------------------")
            print(response.model_dump_json(indent=2))
            print("--------------------------------")

        function_calls = provider.add_response_output(response)
        if not function_calls:
            if verbose:
                print("--------------------------------")
                print(provider.serialized_history("openai"))
                print("--------------------------------")
            print(f"Final answer: {get_output_text(response)}")
            return

        for call in function_calls:
            try:
                arguments = _tool_arguments(call)
            except ToolArgumentsError as exc:
                print(f"Calling tool: {call.name}(<invalid arguments>)")
                output = str(exc)
            else:
                print(f"Calling tool: {call.name}({json.dumps(arguments)})")
                output = tool_registry.execute(call.name, arguments)
            print(f"Tool result: {output}")
            provider.add_tool_output(call.call_id, output)

    print("Max turns reached without a final response.")


def get_output_text(response: ToolLoopResponse) -> str | None:
    """Extract the first text message from a model response."""
    for item in response.output:
        if getattr(item, "type", None) != "message":
            continue
        message = from_openai_output_item(item)
        if message is not None:
            return get_message_text(message)
    return None


def _tool_arguments(call: FunctionCallMessage) -> dict[str, object]:
    if call.arguments_parsed is not None:
        return call.arguments_parsed
    try:
        arguments = json.loads(call.arguments_raw)
    except json.JSONDecodeError as exc:
        raise ToolArgumentsError(
            f"Invalid arguments for tool '{call.name}': arguments must be valid JSON."
        ) from exc
    if not isinstance(arguments, dict):
        raise ToolArgumentsError(
            f"Invalid arguments for tool '{call.name}': "
            "arguments must be a JSON object."
        )
    return arguments


class ToolArgumentsError(ValueError):
    """Raised when a provider returns an unusable tool argument payload."""
