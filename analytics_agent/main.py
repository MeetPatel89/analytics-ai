"""Entry point for the analytics agent tool-calling loop."""

import json
import os
from pathlib import Path

import dotenv
from openai.types.responses import Response

from analytics_agent.messages import (
    FunctionCallMessage,
    from_openai_output_item,
    generate_initial_messages,
    get_message_text,
)
from analytics_agent.providers.openai_provider import OpenAIProvider
from analytics_agent.tools import DataframeCatalog, configure_tools, load_dataset_specs

dotenv.load_dotenv()

MAX_TURNS = 10

DATA_PATH = str(Path(__file__).resolve().parent / "data")

specs = load_dataset_specs(DATA_PATH)
catalog = DataframeCatalog.from_specs(specs)
tool_registry, tool_schemas = configure_tools(catalog)


def get_output_text(response: Response) -> str | None:
    """Extract the first text message from a model response.

    Parameters
    ----------
    response
        OpenAI Responses API result with an ``output`` list of items.

    Returns
    -------
    str or None
        The first text or output_text content, or None if absent.
    """
    for item in response.output:
        if item.type != "message":
            continue
        message = provider_output_to_text(item)
        if message is not None:
            return message
    return None


def provider_output_to_text(item: object) -> str | None:
    """Extract normalized text from a provider output message item."""
    message = from_openai_output_item(item)
    if message is None:
        return None
    return get_message_text(message)


def run_tool_loop(provider: OpenAIProvider) -> None:
    """Run the tool-calling loop until the model returns a final answer.

    Parameters
    ----------
    provider
        Configured OpenAI provider with tools and conversation history.
    """
    for _ in range(MAX_TURNS):
        response = provider.generate()

        print("--------------------------------")
        print(response.model_dump_json(indent=2))
        print("--------------------------------")

        function_calls = provider.add_response_output(response)
        if not function_calls:
            print("--------------------------------")
            print(provider.serialized_history("openai"))
            print("--------------------------------")
            print(get_output_text(response))
            print("--------------------------------")
            return

        for call in function_calls:
            tool = tool_registry.get(call.name)
            if tool is None:
                output = f"Unknown tool: {call.name}"
            else:
                output = tool(**_tool_arguments(call))
            provider.add_tool_output(call.call_id, str(output))

    print("Max turns reached without a final response.")


def main() -> None:
    """Run the main entry point for the analytics agent tool-calling loop."""
    system_prompt = """
    You are a smart data assistant capable of reading multiple CSV files.
- You have access to 4 different datasets: SaaS Docs, Credit Card Terms,
  Hospital Policy, and Ecommerce FAQs.
- When asked a question, determine which DataFrame is most relevant.
- Do NOT answer from general knowledge.
- Answer in plain English.
    """
    messages = generate_initial_messages(
        system_prompt, "What is the visiting hour in the hospital?"
    )
    provider = OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        tools=tool_schemas,
        messages=messages,
    )
    run_tool_loop(provider)


def _tool_arguments(call: FunctionCallMessage) -> dict[str, object]:
    """Return parsed tool call arguments or fall back to JSON decoding."""
    if call.arguments_parsed is not None:
        return call.arguments_parsed
    return json.loads(call.arguments_raw)
