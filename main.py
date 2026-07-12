import json
import os

import dotenv

dotenv.load_dotenv()

from messages import ChatMessage
from providers.openai_provider import OpenAIProvider
from tools import DataframeCatalog, load_dataset_specs, configure_tools


MAX_TURNS = 10

DATA_PATH = "data/"

specs = load_dataset_specs(DATA_PATH)
catalog = DataframeCatalog.from_specs(specs)
tool_registry, tool_schemas = configure_tools(catalog)

def get_output_text(response) -> str | None:
    for item in response.output:
        if item.type != "message":
            continue
        for content in item.content:
            if content.type in ("text", "output_text"):
                return content.text
    return None


def run_tool_loop(provider: OpenAIProvider) -> None:
    for _ in range(MAX_TURNS):
        response = provider.generate()

        print("--------------------------------")
        print(response.model_dump_json(indent=2))
        print("--------------------------------")

        function_calls = provider.add_response_output(response)
        if not function_calls:
            print("--------------------------------")
            print(provider.history)
            print("--------------------------------")
            print(get_output_text(response))
            print("--------------------------------")
            return

        for call in function_calls:
            tool = tool_registry.get(call.name)
            if tool is None:
                output = f"Unknown tool: {call.name}"
            else:
                output = tool(**json.loads(call.arguments))
            provider.add_tool_output(call.call_id, str(output))

    print("Max turns reached without a final response.")


if __name__ == "__main__":
    system_prompt = """
    You are a smart data assistant capable of reading multiple CSV files.
- You have access to 4 different datasets: SaaS Docs, Credit Card Terms, Hospital Policy, and Ecommerce FAQs.
- When asked a question, determine which DataFrame is most relevant.
- Do NOT answer from general knowledge.
- Answer in plain English.
    """
    messages = [
        ChatMessage(
            role="system",
            content=system_prompt,
        ),
        ChatMessage(
            role="user",
            content="What is the visiting hour in the hospital?",
        ),
    ]
    provider = OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        tools=tool_schemas,
        messages=messages,
    )
    run_tool_loop(provider)
