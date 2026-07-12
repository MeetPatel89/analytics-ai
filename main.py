import json
import os

import dotenv

dotenv.load_dotenv()

from messages import ChatMessage
from providers.openai_provider import OpenAIProvider
from tools import TOOL_FUNCTIONS, TOOLS

MAX_TURNS = 10


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
            tool = TOOL_FUNCTIONS.get(call.name)
            if tool is None:
                output = f"Unknown tool: {call.name}"
            else:
                output = tool(**json.loads(call.arguments))
            provider.add_tool_output(call.call_id, str(output))

    print("Max turns reached without a final response.")


if __name__ == "__main__":
    provider = OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        tools=TOOLS,
        messages=[
            ChatMessage(
                role="user",
                content="What is my horoscope? I am an Aquarius.",
            ),
        ],
    )
    run_tool_loop(provider)
