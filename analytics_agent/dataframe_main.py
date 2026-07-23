"""Entry point for the analytics agent tool-calling loop."""

import os

import dotenv

from analytics_agent.messages import generate_initial_messages
from analytics_agent.providers.openai_provider import OpenAIProvider
from analytics_agent.tools import (
    ToolChain,
    build_tools_for_chains,
    default_system_prompt,
    default_user_prompt,
    run_tool_loop,
)


def main() -> None:
    """Run the main entry point for the analytics agent tool-calling loop."""
    dotenv.load_dotenv()
    tool_chains = (ToolChain.DATAFRAME,)
    tool_registry, tool_schemas = build_tools_for_chains(tool_chains)
    messages = generate_initial_messages(
        default_system_prompt(tool_chains),
        default_user_prompt(tool_chains),
    )
    provider = OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        tools=tool_schemas,
        messages=messages,
    )
    run_tool_loop(provider, tool_registry)
