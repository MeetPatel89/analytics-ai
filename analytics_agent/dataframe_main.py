"""Entry point for the analytics agent tool-calling loop."""

import os

import dotenv

from analytics_agent.messages import generate_initial_messages
from analytics_agent.observability import configure_tracing
from analytics_agent.providers.openai_provider import (
    OpenAIGenerationModel,
    OpenAIProvider,
)
from analytics_agent.tools import (
    ToolChain,
    ToolChainDependencies,
    build_tools_for_chains,
    default_system_prompt,
    default_user_prompt,
    run_tool_loop,
)
from analytics_agent.tools.tool_loop import DEFAULT_MAX_TURNS


def main() -> None:
    """Run the main entry point for the analytics agent tool-calling loop."""
    dotenv.load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = "gpt-4o-mini"
    tool_chains = (ToolChain.DATAFRAME,)
    tracer = configure_tracing()

    with tracer.start_as_current_span(
        "agent_run",
        attributes={
            "agent.model": model,
            "agent.tool_chains": tuple(chain.value for chain in tool_chains),
            "agent.max_turns": DEFAULT_MAX_TURNS,
        },
    ):
        tool_registry, tool_schemas = build_tools_for_chains(
            tool_chains,
            ToolChainDependencies(
                create_generation_model=lambda: OpenAIGenerationModel(
                    api_key,
                    model,
                )
            ),
        )
        messages = generate_initial_messages(
            default_system_prompt(tool_chains),
            default_user_prompt(tool_chains),
        )
        provider = OpenAIProvider(
            api_key=api_key,
            model=model,
            tools=tool_schemas,
            messages=messages,
        )
        run_tool_loop(provider, tool_registry)
