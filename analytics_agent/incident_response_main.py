"""Entry point for verifying the l1 server incident-response tool loop."""

import os

import dotenv

from analytics_agent.messages import generate_initial_messages
from analytics_agent.providers.openai_provider import OpenAIProvider
from analytics_agent.tools import create_incident_response_tools, run_tool_loop


def main() -> None:
    """Investigate a sample incident through the OpenAI tool-calling loop."""
    dotenv.load_dotenv()
    tool_registry, tool_schemas = create_incident_response_tools()

    messages = generate_initial_messages(
        """You are an incident-response agent.
Use the available tools to inspect the reported server before taking action.
Restart a service only when the evidence supports it. Escalate dependency or
unresolved failures. Conclude with a concise summary of evidence and actions.""",
        "Investigate payment-server-01 and resolve the incident.",
    )
    provider = OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        tools=tool_schemas,
        messages=messages,
    )
    run_tool_loop(provider, tool_registry)
