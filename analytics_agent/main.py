"""Entry point for the analytics agent tool-calling loop."""

import os
from pathlib import Path

import dotenv

from analytics_agent.messages import generate_initial_messages
from analytics_agent.providers.openai_provider import OpenAIProvider
from analytics_agent.tool_loop import run_tool_loop
from analytics_agent.tools import (
    DataframeCatalog,
    create_dataframe_tools,
    load_dataset_specs,
)

dotenv.load_dotenv()

DATA_PATH = str(Path(__file__).resolve().parent / "data")

specs = load_dataset_specs(DATA_PATH)
catalog = DataframeCatalog.from_specs(specs)
tool_registry, tool_schemas = create_dataframe_tools(catalog)


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
    run_tool_loop(provider, tool_registry)
