"""Composition and display metadata for selectable tool chains."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from analytics_agent.tools.dataframe.catalog import DataframeCatalog, load_dataset_specs
from analytics_agent.tools.dataframe.dataframe_registry import (
    build_dataframe_definitions,
)
from analytics_agent.tools.incident_response.registry import (
    build_incident_response_definitions,
)
from analytics_agent.tools.provider_factories import (
    OpenAIToolSchema,
    create_openai_tools,
)
from analytics_agent.tools.registry import ToolDefinition, ToolRegistry


class ToolChain(StrEnum):
    """Tool collections selectable by an agent run."""

    DATAFRAME = "dataframe"
    INCIDENT_RESPONSE = "incident_response"


@dataclass(frozen=True)
class ToolChainInfo:
    """Human-readable metadata for a selectable tool chain."""

    chain: ToolChain
    label: str
    description: str
    tool_names: tuple[str, ...]


_TOOL_CHAIN_INFO = {
    ToolChain.DATAFRAME: ToolChainInfo(
        chain=ToolChain.DATAFRAME,
        label="Dataframe analysis",
        description="Query locally configured CSV datasets.",
        tool_names=(
            "list_dataframes",
            "describe_dataframe",
            "preview_dataframe",
            "search_rows",
            "filter_rows",
            "aggregate_rows",
            "distinct_values",
        ),
    ),
    ToolChain.INCIDENT_RESPONSE: ToolChainInfo(
        chain=ToolChain.INCIDENT_RESPONSE,
        label="Incident response",
        description=(
            "Inspect simulated server health and logs, then restart or escalate."
        ),
        tool_names=(
            "get_server_health",
            "fetch_recent_logs",
            "restart_service",
            "escalate_incident",
        ),
    ),
}


def available_tool_chains() -> tuple[ToolChainInfo, ...]:
    """Return tool-chain metadata in display order."""
    return tuple(_TOOL_CHAIN_INFO.values())


def build_tools_for_chains(
    chains: tuple[ToolChain, ...],
    *,
    data_path: Path | None = None,
) -> tuple[ToolRegistry, list[OpenAIToolSchema]]:
    """Compose selected tool chains into one validated OpenAI tool set."""
    definitions: list[ToolDefinition] = []
    for chain in _unique_chains(chains):
        if chain is ToolChain.DATAFRAME:
            path = data_path or Path(__file__).resolve().parents[1] / "data"
            catalog = DataframeCatalog.from_specs(load_dataset_specs(str(path)))
            definitions.extend(build_dataframe_definitions(catalog))
        else:
            definitions.extend(build_incident_response_definitions())

    return create_openai_tools(definitions)


def default_system_prompt(chains: tuple[ToolChain, ...]) -> str:
    """Build a safe default system prompt for the selected tool chains."""
    selected = _unique_chains(chains)
    prompts: list[str] = []
    if ToolChain.DATAFRAME in selected:
        prompts.append(
            "You are a data assistant with access to configured CSV datasets. "
            "Use dataframe tools for factual answers, do not answer from general "
            "knowledge, and respond in plain English."
        )
    if ToolChain.INCIDENT_RESPONSE in selected:
        prompts.append(
            "You are an incident-response agent. Inspect server health and logs "
            "before taking action, restart only when evidence supports it, and "
            "escalate unresolved dependency failures. Summarize evidence and actions."
        )
    if len(selected) > 1:
        prompts.append("Use only the tool chain relevant to the user's request.")
    return "\n\n".join(prompts)


def default_user_prompt(chains: tuple[ToolChain, ...]) -> str:
    """Return a starter task for the selected tool chains."""
    selected = _unique_chains(chains)
    if selected == (ToolChain.DATAFRAME,):
        return "What are the visiting hours in the hospital?"
    if selected == (ToolChain.INCIDENT_RESPONSE,):
        return "Investigate payment-server-01 and resolve the incident."
    return (
        "Investigate payment-server-01. Use dataframe tools only when they provide "
        "relevant supporting information."
    )


def _unique_chains(chains: tuple[ToolChain, ...]) -> tuple[ToolChain, ...]:
    if not chains:
        raise ValueError("At least one tool chain must be selected.")
    return tuple(dict.fromkeys(chains))
