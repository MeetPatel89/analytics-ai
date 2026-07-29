"""Composition and display metadata for selectable tool chains."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from analytics_agent.filesystem import LocationCatalog, load_location_catalog
from analytics_agent.providers.generation import GenerationModel
from analytics_agent.tools.filesystem_analytics.registry import (
    build_filesystem_definitions,
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

    FILESYSTEM_ANALYTICS = "filesystem_analytics"
    INCIDENT_RESPONSE = "incident_response"


@dataclass(frozen=True)
class ToolChainInfo:
    """Human-readable metadata for a selectable tool chain."""

    chain: ToolChain
    label: str
    description: str
    tool_names: tuple[str, ...]


GenerationModelFactory = Callable[[], GenerationModel]


@dataclass(frozen=True)
class ToolChainDependencies:
    """Lazy runtime dependencies shared by selectable tool-chain builders."""

    create_generation_model: GenerationModelFactory | None = None
    data_path: Path | None = None
    locations_path: Path | None = None
    location_catalog: LocationCatalog | None = None


_TOOL_CHAIN_INFO = {
    ToolChain.FILESYSTEM_ANALYTICS: ToolChainInfo(
        chain=ToolChain.FILESYSTEM_ANALYTICS,
        label="Filesystem analytics",
        description="Navigate and query configured local or ADLS data roots.",
        tool_names=(
            "list_locations",
            "list_directory",
            "get_file_info",
            "inspect_schema",
            "preview_data",
            "read_text_file",
            "query_data",
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
    dependencies: ToolChainDependencies,
) -> tuple[ToolRegistry, list[OpenAIToolSchema]]:
    """Compose selected tool chains into one validated OpenAI tool set."""
    definitions: list[ToolDefinition] = []
    for chain in _unique_chains(chains):
        definitions.extend(_TOOL_CHAIN_BUILDERS[chain](dependencies))

    return create_openai_tools(definitions)


def _build_filesystem_tools(
    dependencies: ToolChainDependencies,
) -> list[ToolDefinition]:
    catalog = dependencies.location_catalog or load_location_catalog(
        dependencies.locations_path,
        data_path=dependencies.data_path,
    )
    return build_filesystem_definitions(catalog)


def _build_incident_response_tools(
    dependencies: ToolChainDependencies,
) -> list[ToolDefinition]:
    del dependencies
    return build_incident_response_definitions()


ToolChainBuilder = Callable[[ToolChainDependencies], list[ToolDefinition]]
_TOOL_CHAIN_BUILDERS: dict[ToolChain, ToolChainBuilder] = {
    ToolChain.FILESYSTEM_ANALYTICS: _build_filesystem_tools,
    ToolChain.INCIDENT_RESPONSE: _build_incident_response_tools,
}


def default_system_prompt(chains: tuple[ToolChain, ...]) -> str:
    """Build a safe default system prompt for the selected tool chains."""
    selected = _unique_chains(chains)
    prompts: list[str] = []
    if ToolChain.FILESYSTEM_ANALYTICS in selected:
        prompts.append(
            "You are a filesystem analytics assistant with read-only access to "
            "configured data locations. Start by discovering locations and files, "
            "inspect schemas before querying unfamiliar data, and ground every "
            "factual answer in tool results. Use relative paths only. Use query_data "
            "for bounded SELECT-only analytics over CSV or Parquet source aliases."
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
    if selected == (ToolChain.FILESYSTEM_ANALYTICS,):
        return (
            "List the available data locations, find tabular files, and summarize "
            "one useful result from the available data."
        )
    if selected == (ToolChain.INCIDENT_RESPONSE,):
        return "Investigate payment-server-01 and resolve the incident."
    if ToolChain.FILESYSTEM_ANALYTICS in selected:
        return (
            "Discover the configured data files and summarize one useful, "
            "evidence-backed result."
        )
    return "Investigate payment-server-01 and summarize the evidence."


def _unique_chains(chains: tuple[ToolChain, ...]) -> tuple[ToolChain, ...]:
    if not chains:
        raise ValueError("At least one tool chain must be selected.")
    return tuple(dict.fromkeys(chains))
