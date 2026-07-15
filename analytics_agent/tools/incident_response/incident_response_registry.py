"""Composition root for incident-response tools."""

from analytics_agent.tools.incident_response.incident_response_tools import (
    escalate_incident,
    fetch_recent_logs,
    get_server_health,
    restart_service,
)
from analytics_agent.tools.incident_response.incident_response_tools_models import (
    EscalateIncidentInput,
    FetchRecentLogsInput,
    GetServerHealthInput,
    RestartServiceInput,
)
from analytics_agent.tools.provider_factories import (
    OpenAIToolSchema,
    create_openai_tools,
)
from analytics_agent.tools.registry import ToolDefinition, ToolRegistry


def build_incident_response_definitions() -> list[ToolDefinition]:
    """Pair incident-response implementations with their input contracts."""
    return [
        ToolDefinition(get_server_health, GetServerHealthInput),
        ToolDefinition(fetch_recent_logs, FetchRecentLogsInput),
        ToolDefinition(restart_service, RestartServiceInput),
        ToolDefinition(escalate_incident, EscalateIncidentInput),
    ]


def create_incident_response_tools() -> tuple[ToolRegistry, list[OpenAIToolSchema]]:
    """Create validated incident-response tools and OpenAI schemas."""
    return create_openai_tools(build_incident_response_definitions())
