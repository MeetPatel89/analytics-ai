"""Models for incident response tools."""

from pydantic import Field

from analytics_agent.tools.registry import ToolInput


class GetServerHealthInput(ToolInput):
    """Arguments for getting server health."""

    server_id: str = Field(description="The ID of the server to get health for.")


class FetchRecentLogsInput(ToolInput):
    """Arguments for fetching recent logs."""

    server_id: str = Field(description="The ID of the server to fetch logs for.")
    lines: int = Field(
        default=5, ge=1, le=100, description="The number of lines of logs to fetch."
    )


class RestartServiceInput(ToolInput):
    """Arguments for restarting a service."""

    server_id: str = Field(description="The ID of the server to restart.")


class EscalateIncidentInput(ToolInput):
    """Arguments for escalating an incident."""

    server_id: str = Field(description="The ID of the server to escalate.")
