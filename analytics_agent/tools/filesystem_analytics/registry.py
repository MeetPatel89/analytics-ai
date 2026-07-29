"""Composition root for filesystem analytics tools."""

from analytics_agent.filesystem import LocationCatalog
from analytics_agent.tools.filesystem_analytics.models import (
    GetFileInfoInput,
    InspectSchemaInput,
    ListDirectoryInput,
    ListLocationsInput,
    PreviewDataInput,
    QueryDataInput,
    ReadTextFileInput,
)
from analytics_agent.tools.filesystem_analytics.tools import FilesystemAnalyticsTools
from analytics_agent.tools.provider_factories import (
    OpenAIToolSchema,
    create_openai_tools,
)
from analytics_agent.tools.registry import ToolDefinition, ToolRegistry


def build_filesystem_definitions(
    catalog: LocationCatalog,
) -> list[ToolDefinition]:
    """Pair filesystem operations with strict validated input contracts."""
    tools = FilesystemAnalyticsTools(catalog)
    return [
        ToolDefinition(tools.list_locations, ListLocationsInput),
        ToolDefinition(tools.list_directory, ListDirectoryInput),
        ToolDefinition(tools.get_file_info, GetFileInfoInput),
        ToolDefinition(tools.inspect_schema, InspectSchemaInput),
        ToolDefinition(tools.preview_data, PreviewDataInput),
        ToolDefinition(tools.read_text_file, ReadTextFileInput),
        ToolDefinition(tools.query_data, QueryDataInput),
    ]


def create_filesystem_analytics_tools(
    catalog: LocationCatalog,
) -> tuple[ToolRegistry, list[OpenAIToolSchema]]:
    """Create executable filesystem tools and matching OpenAI schemas."""
    return create_openai_tools(build_filesystem_definitions(catalog))
