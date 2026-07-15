"""Composition root for dataframe tools."""

from analytics_agent.tools.dataframe.catalog import DataframeCatalog
from analytics_agent.tools.dataframe.dataframe_tools import DataframeTools
from analytics_agent.tools.dataframe.dataframe_tools_models import (
    AggregateRowsInput,
    DescribeDataframeInput,
    DistinctValuesInput,
    FilterRowsInput,
    ListDataframesInput,
    PreviewDataframeInput,
    SearchRowsInput,
)
from analytics_agent.tools.provider_factories import (
    OpenAIToolSchema,
    create_openai_tools,
)
from analytics_agent.tools.registry import ToolDefinition, ToolRegistry


def build_dataframe_definitions(catalog: DataframeCatalog) -> list[ToolDefinition]:
    """Pair catalog-bound dataframe operations with their input contracts."""
    tools = DataframeTools(catalog)
    return [
        ToolDefinition(tools.list_dataframes, ListDataframesInput),
        ToolDefinition(tools.describe_dataframe, DescribeDataframeInput),
        ToolDefinition(tools.preview_dataframe, PreviewDataframeInput),
        ToolDefinition(tools.search_rows, SearchRowsInput),
        ToolDefinition(tools.filter_rows, FilterRowsInput),
        ToolDefinition(tools.aggregate_rows, AggregateRowsInput),
        ToolDefinition(tools.distinct_values, DistinctValuesInput),
    ]


def create_dataframe_tools(
    catalog: DataframeCatalog,
) -> tuple[ToolRegistry, list[OpenAIToolSchema]]:
    """Create validated dataframe tools and OpenAI schemas."""
    return create_openai_tools(build_dataframe_definitions(catalog))
