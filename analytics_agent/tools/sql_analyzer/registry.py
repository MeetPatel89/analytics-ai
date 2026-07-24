"""Composition root for SQL analyzer tools."""

from pathlib import Path

from analytics_agent.providers.generation import GenerationModel
from analytics_agent.tools.provider_factories import (
    OpenAIToolSchema,
    create_openai_tools,
)
from analytics_agent.tools.registry import ToolDefinition, ToolRegistry
from analytics_agent.tools.sql_analyzer.models import (
    AnalyzeSalesDataInput,
    GenerateVisualizationInput,
    LookupSalesDataInput,
    SalesQueryResult,
)
from analytics_agent.tools.sql_analyzer.tools import SQLAnalyzerTools


def build_sql_analyzer_definitions(
    model: GenerationModel,
    data_path: Path | None = None,
) -> list[ToolDefinition]:
    """Pair SQL analyzer operations with validated input and output contracts."""
    tools = SQLAnalyzerTools(model=model, data_path=data_path)
    return [
        ToolDefinition(
            tools.lookup_sales_data,
            LookupSalesDataInput,
            output_model=SalesQueryResult,
        ),
        ToolDefinition(tools.analyze_sales_data, AnalyzeSalesDataInput),
        ToolDefinition(tools.generate_visualization, GenerateVisualizationInput),
    ]


def create_sql_analyzer_tools(
    model: GenerationModel,
    data_path: Path | None = None,
) -> tuple[ToolRegistry, list[OpenAIToolSchema]]:
    """Create validated SQL analyzer tools and OpenAI schemas."""
    return create_openai_tools(build_sql_analyzer_definitions(model, data_path))
