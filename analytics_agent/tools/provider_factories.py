"""Provider adapters for provider-neutral tool definitions."""

from inspect import getdoc

from analytics_agent.tools.registry import (
    ToolDefinition,
    ToolRegistry,
    create_tool_registry,
)

OpenAIToolSchema = dict[str, object]


def create_openai_tools(
    definitions: list[ToolDefinition],
) -> tuple[ToolRegistry, list[OpenAIToolSchema]]:
    """Create executable tools and their OpenAI function schemas."""
    registry = create_tool_registry(definitions)
    schemas = [_to_openai_schema(definition) for definition in definitions]
    return registry, schemas


def _to_openai_schema(definition: ToolDefinition) -> OpenAIToolSchema:
    description = definition.description or getdoc(definition.handler)
    if not description:
        raise ValueError(f"Missing description for tool '{definition.name}'.")
    return {
        "type": "function",
        "name": definition.name,
        "description": description,
        "parameters": definition.input_model.model_json_schema(),
    }
