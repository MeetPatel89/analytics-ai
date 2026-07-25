"""Provider adapters for provider-neutral tool definitions."""

from inspect import getdoc
from typing import Literal, TypedDict, cast

from openai import pydantic_function_tool

from analytics_agent.tools.registry import (
    ToolDefinition,
    ToolRegistry,
    create_tool_registry,
)


class OpenAIToolSchema(TypedDict):
    """OpenAI function schema."""

    type: Literal["function"]
    name: str
    description: str
    parameters: dict[str, object]
    strict: Literal[True]


def create_openai_tools(
    definitions: list[ToolDefinition],
) -> tuple[ToolRegistry, list[OpenAIToolSchema]]:
    """Create executable tools and their OpenAI function schemas."""
    registry = create_tool_registry(definitions)
    schemas = [_to_openai_schema(definition) for definition in definitions]
    print("--------------------------------")
    print(schemas)
    print("--------------------------------")
    return registry, schemas


def _to_openai_schema(definition: ToolDefinition) -> OpenAIToolSchema:
    description = definition.description or getdoc(definition.handler)
    if not description:
        raise ValueError(f"Missing description for tool '{definition.name}'.")

    sdk_tool = pydantic_function_tool(
        definition.input_model,
        name=definition.name,
        description=description,
    )
    function = sdk_tool["function"]
    parameters = function.get("parameters")
    if parameters is None:
        raise ValueError(f"Missing parameters schema for tool '{definition.name}'.")

    cleaned_parameters = cast(
        dict[str, object],
        _without_titles(parameters),
    )
    _validate_strict_schema(cleaned_parameters, path="parameters")

    return {
        "type": "function",
        "name": definition.name,
        "description": description,
        "parameters": cleaned_parameters,
        "strict": True,
    }


def _without_titles(value: object) -> object:
    """Remove Pydantic display titles that do not help the model call a tool."""
    if isinstance(value, dict):
        return {
            key: _without_titles(item) for key, item in value.items() if key != "title"
        }
    if isinstance(value, list):
        return [_without_titles(item) for item in value]
    return value


def _validate_strict_schema(schema: object, *, path: str) -> None:
    """Fail fast when a generated schema cannot be used with OpenAI strict mode."""
    if not isinstance(schema, dict):
        return

    unsupported_keywords = (
        "allOf",
        "not",
        "dependentRequired",
        "dependentSchemas",
        "if",
        "then",
        "else",
    )
    unsupported = [key for key in unsupported_keywords if key in schema]
    if unsupported:
        names = ", ".join(unsupported)
        raise ValueError(f"Unsupported strict-schema keywords at {path}: {names}.")

    if schema.get("type") == "object":
        if schema.get("additionalProperties") is not False:
            raise ValueError(
                f"Object schema at {path} must set additionalProperties to false."
            )
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            raise ValueError(f"Object schema at {path} must define properties.")
        required = schema.get("required")
        if required != list(properties):
            raise ValueError(
                f"Object schema at {path} must require every declared property."
            )

    for definitions_key in ("$defs", "definitions"):
        definitions = schema.get(definitions_key)
        if isinstance(definitions, dict):
            for name, child in definitions.items():
                _validate_strict_schema(
                    child,
                    path=f"{path}.{definitions_key}.{name}",
                )

    properties = schema.get("properties")
    if isinstance(properties, dict):
        for name, child in properties.items():
            if (
                isinstance(child, dict)
                and not child.get("description")
                and "$ref" not in child
            ):
                raise ValueError(
                    f"Property schema at {path}.properties.{name} must include "
                    "a description."
                )
            _validate_strict_schema(child, path=f"{path}.properties.{name}")

    items = schema.get("items")
    if items is not None:
        _validate_strict_schema(items, path=f"{path}.items")

    variants = schema.get("anyOf")
    if isinstance(variants, list):
        for index, child in enumerate(variants):
            _validate_strict_schema(child, path=f"{path}.anyOf[{index}]")
