"""Provider-neutral tool definitions and validated execution."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from functools import wraps

from pydantic import BaseModel, ConfigDict, ValidationError

ToolFunction = Callable[..., str]
ToolHandler = Callable[..., object]


class ToolInput(BaseModel):
    """Base model for tool arguments."""

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class ToolDefinition:
    """Bind one tool implementation to its validated input contract."""

    handler: ToolHandler
    input_model: type[BaseModel]
    description: str | None = None

    @property
    def name(self) -> str:
        """Name exposed to model providers."""
        return self.handler.__name__


class ToolRegistry(Mapping[str, ToolFunction]):
    """Read-only collection of validated, executable tools."""

    def __init__(self, definitions: list[ToolDefinition]) -> None:
        self._definitions = tuple(definitions)
        self._tools: dict[str, ToolFunction] = {}
        for definition in self._definitions:
            if definition.name in self._tools:
                raise ValueError(f"Duplicate tool name: {definition.name}")
            self._tools[definition.name] = _validated_tool(definition)

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        """Definitions in registration order."""
        return self._definitions

    def execute(self, name: str, arguments: Mapping[str, object]) -> str:
        """Execute a tool by name and return a model-readable result."""
        tool = self._tools.get(name)
        if tool is None:
            return f"Unknown tool: {name}"
        return tool(**arguments)

    def __getitem__(self, name: str) -> ToolFunction:
        """Return a validated tool by name."""
        return self._tools[name]

    def __iter__(self) -> Iterator[str]:
        """Iterate over tool names in registration order."""
        return iter(self._tools)

    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)


def create_tool_registry(definitions: list[ToolDefinition]) -> ToolRegistry:
    """Create a validated registry from explicit tool definitions."""
    return ToolRegistry(definitions)


def _validated_tool(definition: ToolDefinition) -> ToolFunction:
    @wraps(definition.handler)
    def wrapped(**kwargs: object) -> str:
        try:
            validated = definition.input_model.model_validate(kwargs)
            arguments = {
                name: getattr(validated, name)
                for name in definition.input_model.model_fields
            }
            return str(definition.handler(**arguments))
        except ValidationError as exc:
            return f"Invalid arguments for tool '{definition.name}': {exc}"
        except Exception as exc:
            return f"Tool '{definition.name}' failed: {exc}"

    return wrapped
