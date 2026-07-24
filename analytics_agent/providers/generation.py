"""Provider-neutral text and structured-generation capability."""

from typing import Protocol, TypeVar

from pydantic import BaseModel

StructuredOutputT = TypeVar("StructuredOutputT", bound=BaseModel)


class GenerationModel(Protocol):
    """Generation behavior available from every registered provider."""

    def generate_text(self, prompt: str) -> str:
        """Generate a non-empty plain-text response."""
        ...

    def generate_structured(
        self,
        prompt: str,
        response_model: type[StructuredOutputT],
    ) -> StructuredOutputT:
        """Generate output validated against a Pydantic model."""
        ...
