"""Validated contracts for the SQL analyzer tool chain."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)

from analytics_agent.tools.registry import ToolInput

NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class SalesQueryResult(BaseModel):
    """A bounded, JSON-safe result returned by a generated sales query."""

    model_config = ConfigDict(extra="forbid")

    sql: NonEmptyString
    columns: list[NonEmptyString]
    rows: list[dict[str, JsonValue]]
    returned_row_count: int = Field(ge=0)
    truncated: bool

    @model_validator(mode="after")
    def validate_result_shape(self) -> SalesQueryResult:
        """Keep row metadata and column names consistent."""
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("Query-result column names must be unique.")
        if self.returned_row_count != len(self.rows):
            raise ValueError("returned_row_count must match the number of rows.")
        expected_columns = set(self.columns)
        for row in self.rows:
            if set(row) != expected_columns:
                raise ValueError("Each result row must contain every declared column.")
        return self


class LookupSalesDataInput(ToolInput):
    """Arguments for generating and running a read-only sales query."""

    prompt: NonEmptyString


class AnalyzeSalesDataInput(ToolInput):
    """Arguments for analyzing a prior sales-query result."""

    prompt: NonEmptyString
    data: SalesQueryResult


class GenerateVisualizationInput(ToolInput):
    """Arguments for generating plotting code from a sales-query result."""

    data: SalesQueryResult
    visualization_goal: NonEmptyString


class SalesSQLQuery(BaseModel):
    """Structured SQL generated through a provider capability."""

    model_config = ConfigDict(extra="forbid")

    sql: NonEmptyString


class VisualizationConfig(BaseModel):
    """Validated chart choices generated before plotting code."""

    model_config = ConfigDict(extra="forbid")

    chart_type: Literal["line", "bar", "scatter"]
    x_axis: NonEmptyString
    y_axis: NonEmptyString
    title: NonEmptyString


class VisualizationCode(BaseModel):
    """Structured Python source generated through a provider capability."""

    model_config = ConfigDict(extra="forbid")

    code: NonEmptyString
