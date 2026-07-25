"""Validated contracts for the SQL analyzer tool chain."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from analytics_agent.tools.registry import ToolInput

NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
SalesValue = str | int | float | bool | None


class SalesQueryResult(BaseModel):
    """A bounded result returned by a generated sales query."""

    model_config = ConfigDict(extra="forbid")

    sql: NonEmptyString = Field(
        description="The validated read-only DuckDB SELECT statement that was run."
    )
    columns: list[NonEmptyString] = Field(
        description="Ordered result-column names shared by every row."
    )
    rows: list[list[SalesValue]] = Field(
        description=(
            "Result rows as value arrays. Each value is aligned by position with "
            "the ordered columns array."
        )
    )
    returned_row_count: int = Field(
        ge=0,
        description="The number of rows included in this result envelope.",
    )
    truncated: bool = Field(
        description=(
            "Whether additional matching rows existed beyond the returned row limit."
        )
    )

    @model_validator(mode="after")
    def validate_result_shape(self) -> SalesQueryResult:
        """Keep row metadata and column names consistent."""
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("Query-result column names must be unique.")
        if self.returned_row_count != len(self.rows):
            raise ValueError("returned_row_count must match the number of rows.")
        for row in self.rows:
            if len(row) != len(self.columns):
                raise ValueError(
                    "Each result row must contain one value for every declared column."
                )
        return self


class LookupSalesDataInput(ToolInput):
    """Arguments for generating and running a read-only sales query."""

    prompt: NonEmptyString = Field(
        description="The sales-data question the generated query must answer."
    )


class AnalyzeSalesDataInput(ToolInput):
    """Arguments for analyzing a prior sales-query result."""

    prompt: NonEmptyString = Field(
        description="The analysis goal to answer from the supplied query result."
    )
    data: SalesQueryResult


class GenerateVisualizationInput(ToolInput):
    """Arguments for generating plotting code from a sales-query result."""

    data: SalesQueryResult
    visualization_goal: NonEmptyString = Field(
        description="The chart or visual relationship the generated code should show."
    )


class SalesSQLQuery(BaseModel):
    """Structured SQL generated through a provider capability."""

    model_config = ConfigDict(extra="forbid")

    sql: NonEmptyString


class VisualizationConfig(BaseModel):
    """Validated chart choices generated before plotting code."""

    model_config = ConfigDict(extra="forbid")

    chart_type: Literal["line", "bar", "scatter"] = Field(
        description="The type of chart to generate."
    )
    x_axis: NonEmptyString = Field(
        description="The column name to use for the x-axis of the chart."
    )
    y_axis: NonEmptyString = Field(
        description="The column name to use for the y-axis of the chart."
    )
    title: NonEmptyString = Field(description="The title of the chart.")


class VisualizationCode(BaseModel):
    """Structured Python source generated through a provider capability."""

    model_config = ConfigDict(extra="forbid")

    code: NonEmptyString
