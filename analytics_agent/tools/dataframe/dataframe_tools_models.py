"""Input contracts for dataframe tools."""

from typing import Literal

from pydantic import Field

from analytics_agent.tools.registry import ToolInput

MAX_RESULT_ROWS = 50
MAX_DISTINCT_VALUES = 100
MAX_PREVIEW_ROWS = 20


class FilterCondition(ToolInput):
    """Condition applied while filtering dataframe rows."""

    column: str = Field(description="The dataframe column to inspect.")
    operator: Literal["eq", "ne", "gt", "gte", "lt", "lte", "contains", "in"] = Field(
        description="The comparison operator to apply."
    )
    value: str | int | float | bool | list[str | int | float | bool] = Field(
        description="The value used by the operator."
    )


class ListDataframesInput(ToolInput):
    """Arguments for listing available dataframes."""


class DescribeDataframeInput(ToolInput):
    """Arguments for describing a dataframe."""

    dataset_name: str = Field(description="The dataset to describe.")


class PreviewDataframeInput(ToolInput):
    """Arguments for previewing a dataframe."""

    dataset_name: str = Field(description="The dataset to preview.")
    limit: int = Field(default=5, ge=1, le=MAX_PREVIEW_ROWS)


class SearchRowsInput(ToolInput):
    """Arguments for searching dataframe rows."""

    query: str = Field(description="The case-insensitive text query to search for.")
    dataset_name: str | None = Field(
        default=None,
        description="Optional dataset name. If omitted, search all datasets.",
    )
    limit: int = Field(default=5, ge=1, le=MAX_RESULT_ROWS)


class FilterRowsInput(ToolInput):
    """Arguments for filtering dataframe rows."""

    dataset_name: str = Field(description="The dataset to filter.")
    conditions: list[FilterCondition] = Field(
        default_factory=list,
        description="Conditions combined with logical AND.",
    )
    columns: list[str] | None = Field(
        default=None,
        description="Optional list of columns to include in the output.",
    )
    sort_by: str | None = Field(
        default=None,
        description="Optional column name to sort by.",
    )
    sort_desc: bool = Field(default=False)
    limit: int = Field(default=10, ge=1, le=MAX_RESULT_ROWS)


class AggregateRowsInput(ToolInput):
    """Arguments for aggregating dataframe rows."""

    dataset_name: str = Field(description="The dataset to aggregate.")
    group_by: list[str] | None = Field(
        default=None,
        description="Optional list of columns used for grouping.",
    )
    metric: Literal["count", "sum", "mean", "min", "max", "nunique"] = Field(
        default="count"
    )
    metric_column: str | None = Field(
        default=None,
        description="Required for all metrics except count.",
    )
    filters: list[FilterCondition] = Field(
        default_factory=list,
        description="Optional filters applied before aggregation.",
    )
    limit: int = Field(default=10, ge=1, le=MAX_RESULT_ROWS)


class DistinctValuesInput(ToolInput):
    """Arguments for counting distinct dataframe values."""

    dataset_name: str = Field(description="The dataset to inspect.")
    column: str = Field(description="The column whose values should be counted.")
    limit: int = Field(default=20, ge=1, le=MAX_DISTINCT_VALUES)
