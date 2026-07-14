from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable, Literal
from dataclasses import dataclass
from inspect import getdoc

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from pathlib import Path
import glob
import os


MAX_RESULT_ROWS = 50
MAX_DISTINCT_VALUES = 100
MAX_PREVIEW_ROWS = 20

@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataframe: pd.DataFrame
    description: str = ""
    source_path: str | None = None


@dataclass(frozen=True)
class DatasetEntry:
    name: str
    dataframe: pd.DataFrame
    description: str
    source_path: str | None
    id_column: str | None

    @property
    def source_name(self) -> str:
        if not self.source_path:
            return ""
        return Path(self.source_path).name

DATASET_METADATA = {
    "saas_docs.csv": (
        "SaaS Docs",
        "Product and API documentation for SaaS platform features and limits.",
    ),
    "credit_card_terms.csv": (
        "Credit Card Terms",
        "Credit card terms, APR details, fees, and account policies.",
    ),
    "hospital_policy.csv": (
        "Hospital Policy",
        "Hospital operations, compliance rules, and patient care policies.",
    ),
    "ecommerce_faqs.csv": (
        "Ecommerce FAQs",
        "Customer-facing ecommerce questions covering shipping, returns, and support.",
    ),
}

class DataframeCatalog:
    def __init__(self, entries: list[DatasetEntry]) -> None:
        self._entries = entries
        self._by_name = {entry.name: entry for entry in entries}

    @classmethod
    def from_specs(cls, specs: list[DatasetSpec]) -> "DataframeCatalog":
        entries: list[DatasetEntry] = []
        seen_names: set[str] = set()
        for spec in specs:
            normalized_name = spec.name.strip()
            if not normalized_name:
                raise ValueError("Dataset names must be non-empty.")
            if normalized_name in seen_names:
                raise ValueError(f"Duplicate dataset name: {normalized_name}")
            if not isinstance(spec.dataframe, pd.DataFrame):
                raise ValueError(
                    f"Dataset '{normalized_name}' must be a pandas DataFrame."
                )

            seen_names.add(normalized_name)
            entries.append(
                DatasetEntry(
                    name=normalized_name,
                    dataframe=spec.dataframe,
                    description=spec.description.strip(),
                    source_path=spec.source_path,
                    id_column=infer_id_column(spec.dataframe),
                )
            )
        return cls(entries)

    def all(self) -> list[DatasetEntry]:
        return list(self._entries)

    def get(self, dataset_name: str) -> DatasetEntry:
        try:
            return self._by_name[dataset_name]
        except KeyError as exc:
            available = ", ".join(self.names())
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Available datasets: {available}."
            ) from exc

    def names(self) -> list[str]:
        return [entry.name for entry in self._entries]


def infer_id_column(dataframe: pd.DataFrame) -> str | None:
    candidates = [
        column
        for column in dataframe.columns
        if "id" in str(column).strip().lower()
    ]
    for column in candidates:
        series = dataframe[column]
        if series.notna().all() and series.is_unique:
            return str(column)
    return str(candidates[0]) if candidates else None



class FilterCondition(BaseModel):
    column: str = Field(description="The dataframe column to inspect.")
    operator: Literal["eq", "ne", "gt", "gte", "lt", "lte", "contains", "in"] = Field(
        description="The comparison operator to apply."
    )
    value: str | int | float | bool | list[str | int | float | bool] = Field(
        description="The value used by the operator."
    )


class DescribeDataframeInput(BaseModel):
    dataset_name: str = Field(description="The dataset to describe.")


class PreviewDataframeInput(BaseModel):
    dataset_name: str = Field(description="The dataset to preview.")
    limit: int = Field(default=5, ge=1, le=MAX_PREVIEW_ROWS)


class SearchRowsInput(BaseModel):
    query: str = Field(description="The case-insensitive text query to search for.")
    dataset_name: str | None = Field(
        default=None,
        description="Optional dataset name. If omitted, search all datasets.",
    )
    limit: int = Field(default=5, ge=1, le=MAX_RESULT_ROWS)


class FilterRowsInput(BaseModel):
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


class AggregateRowsInput(BaseModel):
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


class DistinctValuesInput(BaseModel):
    dataset_name: str = Field(description="The dataset to inspect.")
    column: str = Field(description="The column whose values should be counted.")
    limit: int = Field(default=20, ge=1, le=MAX_DISTINCT_VALUES)


class ListDataframesInput(BaseModel):
    pass


def build_tools(catalog: DataframeCatalog) -> list:
    
    def list_dataframes() -> str:
        """List all available datasets with source and shape metadata."""
        lines = ["Available datasets:"]
        for entry in catalog.all():
            lines.append(
                (
                    f"- {entry.name}: {len(entry.dataframe)} rows x "
                    f"{len(entry.dataframe.columns)} columns"
                    f"{_format_source(entry)}"
                    f"{_format_description(entry)}"
                )
            )
        return "\n".join(lines)

    def describe_dataframe(dataset_name: str) -> str:
        """Describe a dataset's columns, dtypes, null counts, and preview."""
        entry = _get_entry(catalog, dataset_name)
        dataframe = entry.dataframe
        lines = [
            f"Dataset: {entry.name}",
            f"Rows: {len(dataframe)}",
            f"Columns: {len(dataframe.columns)}",
            f"ID column: {entry.id_column or 'None detected'}",
            "Column summary:",
        ]
        for column in dataframe.columns:
            series = dataframe[column]
            lines.append(
                f"- {column}: dtype={series.dtype}, nulls={int(series.isna().sum())}"
            )
        lines.append("Preview:")
        lines.append(_frame_to_table(dataframe.head(3)))
        return "\n".join(lines)

    def preview_dataframe(dataset_name: str, limit: int = 5) -> str:
        """Show the first few rows of a dataset."""
        entry = _get_entry(catalog, dataset_name)
        preview = entry.dataframe.head(limit)
        return (
            f"Dataset: {entry.name}\n"
            f"Showing first {len(preview)} rows.\n"
            f"{_frame_to_table(preview)}"
        )

    def search_rows(
        query: str,
        dataset_name: str | None = None,
        limit: int = 5,
    ) -> str:
        """Search string columns for a keyword or phrase across one or all datasets."""
        normalized_query = query.strip()
        if not normalized_query:
            return "Search query must be non-empty."
        entries = [catalog.get(dataset_name)] if dataset_name else catalog.all()
        matches: list[tuple[int, DatasetEntry, pd.Series]] = []
        query_tokens = {
            token.lower() for token in normalized_query.split() if token.strip()
        }

        for entry in entries:
            searchable = _string_columns(entry.dataframe)
            if not searchable:
                continue
            for _, row in entry.dataframe.iterrows():
                values = [str(row[column]) for column in searchable if pd.notna(row[column])]
                haystack = " ".join(values)
                lowered = haystack.lower()
                if normalized_query.lower() not in lowered and not all(
                    token in lowered for token in query_tokens
                ):
                    continue
                score = sum(1 for token in query_tokens if token in lowered)
                if normalized_query.lower() in lowered:
                    score += len(query_tokens) + 1
                matches.append((score, entry, row))

        matches.sort(
            key=lambda item: (
                -item[0],
                item[1].name,
                _row_identifier(item[1], item[2]),
            )
        )
        selected = matches[:limit]
        if not selected:
            target = dataset_name or "all datasets"
            return f"No matching rows found in {target} for query '{normalized_query}'."

        lines = [f"Found {len(selected)} matching rows for '{normalized_query}':"]
        for _, entry, row in selected:
            lines.append(
                f"- {entry.name} | {_row_identifier(entry, row)} | "
                f"{_row_excerpt(entry, row)}"
            )
        return "\n".join(lines)

    def filter_rows(
        dataset_name: str,
        conditions: list[FilterCondition],
        columns: list[str] | None = None,
        sort_by: str | None = None,
        sort_desc: bool = False,
        limit: int = 10,
    ) -> str:
        """Filter rows in a dataset using validated column conditions."""
        entry = _get_entry(catalog, dataset_name)
        filtered = _apply_filters(entry.dataframe, conditions)
        if sort_by:
            _require_columns(entry.dataframe, [sort_by])
            filtered = filtered.sort_values(by=sort_by, ascending=not sort_desc)
        if columns:
            _require_columns(entry.dataframe, columns)
            filtered = filtered.loc[:, columns]

        limited = filtered.head(limit)
        if limited.empty:
            return f"No rows matched in dataset '{entry.name}'."
        return (
            f"Dataset: {entry.name}\n"
            f"Matched rows: {len(filtered)}\n"
            f"Showing first {len(limited)} rows.\n"
            f"{_frame_to_table(limited)}"
        )

    def aggregate_rows(
        dataset_name: str,
        group_by: list[str] | None = None,
        metric: Literal["count", "sum", "mean", "min", "max", "nunique"] = "count",
        metric_column: str | None = None,
        filters: list[FilterCondition] | None = None,
        limit: int = 10,
    ) -> str:
        """Aggregate rows with optional filtering and grouping."""
        entry = _get_entry(catalog, dataset_name)
        dataframe = _apply_filters(entry.dataframe, filters or [])
        group_by = group_by or []
        if group_by:
            _require_columns(dataframe, group_by)
        if metric != "count" and not metric_column:
            return f"Metric '{metric}' requires metric_column."
        if metric_column:
            _require_columns(dataframe, [metric_column])

        if group_by:
            grouped = dataframe.groupby(group_by, dropna=False)
            result = _grouped_metric(grouped, metric, metric_column)
        else:
            value = _scalar_metric(dataframe, metric, metric_column)
            result = pd.DataFrame([{metric_column or metric: value}])

        limited = result.head(limit)
        return (
            f"Dataset: {entry.name}\n"
            f"Rows after filters: {len(dataframe)}\n"
            f"{_frame_to_table(limited)}"
        )

    def distinct_values(dataset_name: str, column: str, limit: int = 20) -> str:
        """Count the most common distinct values for a dataset column."""
        entry = _get_entry(catalog, dataset_name)
        _require_columns(entry.dataframe, [column])
        counts = (
            entry.dataframe[column]
            .fillna("<NA>")
            .astype(str)
            .value_counts(dropna=False)
            .head(limit)
            .reset_index()
        )
        counts.columns = [column, "count"]
        return (
            f"Dataset: {entry.name}\n"
            f"Top {len(counts)} values for {column}:\n"
            f"{_frame_to_table(counts)}"
        )

    return [
        list_dataframes,
        describe_dataframe,
        preview_dataframe,
        search_rows,
        filter_rows,
        aggregate_rows,
        distinct_values,
    ]


ToolFunction = Callable[..., str]
ToolRegistry = dict[str, ToolFunction]

_TOOL_INPUT_MODELS: dict[str, type[BaseModel]] = {
    "list_dataframes": ListDataframesInput,
    "describe_dataframe": DescribeDataframeInput,
    "preview_dataframe": PreviewDataframeInput,
    "search_rows": SearchRowsInput,
    "filter_rows": FilterRowsInput,
    "aggregate_rows": AggregateRowsInput,
    "distinct_values": DistinctValuesInput,
}


def _tool_to_openai_schema(name: str, model: type[BaseModel], description: str) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": model.model_json_schema(),
    }

def load_dataset_specs(data_path: str) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []
    files = sorted(glob.glob(os.path.join(data_path, "*.csv")))
    if not files:
        raise ValueError(f"No CSV files found in {data_path}.")
    for file_path in files:
        try:
            dataframe = pd.read_csv(file_path)
        except Exception as exc:
            raise ValueError(f"Failed to load '{file_path}' as a dataframe: {exc}") from exc
        file_name = os.path.basename(file_path)
        dataset_name, description = DATASET_METADATA.get(
            file_name,
            (os.path.splitext(file_name)[0], f"Dataset loaded from {file_name}."),
        )
        specs.append(
            DatasetSpec(name=dataset_name, dataframe=dataframe, description=description, source_path=file_path)
        )

    return specs

def _wrap_tool(tool: ToolFunction, input_model: type[BaseModel]) -> ToolFunction:
    def wrapped(**kwargs: Any) -> str:
        try:
            validated = input_model.model_validate(kwargs)
            return str(tool(**validated.model_dump()))
        except ValidationError as exc:
            return f"Invalid arguments for tool '{tool.__name__}': {exc}"
        except Exception as exc:
            return f"Tool '{tool.__name__}' failed: {exc}"

    wrapped.__name__ = tool.__name__
    wrapped.__doc__ = tool.__doc__
    return wrapped


def create_tool_registry(catalog: DataframeCatalog) -> tuple[ToolRegistry, list[dict[str, Any]]]:
    tool_registry: ToolRegistry = {}
    tool_schemas: list[dict[str, Any]] = []

    for tool in build_tools(catalog):
        input_model = _TOOL_INPUT_MODELS.get(tool.__name__)
        if input_model is None:
            raise ValueError(f"Missing input model for tool '{tool.__name__}'.")
        description = getdoc(tool) or f"Execute the {tool.__name__} tool."
        tool_registry[tool.__name__] = _wrap_tool(tool, input_model)
        tool_schemas.append(_tool_to_openai_schema(tool.__name__, input_model, description))

    return tool_registry, tool_schemas


def configure_tools(catalog: DataframeCatalog) -> tuple[ToolRegistry, list[dict[str, Any]]]:
    global TOOL_FUNCTIONS, TOOL_SCHEMA, TOOLS
    TOOL_SCHEMA, TOOLS = create_tool_registry(catalog)
    TOOL_FUNCTIONS = TOOL_SCHEMA
    return TOOL_SCHEMA, TOOLS


TOOL_SCHEMA, TOOLS = create_tool_registry(DataframeCatalog.from_specs([]))
TOOL_FUNCTIONS = TOOL_SCHEMA


def _get_entry(catalog: DataframeCatalog, dataset_name: str) -> DatasetEntry:
    return catalog.get(dataset_name)


def _format_source(entry: DatasetEntry) -> str:
    return f", source={entry.source_name}" if entry.source_name else ""


def _format_description(entry: DatasetEntry) -> str:
    return f", description={entry.description}" if entry.description else ""


def _frame_to_table(dataframe: pd.DataFrame) -> str:
    if dataframe.empty:
        return "No rows to display."
    return dataframe.fillna("<NA>").to_markdown(index=False)


def _string_columns(dataframe: pd.DataFrame) -> list[str]:
    return [
        str(column)
        for column in dataframe.columns
        if pd.api.types.is_object_dtype(dataframe[column])
        or pd.api.types.is_string_dtype(dataframe[column])
    ]


def _row_identifier(entry: DatasetEntry, row: pd.Series) -> str:
    if entry.id_column and entry.id_column in row:
        return f"{entry.id_column}={row[entry.id_column]}"
    return f"row_index={row.name}"


def _row_excerpt(entry: DatasetEntry, row: pd.Series, max_fields: int = 3) -> str:
    fields: list[str] = []
    for column in entry.dataframe.columns:
        if entry.id_column and column == entry.id_column:
            continue
        value = row[column]
        if pd.isna(value):
            continue
        fields.append(f"{column}={value}")
        if len(fields) >= max_fields:
            break
    return "; ".join(fields)


def _require_columns(dataframe: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        available = ", ".join(map(str, dataframe.columns.tolist()))
        raise ValueError(
            f"Unknown columns: {', '.join(missing)}. Available columns: {available}."
        )


def _apply_filters(
    dataframe: pd.DataFrame,
    conditions: list[FilterCondition],
) -> pd.DataFrame:
    filtered = dataframe.copy()
    for condition in conditions:
        _require_columns(filtered, [condition.column])
        series = filtered[condition.column]
        operator = condition.operator
        value = condition.value

        if operator == "eq":
            mask = series == value
        elif operator == "ne":
            mask = series != value
        elif operator == "gt":
            mask = series > value
        elif operator == "gte":
            mask = series >= value
        elif operator == "lt":
            mask = series < value
        elif operator == "lte":
            mask = series <= value
        elif operator == "contains":
            mask = series.astype(str).str.contains(str(value), case=False, na=False)
        elif operator == "in":
            if not isinstance(value, list):
                raise ValueError("Operator 'in' requires a list value.")
            mask = series.isin(value)
        else:
            raise ValueError(f"Unsupported operator '{operator}'.")
        filtered = filtered.loc[mask]
    return filtered


def _grouped_metric(
    grouped: pd.core.groupby.generic.DataFrameGroupBy,
    metric: str,
    metric_column: str | None,
) -> pd.DataFrame:
    if metric == "count":
        return grouped.size().reset_index(name="count")
    assert metric_column is not None
    aggregator = getattr(grouped[metric_column], metric)
    return aggregator().reset_index(name=f"{metric}_{metric_column}")


def _scalar_metric(
    dataframe: pd.DataFrame,
    metric: str,
    metric_column: str | None,
) -> int | float:
    if metric == "count":
        return int(len(dataframe))
    assert metric_column is not None
    value = getattr(dataframe[metric_column], metric)()
    if pd.isna(value):
        return float("nan")
    return value
