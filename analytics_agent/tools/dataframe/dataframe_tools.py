"""Catalog-bound dataframe query tools."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import pandas as pd

from analytics_agent.tools.dataframe.catalog import DataframeCatalog, DatasetEntry
from analytics_agent.tools.dataframe.dataframe_tools_models import FilterCondition

_Metric = Literal["count", "sum", "mean", "min", "max", "nunique"]


class DataframeTools:
    """Dataframe query operations bound to a dataset catalog."""

    def __init__(self, catalog: DataframeCatalog) -> None:
        self._catalog = catalog

    def list_dataframes(self) -> str:
        """List all available datasets with source and shape metadata."""
        lines = ["Available datasets:"]
        for entry in self._catalog.all():
            lines.append(
                (
                    f"- {entry.name}: {len(entry.dataframe)} rows x "
                    f"{len(entry.dataframe.columns)} columns"
                    f"{_format_source(entry)}"
                    f"{_format_description(entry)}"
                )
            )
        return "\n".join(lines)

    def describe_dataframe(self, dataset_name: str) -> str:
        """Describe a dataset's columns, dtypes, null counts, and preview."""
        entry = self._catalog.get(dataset_name)
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

    def preview_dataframe(self, dataset_name: str, limit: int = 5) -> str:
        """Show the first few rows of a dataset."""
        entry = self._catalog.get(dataset_name)
        preview = entry.dataframe.head(limit)
        return (
            f"Dataset: {entry.name}\n"
            f"Showing first {len(preview)} rows.\n"
            f"{_frame_to_table(preview)}"
        )

    def search_rows(
        self,
        query: str,
        dataset_name: str | None = None,
        limit: int = 5,
    ) -> str:
        """Search string columns for a keyword or phrase across available datasets."""
        normalized_query = query.strip()
        if not normalized_query:
            return "Search query must be non-empty."
        entries = (
            [self._catalog.get(dataset_name)] if dataset_name else self._catalog.all()
        )
        matches: list[tuple[int, DatasetEntry, pd.Series]] = []
        query_tokens = {
            token.lower() for token in normalized_query.split() if token.strip()
        }

        for entry in entries:
            searchable = _string_columns(entry.dataframe)
            if not searchable:
                continue
            for _, row in entry.dataframe.iterrows():
                values = [
                    str(row[column]) for column in searchable if pd.notna(row[column])
                ]
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
        self,
        dataset_name: str,
        conditions: list[FilterCondition],
        columns: list[str] | None = None,
        sort_by: str | None = None,
        sort_desc: bool = False,
        limit: int = 10,
    ) -> str:
        """Filter rows in a dataset using validated column conditions."""
        entry = self._catalog.get(dataset_name)
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
        self,
        dataset_name: str,
        group_by: list[str] | None = None,
        metric: _Metric = "count",
        metric_column: str | None = None,
        filters: list[FilterCondition] | None = None,
        limit: int = 10,
    ) -> str:
        """Aggregate rows with optional filtering and grouping."""
        entry = self._catalog.get(dataset_name)
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

    def distinct_values(self, dataset_name: str, column: str, limit: int = 20) -> str:
        """Count the most common distinct values for a dataset column."""
        entry = self._catalog.get(dataset_name)
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
    metric: _Metric,
    metric_column: str | None,
) -> pd.DataFrame:
    if metric == "count":
        return grouped.size().reset_index(name="count")
    assert metric_column is not None
    aggregator = getattr(grouped[metric_column], metric)
    return aggregator().reset_index(name=f"{metric}_{metric_column}")


def _scalar_metric(
    dataframe: pd.DataFrame,
    metric: _Metric,
    metric_column: str | None,
) -> int | float:
    if metric == "count":
        return int(len(dataframe))
    assert metric_column is not None
    value = getattr(dataframe[metric_column], metric)()
    if pd.isna(value):
        return float("nan")
    return value
