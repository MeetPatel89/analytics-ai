"""Validated input contracts for filesystem analytics tools."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, StringConstraints

from analytics_agent.tools.registry import ToolInput

MAX_DIRECTORY_ENTRIES = 200
MAX_PREVIEW_ROWS = 20
MAX_QUERY_ROWS = 50
MAX_QUERY_SOURCES = 10
MAX_SOURCE_FILES = 100
MAX_TEXT_BYTES = 32 * 1024
MAX_TOOL_OUTPUT_BYTES = 64 * 1024

NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=4096),
]


class ListLocationsInput(ToolInput):
    """Arguments for listing configured filesystem locations."""


class ListDirectoryInput(ToolInput):
    """Arguments for listing a directory or safe glob."""

    location_name: NonEmptyString = Field(
        description="The configured data-location name."
    )
    path: str = Field(
        default="",
        max_length=4096,
        description=(
            "A forward-slash path relative to the location root. It may contain "
            "glob wildcards; use an empty string for the root."
        ),
    )
    limit: int = Field(
        default=100,
        ge=1,
        le=MAX_DIRECTORY_ENTRIES,
        description="The maximum entries to return; use 100 by default.",
    )


class GetFileInfoInput(ToolInput):
    """Arguments for inspecting one filesystem entry."""

    location_name: NonEmptyString = Field(
        description="The configured data-location name."
    )
    path: NonEmptyString = Field(
        description="The forward-slash file path relative to the location root."
    )


class InspectSchemaInput(ToolInput):
    """Arguments for inspecting one tabular file schema."""

    location_name: NonEmptyString = Field(
        description="The configured data-location name."
    )
    path: NonEmptyString = Field(
        description="The relative path of one CSV or Parquet file."
    )


class PreviewDataInput(ToolInput):
    """Arguments for previewing one tabular file."""

    location_name: NonEmptyString = Field(
        description="The configured data-location name."
    )
    path: NonEmptyString = Field(
        description="The relative path of one CSV or Parquet file."
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=MAX_PREVIEW_ROWS,
        description="The number of leading rows to return; use 5 by default.",
    )


class ReadTextFileInput(ToolInput):
    """Arguments for a bounded text-file read."""

    location_name: NonEmptyString = Field(
        description="The configured data-location name."
    )
    path: NonEmptyString = Field(
        description="The relative path of one txt, log, or JSON file."
    )
    offset: int = Field(
        default=0,
        ge=0,
        le=2**63 - 1,
        description="The zero-based byte offset at which to start reading.",
    )
    max_bytes: int = Field(
        default=8192,
        ge=1,
        le=MAX_TEXT_BYTES,
        description=(
            "The maximum bytes to read; use 8192 by default and never exceed 32768."
        ),
    )


class QuerySource(ToolInput):
    """One file or glob exposed to a DuckDB query as a safe view."""

    alias: NonEmptyString = Field(
        description=(
            "A simple SQL identifier used as the source's temporary view name."
        )
    )
    location_name: NonEmptyString = Field(
        description="The configured location containing this source."
    )
    path: NonEmptyString = Field(
        description=(
            "A relative CSV or Parquet file path or glob. Traversal and absolute "
            "paths are rejected."
        )
    )


class QueryDataInput(ToolInput):
    """Arguments for a bounded, read-only DuckDB query."""

    sql: NonEmptyString = Field(
        description=(
            "Exactly one DuckDB SELECT statement. Query the temporary view aliases "
            "declared in sources."
        )
    )
    sources: list[QuerySource] = Field(
        min_length=1,
        max_length=MAX_QUERY_SOURCES,
        description=(
            "One to ten CSV or Parquet files/globs exposed as temporary views."
        ),
    )
