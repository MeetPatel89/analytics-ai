"""Navigation and inspection operations over a location catalog."""

from __future__ import annotations

import glob
from collections.abc import Iterable
from datetime import date, datetime, time
from pathlib import PurePosixPath
from typing import BinaryIO, cast

import pyarrow as pa
import pyarrow.csv as arrow_csv
import pyarrow.parquet as arrow_parquet
from fsspec.spec import AbstractFileSystem

from analytics_agent.filesystem import LocationCatalog
from analytics_agent.tools.filesystem_analytics.models import (
    MAX_DIRECTORY_ENTRIES,
    QuerySource,
)
from analytics_agent.tools.filesystem_analytics.query import DuckDBQueryEngine
from analytics_agent.tools.filesystem_analytics.serialization import (
    bounded_list_result,
    bounded_rows_result,
    bounded_text_result,
    json_result,
    normalize_value,
)

_TEXT_SUFFIXES = frozenset({".json", ".log", ".txt"})
_TABULAR_SUFFIXES = {
    ".csv": "csv",
    ".parquet": "parquet",
    ".pq": "parquet",
}
_MAX_SCHEMA_COLUMNS = 500
_MAX_PREVIEW_COLUMNS = 200


class FilesystemAnalyticsTools:
    """Read-only navigation, inspection, and SQL over named locations."""

    def __init__(self, catalog: LocationCatalog) -> None:
        self._catalog = catalog
        self._query_engine = DuckDBQueryEngine(catalog)

    def list_locations(self) -> str:
        """List configured roots and their backend kinds."""
        locations = [
            {
                "name": location.name,
                "uri": location.uri,
                "backend": location.backend,
            }
            for location in self._catalog.locations()
        ]
        return bounded_list_result(
            {},
            "locations",
            locations,
        )

    def list_directory(
        self,
        location_name: str,
        path: str = "",
        limit: int = 100,
    ) -> str:
        """List directory entries or a safe glob with bounded output."""
        filesystem = self._catalog.filesystem(location_name)
        if glob.has_magic(path):
            matched = self._catalog.glob(location_name, path)
            raw_entries = [
                filesystem.info(item)
                for item in matched[: min(limit, MAX_DIRECTORY_ENTRIES)]
            ]
            matched_entry_count = len(matched)
        else:
            resolved = self._catalog.resolve(location_name, path)
            listed = filesystem.ls(resolved, detail=True)
            raw_entries = [
                filesystem.info(item) if isinstance(item, str) else item
                for item in listed
            ]
            matched_entry_count = len(raw_entries)

        entries = sorted(
            (
                _directory_entry(
                    self._catalog,
                    location_name,
                    entry,
                )
                for entry in raw_entries
            ),
            key=lambda entry: str(entry["path"]),
        )
        capped = entries[: min(limit, MAX_DIRECTORY_ENTRIES)]
        return bounded_list_result(
            {
                "location_name": location_name,
                "path": path or ".",
                "matched_entry_count": matched_entry_count,
            },
            "entries",
            capped,
            requested_truncated=matched_entry_count > len(capped),
        )

    def get_file_info(self, location_name: str, path: str) -> str:
        """Return metadata and extension-based format detection for one path."""
        resolved = self._catalog.resolve(location_name, path)
        info = self._catalog.filesystem(location_name).info(resolved)
        return json_result(
            {
                "location_name": location_name,
                "path": self._catalog.relative_path(location_name, resolved),
                "type": info.get("type", "unknown"),
                "size": info.get("size"),
                "modified": _modified_value(info),
                "format": _detect_format(path),
            }
        )

    def inspect_schema(self, location_name: str, path: str) -> str:
        """Inspect columns and Arrow types for one CSV or Parquet file."""
        resolved, file_format = self._tabular_file(location_name, path)
        schema = _read_schema(
            self._catalog.filesystem(location_name),
            resolved,
            file_format,
        )
        fields = [{"name": field.name, "type": str(field.type)} for field in schema]
        capped = fields[:_MAX_SCHEMA_COLUMNS]
        return bounded_list_result(
            {
                "location_name": location_name,
                "path": path,
                "format": file_format,
                "column_count": len(fields),
            },
            "columns",
            capped,
            requested_truncated=len(fields) > len(capped),
        )

    def preview_data(
        self,
        location_name: str,
        path: str,
        limit: int = 5,
    ) -> str:
        """Return the first bounded rows from one CSV or Parquet file."""
        resolved, file_format = self._tabular_file(location_name, path)
        table = _read_preview(
            self._catalog.filesystem(location_name),
            resolved,
            file_format,
            limit + 1,
        )
        has_more_rows = table.num_rows > limit
        table = table.slice(0, limit)
        columns = [str(name) for name in table.column_names]
        if len(columns) > _MAX_PREVIEW_COLUMNS:
            raise ValueError(
                f"Preview supports at most {_MAX_PREVIEW_COLUMNS} columns."
            )
        rows = [
            [normalize_value(record.get(column)) for column in columns]
            for record in table.to_pylist()
        ]
        return bounded_rows_result(
            {
                "location_name": location_name,
                "path": path,
                "format": file_format,
                "columns": columns,
            },
            rows,
            requested_truncated=has_more_rows,
        )

    def read_text_file(
        self,
        location_name: str,
        path: str,
        offset: int = 0,
        max_bytes: int = 8192,
    ) -> str:
        """Read a bounded UTF-8 byte range from txt, log, or JSON."""
        suffix = PurePosixPath(path).suffix.lower()
        if suffix not in _TEXT_SUFFIXES:
            raise ValueError("read_text_file supports only .txt, .log, and .json.")
        resolved = self._catalog.resolve(location_name, path)
        _require_file(self._catalog.filesystem(location_name).info(resolved), path)
        raw = cast(BinaryIO, self._catalog.filesystem(location_name).open(resolved))
        with raw:
            raw.seek(offset)
            chunk = raw.read(max_bytes + 1)
        returned = chunk[:max_bytes]
        truncated = len(chunk) > max_bytes
        content = returned.decode("utf-8", errors="replace")
        return bounded_text_result(
            {
                "location_name": location_name,
                "path": path,
                "offset": offset,
                "bytes_read": len(returned),
                "truncated": truncated,
                "output_truncated": False,
            },
            content,
        )

    def query_data(self, sql: str, sources: list[QuerySource]) -> str:
        """Run a bounded SELECT over safe CSV or Parquet source aliases."""
        return self._query_engine.execute(sql, sources)

    def _tabular_file(
        self,
        location_name: str,
        path: str,
    ) -> tuple[str, str]:
        resolved = self._catalog.resolve(location_name, path)
        info = self._catalog.filesystem(location_name).info(resolved)
        _require_file(info, path)
        suffix = PurePosixPath(path).suffix.lower()
        try:
            file_format = _TABULAR_SUFFIXES[suffix]
        except KeyError as exc:
            raise ValueError(
                "Tabular inspection supports only CSV and Parquet files."
            ) from exc
        return resolved, file_format


def _directory_entry(
    catalog: LocationCatalog,
    location_name: str,
    info: dict[str, object],
) -> dict[str, object]:
    backend_path = str(info.get("name", ""))
    if not backend_path:
        raise ValueError("Filesystem directory metadata omitted the entry name.")
    return {
        "path": catalog.relative_path(location_name, backend_path),
        "type": info.get("type", "unknown"),
        "size": info.get("size"),
        "modified": _modified_value(info),
    }


def _modified_value(info: dict[str, object]) -> object:
    for key in ("modified", "last_modified", "mtime", "LastModified"):
        value = info.get(key)
        if value is not None:
            if isinstance(value, (datetime, date, time)):
                return value.isoformat()
            return normalize_value(value)
    return None


def _detect_format(path: str) -> str:
    suffix = PurePosixPath(path).suffix.lower()
    if suffix in _TABULAR_SUFFIXES:
        return _TABULAR_SUFFIXES[suffix]
    if suffix == ".json":
        return "json"
    if suffix in {".log", ".txt"}:
        return "text"
    return "unknown"


def _require_file(info: dict[str, object], path: str) -> None:
    if info.get("type") == "directory":
        raise ValueError(f"Expected a file but found a directory: {path!r}.")


def _read_schema(
    filesystem: AbstractFileSystem,
    path: str,
    file_format: str,
) -> pa.Schema:
    raw = cast(BinaryIO, filesystem.open(path))
    with raw:
        if file_format == "parquet":
            return arrow_parquet.ParquetFile(raw).schema_arrow
        reader = arrow_csv.open_csv(
            raw,
            read_options=arrow_csv.ReadOptions(
                block_size=1024 * 1024,
                use_threads=False,
            ),
        )
        return reader.schema


def _read_preview(
    filesystem: AbstractFileSystem,
    path: str,
    file_format: str,
    limit: int,
) -> pa.Table:
    raw = cast(BinaryIO, filesystem.open(path))
    with raw:
        if file_format == "parquet":
            parquet_file = arrow_parquet.ParquetFile(raw)
            return _take_batches(
                parquet_file.iter_batches(batch_size=limit),
                parquet_file.schema_arrow,
                limit,
            )

        reader = arrow_csv.open_csv(
            raw,
            read_options=arrow_csv.ReadOptions(
                block_size=1024 * 1024,
                use_threads=False,
            ),
        )
        batches: list[pa.RecordBatch] = []
        row_count = 0
        while row_count < limit:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break
            remaining = limit - row_count
            selected = batch.slice(0, remaining)
            batches.append(selected)
            row_count += selected.num_rows
        return pa.Table.from_batches(batches, schema=reader.schema)


def _take_batches(
    batches: Iterable[pa.RecordBatch],
    schema: pa.Schema,
    limit: int,
) -> pa.Table:
    selected_batches: list[pa.RecordBatch] = []
    row_count = 0
    for batch in batches:
        remaining = limit - row_count
        if remaining <= 0:
            break
        selected = batch.slice(0, remaining)
        selected_batches.append(selected)
        row_count += selected.num_rows
    return pa.Table.from_batches(selected_batches, schema=schema)
