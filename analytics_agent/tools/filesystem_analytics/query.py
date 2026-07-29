"""Guarded DuckDB execution over catalog-resolved fsspec sources."""

from __future__ import annotations

import glob
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import urlsplit

import duckdb

from analytics_agent.filesystem import LocationCatalog
from analytics_agent.tools.filesystem_analytics.models import (
    MAX_QUERY_ROWS,
    MAX_SOURCE_FILES,
    QuerySource,
)
from analytics_agent.tools.filesystem_analytics.serialization import (
    bounded_rows_result,
    normalize_rows,
)

_SAFE_ALIAS = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
_MAX_RESULT_COLUMNS = 200


@dataclass(frozen=True, slots=True)
class _ResolvedSource:
    alias: str
    location_name: str
    paths: tuple[str, ...]
    uris: tuple[str, ...]
    format: str


class DuckDBQueryEngine:
    """Execute bounded SELECT statements against safe temporary source views."""

    def __init__(self, catalog: LocationCatalog) -> None:
        self._catalog = catalog

    def execute(self, sql: str, sources: list[QuerySource]) -> str:
        """Execute one SELECT statement and return a bounded JSON envelope."""
        connection = duckdb.connect(database=":memory:")
        try:
            validated_sql = validate_select_sql(connection, sql)
            resolved_sources = self._resolve_sources(sources)
            self._register_filesystems(connection, resolved_sources)
            self._create_source_views(connection, resolved_sources)
            self._lock_external_access(connection, resolved_sources)

            try:
                cursor = connection.execute(validated_sql)
                description = cursor.description or []
                columns = [str(column[0]) for column in description]
                if len(columns) > _MAX_RESULT_COLUMNS:
                    raise ValueError(
                        f"Queries may return at most {_MAX_RESULT_COLUMNS} columns."
                    )
                fetched = cursor.fetchmany(MAX_QUERY_ROWS + 1)
            except Exception as exc:
                raise ValueError(
                    f"Filesystem query could not be executed: {exc}"
                ) from exc

            truncated = len(fetched) > MAX_QUERY_ROWS
            rows = normalize_rows(fetched[:MAX_QUERY_ROWS])
            return bounded_rows_result(
                {
                    "sql": validated_sql,
                    "columns": columns,
                    "sources": {
                        source.alias: {
                            "location_name": source.location_name,
                            "format": source.format,
                            "matched_file_count": len(source.paths),
                        }
                        for source in resolved_sources
                    },
                },
                rows,
                requested_truncated=truncated,
            )
        finally:
            connection.close()

    def _resolve_sources(
        self,
        sources: list[QuerySource],
    ) -> tuple[_ResolvedSource, ...]:
        aliases: set[str] = set()
        resolved: list[_ResolvedSource] = []
        matched_file_count = 0
        for source in sources:
            alias = source.alias.strip()
            if not _SAFE_ALIAS.fullmatch(alias):
                raise ValueError(
                    f"Invalid source alias {alias!r}; use a simple SQL identifier."
                )
            normalized_alias = alias.casefold()
            if normalized_alias in aliases:
                raise ValueError(f"Duplicate source alias: {alias!r}.")
            aliases.add(normalized_alias)

            if glob.has_magic(source.path):
                paths = self._catalog.glob(source.location_name, source.path)
            else:
                paths = [
                    self._catalog.resolve(source.location_name, source.path),
                ]
            if not paths:
                raise FileNotFoundError(f"Source {alias!r} did not match any files.")
            if matched_file_count + len(paths) > MAX_SOURCE_FILES:
                raise ValueError(
                    f"A query may read at most {MAX_SOURCE_FILES} matched files."
                )

            file_paths: list[str] = []
            for path in paths:
                info = self._catalog.filesystem(source.location_name).info(path)
                if info.get("type") == "directory":
                    continue
                file_paths.append(path)
            if not file_paths:
                raise ValueError(f"Source {alias!r} matched no readable files.")
            matched_file_count += len(file_paths)
            if matched_file_count > MAX_SOURCE_FILES:
                raise ValueError(
                    f"A query may read at most {MAX_SOURCE_FILES} matched files."
                )

            formats = {_tabular_format(path) for path in file_paths}
            if len(formats) != 1:
                raise ValueError(f"Source {alias!r} mixes CSV and Parquet files.")
            uris = tuple(
                self._catalog.filesystem(source.location_name).unstrip_protocol(path)
                for path in file_paths
            )
            resolved.append(
                _ResolvedSource(
                    alias=alias,
                    location_name=source.location_name,
                    paths=tuple(file_paths),
                    uris=uris,
                    format=formats.pop(),
                )
            )
        return tuple(resolved)

    def _register_filesystems(
        self,
        connection: duckdb.DuckDBPyConnection,
        sources: tuple[_ResolvedSource, ...],
    ) -> None:
        registered: set[str] = set()
        adls_accounts: set[str] = set()
        for source in sources:
            location = self._catalog.get(source.location_name)
            filesystem = self._catalog.filesystem(source.location_name)
            protocol = filesystem.protocol
            primary_protocol = protocol[0] if isinstance(protocol, tuple) else protocol
            if location.backend == "adls":
                account = _adls_account(location.uri)
                if account:
                    adls_accounts.add(account)
                if len(adls_accounts) > 1:
                    raise ValueError(
                        "One query cannot span multiple ADLS storage accounts."
                    )
            if primary_protocol in registered:
                continue
            connection.register_filesystem(filesystem)
            registered.add(primary_protocol)

    @staticmethod
    def _create_source_views(
        connection: duckdb.DuckDBPyConnection,
        sources: tuple[_ResolvedSource, ...],
    ) -> None:
        for source in sources:
            path_expression = _sql_path_expression(source.uris)
            if source.format == "parquet":
                reader = f"read_parquet({path_expression}, union_by_name = true)"
            else:
                reader = (
                    f"read_csv_auto({path_expression}, union_by_name = true, "
                    "header = true)"
                )
            quoted_alias = f'"{source.alias.replace(chr(34), chr(34) * 2)}"'
            connection.execute(
                f"CREATE TEMP VIEW {quoted_alias} AS SELECT * FROM {reader}"
            )

    def _lock_external_access(
        self,
        connection: duckdb.DuckDBPyConnection,
        sources: tuple[_ResolvedSource, ...],
    ) -> None:
        allowed_paths: list[str] = []
        allowed_directories: list[str] = []
        seen_locations: set[str] = set()
        for source in sources:
            allowed_paths.extend(source.uris)
            location = self._catalog.get(source.location_name)
            if location.backend == "local":
                allowed_paths.extend(source.paths)
            if source.location_name in seen_locations:
                continue
            seen_locations.add(source.location_name)
            root = self._catalog.root_path(source.location_name)
            root_uri = self._catalog.filesystem(source.location_name).unstrip_protocol(
                root
            )
            allowed_directories.append(root_uri)
            if location.backend == "local":
                allowed_directories.append(root)

        connection.execute("SET allowed_paths = ?", [allowed_paths])
        connection.execute("SET allowed_directories = ?", [allowed_directories])
        connection.execute("SET enable_external_access = false")
        connection.execute("SET lock_configuration = true")


def validate_select_sql(
    connection: duckdb.DuckDBPyConnection,
    sql: str,
) -> str:
    """Parse SQL and require exactly one SELECT statement."""
    try:
        statements = connection.extract_statements(sql)
    except Exception as exc:
        raise ValueError(f"SQL is malformed: {exc}") from exc
    if len(statements) != 1:
        raise ValueError("SQL must contain exactly one statement.")
    statement = statements[0]
    if statement.type != duckdb.StatementType.SELECT:
        raise ValueError("SQL must be a SELECT statement.")
    return statement.query.strip()


def _tabular_format(path: str) -> str:
    suffix = PurePosixPath(path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    raise ValueError(
        f"Unsupported query source format for {path!r}; use CSV or Parquet."
    )


def _sql_path_expression(uris: tuple[str, ...]) -> str:
    literals = [_sql_literal(uri) for uri in uris]
    return literals[0] if len(literals) == 1 else f"[{','.join(literals)}]"


def _sql_literal(value: str) -> str:
    return f"'{value.replace(chr(39), chr(39) * 2)}'"


def _adls_account(uri: str) -> str:
    host = urlsplit(uri).netloc.rsplit("@", maxsplit=1)[-1].lower()
    if host.endswith((".dfs.core.windows.net", ".blob.core.windows.net")):
        return host
    return ""
