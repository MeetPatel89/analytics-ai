"""Offline tests for filesystem navigation, inspection, and querying."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as arrow_parquet
from fsspec.implementations.memory import MemoryFileSystem

from analytics_agent.filesystem import DataLocation, LocationCatalog
from analytics_agent.tools import (
    FILESYSTEM_ANALYTICS_TOOL_NAMES,
    create_filesystem_analytics_tools,
)
from analytics_agent.tools.filesystem_analytics import MAX_TOOL_OUTPUT_BYTES


def _parquet_bytes() -> bytes:
    sink = pa.BufferOutputStream()
    arrow_parquet.write_table(
        pa.table(
            {
                "record_id": list(range(60)),
                "category": [
                    "even" if value % 2 == 0 else "odd" for value in range(60)
                ],
                "amount": [value + 0.5 for value in range(60)],
            }
        ),
        sink,
    )
    return sink.getvalue().to_pybytes()


def _memory_catalog() -> LocationCatalog:
    memory = MemoryFileSystem()
    memory.pipe_file(
        "/datasets/records.csv",
        b"record_id,name,amount\n1,alpha,10.5\n2,beta,20.25\n",
    )
    memory.pipe_file("/datasets/records.parquet", _parquet_bytes())
    memory.pipe_file("/datasets/notes.txt", b"first line\nsecond line\n")
    return LocationCatalog(
        [DataLocation("remote", "memory://datasets", "memory")],
        filesystems={"remote": memory},
    )


class TestFilesystemAnalyticsTools:
    """Verify all public filesystem tools against a remote-like memory backend."""

    def setup_method(self) -> None:
        """Create a complete registry over an in-memory filesystem."""
        self.catalog = _memory_catalog()
        self.registry, self.schemas = create_filesystem_analytics_tools(self.catalog)

    def test_factory_registers_all_strict_tools(self) -> None:
        """Tool names and OpenAI schemas should remain in lockstep."""
        expected = list(FILESYSTEM_ANALYTICS_TOOL_NAMES)

        assert list(self.registry) == expected
        assert [schema["name"] for schema in self.schemas] == expected
        assert all(schema["strict"] is True for schema in self.schemas)
        assert "title" not in json.dumps(self.schemas)

    def test_navigation_reports_locations_entries_and_file_metadata(self) -> None:
        """Navigation tools should emit bounded structured metadata."""
        locations = json.loads(self.registry["list_locations"]())
        listing = json.loads(
            self.registry["list_directory"](
                location_name="remote",
                path="*.csv",
                limit=10,
            )
        )
        info = json.loads(
            self.registry["get_file_info"](
                location_name="remote",
                path="records.parquet",
            )
        )

        assert locations["locations"][0]["backend"] == "memory"
        assert listing["entries"][0]["path"] == "records.csv"
        assert not listing["truncated"]
        assert info["format"] == "parquet"
        assert info["size"] > 0

    def test_schema_preview_and_text_reads_are_backend_agnostic(self) -> None:
        """Inspection should read CSV, Parquet, and bounded text via fsspec."""
        csv_schema = json.loads(
            self.registry["inspect_schema"](
                location_name="remote",
                path="records.csv",
            )
        )
        parquet_preview = json.loads(
            self.registry["preview_data"](
                location_name="remote",
                path="records.parquet",
                limit=2,
            )
        )
        text = json.loads(
            self.registry["read_text_file"](
                location_name="remote",
                path="notes.txt",
                offset=6,
                max_bytes=4,
            )
        )

        assert [column["name"] for column in csv_schema["columns"]] == [
            "record_id",
            "name",
            "amount",
        ]
        assert parquet_preview["columns"] == ["record_id", "category", "amount"]
        assert parquet_preview["rows"][1] == [1, "odd", 1.5]
        assert parquet_preview["truncated"]
        assert text["content"] == "line"
        assert text["bytes_read"] == 4
        assert text["truncated"]

    def test_validation_and_traversal_failures_are_model_readable(self) -> None:
        """Invalid inputs and unsafe paths should be returned as tool errors."""
        traversal = self.registry["get_file_info"](
            location_name="remote",
            path="../secret.txt",
        )
        bad_text = self.registry["read_text_file"](
            location_name="remote",
            path="records.csv",
            offset=0,
            max_bytes=100,
        )
        bad_limit = self.registry["preview_data"](
            location_name="remote",
            path="records.csv",
            limit=21,
        )

        assert "Path traversal" in traversal
        assert "supports only" in bad_text
        assert "Invalid arguments" in bad_limit

    def test_text_output_has_a_hard_serialized_byte_cap(self) -> None:
        """Escaped control characters must not exceed the global output limit."""
        memory = MemoryFileSystem()
        memory.pipe_file("/bounded/noisy.log", b"\x00" * (32 * 1024))
        catalog = LocationCatalog(
            [DataLocation("bounded", "memory://bounded", "memory")],
            filesystems={"bounded": memory},
        )
        registry, _ = create_filesystem_analytics_tools(catalog)

        output = registry["read_text_file"](
            location_name="bounded",
            path="noisy.log",
            offset=0,
            max_bytes=32 * 1024,
        )
        result = json.loads(output)

        assert len(output.encode("utf-8")) <= MAX_TOOL_OUTPUT_BYTES
        assert result["output_truncated"]
        assert result["truncated"]

    def test_query_data_aggregates_sources_and_caps_rows(self) -> None:
        """DuckDB should query fsspec views and preserve the 50-row cap."""
        aggregate = json.loads(
            self.registry["query_data"](
                sql=(
                    "SELECT category, SUM(amount) AS total "
                    "FROM sales GROUP BY category ORDER BY category"
                ),
                sources=[
                    {
                        "alias": "sales",
                        "location_name": "remote",
                        "path": "records.parquet",
                    }
                ],
            )
        )
        bounded = json.loads(
            self.registry["query_data"](
                sql="SELECT * FROM sales ORDER BY record_id",
                sources=[
                    {
                        "alias": "sales",
                        "location_name": "remote",
                        "path": "*.parquet",
                    }
                ],
            )
        )

        assert aggregate["columns"] == ["category", "total"]
        assert aggregate["returned_row_count"] == 2
        assert bounded["returned_row_count"] == 50
        assert bounded["truncated"]
        assert bounded["rows"][-1][0] == 49

    def test_query_data_rejects_unsafe_or_non_select_sql(self) -> None:
        """SQL guardrails should block mutation, multiple statements, and escape."""
        sources = [
            {
                "alias": "sales",
                "location_name": "remote",
                "path": "records.parquet",
            }
        ]

        delete = self.registry["query_data"](
            sql="DELETE FROM sales",
            sources=sources,
        )
        multiple = self.registry["query_data"](
            sql="SELECT 1; SELECT 2",
            sources=sources,
        )
        escaped = self.registry["query_data"](
            sql="SELECT * FROM read_csv_auto('/etc/passwd')",
            sources=sources,
        )

        assert "SELECT statement" in delete
        assert "exactly one" in multiple
        assert "disabled by configuration" in escaped.lower()


class TestFilesystemComposition:
    """Verify filesystem tools compose without unrelated dependencies."""

    def test_filesystem_tool_names_match_registry_and_schemas(self) -> None:
        """Public metadata, handlers, and schemas should preserve one order."""
        catalog = _memory_catalog()
        registry, schemas = create_filesystem_analytics_tools(catalog)

        assert list(registry) == [schema["name"] for schema in schemas]
        assert tuple(registry) == FILESYSTEM_ANALYTICS_TOOL_NAMES

    def test_local_filesystem_query_uses_the_same_engine(self) -> None:
        """Local files should be queryable through the same catalog abstraction."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "values.csv").write_text(
                "value\n2\n3\n",
                encoding="utf-8",
            )
            catalog = LocationCatalog([DataLocation("local", root.as_uri(), "local")])
            registry, _ = create_filesystem_analytics_tools(catalog)

            result = json.loads(
                registry["query_data"](
                    sql="SELECT SUM(value) AS total FROM values_data",
                    sources=[
                        {
                            "alias": "values_data",
                            "location_name": "local",
                            "path": "values.csv",
                        }
                    ],
                )
            )

        assert result["rows"] == [[5]]
