"""Tests for dataset catalogs and dataframe tool registration."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analytics_agent.tools import (
    DataframeCatalog,
    DatasetSpec,
    create_dataframe_tools,
    create_incident_response_tools,
    load_dataset_specs,
)
from analytics_agent.tools.registry import ToolDefinition, ToolInput, ToolRegistry


class DataframeCatalogTests(unittest.TestCase):
    """Verify catalog validation and CSV loading."""

    def test_catalog_normalizes_names_and_infers_unique_id(self) -> None:
        """Catalog entries should have normalized names and inferred identifiers."""
        dataframe = pd.DataFrame({"record_id": [1, 2], "name": ["A", "B"]})

        catalog = DataframeCatalog.from_specs(
            [DatasetSpec(" Records ", dataframe, " Example data ")]
        )

        entry = catalog.get("Records")
        self.assertEqual(entry.description, "Example data")
        self.assertEqual(entry.id_column, "record_id")
        self.assertEqual(catalog.names(), ["Records"])

    def test_catalog_rejects_invalid_specs(self) -> None:
        """Empty, duplicate, and non-dataframe specs should be rejected."""
        dataframe = pd.DataFrame({"value": [1]})

        with self.assertRaisesRegex(ValueError, "non-empty"):
            DataframeCatalog.from_specs([DatasetSpec(" ", dataframe)])
        with self.assertRaisesRegex(ValueError, "Duplicate dataset name"):
            DataframeCatalog.from_specs(
                [DatasetSpec("Example", dataframe), DatasetSpec("Example", dataframe)]
            )
        with self.assertRaisesRegex(ValueError, "must be a pandas DataFrame"):
            DataframeCatalog.from_specs(
                [DatasetSpec("Invalid", "not a dataframe")]  # type: ignore[arg-type]
            )

    def test_load_dataset_specs_uses_known_and_fallback_metadata(self) -> None:
        """CSV loading should apply configured metadata and sensible defaults."""
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory)
            pd.DataFrame({"policy_id": [1]}).to_csv(
                data_path / "hospital_policy.csv", index=False
            )
            pd.DataFrame({"item_id": [2]}).to_csv(
                data_path / "inventory.csv", index=False
            )

            specs = load_dataset_specs(directory)

        self.assertEqual(
            [spec.name for spec in specs], ["Hospital Policy", "inventory"]
        )
        self.assertIn("patient care policies", specs[0].description)
        self.assertEqual(specs[1].description, "Dataset loaded from inventory.csv.")
        self.assertTrue(specs[1].source_path.endswith("inventory.csv"))

    def test_load_dataset_specs_reports_missing_and_unreadable_csvs(self) -> None:
        """CSV discovery and read failures should be translated to clear errors."""
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "No CSV files found"):
                load_dataset_specs(directory)

            unreadable_csv = Path(directory) / "broken.csv"
            unreadable_csv.mkdir()
            with self.assertRaisesRegex(ValueError, "Failed to load"):
                load_dataset_specs(directory)


class ToolRegistryTests(unittest.TestCase):
    """Verify schema generation and tool execution through the public registry."""

    def setUp(self) -> None:
        """Create a representative in-memory catalog and tool registry."""
        dataframe = pd.DataFrame(
            {
                "record_id": [1, 2, 3],
                "category": ["A", "B", "A"],
                "name": ["Alpha", "Beta", "Gamma"],
                "value": [10, 20, 30],
            }
        )
        catalog = DataframeCatalog.from_specs(
            [
                DatasetSpec(
                    "Records",
                    dataframe,
                    "Example records.",
                    "/tmp/records.csv",
                )
            ]
        )
        self.registry, self.schemas = create_dataframe_tools(catalog)

    def test_registry_contains_all_tools_and_matching_schemas(self) -> None:
        """The registry and schema list should expose the same seven tools."""
        expected_names = [
            "list_dataframes",
            "describe_dataframe",
            "preview_dataframe",
            "search_rows",
            "filter_rows",
            "aggregate_rows",
            "distinct_values",
        ]

        self.assertEqual(list(self.registry), expected_names)
        self.assertEqual([schema["name"] for schema in self.schemas], expected_names)
        self.assertTrue(all(schema["type"] == "function" for schema in self.schemas))

    def test_catalog_inspection_tools_return_expected_content(self) -> None:
        """List, describe, and preview tools should retain their text contracts."""
        listed = self.registry["list_dataframes"]()
        described = self.registry["describe_dataframe"](dataset_name="Records")
        previewed = self.registry["preview_dataframe"](dataset_name="Records", limit=2)

        self.assertIn("Records: 3 rows x 4 columns", listed)
        self.assertIn("source=records.csv", listed)
        self.assertIn("ID column: record_id", described)
        self.assertIn("Showing first 2 rows.", previewed)
        self.assertIn("Beta", previewed)

    def test_query_tools_search_filter_aggregate_and_count_values(self) -> None:
        """Query tools should execute validated dataframe operations end to end."""
        searched = self.registry["search_rows"](query="Gamma")
        filtered = self.registry["filter_rows"](
            dataset_name="Records",
            conditions=[{"column": "category", "operator": "eq", "value": "A"}],
            columns=["name", "value"],
            sort_by="value",
            sort_desc=True,
        )
        aggregated = self.registry["aggregate_rows"](
            dataset_name="Records",
            metric="sum",
            metric_column="value",
            filters=[{"column": "category", "operator": "eq", "value": "A"}],
        )
        distinct = self.registry["distinct_values"](
            dataset_name="Records", column="category"
        )

        self.assertIn("record_id=3", searched)
        self.assertIn("Matched rows: 2", filtered)
        self.assertLess(filtered.index("Gamma"), filtered.index("Alpha"))
        self.assertIn("Rows after filters: 2", aggregated)
        self.assertIn("40", aggregated)
        self.assertIn("A", distinct)
        self.assertIn("2", distinct)

    def test_validation_and_execution_errors_are_returned_to_the_model(self) -> None:
        """Invalid arguments and catalog failures should remain readable strings."""
        invalid = self.registry["preview_dataframe"](dataset_name="Records", limit=21)
        failed = self.registry["describe_dataframe"](dataset_name="Missing")

        self.assertIn("Invalid arguments for tool 'preview_dataframe'", invalid)
        self.assertIn("Tool 'describe_dataframe' failed", failed)
        self.assertIn("Unknown dataset 'Missing'", failed)


class IncidentResponseToolTests(unittest.TestCase):
    """Verify incident-response composition and shared registry behavior."""

    def setUp(self) -> None:
        """Create incident tools through the public factory."""
        self.registry, self.schemas = create_incident_response_tools()

    def test_factory_pairs_each_tool_with_an_openai_schema(self) -> None:
        """Runtime tools and provider schemas should stay in lockstep."""
        expected_names = [
            "get_server_health",
            "fetch_recent_logs",
            "restart_service",
            "escalate_incident",
        ]

        self.assertEqual(list(self.registry), expected_names)
        self.assertEqual([schema["name"] for schema in self.schemas], expected_names)
        self.assertTrue(
            all(
                schema["parameters"]["additionalProperties"] is False
                for schema in self.schemas
            )
        )

    def test_registry_validates_and_executes_incident_tools(self) -> None:
        """Defaults and validation should be applied before tool execution."""
        health = json.loads(
            self.registry.execute(
                "get_server_health", {"server_id": "payment-server-01"}
            )
        )
        logs = json.loads(
            self.registry.execute(
                "fetch_recent_logs", {"server_id": "db-node-02", "lines": 2}
            )
        )
        invalid = self.registry.execute(
            "restart_service", {"server_id": "db-node-02", "unexpected": True}
        )

        self.assertEqual(health["cpu"], "98%")
        self.assertEqual(len(logs["logs"]), 2)
        self.assertIn("Invalid arguments for tool 'restart_service'", invalid)
        self.assertEqual(
            self.registry.execute("missing_tool", {}), "Unknown tool: missing_tool"
        )

    def test_registry_rejects_duplicate_tool_names(self) -> None:
        """Ambiguous dispatch should fail while composing a registry."""

        class NoInput(ToolInput):
            """Arguments for a tool without parameters."""

        def duplicate() -> str:
            return "first"

        first = ToolDefinition(duplicate, NoInput)
        second = ToolDefinition(duplicate, NoInput)

        with self.assertRaisesRegex(ValueError, "Duplicate tool name: duplicate"):
            ToolRegistry([first, second])


if __name__ == "__main__":
    unittest.main()
