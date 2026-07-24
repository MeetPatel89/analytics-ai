"""Tests for the local sales SQL analyzer tool chain."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import duckdb
from pydantic import BaseModel

from analytics_agent.providers.generation import StructuredOutputT
from analytics_agent.providers.openai_provider import OpenAIGenerationModel
from analytics_agent.tools import (
    ToolChain,
    ToolChainDependencies,
    build_tools_for_chains,
)
from analytics_agent.tools.registry import ToolDefinition, ToolInput, ToolRegistry
from analytics_agent.tools.sql_analyzer import (
    SalesQueryResult,
    VisualizationConfig,
    create_sql_analyzer_tools,
)
from analytics_agent.tools.sql_analyzer.models import (
    SalesSQLQuery,
    VisualizationCode,
)


class FakeGenerationModel:
    """Return queued structured values and deterministic analysis text."""

    def __init__(
        self,
        structured: list[object] | None = None,
        text: str = "Sales increased.",
    ) -> None:
        self.structured = list(structured or [])
        self.text = text
        self.structured_prompts: list[tuple[str, type[BaseModel]]] = []
        self.text_prompts: list[str] = []

    def generate_structured(
        self,
        prompt: str,
        response_model: type[StructuredOutputT],
    ) -> StructuredOutputT:
        """Validate the next queued value as the requested response model."""
        self.structured_prompts.append((prompt, response_model))
        if not self.structured:
            raise RuntimeError("no structured response")
        return response_model.model_validate(self.structured.pop(0))

    def generate_text(self, prompt: str) -> str:
        """Return configured analysis text while retaining the prompt."""
        self.text_prompts.append(prompt)
        return self.text


class FailingGenerationModel(FakeGenerationModel):
    """Raise a model failure for every structured request."""

    def generate_structured(
        self,
        prompt: str,
        response_model: type[StructuredOutputT],
    ) -> StructuredOutputT:
        """Simulate a provider failure."""
        del prompt, response_model
        raise RuntimeError("generation model unavailable")


def _write_sales_parquet(path: Path) -> None:
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute(
            """
            COPY (
                SELECT
                    i::INTEGER AS sale_id,
                    DATE '2026-01-01' + i::INTEGER AS sale_date,
                    CAST(i + 0.25 AS DECIMAL(10, 2)) AS revenue,
                    CASE WHEN i % 2 = 0 THEN 'east' ELSE 'west' END AS region
                FROM range(60) AS values_table(i)
            ) TO ? (FORMAT PARQUET)
            """,
            [str(path)],
        )
    finally:
        connection.close()


class SQLAnalyzerToolTests(unittest.TestCase):
    """Verify SQL lookup, analysis, and visualization behavior."""

    def setUp(self) -> None:
        """Create a temporary Parquet fixture for each test."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.data_path = Path(self.temporary_directory.name) / "sales.parquet"
        _write_sales_parquet(self.data_path)

    def _create_tools(
        self,
        structured: list[object],
        text: str = "Sales increased.",
    ) -> tuple[FakeGenerationModel, ToolRegistry]:
        model = FakeGenerationModel(structured, text)
        registry, _ = create_sql_analyzer_tools(model, self.data_path)
        return model, registry

    def test_factory_registers_three_validated_public_tools(self) -> None:
        """Runtime names and OpenAI schemas should stay in lockstep."""
        model = FakeGenerationModel()
        registry, schemas = create_sql_analyzer_tools(model, self.data_path)

        expected = [
            "lookup_sales_data",
            "analyze_sales_data",
            "generate_visualization",
        ]
        self.assertEqual(list(registry), expected)
        self.assertEqual([schema["name"] for schema in schemas], expected)
        nested_schema = schemas[1]["parameters"]["$defs"]["SalesQueryResult"]
        self.assertIn("returned_row_count", nested_schema["required"])

    def test_lookup_executes_aggregation_and_serializes_json_safe_values(self) -> None:
        """Dates and decimals should be normalized in the result envelope."""
        model, registry = self._create_tools(
            [
                SalesSQLQuery(
                    sql=(
                        "SELECT sale_date, SUM(revenue) AS revenue "
                        "FROM sales GROUP BY sale_date ORDER BY sale_date LIMIT 2"
                    )
                )
            ]
        )

        result = json.loads(registry["lookup_sales_data"](prompt="Daily revenue"))

        self.assertEqual(result["columns"], ["sale_date", "revenue"])
        self.assertEqual(result["returned_row_count"], 2)
        self.assertEqual(result["rows"][0]["sale_date"], "2026-01-01")
        self.assertEqual(result["rows"][0]["revenue"], 0.25)
        self.assertFalse(result["truncated"])
        self.assertIn("sale_date", model.structured_prompts[0][0])

    def test_lookup_caps_rows_and_reports_truncation(self) -> None:
        """The query wrapper should fetch one extra row to detect truncation."""
        _, registry = self._create_tools(
            [{"sql": "SELECT * FROM sales ORDER BY sale_id"}]
        )

        result = json.loads(registry["lookup_sales_data"](prompt="All sales"))

        self.assertEqual(result["returned_row_count"], 50)
        self.assertEqual(len(result["rows"]), 50)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["rows"][-1]["sale_id"], 49)

    def test_lookup_rejects_malformed_multiple_and_non_select_sql(self) -> None:
        """Only one parsed SELECT statement may reach execution."""
        invalid_cases = [
            ("SELECT FROM", "malformed"),
            ("SELECT 1; SELECT 2", "exactly one"),
            ("DELETE FROM sales", "SELECT statement"),
        ]
        for sql, expected_error in invalid_cases:
            with self.subTest(sql=sql):
                _, registry = self._create_tools([{"sql": sql}])
                result = registry["lookup_sales_data"](prompt="Invalid query")
                self.assertIn("Tool 'lookup_sales_data' failed", result)
                self.assertIn(expected_error, result)

    def test_lookup_blocks_external_file_access(self) -> None:
        """SELECT table functions must not bypass the local sales table boundary."""
        _, registry = self._create_tools(
            [{"sql": "SELECT * FROM read_csv_auto('/etc/passwd')"}]
        )

        result = registry["lookup_sales_data"](prompt="Read another file")

        self.assertIn("Tool 'lookup_sales_data' failed", result)
        self.assertIn("disabled by configuration", result.lower())

    def test_lookup_reports_missing_unreadable_and_model_failures(self) -> None:
        """Expected local and provider failures should become readable tool errors."""
        missing_registry, _ = create_sql_analyzer_tools(
            FakeGenerationModel([{"sql": "SELECT 1"}]),
            self.data_path.with_name("missing.parquet"),
        )
        missing = missing_registry["lookup_sales_data"](prompt="Sales")
        self.assertIn("Sales Parquet file was not found", missing)

        invalid_path = self.data_path.with_name("invalid.parquet")
        invalid_path.write_text("not parquet", encoding="utf-8")
        invalid_registry, _ = create_sql_analyzer_tools(
            FakeGenerationModel([{"sql": "SELECT 1"}]),
            invalid_path,
        )
        unreadable = invalid_registry["lookup_sales_data"](prompt="Sales")
        self.assertIn("Unable to load sales Parquet file", unreadable)

        failed_registry, _ = create_sql_analyzer_tools(
            FailingGenerationModel(),
            self.data_path,
        )
        failed = failed_registry["lookup_sales_data"](prompt="Sales")
        self.assertIn("generation model unavailable", failed)

    def test_analysis_validates_nested_data_and_forwards_the_envelope(self) -> None:
        """Analysis should receive the complete validated lookup result."""
        data = SalesQueryResult(
            sql="SELECT region, COUNT(*) AS count FROM sales GROUP BY region",
            columns=["region", "count"],
            rows=[{"region": "east", "count": 30}],
            returned_row_count=1,
            truncated=False,
        )
        model, registry = self._create_tools([], text="East has 30 sales.")

        result = registry["analyze_sales_data"](
            prompt="Summarize regional sales",
            data=data.model_dump(),
        )

        self.assertEqual(result, "East has 30 sales.")
        self.assertIn("Summarize regional sales", model.text_prompts[0])
        self.assertIn('"region": "east"', model.text_prompts[0])
        invalid = registry["analyze_sales_data"](
            prompt="Summarize",
            data={**data.model_dump(), "returned_row_count": 2},
        )
        self.assertIn("Invalid arguments for tool 'analyze_sales_data'", invalid)

    def test_visualization_validates_axes_and_python_source(self) -> None:
        """Chart axes must exist and generated source must parse as Python."""
        data = SalesQueryResult(
            sql="SELECT sale_date, revenue FROM sales LIMIT 2",
            columns=["sale_date", "revenue"],
            rows=[
                {"sale_date": "2026-01-01", "revenue": 0.25},
                {"sale_date": "2026-01-02", "revenue": 1.25},
            ],
            returned_row_count=2,
            truncated=False,
        )
        code = (
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            f"rows = {data.rows!r}\n"
            "df = pd.DataFrame(rows)\n"
            "df.plot(x='sale_date', y='revenue', kind='line')\n"
            "plt.show()"
        )
        model, registry = self._create_tools(
            [
                VisualizationConfig(
                    chart_type="line",
                    x_axis="sale_date",
                    y_axis="revenue",
                    title="Revenue by day",
                ),
                VisualizationCode(code=code),
            ]
        )

        result = registry["generate_visualization"](
            data=data.model_dump(),
            visualization_goal="Plot revenue over time",
        )

        self.assertEqual(result, code)
        self.assertIn('"chart_type": "line"', model.structured_prompts[1][0])
        self.assertIn('"sale_date": "2026-01-01"', model.structured_prompts[1][0])

        _, invalid_axis_registry = self._create_tools(
            [
                {
                    "chart_type": "bar",
                    "x_axis": "missing",
                    "y_axis": "revenue",
                    "title": "Invalid",
                }
            ]
        )
        invalid_axis = invalid_axis_registry["generate_visualization"](
            data=data.model_dump(),
            visualization_goal="Plot",
        )
        self.assertIn("axes are not result columns", invalid_axis)

        _, invalid_code_registry = self._create_tools(
            [
                {
                    "chart_type": "scatter",
                    "x_axis": "sale_date",
                    "y_axis": "revenue",
                    "title": "Invalid source",
                },
                {"code": "if broken syntax"},
            ]
        )
        invalid_code = invalid_code_registry["generate_visualization"](
            data=data.model_dump(),
            visualization_goal="Plot",
        )
        self.assertIn("invalid Python", invalid_code)


class SQLAnalyzerCompositionTests(unittest.TestCase):
    """Verify SQL-chain composition and registry output validation."""

    def test_sql_only_and_combined_composition_use_real_sql_tools(self) -> None:
        """SQL selection should never fall through to incident definitions."""
        model = FakeGenerationModel()
        dependencies = ToolChainDependencies(
            create_generation_model=lambda: model,
        )

        sql_registry, _ = build_tools_for_chains(
            (ToolChain.SQL_ANALYZER,),
            dependencies,
        )
        combined_registry, _ = build_tools_for_chains(
            (ToolChain.SQL_ANALYZER, ToolChain.INCIDENT_RESPONSE),
            dependencies,
        )

        self.assertEqual(
            list(sql_registry),
            [
                "lookup_sales_data",
                "analyze_sales_data",
                "generate_visualization",
            ],
        )
        self.assertEqual(len(combined_registry), 7)
        self.assertIn("get_server_health", combined_registry)

    def test_generation_model_factory_is_only_called_for_sql_composition(self) -> None:
        """Unrelated chains should not eagerly construct a generation model."""
        generation_factory = Mock(return_value=FakeGenerationModel())
        dependencies = ToolChainDependencies(
            create_generation_model=generation_factory,
        )

        registry, _ = build_tools_for_chains(
            (ToolChain.INCIDENT_RESPONSE,),
            dependencies,
        )
        self.assertIn("get_server_health", registry)
        generation_factory.assert_not_called()

        build_tools_for_chains((ToolChain.SQL_ANALYZER,), dependencies)
        generation_factory.assert_called_once_with()

    def test_registry_validates_and_serializes_declared_output_models(self) -> None:
        """Pydantic output contracts should be JSON envelopes."""

        class NoInput(ToolInput):
            """No tool arguments."""

        def valid_result() -> dict[str, object]:
            return {
                "sql": "SELECT 1 AS value",
                "columns": ["value"],
                "rows": [{"value": 1}],
                "returned_row_count": 1,
                "truncated": False,
            }

        definition = ToolDefinition(
            valid_result,
            NoInput,
            output_model=SalesQueryResult,
        )
        registry = ToolRegistry([definition])
        self.assertEqual(json.loads(registry["valid_result"]())["rows"], [{"value": 1}])

        def invalid_result() -> dict[str, object]:
            return {"sql": "SELECT 1"}

        invalid_registry = ToolRegistry(
            [
                ToolDefinition(
                    invalid_result,
                    NoInput,
                    output_model=SalesQueryResult,
                )
            ]
        )
        self.assertIn(
            "Tool 'invalid_result' failed",
            invalid_registry["invalid_result"](),
        )


class OpenAIGenerationModelTests(unittest.TestCase):
    """Verify the OpenAI adapter without issuing live requests."""

    def setUp(self) -> None:
        """Create a fake Responses API client."""
        self.responses = Mock()
        self.client = SimpleNamespace(responses=self.responses)
        patcher = patch(
            "analytics_agent.providers.openai_provider.OpenAI",
            return_value=self.client,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.model = OpenAIGenerationModel("test-key", "selected-model")

    def test_structured_and_text_generation_use_the_selected_model(self) -> None:
        """Both response styles should retain the selected outer model ID."""
        parsed = SalesSQLQuery(sql="SELECT 1")
        self.responses.parse.return_value = SimpleNamespace(
            output=[],
            output_parsed=parsed,
        )
        self.responses.create.return_value = SimpleNamespace(
            output=[],
            output_text="  analysis  ",
        )

        structured = self.model.generate_structured("SQL prompt", SalesSQLQuery)
        text = self.model.generate_text("analysis prompt")

        self.assertEqual(structured, parsed)
        self.assertEqual(text, "analysis")
        self.responses.parse.assert_called_once_with(
            model="selected-model",
            input="SQL prompt",
            text_format=SalesSQLQuery,
        )
        self.responses.create.assert_called_once_with(
            model="selected-model",
            input="analysis prompt",
        )

    def test_missing_refused_and_invalid_outputs_fail_clearly(self) -> None:
        """Provider response problems should raise normal runtime failures."""
        self.responses.parse.return_value = SimpleNamespace(
            output=[],
            output_parsed=None,
        )
        with self.assertRaisesRegex(RuntimeError, "no structured output"):
            self.model.generate_structured("prompt", SalesSQLQuery)

        refusal_content = SimpleNamespace(type="refusal", refusal="not allowed")
        self.responses.create.return_value = SimpleNamespace(
            output=[SimpleNamespace(content=[refusal_content])],
            output_text="",
        )
        with self.assertRaisesRegex(RuntimeError, "refused.*not allowed"):
            self.model.generate_text("prompt")

        self.responses.parse.return_value = SimpleNamespace(
            output=[],
            output_parsed={"sql": ""},
        )
        with self.assertRaises(ValueError):
            self.model.generate_structured("prompt", SalesSQLQuery)


if __name__ == "__main__":
    unittest.main()
