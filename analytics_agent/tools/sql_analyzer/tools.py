"""DuckDB-backed sales lookup, analysis, and visualization tools."""

from __future__ import annotations

import ast
import json
import math
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import cast

from duckdb import DuckDBPyConnection, StatementType
from pydantic import JsonValue

from analytics_agent.providers.generation import GenerationModel, StructuredOutputT
from analytics_agent.tools.sql_analyzer.models import (
    SalesQueryResult,
    SalesSQLQuery,
    VisualizationCode,
    VisualizationConfig,
)

DEFAULT_SALES_DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "Store_Sales_Price_Elasticity_Promotions_Data.parquet"
)
MAX_QUERY_ROWS = 50


class SQLAnalyzerTools:
    """Model-assisted operations over one local sales Parquet dataset."""

    def __init__(
        self,
        model: GenerationModel,
        data_path: Path | None = None,
    ) -> None:
        self._model = model
        self._data_path = (
            Path(data_path) if data_path is not None else DEFAULT_SALES_DATA_PATH
        )
        self._connection: DuckDBPyConnection | None = None

    def lookup_sales_data(self, prompt: str) -> SalesQueryResult:
        """Generate one read-only sales query and return at most 50 rows."""
        connection = self._get_connection()
        schema = connection.execute("DESCRIBE sales").fetchall()
        schema_text = "\n".join(
            f"- {name}: {data_type}" for name, data_type, *_ in schema
        )
        generated = self._generate_structured(
            (
                "Write one DuckDB SELECT statement that answers the sales-data "
                "request below. Query only the in-memory table named sales. Do not "
                "read files, attach databases, use extensions, mutate data, or emit "
                "multiple statements. Return the SQL in the required structure.\n\n"
                f"Sales table schema:\n{schema_text}\n\n"
                f"Request:\n{prompt}"
            ),
            SalesSQLQuery,
        )
        sql = self._validate_select(connection, generated.sql)

        try:
            cursor = connection.execute(
                f"SELECT * FROM ({sql}) AS generated_sales_query "
                f"LIMIT {MAX_QUERY_ROWS + 1}"
            )
            columns = [description[0] for description in cursor.description]
            fetched_rows = cursor.fetchall()
        except Exception as exc:
            raise ValueError(
                f"Generated sales query could not be executed: {exc}"
            ) from exc

        truncated = len(fetched_rows) > MAX_QUERY_ROWS
        normalized_rows = [
            {
                column: _normalize_json_value(value)
                for column, value in zip(columns, row, strict=True)
            }
            for row in fetched_rows[:MAX_QUERY_ROWS]
        ]
        return SalesQueryResult(
            sql=sql,
            columns=columns,
            rows=normalized_rows,
            returned_row_count=len(normalized_rows),
            truncated=truncated,
        )

    def analyze_sales_data(self, prompt: str, data: SalesQueryResult) -> str:
        """Analyze a validated sales-query result in response to a user goal."""
        analysis = self._model.generate_text(
            "Analyze the validated sales-query result below. Base every factual "
            "claim on the supplied data, call out truncation when present, and "
            "answer the requested analysis goal in clear prose.\n\n"
            f"Analysis goal:\n{prompt}\n\n"
            f"Sales query result:\n{data.model_dump_json(indent=2)}"
        )
        if not isinstance(analysis, str) or not analysis.strip():
            raise RuntimeError("SQL analyzer model returned no analysis text.")
        return analysis.strip()

    def generate_visualization(
        self,
        data: SalesQueryResult,
        visualization_goal: str,
    ) -> str:
        """Generate syntactically valid pandas/Matplotlib code without running it."""
        data_json = data.model_dump_json(indent=2)
        config = self._generate_structured(
            (
                "Choose a visualization configuration for the validated query "
                "result. Use only line, bar, or scatter. Both axis names must "
                "exactly match a supplied column.\n\n"
                f"Visualization goal:\n{visualization_goal}\n\n"
                f"Sales query result:\n{data_json}"
            ),
            VisualizationConfig,
        )
        missing_axes = [
            axis for axis in (config.x_axis, config.y_axis) if axis not in data.columns
        ]
        if missing_axes:
            missing = ", ".join(repr(axis) for axis in missing_axes)
            raise ValueError(f"Visualization axes are not result columns: {missing}")

        generated = self._generate_structured(
            (
                "Write self-contained Python source using pandas and Matplotlib for "
                "the requested chart. Embed the supplied result rows directly in "
                "the source; do not read files, access the network, execute SQL, or "
                "use placeholders. Create the configured chart and call "
                "plt.show(). Return only the source in the required structure. "
                "This code will be reviewed before anyone executes it.\n\n"
                f"Visualization goal:\n{visualization_goal}\n\n"
                f"Configuration:\n{config.model_dump_json(indent=2)}\n\n"
                f"Sales query result:\n{data_json}"
            ),
            VisualizationCode,
        )
        source = generated.code.strip()
        try:
            ast.parse(source)
        except SyntaxError as exc:
            raise ValueError(
                f"Generated visualization is invalid Python: {exc}"
            ) from exc
        return source

    def _get_connection(self) -> DuckDBPyConnection:
        if self._connection is not None:
            return self._connection
        if not self._data_path.is_file():
            raise FileNotFoundError(
                f"Sales Parquet file was not found: {self._data_path}"
            )

        connection = DuckDBPyConnection(database=":memory:")
        try:
            connection.execute(
                "CREATE TABLE sales AS SELECT * FROM read_parquet(?)",
                [str(self._data_path)],
            )
            connection.execute("SET enable_external_access = false")
        except Exception as exc:
            connection.close()
            raise ValueError(
                f"Unable to load sales Parquet file '{self._data_path}': {exc}"
            ) from exc
        self._connection = connection
        return connection

    @staticmethod
    def _validate_select(
        connection: DuckDBPyConnection,
        sql: str,
    ) -> str:
        try:
            statements = connection.extract_statements(sql)
        except Exception as exc:
            raise ValueError(f"Generated SQL is malformed: {exc}") from exc
        if len(statements) != 1:
            raise ValueError("Generated SQL must contain exactly one statement.")
        statement = statements[0]
        if statement.type != StatementType.SELECT:
            raise ValueError("Generated SQL must be a SELECT statement.")
        return statement.query.strip()

    def _generate_structured(
        self,
        prompt: str,
        response_model: type[StructuredOutputT],
    ) -> StructuredOutputT:
        generated = self._model.generate_structured(prompt, response_model)
        return response_model.model_validate(generated)


def _normalize_json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return cast(JsonValue, value)
    if isinstance(value, float):
        return cast(JsonValue, value if math.isfinite(value) else None)
    if isinstance(value, Decimal):
        normalized_decimal = (
            int(value) if value == value.to_integral_value() else float(value)
        )
        return cast(JsonValue, normalized_decimal)
    if isinstance(value, (datetime, date, time)):
        return cast(JsonValue, value.isoformat())
    if isinstance(value, dict):
        return cast(
            JsonValue,
            {str(key): _normalize_json_value(item) for key, item in value.items()},
        )
    if isinstance(value, (list, tuple)):
        return cast(JsonValue, [_normalize_json_value(item) for item in value])

    scalar_item = getattr(value, "item", None)
    if callable(scalar_item):
        return _normalize_json_value(scalar_item())
    try:
        json.dumps(value, allow_nan=False)
    except TypeError, ValueError:
        return cast(JsonValue, str(value))
    return cast(JsonValue, value)
