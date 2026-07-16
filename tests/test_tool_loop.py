"""Tests for the shared agent tool loop."""

import contextlib
import io
import unittest
from types import SimpleNamespace

import pandas as pd

from analytics_agent.messages import FunctionCallMessage, function_call_message
from analytics_agent.tools import (
    DataframeCatalog,
    DatasetSpec,
    create_dataframe_tools,
    create_incident_response_tools,
    run_tool_loop,
)


class FakeProvider:
    """Deterministic provider that requests one tool and then finishes."""

    def __init__(self, call: FunctionCallMessage) -> None:
        self.turn = 0
        self.tool_outputs: list[tuple[str, str]] = []
        self.call = call

    def generate(self) -> SimpleNamespace:
        """Return a minimal response object for the current turn."""
        self.turn += 1
        return SimpleNamespace(
            output=[],
            model_dump_json=lambda indent: f'{{"turn": {self.turn}}}',
        )

    def add_response_output(self, response: object) -> list[FunctionCallMessage]:
        """Request the health tool on the first turn only."""
        del response
        return [self.call] if self.turn == 1 else []

    def add_tool_output(self, call_id: str, output: str) -> None:
        """Record a tool result returned to the model."""
        self.tool_outputs.append((call_id, output))

    def serialized_history(self, provider: str) -> list[object]:
        """Return an empty display history for the test."""
        del provider
        return []


class ToolLoopTests(unittest.TestCase):
    """Verify tool dispatch independently of the OpenAI network client."""

    def test_incident_tool_call_is_executed_and_returned(self) -> None:
        """The shared loop should route incident calls through their registry."""
        provider = FakeProvider(
            function_call_message(
                call_id="call_health",
                name="get_server_health",
                arguments_raw='{"server_id":"payment-server-01"}',
            )
        )
        registry, _ = create_incident_response_tools()

        with contextlib.redirect_stdout(io.StringIO()):
            run_tool_loop(provider, registry, max_turns=2)  # type: ignore[arg-type]

        self.assertEqual(provider.turn, 2)
        self.assertEqual(provider.tool_outputs[0][0], "call_health")
        self.assertIn('"cpu": "98%"', provider.tool_outputs[0][1])

    def test_dataframe_tool_call_is_executed_and_returned(self) -> None:
        """The shared loop should also route dataframe calls through their registry."""
        catalog = DataframeCatalog.from_specs(
            [DatasetSpec("Records", pd.DataFrame({"record_id": [1, 2]}))]
        )
        registry, _ = create_dataframe_tools(catalog)
        provider = FakeProvider(
            function_call_message(
                call_id="call_list_dataframes",
                name="list_dataframes",
                arguments_raw="{}",
            )
        )

        with contextlib.redirect_stdout(io.StringIO()):
            run_tool_loop(provider, registry, max_turns=2)  # type: ignore[arg-type]

        self.assertEqual(provider.turn, 2)
        self.assertEqual(provider.tool_outputs[0][0], "call_list_dataframes")
        self.assertIn("Records: 2 rows x 1 columns", provider.tool_outputs[0][1])


if __name__ == "__main__":
    unittest.main()
