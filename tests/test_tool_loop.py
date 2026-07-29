"""Tests for the shared agent tool loop."""

import contextlib
import io
import tempfile
from pathlib import Path
from types import SimpleNamespace

from analytics_agent.filesystem import DataLocation, LocationCatalog
from analytics_agent.messages import FunctionCallMessage, function_call_message
from analytics_agent.tools import (
    ToolDefinition,
    ToolInput,
    ToolRegistry,
    create_filesystem_analytics_tools,
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


class TestToolLoop:
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

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            run_tool_loop(provider, registry, max_turns=2)

        assert provider.turn == 2
        assert provider.tool_outputs[0][0] == "call_health"
        assert '"cpu": "98%"' in provider.tool_outputs[0][1]
        assert "Calling tool: get_server_health" in output.getvalue()
        assert "Tool result:" in output.getvalue()
        assert "Final answer: None" in output.getvalue()

    def test_filesystem_tool_call_is_executed_and_returned(self) -> None:
        """The shared loop should route filesystem calls through their registry."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = LocationCatalog([DataLocation("records", root.as_uri(), "local")])
            registry, _ = create_filesystem_analytics_tools(catalog)
            provider = FakeProvider(
                function_call_message(
                    call_id="call_list_locations",
                    name="list_locations",
                    arguments_raw="{}",
                )
            )

            with contextlib.redirect_stdout(io.StringIO()):
                run_tool_loop(provider, registry, max_turns=2)

        assert provider.turn == 2
        assert provider.tool_outputs[0][0] == "call_list_locations"
        assert '"name":"records"' in provider.tool_outputs[0][1]

    def test_display_formatter_does_not_change_the_provider_tool_output(self) -> None:
        """Console-friendly output should remain separate from model payloads."""

        class NoInput(ToolInput):
            """A tool with no arguments."""

        def raw_result() -> str:
            """Return a structured placeholder payload."""
            return '{"rows":[[1],[2]]}'

        registry = ToolRegistry(
            [
                ToolDefinition(
                    raw_result,
                    NoInput,
                    output_formatter=lambda output: f"formatted table\n{output}",
                )
            ]
        )
        provider = FakeProvider(
            function_call_message(
                call_id="call_raw_result",
                name="raw_result",
                arguments_raw="{}",
            )
        )
        console = io.StringIO()

        with contextlib.redirect_stdout(console):
            run_tool_loop(provider, registry, max_turns=2)

        assert provider.tool_outputs[0][1] == '{"rows":[[1],[2]]}'
        assert 'Tool result:\nformatted table\n{"rows":[[1],[2]]}' in console.getvalue()

    def test_verbose_output_includes_provider_diagnostics(self) -> None:
        """Verbose runs should retain the raw provider diagnostics."""
        provider = FakeProvider(
            function_call_message(
                call_id="call_health",
                name="get_server_health",
                arguments_raw='{"server_id":"payment-server-01"}',
            )
        )
        registry, _ = create_incident_response_tools()
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            run_tool_loop(provider, registry, max_turns=2, verbose=True)

        assert '"turn": 1' in output.getvalue()

    def test_invalid_tool_arguments_are_returned_to_the_provider(self) -> None:
        """Malformed provider arguments should not terminate the agent loop."""
        provider = FakeProvider(
            function_call_message(
                call_id="call_health",
                name="get_server_health",
                arguments_raw="{invalid-json",
            )
        )
        registry, _ = create_incident_response_tools()

        with contextlib.redirect_stdout(io.StringIO()):
            run_tool_loop(provider, registry, max_turns=2)

        assert provider.turn == 2
        assert provider.tool_outputs[0][0] == "call_health"
        assert "arguments must be valid JSON" in provider.tool_outputs[0][1]
