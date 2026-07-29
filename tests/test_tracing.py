"""Tests for the first phase of OpenTelemetry tracing."""

import json
import re
import subprocess
import sys
import textwrap


class TestTracingConfiguration:
    """Verify tracing is inert by default and visible after configuration."""

    def test_unconfigured_tracer_does_not_export_spans(self) -> None:
        """Using the API alone should remain silent in a fresh process."""
        completed = self._run_python(
            """
            from analytics_agent.observability import get_tracer

            tracer = get_tracer()
            with tracer.start_as_current_span("silent_span"):
                pass
            """
        )

        assert completed.stdout == ""

    def test_configured_tracer_exports_span_json_to_console(self) -> None:
        """The console pipeline should export attributes and resource metadata."""
        completed = self._run_python(
            """
            from analytics_agent.observability import configure_tracing

            tracer = configure_tracing()
            with tracer.start_as_current_span(
                "agent_run",
                attributes={
                    "agent.model": "learning-model",
                    "agent.tool_set": "filesystem_analytics",
                    "agent.max_turns": 10,
                },
            ):
                pass
            """
        )
        exported_span = json.loads(completed.stdout)

        assert exported_span["name"] == "agent_run"
        assert exported_span["parent_id"] is None
        assert re.search(r"^0x[0-9a-f]{32}$", exported_span["context"]["trace_id"])
        assert re.search(r"^0x[0-9a-f]{16}$", exported_span["context"]["span_id"])
        assert exported_span["attributes"] == (
            {
                "agent.model": "learning-model",
                "agent.tool_set": "filesystem_analytics",
                "agent.max_turns": 10,
            }
        )
        assert (
            exported_span["resource"]["attributes"]["service.name"] == "analytics-agent"
        )

    def test_agent_operations_form_one_parent_child_trace(self) -> None:
        """Turn, LLM, and tool spans should inherit the active parent context."""
        completed = self._run_python(
            """
            import contextlib
            import io
            import json
            from types import SimpleNamespace
            from unittest.mock import Mock, patch

            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
                InMemorySpanExporter,
            )

            from analytics_agent.messages import generate_initial_messages
            from analytics_agent.providers.openai_provider import OpenAIProvider
            from analytics_agent.tools.registry import (
                ToolDefinition,
                ToolInput,
                ToolRegistry,
            )
            from analytics_agent.tools.tool_loop import run_tool_loop

            class EchoInput(ToolInput):
                value: str

            def echo(value: str) -> str:
                return value

            tool_call = SimpleNamespace(
                type="function_call",
                id="fc_1",
                call_id="call_1",
                name="echo",
                arguments='{"value":"hello"}',
                status="completed",
            )
            responses = Mock()
            responses.create.side_effect = [
                SimpleNamespace(
                    id="resp_1",
                    model="learning-model",
                    output=[tool_call],
                ),
                SimpleNamespace(
                    id="resp_2",
                    model="learning-model",
                    output=[],
                ),
            ]
            client = SimpleNamespace(responses=responses)
            with patch(
                "analytics_agent.providers.openai_provider.OpenAI",
                return_value=client,
            ):
                agent_provider = OpenAIProvider(
                    api_key="test-key",
                    model="learning-model",
                    tools=[{"name": "echo"}],
                    messages=generate_initial_messages("system", "user"),
                )

            exporter = InMemorySpanExporter()
            tracer_provider = TracerProvider()
            tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
            trace.set_tracer_provider(tracer_provider)

            registry = ToolRegistry([ToolDefinition(echo, EchoInput)])
            root_tracer = trace.get_tracer("hierarchy-test")
            with contextlib.redirect_stdout(io.StringIO()):
                with root_tracer.start_as_current_span("agent_run"):
                    run_tool_loop(agent_provider, registry, max_turns=2)

            exported = []
            for span in exporter.get_finished_spans():
                context = span.get_span_context()
                exported.append(
                    {
                        "name": span.name,
                        "trace_id": f"{context.trace_id:032x}",
                        "span_id": f"{context.span_id:016x}",
                        "parent_id": (
                            f"{span.parent.span_id:016x}" if span.parent else None
                        ),
                        "attributes": dict(span.attributes),
                    }
                )
            print(json.dumps(exported))
            """
        )
        spans = json.loads(completed.stdout)

        assert [span["name"] for span in spans] == (
            [
                "llm.generate",
                "tool.execute",
                "agent.turn",
                "llm.generate",
                "agent.turn",
                "agent_run",
            ]
        )
        assert len({span["trace_id"] for span in spans}) == 1

        root = spans[-1]
        turns = [span for span in spans if span["name"] == "agent.turn"]
        turns.sort(key=lambda span: span["attributes"]["agent.turn.number"])
        llm_calls = [span for span in spans if span["name"] == "llm.generate"]
        llm_calls.sort(
            key=lambda span: span["attributes"]["llm.request.input_item_count"]
        )
        tool_call = next(span for span in spans if span["name"] == "tool.execute")

        assert root["parent_id"] is None
        assert [turn["parent_id"] for turn in turns] == [
            root["span_id"],
            root["span_id"],
        ]
        assert llm_calls[0]["parent_id"] == turns[0]["span_id"]
        assert tool_call["parent_id"] == turns[0]["span_id"]
        assert llm_calls[1]["parent_id"] == turns[1]["span_id"]
        assert turns[0]["attributes"] == {"agent.turn.number": 1, "agent.turn.max": 2}
        assert llm_calls[0]["attributes"] == (
            {
                "llm.model": "learning-model",
                "llm.request.input_item_count": 2,
                "llm.request.tool_count": 1,
                "llm.response.output_item_count": 1,
                "llm.response.id": "resp_1",
                "llm.response.model": "learning-model",
            }
        )
        assert tool_call["attributes"] == (
            {
                "tool.name": "echo",
                "tool.arguments": '{"value": "hello"}',
            }
        )

    def _run_python(self, source: str) -> subprocess.CompletedProcess[str]:
        """Run a tracing example in an isolated interpreter process."""
        return subprocess.run(
            [sys.executable, "-c", textwrap.dedent(source)],
            check=True,
            capture_output=True,
            text=True,
        )
