"""Tests for the first phase of OpenTelemetry tracing."""

import json
import subprocess
import sys
import textwrap
import unittest


class TracingConfigurationTests(unittest.TestCase):
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

        self.assertEqual(completed.stdout, "")

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
                    "agent.tool_chains": ("incident_response",),
                    "agent.max_turns": 10,
                },
            ):
                pass
            """
        )
        exported_span = json.loads(completed.stdout)

        self.assertEqual(exported_span["name"], "agent_run")
        self.assertIsNone(exported_span["parent_id"])
        self.assertRegex(exported_span["context"]["trace_id"], r"^0x[0-9a-f]{32}$")
        self.assertRegex(exported_span["context"]["span_id"], r"^0x[0-9a-f]{16}$")
        self.assertEqual(
            exported_span["attributes"],
            {
                "agent.model": "learning-model",
                "agent.tool_chains": ["incident_response"],
                "agent.max_turns": 10,
            },
        )
        self.assertEqual(
            exported_span["resource"]["attributes"]["service.name"],
            "analytics-agent",
        )

    def _run_python(self, source: str) -> subprocess.CompletedProcess[str]:
        """Run a tracing example in an isolated interpreter process."""
        return subprocess.run(
            [sys.executable, "-c", textwrap.dedent(source)],
            check=True,
            capture_output=True,
            text=True,
        )
