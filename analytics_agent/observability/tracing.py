"""OpenTelemetry tracing configuration for application entry points."""

from __future__ import annotations

import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.trace import Tracer

INSTRUMENTATION_SCOPE = "analytics_agent"
SERVICE_NAME_VALUE = "analytics-agent"
TRACES_EXPORTER_ENV_VAR = "OTEL_TRACES_EXPORTER"
DEFAULT_TRACES_EXPORTER = "console"
SUPPORTED_TRACES_EXPORTERS = frozenset({"console", "otlp", "none"})

_tracer_provider: TracerProvider | None = None


def get_tracer() -> Tracer:
    """Return the application tracer.

    Before :func:`configure_tracing` is called, OpenTelemetry's API supplies a
    non-recording tracer. This keeps imports and offline tests inert by default.
    """
    return trace.get_tracer(INSTRUMENTATION_SCOPE)


def configure_tracing() -> Tracer:
    """Configure the environment-selected trace exporter once per process.

    ``OTEL_TRACES_EXPORTER`` accepts ``console`` (the default), ``otlp``, or
    ``none``. The OTLP HTTP exporter reads its endpoint, headers, and other
    transport settings from the standard OpenTelemetry environment variables.
    """
    global _tracer_provider

    if _tracer_provider is None:
        exporter_name = (
            os.getenv(TRACES_EXPORTER_ENV_VAR, DEFAULT_TRACES_EXPORTER).strip().lower()
            or DEFAULT_TRACES_EXPORTER
        )
        if exporter_name not in SUPPORTED_TRACES_EXPORTERS:
            supported = ", ".join(sorted(SUPPORTED_TRACES_EXPORTERS))
            msg = (
                f"Unsupported {TRACES_EXPORTER_ENV_VAR} value "
                f"{exporter_name!r}; expected one of: {supported}"
            )
            raise ValueError(msg)

        resource = Resource.create({SERVICE_NAME: SERVICE_NAME_VALUE})
        provider = TracerProvider(resource=resource)
        if exporter_name == "console":
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        elif exporter_name == "otlp":
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        _tracer_provider = provider

    return get_tracer()


def shutdown_tracing() -> None:
    """Flush pending spans and shut down the configured tracing pipeline."""
    if _tracer_provider is not None:
        _tracer_provider.shutdown()
