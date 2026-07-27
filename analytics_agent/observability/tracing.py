"""OpenTelemetry tracing configuration for application entry points."""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.trace import Tracer

INSTRUMENTATION_SCOPE = "analytics_agent"
SERVICE_NAME_VALUE = "analytics-agent"

_tracer_provider: TracerProvider | None = None


def get_tracer() -> Tracer:
    """Return the application tracer.

    Before :func:`configure_tracing` is called, OpenTelemetry's API supplies a
    non-recording tracer. This keeps imports and offline tests inert by default.
    """
    return trace.get_tracer(INSTRUMENTATION_SCOPE)


def configure_tracing() -> Tracer:
    """Configure immediate JSON span export to the console once per process."""
    global _tracer_provider

    if _tracer_provider is None:
        resource = Resource.create({SERVICE_NAME: SERVICE_NAME_VALUE})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        _tracer_provider = provider

    return get_tracer()
