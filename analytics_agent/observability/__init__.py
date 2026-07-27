"""Observability configuration for the analytics agent."""

from analytics_agent.observability.tracing import configure_tracing, get_tracer

__all__ = [
    "configure_tracing",
    "get_tracer",
]
