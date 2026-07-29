"""Tests for the filesystem analytics agent runtime."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from analytics_agent.agent_runtime import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    AgentRunConfig,
    ProviderDefinition,
    available_providers,
    build_run_tools,
    create_openai_provider,
)
from analytics_agent.providers.openai_provider import list_available_models


class TestAgentRunConfig:
    """Verify validation for configurations collected by the interactive CLI."""

    def test_config_requires_model_and_prompts(self) -> None:
        """Incomplete configurations should fail before a provider is created."""
        with pytest.raises(ValueError, match="model"):
            AgentRunConfig("openai", "", "system", "task")
        with pytest.raises(ValueError, match="system prompt"):
            AgentRunConfig("openai", "model", " ", "task")
        with pytest.raises(ValueError, match="user task"):
            AgentRunConfig("openai", "model", "system", " ")

    def test_defaults_describe_safe_filesystem_discovery_and_querying(self) -> None:
        """Built-in prompts should guide discovery and bounded analytics."""
        assert "filesystem analytics assistant" in DEFAULT_SYSTEM_PROMPT
        assert "read-only" in DEFAULT_SYSTEM_PROMPT
        assert "inspect schemas" in DEFAULT_SYSTEM_PROMPT
        assert "SELECT-only" in DEFAULT_SYSTEM_PROMPT
        assert "available data locations" in DEFAULT_USER_PROMPT

    def test_provider_registry_exposes_openai_runtime_metadata(self) -> None:
        """The interactive runtime should discover OpenAI through its registry."""
        providers = available_providers()

        assert len(providers) == 1
        assert providers[0].name == "openai"
        assert providers[0].label == "OpenAI"
        assert providers[0].credential_env_var == "OPENAI_API_KEY"
        assert providers[0].list_models is list_available_models
        assert providers[0].create_provider is create_openai_provider


class TestFilesystemToolComposition:
    """Verify every run receives the filesystem analytics tool set."""

    def test_build_run_tools_uses_the_selected_zero_config_data_path(self) -> None:
        """The CLI data-path override should become the fallback local catalog."""
        with tempfile.TemporaryDirectory() as directory:
            config = AgentRunConfig(
                provider="openai",
                model="selected-model",
                system_prompt="system",
                user_prompt="task",
                data_path=Path(directory),
            )

            registry, schemas = build_run_tools(config)
            locations = json.loads(registry["list_locations"]())

        assert len(registry) == 7
        assert list(registry) == [schema["name"] for schema in schemas]
        assert locations["locations"][0]["uri"] == Path(directory).resolve().as_uri()


class TestOpenAIModelDiscovery:
    """Verify model discovery is account-scoped and deterministic."""

    def test_list_available_models_sorts_and_deduplicates_ids(self) -> None:
        """The CLI should receive a stable list for filtering and paging."""
        fake_client = SimpleNamespace(
            models=SimpleNamespace(
                list=lambda: SimpleNamespace(
                    data=[
                        SimpleNamespace(id="gpt-4o-mini"),
                        SimpleNamespace(id="gpt-4.1"),
                        SimpleNamespace(id="gpt-4o-mini"),
                    ]
                )
            )
        )
        with patch(
            "analytics_agent.providers.openai_provider.OpenAI",
            return_value=fake_client,
        ):
            models = list_available_models("test-key")

        assert models == ["gpt-4.1", "gpt-4o-mini"]

    def test_list_available_models_requires_an_api_key(self) -> None:
        """Discovery should fail before issuing a request without credentials."""
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            list_available_models("")


def test_provider_definition_does_not_require_tool_specific_generation() -> None:
    """Provider registration should expose only the shared agent-loop behavior."""
    definition = ProviderDefinition(
        name="openai",
        label="OpenAI",
        credential_env_var="OPENAI_API_KEY",
        list_models=Mock(),
        create_provider=Mock(),
    )

    assert definition.name == "openai"
