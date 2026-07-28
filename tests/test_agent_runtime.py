"""Tests for interactive agent runtime composition."""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from analytics_agent.agent_runtime import (
    AgentRunConfig,
    ProviderDefinition,
    available_providers,
    build_run_tools,
    create_openai_provider,
)
from analytics_agent.providers.openai_provider import (
    OpenAIGenerationModel,
    list_available_models,
)
from analytics_agent.tools import (
    ToolChain,
    ToolChainDependencies,
    build_tools_for_chains,
    default_system_prompt,
    default_user_prompt,
)


class TestAgentRunConfig:
    """Verify validation for configurations collected by the interactive CLI."""

    def test_config_requires_model_chains_and_prompts(self) -> None:
        """Incomplete configurations should fail before a provider is created."""
        with pytest.raises(ValueError, match="model"):
            AgentRunConfig("openai", "", (ToolChain.DATAFRAME,), "system", "task")
        with pytest.raises(ValueError, match="tool chain"):
            AgentRunConfig("openai", "model", (), "system", "task")
        with pytest.raises(ValueError, match="system prompt"):
            AgentRunConfig("openai", "model", (ToolChain.DATAFRAME,), " ", "task")
        with pytest.raises(ValueError, match="user task"):
            AgentRunConfig("openai", "model", (ToolChain.DATAFRAME,), "system", " ")

    def test_defaults_change_with_selected_tool_chains(self) -> None:
        """Generated prompts should mention each selected capability."""
        both = (ToolChain.DATAFRAME, ToolChain.INCIDENT_RESPONSE)

        assert "data assistant" in default_system_prompt(both)
        assert "incident-response" in default_system_prompt(both)
        assert "payment-server-01" in default_user_prompt(both)

    def test_provider_registry_exposes_openai_runtime_metadata(self) -> None:
        """The interactive runtime should discover OpenAI through its registry."""
        providers = available_providers()

        assert len(providers) == 1
        assert providers[0].name == "openai"
        assert providers[0].label == "OpenAI"
        assert providers[0].credential_env_var == "OPENAI_API_KEY"
        assert providers[0].list_models is list_available_models
        assert providers[0].create_provider is create_openai_provider
        assert providers[0].create_generation_model is OpenAIGenerationModel


class TestToolChainComposition:
    """Verify selected chains produce one combined executable tool set."""

    def test_importing_dataframe_entry_point_has_no_runtime_side_effects(self) -> None:
        """Importing an entry point should not read datasets or credentials."""
        with patch(
            "analytics_agent.tools.tool_chains.load_dataset_specs"
        ) as load_dataset_specs:
            from analytics_agent import dataframe_main

            importlib.reload(dataframe_main)

        load_dataset_specs.assert_not_called()

    def test_combined_chains_include_all_tools_and_schemas(self) -> None:
        """Both chains should preserve order and have matching OpenAI schemas."""
        registry, schemas = build_tools_for_chains(
            (ToolChain.DATAFRAME, ToolChain.INCIDENT_RESPONSE),
            ToolChainDependencies(create_generation_model=Mock()),
        )

        assert len(registry) == 11
        assert list(registry) == [schema["name"] for schema in schemas]
        assert "list_dataframes" in registry
        assert "get_server_health" in registry

    def test_sql_defaults_describe_the_three_step_workflow(self) -> None:
        """SQL-only prompts should exercise lookup, analysis, and visualization."""
        system = default_system_prompt((ToolChain.SQL_ANALYZER,))
        user = default_user_prompt((ToolChain.SQL_ANALYZER,))

        assert "lookup_sales_data first" in system
        assert "analyze_sales_data" in system
        assert "generate_visualization" in system
        assert "analyze" in user
        assert "visualization" in user


class TestRunToolComposition:
    """Verify provider generation capabilities are bound once at runtime."""

    def _definition(self, generation_factory: Mock) -> ProviderDefinition:
        return ProviderDefinition(
            name="openai",
            label="OpenAI",
            credential_env_var="OPENAI_API_KEY",
            list_models=Mock(),
            create_provider=Mock(),
            create_generation_model=generation_factory,
        )

    def _config(self, chain: ToolChain) -> AgentRunConfig:
        return AgentRunConfig(
            provider="openai",
            model="selected-model",
            tool_chains=(chain,),
            system_prompt="system",
            user_prompt="task",
        )

    def test_generation_model_is_lazy_for_chains_that_do_not_use_it(self) -> None:
        """Composing an unrelated chain should not construct a generation client."""
        generation_factory = Mock(return_value=Mock())

        registry, _ = build_run_tools(
            self._definition(generation_factory),
            self._config(ToolChain.INCIDENT_RESPONSE),
            "test-key",
        )

        assert "get_server_health" in registry
        generation_factory.assert_not_called()

    def test_sql_chain_binds_the_selected_credential_and_model(self) -> None:
        """SQL composition should request the provider's generic capability."""
        generation_factory = Mock(return_value=Mock())

        registry, _ = build_run_tools(
            self._definition(generation_factory),
            self._config(ToolChain.SQL_ANALYZER),
            "test-key",
        )

        assert "lookup_sales_data" in registry
        generation_factory.assert_called_once_with("test-key", "selected-model")


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
