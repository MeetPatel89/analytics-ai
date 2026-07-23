"""Tests for interactive agent runtime composition."""

import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from analytics_agent.agent_runtime import (
    AgentRunConfig,
    available_providers,
    create_openai_provider,
)
from analytics_agent.providers.openai_provider import list_available_models
from analytics_agent.tools import (
    ToolChain,
    build_tools_for_chains,
    default_system_prompt,
    default_user_prompt,
)


class AgentRunConfigTests(unittest.TestCase):
    """Verify validation for configurations collected by the interactive CLI."""

    def test_config_requires_model_chains_and_prompts(self) -> None:
        """Incomplete configurations should fail before a provider is created."""
        with self.assertRaisesRegex(ValueError, "model"):
            AgentRunConfig("openai", "", (ToolChain.DATAFRAME,), "system", "task")
        with self.assertRaisesRegex(ValueError, "tool chain"):
            AgentRunConfig("openai", "model", (), "system", "task")
        with self.assertRaisesRegex(ValueError, "system prompt"):
            AgentRunConfig("openai", "model", (ToolChain.DATAFRAME,), " ", "task")
        with self.assertRaisesRegex(ValueError, "user task"):
            AgentRunConfig("openai", "model", (ToolChain.DATAFRAME,), "system", " ")

    def test_defaults_change_with_selected_tool_chains(self) -> None:
        """Generated prompts should mention each selected capability."""
        both = (ToolChain.DATAFRAME, ToolChain.INCIDENT_RESPONSE)

        self.assertIn("data assistant", default_system_prompt(both))
        self.assertIn("incident-response", default_system_prompt(both))
        self.assertIn("payment-server-01", default_user_prompt(both))

    def test_provider_registry_exposes_openai_runtime_metadata(self) -> None:
        """The interactive runtime should discover OpenAI through its registry."""
        providers = available_providers()

        self.assertEqual(len(providers), 1)
        self.assertEqual(providers[0].name, "openai")
        self.assertEqual(providers[0].label, "OpenAI")
        self.assertEqual(providers[0].credential_env_var, "OPENAI_API_KEY")
        self.assertIs(providers[0].list_models, list_available_models)
        self.assertIs(providers[0].create_provider, create_openai_provider)


class ToolChainCompositionTests(unittest.TestCase):
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
            (ToolChain.DATAFRAME, ToolChain.INCIDENT_RESPONSE)
        )

        self.assertEqual(len(registry), 11)
        self.assertEqual(list(registry), [schema["name"] for schema in schemas])
        self.assertIn("list_dataframes", registry)
        self.assertIn("get_server_health", registry)


class OpenAIModelDiscoveryTests(unittest.TestCase):
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

        self.assertEqual(models, ["gpt-4.1", "gpt-4o-mini"])

    def test_list_available_models_requires_an_api_key(self) -> None:
        """Discovery should fail before issuing a request without credentials."""
        with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
            list_available_models("")


if __name__ == "__main__":
    unittest.main()
