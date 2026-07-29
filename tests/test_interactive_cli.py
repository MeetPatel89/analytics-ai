"""Tests for provider selection in the interactive CLI."""

import io
from contextlib import nullcontext
from unittest.mock import Mock, patch

from rich.console import Console

from analytics_agent.agent_runtime import ProviderDefinition
from analytics_agent.interactive_cli import InteractiveCLI
from analytics_agent.tools import ToolChain


class TestInteractiveCLIProvider:
    """Verify provider-first model discovery and runtime creation."""

    def setup_method(self) -> None:
        """Create an isolated CLI with one fake provider registration."""
        self.output = io.StringIO()
        self.list_models = Mock(return_value=["model-a"])
        self.create_provider = Mock(return_value=Mock())
        self.create_generation_model = Mock(return_value=Mock())
        self.tracer = Mock()
        self.tracer.start_as_current_span.return_value = nullcontext()
        self.provider = ProviderDefinition(
            name="openai",
            label="OpenAI",
            credential_env_var="OPENAI_API_KEY",
            list_models=self.list_models,
            create_provider=self.create_provider,
            create_generation_model=self.create_generation_model,
        )
        self.cli = InteractiveCLI(
            console=Console(file=self.output, force_terminal=False),
            providers=(self.provider,),
            tracer=self.tracer,
        )

    def test_select_provider_returns_registered_definition(self) -> None:
        """The picker should return the provider selected by number."""
        with patch(
            "analytics_agent.interactive_cli.Prompt.ask",
            return_value="1",
        ):
            selected = self.cli._select_provider()

        assert selected is self.provider
        assert "OpenAI" in self.output.getvalue()

    def test_select_provider_can_return_to_main_menu(self) -> None:
        """Quitting the provider picker should cancel configuration."""
        with patch(
            "analytics_agent.interactive_cli.Prompt.ask",
            return_value="q",
        ):
            selected = self.cli._select_provider()

        assert selected is None

    def test_missing_provider_credential_skips_model_discovery(self) -> None:
        """Credentials should be checked only after a provider is selected."""
        with (
            patch.object(self.cli, "_select_provider", return_value=self.provider),
            patch("analytics_agent.interactive_cli.os.getenv", return_value=""),
        ):
            self.cli._configure_and_run()

        self.list_models.assert_not_called()
        assert "OPENAI_API_KEY is not configured" in self.output.getvalue()

    def test_selected_provider_discovers_models_with_its_credential(self) -> None:
        """The selected provider should receive its configured API key."""
        with (
            patch.object(self.cli, "_select_provider", return_value=self.provider),
            patch(
                "analytics_agent.interactive_cli.os.getenv",
                return_value="test-key",
            ),
            patch.object(self.cli, "_select_model", return_value=None),
        ):
            self.cli._configure_and_run()

        self.list_models.assert_called_once_with("test-key")

    def test_model_discovery_error_is_displayed_and_cancels_selection(self) -> None:
        """Provider discovery failures should return safely to the menu."""
        self.list_models.side_effect = RuntimeError("discovery failed")

        models = self.cli._load_models(self.provider, "test-key")

        assert models is None
        assert "discovery failed" in self.output.getvalue()

    def test_empty_model_catalog_is_displayed_and_cancels_selection(self) -> None:
        """An account with no visible models should not enter the picker."""
        self.list_models.return_value = []

        models = self.cli._load_models(self.provider, "test-key")

        assert models is None
        assert "No OpenAI models are available" in self.output.getvalue()

    def test_selected_provider_factory_creates_the_runtime(self) -> None:
        """The provider registration should own runtime construction."""
        tool_registry = Mock()
        tool_schemas = [{"type": "function"}]
        with (
            patch.object(self.cli, "_select_provider", return_value=self.provider),
            patch(
                "analytics_agent.interactive_cli.os.getenv",
                return_value="test-key",
            ),
            patch.object(self.cli, "_select_model", return_value="model-a"),
            patch.object(
                self.cli,
                "_select_tool_chains",
                return_value=(ToolChain.INCIDENT_RESPONSE,),
            ),
            patch.object(
                self.cli,
                "_select_system_prompt",
                return_value="system prompt",
            ),
            patch(
                "analytics_agent.interactive_cli.Prompt.ask",
                return_value="user task",
            ),
            patch(
                "analytics_agent.interactive_cli.Confirm.ask",
                side_effect=[False, True],
            ),
            patch(
                "analytics_agent.interactive_cli.build_run_tools",
                return_value=(tool_registry, tool_schemas),
            ) as build_run_tools,
            patch("analytics_agent.interactive_cli.run_tool_loop") as run_tool_loop,
        ):
            self.cli._configure_and_run()

        config = self.create_provider.call_args.args[0]
        assert config.provider == "openai"
        assert config.model == "model-a"
        build_run_tools.assert_called_once_with(
            self.provider,
            config,
            "test-key",
        )
        self.create_provider.assert_called_once_with(
            config,
            "test-key",
            tool_schemas,
        )
        run_tool_loop.assert_called_once_with(
            self.create_provider.return_value,
            tool_registry,
            verbose=False,
        )
        self.tracer.start_as_current_span.assert_called_once_with(
            "agent_run",
            attributes={
                "agent.provider": "OpenAI",
                "agent.model": "model-a",
                "agent.tool_chains": ("incident_response",),
                "agent.max_turns": 10,
                "agent.system_prompt": "system prompt",
                "agent.user_prompt": "user task",
            },
        )

    def test_filesystem_chain_uses_the_same_runtime_composition_path(self) -> None:
        """The CLI should use the shared composition path for filesystem tools."""
        tool_registry = Mock()
        tool_schemas = [{"type": "function"}]
        with (
            patch.object(self.cli, "_select_provider", return_value=self.provider),
            patch(
                "analytics_agent.interactive_cli.os.getenv",
                return_value="test-key",
            ),
            patch.object(self.cli, "_select_model", return_value="model-a"),
            patch.object(
                self.cli,
                "_select_tool_chains",
                return_value=(ToolChain.FILESYSTEM_ANALYTICS,),
            ),
            patch.object(
                self.cli,
                "_select_system_prompt",
                return_value="system prompt",
            ),
            patch(
                "analytics_agent.interactive_cli.Prompt.ask",
                return_value="user task",
            ),
            patch(
                "analytics_agent.interactive_cli.Confirm.ask",
                side_effect=[False, True],
            ),
            patch(
                "analytics_agent.interactive_cli.build_run_tools",
                return_value=(tool_registry, tool_schemas),
            ) as build_run_tools,
            patch("analytics_agent.interactive_cli.run_tool_loop"),
        ):
            self.cli._configure_and_run()

        config = self.create_provider.call_args.args[0]
        build_run_tools.assert_called_once_with(
            self.provider,
            config,
            "test-key",
        )
