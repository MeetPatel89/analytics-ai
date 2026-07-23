"""Interactive terminal interface for configuring and running an agent."""

from __future__ import annotations

import os

import dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from analytics_agent.agent_runtime import (
    AgentRunConfig,
    ProviderDefinition,
    available_providers,
)
from analytics_agent.tools import (
    ToolChain,
    available_tool_chains,
    build_tools_for_chains,
    default_system_prompt,
    default_user_prompt,
    run_tool_loop,
)

MODEL_PAGE_SIZE = 20


class InteractiveCLI:
    """Prompt for agent configuration and run the selected tool loop."""

    def __init__(
        self,
        console: Console | None = None,
        providers: tuple[ProviderDefinition, ...] | None = None,
    ) -> None:
        self.console = console or Console()
        self.providers = providers if providers is not None else available_providers()

    def run(self) -> None:
        """Run the top-level interactive menu until the user exits."""
        dotenv.load_dotenv()
        self.console.print(Panel.fit("Analytics Agent", style="bold cyan"))
        while True:
            self.console.print(
                Panel(
                    "[1] Configure and run an agent\n"
                    "[2] View available providers, models, and tool chains\n"
                    "[3] Exit",
                    title="Main menu",
                )
            )
            choice = Prompt.ask(
                "Select an action",
                choices=["1", "2", "3"],
                default="1",
                console=self.console,
            )
            if choice == "1":
                self._configure_and_run()
            elif choice == "2":
                self._show_available_configuration()
            else:
                self.console.print("Goodbye.")
                return

    def _configure_and_run(self) -> None:
        provider_definition = self._select_provider()
        if provider_definition is None:
            return

        api_key = os.getenv(provider_definition.credential_env_var, "").strip()
        if not api_key:
            self.console.print(
                f"[red]{provider_definition.credential_env_var} is not configured. "
                "Add it to .env and try again.[/red]"
            )
            return

        models = self._load_models(provider_definition, api_key)
        if not models:
            return
        model = self._select_model(models)
        if model is None:
            return

        tool_chains = self._select_tool_chains()
        system_prompt = self._select_system_prompt(tool_chains)
        user_prompt = Prompt.ask(
            "Task for the agent",
            default=default_user_prompt(tool_chains),
            console=self.console,
        ).strip()
        verbose = Confirm.ask(
            "Show verbose provider diagnostics?",
            default=False,
            console=self.console,
        )

        try:
            config = AgentRunConfig(
                provider=provider_definition.name,
                model=model,
                tool_chains=tool_chains,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                verbose=verbose,
            )
        except ValueError as exc:
            self.console.print(f"[red]Invalid configuration: {exc}[/red]")
            return

        self._show_run_summary(config, provider_definition.label)
        if not Confirm.ask(
            "Start this agent run?", default=False, console=self.console
        ):
            self.console.print("Run cancelled.")
            return

        try:
            tool_registry, tool_schemas = build_tools_for_chains(config.tool_chains)
            provider = provider_definition.create_provider(
                config,
                api_key,
                tool_schemas,
            )
            run_tool_loop(provider, tool_registry, verbose=config.verbose)
        except KeyboardInterrupt:
            self.console.print("\nRun interrupted.")
        except Exception as exc:
            self.console.print(f"[red]Agent run failed: {exc}[/red]")

    def _select_provider(self) -> ProviderDefinition | None:
        """Display registered providers and return the selected definition."""
        if not self.providers:
            self.console.print("[yellow]No providers are registered.[/yellow]")
            return None

        table = Table(title="Providers")
        table.add_column("#", justify="right")
        table.add_column("Provider")
        for index, provider in enumerate(self.providers, start=1):
            table.add_row(str(index), provider.label)
        self.console.print(table)

        choices = [str(index) for index in range(1, len(self.providers) + 1)]
        choice = Prompt.ask(
            "Select a provider or (q)uit",
            choices=[*choices, "q"],
            default="1",
            console=self.console,
        ).lower()
        if choice == "q":
            return None
        return self.providers[int(choice) - 1]

    def _load_models(
        self,
        provider: ProviderDefinition,
        api_key: str,
    ) -> list[str] | None:
        try:
            models = provider.list_models(api_key)
        except (RuntimeError, ValueError) as exc:
            self.console.print(f"[red]{exc}[/red]")
            return None
        if not models:
            self.console.print(
                f"[yellow]No {provider.label} models are available for this API "
                "key.[/yellow]"
            )
            return None
        return models

    def _select_model(self, models: list[str]) -> str | None:
        """Display account-available models and select one model ID."""
        return self._select_model_page(models, all_models=models)

    def _filter_and_select_model(self, models: list[str]) -> str | None:
        """Filter the available model IDs before returning to selection."""
        while True:
            filter_text = Prompt.ask(
                "Model ID filter", default="", console=self.console
            ).strip()
            matches = [
                model for model in models if filter_text.lower() in model.lower()
            ]
            if not matches:
                self.console.print(
                    "[yellow]No matching models. Try another filter.[/yellow]"
                )
                continue
            return self._select_model_page(matches, all_models=models)

    def _select_model_page(
        self, models: list[str], *, all_models: list[str]
    ) -> str | None:
        """Display matching models in pages and return the selected model ID."""
        page = 0
        while True:
            start = page * MODEL_PAGE_SIZE
            page_models = models[start : start + MODEL_PAGE_SIZE]
            table = Table(
                title=(
                    f"Available models ({start + 1}-{start + len(page_models)} "
                    f"of {len(models)})"
                )
            )
            table.add_column("#", justify="right")
            table.add_column("Model ID")
            for index, model in enumerate(page_models, start=1):
                table.add_row(str(index), model)
            self.console.print(table)

            choice = Prompt.ask(
                "Select a number, (n)ext, (p)revious, (f)ilter, or (q)uit",
                default="q",
                console=self.console,
            ).lower()
            if choice == "q":
                return None
            if choice == "f":
                return self._filter_and_select_model(all_models)
            if choice == "n" and start + MODEL_PAGE_SIZE < len(models):
                page += 1
                continue
            if choice == "p" and page > 0:
                page -= 1
                continue
            try:
                selection = int(choice)
                if 1 <= selection <= len(page_models):
                    return page_models[selection - 1]
            except ValueError:
                pass
            self.console.print(
                "[yellow]Choose a listed number or a valid command.[/yellow]"
            )

    def _select_tool_chains(self) -> tuple[ToolChain, ...]:
        """Prompt until the user selects one or more registered tool chains."""
        choices = available_tool_chains()
        table = Table(title="Tool chains")
        table.add_column("#", justify="right")
        table.add_column("Tool chain")
        table.add_column("Description")
        for index, option in enumerate(choices, start=1):
            table.add_row(str(index), option.label, option.description)
        self.console.print(table)

        while True:
            raw_selection = Prompt.ask(
                "Select one or both chains (for example: 1,2)",
                default="1",
                console=self.console,
            )
            try:
                indexes = [int(value.strip()) for value in raw_selection.split(",")]
                if not indexes or any(
                    index not in range(1, len(choices) + 1) for index in indexes
                ):
                    raise ValueError
            except ValueError:
                self.console.print(
                    "[yellow]Enter one or more valid comma-separated numbers.[/yellow]"
                )
                continue

            return tuple(dict.fromkeys(choices[index - 1].chain for index in indexes))

    def _select_system_prompt(self, tool_chains: tuple[ToolChain, ...]) -> str:
        """Accept a generated system prompt or collect a multiline replacement."""
        default_prompt = default_system_prompt(tool_chains)
        self.console.print(Panel(default_prompt, title="Default system prompt"))
        if Confirm.ask("Use this system prompt?", default=True, console=self.console):
            return default_prompt

        self.console.print(
            "Enter a system prompt. Finish with a line containing only '.'."
        )
        lines: list[str] = []
        while True:
            line = self.console.input("> ")
            if line == ".":
                break
            lines.append(line)
        return "\n".join(lines).strip()

    def _show_available_configuration(self) -> None:
        """Display supported built-ins and account configuration without secrets."""
        self.console.print(
            "[bold]Saved profiles:[/bold] none (built-in configuration only)"
        )

        provider_table = Table(title="Registered providers")
        provider_table.add_column("Provider")
        provider_table.add_column("Credential")
        provider_table.add_column("Configured")
        provider_credentials: list[tuple[ProviderDefinition, str]] = []
        for provider in self.providers:
            api_key = os.getenv(provider.credential_env_var, "").strip()
            provider_credentials.append((provider, api_key))
            provider_table.add_row(
                provider.label,
                provider.credential_env_var,
                "yes" if api_key else "no",
            )
        self.console.print(provider_table)

        table = Table(title="Registered tool chains")
        table.add_column("Tool chain")
        table.add_column("Tools")
        for option in available_tool_chains():
            table.add_row(option.label, ", ".join(option.tool_names))
        self.console.print(table)

        for provider, api_key in provider_credentials:
            if not api_key:
                self.console.print(
                    f"Set {provider.credential_env_var} to inspect "
                    f"account-available {provider.label} models."
                )
                continue
            models = self._load_models(provider, api_key)
            if models:
                preview = "\n".join(models[:MODEL_PAGE_SIZE])
                suffix = "\n..." if len(models) > MODEL_PAGE_SIZE else ""
                self.console.print(
                    Panel(
                        f"{len(models)} available models\n\n{preview}{suffix}",
                        title=f"{provider.label} models",
                    )
                )

    def _show_run_summary(
        self,
        config: AgentRunConfig,
        provider_label: str,
    ) -> None:
        """Display the selected values before requiring execution confirmation."""
        chains = ", ".join(chain.replace("_", " ") for chain in config.tool_chains)
        self.console.print(
            Panel(
                "\n".join(
                    [
                        f"Provider: {provider_label}",
                        f"Model: {config.model}",
                        f"Tool chains: {chains}",
                        f"Verbose diagnostics: {'yes' if config.verbose else 'no'}",
                        f"Task: {config.user_prompt}",
                        "System prompt:",
                        config.system_prompt,
                    ]
                ),
                title="Run configuration",
            )
        )


def main() -> None:
    """Run the interactive agent CLI."""
    try:
        InteractiveCLI().run()
    except KeyboardInterrupt:
        Console().print("\nGoodbye.")
