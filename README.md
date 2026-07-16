# Analytics Agent

## Setup

Clone the repository, then create a `.env` file containing your OpenAI API key:

```sh
git clone <repository-url>
cd analytics-agent
printf 'OPENAI_API_KEY=your-key-here\n' > .env
```

`uv` is preferred. It creates the virtual environment and installs the application
and development dependencies:

```sh
uv sync
```

Alternatively, use `pip` in a Python 3.14+ virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Run Tool Loops

After setup, run either loop with `uv run` (or activate `.venv` first when using
`pip`):

```sh
uv run run_dataframe_agent
uv run run_incident_agent
```

`run_dataframe_agent` answers questions from the bundled CSV datasets using
dataframe tools. `run_incident_agent` investigates a sample server incident using
health, log, restart, and escalation tools.

For an interactive configuration flow, run:

```sh
uv run run_interactive_agent
```

It lets you select an account-available OpenAI model, one or both tool chains,
prompts, and output detail before confirming the run. Select `View configuration`
to inspect built-in tool chains and account-available models without running one.
