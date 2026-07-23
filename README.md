# Analytics Agent

Analytics Agent is an experimental Python CLI for running OpenAI Responses API
tool loops over local CSV data or a simulated server-incident environment.

## Status and capabilities

The repository currently provides:

- Seven dataframe tools for discovering, describing, searching, filtering, and
  aggregating locally supplied CSV files.
- Four deterministic incident-response tools with simulated health, log, restart,
  and escalation results.
- Canonical typed chat messages with an OpenAI Responses API adapter.
- Fixed sample entry points and an interactive flow for selecting an
  account-available model, tool chains, prompts, and verbose diagnostics.

This is a learning and development project, not a production incident-response
system. It has no persistence, authentication layer, per-tool approval workflow,
retrieval index, or automated model-quality evaluation.

## Quickstart

Prerequisites:

- Python 3.14 or newer
- An OpenAI API key
- [`uv`](https://docs.astral.sh/uv/) for the preferred installation path

From the repository root, install the application and development dependencies:

```sh
uv sync
```

Create a local `.env` file:

```sh
printf 'OPENAI_API_KEY=your-key-here\n' > .env
```

The incident scenario requires no local data, so it is the shortest working run:

```sh
uv run run_incident_agent
```

It uses `gpt-4o-mini` to investigate `payment-server-01`. Tool calls and results
are printed as the run progresses, followed by the model's final response.

To install without `uv`, use a Python 3.14+ virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Run installed commands directly when the virtual environment is active.

## Local CSV setup

CSV data is intentionally local and is not tracked by Git. Create
`analytics_agent/data/` and place one or more `.csv` files directly inside it:

```sh
mkdir -p analytics_agent/data
cp /path/to/your-dataset.csv analytics_agent/data/
uv run run_dataframe_agent
```

The loader is non-recursive and reads every `*.csv` file in filename order. Any
CSV filename is accepted. These filenames receive friendlier built-in names and
descriptions:

| Filename | Dataset name |
| --- | --- |
| `saas_docs.csv` | `SaaS Docs` |
| `credit_card_terms.csv` | `Credit Card Terms` |
| `hospital_policy.csv` | `Hospital Policy` |
| `ecommerce_faqs.csv` | `Ecommerce FAQs` |

Other files use their filename stem as the dataset name. The fixed dataframe
entry point asks, “What are the visiting hours in the hospital?”; use the
interactive command for a task tailored to different data.

Only CSV files are loaded. Parquet, JSON, spreadsheets, nested directories, and
remote data sources are not supported.

## Interactive usage

Start the configuration flow with:

```sh
uv run run_interactive_agent
```

The flow can:

1. List model IDs visible to the configured OpenAI account.
2. Select the dataframe chain, incident-response chain, or both.
3. Accept generated system and user prompts or collect replacements.
4. Enable raw provider-response and serialized-history diagnostics.
5. Show a summary and require confirmation before making the model request.

The `View available providers, models, and tool chains` menu is read-only. Model
listing still requires `OPENAI_API_KEY` and network access. Configurations are not
saved between runs.

## Tool behavior

The dataframe chain exposes:

- `list_dataframes`
- `describe_dataframe`
- `preview_dataframe`
- `search_rows`
- `filter_rows`
- `aggregate_rows`
- `distinct_values`

Tool inputs are validated before execution. Preview results are limited to 20
rows, general query results to 50 rows, and distinct-value results to 100 rows.
The catalog infers an ID column from columns whose names contain `id`, preferring
a complete unique column.

The incident-response chain exposes:

- `get_server_health`
- `fetch_recent_logs`
- `restart_service`
- `escalate_incident`

All four operate on in-memory fixtures. In particular, restart and escalation only
return simulated success JSON; they do not contact servers, restart processes,
page responders, or change external state.

## Architecture and workflow

The entry points are composition roots: they load configuration, assemble tool
definitions and schemas, create the provider, and start the shared loop. Domain
tools remain provider-neutral; the OpenAI schema factory adapts their Pydantic
input contracts at the provider boundary.

For each model turn:

1. `OpenAIProvider` serializes canonical history and calls the Responses API.
2. The provider adapter normalizes response messages and function calls.
3. The shared loop dispatches each call through `ToolRegistry`.
4. Pydantic rejects missing, extra, or invalid tool arguments.
5. Tool results are appended to history and supplied on the next model turn.
6. The loop prints a final answer when no function calls remain.

Malformed JSON tool arguments are returned to the model as readable tool errors,
allowing a later turn to correct them. The loop stops after 10 model turns if no
final response is produced.

## Configuration reference

| Setting | Default | Notes |
| --- | --- | --- |
| `OPENAI_API_KEY` | none | Required for model listing and agent runs; loaded from the environment or `.env`. |
| Static entry-point model | `gpt-4o-mini` | Used by `run_dataframe_agent` and `run_incident_agent`. |
| Interactive model | none | Selected from IDs returned for the configured account. |
| CSV directory | `analytics_agent/data/` | Local, Git-ignored, and required only for the dataframe chain. |
| Maximum model turns | `10` | The loop stops without a final response after this limit. |
| Verbose diagnostics | off | Available through the interactive flow. |

The interactive model list is account-scoped but is not filtered to models that
support the Responses API or function tools. Selecting an incompatible model
causes the provider request to fail with an API error.

## Data handling and limitations

- CSV files are read locally with pandas when the dataframe chain is assembled.
- The model receives the user prompt, tool schemas, and tool results. A tool result
  can contain values from local CSV rows, so do not use sensitive data unless its
  transmission to OpenAI is acceptable.
- CSV contents are not indexed or automatically uploaded in full by this code.
- Responses are grounded in tool output only to the extent that the selected model
  follows the prompt and calls the appropriate tools.
- Tool output is printed to the terminal and retained only in in-memory
  conversation history for the current process.
- Provider and CSV-loading errors are reported, but there is no retry, rate-limit
  backoff, checkpointing, or recovery across process restarts.
- Verbose mode can print model responses and conversation history. Avoid it when
  terminal output may be retained in an insecure location.

## Testing and quality checks

The unit suite uses in-memory data and test doubles; it does not make live OpenAI
requests:

```sh
uv run python -m unittest discover -s tests
```

Run lint and formatting checks with:

```sh
uv run ruff check .
uv run ruff format --check .
```

The tests cover message normalization, provider history, configuration
validation, tool composition and validation, dataframe operations, incident
fixtures, and loop dispatch. There are no model-quality benchmarks or end-to-end
live API tests.

## Project structure

```text
analytics_agent/
├── agent_runtime.py          # Validated interactive run configuration
├── dataframe_main.py         # Fixed dataframe sample entry point
├── incident_response_main.py # Fixed simulated-incident entry point
├── interactive_cli.py        # Interactive terminal configuration
├── messages/                 # Canonical message models and OpenAI adapters
├── providers/                # Provider boundary and OpenAI implementation
└── tools/
    ├── dataframe/            # CSV catalog, contracts, and dataframe operations
    ├── incident_response/    # Simulated incident contracts and operations
    ├── registry.py           # Validated provider-neutral dispatch
    ├── provider_factories.py # OpenAI tool-schema adapter
    ├── tool_chains.py        # Selectable tool-chain composition
    └── tool_loop.py          # Shared model/tool orchestration
tests/                        # Offline unit tests
```
