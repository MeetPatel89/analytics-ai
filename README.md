# Analytics Agent

Analytics Agent is an experimental Python CLI for running OpenAI Responses API
tool loops over local CSV data, a local sales Parquet dataset, or a simulated
server-incident environment.

## Status and capabilities

The repository currently provides:

- Seven dataframe tools for discovering, describing, searching, filtering, and
  aggregating locally supplied CSV files.
- Four deterministic incident-response tools with simulated health, log, restart,
  and escalation results.
- Three SQL analyzer tools for querying a local sales Parquet file with DuckDB,
  analyzing bounded results, and generating reviewable pandas/Matplotlib code.
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
uv run incident_agent
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
uv run dataframe_agent
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

The dataframe chain loads only CSV files. JSON, spreadsheets, nested directories,
and remote data sources are not supported by that chain.

## Local sales Parquet setup

The SQL analyzer expects this exact Git-ignored local filename:

```text
analytics_agent/data/Store_Sales_Price_Elasticity_Promotions_Data.parquet
```

Create the directory and copy the sales dataset into place before selecting the
SQL analyzer chain:

```sh
mkdir -p analytics_agent/data
cp /path/to/Store_Sales_Price_Elasticity_Promotions_Data.parquet \
  analytics_agent/data/Store_Sales_Price_Elasticity_Promotions_Data.parquet
uv run agent
```

DuckDB materializes the Parquet data into a private in-memory `sales` table and
then disables external access. Generated SQL must parse as exactly one `SELECT`
statement. Results are limited to 50 rows and include a `truncated` flag.

## Interactive usage

Start the configuration flow with:

```sh
uv run agent
```

The flow can:

1. Select a registered provider.
2. Fetch and select from the model IDs currently visible to that provider account.
3. Select any combination of the dataframe, incident-response, and SQL analyzer
   chains.
4. Accept generated system and user prompts or collect replacements.
5. Enable raw provider-response and serialized-history diagnostics.
6. Show a summary and require confirmation before making the model request.

The `View available providers, models, and tool chains` menu is read-only. Model
listing still requires the selected provider's credential and network access.
OpenAI currently uses `OPENAI_API_KEY`. Model lists are fetched fresh and are not
cached between runs; configurations are not saved.

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

The SQL analyzer chain exposes:

- `lookup_sales_data`
- `analyze_sales_data`
- `generate_visualization`

`lookup_sales_data` asks the selected model for a DuckDB query, validates it, and
returns a JSON envelope containing the SQL, ordered columns, normalized positional
row arrays, returned row count, and truncation status. Each row value aligns with
the column at the same index. `analyze_sales_data` accepts that complete envelope
and generates a grounded text analysis. `generate_visualization` first selects
validated line, bar, or scatter axes, then returns self-contained pandas/Matplotlib
source with the result rows embedded.

Visualization code is syntax-checked but never executed by the application.
Review generated code before running it.

## Architecture and workflow

The entry points are composition roots: they load configuration, assemble tool
definitions and schemas, create the provider, and start the shared loop. The
interactive runtime uses a provider registry for display metadata, credential
lookup, model discovery, and provider construction. Domain tools remain
provider-neutral; the OpenAI schema factory adapts their Pydantic input contracts
at the provider boundary. Registered providers expose a provider-neutral
text/structured-generation capability that tool chains can request lazily. The
SQL analyzer uses that capability with the credential and model selected for the
outer agent run.

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
| Static entry-point model | `gpt-4o-mini` | Used by `dataframe_agent` and `incident_agent`. |
| Interactive provider | `openai` only | Explicitly selected from the provider registry. |
| Interactive model | none | Fetched after provider selection and selected from IDs returned for the configured account. |
| CSV directory | `analytics_agent/data/` | Local, Git-ignored, and required only for the dataframe chain. |
| Sales Parquet file | `analytics_agent/data/Store_Sales_Price_Elasticity_Promotions_Data.parquet` | Local, Git-ignored, and required only for the SQL analyzer chain. |
| Maximum model turns | `10` | The loop stops without a final response after this limit. |
| Verbose diagnostics | off | Available through the interactive flow. |

The interactive model list is account-scoped but is not filtered to models that
support the Responses API or function tools. Selecting an incompatible model
causes the provider request to fail with an API error.

## Data handling and limitations

- CSV files are read locally with pandas when the dataframe chain is assembled.
  The sales Parquet file is read locally by DuckDB only when sales lookup runs.
- The outer model receives the user prompt, tool schemas, and tool results. SQL
  analyzer generation calls also receive the sales schema, the analysis or
  visualization goal, and relevant bounded query-result rows. Do not use
  sensitive local data unless transmitting those values to the selected provider
  is acceptable.
- CSV and Parquet files are not automatically uploaded in full. A generated sales
  query can nevertheless select up to 50 rows, and those rows are returned to the
  outer model. Analysis and visualization calls send that result envelope to the
  generation model as well.
- SQL analyzer workflows make additional provider requests with the same selected
  model: one for SQL generation, one for analysis, and two for visualization
  configuration and source generation. These calls add latency and may add usage
  charges under the provider account.
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
fixtures, bounded DuckDB queries, SQL restrictions, visualization source
validation, and loop dispatch. There are no model-quality benchmarks or
end-to-end live API tests.

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
    ├── sql_analyzer/         # DuckDB sales lookup and model-assisted analysis
    ├── registry.py           # Validated provider-neutral dispatch
    ├── provider_factories.py # OpenAI tool-schema adapter
    ├── tool_chains.py        # Selectable tool-chain composition
    └── tool_loop.py          # Shared model/tool orchestration
tests/                        # Offline unit tests
```
