# Filesystem Analytics Agent

Filesystem Analytics Agent is a Python CLI that lets an OpenAI tool-calling agent
navigate, inspect, and query data in named filesystem roots. Phase 1 supports local
filesystems and Azure Data Lake Storage Gen2 (ADLS Gen2) through one read-only
`fsspec` abstraction.

The agent can:

- Discover configured locations and browse directories or globs.
- Inspect metadata and CSV or Parquet schemas.
- Preview at most 20 tabular rows.
- Read bounded ranges from txt, log, and JSON files.
- Run single-statement DuckDB `SELECT` queries over one or more CSV or Parquet
  files/globs, returning at most 50 rows.

## Quickstart

Prerequisites:

- Python 3.14 or newer
- [`uv`](https://docs.astral.sh/uv/)
- An OpenAI API key

Install the application and test dependencies:

```sh
uv sync --extra dev
```

Add the model-provider credential to `.env`:

```dotenv
OPENAI_API_KEY=your-key-here
```

For a zero-configuration local run, create the default project-root `data/`
directory and add CSV, Parquet, txt, log, or JSON files:

```sh
mkdir -p data
cp /path/to/example.parquet data/
uv run agent
```

The CLI lets you select an account-available model, edit the generated prompts,
inspect the run summary, and confirm before the first model request. Every run has
the same bounded filesystem analytics tools; there is no unrelated tool-set
selection step.

To use a different local root without creating `locations.toml`, pass:

```sh
uv run agent --data-path /srv/analytics
```

## Named locations

Create `locations.toml` in the project root to expose more than one root. Locations
can be local or ADLS Gen2:

```toml
[locations.local]
uri = "file:///srv/analytics"
backend = "local"

[locations.lake]
uri = "abfs://curated@myaccount.dfs.core.windows.net/sales"
backend = "adls"
```

`backend` may be omitted when the URI scheme identifies it. Local filesystem paths
without a scheme are also accepted, although absolute `file://` URIs make the root
unambiguous.

To keep configuration elsewhere, set:

```sh
export ANALYTICS_AGENT_LOCATIONS=/etc/analytics-agent/locations.toml
```

An explicit override must exist and contain either `[locations.<name>]` tables, as
above, or `[[locations]]` entries:

```toml
[[locations]]
name = "local"
uri = "file:///srv/analytics"

[[locations]]
name = "lake"
uri = "abfs://curated@myaccount.dfs.core.windows.net/sales"
```

Credentials are rejected in `locations.toml`. Keep them in the environment or
`.env`.

## ADLS Gen2 authentication

The ADLS backend resolves credentials in this order:

1. `AZURE_STORAGE_ACCOUNT_KEY`
2. `AZURE_STORAGE_SAS_TOKEN`
3. Azure `DefaultAzureCredential`

Include the account in the URI, as in
`container@account.dfs.core.windows.net`, or set
`AZURE_STORAGE_ACCOUNT_NAME`. `DefaultAzureCredential` covers common developer
and deployed identities, including Azure CLI login, service-principal environment
variables, and managed identity.

Example with an account key:

```dotenv
AZURE_STORAGE_ACCOUNT_NAME=myaccount
AZURE_STORAGE_ACCOUNT_KEY=your-account-key
```

Example with a SAS token:

```dotenv
AZURE_STORAGE_ACCOUNT_NAME=myaccount
AZURE_STORAGE_SAS_TOKEN=your-sas-token
```

Do not put secrets or SAS query strings in location URIs.

## Analytics tools

| Tool | Purpose | Phase 1 bounds |
| --- | --- | --- |
| `list_locations` | List named roots, URIs, and backend kinds. | Configured roots only |
| `list_directory` | List a relative directory or glob with type, size, and modification metadata. | At most 200 entries and 64 KiB output |
| `get_file_info` | Inspect one relative path and detect its format by extension. | One entry |
| `inspect_schema` | Return Arrow column names and types for CSV or Parquet. | At most 500 columns and 64 KiB output |
| `preview_data` | Return leading CSV or Parquet rows. | At most 20 rows and 64 KiB output |
| `read_text_file` | Read UTF-8 from txt, log, or JSON at a byte offset. | At most 32 KiB read and 64 KiB output |
| `query_data` | Run DuckDB SQL over safe source aliases. | One `SELECT`, 10 sources, 100 matched files, 50 rows, and 64 KiB output |

`query_data` receives a list of source declarations. Each declaration gives a
temporary SQL view alias, a location name, and a relative file or glob. For
example, a tool call can expose `2026/*.parquet` as `sales`, then run:

```sql
SELECT region, SUM(revenue) AS revenue
FROM sales
GROUP BY region
ORDER BY revenue DESC
```

CSV and Parquet are supported by schema, preview, and query operations. Txt, log,
and JSON are supported by bounded text reads; JSON is not a tabular query format
in Phase 1.

## Safety boundary

Phase 1 is read-only:

- Every tool path is resolved through `LocationCatalog`. Absolute paths,
  backslashes, URI paths, encoded traversal, `..` segments, and local symlinks
  escaping a configured root are rejected.
- The `fsspec` wrapper hides mutation APIs and accepts only binary-read mode.
- DuckDB sees only catalog-resolved source views and configured roots.
- User SQL must parse as exactly one `SELECT` statement.
- After source registration, DuckDB external access is disabled except for
  explicitly allowed configured roots, and its configuration is locked.
- Directory, source-file, byte, column, and row limits bound results sent back to
  the model.

These controls reduce accidental access and mutation; they do not make the CLI a
multi-tenant sandbox. Run it under an operating-system identity that already has
the minimum required read permissions.

Tool outputs, prompts, and selected result rows are sent to the configured model
provider. Do not expose data that may not be transmitted to that provider.
Verbose diagnostics and trace exports can also contain prompts and tool arguments.

## Architecture

```text
interactive_cli
└── agent_runtime + shared tool loop
    └── filesystem_analytics tools
        ├── navigation tools
        ├── Arrow-backed schema and preview tools
        └── guarded DuckDB query tool

filesystem_analytics
└── LocationCatalog
    └── ReadOnlyFileSystem (fsspec)
        ├── LocalFileSystem
        └── AzureBlobFileSystem (adlfs / ADLS Gen2)
```

The shared tool registry validates every call with Pydantic models configured to
reject extra fields. Provider adapters translate those provider-neutral
definitions into strict OpenAI function schemas. Malformed calls return readable
tool errors to the model so a later turn can correct them.

Each CLI run creates an OpenTelemetry `agent_run` span with child spans for model
turns, provider generation, and tool execution. Traces use the console exporter
by default. Set `OTEL_TRACES_EXPORTER=otlp` to send them through OTLP over
HTTP/protobuf; the exporter reads the standard
`OTEL_EXPORTER_OTLP_ENDPOINT` and `OTEL_EXPORTER_OTLP_HEADERS` settings. OTLP
spans are batched during a run and flushed when the CLI exits. Set
`OTEL_TRACES_EXPORTER=none` to disable export.

## Configuration reference

| Setting | Default | Notes |
| --- | --- | --- |
| `OPENAI_API_KEY` | none | Required for model discovery and agent runs. |
| `ANALYTICS_AGENT_LOCATIONS` | project `locations.toml` | Optional alternate TOML path. |
| `agent --data-path PATH` | project `data/` | Alternate zero-config local root; ignored when locations TOML is present. |
| Local data root without TOML | project `data/` | Created by the user; never written by the agent. |
| `AZURE_STORAGE_ACCOUNT_NAME` | account parsed from URI | Required if an ADLS URI omits its account host. |
| `AZURE_STORAGE_ACCOUNT_KEY` | none | Highest-priority ADLS credential. |
| `AZURE_STORAGE_SAS_TOKEN` | none | Used only when an account key is absent. |
| Azure identity variables/login | environment-dependent | Used by `DefaultAzureCredential` as the fallback. |
| Maximum model turns | `10` | Stops a tool loop that never produces a final answer. |
| `OTEL_TRACES_EXPORTER` | `console` | Trace exporter: `console`, `otlp` (HTTP/protobuf), or `none`. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | SDK default | OTLP base endpoint; used when the trace exporter is `otlp`. |
| `OTEL_EXPORTER_OTLP_HEADERS` | none | Comma-separated OTLP request headers, such as backend authorization. |

## Testing and quality checks

The suite is fully offline. It uses temporary local directories and fsspec's
`MemoryFileSystem` as a remote backend, and mocks ADLS construction:

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Coverage includes location configuration, traversal and symlink rejection,
read-only enforcement, Azure authentication priority, local/remote navigation,
CSV and Parquet inspection, bounded text reads, DuckDB query caps, external-path
blocking, tool schemas, runtime composition, provider adapters, message handling,
the shared loop, and tracing.

## Project structure

```text
analytics_agent/
├── agent_runtime.py
├── filesystem/
│   ├── backends.py           # Local/ADLS factories and read-only wrapper
│   ├── config.py             # locations.toml and zero-config fallback
│   └── locations.py          # Named roots and containment
├── interactive_cli.py
├── messages/
├── observability/
├── providers/
└── tools/
    ├── filesystem_analytics/ # Navigation, inspection, and DuckDB query tools
    ├── provider_factories.py
    ├── registry.py
    └── tool_loop.py
tests/
```

## Phase 2 roadmap

The following are future work, not Phase 1 features:

- Delta Lake readers through `deltalake`
- Iceberg readers through `pyiceberg`
- S3 backend configuration
- Dataset profiling such as null counts and distributions
- Partition-aware dataset discovery
- Explicit scratch locations for saving results

The storage catalog and tool boundaries are designed so new format handlers and
fsspec backends can be added without changing the shared agent runtime.
