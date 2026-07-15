"""Dataset catalog models and CSV loading."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATASET_METADATA = {
    "saas_docs.csv": (
        "SaaS Docs",
        "Product and API documentation for SaaS platform features and limits.",
    ),
    "credit_card_terms.csv": (
        "Credit Card Terms",
        "Credit card terms, APR details, fees, and account policies.",
    ),
    "hospital_policy.csv": (
        "Hospital Policy",
        "Hospital operations, compliance rules, and patient care policies.",
    ),
    "ecommerce_faqs.csv": (
        "Ecommerce FAQs",
        "Customer-facing ecommerce questions covering shipping, returns, and support.",
    ),
}


@dataclass(frozen=True)
class DatasetSpec:
    """User-provided dataframe and its descriptive metadata."""

    name: str
    dataframe: pd.DataFrame
    description: str = ""
    source_path: str | None = None


@dataclass(frozen=True)
class DatasetEntry:
    """Validated dataset stored in a dataframe catalog."""

    name: str
    dataframe: pd.DataFrame
    description: str
    source_path: str | None
    id_column: str | None

    @property
    def source_name(self) -> str:
        """Source file name without its parent path."""
        if not self.source_path:
            return ""
        return Path(self.source_path).name


class DataframeCatalog:
    """Collection of validated dataframes addressable by name."""

    def __init__(self, entries: list[DatasetEntry]) -> None:
        self._entries = entries
        self._by_name = {entry.name: entry for entry in entries}

    @classmethod
    def from_specs(cls, specs: list[DatasetSpec]) -> "DataframeCatalog":
        """Build a catalog from dataset specifications."""
        entries: list[DatasetEntry] = []
        seen_names: set[str] = set()
        for spec in specs:
            normalized_name = spec.name.strip()
            if not normalized_name:
                raise ValueError("Dataset names must be non-empty.")
            if normalized_name in seen_names:
                raise ValueError(f"Duplicate dataset name: {normalized_name}")
            if not isinstance(spec.dataframe, pd.DataFrame):
                raise ValueError(
                    f"Dataset '{normalized_name}' must be a pandas DataFrame."
                )

            seen_names.add(normalized_name)
            entries.append(
                DatasetEntry(
                    name=normalized_name,
                    dataframe=spec.dataframe,
                    description=spec.description.strip(),
                    source_path=spec.source_path,
                    id_column=infer_id_column(spec.dataframe),
                )
            )
        return cls(entries)

    def all(self) -> list[DatasetEntry]:
        """Return all entries in insertion order."""
        return list(self._entries)

    def get(self, dataset_name: str) -> DatasetEntry:
        """Return a dataset by name or raise an informative error."""
        try:
            return self._by_name[dataset_name]
        except KeyError as exc:
            available = ", ".join(self.names())
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Available datasets: {available}."
            ) from exc

    def names(self) -> list[str]:
        """Return dataset names in insertion order."""
        return [entry.name for entry in self._entries]


def infer_id_column(dataframe: pd.DataFrame) -> str | None:
    """Infer an identifier column, preferring complete unique candidates."""
    candidates = [
        column for column in dataframe.columns if "id" in str(column).strip().lower()
    ]
    for column in candidates:
        series = dataframe[column]
        if series.notna().all() and series.is_unique:
            return str(column)
    return str(candidates[0]) if candidates else None


def load_dataset_specs(data_path: str) -> list[DatasetSpec]:
    """Load all CSV files in a directory as dataset specifications."""
    directory = Path(data_path)
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise ValueError(f"No CSV files found in {data_path}.")

    specs: list[DatasetSpec] = []
    for file_path in files:
        try:
            dataframe = pd.read_csv(file_path)
        except Exception as exc:
            raise ValueError(
                f"Failed to load '{file_path}' as a dataframe: {exc}"
            ) from exc

        dataset_name, description = DATASET_METADATA.get(
            file_path.name,
            (file_path.stem, f"Dataset loaded from {file_path.name}."),
        )
        specs.append(
            DatasetSpec(
                name=dataset_name,
                dataframe=dataframe,
                description=description,
                source_path=str(file_path),
            )
        )

    return specs
