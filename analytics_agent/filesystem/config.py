"""TOML configuration loading for named data locations."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path

from analytics_agent.filesystem.locations import DataLocation, LocationCatalog

LOCATIONS_ENV_VAR = "ANALYTICS_AGENT_LOCATIONS"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCATIONS_PATH = PROJECT_ROOT / "locations.toml"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data"
_ALLOWED_LOCATION_KEYS = frozenset({"backend", "name", "uri"})


def load_locations(
    config_path: str | Path | None = None,
    *,
    data_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[DataLocation, ...]:
    """Load locations from TOML or return the zero-config local fallback."""
    environment = os.environ if environ is None else environ
    selected_path, required = _select_config_path(
        config_path,
        environment,
    )
    if not selected_path.is_file():
        if required:
            raise FileNotFoundError(
                f"Locations configuration was not found: {selected_path}"
            )
        root = Path(data_path or DEFAULT_DATA_PATH).expanduser().resolve()
        return (DataLocation(name="local", uri=root.as_uri(), backend="local"),)

    try:
        with selected_path.open("rb") as config_file:
            document = tomllib.load(config_file)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid locations TOML in '{selected_path}': {exc}") from exc
    return _parse_locations(document, selected_path)


def load_location_catalog(
    config_path: str | Path | None = None,
    *,
    data_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> LocationCatalog:
    """Build a location catalog from configuration and environment."""
    environment = os.environ if environ is None else environ
    locations = load_locations(
        config_path,
        data_path=data_path,
        environ=environment,
    )

    from analytics_agent.filesystem.backends import create_filesystem

    return LocationCatalog(
        locations,
        filesystem_factory=lambda location: create_filesystem(
            location,
            environ=environment,
        ),
    )


def _select_config_path(
    config_path: str | Path | None,
    environ: Mapping[str, str],
) -> tuple[Path, bool]:
    if config_path is not None:
        return Path(config_path).expanduser().resolve(), True
    override = environ.get(LOCATIONS_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser().resolve(), True
    return DEFAULT_LOCATIONS_PATH, False


def _parse_locations(
    document: dict[str, object],
    source_path: Path,
) -> tuple[DataLocation, ...]:
    raw_locations = document.get("locations")
    if isinstance(raw_locations, dict):
        records = [
            _location_from_mapping(name, value, source_path)
            for name, value in raw_locations.items()
        ]
    elif isinstance(raw_locations, list):
        records = [
            _location_from_array_entry(value, index, source_path)
            for index, value in enumerate(raw_locations, start=1)
        ]
    else:
        raise ValueError(
            f"'{source_path}' must define [locations.<name>] tables or "
            "[[locations]] entries."
        )
    if not records:
        raise ValueError(f"'{source_path}' does not define any data locations.")
    names = [record.name for record in records]
    if len(names) != len(set(names)):
        raise ValueError(f"'{source_path}' defines duplicate data-location names.")
    return tuple(records)


def _location_from_mapping(
    name: object,
    value: object,
    source_path: Path,
) -> DataLocation:
    if not isinstance(name, str) or not isinstance(value, dict):
        raise ValueError(f"Invalid location table in '{source_path}'.")
    settings = _validate_location_settings(value, source_path)
    return _build_location(name, settings, source_path)


def _location_from_array_entry(
    value: object,
    index: int,
    source_path: Path,
) -> DataLocation:
    if not isinstance(value, dict):
        raise ValueError(
            f"Location entry {index} in '{source_path}' must be a TOML table."
        )
    settings = _validate_location_settings(value, source_path)
    name = settings.get("name")
    if not isinstance(name, str):
        raise ValueError(
            f"Location entry {index} in '{source_path}' requires a string name."
        )
    return _build_location(name, settings, source_path)


def _validate_location_settings(
    value: dict[object, object],
    source_path: Path,
) -> dict[str, object]:
    settings = {str(key): item for key, item in value.items()}
    unexpected = sorted(set(settings) - _ALLOWED_LOCATION_KEYS)
    if unexpected:
        names = ", ".join(unexpected)
        raise ValueError(
            f"Unsupported keys in '{source_path}': {names}. Credentials must be "
            "provided through Azure environment variables."
        )
    return settings


def _build_location(
    name: str,
    settings: dict[str, object],
    source_path: Path,
) -> DataLocation:
    uri = settings.get("uri")
    backend = settings.get("backend")
    if not isinstance(uri, str):
        raise ValueError(f"Location {name!r} in '{source_path}' requires a string URI.")
    if backend is not None and not isinstance(backend, str):
        raise ValueError(
            f"Location {name!r} in '{source_path}' has a non-string backend."
        )
    if "://" not in uri and (backend is None or backend.lower() in {"file", "local"}):
        uri = os.fspath((source_path.parent / Path(uri)).resolve())
    return DataLocation(name=name, uri=uri, backend=backend)
