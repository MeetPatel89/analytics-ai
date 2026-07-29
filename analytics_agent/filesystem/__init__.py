"""Backend-agnostic, read-only filesystem access."""

from analytics_agent.filesystem.backends import (
    ReadOnlyFileSystem,
    build_filesystem,
    create_filesystem,
)
from analytics_agent.filesystem.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_LOCATIONS_PATH,
    LOCATIONS_ENV_VAR,
    load_location_catalog,
    load_locations,
)
from analytics_agent.filesystem.locations import DataLocation, LocationCatalog

__all__ = [
    "DEFAULT_DATA_PATH",
    "DEFAULT_LOCATIONS_PATH",
    "LOCATIONS_ENV_VAR",
    "DataLocation",
    "LocationCatalog",
    "ReadOnlyFileSystem",
    "build_filesystem",
    "create_filesystem",
    "load_location_catalog",
    "load_locations",
]
