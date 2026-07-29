"""Named filesystem locations and path-containment enforcement."""

from __future__ import annotations

import os
import posixpath
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from urllib.parse import SplitResult, unquote, urlsplit
from urllib.request import url2pathname

from fsspec.spec import AbstractFileSystem

_LOCATION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")
_GLOB_MAGIC = re.compile(r"[*?[]")


@dataclass(frozen=True, slots=True)
class DataLocation:
    """A named root URI backed by one filesystem implementation."""

    name: str
    uri: str
    backend: str | None = None

    def __post_init__(self) -> None:
        """Normalize and validate location metadata."""
        name = self.name.strip()
        uri = self.uri.strip()
        if not _LOCATION_NAME.fullmatch(name):
            raise ValueError(
                "Location names must contain 1-64 letters, numbers, dots, "
                "underscores, or hyphens."
            )
        if not uri:
            raise ValueError("A data location URI cannot be empty.")
        if len(uri) > 8192:
            raise ValueError("A data location URI cannot exceed 8192 characters.")

        parsed = urlsplit(uri)
        windows_path = bool(_WINDOWS_DRIVE.match(uri))
        backend = self.backend or (
            "local" if windows_path else _backend_from_scheme(parsed.scheme)
        )
        backend = backend.lower()
        if backend == "file":
            backend = "local"
        elif backend in {"abfs", "abfss", "az", "azure"}:
            backend = "adls"

        if backend == "local":
            if not windows_path and (
                parsed.query or parsed.fragment or parsed.password
            ):
                raise ValueError(
                    "Data location URIs cannot contain credentials, query strings, "
                    "or fragments."
                )
            uri = _normalize_local_uri(uri, parsed.scheme)
        elif backend == "adls":
            _validate_remote_root(parsed)
            if parsed.scheme.lower() not in {"abfs", "abfss", "az"}:
                raise ValueError("ADLS locations require an abfs:// URI.")
            if not parsed.netloc:
                raise ValueError("ADLS locations require a container in the URI.")
        elif backend == "memory":
            _validate_remote_root(parsed)
            if parsed.scheme.lower() != "memory":
                raise ValueError("Memory locations require a memory:// URI.")
        else:
            raise ValueError(f"Unsupported filesystem backend: {backend!r}.")

        object.__setattr__(self, "name", name)
        normalized_uri = uri.rstrip("/") if urlsplit(uri).path not in {"", "/"} else uri
        object.__setattr__(self, "uri", normalized_uri)
        object.__setattr__(self, "backend", backend)

    @property
    def kind(self) -> str:
        """Normalized backend kind."""
        return str(self.backend)


@dataclass(frozen=True, slots=True)
class _CatalogEntry:
    location: DataLocation
    filesystem: AbstractFileSystem
    root_path: str
    local_root: Path | None


FilesystemFactory = Callable[[DataLocation], AbstractFileSystem]


class LocationCatalog:
    """Resolve safe agent-facing paths within named filesystem roots."""

    def __init__(
        self,
        locations: Iterable[DataLocation],
        *,
        filesystem_factory: FilesystemFactory | None = None,
        filesystems: Mapping[str, AbstractFileSystem] | None = None,
    ) -> None:
        if filesystem_factory is None:
            from analytics_agent.filesystem.backends import create_filesystem

            filesystem_factory = create_filesystem

        supplied_filesystems = filesystems or {}
        entries: dict[str, _CatalogEntry] = {}
        for location in locations:
            if location.name in entries:
                raise ValueError(f"Duplicate data location: {location.name!r}.")
            filesystem = supplied_filesystems.get(location.name)
            if filesystem is None:
                filesystem = filesystem_factory(location)
            filesystem = _ensure_read_only(filesystem)
            root_path = filesystem._strip_protocol(location.uri)
            local_root = None
            if location.backend == "local":
                local_root = Path(root_path).resolve()
                root_path = os.fspath(local_root)
            entries[location.name] = _CatalogEntry(
                location=location,
                filesystem=filesystem,
                root_path=root_path.rstrip("/") or root_path,
                local_root=local_root,
            )
        if not entries:
            raise ValueError("At least one data location must be configured.")
        self._entries = MappingProxyType(entries)

    def locations(self) -> tuple[DataLocation, ...]:
        """Return configured locations in declaration order."""
        return tuple(entry.location for entry in self._entries.values())

    def names(self) -> tuple[str, ...]:
        """Return configured location names in declaration order."""
        return tuple(self._entries)

    def get(self, location_name: str) -> DataLocation:
        """Return a location by name."""
        return self._entry(location_name).location

    def filesystem(self, location_name: str) -> AbstractFileSystem:
        """Return a location's read-only filesystem."""
        return self._entry(location_name).filesystem

    def root_path(self, location_name: str) -> str:
        """Return the backend-native configured root path."""
        return self._entry(location_name).root_path

    def resolve(
        self,
        location_name: str,
        relative_path: str = "",
        *,
        allow_glob: bool = False,
    ) -> str:
        """Resolve a relative path while rejecting traversal and root escape."""
        entry = self._entry(location_name)
        safe_path = _safe_relative_path(relative_path, allow_glob=allow_glob)
        if entry.local_root is not None:
            if allow_glob and _has_glob_magic(safe_path):
                return os.fspath(entry.local_root / Path(*safe_path.split("/")))
            resolved = (entry.local_root / Path(*safe_path.split("/"))).resolve()
            if not resolved.is_relative_to(entry.local_root):
                raise ValueError("Path resolves outside the configured location root.")
            return os.fspath(resolved)

        root = entry.root_path.rstrip("/")
        return posixpath.join(root, safe_path) if safe_path else root

    def uri(self, location_name: str, relative_path: str = "") -> str:
        """Return a fully qualified filesystem URI for a safe path."""
        entry = self._entry(location_name)
        resolved = self.resolve(location_name, relative_path)
        return entry.filesystem.unstrip_protocol(resolved)

    def relative_path(self, location_name: str, backend_path: str) -> str:
        """Return an agent-facing path relative to a configured root."""
        entry = self._entry(location_name)
        stripped = entry.filesystem._strip_protocol(backend_path)
        if entry.local_root is not None:
            resolved = Path(stripped).resolve()
            if not resolved.is_relative_to(entry.local_root):
                raise ValueError("Backend returned a path outside the location root.")
            relative = resolved.relative_to(entry.local_root)
            return relative.as_posix() or "."

        root = entry.root_path.rstrip("/")
        candidate = stripped.rstrip("/")
        if candidate == root:
            return "."
        prefix = f"{root}/" if root else ""
        if prefix and not candidate.startswith(prefix):
            raise ValueError("Backend returned a path outside the location root.")
        return candidate[len(prefix) :] if prefix else candidate.lstrip("/")

    def glob(self, location_name: str, pattern: str) -> list[str]:
        """Expand a safe glob and verify every result remains under its root."""
        entry = self._entry(location_name)
        resolved_pattern = self.resolve(
            location_name,
            pattern,
            allow_glob=True,
        )
        matches = entry.filesystem.glob(resolved_pattern)
        paths = list(matches) if not isinstance(matches, dict) else list(matches)
        for path in paths:
            self.relative_path(location_name, path)
        return sorted(paths)

    def _entry(self, location_name: str) -> _CatalogEntry:
        name = location_name.strip()
        try:
            return self._entries[name]
        except KeyError as exc:
            available = ", ".join(self._entries)
            raise ValueError(
                f"Unknown data location {name!r}. Available locations: {available}."
            ) from exc


def _ensure_read_only(filesystem: AbstractFileSystem) -> AbstractFileSystem:
    from analytics_agent.filesystem.backends import ReadOnlyFileSystem

    if isinstance(filesystem, ReadOnlyFileSystem):
        return filesystem
    return ReadOnlyFileSystem(filesystem)


def _backend_from_scheme(scheme: str) -> str:
    normalized = scheme.lower()
    if normalized in {"", "file", "local"}:
        return "local"
    if normalized in {"abfs", "abfss", "az"}:
        return "adls"
    if normalized == "memory":
        return "memory"
    raise ValueError(f"Unsupported filesystem URI scheme: {scheme!r}.")


def _normalize_local_uri(uri: str, scheme: str) -> str:
    if _WINDOWS_DRIVE.match(uri):
        return Path(uri).expanduser().resolve().as_uri()
    if scheme.lower() not in {"", "file", "local"}:
        raise ValueError("Local locations require a file:// URI or filesystem path.")
    if scheme.lower() == "file":
        parsed = urlsplit(uri)
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError("Remote hosts are not supported in file:// locations.")
        path = Path(url2pathname(unquote(parsed.path)))
    elif scheme.lower() == "local":
        path = Path(unquote(urlsplit(uri).path))
    else:
        path = Path(uri)
    return path.expanduser().resolve().as_uri()


def _validate_remote_root(parsed: SplitResult) -> None:
    if parsed.query or parsed.fragment or parsed.password:
        raise ValueError(
            "Data location URIs cannot contain credentials, query strings, "
            "or fragments."
        )
    remote_path = _decode_path(parsed.path)
    if "\\" in remote_path or any(part == ".." for part in remote_path.split("/")):
        raise ValueError("Data location URI roots cannot contain traversal.")


def _safe_relative_path(path: str, *, allow_glob: bool) -> str:
    decoded = _decode_path(path.strip())
    if "\x00" in decoded:
        raise ValueError("Paths cannot contain null bytes.")
    if "\\" in decoded:
        raise ValueError("Use forward slashes in agent-facing paths.")
    parsed = urlsplit(decoded)
    if parsed.scheme or parsed.netloc or decoded.startswith("/"):
        raise ValueError("Paths must be relative to a configured location.")
    if _WINDOWS_DRIVE.match(decoded):
        raise ValueError("Paths must not contain a Windows drive prefix.")
    if any(part == ".." for part in decoded.split("/")):
        raise ValueError("Path traversal with '..' is not allowed.")
    if not allow_glob and _has_glob_magic(decoded):
        raise ValueError("Glob patterns are not allowed for this operation.")

    normalized = posixpath.normpath(decoded or ".")
    if normalized in {"", "."}:
        return ""
    if normalized == ".." or normalized.startswith("../"):
        raise ValueError("Path resolves outside the configured location root.")
    return normalized


def _has_glob_magic(path: str) -> bool:
    return bool(_GLOB_MAGIC.search(path))


def _decode_path(path: str) -> str:
    decoded = path
    for _ in range(3):
        expanded = unquote(decoded)
        if expanded == decoded:
            break
        decoded = expanded
    return decoded
