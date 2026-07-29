"""Read-only fsspec backends used by filesystem analytics."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import BinaryIO, cast
from urllib.parse import urlsplit

from adlfs import AzureBlobFileSystem
from azure.identity.aio import DefaultAzureCredential
from fsspec.implementations.local import LocalFileSystem
from fsspec.implementations.memory import MemoryFileSystem
from fsspec.spec import AbstractFileSystem

from analytics_agent.filesystem.locations import DataLocation

_WRITE_METHODS = frozenset(
    {
        "_rm",
        "copy",
        "cp",
        "cp_file",
        "delete",
        "download",
        "end_transaction",
        "get",
        "get_file",
        "makedir",
        "makedirs",
        "mkdir",
        "mkdirs",
        "move",
        "mv",
        "pipe",
        "pipe_file",
        "put",
        "put_file",
        "rename",
        "rm",
        "rm_file",
        "rmdir",
        "start_transaction",
        "touch",
        "transaction",
        "upload",
        "write_bytes",
        "write_text",
    }
)


class ReadOnlyFileSystem(AbstractFileSystem):
    """Restrict an fsspec filesystem to non-mutating operations."""

    cachable = False
    protocol = "read-only"

    def __init__(self, filesystem: AbstractFileSystem) -> None:
        super().__init__(skip_instance_cache=True)
        self.__filesystem = filesystem
        self.protocol = filesystem.protocol
        self.storage_options = {}

    def __getattribute__(self, name: str) -> object:
        """Hide fsspec mutation APIs from callers."""
        if name in _WRITE_METHODS:
            raise AttributeError(
                f"{type(self).__name__!s} does not expose the mutating method {name!r}."
            )
        return super().__getattribute__(name)

    def open(
        self,
        path: str,
        mode: str = "rb",
        block_size: int | None = None,
        cache_options: dict[str, object] | None = None,
        compression: str | None = None,
        **kwargs: object,
    ) -> BinaryIO:
        """Open a path for binary reads only."""
        if mode != "rb":
            raise PermissionError("The filesystem is read-only; use mode='rb'.")
        return cast(
            BinaryIO,
            self.__filesystem.open(
                path,
                mode=mode,
                block_size=block_size,
                cache_options=cache_options,
                compression=compression,
                **kwargs,
            ),
        )

    def ls(
        self,
        path: str,
        detail: bool = True,
        **kwargs: object,
    ) -> list[str] | list[dict[str, object]]:
        """List a directory without permitting mutation."""
        return cast(
            list[str] | list[dict[str, object]],
            self.__filesystem.ls(path, detail=detail, **kwargs),
        )

    def info(self, path: str, **kwargs: object) -> dict[str, object]:
        """Return metadata for a path."""
        return cast(dict[str, object], self.__filesystem.info(path, **kwargs))

    def glob(
        self,
        path: str,
        maxdepth: int | None = None,
        **kwargs: object,
    ) -> list[str] | dict[str, dict[str, object]]:
        """Return paths matching a filesystem glob."""
        return cast(
            list[str] | dict[str, dict[str, object]],
            self.__filesystem.glob(path, maxdepth=maxdepth, **kwargs),
        )

    def exists(self, path: str, **kwargs: object) -> bool:
        """Return whether a path exists."""
        return bool(self.__filesystem.exists(path, **kwargs))

    def invalidate_cache(self, path: str | None = None) -> None:
        """Invalidate read caches maintained by the wrapped filesystem."""
        self.__filesystem.invalidate_cache(path)

    def modified(self, path: str) -> object:
        """Return the last-modified value when the backend provides one."""
        return self.__filesystem.modified(path)

    def created(self, path: str) -> object:
        """Return the creation value when the backend provides one."""
        return self.__filesystem.created(path)

    def _strip_protocol(self, path: str) -> str:
        return cast(str, self.__filesystem._strip_protocol(path))

    def unstrip_protocol(self, name: str) -> str:
        """Add the wrapped filesystem's primary protocol to a path."""
        return self.__filesystem.unstrip_protocol(name)


def create_filesystem(
    location: DataLocation,
    *,
    environ: Mapping[str, str] | None = None,
) -> ReadOnlyFileSystem:
    """Create the authenticated, read-only filesystem for a data location."""
    environment = os.environ if environ is None else environ
    if location.backend == "local":
        filesystem: AbstractFileSystem = LocalFileSystem(auto_mkdir=False)
    elif location.backend == "adls":
        filesystem = _create_adls_filesystem(location, environment)
    elif location.backend == "memory":
        filesystem = MemoryFileSystem()
    else:
        raise ValueError(f"Unsupported filesystem backend: {location.backend!r}.")
    return ReadOnlyFileSystem(filesystem)


def build_filesystem(
    location: DataLocation,
    *,
    environ: Mapping[str, str] | None = None,
) -> ReadOnlyFileSystem:
    """Compatibility alias for the filesystem backend factory."""
    return create_filesystem(location, environ=environ)


def _create_adls_filesystem(
    location: DataLocation,
    environ: Mapping[str, str],
) -> AbstractFileSystem:
    account_name = _adls_account_name(location.uri, environ)
    options: dict[str, object] = {
        "account_name": account_name,
        "anon": False,
    }
    account_key = environ.get("AZURE_STORAGE_ACCOUNT_KEY", "").strip()
    sas_token = environ.get("AZURE_STORAGE_SAS_TOKEN", "").strip()
    if account_key:
        options["account_key"] = account_key
    elif sas_token:
        options["sas_token"] = sas_token
    else:
        options["credential"] = DefaultAzureCredential()
    return AzureBlobFileSystem(**options)


def _adls_account_name(
    uri: str,
    environ: Mapping[str, str],
) -> str:
    parsed = urlsplit(uri)
    host = parsed.netloc.rsplit("@", maxsplit=1)[-1]
    account_name = ""
    for suffix in (".dfs.core.windows.net", ".blob.core.windows.net"):
        if host.lower().endswith(suffix):
            account_name = host[: -len(suffix)]
            break
    if not account_name:
        account_name = environ.get("AZURE_STORAGE_ACCOUNT_NAME", "").strip()
    if not account_name:
        raise ValueError(
            "An ADLS location must include an account host in its URI or set "
            "AZURE_STORAGE_ACCOUNT_NAME."
        )
    return account_name
