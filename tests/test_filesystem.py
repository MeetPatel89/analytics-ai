"""Tests for filesystem locations, configuration, and backend safety."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fsspec.implementations.memory import MemoryFileSystem

from analytics_agent.filesystem import (
    DataLocation,
    LocationCatalog,
    ReadOnlyFileSystem,
    create_filesystem,
    load_locations,
)


class TestDataLocations:
    """Verify normalization and path containment."""

    def test_filesystem_root_uri_is_preserved(self) -> None:
        """Trailing slash normalization must not corrupt the filesystem root."""
        location = DataLocation("root", "file:///", "local")

        assert location.uri == "file:///"

    def test_local_paths_are_resolved_within_the_named_root(self) -> None:
        """Safe paths should resolve while traversal and absolute paths fail."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = LocationCatalog([DataLocation("local", root.as_uri(), "local")])

            assert catalog.resolve("local", "sales/2026.csv") == str(
                root / "sales" / "2026.csv"
            )
            with pytest.raises(ValueError, match="traversal"):
                catalog.resolve("local", "../secret.csv")
            with pytest.raises(ValueError, match="relative"):
                catalog.resolve("local", "/etc/passwd")
            with pytest.raises(ValueError, match="traversal"):
                catalog.resolve("local", "%2e%2e/secret.csv")
            with pytest.raises(ValueError, match="forward slashes"):
                catalog.resolve("local", "%2e%2e%5csecret.csv")
            with pytest.raises(ValueError, match="traversal"):
                catalog.resolve("local", "%252e%252e/secret.csv")

    def test_location_roots_reject_embedded_credentials_and_traversal(self) -> None:
        """Configured roots must not contain secrets or traversal components."""
        with pytest.raises(ValueError, match="credentials"):
            DataLocation(
                "lake",
                "abfs://container:secret@account.dfs.core.windows.net/data",
                "adls",
            )
        with pytest.raises(ValueError, match="traversal"):
            DataLocation(
                "lake",
                "abfs://container@account.dfs.core.windows.net/../secret",
                "adls",
            )

    def test_local_symlinks_cannot_escape_the_root(self) -> None:
        """A symlink inside a root must not expose a path outside that root."""
        with (
            tempfile.TemporaryDirectory() as directory,
            tempfile.TemporaryDirectory() as outside_directory,
        ):
            root = Path(directory)
            outside = Path(outside_directory)
            (outside / "secret.txt").write_text("secret", encoding="utf-8")
            (root / "escape").symlink_to(outside, target_is_directory=True)
            catalog = LocationCatalog([DataLocation("local", root.as_uri(), "local")])

            with pytest.raises(ValueError, match="outside"):
                catalog.resolve("local", "escape/secret.txt")

    def test_memory_filesystem_proves_backend_agnostic_resolution(self) -> None:
        """Catalog operations should work with an injected remote-like backend."""
        memory = MemoryFileSystem()
        memory.pipe_file("/datasets/example.txt", b"hello")
        catalog = LocationCatalog(
            [DataLocation("remote", "memory://datasets", "memory")],
            filesystems={"remote": memory},
        )

        assert catalog.resolve("remote", "example.txt") == "/datasets/example.txt"
        assert catalog.glob("remote", "*.txt") == ["/datasets/example.txt"]
        assert catalog.relative_path("remote", "/datasets/example.txt") == "example.txt"


class TestReadOnlyFilesystem:
    """Verify the storage wrapper cannot mutate its backend."""

    def test_read_operations_work_and_mutations_are_hidden(self) -> None:
        """Binary reads should work while writes and deletes remain unavailable."""
        memory = MemoryFileSystem()
        memory.pipe_file("/data/example.txt", b"hello")
        filesystem = ReadOnlyFileSystem(memory)

        with filesystem.open("/data/example.txt", "rb") as source:
            assert source.read() == b"hello"
        assert filesystem.exists("/data/example.txt")
        assert not hasattr(filesystem, "rm")
        assert not hasattr(filesystem, "pipe_file")
        with pytest.raises(PermissionError, match="read-only"):
            filesystem.open("/data/new.txt", "wb")


class TestLocationConfiguration:
    """Verify TOML parsing, overrides, and zero-config behavior."""

    def test_zero_config_uses_a_named_local_data_root(self) -> None:
        """An absent default file should create the local fallback location."""
        with tempfile.TemporaryDirectory() as directory:
            locations = load_locations(
                data_path=directory,
                environ={},
            )

        assert locations == (
            DataLocation("local", Path(directory).resolve().as_uri(), "local"),
        )

    def test_toml_supports_named_tables_and_environment_override(self) -> None:
        """The override should load local and ADLS roots without credentials."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "custom-locations.toml"
            config_path.write_text(
                """
[locations.local]
uri = "data"
backend = "local"

[locations.lake]
uri = "abfs://container@account.dfs.core.windows.net/curated"
backend = "adls"
""".strip(),
                encoding="utf-8",
            )

            locations = load_locations(
                environ={"ANALYTICS_AGENT_LOCATIONS": str(config_path)}
            )

        assert [location.name for location in locations] == ["local", "lake"]
        assert locations[0].backend == "local"
        assert locations[0].uri == (root / "data").resolve().as_uri()
        assert locations[1].backend == "adls"

    def test_toml_rejects_credentials_and_missing_explicit_paths(self) -> None:
        """Secrets in configuration and bad override paths should fail clearly."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "locations.toml"
            config_path.write_text(
                """
[locations.lake]
uri = "abfs://container@account.dfs.core.windows.net/data"
account_key = "secret"
""".strip(),
                encoding="utf-8",
            )

            with pytest.raises(ValueError, match="Credentials"):
                load_locations(config_path)
            with pytest.raises(FileNotFoundError, match="not found"):
                load_locations(root / "missing.toml")


class TestADLSBackend:
    """Verify deterministic Azure authentication priority."""

    location = DataLocation(
        "lake",
        "abfs://container@testaccount.dfs.core.windows.net/curated",
        "adls",
    )

    @patch("analytics_agent.filesystem.backends.AzureBlobFileSystem")
    @patch("analytics_agent.filesystem.backends.DefaultAzureCredential")
    def test_account_key_precedes_sas_and_default_credential(
        self,
        credential_factory: Mock,
        azure_filesystem: Mock,
    ) -> None:
        """An explicit account key should win over every lower-priority option."""
        azure_filesystem.return_value = MemoryFileSystem()

        create_filesystem(
            self.location,
            environ={
                "AZURE_STORAGE_ACCOUNT_KEY": "account-key",
                "AZURE_STORAGE_SAS_TOKEN": "sas-token",
            },
        )

        azure_filesystem.assert_called_once_with(
            account_name="testaccount",
            anon=False,
            account_key="account-key",
        )
        credential_factory.assert_not_called()

    @patch("analytics_agent.filesystem.backends.AzureBlobFileSystem")
    @patch("analytics_agent.filesystem.backends.DefaultAzureCredential")
    def test_sas_precedes_default_credential(
        self,
        credential_factory: Mock,
        azure_filesystem: Mock,
    ) -> None:
        """A SAS token should be used when an account key is absent."""
        azure_filesystem.return_value = MemoryFileSystem()

        create_filesystem(
            self.location,
            environ={"AZURE_STORAGE_SAS_TOKEN": "sas-token"},
        )

        azure_filesystem.assert_called_once_with(
            account_name="testaccount",
            anon=False,
            sas_token="sas-token",
        )
        credential_factory.assert_not_called()

    @patch("analytics_agent.filesystem.backends.AzureBlobFileSystem")
    @patch("analytics_agent.filesystem.backends.DefaultAzureCredential")
    def test_default_azure_credential_is_the_fallback(
        self,
        credential_factory: Mock,
        azure_filesystem: Mock,
    ) -> None:
        """Identity-chain authentication should be constructed as the fallback."""
        credential = Mock()
        credential_factory.return_value = credential
        azure_filesystem.return_value = MemoryFileSystem()

        create_filesystem(self.location, environ={})

        credential_factory.assert_called_once_with()
        azure_filesystem.assert_called_once_with(
            account_name="testaccount",
            anon=False,
            credential=credential,
        )
