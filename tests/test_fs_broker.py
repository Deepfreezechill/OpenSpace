"""Tests for EPIC 2.2 — Filesystem Broker.

Issues:
- #89: Virtual path resolution
- #90: Read/write/deny enforcement
- #91: Chroot-style jailing / traversal prevention
- #92: TOCTOU protection (O_NOFOLLOW, post-open verify)
- #93: Property-based tests for jail/deny escapes
- #94: Concurrent symlink-creation race tests
"""

from __future__ import annotations

import os
import platform
import tempfile
import threading
from pathlib import Path

import pytest

from openspace.sandbox.fs_broker import (
    DeniedPathError,
    FileSizeLimitError,
    FilesystemBroker,
    JailConfig,
    PathEscapeError,
    ReadNotAllowedError,
    WriteNotAllowedError,
    bounded_write,
    check_denied,
    check_read,
    check_write,
    ensure_jailed,
    resolve_virtual_path,
    safe_open_read,
    safe_open_write,
)
from openspace.sandbox.leases import FilesystemCapability

_IS_WINDOWS = platform.system() == "Windows"
_SUPPORTS_SYMLINKS = not _IS_WINDOWS  # Conservative; some Windows configs allow them


@pytest.fixture
def jail(tmp_path: Path) -> Path:
    """Create a jail directory with test files."""
    (tmp_path / "allowed.txt").write_text("hello")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested.txt").write_text("nested")
    return tmp_path


@pytest.fixture
def basic_config(jail: Path) -> JailConfig:
    """JailConfig that allows reads everywhere, writes to subdir only."""
    return JailConfig(
        jail_root=jail,
        read_patterns=("**",),
        write_patterns=("subdir/**",),
        denied_patterns=("/etc/shadow", "/etc/passwd", "~/.ssh/*", "**/.env"),
        max_file_size_bytes=1024,
        temp_dir_only=False,
    )


@pytest.fixture
def broker(basic_config: JailConfig) -> FilesystemBroker:
    return FilesystemBroker(basic_config)


# ---------------------------------------------------------------------------
# #89 — Virtual Path Resolution
# ---------------------------------------------------------------------------


class TestVirtualPathResolution:
    def test_resolve_current_namespace(self, jail: Path) -> None:
        result = resolve_virtual_path("skills://current/foo.txt", jail)
        assert result == jail / "foo.txt"

    def test_resolve_nested_path(self, jail: Path) -> None:
        result = resolve_virtual_path("skills://current/subdir/nested.txt", jail)
        assert result == jail / "subdir" / "nested.txt"

    def test_resolve_root_returns_jail(self, jail: Path) -> None:
        result = resolve_virtual_path("skills://current/", jail)
        assert result == jail

    def test_resolve_bare_namespace(self, jail: Path) -> None:
        result = resolve_virtual_path("skills://current", jail)
        assert result == jail

    def test_invalid_scheme_raises(self, jail: Path) -> None:
        with pytest.raises(ValueError, match="Not a virtual path"):
            resolve_virtual_path("/some/real/path", jail)

    def test_different_namespace(self, jail: Path) -> None:
        result = resolve_virtual_path("skills://shared/data.json", jail)
        assert result == jail / "data.json"

    def test_traversal_in_virtual_path_rejected(self, jail: Path) -> None:
        """skills://current/../../../etc/passwd must be rejected."""
        with pytest.raises(ValueError, match="Traversal"):
            resolve_virtual_path("skills://current/../../../etc/passwd", jail)

    def test_dotdot_in_virtual_path_rejected(self, jail: Path) -> None:
        with pytest.raises(ValueError, match="Traversal"):
            resolve_virtual_path("skills://current/subdir/../../escape", jail)


# ---------------------------------------------------------------------------
# #91 — Jail Enforcement
# ---------------------------------------------------------------------------


class TestJailEnforcement:
    def test_path_inside_jail(self, jail: Path) -> None:
        result = ensure_jailed(jail / "allowed.txt", jail)
        assert result == (jail / "allowed.txt").resolve()

    def test_dotdot_traversal_blocked(self, jail: Path) -> None:
        with pytest.raises(PathEscapeError, match="escapes jail"):
            ensure_jailed(jail / ".." / "etc" / "passwd", jail)

    def test_absolute_escape_blocked(self, jail: Path) -> None:
        with pytest.raises(PathEscapeError):
            ensure_jailed(Path("/etc/shadow"), jail)

    @pytest.mark.skipif(_IS_WINDOWS, reason="Symlinks unreliable on Windows")
    def test_symlink_escape_blocked(self, jail: Path) -> None:
        """Symlink pointing outside jail is caught."""
        link = jail / "escape_link"
        link.symlink_to("/tmp")
        with pytest.raises(PathEscapeError, match="escapes jail"):
            ensure_jailed(link, jail)

    @pytest.mark.skipif(_IS_WINDOWS, reason="Symlinks unreliable on Windows")
    def test_nested_symlink_escape_blocked(self, jail: Path) -> None:
        """Deeply nested symlink escape is caught."""
        subdir = jail / "deep" / "nested"
        subdir.mkdir(parents=True)
        link = subdir / "sneaky"
        link.symlink_to("/etc")
        with pytest.raises(PathEscapeError, match="escapes jail"):
            ensure_jailed(link / "shadow", jail)

    def test_jail_root_itself_is_allowed(self, jail: Path) -> None:
        result = ensure_jailed(jail, jail)
        assert result == jail.resolve()

    @pytest.mark.skipif(_IS_WINDOWS, reason="Symlinks unreliable on Windows")
    def test_symlink_within_jail_is_allowed(self, jail: Path) -> None:
        """Symlink pointing to another file inside jail is OK."""
        target = jail / "allowed.txt"
        link = jail / "internal_link"
        link.symlink_to(target)
        result = ensure_jailed(link, jail)
        assert result == target.resolve()


# ---------------------------------------------------------------------------
# #90 — Read/Write/Deny Enforcement
# ---------------------------------------------------------------------------


class TestDenyEnforcement:
    def test_denied_path_raises(self, basic_config: JailConfig) -> None:
        with pytest.raises(DeniedPathError, match="denied"):
            check_denied("/etc/shadow", basic_config)

    def test_denied_env_file(self, basic_config: JailConfig) -> None:
        with pytest.raises(DeniedPathError, match="denied"):
            check_denied("/workspace/.env", basic_config)

    def test_allowed_path_passes(self, basic_config: JailConfig) -> None:
        check_denied("/workspace/main.py", basic_config)  # Should not raise

    def test_denied_ssh_key(self, basic_config: JailConfig) -> None:
        with pytest.raises(DeniedPathError):
            check_denied("~/.ssh/id_rsa", basic_config)


class TestReadEnforcement:
    def test_read_allowed_by_pattern(self, basic_config: JailConfig) -> None:
        check_read("/workspace/file.py", basic_config)  # "**" matches all

    def test_read_denied_path_raises(self, basic_config: JailConfig) -> None:
        with pytest.raises(DeniedPathError):
            check_read("/etc/shadow", basic_config)

    def test_read_not_in_allowlist(self, jail: Path) -> None:
        config = JailConfig(
            jail_root=jail,
            read_patterns=("docs/**",),
            denied_patterns=(),
        )
        with pytest.raises(ReadNotAllowedError, match="not in read allowlist"):
            check_read("/workspace/secret.py", config)

    def test_empty_read_patterns_allows_all(self, jail: Path) -> None:
        config = JailConfig(jail_root=jail, read_patterns=(), denied_patterns=())
        check_read("/any/file.txt", config)  # Should not raise


class TestWriteEnforcement:
    def test_write_allowed_by_pattern(self, basic_config: JailConfig) -> None:
        check_write("subdir/file.txt", basic_config)

    def test_write_denied_path_raises(self, basic_config: JailConfig) -> None:
        with pytest.raises(DeniedPathError):
            check_write("/etc/shadow", basic_config)

    def test_write_not_in_allowlist(self, basic_config: JailConfig) -> None:
        with pytest.raises(WriteNotAllowedError, match="not in write allowlist"):
            check_write("/root_level.txt", basic_config)

    def test_write_size_exceeded(self, basic_config: JailConfig) -> None:
        with pytest.raises(FileSizeLimitError, match="exceeds limit"):
            check_write("subdir/big.bin", basic_config, size_bytes=9999)

    def test_write_size_at_limit_ok(self, basic_config: JailConfig) -> None:
        check_write("subdir/ok.bin", basic_config, size_bytes=1024)

    def test_temp_dir_only_blocks_non_temp(self, jail: Path) -> None:
        config = JailConfig(
            jail_root=jail,
            write_patterns=("**",),
            denied_patterns=(),
            temp_dir_only=True,
        )
        with pytest.raises(WriteNotAllowedError, match="temp dir"):
            check_write("/workspace/file.txt", config)

    def test_temp_dir_only_allows_tmp(self, jail: Path) -> None:
        config = JailConfig(
            jail_root=jail,
            write_patterns=("**",),
            denied_patterns=(),
            temp_dir_only=True,
        )
        check_write(os.path.join(tempfile.gettempdir(), "output.txt"), config)


# ---------------------------------------------------------------------------
# #92 — TOCTOU-safe Operations
# ---------------------------------------------------------------------------


class TestTocTouSafeOps:
    def test_safe_open_read_existing_file(self, jail: Path) -> None:
        fd = safe_open_read(jail / "allowed.txt", jail)
        try:
            content = os.read(fd, 1024)
            assert content == b"hello"
        finally:
            os.close(fd)

    def test_safe_open_read_escape_blocked(self, jail: Path) -> None:
        with pytest.raises(PathEscapeError):
            safe_open_read(jail / ".." / ".." / "etc" / "hosts", jail)

    @pytest.mark.skipif(_IS_WINDOWS, reason="O_NOFOLLOW not available")
    def test_safe_open_read_symlink_rejected(self, jail: Path) -> None:
        """Final-component symlink is caught by O_NOFOLLOW."""
        target = jail / "allowed.txt"
        link = jail / "link_to_allowed"
        link.symlink_to(target)
        # O_NOFOLLOW causes ELOOP on final-component symlinks
        with pytest.raises((PathEscapeError, OSError)):
            safe_open_read(link, jail)

    def test_safe_open_write_creates_file(self, jail: Path) -> None:
        new_file = jail / "new_file.txt"
        fd = safe_open_write(new_file, jail, create=True)
        try:
            os.write(fd, b"written")
        finally:
            os.close(fd)
        assert new_file.read_text() == "written"

    def test_safe_open_write_escape_blocked(self, jail: Path) -> None:
        with pytest.raises(PathEscapeError):
            safe_open_write(jail / ".." / "escaped.txt", jail)

    def test_safe_open_write_size_check(self, jail: Path) -> None:
        big = jail / "big.bin"
        big.write_bytes(b"x" * 2000)
        with pytest.raises(FileSizeLimitError):
            safe_open_write(big, jail, max_size_bytes=1000)


# ---------------------------------------------------------------------------
# Broker Integration Tests
# ---------------------------------------------------------------------------


class TestFilesystemBroker:
    def test_resolve_virtual(self, broker: FilesystemBroker, jail: Path) -> None:
        result = broker.resolve("skills://current/allowed.txt")
        assert result == (jail / "allowed.txt").resolve()

    def test_resolve_real_path(self, broker: FilesystemBroker, jail: Path) -> None:
        result = broker.resolve(str(jail / "subdir" / "nested.txt"))
        assert result == (jail / "subdir" / "nested.txt").resolve()

    def test_resolve_escape_blocked(self, broker: FilesystemBroker) -> None:
        with pytest.raises(PathEscapeError):
            broker.resolve("/etc/passwd")

    def test_check_read_valid(self, broker: FilesystemBroker, jail: Path) -> None:
        result = broker.check_read(str(jail / "allowed.txt"))
        assert result.exists()

    def test_check_write_valid(self, broker: FilesystemBroker, jail: Path) -> None:
        result = broker.check_write(str(jail / "subdir" / "new.txt"))
        assert result.parent.exists()

    def test_open_read(self, broker: FilesystemBroker, jail: Path) -> None:
        fd = broker.open_read(str(jail / "allowed.txt"))
        try:
            assert os.read(fd, 1024) == b"hello"
        finally:
            os.close(fd)

    def test_open_write(self, broker: FilesystemBroker, jail: Path) -> None:
        target = jail / "subdir" / "output.txt"
        fd = broker.open_write(str(target))
        try:
            os.write(fd, b"output")
        finally:
            os.close(fd)
        assert target.read_text() == "output"

    def test_jail_root_property(self, broker: FilesystemBroker, jail: Path) -> None:
        assert broker.jail_root == jail.resolve()


# ---------------------------------------------------------------------------
# JailConfig.from_capability
# ---------------------------------------------------------------------------


class TestJailConfigFromCapability:
    def test_basic_conversion(self, jail: Path) -> None:
        cap = FilesystemCapability(
            read_paths=["workspace/**"],
            write_paths=["workspace/out/**"],
            max_file_size_mb=5,
            temp_dir_only=False,
        )
        config = JailConfig.from_capability(cap, jail)
        assert config.jail_root == jail
        assert "workspace/**" in config.read_patterns
        assert "workspace/out/**" in config.write_patterns
        assert config.max_file_size_bytes == 5 * 1024 * 1024
        assert config.temp_dir_only is False

    def test_denied_paths_propagated(self, jail: Path) -> None:
        cap = FilesystemCapability()
        config = JailConfig.from_capability(cap, jail)
        assert "/etc/shadow" in config.denied_patterns
        assert "/etc/passwd" in config.denied_patterns


# ---------------------------------------------------------------------------
# #93 — Property-based Escape Tests
# ---------------------------------------------------------------------------


class TestPropertyBasedEscapes:
    """Systematic traversal / escape attempts."""

    _ESCAPE_PAYLOADS = [
        "..",
        "../..",
        "../../..",
        "../../../etc/shadow",
        "..\\..\\..\\windows\\system32",
        "subdir/../../..",
        "subdir/../../../etc/passwd",
        "./././../../..",
        "%2e%2e/%2e%2e",  # URL-encoded (should not be decoded)
        "....//....//",
        "..;/..;/",
    ]

    @pytest.mark.parametrize("payload", _ESCAPE_PAYLOADS)
    def test_traversal_payload_blocked(self, jail: Path, payload: str) -> None:
        """No traversal payload can escape the jail."""
        target = jail / payload
        try:
            result = ensure_jailed(target, jail)
            # If it didn't raise, the resolved path must be inside jail
            jail_str = str(jail.resolve())
            result_str = str(result)
            if _IS_WINDOWS:
                jail_str = jail_str.lower()
                result_str = result_str.lower()
            assert result_str == jail_str or result_str.startswith(jail_str + os.sep), (
                f"Payload {payload!r} resolved to {result}, outside {jail.resolve()}"
            )
        except (PathEscapeError, OSError):
            pass  # Expected — blocked

    @pytest.mark.skipif(_IS_WINDOWS, reason="Symlinks unreliable on Windows")
    def test_chained_symlink_escape(self, jail: Path) -> None:
        """Chain of symlinks that eventually escapes is caught."""
        (jail / "a").symlink_to(jail / "b")
        (jail / "b").mkdir()
        (jail / "b" / "c").symlink_to("/tmp")
        with pytest.raises(PathEscapeError):
            ensure_jailed(jail / "a" / "c" / "escape.txt", jail)

    def test_null_byte_in_path(self, jail: Path) -> None:
        """Null bytes in paths must not bypass checks."""
        with pytest.raises((ValueError, OSError, PathEscapeError)):
            ensure_jailed(jail / "file\x00.txt", jail)

    def test_very_long_path(self, jail: Path) -> None:
        """Extremely long paths should not crash or escape."""
        long_name = "a" * 200
        try:
            result = ensure_jailed(jail / long_name / long_name, jail)
            # If it succeeds, must be inside jail
            jail_str = str(jail.resolve())
            result_str = str(result)
            if _IS_WINDOWS:
                jail_str = jail_str.lower()
                result_str = result_str.lower()
            assert result_str.startswith(jail_str)
        except (OSError, PathEscapeError):
            pass  # Also acceptable — blocked


# ---------------------------------------------------------------------------
# #94 — Race Condition Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_IS_WINDOWS, reason="Symlink race tests require POSIX")
class TestSymlinkRaceConditions:
    """Concurrent symlink creation must never allow jail escape."""

    def test_concurrent_symlink_swap(self, jail: Path) -> None:
        """Race: thread creates symlink while main thread resolves.

        The file starts as a regular file. A background thread repeatedly
        swaps it for a symlink to /tmp. Even if timing aligns, the
        safe_open_read path must never return an fd outside the jail.
        """
        target = jail / "race_target.txt"
        target.write_text("safe content")
        escape_target = "/tmp"

        stop = threading.Event()
        escapes_detected = []

        def symlink_swapper() -> None:
            """Repeatedly swap between file and symlink."""
            while not stop.is_set():
                try:
                    if target.is_symlink() or not target.exists():
                        target.unlink(missing_ok=True)
                        target.write_text("safe content")
                    else:
                        target.unlink()
                        target.symlink_to(escape_target)
                except OSError:
                    pass

        swapper = threading.Thread(target=symlink_swapper, daemon=True)
        swapper.start()

        try:
            for _ in range(100):
                try:
                    fd = safe_open_read(target, jail)
                    try:
                        # Verify fd is inside jail via /proc if available
                        proc_link = f"/proc/self/fd/{fd}"
                        if os.path.exists(proc_link):
                            actual = Path(os.readlink(proc_link)).resolve()
                            jail_str = str(jail.resolve())
                            actual_str = str(actual)
                            if not (actual_str == jail_str or actual_str.startswith(jail_str + "/")):
                                escapes_detected.append(actual_str)
                    finally:
                        os.close(fd)
                except (PathEscapeError, OSError):
                    pass  # Expected — race caught
        finally:
            stop.set()
            swapper.join(timeout=2)
            # Clean up
            target.unlink(missing_ok=True)

        assert not escapes_detected, f"Jail escapes detected: {escapes_detected}"

    def test_directory_to_symlink_race(self, jail: Path) -> None:
        """Race: directory swapped to symlink mid-traversal."""
        subdir = jail / "race_dir"
        subdir.mkdir()
        inner = subdir / "file.txt"
        inner.write_text("inner content")

        stop = threading.Event()

        def dir_swapper() -> None:
            while not stop.is_set():
                try:
                    if subdir.is_symlink():
                        subdir.unlink()
                        subdir.mkdir()
                        (subdir / "file.txt").write_text("inner content")
                    else:
                        import shutil

                        shutil.rmtree(subdir, ignore_errors=True)
                        subdir.symlink_to("/tmp")
                except OSError:
                    pass

        swapper = threading.Thread(target=dir_swapper, daemon=True)
        swapper.start()

        escapes = []
        try:
            for _ in range(100):
                try:
                    resolved = ensure_jailed(subdir / "file.txt", jail)
                    jail_str = str(jail.resolve())
                    res_str = str(resolved)
                    if not (res_str == jail_str or res_str.startswith(jail_str + "/")):
                        escapes.append(res_str)
                except (PathEscapeError, OSError):
                    pass
        finally:
            stop.set()
            swapper.join(timeout=2)
            # Cleanup
            if subdir.is_symlink():
                subdir.unlink()
            elif subdir.exists():
                import shutil

                shutil.rmtree(subdir, ignore_errors=True)

        assert not escapes, f"Jail escapes during race: {escapes}"


# ---------------------------------------------------------------------------
# Security Regression Tests (R1 review fixes)
# ---------------------------------------------------------------------------


class TestSecurityRegressions:
    """Regression tests for R1 review findings."""

    def test_temp_dir_substring_bypass_blocked(self, jail: Path) -> None:
        """Path containing '/tmp' as substring must NOT be treated as temp."""
        config = JailConfig(
            jail_root=jail,
            write_patterns=("**",),
            denied_patterns=(),
            temp_dir_only=True,
        )
        # A path like /home/user/tmpfake/evil.txt should be blocked
        with pytest.raises(WriteNotAllowedError, match="temp dir"):
            check_write("/home/user/tmpfake/evil.txt", config)

    def test_virtual_path_traversal_blocked(self, jail: Path) -> None:
        """skills://current/../../../etc/passwd must be rejected."""
        with pytest.raises(ValueError, match="Traversal"):
            resolve_virtual_path("skills://current/../../../etc/passwd", jail)

    def test_bounded_write_enforces_limit(self, jail: Path) -> None:
        """bounded_write rejects writes that would exceed the limit."""
        target = jail / "bounded.txt"
        fd = os.open(str(target), os.O_WRONLY | os.O_CREAT, 0o644)
        try:
            bounded_write(fd, b"small", max_size_bytes=1024)  # OK
            with pytest.raises(FileSizeLimitError):
                bounded_write(fd, b"x" * 2000, max_size_bytes=1024)
        finally:
            os.close(fd)

    def test_bounded_write_zero_limit_allows_all(self, jail: Path) -> None:
        """max_size_bytes=0 means no limit enforced."""
        target = jail / "unlimited.txt"
        fd = os.open(str(target), os.O_WRONLY | os.O_CREAT, 0o644)
        try:
            written = bounded_write(fd, b"data", max_size_bytes=0)
            assert written == 4
        finally:
            os.close(fd)

    def test_device_check_same_filesystem(self, jail: Path) -> None:
        """Files inside jail must be on same device as jail root."""
        target = jail / "same_dev.txt"
        target.write_text("ok")
        fd = safe_open_read(target, jail)
        os.close(fd)  # Should not raise — same device

    def test_temp_traversal_bypass_blocked(self, jail: Path) -> None:
        """_is_temp_path must reject /tmp/../etc/shadow after normalization."""
        import tempfile as _tf

        config = JailConfig(
            jail_root=jail,
            write_patterns=("**",),
            denied_patterns=(),
            temp_dir_only=True,
        )
        # Path that starts with a temp dir but traverses out
        crafted = os.path.join(_tf.gettempdir(), "..", "etc", "shadow")
        with pytest.raises((WriteNotAllowedError, PathEscapeError)):
            check_write(crafted, config)

    def test_standalone_check_read_enforces_jail(self, jail: Path) -> None:
        """FilesystemBroker.check_read() must reject paths outside the jail."""
        config = JailConfig(
            jail_root=jail,
            read_patterns=("**",),
            denied_patterns=(),
        )
        broker = FilesystemBroker(config)
        with pytest.raises(PathEscapeError):
            broker.check_read("/etc/hosts")

    def test_standalone_check_write_enforces_jail(self, jail: Path) -> None:
        """FilesystemBroker.check_write() must reject paths outside the jail."""
        config = JailConfig(
            jail_root=jail,
            write_patterns=("**",),
            denied_patterns=(),
        )
        broker = FilesystemBroker(config)
        with pytest.raises(PathEscapeError):
            broker.check_write("/etc/hosts")

    @pytest.mark.skipif(_IS_WINDOWS, reason="Windows rejects os.open on directories")
    def test_directory_fd_rejected(self, jail: Path) -> None:
        """safe_open_read must reject directories to prevent openat escape."""
        subdir = jail / "subdir"
        subdir.mkdir(exist_ok=True)
        with pytest.raises(PathEscapeError, match="Not a regular file"):
            safe_open_read(subdir, jail)

    def test_bounded_write_seek_bypass_blocked(self, jail: Path) -> None:
        """bounded_write must account for seek offset, not just file size."""
        target = jail / "sparse.bin"
        fd = os.open(str(target), os.O_WRONLY | os.O_CREAT, 0o644)
        try:
            # Seek far past the limit, then try to write
            os.lseek(fd, 9_000_000, os.SEEK_SET)
            with pytest.raises(FileSizeLimitError):
                bounded_write(fd, b"x" * 2_000_000, max_size_bytes=10_000_000)
        finally:
            os.close(fd)
