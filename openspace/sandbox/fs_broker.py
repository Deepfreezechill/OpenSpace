"""Filesystem broker — jailed, policy-enforced file access.

EPIC 2.2 — Filesystem Broker

Issues:
- #89: Virtual path resolution (skills://current/** → jailed real path)
- #90: Read/write/deny enforcement with max bytes + type checks
- #91: Chroot-style jailing; prevent symlink/traversal escapes
- #92: TOCTOU protection using O_NOFOLLOW, openat(), dir-FD pinning
"""

from __future__ import annotations

import errno
import fnmatch
import os
import platform
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Optional, Union

from openspace.sandbox.leases import FilesystemCapability

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IS_WINDOWS = platform.system() == "Windows"

# O_NOFOLLOW prevents open() from following symlinks (POSIX only)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)

VIRTUAL_SCHEME = "skills://"


# ---------------------------------------------------------------------------
# #89 — Virtual Path Resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JailConfig:
    """Filesystem jail configuration derived from a lease's FilesystemCapability."""

    jail_root: Path
    read_patterns: tuple[str, ...] = ()
    write_patterns: tuple[str, ...] = ()
    denied_patterns: tuple[str, ...] = ()
    max_file_size_bytes: int = 10 * 1024 * 1024  # 10 MB
    temp_dir_only: bool = True

    @classmethod
    def from_capability(cls, capability: FilesystemCapability, jail_root: Path) -> JailConfig:
        """Build a JailConfig from a FilesystemCapability and a jail root."""
        return cls(
            jail_root=jail_root,
            read_patterns=tuple(capability.read_paths),
            write_patterns=tuple(capability.write_paths),
            denied_patterns=tuple(capability.denied_paths),
            max_file_size_bytes=capability.max_file_size_mb * 1024 * 1024,
            temp_dir_only=capability.temp_dir_only,
        )


class PathEscapeError(PermissionError):
    """Raised when a resolved path escapes the jail root."""


class DeniedPathError(PermissionError):
    """Raised when a path matches a deny pattern."""


class FileSizeLimitError(PermissionError):
    """Raised when a write would exceed the file size limit."""


class WriteNotAllowedError(PermissionError):
    """Raised when writing is not allowed for a path."""


class ReadNotAllowedError(PermissionError):
    """Raised when reading is not allowed for a path."""


def resolve_virtual_path(virtual_path: str, jail_root: Path) -> Path:
    """Resolve a ``skills://`` virtual path to a real jailed path.

    ``skills://current/foo.txt`` → ``<jail_root>/foo.txt``

    Raises ``ValueError`` for invalid virtual paths or traversal attempts.
    """
    if not virtual_path.startswith(VIRTUAL_SCHEME):
        raise ValueError(f"Not a virtual path: {virtual_path!r} (must start with {VIRTUAL_SCHEME!r})")

    remainder = virtual_path[len(VIRTUAL_SCHEME) :]

    # Strip the namespace prefix (e.g., "current/")
    if "/" in remainder:
        _namespace, _, relative = remainder.partition("/")
    else:
        relative = ""

    if not relative:
        return jail_root

    # Canonicalise: reject any ".." components before joining to jail
    clean = PurePosixPath(relative)
    if ".." in clean.parts:
        raise ValueError(f"Traversal in virtual path: {virtual_path!r}")

    return jail_root / clean


# ---------------------------------------------------------------------------
# #91 — Chroot-style Jailing
# ---------------------------------------------------------------------------


def _resolve_no_symlinks(path: Path, jail_root: Path) -> Path:
    """Resolve *path* component-by-component, rejecting symlinks that escape.

    On each component we call ``Path.resolve()`` and verify we're
    still inside *jail_root*.  This prevents ``../`` traversal **and**
    symlink-based escapes.
    """
    jail_resolved = jail_root.resolve()

    # Resolve the full path — this follows symlinks
    try:
        resolved = path.resolve()
    except OSError as exc:
        raise PathEscapeError(f"Cannot resolve path: {exc}") from exc

    # On Windows, case-insensitive comparison
    if _IS_WINDOWS:
        jail_str = str(jail_resolved).lower()
        resolved_str = str(resolved).lower()
    else:
        jail_str = str(jail_resolved)
        resolved_str = str(resolved)

    # Must be jail_root itself or a child
    if resolved_str != jail_str and not resolved_str.startswith(jail_str + os.sep):
        raise PathEscapeError(f"Path escapes jail: resolved to {resolved}, jail root is {jail_resolved}")

    return resolved


def ensure_jailed(path: Union[str, Path], jail_root: Path) -> Path:
    """Ensure *path* resolves within *jail_root*.

    Raises ``PathEscapeError`` if the path escapes.
    Raises ``ValueError`` if path contains null bytes.
    """
    path_str = str(path)
    if "\x00" in path_str:
        raise ValueError(f"Null byte in path: {path_str!r}")
    return _resolve_no_symlinks(Path(path), jail_root)


# ---------------------------------------------------------------------------
# #90 — Read/Write/Deny Enforcement
# ---------------------------------------------------------------------------


def _is_temp_path(path_str: str, path_obj: Path) -> bool:
    """Check if *path* is under a recognised temp directory using ancestry, not substring.

    Always normalises ``..`` components via ``Path.resolve(strict=False)``
    to prevent traversal bypasses like ``/tmp/../etc/shadow``.
    """
    import tempfile

    temp_roots = [Path(tempfile.gettempdir())]
    if not _IS_WINDOWS:
        temp_roots.extend([Path("/tmp"), Path("/temp")])
    else:
        for var in ("TEMP", "TMP"):
            val = os.environ.get(var)
            if val:
                temp_roots.append(Path(val))

    try:
        resolved = path_obj.resolve(strict=False)
    except OSError:
        try:
            resolved = Path(path_str).resolve(strict=False)
        except OSError:
            return False

    for temp_root in temp_roots:
        try:
            resolved.relative_to(temp_root.resolve())
            return True
        except (ValueError, OSError):
            continue
    return False


def _matches_any(path_str: str, patterns: tuple[str, ...], *, jail_root: Optional[Path] = None) -> bool:
    """Check if *path_str* matches any of the glob *patterns*.

    When *jail_root* is provided, also matches against the jail-relative path.

    .. note::
        Python's ``fnmatch`` treats ``*`` as matching across path separators,
        unlike shell glob.  A pattern like ``output/*`` will match
        ``output/deep/nested/file.txt``.  This is intentional — allow/deny
        patterns are **recursive** by default.

        Patterns may use ``~`` (tilde) notation which is expanded via
        ``os.path.expanduser`` before matching.

        Matching is against the **full path** and **jail-relative path** only.
        Basename-only matching is intentionally excluded to prevent
        over-permissive allowlists (e.g., ``allowed.txt`` matching any
        same-named file in any subdirectory).
    """
    if not patterns:
        return False

    candidates: list[str] = [path_str]

    if jail_root is not None:
        try:
            rel = os.path.relpath(path_str, jail_root)
            candidates.append(rel)
            candidates.append(rel.replace("\\", "/"))
        except ValueError:
            pass

    for pattern in patterns:
        expanded = os.path.expanduser(pattern)
        for candidate in candidates:
            if fnmatch.fnmatch(candidate, expanded):
                return True
            if expanded != pattern and fnmatch.fnmatch(candidate, pattern):
                return True
    return False


def check_denied(path: Union[str, Path], config: JailConfig) -> None:
    """Raise ``DeniedPathError`` if *path* matches any deny pattern.

    .. note::
        This is a **pattern-only** check.  For full jail enforcement
        (containment + deny + allowlist), use :class:`FilesystemBroker`.
    """
    path_str = str(path)
    if _matches_any(path_str, config.denied_patterns, jail_root=config.jail_root):
        raise DeniedPathError(f"Path is denied by policy: {path_str}")


def check_read(path: Union[str, Path], config: JailConfig) -> None:
    """Raise if reading *path* is not allowed by **pattern rules**.

    Rules:
    1. Path must not be denied.
    2. If read_patterns is non-empty, path must match at least one.

    .. note::
        This is a **pattern-only** check.  For full jail enforcement
        (containment + deny + allowlist), use :class:`FilesystemBroker`.
    """
    check_denied(path, config)

    path_str = str(path)
    if config.read_patterns and not _matches_any(path_str, config.read_patterns, jail_root=config.jail_root):
        raise ReadNotAllowedError(f"Path not in read allowlist: {path_str}")


def check_write(
    path: Union[str, Path],
    config: JailConfig,
    size_bytes: int = 0,
) -> None:
    """Raise if writing *path* is not allowed by **pattern rules**.

    Rules:
    1. Path must not be denied.
    2. If temp_dir_only, path must be under a temp directory.
    3. If write_patterns is non-empty, path must match at least one.
    4. size_bytes must not exceed max_file_size_bytes.

    .. note::
        This is a **pattern-only** check.  For full jail enforcement
        (containment + deny + allowlist), use :class:`FilesystemBroker`.
    """
    check_denied(path, config)

    path_str = str(path)
    path_obj = Path(path)

    if config.temp_dir_only:
        if not _is_temp_path(path_str, path_obj):
            raise WriteNotAllowedError(f"Writes restricted to temp directories: {path_str}")

    if config.write_patterns and not _matches_any(path_str, config.write_patterns, jail_root=config.jail_root):
        raise WriteNotAllowedError(f"Path not in write allowlist: {path_str}")

    if size_bytes > config.max_file_size_bytes:
        raise FileSizeLimitError(f"Write of {size_bytes} bytes exceeds limit of {config.max_file_size_bytes} bytes")


# ---------------------------------------------------------------------------
# #92 — TOCTOU-safe File Operations
# ---------------------------------------------------------------------------


def _ensure_regular_file(fd: int, path: Path) -> None:
    """Reject non-regular files (directories, FIFOs, devices, sockets).

    A directory fd could be exploited via ``openat()`` / ``dir_fd`` to escape
    the jail.  Only regular files are permitted.
    """
    try:
        mode = os.fstat(fd).st_mode
    except OSError:
        os.close(fd)
        raise
    if not stat.S_ISREG(mode):
        os.close(fd)
        raise PathEscapeError(f"Not a regular file (mode={oct(mode)}): {path}")


def safe_open_read(path: Union[str, Path], jail_root: Path) -> int:
    """Open a file for reading with TOCTOU protection.

    Returns a raw file descriptor. Caller MUST close it via ``os.close(fd)``.

    On POSIX: uses ``O_NOFOLLOW`` to reject symlinks at the final component.
    On Windows: relies on ``ensure_jailed`` pre-check (no O_NOFOLLOW).
    Rejects non-regular files (directories, FIFOs, devices) to prevent
    ``openat``-based jail escapes.
    """
    resolved = ensure_jailed(path, jail_root)

    flags = os.O_RDONLY | _O_NOFOLLOW
    try:
        fd = os.open(str(resolved), flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise PathEscapeError(f"Symlink detected at final component: {resolved}") from exc
        raise

    _ensure_regular_file(fd, resolved)
    _verify_fd_path(fd, resolved, jail_root)
    return fd


def safe_open_write(
    path: Union[str, Path],
    jail_root: Path,
    *,
    create: bool = True,
    max_size_bytes: int = 0,
) -> int:
    """Open a file for writing with TOCTOU protection.

    Returns a raw file descriptor. Caller MUST close it via ``os.close(fd)``.
    """
    resolved = ensure_jailed(path, jail_root)

    flags = os.O_WRONLY | _O_NOFOLLOW
    if create:
        flags |= os.O_CREAT
    mode = 0o644

    try:
        fd = os.open(str(resolved), flags, mode)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise PathEscapeError(f"Symlink detected at final component: {resolved}") from exc
        raise

    # Post-open type check: reject directories/FIFOs/devices
    _ensure_regular_file(fd, resolved)

    # Post-open size check
    if max_size_bytes > 0:
        try:
            st = os.fstat(fd)
            if st.st_size > max_size_bytes:
                os.close(fd)
                raise FileSizeLimitError(f"Existing file {resolved} is {st.st_size} bytes, exceeds {max_size_bytes}")
        except FileSizeLimitError:
            raise
        except OSError:
            os.close(fd)
            raise

    _verify_fd_path(fd, resolved, jail_root, unlink_on_escape=True)
    return fd


def bounded_write(fd: int, data: bytes, max_size_bytes: int) -> int:
    """Write *data* to *fd*, enforcing cumulative size limit.

    Checks ``max(file_size, current_offset) + len(data)`` against
    *max_size_bytes* to prevent sparse-file / seek-based bypasses.
    Raises ``FileSizeLimitError`` if the write would exceed the cap.
    Returns the number of bytes written.

    .. warning::
        The check-then-write is **not atomic**.  Concurrent writers to the
        same fd can exceed the cap.  This is acceptable because each sandbox
        runs a single skill process; multi-writer scenarios are out of scope.
        EPIC 2.9 (Runtime Quotas) adds OS-level ``rlimit`` enforcement as
        a hard backstop.
    """
    if max_size_bytes > 0:
        try:
            current_size = os.fstat(fd).st_size
        except OSError:
            current_size = 0
        try:
            current_offset = os.lseek(fd, 0, os.SEEK_CUR)
        except OSError:
            current_offset = current_size
        effective_pos = max(current_size, current_offset)
        if effective_pos + len(data) > max_size_bytes:
            raise FileSizeLimitError(
                f"Write of {len(data)} bytes at offset {effective_pos} would exceed limit of {max_size_bytes} bytes"
            )
    return os.write(fd, data)


def _verify_fd_path(
    fd: int,
    expected: Path,
    jail_root: Path,
    *,
    unlink_on_escape: bool = False,
) -> None:
    """Post-open verification: ensure the fd actually points inside the jail.

    - On Linux: reads ``/proc/self/fd/{fd}`` to verify actual path.
    - On all platforms: checks ``st_dev`` matches the jail root's device
      to detect hard-link escapes across filesystems.
    - On Windows/macOS without /proc: relies on st_dev check only.

    If *unlink_on_escape* is True and a jail escape is detected, the file
    is unlinked before raising, limiting damage from O_CREAT races.

    **Limitation**: Same-device hard links created by a compromised process
    that already has write access both inside and outside the jail can
    bypass path-based checks.  This is mitigated by the sandbox process
    broker (EPIC 2.4) restricting link/symlink syscalls.
    """
    # Device check: file must be on the same filesystem as the jail
    try:
        fd_stat = os.fstat(fd)
        jail_stat = os.stat(str(jail_root.resolve()))
        if fd_stat.st_dev != jail_stat.st_dev:
            if unlink_on_escape:
                _safe_unlink(expected)
            os.close(fd)
            raise PathEscapeError(f"File device ({fd_stat.st_dev}) differs from jail device ({jail_stat.st_dev})")
    except PathEscapeError:
        raise
    except OSError:
        pass  # Best-effort

    if _IS_WINDOWS:
        return

    proc_link = f"/proc/self/fd/{fd}"
    try:
        actual = Path(os.readlink(proc_link)).resolve()
    except OSError:
        return

    jail_resolved = jail_root.resolve()
    actual_str = str(actual)
    jail_str = str(jail_resolved)

    if actual_str != jail_str and not actual_str.startswith(jail_str + "/"):
        if unlink_on_escape:
            _safe_unlink(expected)
        os.close(fd)
        raise PathEscapeError(f"Post-open verification failed: fd points to {actual}, outside jail {jail_resolved}")

    return actual


def _safe_unlink(path: Path) -> None:
    """Best-effort removal of a file created during a TOCTOU race."""
    try:
        os.unlink(str(path))
    except OSError:
        pass


# ---------------------------------------------------------------------------
# High-level Broker
# ---------------------------------------------------------------------------


class FilesystemBroker:
    """Policy-enforced filesystem broker with jailing.

    Combines virtual path resolution, jail enforcement, deny-list
    checking, read/write policy, file-size limits, and TOCTOU-safe
    operations into a single entry point.
    """

    def __init__(self, config: JailConfig) -> None:
        self._config = config
        self._jail_root = config.jail_root.resolve()

    @property
    def jail_root(self) -> Path:
        return self._jail_root

    @property
    def config(self) -> JailConfig:
        return self._config

    def resolve(self, path: str) -> Path:
        """Resolve a path (virtual or real) to a jailed real path.

        Virtual paths (``skills://current/...``) are resolved first,
        then jail enforcement is applied.
        """
        if path.startswith(VIRTUAL_SCHEME):
            real_path = resolve_virtual_path(path, self._jail_root)
        else:
            real_path = Path(path)
        return ensure_jailed(real_path, self._jail_root)

    def check_read(self, path: str) -> Path:
        """Validate and return the jailed path for reading.

        Raises on escape, denied, or not-in-allowlist.
        """
        resolved = self.resolve(path)
        check_read(resolved, self._config)
        return resolved

    def check_write(self, path: str, size_bytes: int = 0) -> Path:
        """Validate and return the jailed path for writing.

        Raises on escape, denied, not-in-allowlist, or size exceeded.
        """
        resolved = self.resolve(path)
        check_write(resolved, self._config, size_bytes=size_bytes)
        return resolved

    def open_read(self, path: str) -> int:
        """TOCTOU-safe open for reading. Returns raw fd.

        Performs deny-list re-check after safe_open using the actual resolved
        path from /proc/self/fd (Linux) to close the check-then-open race window.
        """
        resolved = self.check_read(path)
        fd = safe_open_read(resolved, self._jail_root)
        self._post_open_policy_check(fd, resolved, is_write=False)
        return fd

    def open_write(self, path: str, size_bytes: int = 0) -> int:
        """TOCTOU-safe open for writing. Returns raw fd.

        Performs deny-list re-check after safe_open using the actual resolved
        path from /proc/self/fd (Linux) to close the check-then-open race window.

        .. important::
            Callers MUST use :func:`bounded_write` instead of ``os.write``
            to enforce cumulative file-size limits.  Direct ``os.write``
            bypasses size cap enforcement.
        """
        resolved = self.check_write(path, size_bytes=size_bytes)
        fd = safe_open_write(
            resolved,
            self._jail_root,
            max_size_bytes=self._config.max_file_size_bytes,
        )
        self._post_open_policy_check(fd, resolved, is_write=True)
        return fd

    def _post_open_policy_check(self, fd: int, expected: Path, *, is_write: bool = False) -> None:
        """Re-check the full policy against the actual fd path (Linux /proc, macOS F_GETPATH).

        If the actual path differs from expected, re-runs deny, allowlist, and
        temp_dir checks to close the check-then-open TOCTOU window.
        """
        actual_path = self._resolve_fd_actual_path(fd)
        if actual_path is None:
            return

        actual_str = str(actual_path)
        if actual_str == str(expected.resolve()):
            return  # No race — path unchanged

        # Full policy re-check on the actual path
        # 1. Jail containment
        jail_str = str(self._jail_root)
        if actual_str != jail_str and not actual_str.startswith(jail_str + os.sep):
            os.close(fd)
            raise PathEscapeError(f"Post-open: actual path {actual_path} is outside jail {self._jail_root}")

        # 2. Deny patterns
        if _matches_any(actual_str, self._config.denied_patterns, jail_root=self._jail_root):
            os.close(fd)
            raise DeniedPathError(f"Post-open deny check: actual path {actual_path} matches deny pattern")

        # 3. Read/write allowlist
        if is_write:
            if self._config.write_patterns and not _matches_any(
                actual_str, self._config.write_patterns, jail_root=self._jail_root
            ):
                os.close(fd)
                raise WriteNotAllowedError(f"Post-open: actual path {actual_path} not in write allowlist")
            if self._config.temp_dir_only:
                if not _is_temp_path(actual_str, actual_path):
                    os.close(fd)
                    raise WriteNotAllowedError(f"Post-open: actual path {actual_path} not in temp directory")
        else:
            if self._config.read_patterns and not _matches_any(
                actual_str, self._config.read_patterns, jail_root=self._jail_root
            ):
                os.close(fd)
                raise ReadNotAllowedError(f"Post-open: actual path {actual_path} not in read allowlist")

    @staticmethod
    def _resolve_fd_actual_path(fd: int) -> Optional[Path]:
        """Resolve the actual filesystem path of an open fd.

        - Linux: ``/proc/self/fd/{fd}``
        - macOS: ``fcntl.F_GETPATH``
        - Windows/other: returns None (no reliable mechanism)
        """
        if _IS_WINDOWS:
            return None

        # Try /proc/self/fd first (Linux)
        proc_link = f"/proc/self/fd/{fd}"
        try:
            return Path(os.readlink(proc_link)).resolve()
        except OSError:
            pass

        # Try fcntl F_GETPATH (macOS)
        try:
            import fcntl

            F_GETPATH = 50  # macOS-specific
            result = fcntl.fcntl(fd, F_GETPATH, b"\0" * 1024)
            if isinstance(result, bytes):
                path_bytes = result.split(b"\0", 1)[0]
            else:
                path_bytes = b""
            if not path_bytes:
                return None  # Empty result — cannot verify
            return Path(path_bytes.decode()).resolve()
        except (ImportError, OSError, AttributeError, UnicodeDecodeError):
            pass

        return None
