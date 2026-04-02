"""Process broker — policy-enforced process execution control.

EPIC 2.4 — Process Broker

Issues:
- #99: Command allow/deny enforcement with blocked-command safety invariant
- #100: Shell invocation control (bash/sh/cmd/powershell restriction)
- #101: Process tracking with concurrency limits and execution time bounds
- #102: Dangerous syscall restriction (link, symlink, hardlink, mount)
- #103: ProcessBrokerPort integration and config-from-capability pattern
"""

from __future__ import annotations

import fnmatch
import threading
import time
from dataclasses import dataclass, field
from pathlib import PurePosixPath, PureWindowsPath

from openspace.sandbox.leases import REQUIRED_BLOCKED_COMMANDS, ProcessCapability

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Shells that must be blocked when allow_shell=False
_SHELL_BINARIES: frozenset[str] = frozenset(
    {
        "sh",
        "bash",
        "dash",
        "zsh",
        "fish",
        "csh",
        "tcsh",
        "ksh",
        "ash",
        "rbash",
        "rksh",
        "cmd",
        "cmd.exe",
        "powershell",
        "powershell.exe",
        "pwsh",
        "pwsh.exe",
    }
)

# Commands that can invoke shells indirectly (shell wrappers).
# When allow_shell=False, these must be checked for shell args.
_SHELL_WRAPPERS: frozenset[str] = frozenset(
    {
        "env",
        "busybox",
        "xargs",
        "script",
        "nohup",
        "strace",
        "ltrace",
        "nice",
        "ionice",
        "taskset",
        "timeout",
        "chrt",
        "setsid",
        "sudo",
        "su",
        "doas",
        "runuser",
        "time",
        "stdbuf",
        "chpst",
        "softlimit",
        "watch",
        "exec",
        "find",
        "parallel",  # -exec / ::: can invoke arbitrary commands
        "flock",  # flock /tmp/lock <command>
    }
)

# Syscalls that can create hard-link / symlink escapes from filesystem jails.
# EPIC 2.2 deferred this to 2.4: "same-device hard links created by a
# compromised process [...] can bypass path-based checks. This is mitigated
# by the sandbox process broker (EPIC 2.4) restricting link/symlink syscalls."
_DANGEROUS_SYSCALLS: frozenset[str] = frozenset(
    {
        "link",
        "linkat",
        "symlink",
        "symlinkat",
        "rename",
        "renameat",
        "renameat2",
        "mount",
        "umount",
        "umount2",
        "pivot_root",
        "chroot",
        "mknod",
        "mknodat",
    }
)

# Commands that create links (user-space equivalents of dangerous syscalls)
_LINK_COMMANDS: frozenset[str] = frozenset(
    {
        "ln",
        "link",
        "mklink",
        "mount",
        "umount",
        "fusermount",
        "mknod",
    }
)

# Wrapper flags that implicitly spawn a shell (must be checked when allow_shell=False)
_SHELL_INVOKING_FLAGS: dict[str, frozenset[str]] = {
    "sudo": frozenset({"-s", "--shell", "-i", "--login"}),
    "su": frozenset({"-", "-l", "--login", "-s", "--shell", "-c", "--command"}),
    "doas": frozenset({"-s"}),
    "runuser": frozenset({"-l", "--login", "-s", "--shell", "-c", "--command"}),
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ProcessPolicyError(Exception):
    """Base for all process broker policy violations."""


class CommandBlockedError(ProcessPolicyError):
    """Raised when a command is on the block list."""


class CommandNotAllowedError(ProcessPolicyError):
    """Raised when a command is not on the allow list."""


class ShellNotAllowedError(ProcessPolicyError):
    """Raised when shell invocation is disabled."""


class ProcessLimitError(ProcessPolicyError):
    """Raised when process concurrency limit is reached."""


class SyscallBlockedError(ProcessPolicyError):
    """Raised when a dangerous syscall is attempted."""


class ExecutionTimeoutError(ProcessPolicyError):
    """Raised when a process exceeds its execution time limit."""


# ---------------------------------------------------------------------------
# #99 — Command Allow/Deny Enforcement
# ---------------------------------------------------------------------------


def _extract_arg_tokens(args: list[str]) -> list[str]:
    """Extract all command-like tokens from an argument list.

    Handles:
    - Plain args: ``["bash"]`` → ``["bash"]``
    - Flag values: ``["-Sbash"]`` → ``["bash"]`` (strip single-char flag)
    - Long flag values: ``["--split-string=bash -c id"]`` → ``["bash", "-c", "id"]``
    - Env-var assignments: ``["FOO=bar"]`` → skipped (not commands)
    - Space-concatenated: ``["sh -c id"]`` → ``["sh", "-c", "id"]``
    - Sentinel ``--``: subsequent args treated as positional (no skipping)
    """
    tokens: list[str] = []
    past_sentinel = False
    for arg in args:
        if arg == "--" and not past_sentinel:
            past_sentinel = True
            continue

        if past_sentinel:
            # After --, everything is positional (no flag/env-var skipping)
            tokens.extend(arg.split())
        elif arg.startswith("--") and "=" in arg:
            # --flag=value → extract value, split on spaces
            _, _, value = arg.partition("=")
            tokens.extend(value.split())
        elif arg.startswith("-") and len(arg) > 2 and not arg.startswith("--"):
            # -Svalue → extract value after single-char flag (e.g., -Sbash)
            value = arg[2:]
            tokens.extend(value.split())
        elif arg.startswith("-"):
            # Plain flag like -c, --verbose → skip
            continue
        elif "=" in arg and not arg.startswith("/") and not arg.startswith("."):
            # ENV_VAR=value → skip entirely (not a command invocation)
            continue
        else:
            # Plain arg — split on spaces for concatenated commands
            tokens.extend(arg.split())

    return tokens


def _extract_basename(command: str) -> str:
    """Extract the command basename from a full path or bare command.

    Handles both Unix and Windows paths, strips quotes, and normalizes
    .exe extensions. Returns lowercase for case-insensitive matching.
    """
    # Strip surrounding quotes (single or double)
    cleaned = command.strip().strip("'\"")

    # Handle Windows paths first (backslash), then Unix
    if "\\" in cleaned:
        name = PureWindowsPath(cleaned).name
        if name:
            return name.lower()
    if "/" in cleaned:
        name = PurePosixPath(cleaned).name
        if name:
            return name.lower()
    return cleaned.lower()


def _strip_exe(basename: str) -> str:
    """Strip all trailing .exe extensions for cross-platform command matching."""
    while basename.endswith(".exe"):
        basename = basename[:-4]
    return basename


def check_command_blocked(command: str, blocked_commands: list[str]) -> None:
    """Raise ``CommandBlockedError`` if *command* matches any blocked pattern.

    Always checks against REQUIRED_BLOCKED_COMMANDS regardless of the
    provided list. Matching is case-insensitive, supports fnmatch globs,
    extracts basenames from full paths, and strips .exe extensions for
    cross-platform consistency (rm.exe → rm).
    """
    basename = _extract_basename(command)
    basename_no_exe = _strip_exe(basename)

    # Merge explicit + required blocks
    all_blocked = set(b.lower() for b in blocked_commands) | {r.lower() for r in REQUIRED_BLOCKED_COMMANDS}

    for pattern in all_blocked:
        pattern_no_exe = _strip_exe(pattern)
        # Match both with and without .exe
        if (
            fnmatch.fnmatch(basename, pattern)
            or fnmatch.fnmatch(basename_no_exe, pattern)
            or fnmatch.fnmatch(basename, pattern_no_exe)
            or fnmatch.fnmatch(basename_no_exe, pattern_no_exe)
        ):
            raise CommandBlockedError(f"Command '{command}' (basename '{basename}') is blocked by pattern '{pattern}'")


def check_command_allowed(command: str, allowed_commands: list[str]) -> None:
    """Raise ``CommandNotAllowedError`` if *command* is not in the allow list.

    An empty allow list means **all non-blocked commands** are permitted
    (open policy). A non-empty list enforces an allowlist (closed policy).
    Matching is case-insensitive and supports fnmatch globs.
    """
    if not allowed_commands:
        return  # Empty allowlist = open policy (block list still applies)

    basename = _extract_basename(command)
    basename_no_exe = _strip_exe(basename)

    for pattern in allowed_commands:
        pat = pattern.lower()
        if fnmatch.fnmatch(basename, pat) or fnmatch.fnmatch(basename_no_exe, pat):
            return  # Allowed

    raise CommandNotAllowedError(f"Command '{command}' (basename '{basename}') not in allowed list")


# ---------------------------------------------------------------------------
# #100 — Shell Invocation Control
# ---------------------------------------------------------------------------


def check_shell_allowed(command: str, allow_shell: bool, args: list[str] | None = None) -> None:
    """Raise ``ShellNotAllowedError`` if shell invocation is disabled.

    Detects shell binaries by basename matching against ``_SHELL_BINARIES``.
    Also detects shell wrappers (env, busybox, sudo, etc.) that invoke
    shells via their arguments.
    """
    if allow_shell:
        return  # Shells permitted

    basename = _extract_basename(command)
    basename_no_exe = _strip_exe(basename)

    if basename in _SHELL_BINARIES or basename_no_exe in _SHELL_BINARIES:
        raise ShellNotAllowedError(f"Shell invocation via '{command}' is not allowed (allow_shell=False)")

    # Check shell wrappers: env bash, busybox sh, sudo bash, etc.
    if basename in _SHELL_WRAPPERS or basename_no_exe in _SHELL_WRAPPERS:
        if args:
            for token in _extract_arg_tokens(args):
                token_basename = _extract_basename(token)
                token_no_exe = _strip_exe(token_basename)
                if token_basename in _SHELL_BINARIES or token_no_exe in _SHELL_BINARIES:
                    raise ShellNotAllowedError(
                        f"Shell invocation via wrapper '{command}' with "
                        f"shell '{token}' is not allowed (allow_shell=False)"
                    )


def check_shell_command(shell_command: str, allow_shell: bool) -> None:
    """Raise ``ShellNotAllowedError`` if shell command execution is disabled.

    This checks the content of a shell command string (e.g., passed to
    ``subprocess.run(cmd, shell=True)``). When ``allow_shell`` is False,
    all shell commands are rejected.
    """
    if allow_shell:
        return

    raise ShellNotAllowedError(f"Shell command execution is not allowed (allow_shell=False): '{shell_command[:100]}'")


# ---------------------------------------------------------------------------
# #102 — Dangerous Syscall Restriction
# ---------------------------------------------------------------------------


def check_syscall_allowed(syscall: str, *args: str) -> None:
    """Raise ``SyscallBlockedError`` if *syscall* is dangerous.

    Blocks link/symlink/rename/mount/chroot/mknod syscalls that can
    create jail escapes (deferred from EPIC 2.2 filesystem broker).
    """
    if syscall.lower() in _DANGEROUS_SYSCALLS:
        raise SyscallBlockedError(f"Syscall '{syscall}' is blocked (jail escape risk). Args: {args[:5]}")


def check_link_command(command: str) -> None:
    """Raise ``SyscallBlockedError`` if command creates links.

    Blocks user-space commands that create hard/symbolic links,
    which can bypass filesystem jail path-based checks.
    Strips .exe extension for cross-platform matching.
    """
    basename = _extract_basename(command)
    basename_no_exe = _strip_exe(basename)

    if basename in _LINK_COMMANDS or basename_no_exe in _LINK_COMMANDS:
        raise SyscallBlockedError(
            f"Command '{command}' (basename '{basename}') creates links and is blocked to prevent jail escapes"
        )


# ---------------------------------------------------------------------------
# #101 — Process Tracking
# ---------------------------------------------------------------------------


@dataclass
class ProcessRecord:
    """Record of a tracked process."""

    pid: int
    command: str
    start_time: float = field(default_factory=time.monotonic)
    terminated: bool = False


@dataclass
class ProcessTracker:
    """Thread-safe process tracker with concurrency and time limits.

    Enforces:
    - Maximum concurrent processes (``max_processes``)
    - Maximum execution time per process (``max_execution_time_s``)
    """

    max_processes: int
    max_execution_time_s: int
    _processes: dict[int, ProcessRecord] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def track(self, pid: int, command: str) -> None:
        """Register a new process. Raises ``ProcessLimitError`` if at limit."""
        with self._lock:
            # Clean up terminated processes
            self._cleanup_locked()

            active = sum(1 for p in self._processes.values() if not p.terminated)
            if active >= self.max_processes:
                raise ProcessLimitError(
                    f"Process limit reached ({active}/{self.max_processes}). Cannot track pid={pid} command='{command}'"
                )

            if pid in self._processes and not self._processes[pid].terminated:
                raise ProcessLimitError(f"Process pid={pid} is already tracked and active")

            self._processes[pid] = ProcessRecord(pid=pid, command=command)

    def release(self, pid: int) -> None:
        """Mark a process as terminated."""
        with self._lock:
            if pid in self._processes:
                self._processes[pid].terminated = True

    def check_timeout(self, pid: int) -> None:
        """Raise ``ExecutionTimeoutError`` if process has exceeded time limit."""
        with self._lock:
            record = self._processes.get(pid)
            if record is None:
                return

            elapsed = time.monotonic() - record.start_time
            if elapsed > self.max_execution_time_s:
                record.terminated = True
                raise ExecutionTimeoutError(
                    f"Process pid={pid} command='{record.command}' exceeded "
                    f"time limit ({elapsed:.1f}s > {self.max_execution_time_s}s)"
                )

    def check_all_timeouts(self) -> list[int]:
        """Check all active processes for timeouts. Returns list of timed-out PIDs."""
        timed_out: list[int] = []
        with self._lock:
            now = time.monotonic()
            for pid, record in self._processes.items():
                if record.terminated:
                    continue
                elapsed = now - record.start_time
                if elapsed > self.max_execution_time_s:
                    record.terminated = True
                    timed_out.append(pid)
        return timed_out

    @property
    def active_count(self) -> int:
        """Number of active (non-terminated) processes."""
        with self._lock:
            self._cleanup_locked()
            return sum(1 for p in self._processes.values() if not p.terminated)

    def list_processes(self) -> list[ProcessRecord]:
        """Return snapshot of all tracked processes (defensive copies)."""
        with self._lock:
            return [
                ProcessRecord(
                    pid=p.pid,
                    command=p.command,
                    start_time=p.start_time,
                    terminated=p.terminated,
                )
                for p in self._processes.values()
            ]

    def _cleanup_locked(self) -> None:
        """Remove terminated processes older than 60 seconds (must hold lock)."""
        now = time.monotonic()
        stale = [pid for pid, rec in self._processes.items() if rec.terminated and (now - rec.start_time) > 60.0]
        for pid in stale:
            del self._processes[pid]


# ---------------------------------------------------------------------------
# Config + Broker (combines all enforcement)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessBrokerConfig:
    """Immutable configuration derived from a ``ProcessCapability``."""

    allowed_commands: tuple[str, ...]
    blocked_commands: tuple[str, ...]
    max_processes: int
    max_execution_time_s: int
    allow_shell: bool

    @classmethod
    def from_capability(cls, cap: ProcessCapability) -> "ProcessBrokerConfig":
        """Create config from a ProcessCapability, enforcing safety invariants."""
        # Merge link commands into blocked set
        merged_blocked = set(cap.blocked_commands) | _LINK_COMMANDS | REQUIRED_BLOCKED_COMMANDS
        return cls(
            allowed_commands=tuple(cap.allowed_commands),
            blocked_commands=tuple(sorted(merged_blocked)),
            max_processes=cap.max_processes,
            max_execution_time_s=cap.max_execution_time_s,
            allow_shell=cap.allow_shell,
        )


class ProcessBroker:
    """Policy-enforced process execution broker.

    Combines command allow/deny, shell control, process tracking,
    execution time limits, and dangerous syscall/command blocking.

    **Usage pattern** (follows FilesystemBroker / NetworkProxy)::

        config = ProcessBrokerConfig.from_capability(lease.process)
        broker = ProcessBroker(config)

        # Before exec:
        broker.check_command("python", ["script.py"])
        broker.track_process(pid, "python")

        # After completion:
        broker.release_process(pid)
    """

    def __init__(self, config: ProcessBrokerConfig) -> None:
        self._config = config
        self._tracker = ProcessTracker(
            max_processes=config.max_processes,
            max_execution_time_s=config.max_execution_time_s,
        )

    @property
    def config(self) -> ProcessBrokerConfig:
        return self._config

    def check_command(self, command: str, args: list[str] | None = None) -> None:
        """Full command validation: blocked → link-check → shell-check → allowlist.

        Inspects both *command* and *args* to catch shell wrapper bypasses
        (e.g., ``env bash -c "..."``). Uses ``_extract_arg_tokens`` to
        parse flag values, env-var values, and concatenated args.
        Raises appropriate policy errors if any check fails.
        """
        basename = _extract_basename(command)
        basename_no_exe = _strip_exe(basename)
        is_wrapper = basename in _SHELL_WRAPPERS or basename_no_exe in _SHELL_WRAPPERS

        # 1. Block list (deny-before-allow, includes REQUIRED_BLOCKED_COMMANDS)
        check_command_blocked(command, list(self._config.blocked_commands))

        # 2. For wrappers: check arg tokens for blocked commands (e.g., env rm)
        #    Non-wrappers don't get arg scanning to avoid false positives
        #    (e.g., `git checkout rm` should not block on `rm` in args)
        if is_wrapper and args:
            for token in _extract_arg_tokens(args):
                check_command_blocked(token, list(self._config.blocked_commands))
                check_link_command(token)

        # 3. Link/symlink command check (EPIC 2.2 deferred hardening)
        check_link_command(command)

        # 4. Shell binary check (with args inspection for wrappers)
        check_shell_allowed(command, self._config.allow_shell, args)

        # 4b. Shell-invoking wrapper flags (sudo -s, su -, doas -s)
        if not self._config.allow_shell and is_wrapper and args:
            shell_flags = _SHELL_INVOKING_FLAGS.get(basename_no_exe)
            if shell_flags:
                for arg in args:
                    stripped = arg.strip().strip("'\"")
                    if stripped in shell_flags:
                        raise ShellNotAllowedError(
                            f"Wrapper '{basename}' with flag '{stripped}' invokes a shell but allow_shell=False"
                        )
                    # Combined short flags: -si means -s + -i
                    if stripped.startswith("-") and not stripped.startswith("--") and len(stripped) > 2:
                        for flag in shell_flags:
                            if len(flag) == 2 and flag[1] in stripped[1:]:
                                raise ShellNotAllowedError(
                                    f"Wrapper '{basename}' with combined flag "
                                    f"'{stripped}' contains shell flag '{flag}' "
                                    f"but allow_shell=False"
                                )

        # 5. Allow list — for wrappers, also check the wrapped command
        check_command_allowed(command, list(self._config.allowed_commands))
        if is_wrapper and args and self._config.allowed_commands:
            allowed = list(self._config.allowed_commands)

            def _is_skippable(basename: str) -> bool:
                """Return True if token should be skipped in allowlist scan."""
                if not basename:
                    return True
                if basename.isdigit():
                    return True
                if basename in _SHELL_WRAPPERS:
                    return True
                if all(c in ".{}+;:" for c in basename):
                    return True
                # Glob patterns (find -name *.py) are not commands
                if any(c in basename for c in "*?["):
                    return True
                return False

            # A) Scan raw args for first plain positional command
            found_plain = False
            for raw_arg in args:
                token = raw_arg.strip().strip("'\"")
                if not token or token.startswith("-"):
                    continue  # flag
                # Env-var assignment (KEY=value) — not a command
                if "=" in token and not token.startswith("/") and not token.startswith("."):
                    continue
                token_basename = _extract_basename(token)
                if _is_skippable(token_basename):
                    continue
                check_command_allowed(token, allowed)
                found_plain = True
                break  # Only check the first actual command

            # B) Scan flag-embedded values (--split-string=curl, -Scurl)
            #    Only runs if pass A didn't find any plain positional command.
            if not found_plain:
                for token in _extract_arg_tokens(args):
                    token_basename = _extract_basename(token)
                    if _is_skippable(token_basename):
                        continue
                    check_command_allowed(token, allowed)
                    break

            # C) Multi-command wrappers: check commands after -exec/-execdir
            #    Catches: find . -exec echo ; -exec curl evil ;
            _EXEC_FLAGS = frozenset({"-exec", "-execdir"})
            _EXEC_TERMINATORS = frozenset({";", "+", "{}"})
            in_exec = False
            exec_checked_first = False
            for raw_arg in args:
                token = raw_arg.strip().strip("'\"")
                if token in _EXEC_FLAGS:
                    in_exec = True
                    exec_checked_first = False
                    continue
                if in_exec:
                    if token in _EXEC_TERMINATORS:
                        continue
                    if not exec_checked_first:
                        # First token after -exec is the command
                        token_basename = _extract_basename(token)
                        if not _is_skippable(token_basename):
                            check_command_allowed(token, allowed)
                        exec_checked_first = True

    def check_shell(self, shell_command: str) -> None:
        """Validate a shell command string (``shell=True`` invocation)."""
        check_shell_command(shell_command, self._config.allow_shell)

    def check_syscall(self, syscall: str, *args: str) -> None:
        """Validate a syscall name against the dangerous syscall list."""
        check_syscall_allowed(syscall, *args)

    def track_process(self, pid: int, command: str) -> None:
        """Register a process for tracking. Raises if at concurrency limit."""
        self._tracker.track(pid, command)

    def check_and_track(self, pid: int, command: str, args: list[str] | None = None) -> None:
        """Atomic command validation + process reservation.

        Combines ``check_command()`` and ``track_process()`` to prevent
        TOCTOU races where multiple workers validate concurrently and
        then all attempt to track, exceeding ``max_processes``.
        """
        # Validate command first (stateless checks — can fail fast)
        self.check_command(command, args)

        # Reserve process slot atomically
        self._tracker.track(pid, command)

    def release_process(self, pid: int) -> None:
        """Mark a tracked process as terminated."""
        self._tracker.release(pid)

    def check_timeout(self, pid: int) -> None:
        """Check if a tracked process has exceeded its time limit."""
        self._tracker.check_timeout(pid)

    def check_all_timeouts(self) -> list[int]:
        """Check all tracked processes for timeouts."""
        return self._tracker.check_all_timeouts()

    @property
    def active_count(self) -> int:
        """Number of currently active tracked processes."""
        return self._tracker.active_count

    def list_processes(self) -> list[ProcessRecord]:
        """Return snapshot of tracked processes."""
        return self._tracker.list_processes()
