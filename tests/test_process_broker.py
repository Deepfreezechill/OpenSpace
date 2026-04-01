"""Tests for openspace.sandbox.process_broker — EPIC 2.4.

Covers:
- #99: Command allow/deny enforcement
- #100: Shell invocation control
- #101: Process tracking with limits and timeouts
- #102: Dangerous syscall / link command restriction
- #103: ProcessBroker integration and config-from-capability
"""

from __future__ import annotations

import time
import threading
import pytest

from openspace.sandbox.process_broker import (
    ProcessBroker,
    ProcessBrokerConfig,
    ProcessRecord,
    ProcessTracker,
    CommandBlockedError,
    CommandNotAllowedError,
    ShellNotAllowedError,
    ProcessLimitError,
    SyscallBlockedError,
    ExecutionTimeoutError,
    _extract_basename,
    _extract_arg_tokens,
    _strip_exe,
    check_command_blocked,
    check_command_allowed,
    check_shell_allowed,
    check_shell_command,
    check_syscall_allowed,
    check_link_command,
    _SHELL_BINARIES,
    _SHELL_WRAPPERS,
    _DANGEROUS_SYSCALLS,
    _LINK_COMMANDS,
)
from openspace.sandbox.leases import ProcessCapability, REQUIRED_BLOCKED_COMMANDS


# ═══════════════════════════════════════════════════════════════════════
# #99 — Command Allow/Deny
# ═══════════════════════════════════════════════════════════════════════


class TestExtractBasename:
    """Tests for _extract_basename path handling."""

    def test_bare_command(self) -> None:
        assert _extract_basename("python") == "python"

    def test_unix_path(self) -> None:
        assert _extract_basename("/usr/bin/python") == "python"

    def test_windows_path(self) -> None:
        assert _extract_basename("C:\\Windows\\System32\\cmd.exe") == "cmd.exe"

    def test_case_insensitive(self) -> None:
        assert _extract_basename("Python3") == "python3"

    def test_relative_path(self) -> None:
        assert _extract_basename("./scripts/run.sh") == "run.sh"

    def test_empty_string(self) -> None:
        assert _extract_basename("") == ""


class TestCommandBlocked:
    """Tests for command blocking enforcement."""

    def test_required_commands_always_blocked(self) -> None:
        """REQUIRED_BLOCKED_COMMANDS must be blocked even with empty list."""
        for cmd in REQUIRED_BLOCKED_COMMANDS:
            with pytest.raises(CommandBlockedError, match=cmd):
                check_command_blocked(cmd, [])

    def test_explicit_block(self) -> None:
        with pytest.raises(CommandBlockedError, match="curl"):
            check_command_blocked("curl", ["curl", "wget"])

    def test_full_path_blocked(self) -> None:
        """Full path to a blocked command must still be caught."""
        with pytest.raises(CommandBlockedError, match="rm"):
            check_command_blocked("/bin/rm", [])

    def test_windows_path_blocked(self) -> None:
        with pytest.raises(CommandBlockedError, match="rm"):
            check_command_blocked("C:\\Tools\\rm", [])

    def test_case_insensitive_blocking(self) -> None:
        with pytest.raises(CommandBlockedError):
            check_command_blocked("RM", [])

    def test_glob_pattern_blocking(self) -> None:
        with pytest.raises(CommandBlockedError, match="python"):
            check_command_blocked("python3.11", ["python*"])

    def test_allowed_command_passes(self) -> None:
        """Non-blocked commands should not raise."""
        check_command_blocked("python", [])

    def test_custom_plus_required_blocked(self) -> None:
        """Custom blocks are additive to required blocks."""
        with pytest.raises(CommandBlockedError, match="wget"):
            check_command_blocked("wget", ["wget"])
        with pytest.raises(CommandBlockedError, match="rm"):
            check_command_blocked("rm", ["wget"])


class TestCommandAllowed:
    """Tests for command allowlist enforcement."""

    def test_empty_allowlist_permits_all(self) -> None:
        """Empty allow list = open policy (anything non-blocked passes)."""
        check_command_allowed("anything", [])

    def test_allowlist_permits_match(self) -> None:
        check_command_allowed("python", ["python", "node"])

    def test_allowlist_rejects_nonmatch(self) -> None:
        with pytest.raises(CommandNotAllowedError, match="curl"):
            check_command_allowed("curl", ["python", "node"])

    def test_allowlist_glob(self) -> None:
        check_command_allowed("python3.11", ["python*"])

    def test_allowlist_case_insensitive(self) -> None:
        check_command_allowed("Python", ["python"])

    def test_allowlist_full_path(self) -> None:
        """Full path should match against basename in allowlist."""
        check_command_allowed("/usr/bin/python", ["python"])


# ═══════════════════════════════════════════════════════════════════════
# #100 — Shell Invocation Control
# ═══════════════════════════════════════════════════════════════════════


class TestShellControl:
    """Tests for shell invocation enforcement."""

    def test_shell_allowed_when_enabled(self) -> None:
        check_shell_allowed("bash", allow_shell=True)

    def test_bash_blocked_when_disabled(self) -> None:
        with pytest.raises(ShellNotAllowedError, match="bash"):
            check_shell_allowed("bash", allow_shell=False)

    def test_sh_blocked(self) -> None:
        with pytest.raises(ShellNotAllowedError):
            check_shell_allowed("sh", allow_shell=False)

    def test_cmd_exe_blocked(self) -> None:
        with pytest.raises(ShellNotAllowedError, match="cmd.exe"):
            check_shell_allowed("cmd.exe", allow_shell=False)

    def test_powershell_blocked(self) -> None:
        with pytest.raises(ShellNotAllowedError):
            check_shell_allowed("powershell", allow_shell=False)

    def test_pwsh_blocked(self) -> None:
        with pytest.raises(ShellNotAllowedError):
            check_shell_allowed("pwsh", allow_shell=False)

    def test_zsh_blocked(self) -> None:
        with pytest.raises(ShellNotAllowedError):
            check_shell_allowed("zsh", allow_shell=False)

    def test_full_path_shell_blocked(self) -> None:
        """Full path to shell must still be caught."""
        with pytest.raises(ShellNotAllowedError, match="bash"):
            check_shell_allowed("/bin/bash", allow_shell=False)

    def test_non_shell_command_allowed(self) -> None:
        """Non-shell commands pass even with allow_shell=False."""
        check_shell_allowed("python", allow_shell=False)

    def test_all_known_shells_blocked(self) -> None:
        """Every shell in _SHELL_BINARIES must be blocked when disabled."""
        for shell in _SHELL_BINARIES:
            with pytest.raises(ShellNotAllowedError):
                check_shell_allowed(shell, allow_shell=False)

    def test_shell_command_blocked_when_disabled(self) -> None:
        with pytest.raises(ShellNotAllowedError, match="allow_shell=False"):
            check_shell_command("echo hello && rm -rf /", allow_shell=False)

    def test_shell_command_allowed_when_enabled(self) -> None:
        check_shell_command("echo hello", allow_shell=True)

    def test_shell_command_truncated_in_error(self) -> None:
        """Long shell commands should be truncated in error messages."""
        long_cmd = "x" * 200
        with pytest.raises(ShellNotAllowedError) as exc_info:
            check_shell_command(long_cmd, allow_shell=False)
        assert len(str(exc_info.value)) < 250


# ═══════════════════════════════════════════════════════════════════════
# #102 — Dangerous Syscall / Link Command Restriction
# ═══════════════════════════════════════════════════════════════════════


class TestSyscallRestriction:
    """Tests for dangerous syscall blocking."""

    def test_link_syscall_blocked(self) -> None:
        with pytest.raises(SyscallBlockedError, match="link"):
            check_syscall_allowed("link", "/source", "/target")

    def test_symlink_blocked(self) -> None:
        with pytest.raises(SyscallBlockedError, match="symlink"):
            check_syscall_allowed("symlink", "/target", "/linkname")

    def test_linkat_blocked(self) -> None:
        with pytest.raises(SyscallBlockedError):
            check_syscall_allowed("linkat")

    def test_symlinkat_blocked(self) -> None:
        with pytest.raises(SyscallBlockedError):
            check_syscall_allowed("symlinkat")

    def test_rename_blocked(self) -> None:
        with pytest.raises(SyscallBlockedError):
            check_syscall_allowed("rename", "/old", "/new")

    def test_mount_blocked(self) -> None:
        with pytest.raises(SyscallBlockedError):
            check_syscall_allowed("mount", "/dev/sda1", "/mnt")

    def test_chroot_blocked(self) -> None:
        with pytest.raises(SyscallBlockedError):
            check_syscall_allowed("chroot", "/new_root")

    def test_mknod_blocked(self) -> None:
        with pytest.raises(SyscallBlockedError):
            check_syscall_allowed("mknod", "/dev/null", "c", "1", "3")

    def test_safe_syscall_allowed(self) -> None:
        """Normal syscalls must not be blocked."""
        check_syscall_allowed("read")
        check_syscall_allowed("write")
        check_syscall_allowed("open")
        check_syscall_allowed("close")

    def test_case_insensitive(self) -> None:
        with pytest.raises(SyscallBlockedError):
            check_syscall_allowed("LINK")

    def test_all_dangerous_syscalls_blocked(self) -> None:
        for syscall in _DANGEROUS_SYSCALLS:
            with pytest.raises(SyscallBlockedError):
                check_syscall_allowed(syscall)


class TestLinkCommandRestriction:
    """Tests for link-creating command restriction."""

    def test_ln_blocked(self) -> None:
        with pytest.raises(SyscallBlockedError, match="ln"):
            check_link_command("ln")

    def test_ln_full_path_blocked(self) -> None:
        with pytest.raises(SyscallBlockedError, match="ln"):
            check_link_command("/usr/bin/ln")

    def test_mklink_blocked(self) -> None:
        with pytest.raises(SyscallBlockedError, match="mklink"):
            check_link_command("mklink")

    def test_mount_command_blocked(self) -> None:
        with pytest.raises(SyscallBlockedError, match="mount"):
            check_link_command("mount")

    def test_all_link_commands_blocked(self) -> None:
        for cmd in _LINK_COMMANDS:
            with pytest.raises(SyscallBlockedError):
                check_link_command(cmd)

    def test_safe_command_passes(self) -> None:
        check_link_command("python")
        check_link_command("node")
        check_link_command("cat")


# ═══════════════════════════════════════════════════════════════════════
# #101 — Process Tracking
# ═══════════════════════════════════════════════════════════════════════


class TestProcessTracker:
    """Tests for ProcessTracker concurrency and timeout."""

    def test_track_and_release(self) -> None:
        tracker = ProcessTracker(max_processes=2, max_execution_time_s=300)
        tracker.track(100, "python")
        assert tracker.active_count == 1
        tracker.release(100)
        assert tracker.active_count == 0

    def test_process_limit_enforced(self) -> None:
        tracker = ProcessTracker(max_processes=2, max_execution_time_s=300)
        tracker.track(100, "python")
        tracker.track(101, "node")
        with pytest.raises(ProcessLimitError, match="2/2"):
            tracker.track(102, "ruby")

    def test_release_frees_slot(self) -> None:
        tracker = ProcessTracker(max_processes=1, max_execution_time_s=300)
        tracker.track(100, "python")
        tracker.release(100)
        tracker.track(101, "node")  # Should succeed after release
        assert tracker.active_count == 1

    def test_duplicate_pid_rejected(self) -> None:
        tracker = ProcessTracker(max_processes=5, max_execution_time_s=300)
        tracker.track(100, "python")
        with pytest.raises(ProcessLimitError, match="already tracked"):
            tracker.track(100, "python")

    def test_terminated_pid_reusable(self) -> None:
        tracker = ProcessTracker(max_processes=1, max_execution_time_s=300)
        tracker.track(100, "python")
        tracker.release(100)
        tracker.track(100, "node")  # Reuse after termination

    def test_timeout_detection(self) -> None:
        tracker = ProcessTracker(max_processes=5, max_execution_time_s=0)
        tracker.track(100, "slow")
        # max_execution_time_s=0 means immediate timeout
        with pytest.raises(ExecutionTimeoutError, match="slow"):
            tracker.check_timeout(100)

    def test_no_timeout_when_within_limit(self) -> None:
        tracker = ProcessTracker(max_processes=5, max_execution_time_s=3600)
        tracker.track(100, "fast")
        tracker.check_timeout(100)  # Should not raise

    def test_check_all_timeouts(self) -> None:
        tracker = ProcessTracker(max_processes=5, max_execution_time_s=0)
        tracker.track(100, "a")
        tracker.track(101, "b")
        tracker.track(102, "c")
        timed_out = tracker.check_all_timeouts()
        assert set(timed_out) == {100, 101, 102}

    def test_list_processes(self) -> None:
        tracker = ProcessTracker(max_processes=5, max_execution_time_s=300)
        tracker.track(100, "python")
        tracker.track(101, "node")
        processes = tracker.list_processes()
        assert len(processes) == 2
        assert {p.pid for p in processes} == {100, 101}

    def test_thread_safety(self) -> None:
        """Concurrent track/release must not corrupt state."""
        tracker = ProcessTracker(max_processes=100, max_execution_time_s=300)
        errors: list[Exception] = []

        def worker(start_pid: int) -> None:
            try:
                for i in range(10):
                    pid = start_pid + i
                    tracker.track(pid, f"cmd-{pid}")
                    tracker.release(pid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i * 100,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety violation: {errors}"
        assert tracker.active_count == 0

    def test_zero_max_processes(self) -> None:
        """max_processes=0 means no processes can be tracked."""
        tracker = ProcessTracker(max_processes=0, max_execution_time_s=300)
        with pytest.raises(ProcessLimitError, match="0/0"):
            tracker.track(100, "python")

    def test_release_unknown_pid_no_error(self) -> None:
        """Releasing an unknown PID should be a no-op."""
        tracker = ProcessTracker(max_processes=5, max_execution_time_s=300)
        tracker.release(999)  # Should not raise

    def test_timeout_unknown_pid_no_error(self) -> None:
        """Checking timeout for unknown PID should be a no-op."""
        tracker = ProcessTracker(max_processes=5, max_execution_time_s=300)
        tracker.check_timeout(999)  # Should not raise


# ═══════════════════════════════════════════════════════════════════════
# #103 — Config + Broker Integration
# ═══════════════════════════════════════════════════════════════════════


class TestProcessBrokerConfig:
    """Tests for ProcessBrokerConfig.from_capability."""

    def test_from_default_capability(self) -> None:
        cap = ProcessCapability()
        config = ProcessBrokerConfig.from_capability(cap)
        assert config.max_processes == 3
        assert config.max_execution_time_s == 300
        assert config.allow_shell is False

    def test_link_commands_merged_into_blocked(self) -> None:
        """Link commands must be auto-merged into blocked list."""
        cap = ProcessCapability()
        config = ProcessBrokerConfig.from_capability(cap)
        for cmd in _LINK_COMMANDS:
            assert cmd in config.blocked_commands

    def test_required_blocked_merged(self) -> None:
        """REQUIRED_BLOCKED_COMMANDS must be in blocked list."""
        cap = ProcessCapability()
        config = ProcessBrokerConfig.from_capability(cap)
        for cmd in REQUIRED_BLOCKED_COMMANDS:
            assert cmd in config.blocked_commands

    def test_custom_allowed_commands(self) -> None:
        cap = ProcessCapability(allowed_commands=["python", "node"])
        config = ProcessBrokerConfig.from_capability(cap)
        assert config.allowed_commands == ("python", "node")

    def test_config_is_frozen(self) -> None:
        cap = ProcessCapability()
        config = ProcessBrokerConfig.from_capability(cap)
        with pytest.raises(AttributeError):
            config.allow_shell = True  # type: ignore[misc]


class TestProcessBroker:
    """Integration tests for ProcessBroker."""

    @pytest.fixture
    def broker(self) -> ProcessBroker:
        cap = ProcessCapability(
            allowed_commands=["python", "node", "git"],
            max_processes=3,
            max_execution_time_s=300,
            allow_shell=False,
        )
        return ProcessBroker(ProcessBrokerConfig.from_capability(cap))

    @pytest.fixture
    def open_broker(self) -> ProcessBroker:
        """Broker with open policy (empty allowlist, shell disabled)."""
        cap = ProcessCapability(
            allowed_commands=[],
            max_processes=5,
            max_execution_time_s=600,
            allow_shell=False,
        )
        return ProcessBroker(ProcessBrokerConfig.from_capability(cap))

    def test_allowed_command_passes(self, broker: ProcessBroker) -> None:
        broker.check_command("python", ["script.py"])

    def test_blocked_command_rejected(self, broker: ProcessBroker) -> None:
        with pytest.raises(CommandBlockedError, match="rm"):
            broker.check_command("rm", ["-rf", "/"])

    def test_link_command_rejected(self, broker: ProcessBroker) -> None:
        with pytest.raises((CommandBlockedError, SyscallBlockedError)):
            broker.check_command("ln", ["-s", "/etc/passwd", "/tmp/link"])

    def test_shell_rejected(self, broker: ProcessBroker) -> None:
        with pytest.raises(ShellNotAllowedError):
            broker.check_command("bash", ["-c", "echo hello"])

    def test_unapproved_command_rejected(self, broker: ProcessBroker) -> None:
        with pytest.raises(CommandNotAllowedError, match="curl"):
            broker.check_command("curl", ["https://evil.com"])

    def test_open_policy_permits_non_blocked(self, open_broker: ProcessBroker) -> None:
        """With empty allowlist, non-blocked commands pass."""
        open_broker.check_command("python", [])
        open_broker.check_command("cat", [])
        open_broker.check_command("grep", [])

    def test_open_policy_still_blocks_dangerous(self, open_broker: ProcessBroker) -> None:
        with pytest.raises(CommandBlockedError):
            open_broker.check_command("rm", [])
        with pytest.raises((CommandBlockedError, SyscallBlockedError)):
            open_broker.check_command("ln", [])

    def test_shell_command_rejected(self, broker: ProcessBroker) -> None:
        with pytest.raises(ShellNotAllowedError):
            broker.check_shell("echo hello && rm -rf /")

    def test_syscall_rejected(self, broker: ProcessBroker) -> None:
        with pytest.raises(SyscallBlockedError):
            broker.check_syscall("symlink", "/target", "/linkname")

    def test_safe_syscall_allowed(self, broker: ProcessBroker) -> None:
        broker.check_syscall("read")
        broker.check_syscall("write")

    def test_process_lifecycle(self, broker: ProcessBroker) -> None:
        """Full process track → check → release cycle."""
        broker.track_process(100, "python")
        assert broker.active_count == 1
        broker.check_timeout(100)
        broker.release_process(100)
        assert broker.active_count == 0

    def test_process_limit(self, broker: ProcessBroker) -> None:
        broker.track_process(100, "python")
        broker.track_process(101, "node")
        broker.track_process(102, "git")
        with pytest.raises(ProcessLimitError):
            broker.track_process(103, "ruby")

    def test_list_processes(self, broker: ProcessBroker) -> None:
        broker.track_process(100, "python")
        broker.track_process(101, "node")
        processes = broker.list_processes()
        assert len(processes) == 2

    def test_full_path_command_validated(self, broker: ProcessBroker) -> None:
        """Full path to an allowed command should pass."""
        broker.check_command("/usr/bin/python", [])

    def test_full_path_blocked_command_rejected(self, broker: ProcessBroker) -> None:
        """Full path to a blocked command must be caught."""
        with pytest.raises(CommandBlockedError):
            broker.check_command("/usr/bin/rm", [])


class TestProcessBrokerWithShell:
    """Tests for ProcessBroker with allow_shell=True."""

    @pytest.fixture
    def shell_broker(self) -> ProcessBroker:
        cap = ProcessCapability(
            allowed_commands=[],
            max_processes=3,
            max_execution_time_s=300,
            allow_shell=True,
        )
        return ProcessBroker(ProcessBrokerConfig.from_capability(cap))

    def test_shell_allowed(self, shell_broker: ProcessBroker) -> None:
        shell_broker.check_command("bash", ["-c", "echo hello"])

    def test_shell_command_allowed(self, shell_broker: ProcessBroker) -> None:
        shell_broker.check_shell("echo hello")

    def test_blocked_still_enforced_with_shell(self, shell_broker: ProcessBroker) -> None:
        """Even with allow_shell=True, blocked commands are still blocked."""
        with pytest.raises(CommandBlockedError):
            shell_broker.check_command("rm", ["-rf", "/"])


# ═══════════════════════════════════════════════════════════════════════
# Security Regressions
# ═══════════════════════════════════════════════════════════════════════


class TestSecurityRegressions:
    """Regression tests for EPIC 2.2 deferred items and attack vectors."""

    def test_hardlink_escape_blocked(self) -> None:
        """EPIC 2.2 deferred: ln must be blocked to prevent hard-link jail escape."""
        cap = ProcessCapability()
        broker = ProcessBroker(ProcessBrokerConfig.from_capability(cap))
        with pytest.raises((CommandBlockedError, SyscallBlockedError)):
            broker.check_command("ln", ["/etc/passwd", "/sandbox/passwd"])

    def test_symlink_syscall_blocked(self) -> None:
        """symlink syscall must be blocked to prevent jail escape."""
        cap = ProcessCapability()
        broker = ProcessBroker(ProcessBrokerConfig.from_capability(cap))
        with pytest.raises(SyscallBlockedError):
            broker.check_syscall("symlink", "/etc/passwd", "/sandbox/link")

    def test_chroot_escape_blocked(self) -> None:
        """chroot syscall must be blocked."""
        cap = ProcessCapability()
        broker = ProcessBroker(ProcessBrokerConfig.from_capability(cap))
        with pytest.raises(SyscallBlockedError):
            broker.check_syscall("chroot", "/tmp/fake_root")

    def test_mount_escape_blocked(self) -> None:
        """mount syscall + command must both be blocked."""
        cap = ProcessCapability()
        broker = ProcessBroker(ProcessBrokerConfig.from_capability(cap))
        with pytest.raises(SyscallBlockedError):
            broker.check_syscall("mount", "/dev/sda1", "/mnt")
        with pytest.raises((CommandBlockedError, SyscallBlockedError)):
            broker.check_command("mount", ["/dev/sda1", "/mnt"])

    def test_t0_single_process_limit(self) -> None:
        """T0 allows max 1 process — broker must enforce this."""
        cap = ProcessCapability(max_processes=1, allow_shell=False)
        broker = ProcessBroker(ProcessBrokerConfig.from_capability(cap))
        broker.track_process(100, "python")
        with pytest.raises(ProcessLimitError):
            broker.track_process(101, "node")

    def test_mknod_device_creation_blocked(self) -> None:
        """mknod must be blocked to prevent device file creation."""
        cap = ProcessCapability()
        broker = ProcessBroker(ProcessBrokerConfig.from_capability(cap))
        with pytest.raises(SyscallBlockedError):
            broker.check_syscall("mknod", "/dev/evil", "c", "1", "3")

    def test_rename_across_jail_blocked(self) -> None:
        """rename/renameat must be blocked (can move files out of jail)."""
        cap = ProcessCapability()
        broker = ProcessBroker(ProcessBrokerConfig.from_capability(cap))
        with pytest.raises(SyscallBlockedError):
            broker.check_syscall("rename", "/sandbox/file", "/etc/file")
        with pytest.raises(SyscallBlockedError):
            broker.check_syscall("renameat2")

    def test_fusermount_blocked(self) -> None:
        """fusermount can bypass jail via FUSE mounts."""
        cap = ProcessCapability()
        broker = ProcessBroker(ProcessBrokerConfig.from_capability(cap))
        with pytest.raises((CommandBlockedError, SyscallBlockedError)):
            broker.check_command("fusermount", ["-u", "/mnt/fuse"])

    def test_pivot_root_blocked(self) -> None:
        """pivot_root must be blocked."""
        cap = ProcessCapability()
        broker = ProcessBroker(ProcessBrokerConfig.from_capability(cap))
        with pytest.raises(SyscallBlockedError):
            broker.check_syscall("pivot_root", "/new_root", "/old_root")

    def test_enforcement_order_deny_before_allow(self) -> None:
        """Blocked commands must be rejected even if in allowlist."""
        cap = ProcessCapability(
            allowed_commands=["rm", "python"],
            blocked_commands=["rm", "rmdir", "mkfs", "dd", "shutdown", "reboot", "kill", "pkill"],
        )
        broker = ProcessBroker(ProcessBrokerConfig.from_capability(cap))
        with pytest.raises(CommandBlockedError):
            broker.check_command("rm", [])
        # But python should still work
        broker.check_command("python", [])


# ═══════════════════════════════════════════════════════════════════════
# R1 Review Fixes — /8eyes + /collab findings
# ═══════════════════════════════════════════════════════════════════════


class TestR1ShellWrapperBypass:
    """R1 finding: env/busybox/sudo can invoke shells via args."""

    @pytest.fixture
    def broker(self) -> ProcessBroker:
        cap = ProcessCapability(allow_shell=False)
        return ProcessBroker(ProcessBrokerConfig.from_capability(cap))

    def test_env_bash_blocked(self, broker: ProcessBroker) -> None:
        """env bash -c 'id' must be blocked."""
        with pytest.raises(ShellNotAllowedError, match="wrapper"):
            broker.check_command("env", ["bash", "-c", "id"])

    def test_busybox_sh_blocked(self, broker: ProcessBroker) -> None:
        with pytest.raises(ShellNotAllowedError, match="wrapper"):
            broker.check_command("busybox", ["sh", "-c", "id"])

    def test_sudo_bash_blocked(self, broker: ProcessBroker) -> None:
        with pytest.raises(ShellNotAllowedError, match="wrapper"):
            broker.check_command("sudo", ["bash"])

    def test_nohup_zsh_blocked(self, broker: ProcessBroker) -> None:
        with pytest.raises(ShellNotAllowedError, match="wrapper"):
            broker.check_command("nohup", ["zsh", "-c", "evil"])

    def test_env_python_allowed(self, broker: ProcessBroker) -> None:
        """env python should pass (python is not a shell)."""
        broker.check_command("env", ["python", "script.py"])

    def test_env_no_args_allowed(self, broker: ProcessBroker) -> None:
        """env with no args should pass."""
        broker.check_command("env", [])

    def test_wrapper_with_blocked_cmd_in_args(self, broker: ProcessBroker) -> None:
        """env rm -rf / must be blocked (rm in args)."""
        with pytest.raises(CommandBlockedError, match="rm"):
            broker.check_command("env", ["rm", "-rf", "/"])


class TestR1ExeExtensionBypass:
    """R1 finding: rm.exe, shutdown.exe, ln.exe bypass block list."""

    def test_rm_exe_blocked(self) -> None:
        with pytest.raises(CommandBlockedError, match="rm"):
            check_command_blocked("rm.exe", [])

    def test_shutdown_exe_blocked(self) -> None:
        with pytest.raises(CommandBlockedError):
            check_command_blocked("shutdown.exe", [])

    def test_ln_exe_blocked(self) -> None:
        """ln.exe must be caught by either block list or link command check."""
        with pytest.raises((CommandBlockedError, SyscallBlockedError)):
            check_command_blocked("ln.exe", list(_LINK_COMMANDS))

    def test_kill_exe_blocked(self) -> None:
        with pytest.raises(CommandBlockedError):
            check_command_blocked("kill.exe", [])

    def test_link_exe_blocked(self) -> None:
        with pytest.raises(SyscallBlockedError):
            check_link_command("ln.exe")

    def test_mklink_exe_blocked(self) -> None:
        with pytest.raises(SyscallBlockedError):
            check_link_command("mklink.exe")

    def test_strip_exe_helper(self) -> None:
        assert _strip_exe("rm.exe") == "rm"
        assert _strip_exe("python") == "python"
        assert _strip_exe("cmd.exe") == "cmd"


class TestR1QuotedPathBypass:
    """R1 finding: quoted paths bypass basename extraction."""

    def test_quoted_cmd_exe_blocked(self) -> None:
        with pytest.raises(ShellNotAllowedError):
            check_shell_allowed('"C:\\Windows\\System32\\cmd.exe"', False)

    def test_single_quoted_bash_blocked(self) -> None:
        with pytest.raises(ShellNotAllowedError):
            check_shell_allowed("'/bin/bash'", False)

    def test_quoted_rm_blocked(self) -> None:
        with pytest.raises(CommandBlockedError):
            check_command_blocked('"rm"', [])

    def test_quoted_extract_basename(self) -> None:
        assert _extract_basename('"C:\\Windows\\cmd.exe"') == "cmd.exe"
        assert _extract_basename("'/usr/bin/bash'") == "bash"
        assert _extract_basename('"rm"') == "rm"


class TestR1MutableProcessRecord:
    """R1 finding: list_processes returned mutable internal records."""

    def test_list_returns_copies(self) -> None:
        tracker = ProcessTracker(max_processes=5, max_execution_time_s=300)
        tracker.track(100, "python")
        processes = tracker.list_processes()
        # Mutating the copy should NOT affect internal state
        processes[0].terminated = True
        assert tracker.active_count == 1  # Still active internally


class TestR1AtomicCheckAndTrack:
    """R1 finding: TOCTOU between check_command and track_process."""

    def test_check_and_track_success(self) -> None:
        cap = ProcessCapability(max_processes=2)
        broker = ProcessBroker(ProcessBrokerConfig.from_capability(cap))
        broker.check_and_track(100, "python", ["script.py"])
        assert broker.active_count == 1

    def test_check_and_track_blocked_command(self) -> None:
        cap = ProcessCapability(max_processes=2)
        broker = ProcessBroker(ProcessBrokerConfig.from_capability(cap))
        with pytest.raises(CommandBlockedError):
            broker.check_and_track(100, "rm", ["-rf", "/"])
        assert broker.active_count == 0  # Not tracked

    def test_check_and_track_limit(self) -> None:
        cap = ProcessCapability(max_processes=1)
        broker = ProcessBroker(ProcessBrokerConfig.from_capability(cap))
        broker.check_and_track(100, "python", [])
        with pytest.raises(ProcessLimitError):
            broker.check_and_track(101, "python", [])

    def test_check_and_track_concurrent_safety(self) -> None:
        """Concurrent check_and_track must not exceed max_processes."""
        cap = ProcessCapability(max_processes=5)
        broker = ProcessBroker(ProcessBrokerConfig.from_capability(cap))
        errors: list[Exception] = []
        successes = []
        lock = threading.Lock()

        def worker(pid: int) -> None:
            try:
                broker.check_and_track(pid, "python", [])
                with lock:
                    successes.append(pid)
            except ProcessLimitError:
                pass
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Unexpected errors: {errors}"
        assert len(successes) <= 5, f"Exceeded max_processes: {len(successes)}"
        assert broker.active_count <= 5


class TestR1MissingShells:
    """R1 finding: ash, rbash not in shell list."""

    def test_ash_blocked(self) -> None:
        with pytest.raises(ShellNotAllowedError):
            check_shell_allowed("ash", False)

    def test_rbash_blocked(self) -> None:
        with pytest.raises(ShellNotAllowedError):
            check_shell_allowed("rbash", False)

    def test_rksh_blocked(self) -> None:
        with pytest.raises(ShellNotAllowedError):
            check_shell_allowed("rksh", False)


# ═══════════════════════════════════════════════════════════════════════
# R2 Review Fixes — /8eyes + Sonnet findings
# ═══════════════════════════════════════════════════════════════════════


class TestR2ConcatenatedShellArg:
    """R2 finding: env -S 'sh -c id' bypasses shell detection."""

    @pytest.fixture
    def broker(self) -> ProcessBroker:
        cap = ProcessCapability(allow_shell=False)
        return ProcessBroker(ProcessBrokerConfig.from_capability(cap))

    def test_env_S_sh_blocked(self, broker: ProcessBroker) -> None:
        """env -S 'sh -c id' — shell name embedded in single arg."""
        with pytest.raises(ShellNotAllowedError):
            broker.check_command("env", ["-S", "sh -c id"])

    def test_env_S_bash_blocked(self, broker: ProcessBroker) -> None:
        with pytest.raises(ShellNotAllowedError):
            broker.check_command("env", ["-S", "bash -c whoami"])

    def test_busybox_concatenated_sh(self, broker: ProcessBroker) -> None:
        with pytest.raises(ShellNotAllowedError):
            broker.check_command("busybox", ["sh -c id"])


class TestR2EnvVarFalsePositive:
    """R2 finding: env SHELL=/bin/bash python was false-positive."""

    @pytest.fixture
    def broker(self) -> ProcessBroker:
        cap = ProcessCapability(allow_shell=False)
        return ProcessBroker(ProcessBrokerConfig.from_capability(cap))

    def test_env_var_assignment_not_flagged(self, broker: ProcessBroker) -> None:
        """env SHELL=/bin/bash python should pass (SHELL=... is env var, not cmd)."""
        broker.check_command("env", ["SHELL=/bin/bash", "python"])

    def test_env_var_path_not_flagged(self, broker: ProcessBroker) -> None:
        """env PATH=/usr/bin node should pass."""
        broker.check_command("env", ["PATH=/usr/bin", "node"])

    def test_env_var_but_shell_arg_still_blocked(self, broker: ProcessBroker) -> None:
        """env MYVAR=1 bash should block on bash (the actual command)."""
        with pytest.raises(ShellNotAllowedError, match="wrapper"):
            broker.check_command("env", ["MYVAR=1", "bash"])


class TestR2MissingWrappers:
    """R2 finding: time, stdbuf, chpst bypass wrapper detection."""

    @pytest.fixture
    def broker(self) -> ProcessBroker:
        cap = ProcessCapability(allow_shell=False)
        return ProcessBroker(ProcessBrokerConfig.from_capability(cap))

    def test_time_bash_blocked(self, broker: ProcessBroker) -> None:
        with pytest.raises(ShellNotAllowedError, match="wrapper"):
            broker.check_command("time", ["bash", "-c", "id"])

    def test_stdbuf_bash_blocked(self, broker: ProcessBroker) -> None:
        with pytest.raises(ShellNotAllowedError, match="wrapper"):
            broker.check_command("stdbuf", ["-o0", "bash", "-c", "id"])

    def test_chpst_bash_blocked(self, broker: ProcessBroker) -> None:
        with pytest.raises(ShellNotAllowedError, match="wrapper"):
            broker.check_command("chpst", ["bash", "-c", "id"])

    def test_watch_sh_blocked(self, broker: ProcessBroker) -> None:
        with pytest.raises(ShellNotAllowedError, match="wrapper"):
            broker.check_command("watch", ["sh", "-c", "id"])


class TestR2DoubleExeExtension:
    """R2 finding: rm.exe.exe bypasses _strip_exe."""

    def test_double_exe_stripped(self) -> None:
        assert _strip_exe("rm.exe.exe") == "rm"

    def test_triple_exe_stripped(self) -> None:
        assert _strip_exe("rm.exe.exe.exe") == "rm"

    def test_double_exe_blocked(self) -> None:
        with pytest.raises(CommandBlockedError):
            check_command_blocked("rm.exe.exe", [])

    def test_double_exe_link_blocked(self) -> None:
        with pytest.raises(SyscallBlockedError):
            check_link_command("ln.exe.exe")


class TestR2ConcatenatedBlockedArgs:
    """R2: blocked commands embedded in space-concatenated args."""

    @pytest.fixture
    def broker(self) -> ProcessBroker:
        cap = ProcessCapability(allow_shell=False)
        return ProcessBroker(ProcessBrokerConfig.from_capability(cap))

    def test_env_S_rm_blocked(self, broker: ProcessBroker) -> None:
        """env -S 'rm -rf /' — rm embedded in concatenated arg."""
        with pytest.raises(CommandBlockedError, match="rm"):
            broker.check_command("env", ["-S", "rm -rf /"])


# ═══════════════════════════════════════════════════════════════════════
# R3 Review Fixes — /collab findings
# ═══════════════════════════════════════════════════════════════════════


class TestR3AllowlistWrapperBypass:
    """R3 finding: env in allowlist lets non-allowed commands through."""

    def test_env_curl_blocked_by_allowlist(self) -> None:
        """With allowed=[env, python], env curl must be rejected."""
        cap = ProcessCapability(allowed_commands=["env", "python"])
        broker = ProcessBroker(ProcessBrokerConfig.from_capability(cap))
        with pytest.raises(CommandNotAllowedError, match="curl"):
            broker.check_command("env", ["curl", "https://example.com"])

    def test_env_python_allowed(self) -> None:
        """env python should pass when both are in allowlist."""
        cap = ProcessCapability(allowed_commands=["env", "python"])
        broker = ProcessBroker(ProcessBrokerConfig.from_capability(cap))
        broker.check_command("env", ["python", "script.py"])

    def test_sudo_curl_blocked(self) -> None:
        """sudo curl must be rejected when curl not in allowlist."""
        cap = ProcessCapability(
            allowed_commands=["sudo", "python"],
            allow_shell=False,
        )
        broker = ProcessBroker(ProcessBrokerConfig.from_capability(cap))
        with pytest.raises(CommandNotAllowedError, match="curl"):
            broker.check_command("sudo", ["curl", "https://evil.com"])

    def test_timeout_python_allowed(self) -> None:
        """timeout python should pass."""
        cap = ProcessCapability(allowed_commands=["timeout", "python"])
        broker = ProcessBroker(ProcessBrokerConfig.from_capability(cap))
        broker.check_command("timeout", ["30", "python", "script.py"])


class TestR3FlagEmbeddedBypass:
    """R3 finding: -Sbash, --split-string=bash, foo=bar/rm bypass scanning."""

    @pytest.fixture
    def broker(self) -> ProcessBroker:
        cap = ProcessCapability(allow_shell=False)
        return ProcessBroker(ProcessBrokerConfig.from_capability(cap))

    def test_Sbash_flag_blocked(self, broker: ProcessBroker) -> None:
        """env -Sbash -c id — shell name in flag value."""
        with pytest.raises(ShellNotAllowedError):
            broker.check_command("env", ["-Sbash", "-c", "id"])

    def test_split_string_bash_blocked(self, broker: ProcessBroker) -> None:
        """env --split-string=bash -c id — shell in long flag value."""
        with pytest.raises(ShellNotAllowedError):
            broker.check_command("env", ["--split-string=bash", "-c", "id"])

    def test_env_var_path_rm_after_sentinel_blocked(self, broker: ProcessBroker) -> None:
        """env -- foo=bar/rm — after sentinel, treated as positional, rm caught."""
        with pytest.raises(CommandBlockedError):
            broker.check_command("env", ["--", "foo=bar/rm", "-rf", "/"])

    def test_split_string_rm_blocked(self, broker: ProcessBroker) -> None:
        """env --split-string=rm -rf / — blocked cmd in long flag value."""
        with pytest.raises(CommandBlockedError):
            broker.check_command("env", ["--split-string=rm", "-rf", "/"])


class TestR3ExtractArgTokens:
    """Tests for the _extract_arg_tokens helper."""

    def test_plain_args(self) -> None:
        assert _extract_arg_tokens(["bash", "script.py"]) == ["bash", "script.py"]

    def test_flags_skipped(self) -> None:
        assert _extract_arg_tokens(["-c", "--verbose"]) == []

    def test_short_flag_value_extracted(self) -> None:
        tokens = _extract_arg_tokens(["-Sbash -c id"])
        assert "bash" in tokens

    def test_long_flag_value_extracted(self) -> None:
        tokens = _extract_arg_tokens(["--split-string=bash -c id"])
        assert "bash" in tokens

    def test_env_var_value_skipped(self) -> None:
        tokens = _extract_arg_tokens(["FOO=bar/bash"])
        assert tokens == []  # Env vars are skipped entirely

    def test_sentinel_skipped(self) -> None:
        assert _extract_arg_tokens(["--", "bash"]) == ["bash"]

    def test_sentinel_makes_env_var_positional(self) -> None:
        """After --, FOO=bar is treated as positional (not env var)."""
        tokens = _extract_arg_tokens(["--", "FOO=bar/rm"])
        assert "FOO=bar/rm" in tokens

    def test_space_split(self) -> None:
        tokens = _extract_arg_tokens(["sh -c id"])
        assert "sh" in tokens


# ── R4 Regressions (false-positive + find/parallel wrapper) ──────────


class TestR4NonWrapperArgSafe:
    """Non-wrapper commands must NOT have args scanned for blocked commands.

    Fixes false-positive: `git checkout rm` wrongly blocking on `rm` in args.
    """

    def test_git_checkout_rm_allowed(self) -> None:
        broker = ProcessBroker(ProcessBrokerConfig(
            allowed_commands=("git",),
            blocked_commands=("rm",),
            max_processes=10, max_execution_time_s=300, allow_shell=False,
        ))
        broker.check_command("git", ["checkout", "rm"])

    def test_cat_file_named_reboot(self) -> None:
        broker = ProcessBroker(ProcessBrokerConfig(
            allowed_commands=("cat",),
            blocked_commands=("reboot",),
            max_processes=10, max_execution_time_s=300, allow_shell=False,
        ))
        broker.check_command("cat", ["/tmp/reboot"])

    def test_grep_pattern_shutdown(self) -> None:
        broker = ProcessBroker(ProcessBrokerConfig(
            allowed_commands=("grep",),
            blocked_commands=("shutdown",),
            max_processes=10, max_execution_time_s=300, allow_shell=False,
        ))
        broker.check_command("grep", ["shutdown", "log.txt"])


class TestR4WrapperArgScanStillWorks:
    """Wrappers must still scan args for blocked commands."""

    def test_env_rm_still_blocked(self) -> None:
        broker = ProcessBroker(ProcessBrokerConfig(
            allowed_commands=(),
            blocked_commands=("rm",),
            max_processes=10, max_execution_time_s=300, allow_shell=True,
        ))
        with pytest.raises(CommandBlockedError):
            broker.check_command("env", ["rm", "-rf", "/"])

    def test_sudo_reboot_still_blocked(self) -> None:
        broker = ProcessBroker(ProcessBrokerConfig(
            allowed_commands=(),
            blocked_commands=("reboot",),
            max_processes=10, max_execution_time_s=300, allow_shell=True,
        ))
        with pytest.raises(CommandBlockedError):
            broker.check_command("sudo", ["reboot"])


class TestR4FindParallelWrapper:
    """find and parallel are wrappers — their -exec args get scanned."""

    def test_find_exec_sh_blocked(self) -> None:
        broker = ProcessBroker(ProcessBrokerConfig(
            allowed_commands=(), blocked_commands=(),
            max_processes=10, max_execution_time_s=300, allow_shell=False,
        ))
        with pytest.raises(ShellNotAllowedError):
            broker.check_command("find", [".", "-exec", "sh", "-c", "id", ";"])

    def test_find_exec_rm_blocked(self) -> None:
        broker = ProcessBroker(ProcessBrokerConfig(
            allowed_commands=(),
            blocked_commands=("rm",),
            max_processes=10, max_execution_time_s=300, allow_shell=True,
        ))
        with pytest.raises(CommandBlockedError):
            broker.check_command("find", [".", "-exec", "rm", "-rf", "{}", ";"])

    def test_parallel_sh_blocked(self) -> None:
        broker = ProcessBroker(ProcessBrokerConfig(
            allowed_commands=(), blocked_commands=(),
            max_processes=10, max_execution_time_s=300, allow_shell=False,
        ))
        with pytest.raises(ShellNotAllowedError):
            broker.check_command("parallel", ["sh", "-c", "echo {}"])

    def test_find_exec_allowed_cmd_passes(self) -> None:
        broker = ProcessBroker(ProcessBrokerConfig(
            allowed_commands=("find", "echo"),
            blocked_commands=(),
            max_processes=10, max_execution_time_s=300, allow_shell=False,
        ))
        broker.check_command("find", [".", "-exec", "echo", "{}", ";"])


class TestR4AllowlistFlagEmbedded:
    """Flag-embedded commands must be checked against the allowlist."""

    def test_env_split_string_curl_blocked(self) -> None:
        broker = ProcessBroker(ProcessBrokerConfig(
            allowed_commands=("env", "python"),
            blocked_commands=(),
            max_processes=10, max_execution_time_s=300, allow_shell=True,
        ))
        with pytest.raises(CommandNotAllowedError):
            broker.check_command("env", ["--split-string=curl https://evil.com"])

    def test_env_short_flag_curl_blocked(self) -> None:
        """env -Scurl must also be caught by allowlist."""
        broker = ProcessBroker(ProcessBrokerConfig(
            allowed_commands=("env", "python"),
            blocked_commands=(),
            max_processes=10, max_execution_time_s=300, allow_shell=True,
        ))
        with pytest.raises(CommandNotAllowedError):
            broker.check_command("env", ["-Scurl"])

    def test_env_split_string_allowed_passes(self) -> None:
        broker = ProcessBroker(ProcessBrokerConfig(
            allowed_commands=("env", "python"),
            blocked_commands=(),
            max_processes=10, max_execution_time_s=300, allow_shell=True,
        ))
        broker.check_command("env", ["--split-string=python script.py"])


class TestR4ExeAllowlistMismatch:
    """python.exe should match allowlist entry 'python'."""

    def test_exe_matches_allowed(self) -> None:
        check_command_allowed("python.exe", ["python"])

    def test_double_exe_matches_allowed(self) -> None:
        check_command_allowed("cmd.exe.exe", ["cmd"])

    def test_non_matching_still_blocked(self) -> None:
        with pytest.raises(CommandNotAllowedError):
            check_command_allowed("curl.exe", ["python"])


class TestR4PortShape:
    """ProcessBrokerPort.active_count must be a property."""

    def test_active_count_is_property(self) -> None:
        cap = ProcessCapability()
        config = ProcessBrokerConfig.from_capability(cap)
        broker = ProcessBroker(config)
        # Must be accessible as property, not method call
        assert broker.active_count == 0


# ── R5 Regressions (flock, multi-exec, shell flags, glob patterns) ───


class TestR5FlockWrapper:
    """flock must be treated as a wrapper."""

    def test_flock_bash_blocked(self) -> None:
        broker = ProcessBroker(ProcessBrokerConfig(
            allowed_commands=(), blocked_commands=(),
            max_processes=10, max_execution_time_s=300, allow_shell=False,
        ))
        with pytest.raises(ShellNotAllowedError):
            broker.check_command("flock", ["/tmp/lock", "bash", "-c", "id"])

    def test_flock_allowed_cmd_passes(self) -> None:
        """flock's first positional arg is a lock file — second is the command."""
        broker = ProcessBroker(ProcessBrokerConfig(
            allowed_commands=("flock", "echo", "lock"),
            blocked_commands=(),
            max_processes=10, max_execution_time_s=300, allow_shell=False,
        ))
        broker.check_command("flock", ["/tmp/lock", "echo", "hello"])


class TestR5MultiExecAllowlist:
    """Multi-exec: all -exec commands are allowlist-checked via pass C."""

    def test_second_exec_not_allowed(self) -> None:
        """Second -exec with non-allowed command is caught by pass C."""
        broker = ProcessBroker(ProcessBrokerConfig(
            allowed_commands=("find", "echo"),
            blocked_commands=(),
            max_processes=10, max_execution_time_s=300, allow_shell=False,
        ))
        with pytest.raises(CommandNotAllowedError):
            broker.check_command("find", [
                ".", "-exec", "echo", "{}", ";",
                "-exec", "curl", "https://evil.example", ";",
            ])

    def test_second_exec_blocked_cmd_caught(self) -> None:
        """Blocked-command check also scans ALL wrapper args."""
        broker = ProcessBroker(ProcessBrokerConfig(
            allowed_commands=("find", "echo"),
            blocked_commands=("curl",),
            max_processes=10, max_execution_time_s=300, allow_shell=False,
        ))
        with pytest.raises(CommandBlockedError):
            broker.check_command("find", [
                ".", "-exec", "echo", "{}", ";",
                "-exec", "curl", "https://evil.example", ";",
            ])

    def test_second_exec_shell_caught(self) -> None:
        """Shell check scans ALL wrapper args — second -exec sh blocked."""
        broker = ProcessBroker(ProcessBrokerConfig(
            allowed_commands=("find", "echo"),
            blocked_commands=(),
            max_processes=10, max_execution_time_s=300, allow_shell=False,
        ))
        with pytest.raises(ShellNotAllowedError):
            broker.check_command("find", [
                ".", "-exec", "echo", "{}", ";",
                "-exec", "sh", "-c", "id", ";",
            ])

    def test_all_exec_allowed_passes(self) -> None:
        broker = ProcessBroker(ProcessBrokerConfig(
            allowed_commands=("find", "echo", "grep"),
            blocked_commands=(),
            max_processes=10, max_execution_time_s=300, allow_shell=False,
        ))
        broker.check_command("find", [
            ".", "-exec", "echo", "{}", ";",
            "-exec", "grep", "pattern", "{}", ";",
        ])


class TestR5ShellInvokingFlags:
    """sudo -s, su -, doas -s must be blocked when allow_shell=False."""

    def test_sudo_dash_s(self) -> None:
        broker = ProcessBroker(ProcessBrokerConfig(
            allowed_commands=("sudo",), blocked_commands=(),
            max_processes=10, max_execution_time_s=300, allow_shell=False,
        ))
        with pytest.raises(ShellNotAllowedError):
            broker.check_command("sudo", ["-s"])

    def test_sudo_dash_i(self) -> None:
        broker = ProcessBroker(ProcessBrokerConfig(
            allowed_commands=("sudo",), blocked_commands=(),
            max_processes=10, max_execution_time_s=300, allow_shell=False,
        ))
        with pytest.raises(ShellNotAllowedError):
            broker.check_command("sudo", ["-i"])

    def test_su_dash(self) -> None:
        broker = ProcessBroker(ProcessBrokerConfig(
            allowed_commands=("su",), blocked_commands=(),
            max_processes=10, max_execution_time_s=300, allow_shell=False,
        ))
        with pytest.raises(ShellNotAllowedError):
            broker.check_command("su", ["-"])

    def test_doas_dash_s(self) -> None:
        broker = ProcessBroker(ProcessBrokerConfig(
            allowed_commands=("doas",), blocked_commands=(),
            max_processes=10, max_execution_time_s=300, allow_shell=False,
        ))
        with pytest.raises(ShellNotAllowedError):
            broker.check_command("doas", ["-s"])

    def test_sudo_dash_s_allowed_when_shell_true(self) -> None:
        broker = ProcessBroker(ProcessBrokerConfig(
            allowed_commands=("sudo",), blocked_commands=(),
            max_processes=10, max_execution_time_s=300, allow_shell=True,
        ))
        broker.check_command("sudo", ["-s"])

    def test_sudo_combined_si_blocked(self) -> None:
        """Combined short flags: -si means -s + -i, both shell-invoking."""
        broker = ProcessBroker(ProcessBrokerConfig(
            allowed_commands=("sudo",), blocked_commands=(),
            max_processes=10, max_execution_time_s=300, allow_shell=False,
        ))
        with pytest.raises(ShellNotAllowedError):
            broker.check_command("sudo", ["-si"])

    def test_sudo_combined_uis_blocked(self) -> None:
        """Even with other flags mixed in, -s is detected."""
        broker = ProcessBroker(ProcessBrokerConfig(
            allowed_commands=("sudo",), blocked_commands=(),
            max_processes=10, max_execution_time_s=300, allow_shell=False,
        ))
        with pytest.raises(ShellNotAllowedError):
            broker.check_command("sudo", ["-uis"])


class TestR5GlobPatternSkip:
    """Glob patterns (find -name *.py) should not be checked as commands."""

    def test_find_name_glob_not_blocked(self) -> None:
        broker = ProcessBroker(ProcessBrokerConfig(
            allowed_commands=("find", "echo"),
            blocked_commands=(),
            max_processes=10, max_execution_time_s=300, allow_shell=False,
        ))
        broker.check_command("find", [".", "-name", "*.py", "-exec", "echo", "{}", ";"])


# ── R7 Regressions (su -c, runuser -c shell bypass) ─────────────────


class TestR7SuRunuserCommand:
    """su -c and runuser -c invoke a shell — must block when allow_shell=False."""

    def test_su_dash_c_blocked(self) -> None:
        broker = ProcessBroker(ProcessBrokerConfig(
            allowed_commands=("su",), blocked_commands=(),
            max_processes=10, max_execution_time_s=300, allow_shell=False,
        ))
        with pytest.raises(ShellNotAllowedError):
            broker.check_command("su", ["-c", "id"])

    def test_su_command_flag_blocked(self) -> None:
        broker = ProcessBroker(ProcessBrokerConfig(
            allowed_commands=("su",), blocked_commands=(),
            max_processes=10, max_execution_time_s=300, allow_shell=False,
        ))
        with pytest.raises(ShellNotAllowedError):
            broker.check_command("su", ["--command", "id"])

    def test_runuser_dash_c_blocked(self) -> None:
        broker = ProcessBroker(ProcessBrokerConfig(
            allowed_commands=("runuser",), blocked_commands=(),
            max_processes=10, max_execution_time_s=300, allow_shell=False,
        ))
        with pytest.raises(ShellNotAllowedError):
            broker.check_command("runuser", ["-c", "id"])

    def test_su_dash_c_allowed_when_shell_true(self) -> None:
        broker = ProcessBroker(ProcessBrokerConfig(
            allowed_commands=("su", "id"), blocked_commands=(),
            max_processes=10, max_execution_time_s=300, allow_shell=True,
        ))
        broker.check_command("su", ["-c", "id"])
