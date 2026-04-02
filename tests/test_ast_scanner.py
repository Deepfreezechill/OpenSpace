"""Comprehensive tests for the AST dangerous-API scanner.

Covers every pattern from blocklist.yml, severity levels,
the check_code_safety() integration function, blocklist loading,
and edge cases (nested functions, classes, lambdas, comprehensions).
"""

from __future__ import annotations

import textwrap

import pytest

from openspace.security import check_code_safety
from openspace.security.ast_scanner import (
    Finding,
    Severity,
    load_blocklist,
    scan_code,
    scan_file,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _names(findings: list[Finding]) -> set[str]:
    """Return set of pattern_name values from findings."""
    return {f.pattern_name for f in findings}


def _severities(findings: list[Finding]) -> set[Severity]:
    """Return set of severity values from findings."""
    return {f.severity for f in findings}


# ---------------------------------------------------------------------------
# Issue #18 — Core pattern detection
# ---------------------------------------------------------------------------


class TestDangerousPatterns:
    """Each dangerous API from the spec MUST be detected."""

    @pytest.mark.parametrize(
        "code,expected_pattern",
        [
            ("eval('1+1')", "eval"),
            ("exec('pass')", "exec"),
            ("result = eval(input())", "eval"),
        ],
    )
    def test_eval_exec(self, code, expected_pattern):
        findings = scan_code(code)
        assert expected_pattern in _names(findings)

    @pytest.mark.parametrize(
        "code,expected_pattern",
        [
            ("import os; os.system('ls')", "os_system"),
            ("import os; os.popen('ls')", "os_popen"),
        ],
    )
    def test_os_command_execution(self, code, expected_pattern):
        findings = scan_code(code)
        assert expected_pattern in _names(findings)

    @pytest.mark.parametrize(
        "code",
        [
            "import subprocess; subprocess.run(['ls'])",
            "import subprocess; subprocess.Popen(['ls'])",
            "import subprocess; subprocess.call(['ls'])",
            "import subprocess; subprocess.check_output(['ls'])",
        ],
    )
    def test_subprocess(self, code):
        findings = scan_code(code)
        assert "subprocess" in _names(findings) or "subprocess_import" in _names(findings)

    def test_dynamic_import(self):
        findings = scan_code("mod = __import__('os')")
        assert "dynamic_import" in _names(findings)

    @pytest.mark.parametrize(
        "code",
        [
            "import socket; socket.socket()",
            "import socket; s = socket.create_connection(('host', 80))",
        ],
    )
    def test_socket(self, code):
        findings = scan_code(code)
        assert "socket" in _names(findings) or "socket_import" in _names(findings)

    @pytest.mark.parametrize(
        "code",
        [
            "import ctypes; ctypes.cdll.LoadLibrary('libc.so')",
            "import ctypes; ctypes.CDLL('libc.so.6')",
        ],
    )
    def test_ctypes(self, code):
        findings = scan_code(code)
        assert "ctypes" in _names(findings) or "ctypes_import" in _names(findings)

    def test_os_environ_attribute(self):
        code = "import os\nval = os.environ"
        findings = scan_code(code)
        assert "env_access" in _names(findings)

    def test_os_getenv(self):
        findings = scan_code("import os; os.getenv('SECRET')")
        assert "env_getenv" in _names(findings)

    @pytest.mark.parametrize("path", ["/proc/self/environ", "/etc/passwd", "/etc/shadow"])
    def test_sensitive_file_open(self, path):
        findings = scan_code(f"f = open('{path}')")
        assert "sensitive_file_open" in _names(findings)

    def test_compile_exec_mode(self):
        code = "code_obj = compile('pass', '<string>', 'exec')"
        findings = scan_code(code)
        assert "compile_exec" in _names(findings)

    def test_compile_eval_mode_not_flagged(self):
        """compile() with 'eval' mode is NOT flagged by compile_exec pattern."""
        code = "code_obj = compile('1+1', '<string>', 'eval')"
        findings = scan_code(code)
        # compile_exec should not fire (mode is 'eval', not 'exec')
        compile_exec_findings = [f for f in findings if f.pattern_name == "compile_exec"]
        assert len(compile_exec_findings) == 0

    def test_getattr_on_module(self):
        code = "import os\ngetattr(os, 'system')('ls')"
        findings = scan_code(code)
        assert "getattr_injection" in _names(findings)

    def test_setattr_on_module(self):
        code = "import os\nsetattr(os, 'foo', 'bar')"
        findings = scan_code(code)
        assert "setattr_injection" in _names(findings)


# ---------------------------------------------------------------------------
# Safe code — no findings
# ---------------------------------------------------------------------------


class TestSafeCode:
    """Safe, everyday Python should produce zero findings."""

    @pytest.mark.parametrize(
        "code",
        [
            "x = 1 + 2",
            "def greet(name): return f'Hello, {name}'",
            "data = [i**2 for i in range(10)]",
            "import json; json.loads('{}')",
            "from pathlib import Path; p = Path('.')",
            "class Foo:\n    def bar(self): return 42",
            "open('myfile.txt', 'r')",  # non-sensitive path
            "compile('1+1', '<string>', 'eval')",  # eval mode, not exec
        ],
    )
    def test_safe_code_clean(self, code):
        findings = scan_code(code)
        # Filter out MEDIUM / informational — only CRITICAL/HIGH matter
        serious = [f for f in findings if f.severity in (Severity.CRITICAL, Severity.HIGH)]
        assert serious == []


# ---------------------------------------------------------------------------
# Severity levels
# ---------------------------------------------------------------------------


class TestSeverityLevels:
    """Verify severity assignments match the blocklist."""

    def test_eval_is_critical(self):
        findings = scan_code("eval('x')")
        evals = [f for f in findings if f.pattern_name == "eval"]
        assert all(f.severity == Severity.CRITICAL for f in evals)

    def test_exec_is_critical(self):
        findings = scan_code("exec('x')")
        execs = [f for f in findings if f.pattern_name == "exec"]
        assert all(f.severity == Severity.CRITICAL for f in execs)

    def test_os_system_is_critical(self):
        findings = scan_code("import os; os.system('ls')")
        hits = [f for f in findings if f.pattern_name == "os_system"]
        assert all(f.severity == Severity.CRITICAL for f in hits)

    def test_subprocess_call_is_critical(self):
        findings = scan_code("import subprocess; subprocess.run(['ls'])")
        hits = [f for f in findings if f.pattern_name == "subprocess"]
        assert all(f.severity == Severity.CRITICAL for f in hits)

    def test_socket_import_is_high(self):
        findings = scan_code("import socket")
        hits = [f for f in findings if f.pattern_name == "socket_import"]
        assert all(f.severity == Severity.HIGH for f in hits)

    def test_env_access_is_high(self):
        """Upgraded to HIGH in EPIC 0.3b (secret isolation)."""
        findings = scan_code("import os\nos.environ")
        hits = [f for f in findings if f.pattern_name == "env_access"]
        assert all(f.severity == Severity.HIGH for f in hits)


# ---------------------------------------------------------------------------
# Issue #19 — Blocklist loading
# ---------------------------------------------------------------------------


class TestBlocklist:
    """Blocklist YAML loading and extensibility."""

    def test_default_blocklist_loads(self):
        patterns = load_blocklist()
        assert len(patterns) > 0
        names = {p.name for p in patterns}
        assert "eval" in names
        assert "exec" in names
        assert "subprocess" in names

    def test_pattern_fields(self):
        patterns = load_blocklist()
        for p in patterns:
            assert p.name
            assert p.description
            assert p.severity in Severity
            assert p.ast_type in ("Call", "Attribute", "Import")
            assert isinstance(p.targets, list)
            assert len(p.targets) > 0

    def test_custom_blocklist(self, tmp_path):
        custom = tmp_path / "custom.yml"
        custom.write_text(
            textwrap.dedent("""\
            patterns:
              - name: custom_danger
                description: "Custom dangerous function"
                severity: HIGH
                ast_type: Call
                targets:
                  - my_dangerous_func
        """),
            encoding="utf-8",
        )

        patterns = load_blocklist(extra_paths=[custom])
        names = {p.name for p in patterns}
        assert "custom_danger" in names
        # default patterns still present
        assert "eval" in names

    def test_missing_blocklist_file_ignored(self, tmp_path):
        fake = tmp_path / "nonexistent.yml"
        patterns = load_blocklist(extra_paths=[fake])
        # Should still load default patterns without error
        assert len(patterns) > 0


# ---------------------------------------------------------------------------
# Issue #20 — check_code_safety integration
# ---------------------------------------------------------------------------


class TestCheckCodeSafety:
    """Integration function for the execution pipeline."""

    def test_safe_code_passes(self):
        is_safe, findings = check_code_safety("x = 1 + 2")
        assert is_safe is True

    def test_critical_code_rejected(self):
        is_safe, findings = check_code_safety("eval('malicious')")
        assert is_safe is False
        assert any(f.severity == Severity.CRITICAL for f in findings)

    def test_high_only_allowed(self):
        """HIGH-severity findings do NOT block execution."""
        is_safe, findings = check_code_safety("import socket")
        assert is_safe is True
        assert any(f.severity == Severity.HIGH for f in findings)

    def test_medium_only_allowed(self):
        """MEDIUM-severity findings do NOT block execution."""
        is_safe, findings = check_code_safety("getattr(os, 'path')")
        assert is_safe is True

    def test_multiple_findings_returned(self):
        code = "eval('x')\nexec('y')\nimport os; os.system('z')"
        is_safe, findings = check_code_safety(code)
        assert is_safe is False
        assert len(findings) >= 3

    def test_syntax_error_returns_finding(self):
        is_safe, findings = check_code_safety("def (invalid syntax")
        # Syntax errors produce a finding but are not CRITICAL
        assert len(findings) == 1
        assert findings[0].pattern_name == "syntax_error"


# ---------------------------------------------------------------------------
# Issue #21 — Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Dangerous code hiding in nested contexts."""

    def test_eval_in_nested_function(self):
        code = textwrap.dedent("""\
            def outer():
                def inner():
                    return eval('42')
                return inner()
        """)
        assert "eval" in _names(scan_code(code))

    def test_exec_in_class_method(self):
        code = textwrap.dedent("""\
            class Sneaky:
                def run(self):
                    exec('import os')
        """)
        assert "exec" in _names(scan_code(code))

    def test_eval_in_list_comprehension(self):
        code = "[eval(x) for x in ['1', '2', '3']]"
        assert "eval" in _names(scan_code(code))

    def test_eval_in_lambda(self):
        code = "f = lambda x: eval(x)"
        assert "eval" in _names(scan_code(code))

    def test_os_system_in_conditional(self):
        code = textwrap.dedent("""\
            import os
            if True:
                os.system('echo hi')
        """)
        assert "os_system" in _names(scan_code(code))

    def test_subprocess_in_try_except(self):
        code = textwrap.dedent("""\
            import subprocess
            try:
                subprocess.run(['ls'])
            except Exception:
                pass
        """)
        findings = scan_code(code)
        assert "subprocess" in _names(findings) or "subprocess_import" in _names(findings)

    def test_nested_attribute_chain(self):
        """os.path is safe; os.system is not."""
        safe_code = "import os; p = os.path.join('a', 'b')"
        findings = scan_code(safe_code)
        critical = [f for f in findings if f.severity == Severity.CRITICAL]
        assert critical == []

    def test_chained_dangerous_calls(self):
        code = "eval(exec('import os'))"
        findings = scan_code(code)
        assert "eval" in _names(findings)
        assert "exec" in _names(findings)

    def test_from_import_subprocess(self):
        code = "from subprocess import run; run(['ls'])"
        findings = scan_code(code)
        # Should detect the import
        assert "subprocess_import" in _names(findings)

    def test_finding_has_line_col(self):
        code = "x = 1\neval('2')"
        findings = scan_code(code)
        evals = [f for f in findings if f.pattern_name == "eval"]
        assert len(evals) == 1
        assert evals[0].line == 2
        assert evals[0].col >= 0


# ---------------------------------------------------------------------------
# scan_file
# ---------------------------------------------------------------------------


class TestScanFile:
    def test_scan_existing_file(self, tmp_path):
        f = tmp_path / "danger.py"
        f.write_text("eval('42')", encoding="utf-8")
        findings = scan_file(f)
        assert "eval" in _names(findings)

    def test_scan_missing_file(self, tmp_path):
        findings = scan_file(tmp_path / "nope.py")
        assert len(findings) == 1
        assert findings[0].pattern_name == "file_read_error"

    def test_scan_safe_file(self, tmp_path):
        f = tmp_path / "safe.py"
        f.write_text("x = 42\n", encoding="utf-8")
        findings = scan_file(f)
        assert findings == []
