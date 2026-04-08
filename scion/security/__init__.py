"""Scion Security — AST-based code safety scanning.

Public API
----------
check_code_safety(source)
    Pre-execution gate.  Returns ``(is_safe, findings)``.

scan_code / scan_file
    Lower-level scanning functions.
"""

from __future__ import annotations

from typing import List, Tuple

from scion.utils.logging import Logger

from .ast_scanner import Finding, Severity, scan_code, scan_file  # noqa: F401

logger = Logger.get_logger(__name__)

__all__ = [
    "check_code_safety",
    "scan_code",
    "scan_file",
    "Finding",
    "Severity",
]


def check_code_safety(source: str) -> Tuple[bool, List[Finding]]:
    """Scan *source* for dangerous APIs and decide whether execution is safe.

    Returns ``(is_safe, findings)`` where *is_safe* is ``False`` when any
    **CRITICAL** finding is present (code must be rejected).

    **HIGH** findings are logged as warnings but do **not** block execution.
    **MEDIUM** findings are informational only.
    """
    findings = scan_code(source)

    for f in findings:
        if f.severity == Severity.CRITICAL:
            logger.warning(
                "CRITICAL security finding — code REJECTED: [%s] %s (line %d)",
                f.pattern_name,
                f.description,
                f.line,
            )
        elif f.severity == Severity.HIGH:
            logger.warning(
                "HIGH security finding: [%s] %s (line %d)",
                f.pattern_name,
                f.description,
                f.line,
            )
        else:
            logger.info(
                "MEDIUM security finding: [%s] %s (line %d)",
                f.pattern_name,
                f.description,
                f.line,
            )

    has_critical = any(f.severity == Severity.CRITICAL for f in findings)
    return (not has_critical, findings)
