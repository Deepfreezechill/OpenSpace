"""AST-based scanner for dangerous API usage in skill code.

Walks a Python AST tree to detect calls to dangerous built-ins,
OS-level APIs, subprocess invocations, dynamic imports, raw socket
access, ctypes FFI, environment variable reads, sensitive file opens,
and attribute-injection patterns.

Patterns are loaded from ``blocklist.yml`` (shipped) and optionally
extended with user-supplied blocklist files.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Sequence

from scion.utils.logging import Logger

logger = Logger.get_logger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"


@dataclass(frozen=True)
class Finding:
    """A single dangerous-API detection result."""

    line: int
    col: int
    severity: Severity
    pattern_name: str
    description: str


@dataclass
class BlocklistPattern:
    """One entry from the blocklist configuration."""

    name: str
    description: str
    severity: Severity
    ast_type: str  # "Call", "Attribute", "Import"
    targets: List[str]


# ---------------------------------------------------------------------------
# Blocklist loader (pure-stdlib, no PyYAML)
# ---------------------------------------------------------------------------

_BLOCKLIST_DIR = Path(__file__).resolve().parent


def _parse_blocklist_yaml(text: str) -> List[Dict[str, Any]]:
    """Minimal parser for the specific blocklist YAML structure.

    Handles a top-level ``patterns:`` key containing a list of mappings
    with scalar values and simple ``targets:`` sub-lists.

    Bounds: max 500 patterns, max 100 targets per pattern.
    """
    MAX_PATTERNS = 500
    MAX_TARGETS = 100

    patterns: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    in_targets = False
    targets_list: List[str] = []

    # Track indent of the pattern-level list dash (typically 2)
    pattern_indent: int | None = None

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Top-level key (e.g. "patterns:")
        if stripped == "patterns:":
            continue

        indent = len(raw_line) - len(raw_line.lstrip())

        # Sub-list item under targets (deeper indent)
        if in_targets and stripped.startswith("- ") and (pattern_indent is not None and indent > pattern_indent):
            if len(targets_list) < MAX_TARGETS:
                targets_list.append(stripped[2:].strip().strip("'\""))
            continue

        # New list item under patterns
        if stripped.startswith("- "):
            if pattern_indent is None:
                pattern_indent = indent
            if indent <= pattern_indent:
                # Flush previous
                if current is not None:
                    if in_targets:
                        current["targets"] = targets_list
                    patterns.append(current)
                    if len(patterns) >= MAX_PATTERNS:
                        break
                current = {}
                in_targets = False
                targets_list = []
                # Parse inline key on same line as dash
                rest = stripped[2:].strip()
                if ":" in rest:
                    k, v = rest.split(":", 1)
                    current[k.strip()] = v.strip()
                continue

        # Key inside a pattern mapping
        if current is not None and ":" in stripped:
            k, v = stripped.split(":", 1)
            k = k.strip()
            v = v.strip()
            if k == "targets" and not v:
                in_targets = True
                targets_list = []
            else:
                in_targets = False
                current[k] = v.strip("'\"")

    # Flush last
    if current is not None and len(patterns) < MAX_PATTERNS:
        if in_targets:
            current["targets"] = targets_list
        patterns.append(current)

    return patterns


def load_blocklist(
    extra_paths: Sequence[str | Path] | None = None,
) -> List[BlocklistPattern]:
    """Load dangerous-API patterns from the default blocklist and any extras.

    FAIL-CLOSED: if the default blocklist cannot be loaded, raises
    ``RuntimeError`` to prevent the scanner from running with no rules.
    """
    default_path = _BLOCKLIST_DIR / "blocklist.yml"
    extra = [Path(p) for p in extra_paths] if extra_paths else []

    results: List[BlocklistPattern] = []
    seen_names: set[str] = set()

    for idx, path in enumerate([default_path, *extra]):
        is_default = idx == 0
        if not path.exists():
            if is_default:
                raise RuntimeError(f"Default blocklist not found: {path}. Cannot scan without rules — refusing to run.")
            logger.warning("Blocklist file not found: %s", path)
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            if is_default:
                raise RuntimeError(f"Cannot read default blocklist {path}: {exc}") from exc
            logger.warning("Cannot read blocklist %s: %s", path, exc)
            continue

        for entry in _parse_blocklist_yaml(text):
            name = entry.get("name", "")
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            try:
                results.append(
                    BlocklistPattern(
                        name=name,
                        description=entry.get("description", ""),
                        severity=Severity(entry.get("severity", "HIGH").upper()),
                        ast_type=entry.get("ast_type", "Call"),
                        targets=entry.get("targets", []),
                    )
                )
            except (ValueError, KeyError) as exc:
                logger.warning("Skipping malformed blocklist entry %r: %s", name, exc)

    if not results:
        raise RuntimeError("Blocklist loaded but produced zero patterns. Cannot scan without rules — refusing to run.")

    return results


# ---------------------------------------------------------------------------
# AST Visitor
# ---------------------------------------------------------------------------

_SENSITIVE_PATH_RE = re.compile(r"^/(proc|etc|sys|dev)/")


class DangerousAPIVisitor(ast.NodeVisitor):
    """Walk a Python AST and collect :class:`Finding` objects for dangerous APIs."""

    def __init__(self, patterns: List[BlocklistPattern] | None = None) -> None:
        self.findings: List[Finding] = []
        self._patterns = patterns or load_blocklist()

        # Pre-index patterns by ast_type for O(1) lookup
        self._call_targets: Dict[str, BlocklistPattern] = {}
        self._attr_targets: Dict[str, BlocklistPattern] = {}
        self._import_targets: Dict[str, BlocklistPattern] = {}

        for pat in self._patterns:
            bucket = {
                "Call": self._call_targets,
                "Attribute": self._attr_targets,
                "Import": self._import_targets,
            }.get(pat.ast_type)
            if bucket is not None:
                for t in pat.targets:
                    bucket[t] = pat

    # -- helpers -------------------------------------------------------------

    def _add(self, node: ast.AST, pat: BlocklistPattern, extra: str = "") -> None:
        desc = pat.description
        if extra:
            desc = f"{desc} ({extra})"
        self.findings.append(
            Finding(
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                severity=pat.severity,
                pattern_name=pat.name,
                description=desc,
            )
        )

    @staticmethod
    def _resolve_call_name(node: ast.Call) -> str | None:
        """Return the dotted name of a Call node's function, or None."""
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            parts: list[str] = [func.attr]
            val = func.value
            while isinstance(val, ast.Attribute):
                parts.append(val.attr)
                val = val.value
            if isinstance(val, ast.Name):
                parts.append(val.id)
                return ".".join(reversed(parts))
        return None

    @staticmethod
    def _resolve_attr_name(node: ast.Attribute) -> str | None:
        """Return the dotted name of an Attribute node."""
        parts: list[str] = [node.attr]
        val = node.value
        while isinstance(val, ast.Attribute):
            parts.append(val.attr)
            val = val.value
        if isinstance(val, ast.Name):
            parts.append(val.id)
            return ".".join(reversed(parts))
        return None

    # -- visitors ------------------------------------------------------------

    # Patterns that require argument inspection — skip in generic match
    _CONTEXT_SENSITIVE = frozenset({"compile", "open"})

    def visit_Call(self, node: ast.Call) -> None:
        name = self._resolve_call_name(node)
        if name:
            # Direct call-target match (e.g. "eval", "os.system")
            # Skip context-sensitive patterns that need argument checks
            if name in self._call_targets and name not in self._CONTEXT_SENSITIVE:
                self._add(node, self._call_targets[name], extra=name)

            # Wildcard match (e.g. "subprocess.*" matches "subprocess.run")
            prefix = name.split(".")[0]
            wildcard_key = f"{prefix}.*"
            if wildcard_key in self._call_targets and wildcard_key != name:
                self._add(node, self._call_targets[wildcard_key], extra=name)

            # compile() with 'exec' mode — positional or keyword
            if name == "compile":
                mode_val = None
                if len(node.args) >= 3:
                    mode_arg = node.args[2]
                    if isinstance(mode_arg, ast.Constant):
                        mode_val = mode_arg.value
                for kw in node.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        mode_val = kw.value.value
                if mode_val == "exec":
                    pat = self._call_targets.get("compile")
                    if pat:
                        self._add(node, pat, extra="compile(..., 'exec')")

            # open() with sensitive paths — positional or keyword
            if name == "open":
                path_val = None
                if node.args:
                    first_arg = node.args[0]
                    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                        path_val = first_arg.value
                for kw in node.keywords:
                    if kw.arg == "file" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        path_val = kw.value.value
                if path_val and _SENSITIVE_PATH_RE.match(path_val):
                    pat = self._call_targets.get("open")
                    if pat:
                        self._add(node, pat, extra=f"open('{path_val}')")

            # getattr/setattr on modules (attribute injection)
            if name in ("getattr", "setattr") and len(node.args) >= 2:
                first_arg = node.args[0]
                if isinstance(first_arg, ast.Name):
                    pat = self._call_targets.get(name)
                    if pat:
                        self._add(node, pat, extra=f"{name}() on '{first_arg.id}'")

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        full_name = self._resolve_attr_name(node)
        if full_name:
            if full_name in self._attr_targets:
                self._add(node, self._attr_targets[full_name], extra=full_name)

            # Wildcard attribute match
            prefix = full_name.split(".")[0]
            wildcard_key = f"{prefix}.*"
            if wildcard_key in self._attr_targets:
                self._add(node, self._attr_targets[wildcard_key], extra=full_name)

        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name in self._import_targets:
                self._add(node, self._import_targets[alias.name], extra=alias.name)
            # Wildcard: "subprocess" matches "subprocess.*" import target
            prefix = alias.name.split(".")[0]
            wildcard = f"{prefix}.*"
            if wildcard in self._import_targets:
                self._add(node, self._import_targets[wildcard], extra=alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if module in self._import_targets:
            self._add(node, self._import_targets[module], extra=module)
        prefix = module.split(".")[0] if module else ""
        wildcard = f"{prefix}.*"
        if prefix and wildcard in self._import_targets:
            self._add(node, self._import_targets[wildcard], extra=module)
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan_code(
    source_code: str,
    extra_blocklists: Sequence[str | Path] | None = None,
) -> List[Finding]:
    """Parse *source_code* and return a list of dangerous-API findings."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        return [
            Finding(
                line=exc.lineno or 0,
                col=exc.offset or 0,
                severity=Severity.HIGH,
                pattern_name="syntax_error",
                description=f"Could not parse source: {exc.msg}",
            )
        ]

    patterns = load_blocklist(extra_paths=extra_blocklists)
    visitor = DangerousAPIVisitor(patterns=patterns)
    visitor.visit(tree)
    return visitor.findings


def scan_file(
    file_path: str | Path,
    extra_blocklists: Sequence[str | Path] | None = None,
) -> List[Finding]:
    """Read a Python file and scan it for dangerous APIs."""
    path = Path(file_path)
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [
            Finding(
                line=0,
                col=0,
                severity=Severity.HIGH,
                pattern_name="file_read_error",
                description=f"Cannot read file: {exc}",
            )
        ]
    return scan_code(source, extra_blocklists=extra_blocklists)
