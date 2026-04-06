"""Tests for openspace.agents.grounding.workspace — workspace helpers.

Epic 5.9 extraction: _get_workspace_path, _scan_workspace_files,
_check_workspace_artifacts.
"""

from __future__ import annotations

import os
import tempfile
import time

import pytest

from openspace.agents.grounding.workspace import (
    _check_workspace_artifacts,
    _get_workspace_path,
    _scan_workspace_files,
)


# ── _get_workspace_path (pure function) ────────────────────────────


class TestGetWorkspacePath:
    def test_returns_value_from_context(self):
        assert _get_workspace_path({"workspace_dir": "/some/path"}) == "/some/path"

    def test_returns_none_when_missing(self):
        assert _get_workspace_path({}) is None

    def test_returns_none_for_empty_context(self):
        assert _get_workspace_path({"other": 123}) is None


# ── _scan_workspace_files (pure function) ──────────────────────────


class TestScanWorkspaceFiles:
    def test_returns_empty_for_none_path(self):
        result = _scan_workspace_files(None)
        assert result["files"] == []
        assert result["file_details"] == {}
        assert result["recent_files"] == []

    def test_returns_empty_for_nonexistent_path(self):
        result = _scan_workspace_files("/nonexistent/path/xyz")
        assert result["files"] == []

    def test_scans_real_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for name in ["alpha.txt", "beta.py", "gamma.md"]:
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write("content")

            result = _scan_workspace_files(tmpdir, recent_threshold=600)
            assert sorted(result["files"]) == ["alpha.txt", "beta.py", "gamma.md"]
            assert len(result["file_details"]) == 3
            # All files just created → should be recent
            assert len(result["recent_files"]) == 3

    def test_excludes_metadata_and_traj(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["metadata.json", "traj.jsonl", "real.txt"]:
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write("x")

            result = _scan_workspace_files(tmpdir)
            assert result["files"] == ["real.txt"]

    def test_skips_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "subdir"))
            with open(os.path.join(tmpdir, "file.txt"), "w") as f:
                f.write("data")

            result = _scan_workspace_files(tmpdir)
            assert result["files"] == ["file.txt"]

    def test_old_files_not_in_recent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "old.txt")
            with open(filepath, "w") as f:
                f.write("data")
            # Set mtime to 2 hours ago
            old_time = time.time() - 7200
            os.utime(filepath, (old_time, old_time))

            result = _scan_workspace_files(tmpdir, recent_threshold=600)
            assert "old.txt" in result["files"]
            assert "old.txt" not in result["recent_files"]

    def test_file_details_contain_expected_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test.txt"), "w") as f:
                f.write("hello world")

            result = _scan_workspace_files(tmpdir)
            details = result["file_details"]["test.txt"]
            assert "size" in details
            assert "modified" in details
            assert "age_seconds" in details
            assert details["size"] > 0

    def test_files_sorted_alphabetically(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["zebra.txt", "apple.txt", "mango.txt"]:
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write("x")

            result = _scan_workspace_files(tmpdir)
            assert result["files"] == ["apple.txt", "mango.txt", "zebra.txt"]


# ── _check_workspace_artifacts ─────────────────────────────────────


class _FakeAgent:
    """Minimal stand-in — workspace module calls module-level functions directly,
    so agent is only used as first parameter for _check_workspace_artifacts."""
    pass


class TestCheckWorkspaceArtifacts:
    @pytest.mark.asyncio
    async def test_empty_when_no_workspace_dir(self):
        agent = _FakeAgent()
        result = await _check_workspace_artifacts(agent, {})
        assert result["has_files"] is False
        assert result["files"] == []

    @pytest.mark.asyncio
    async def test_finds_files_in_workspace(self):
        agent = _FakeAgent()
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["report.pdf", "data.csv"]:
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write("content")

            result = await _check_workspace_artifacts(agent, {"workspace_dir": tmpdir})
            assert result["has_files"] is True
            assert "report.pdf" in result["files"]
            assert "data.csv" in result["files"]

    @pytest.mark.asyncio
    async def test_detects_matching_files_from_instruction(self):
        agent = _FakeAgent()
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "output.png"), "w") as f:
                f.write("image data")

            context = {
                "workspace_dir": tmpdir,
                "instruction": 'Generate "output.png" from the data',
            }
            result = await _check_workspace_artifacts(agent, context)
            assert result.get("matching_files") == ["output.png"]

    @pytest.mark.asyncio
    async def test_no_matching_files_when_instruction_empty(self):
        agent = _FakeAgent()
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "file.txt"), "w") as f:
                f.write("x")

            result = await _check_workspace_artifacts(agent, {"workspace_dir": tmpdir})
            assert "matching_files" not in result

    @pytest.mark.asyncio
    async def test_resilient_to_errors(self):
        """Should not raise even when context is weird."""
        agent = _FakeAgent()
        result = await _check_workspace_artifacts(agent, {"workspace_dir": "/nonexistent/xyz"})
        assert result["has_files"] is False


# ── Delegation seam tests ──────────────────────────────────────────


class TestWorkspaceDelegationSeams:
    """Verify grounding_agent.py properly delegates to workspace module."""

    def test_get_workspace_path_is_static(self):
        from openspace.agents.grounding_agent import GroundingAgent

        assert callable(GroundingAgent._get_workspace_path)
        assert GroundingAgent._get_workspace_path is _get_workspace_path

    def test_scan_workspace_files_is_static(self):
        from openspace.agents.grounding_agent import GroundingAgent

        assert callable(GroundingAgent._scan_workspace_files)
        assert GroundingAgent._scan_workspace_files is _scan_workspace_files

    def test_check_workspace_artifacts_delegates(self):
        from openspace.agents.grounding_agent import GroundingAgent

        assert "_check_workspace_artifacts" in dir(GroundingAgent)
        method = getattr(GroundingAgent, "_check_workspace_artifacts")
        assert callable(method)
