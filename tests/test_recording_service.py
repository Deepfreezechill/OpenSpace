"""Tests for RecordingService — extracted recording factory/wiring from OpenSpace."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

try:
    from openspace.tool_layer import OpenSpace, OpenSpaceConfig
    from openspace.recording_service import RecordingService

    _HAS_TOOL_LAYER = True
except Exception:
    _HAS_TOOL_LAYER = False

pytestmark = pytest.mark.skipif(not _HAS_TOOL_LAYER, reason="tool_layer deps unavailable")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return OpenSpaceConfig(
        enable_recording=True,
        recording_backends=["json"],
        recording_log_dir="/tmp/recordings",
        enable_screenshot=False,
        enable_video=False,
        enable_conversation_log=True,
    )


@pytest.fixture
def disabled_config():
    return OpenSpaceConfig(enable_recording=False)


@pytest.fixture
def mock_llm_client():
    return MagicMock()


@pytest.fixture
def mock_grounding_client():
    client = MagicMock()
    client.recording_manager = None
    return client


# ---------------------------------------------------------------------------
# RecordingService.__init__
# ---------------------------------------------------------------------------

class TestRecordingServiceInit:

    def test_initial_state(self, config):
        svc = RecordingService(config=config)
        assert svc.manager is None

    def test_stores_config(self, config):
        svc = RecordingService(config=config)
        assert svc._config is config


# ---------------------------------------------------------------------------
# RecordingService.create()
# ---------------------------------------------------------------------------

class TestRecordingServiceCreate:

    def test_creates_manager_when_enabled(self, config, mock_llm_client):
        svc = RecordingService(config=config)
        mgr = svc.create(llm_client=mock_llm_client)
        assert mgr is not None
        assert svc.manager is mgr

    def test_returns_none_when_disabled(self, disabled_config, mock_llm_client):
        svc = RecordingService(config=disabled_config)
        mgr = svc.create(llm_client=mock_llm_client)
        assert mgr is None
        assert svc.manager is None

    def test_passes_config_fields_to_manager(self, config, mock_llm_client):
        with patch("openspace.recording_service.RecordingManager") as MockRM:
            mock_instance = MagicMock()
            MockRM.return_value = mock_instance

            svc = RecordingService(config=config)
            svc.create(llm_client=mock_llm_client)

            MockRM.assert_called_once_with(
                enabled=True,
                task_id="",
                log_dir="/tmp/recordings",
                backends=["json"],
                enable_screenshot=False,
                enable_video=False,
                enable_conversation_log=True,
                agent_name="OpenSpace",
            )

    def test_registers_to_llm(self, config, mock_llm_client):
        with patch("openspace.recording_service.RecordingManager") as MockRM:
            mock_instance = MagicMock()
            MockRM.return_value = mock_instance

            svc = RecordingService(config=config)
            svc.create(llm_client=mock_llm_client)
            mock_instance.register_to_llm.assert_called_once_with(mock_llm_client)


# ---------------------------------------------------------------------------
# RecordingService.wire()
# ---------------------------------------------------------------------------

class TestRecordingServiceWire:

    def test_injects_manager_into_grounding_client(self, config, mock_llm_client, mock_grounding_client):
        with patch("openspace.recording_service.RecordingManager") as MockRM:
            mock_instance = MagicMock()
            MockRM.return_value = mock_instance

            svc = RecordingService(config=config)
            svc.create(llm_client=mock_llm_client)
            svc.wire(grounding_client=mock_grounding_client)
            assert mock_grounding_client.recording_manager is mock_instance

    def test_noop_when_no_manager(self, disabled_config, mock_llm_client, mock_grounding_client):
        svc = RecordingService(config=disabled_config)
        svc.create(llm_client=mock_llm_client)
        svc.wire(grounding_client=mock_grounding_client)
        assert mock_grounding_client.recording_manager is None


# ---------------------------------------------------------------------------
# RecordingService.cleanup()
# ---------------------------------------------------------------------------

class TestRecordingServiceCleanup:

    @pytest.mark.asyncio
    async def test_stops_active_recording(self, config, mock_llm_client):
        with patch("openspace.recording_service.RecordingManager") as MockRM:
            mock_instance = MagicMock()
            mock_instance.recording_status = True
            mock_instance.stop = AsyncMock()
            MockRM.return_value = mock_instance

            svc = RecordingService(config=config)
            svc.create(llm_client=mock_llm_client)
            await svc.cleanup()
            mock_instance.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_noop_when_not_recording(self, config, mock_llm_client):
        with patch("openspace.recording_service.RecordingManager") as MockRM:
            mock_instance = MagicMock()
            mock_instance.recording_status = False
            mock_instance.stop = AsyncMock()
            MockRM.return_value = mock_instance

            svc = RecordingService(config=config)
            svc.create(llm_client=mock_llm_client)
            await svc.cleanup()
            mock_instance.stop.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_noop_when_no_manager(self, disabled_config):
        svc = RecordingService(config=disabled_config)
        await svc.cleanup()  # No raise

    @pytest.mark.asyncio
    async def test_stop_exception_swallowed_and_logged(self, config, mock_llm_client):
        with patch("openspace.recording_service.RecordingManager") as MockRM:
            mock_instance = MagicMock()
            mock_instance.recording_status = True
            mock_instance.stop = AsyncMock(side_effect=RuntimeError("stop boom"))
            MockRM.return_value = mock_instance

            svc = RecordingService(config=config)
            svc.create(llm_client=mock_llm_client)
            await svc.cleanup()  # No raise — exception logged


# ---------------------------------------------------------------------------
# OpenSpace backward compatibility
# ---------------------------------------------------------------------------

class TestOpenSpaceDelegation:

    def test_openspace_has_recording_service_attr(self):
        os_instance = OpenSpace()
        assert hasattr(os_instance, "_recording_service")

    def test_recording_manager_still_accessible(self):
        os_instance = OpenSpace()
        assert hasattr(os_instance, "_recording_manager")
        assert os_instance._recording_manager is None
