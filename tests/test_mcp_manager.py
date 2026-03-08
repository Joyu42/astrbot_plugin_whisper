"""Tests for MCPManager."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from astrbot_plugin_whisper.models import LLMDecision, WhisperConfig
from astrbot_plugin_whisper.mcp_manager import MCPManager


class TestMCPManager:
    """Test cases for MCPManager."""

    def test_format_combined_suggestions_with_action(self):
        """Test format_combined_suggestions when LLMDecision has spotify_action."""
        # Create a mock SpotifyMCPService
        mock_spotify = MagicMock()
        mock_spotify.format_suggestion = MagicMock(
            return_value="\n[Spotify 推荐建议: 放松的爵士乐]"
        )

        # Create MCPManager and manually inject the mock service
        manager = MCPManager()
        manager._services["spotify"] = mock_spotify

        # Create LLMDecision with spotify_action
        decision = LLMDecision(
            should_send=True,
            content="Test content",
            reason="Test reason",
            spotify_action={"type": "recommend", "query": "放松的爵士乐"},
        )

        # Call format_combined_suggestions
        result = manager.format_combined_suggestions(decision)

        # Verify the result
        assert result == "\n[Spotify 推荐建议: 放松的爵士乐]"
        mock_spotify.format_suggestion.assert_called_once_with(
            {"type": "recommend", "query": "放松的爵士乐"}
        )

    def test_format_combined_suggestions_no_action(self):
        """Test format_combined_suggestions when spotify_action is None."""
        # Create MCPManager with no services
        manager = MCPManager()

        # Create LLMDecision without spotify_action
        decision = LLMDecision(
            should_send=True,
            content="Test content",
            reason="Test reason",
            spotify_action=None,
        )

        # Call format_combined_suggestions
        result = manager.format_combined_suggestions(decision)

        # Verify the result is empty string
        assert result == ""

    def test_format_combined_suggestions_no_spotify_service(self):
        """Test format_combined_suggestions when spotify service not loaded."""
        # Create MCPManager with empty services
        manager = MCPManager()

        # Create LLMDecision with spotify_action but no service
        decision = LLMDecision(
            should_send=True,
            content="Test content",
            reason="Test reason",
            spotify_action={"type": "recommend", "query": "test"},
        )

        # Call format_combined_suggestions
        result = manager.format_combined_suggestions(decision)

        # Verify the result is empty string
        assert result == ""

    @pytest.mark.asyncio
    async def test_get_combined_context(self):
        """Test get_combined_context gathers context from active services."""
        # Create mock SpotifyMCPService
        mock_spotify = MagicMock()
        mock_spotify.get_context = AsyncMock(
            return_value="用户当前正在听 Spotify: Test Song - Test Artist"
        )

        # Create MCPManager and manually inject the mock service
        manager = MCPManager()
        manager._services["spotify"] = mock_spotify

        # Create config with spotify_context_enabled
        config = WhisperConfig(
            mcp_enabled=True, mcp_services=["spotify"], spotify_context_enabled=True
        )

        # Call get_combined_context
        result = await manager.get_combined_context(config)

        # Verify the result
        assert result == "用户当前正在听 Spotify: Test Song - Test Artist"
        mock_spotify.get_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_combined_context_no_context_enabled(self):
        """Test get_combined_context when spotify_context_enabled is False."""
        # Create MCPManager with mock service
        mock_spotify = MagicMock()
        mock_spotify.get_context = AsyncMock(return_value="Some context")

        manager = MCPManager()
        manager._services["spotify"] = mock_spotify

        # Create config with spotify_context_enabled=False
        config = WhisperConfig(
            mcp_enabled=True, mcp_services=["spotify"], spotify_context_enabled=False
        )

        # Call get_combined_context
        result = await manager.get_combined_context(config)

        # Verify result is empty
        assert result == ""
        mock_spotify.get_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_combined_context_no_services(self):
        """Test get_combined_context when no services are loaded."""
        # Create MCPManager with no services
        manager = MCPManager()

        # Create config
        config = WhisperConfig(mcp_enabled=True, spotify_context_enabled=True)

        # Call get_combined_context
        result = await manager.get_combined_context(config)

        # Verify result is empty
        assert result == ""

    @pytest.mark.asyncio
    async def test_load_services_disabled(self):
        """Test load_services when mcp_enabled is False."""
        manager = MCPManager()

        config = WhisperConfig(mcp_enabled=False)

        await manager.load_services(config)

        assert len(manager._services) == 0

    @pytest.mark.asyncio
    async def test_stop_all(self):
        """Test stop_all stops all services."""
        # Create mock services
        mock_spotify = AsyncMock()
        mock_other = AsyncMock()

        manager = MCPManager()
        manager._services["spotify"] = mock_spotify
        manager._services["other"] = mock_other

        await manager.stop_all()

        mock_spotify.stop.assert_called_once()
        mock_other.stop.assert_called_once()
        assert len(manager._services) == 0
