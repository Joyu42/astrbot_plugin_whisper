"""
Unit tests for SpotifyMCPService.

This test module covers:
1. format_suggestion - formatting Spotify recommendations
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

# Add the workspace directory to path so the plugin package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Mock the astrbot module before importing
sys.modules["astrbot"] = MagicMock()
sys.modules["astrbot.api"] = MagicMock()
sys.modules["astrbot.api.message"] = MagicMock()
sys.modules["astrbot.api.components"] = MagicMock()

# Import the service to test
from astrbot_plugin_whisper.spotify_service import SpotifyMCPService


class TestFormatSuggestion:
    """Test format_suggestion method."""

    def test_format_suggestion_valid_recommend(self):
        """format_suggestion returns formatted string for valid recommend action."""
        service = SpotifyMCPService(command=["dummy"])

        action = {"type": "recommend", "query": "Beatles"}
        result = service.format_suggestion(action)

        assert result == "\n[Spotify 推荐建议: Beatles]"

    def test_format_suggestion_missing_query(self):
        """format_suggestion returns empty string when query is missing."""
        service = SpotifyMCPService(command=["dummy"])

        action = {"type": "recommend"}
        result = service.format_suggestion(action)

        assert result == ""

    def test_format_suggestion_missing_type(self):
        """format_suggestion returns empty string when type is missing."""
        service = SpotifyMCPService(command=["dummy"])

        action = {"query": "Beatles"}
        result = service.format_suggestion(action)

        assert result == ""

    def test_format_suggestion_invalid_type(self):
        """format_suggestion returns empty string for invalid type."""
        service = SpotifyMCPService(command=["dummy"])

        action = {"type": "invalid_type", "query": "Beatles"}
        result = service.format_suggestion(action)

        assert result == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
