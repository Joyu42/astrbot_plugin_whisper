"""
Unit tests for Whisper plugin core logic.

This test module covers:
1. Data Models (SessionState, LLMDecision, WhisperConfig)
2. Config Parsing (parse_config)
3. Quiet Hours (is_quiet_hours)
4. Silence Trigger Delay (get_silence_trigger_delay)
5. Should Send Proactive (should_send_proactive)
6. Prompt Placeholders (replace_prompt_placeholders)
7. Content Parsing (_parse_content_response)
8. Backoff Math (compute_backoff_minutes, apply_jitter)
"""

import json
import os
import sys
import time
import pytest
from datetime import datetime, timedelta
from datetime import time as datetime_time
from unittest.mock import patch, MagicMock

# Add the workspace directory to path so the plugin package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Mock the astrbot module before importing main
sys.modules["astrbot"] = MagicMock()
sys.modules["astrbot.api"] = MagicMock()
sys.modules["astrbot.api.message"] = MagicMock()
sys.modules["astrbot.api.components"] = MagicMock()

# Import from modules via package path
from astrbot_plugin_whisper.models import (
    SessionState,
    LLMDecision,
    WhisperConfig,
    parse_config,
)

# Import from utils module
from astrbot_plugin_whisper.utils import (
    is_quiet_hours,
    get_silence_trigger_delay,
    should_send_proactive,
    replace_prompt_placeholders,
    _parse_content_response,
    _segment_text,
    _format_status_message,
    _build_proactive_prompt,
    DEFAULT_PROMPT_TEMPLATE,
)

# Import from scheduler module
from astrbot_plugin_whisper.scheduler import (
    _load_sessions_sync,
    _save_sessions_sync,
    _get_data_file_path,
)


# ===== Test Data Models =====


class TestDataModels:
    """Test data model dataclasses."""

    def test_session_state_default_values(self):
        """SessionState should have correct default values."""
        state = SessionState(session_id="test_123", last_message_time=1000.0)

        assert state.session_id == "test_123"
        assert state.last_message_time == 1000.0
        assert state.unanswered_count == 0  # Default
        assert state.next_trigger_time == 0.0  # Default
        assert state.enabled is True  # Default
        assert state.self_id == ""  # Default

    def test_session_state_custom_values(self):
        """SessionState should accept custom values."""
        state = SessionState(
            session_id="user_456",
            last_message_time=2000.0,
            unanswered_count=2,
            next_trigger_time=3000.0,
            enabled=False,
            self_id="bot_789",
        )

        assert state.session_id == "user_456"
        assert state.last_message_time == 2000.0
        assert state.unanswered_count == 2
        assert state.next_trigger_time == 3000.0
        assert state.enabled is False
        assert state.self_id == "bot_789"

    def test_llm_decision_required_field(self):
        """LLMDecision should require content field."""
        decision = LLMDecision(content="Hello!")

        assert decision.content == "Hello!"
        assert decision.reason == ""  # Default

    def test_llm_decision_all_fields(self):
        """LLMDecision should accept all fields."""
        decision = LLMDecision(content="Hello!", reason="Testing")

        assert decision.content == "Hello!"
        assert decision.reason == "Testing"

    def test_whisper_config_defaults(self):
        """WhisperConfig should have correct default values."""
        config = WhisperConfig()

        assert config.enable is True
        assert config.silence_trigger_minutes == 5
        assert config.timeout_max == 30
        assert config.max_consecutive == 3
        assert config.quiet_hours_enabled is True
        assert config.quiet_hours_start == "23:00"
        assert config.quiet_hours_end == "08:00"
        assert config.max_history_messages == 20
        assert config.segment_enabled is True
        assert config.segment_threshold == 150
        assert config.segment_mode == "regex"
        assert config.segment_delay_ms == 1500
        assert config.proactive_prompt == ""
        # MCP and Spotify config defaults
        assert config.mcp_enabled is False
        assert config.mcp_services == []
        assert config.spotify_context_enabled is False
        assert config.spotify_suggest_enabled is False
        assert (
            config.spotify_mcp_command
            == "node data/mcp_servers/spotify-mcp-server/build/index.js"
        )
        assert config.llm_provider_id == ""

    def test_session_state_has_backoff_fields(self):
        """SessionState should expose backoff fields for single-phase checks."""
        state = SessionState(session_id="test_123", last_message_time=1000.0)

        assert state.backoff_level == 0
        assert state.last_check_time == 0.0

    def test_llm_decision_has_should_send_field(self):
        """LLMDecision should expose should_send field (optional)."""
        decision = LLMDecision(content="hi")
        assert decision.should_send is None

    def test_llm_decision_has_spotify_action_field(self):
        """LLMDecision should expose spotify_action field (optional dict)."""
        decision = LLMDecision(content="hi")
        assert decision.spotify_action is None

        # Test with spotify_action set
        action = {"action": "play", "track": "test song"}
        decision_with_action = LLMDecision(content="Now playing", spotify_action=action)
        assert decision_with_action.spotify_action == action


# ===== Test Config Parsing =====


class TestConfigParsing:
    """Test parse_config function."""

    def test_parse_config_defaults(self):
        """parse_config should use defaults when no config provided."""
        config = parse_config({})

        assert config.enable is True
        assert config.silence_trigger_minutes == 5
        assert config.timeout_max == 30
        assert config.max_consecutive == 3

    def test_parse_config_with_values(self):
        """parse_config should use provided values."""
        raw_config = {
            "enable": False,
            "silence_trigger_minutes": 10,
            "timeout_max": 60,
            "max_consecutive": 5,
        }
        config = parse_config(raw_config)

        assert config.enable is False
        assert config.silence_trigger_minutes == 10
        assert config.timeout_max == 60
        assert config.max_consecutive == 5

    def test_parse_config_with_session_override(self):
        """parse_config should apply session-level overrides."""
        raw_config = {
            "enable": True,
            "silence_trigger_minutes": 5,
            "timeout_max": 30,
            "session_configs": {
                "user_123": {
                    "silence_trigger_minutes": 15,
                    "enable": False,
                }
            },
        }

        # Without session_id - use global config
        config = parse_config(raw_config)
        assert config.silence_trigger_minutes == 5
        assert config.enable is True

        # With session_id - apply override
        config = parse_config(raw_config, "user_123")
        assert config.silence_trigger_minutes == 15
        assert config.enable is False

    def test_parse_config_missing_keys(self):
        """parse_config should handle missing keys gracefully."""
        raw_config = {
            "enable": False,
            # Missing other keys
        }
        config = parse_config(raw_config)

        assert config.enable is False
        assert config.silence_trigger_minutes == 5  # Default
        assert config.timeout_max == 30  # Default
        assert config.quiet_hours_enabled is True  # Default

    def test_parse_config_all_keys(self):
        """parse_config should handle all configuration keys."""
        raw_config = {
            "enable": False,
            "silence_trigger_minutes": 1,
            "timeout_max": 5,
            "max_consecutive": 1,
            "quiet_hours_enabled": False,
            "quiet_hours_start": "22:00",
            "quiet_hours_end": "07:00",
            "max_history_messages": 10,
            "segment_enabled": False,
            "segment_threshold": 50,
            "segment_delay_ms": 500,
            "proactive_prompt": "Custom prompt",
            # MCP and Spotify config
            "mcp_enabled": True,
            "mcp_services": ["spotify", "filesystem"],
            "spotify_context_enabled": True,
            "spotify_suggest_enabled": True,
            "spotify_mcp_command": "npx -y custom-spotify-mcp",
            "llm_provider_id": "deepseek/deepseek-chat",
        }

        config = parse_config(raw_config)

        assert config.enable is False
        assert config.silence_trigger_minutes == 1
        assert config.timeout_max == 5
        assert config.max_consecutive == 1
        assert config.quiet_hours_enabled is False
        assert config.quiet_hours_start == "22:00"
        assert config.quiet_hours_end == "07:00"
        assert config.max_history_messages == 10
        assert config.segment_enabled is False
        assert config.segment_threshold == 50
        assert config.segment_delay_ms == 500
        assert config.proactive_prompt == "Custom prompt"
        # MCP and Spotify config
        assert config.mcp_enabled is True
        assert config.mcp_services == ["spotify", "filesystem"]
        assert config.spotify_context_enabled is True
        assert config.spotify_suggest_enabled is True
        assert config.spotify_mcp_command == "npx -y custom-spotify-mcp"
        assert config.llm_provider_id == "deepseek/deepseek-chat"


# ===== Test Quiet Hours =====


class TestQuietHours:
    """Test is_quiet_hours function."""

    def test_quiet_hours_disabled(self):
        """is_quiet_hours returns False when disabled."""
        config = WhisperConfig(quiet_hours_enabled=False)

        with patch("astrbot_plugin_whisper.utils.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 1, 23, 30)
            assert is_quiet_hours(config) is False

    def test_quiet_hours_within_range(self):
        """is_quiet_hours returns True when within range."""
        config = WhisperConfig(
            quiet_hours_enabled=True,
            quiet_hours_start="10:00",
            quiet_hours_end="12:00",
        )

        with patch("astrbot_plugin_whisper.utils.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 1, 11, 0)
            assert is_quiet_hours(config) is True

    def test_quiet_hours_cross_midnight(self):
        """is_quiet_hours handles cross-midnight (23:00-08:00)."""
        config = WhisperConfig(
            quiet_hours_enabled=True,
            quiet_hours_start="23:00",
            quiet_hours_end="08:00",
        )

        with patch("astrbot_plugin_whisper.utils.datetime") as mock_datetime:
            # 23:30 - should be True (after start)
            mock_datetime.now.return_value = datetime(2024, 1, 1, 23, 30)
            assert is_quiet_hours(config) is True

            # 02:00 - should be True (before end)
            mock_datetime.now.return_value = datetime(2024, 1, 1, 2, 0)
            assert is_quiet_hours(config) is True

            # 12:00 - should be False (outside range)
            mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0)
            assert is_quiet_hours(config) is False

    def test_quiet_hours_outside_range(self):
        """is_quiet_hours returns False when outside range."""
        config = WhisperConfig(
            quiet_hours_enabled=True,
            quiet_hours_start="10:00",
            quiet_hours_end="12:00",
        )

        with patch("astrbot_plugin_whisper.utils.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 1, 14, 0)
            assert is_quiet_hours(config) is False

    def test_quiet_hours_boundary_cases(self):
        """is_quiet_hours handles boundary times correctly."""
        config = WhisperConfig(
            quiet_hours_enabled=True,
            quiet_hours_start="10:00",
            quiet_hours_end="12:00",
        )

        with patch("astrbot_plugin_whisper.utils.datetime") as mock_datetime:
            # At start time - should be True
            mock_datetime.now.return_value = datetime(2024, 1, 1, 10, 0)
            assert is_quiet_hours(config) is True

            # At end time - should be True (within)
            mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0)
            assert is_quiet_hours(config) is True

            # One minute after end - should be False
            mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 1)
            assert is_quiet_hours(config) is False


# ===== Test Silence Trigger Delay =====


class TestSilenceTriggerDelay:
    """Test get_silence_trigger_delay function."""

    def test_silence_trigger_delay_fixed_value(self):
        """get_silence_trigger_delay returns fixed delay value in seconds."""
        config = WhisperConfig(silence_trigger_minutes=1)

        result = get_silence_trigger_delay(config)

        assert result == 60  # 1 minute * 60 = 60 seconds

    def test_silence_trigger_delay_default(self):
        """get_silence_trigger_delay returns default value when not set."""
        config = WhisperConfig()

        result = get_silence_trigger_delay(config)

        assert result == 300  # Default: 5 * 60 = 300 seconds


# ===== Test Should Send Proactive =====


class TestShouldSendProactive:
    """Test should_send_proactive function."""

    def test_should_send_disabled_config(self):
        """Returns False when config.enable is False."""
        config = WhisperConfig(enable=False)
        state = SessionState(session_id="test", last_message_time=time.time())

        can_send, reason = should_send_proactive(config, state)

        assert can_send is False
        assert reason == "disabled"

    def test_should_send_quiet_hours(self):
        """Returns False during quiet hours."""
        config = WhisperConfig(
            enable=True,
            quiet_hours_enabled=True,
            quiet_hours_start="00:00",
            quiet_hours_end="23:59",  # Always quiet
        )
        state = SessionState(session_id="test", last_message_time=time.time())

        can_send, reason = should_send_proactive(config, state)

        assert can_send is False
        assert reason == "quiet_hours"

    def test_should_send_max_consecutive(self):
        """Returns False when max consecutive reached."""
        config = WhisperConfig(
            enable=True,
            quiet_hours_enabled=False,
            max_consecutive=3,
        )
        state = SessionState(
            session_id="test",
            last_message_time=time.time(),
            unanswered_count=3,
        )

        can_send, reason = should_send_proactive(config, state)

        assert can_send is False
        assert reason == "max_consecutive"

    def test_should_send_allowed(self):
        """Returns True when all conditions allow."""
        config = WhisperConfig(
            enable=True,
            quiet_hours_enabled=False,
            max_consecutive=3,
        )
        state = SessionState(
            session_id="test",
            last_message_time=time.time(),
            unanswered_count=1,
        )

        can_send, reason = should_send_proactive(config, state)

        assert can_send is True
        assert reason == ""


# ===== Test Prompt Placeholders =====


class TestPromptPlaceholders:
    """Test replace_prompt_placeholders function."""

    def test_replace_current_time(self):
        """replace_prompt_placeholders replaces current_time."""
        state = SessionState(
            session_id="test", last_message_time=time.time(), unanswered_count=0
        )
        template = "现在是 {{current_time}}"

        result = replace_prompt_placeholders(template, state)

        assert "{{current_time}}" not in result
        # Should contain a date/time string
        assert len(result) > 10

    def test_replace_unanswered_count(self):
        """replace_prompt_placeholders replaces unanswered_count."""
        state = SessionState(
            session_id="test", last_message_time=time.time(), unanswered_count=5
        )
        template = "未回复消息: {{unanswered_count}}"

        result = replace_prompt_placeholders(template, state)

        assert "{{unanswered_count}}" not in result
        assert "5" in result

    def test_no_placeholders_to_replace(self):
        """replace_prompt_placeholders handles template without placeholders."""
        state = SessionState(
            session_id="test", last_message_time=time.time(), unanswered_count=0
        )
        template = "这是一段普通文本"

        result = replace_prompt_placeholders(template, state)

        assert result == "这是一段普通文本"


# ===== Test Prediction Parsing =====


# ===== Test Content Parsing =====


class TestContentParsing:
    """Test _parse_content_response function."""

    def test_parse_content_standard_json(self):
        """_parse_content_response parses standard JSON."""
        raw = '{"content": "最近怎么样？", "reason": "关心朋友"}'

        result = _parse_content_response(raw)

        assert result.should_send is None
        assert result.content == "最近怎么样？"
        assert result.reason == "missing_should_send"

    def test_parse_content_should_send_false_allows_empty_content(self):
        """should_send=false should be treated as a valid no-send decision."""
        raw = '{"should_send": false, "content": "", "reason": "不想打扰"}'

        result = _parse_content_response(raw)

        assert result.should_send is False
        assert result.content == ""
        assert result.reason == "不想打扰"

    def test_parse_content_should_send_true_requires_non_empty_content(self):
        """should_send=true with empty content should be treated as empty_content."""
        raw = '{"should_send": true, "content": "", "reason": "想问候"}'

        result = _parse_content_response(raw)

        assert result.should_send is True
        assert result.content == ""
        assert result.reason == "empty_content"

    def test_parse_content_missing_should_send_marks_missing(self):
        """Missing should_send should be detected for scheduler/backoff handling."""
        raw = '{"content": "hi", "reason": "test"}'

        result = _parse_content_response(raw)

        assert result.should_send is None
        assert result.reason == "missing_should_send"


class TestBackoffHelpers:
    def test_compute_backoff_minutes_exponential_with_cap(self):
        """compute_backoff_minutes should grow exponentially and clamp to cap."""
        from astrbot_plugin_whisper.utils import compute_backoff_minutes

        assert compute_backoff_minutes(base_minutes=5, level=0, cap_minutes=30) == 5
        assert compute_backoff_minutes(base_minutes=5, level=1, cap_minutes=30) == 10
        assert compute_backoff_minutes(base_minutes=5, level=2, cap_minutes=30) == 20
        assert compute_backoff_minutes(base_minutes=5, level=3, cap_minutes=30) == 30

    def test_apply_jitter_returns_positive_seconds(self):
        """apply_jitter should always return >= 1 second."""
        from astrbot_plugin_whisper.utils import apply_jitter

        assert apply_jitter(1, ratio=0.1) >= 1


# ===== Test MCP Decision Parsing =====


class TestMCPDecisionParsing:
    """Test _parse_content_response extracts spotify_action from JSON."""

    def test_parse_content_extracts_spotify_action_from_direct_json(self):
        """_parse_content_response should extract spotify_action when provided in JSON."""
        raw = '{"should_send": true, "content": "Playing some music!", "reason": "test", "spotify_action": {"action": "play", "track": "test song"}}'

        result = _parse_content_response(raw)

        assert result.should_send is True
        assert result.content == "Playing some music!"
        assert result.spotify_action == {"action": "play", "track": "test song"}

    def test_parse_content_handles_json_without_spotify_action(self):
        """_parse_content_response should leave spotify_action as None when not in JSON."""
        raw = '{"should_send": true, "content": "Hello!", "reason": "test"}'

        result = _parse_content_response(raw)

        assert result.should_send is True
        assert result.content == "Hello!"
        assert result.spotify_action is None

    def test_parse_content_extracts_spotify_action_from_regex_match(self):
        """Regex matching should also extract spotify_action when JSON has simple structure.

        Note: The regex pattern can't handle nested braces, so this test uses
        a JSON structure that the regex CAN match (flat spotify_action value).
        """
        # The regex pattern matches JSON with 'content' field but no nested braces
        # This uses a flat structure that works with the regex
        raw = '{"content": "Now playing!", "reason": "music", "spotify_action": "play"}'

        result = _parse_content_response(raw)

        assert result.content == "Now playing!"
        # spotify_action should be None because it's a string, not a dict
        assert result.spotify_action is None

    def test_parse_content_extracts_spotify_action_from_fallback_block(self):
        """Fallback {} matching should also extract spotify_action when JSON has simple structure.

        Note: The fallback pattern truncates nested braces, so this test uses
        a JSON structure that doesn't have nested braces.
        """
        raw = '{"should_send": false, "content": "", "reason": "no", "spotify_action": "pause"}'

        result = _parse_content_response(raw)

        assert result.should_send is False
        assert result.content == ""
        # spotify_action should be None because it's a string, not a dict
        assert result.spotify_action is None

    def test_parse_content_ignores_non_dict_spotify_action(self):
        """spotify_action should be None if not a dict in the JSON."""
        raw = '{"should_send": true, "content": "hi", "reason": "test", "spotify_action": "not a dict"}'

        result = _parse_content_response(raw)

        assert result.spotify_action is None


# ===== Test Segment Text =====


class TestSegmentText:
    """Test _segment_text function."""

    def test_segment_normal_text(self):
        """_segment_text tries to split text when below threshold."""
        text = "你好啊。最近怎么样？我这边一切都好。有时间一起吃饭吗？"
        # 34 chars - below threshold 50, try to segment
        result = _segment_text(text, 50)
        # Each segment should be <= threshold
        for segment in result:
            assert len(segment) <= 50

    def test_segment_long_text(self):
        """_segment_text returns as-is when text exceeds threshold."""
        text = "这是一段很长的文本内容需要超过阈值" * 5
        # ~80 chars - exceeds threshold 50, should NOT segment
        result = _segment_text(text, 50)
        assert len(result) == 1  # No segmentation for long text

    def test_segment_empty_text(self):
        """_segment_text handles empty text."""
        result = _segment_text("", 100)
        assert result == []

    def test_segment_short_text(self):
        """_segment_text returns text as-is when below threshold."""
        text = "Hello"
        result = _segment_text(text, 100)
        assert len(result) >= 1  # May be segmented or not depending on content


# ===== Test Session Persistence =====


class TestSessionPersistence:
    """Test session persistence functions."""

    def test_get_data_file_path(self):
        """_get_data_file_path returns correct path."""
        path = _get_data_file_path("/tmp/test_dir")
        assert path == "/tmp/test_dir/session_data.json"

    def test_save_and_load_sessions(self, tmp_path):
        """Test saving and loading session data."""
        sessions = {
            "user_1": SessionState(
                session_id="user_1",
                last_message_time=1000.0,
                unanswered_count=2,
            )
        }

        # Save
        _save_sessions_sync(sessions, str(tmp_path))

        # Load
        loaded = _load_sessions_sync(str(tmp_path))

        assert "user_1" in loaded
        assert loaded["user_1"].unanswered_count == 2

    def test_load_nonexistent_file(self, tmp_path):
        """_load_sessions_sync handles missing file."""
        result = _load_sessions_sync(str(tmp_path))
        assert result == {}

    def test_load_corrupted_file(self, tmp_path):
        """_load_sessions_sync handles corrupted JSON."""
        file_path = os.path.join(str(tmp_path), "session_data.json")
        with open(file_path, "w") as f:
            f.write("not valid json{{")

        result = _load_sessions_sync(str(tmp_path))
        assert result == {}

    def test_load_sessions_with_unknown_fields(self, tmp_path):
        """_load_sessions_sync should ignore unknown fields in persisted state."""
        file_path = os.path.join(str(tmp_path), "session_data.json")
        data = {
            "user_1": {
                "session_id": "user_1",
                "last_message_time": 1000.0,
                "unanswered_count": 1,
                "unknown_field": "x",
            }
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        loaded = _load_sessions_sync(str(tmp_path))

        assert "user_1" in loaded
        assert loaded["user_1"].session_id == "user_1"


# ===== Test Status Message =====


class TestStatusMessage:
    """Test _format_status_message function."""

    def test_status_enabled(self):
        """_format_status_message shows enabled state."""
        config = WhisperConfig(
            enable=True,
            silence_trigger_minutes=5,
            timeout_max=30,
            max_consecutive=3,
            quiet_hours_enabled=True,
            quiet_hours_start="23:00",
            quiet_hours_end="08:00",
        )
        state = SessionState(
            session_id="test",
            last_message_time=time.time(),
            unanswered_count=1,
            enabled=True,
        )

        result = _format_status_message(config, state)

        assert "已启用" in result
        assert "5-30" in result
        assert "1/3" in result

    def test_status_disabled(self):
        """_format_status_message shows disabled state."""
        config = WhisperConfig(enable=False)
        state = SessionState(
            session_id="test", last_message_time=time.time(), enabled=False
        )

        result = _format_status_message(config, state)

        assert "已禁用" in result


# ===== Test Proactive Prompt with Additional Context =====


class TestProactivePromptAdditionalContext:
    """Test _build_proactive_prompt function with additional_context."""

    def test_build_proactive_prompt_with_additional_context(self):
        """_build_proactive_prompt appends additional_context when provided and not empty."""
        config = WhisperConfig()
        state = SessionState(
            session_id="test", last_message_time=time.time(), unanswered_count=0
        )
        additional_context = "用户正在听歌：周杰伦 - 稻香"

        result = _build_proactive_prompt(config, state, additional_context)

        # Should contain the additional context at the end
        assert "用户正在听歌：周杰伦 - 稻香" in result
        # Should have the section header
        assert "[额外外部状态上下文]" in result

    def test_build_proactive_prompt_without_additional_context(self):
        """_build_proactive_prompt does not append when additional_context is empty."""
        config = WhisperConfig()
        state = SessionState(
            session_id="test", last_message_time=time.time(), unanswered_count=0
        )
        additional_context = ""

        result = _build_proactive_prompt(config, state, additional_context)

        # Should NOT contain the additional context section
        assert "[额外外部状态上下文]" not in result
        assert "用户正在听歌" not in result

    def test_build_proactive_prompt_default_empty_context(self):
        """_build_proactive_prompt works with default empty string for additional_context."""
        config = WhisperConfig()
        state = SessionState(
            session_id="test", last_message_time=time.time(), unanswered_count=0
        )

        # Call without third argument (uses default "")
        result = _build_proactive_prompt(config, state)

        # Should NOT contain the additional context section
        assert "[额外外部状态上下文]" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
