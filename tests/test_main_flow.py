import os
import sys
import time
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def _install_astrbot_stubs():
    astrbot_module = types.ModuleType("astrbot")

    api_module = types.ModuleType("astrbot.api")
    api_module.logger = MagicMock()

    event_module = types.ModuleType("astrbot.api.event")

    class _Filter:
        class EventMessageType:
            PRIVATE_MESSAGE = "private"

        @staticmethod
        def event_message_type(*_args, **_kwargs):
            return lambda fn: fn

        @staticmethod
        def command(*_args, **_kwargs):
            return lambda fn: fn

    event_module.filter = _Filter
    event_module.AstrMessageEvent = object

    star_module = types.ModuleType("astrbot.api.star")

    class _Star:
        def __init__(self, context):
            self.context = context

    class _StarTools:
        @staticmethod
        def get_data_dir(_name):
            return "/tmp"

    def _register(*_args, **_kwargs):
        return lambda cls: cls

    star_module.Context = object
    star_module.Star = _Star
    star_module.StarTools = _StarTools
    star_module.register = _register

    components_module = types.ModuleType("astrbot.core.message.components")

    class _Plain:
        def __init__(self, text):
            self.text = text

    components_module.Plain = _Plain

    result_module = types.ModuleType("astrbot.core.message.message_event_result")

    class _MessageChain(list):
        pass

    result_module.MessageChain = _MessageChain

    agent_message_module = types.ModuleType("astrbot.core.agent.message")

    class _TextPart:
        def __init__(self, text):
            self.text = text

    class _UserMessageSegment:
        def __init__(self, content):
            self.content = content

    class _AssistantMessageSegment:
        def __init__(self, content):
            self.content = content

    agent_message_module.TextPart = _TextPart
    agent_message_module.UserMessageSegment = _UserMessageSegment
    agent_message_module.AssistantMessageSegment = _AssistantMessageSegment

    sys.modules["astrbot"] = astrbot_module
    sys.modules["astrbot.api"] = api_module
    sys.modules["astrbot.api.event"] = event_module
    sys.modules["astrbot.api.star"] = star_module
    sys.modules["astrbot.core.message.components"] = components_module
    sys.modules["astrbot.core.message.message_event_result"] = result_module
    sys.modules["astrbot.core.agent.message"] = agent_message_module


_install_astrbot_stubs()

from astrbot_plugin_whisper.main import WhisperPlugin
from astrbot_plugin_whisper.models import LLMDecision, SessionState


@pytest.mark.asyncio
async def test_execute_check_rechecks_quiet_hours_before_send():
    state = SessionState(session_id="session_1", last_message_time=time.time())

    plugin = SimpleNamespace()
    plugin.raw_config = {
        "enable": True,
        "quiet_hours_enabled": False,
        "silence_trigger_minutes": 5,
        "timeout_max": 30,
    }
    plugin.scheduler = MagicMock()
    plugin.mcp_manager = MagicMock()
    plugin.mcp_manager.get_combined_context = AsyncMock(return_value="")
    plugin.mcp_manager.format_combined_suggestions = MagicMock(return_value="")
    plugin.context = MagicMock()
    plugin.context.get_current_chat_provider_id = AsyncMock(return_value="provider_1")
    plugin.context.llm_generate = AsyncMock(
        return_value=SimpleNamespace(
            completion_text='{"should_send": true, "content": "hello", "reason": "ok"}'
        )
    )
    plugin._get_session = lambda _session_id: state
    plugin._get_filtered_history = AsyncMock(return_value=([], "", None))
    plugin._send_message = AsyncMock()

    with (
        patch(
            "astrbot_plugin_whisper.main.should_send_proactive",
            side_effect=[(True, ""), (False, "quiet_hours")],
        ),
        patch(
            "astrbot_plugin_whisper.main.get_quiet_hours_end_delay", return_value=120
        ),
        patch("astrbot_plugin_whisper.main._schedule_check") as mock_schedule_check,
    ):
        await WhisperPlugin._execute_check(plugin, "session_1")

    plugin._send_message.assert_not_called()
    mock_schedule_check.assert_called_once_with(plugin.scheduler, "session_1", 120)
    assert state.next_trigger_time > 0


@pytest.mark.asyncio
async def test_send_message_failure_does_not_mutate_state_or_archive():
    state = SessionState(session_id="session_2", last_message_time=time.time())

    plugin = SimpleNamespace()
    plugin.raw_config = {
        "segment_enabled": False,
        "silence_trigger_minutes": 5,
        "timeout_max": 30,
    }
    plugin.scheduler = MagicMock()
    plugin.context = MagicMock()
    plugin.context.send_message = AsyncMock(side_effect=RuntimeError("send failed"))
    plugin.context.conversation_manager = MagicMock()
    plugin.context.conversation_manager.get_curr_conversation_id = AsyncMock(
        return_value="conv-1"
    )
    plugin.context.conversation_manager.add_message_pair = AsyncMock()
    plugin._get_session = lambda _session_id: state
    plugin._save_session = AsyncMock()

    with patch("astrbot_plugin_whisper.main._schedule_check") as mock_schedule_check:
        await WhisperPlugin._send_message(
            plugin,
            "session_2",
            LLMDecision(content="hello", prompt="p", should_send=True),
        )

    assert state.unanswered_count == 0
    assert state.next_trigger_time == 0
    plugin.context.conversation_manager.add_message_pair.assert_not_called()
    plugin._save_session.assert_not_called()
    mock_schedule_check.assert_not_called()
