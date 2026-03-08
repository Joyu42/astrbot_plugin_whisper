# Whisper Plugin

import asyncio
import json
import time
from datetime import datetime, timedelta
import inspect
from typing import Optional, Dict

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from astrbot.api import logger
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.core.message.components import Plain
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.agent.message import (
    AssistantMessageSegment,
    TextPart,
    UserMessageSegment,
)

# Import from modules
from .models import (
    SessionState,
    LLMDecision,
    WhisperConfig,
    parse_config,
)
from .mcp_manager import MCPManager
from .utils import (
    is_quiet_hours,
    get_quiet_hours_end_delay,
    get_silence_trigger_delay,
    should_send_proactive,
    compute_backoff_minutes,
    apply_jitter,
    replace_prompt_placeholders,
    _build_proactive_prompt,
    _parse_content_response,
    _segment_text,
    _format_status_message,
)
from .scheduler import (
    _init_scheduler,
    _schedule_check,
    _cancel_all_session_jobs,
    _cancel_all_checks,
    _set_plugin_instance,
    _get_data_file_path,
    _save_sessions_sync,
    _load_sessions_sync,
    _plugin_instance,
)


# ===== WhisperPlugin Main Class =====


@register(
    "astrbot_plugin_whisper",
    "Your Name",
    "基于对话感知的私聊主动消息插件，单阶段 LLM 调用 + MCP 状态感知 (v0.6.3)",
    "0.6.3",
)
class WhisperPlugin(Star):
    """Whisper - 基于对话感知的私聊主动消息插件"""

    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.context = context
        self.raw_config = config
        self._sessions: Dict[str, SessionState] = {}
        self._save_lock = asyncio.Lock()
        self.scheduler: Optional[AsyncIOScheduler] = None
        self._plugin_instance = self  # Reference for callbacks
        self.mcp_manager = MCPManager()
        _set_plugin_instance(self)  # Set global reference for scheduler

    async def initialize(self):
        """Initialize the plugin - load sessions and start scheduler."""
        # Load persisted sessions
        try:
            data_dir = StarTools.get_data_dir("astrbot_plugin_whisper")
            self._sessions = _load_sessions_sync(data_dir)
        except Exception as e:
            logger.warning(f"[Whisper] 加载会话数据失败: {e}")
            self._sessions = {}

        # Initialize scheduler
        self.scheduler = _init_scheduler(self)

        # Load MCP services
        config = parse_config(self.raw_config)
        await self.mcp_manager.load_services(config)

        # Resume scheduled checks for enabled sessions
        count = 0
        expired_count = 0

        for session_id, state in self._sessions.items():
            if state.enabled and state.next_trigger_time > 0:
                remaining = state.next_trigger_time - time.time()
                if remaining > 0:
                    _schedule_check(self.scheduler, session_id, int(remaining))
                    count += 1
                else:
                    # Expired - reset to silence trigger
                    config = parse_config(self.raw_config, session_id)
                    delay = get_silence_trigger_delay(config)
                    state.next_trigger_time = time.time() + delay
                    _schedule_check(self.scheduler, session_id, delay)
                    logger.info(
                        f"[Whisper] 会话 {session_id} 的定时任务已过期，将重新检查"
                    )
                    expired_count += 1

        logger.info(f"[Whisper] 已恢复 {count} 个定时任务（过期重置: {expired_count}）")

    async def terminate(self):
        """Cleanup - cancel all jobs and save state."""
        # Stop MCP services
        await self.mcp_manager.stop_all()

        if self.scheduler and self.scheduler.running:
            _cancel_all_checks(self.scheduler)
            self.scheduler.shutdown(wait=False)

        # Save sessions
        try:
            data_dir = StarTools.get_data_dir("astrbot_plugin_whisper")
            _save_sessions_sync(self._sessions, data_dir)
            logger.info(f"[Whisper] 已保存 {len(self._sessions)} 个会话状态")
        except Exception as e:
            logger.warning(f"[Whisper] 保存会话数据失败: {e}")

        logger.info("[Whisper] 插件已终止")

    def _get_session(self, session_id: str) -> SessionState:
        """Get or create session state."""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(
                session_id=session_id, last_message_time=time.time()
            )
        return self._sessions[session_id]

    async def _save_session(self, session_id: str):
        """Save session state asynchronously."""
        async with self._save_lock:
            try:
                data_dir = StarTools.get_data_dir("astrbot_plugin_whisper")
                _save_sessions_sync(self._sessions, data_dir)
            except Exception as e:
                logger.warning(f"[Whisper] 保存会话失败: {e}")

    @filter.event_message_type(filter.EventMessageType.PRIVATE_MESSAGE, priority=999)
    async def on_private_message(self, event: AstrMessageEvent):
        """Handle incoming private message - reset silence timer."""
        # Safety check: skip empty messages
        if not event.get_messages():
            return

        session_id = event.unified_msg_origin
        state = self._get_session(session_id)

        # Update state
        state.last_message_time = time.time()
        state.unanswered_count = 0  # Reset on user message
        state.self_id = str(event.get_self_id()) if event.get_self_id() else ""

        # Cancel existing scheduled check
        if self.scheduler:
            has_check = (
                self.scheduler.get_job(f"whisper_check_{session_id}") is not None
            )
            if has_check:
                logger.info("[Whisper] 用户发送了新消息，取消当前主动消息计划。")
            _cancel_all_session_jobs(self.scheduler, session_id)

        # Get config for this session
        config = parse_config(self.raw_config, session_id)

        if state.enabled:
            state.backoff_level = 0

            delay = get_silence_trigger_delay(config)
            trigger_time = datetime.now() + timedelta(seconds=delay)
            state.next_trigger_time = time.time() + delay
            _schedule_check(self.scheduler, session_id, delay)
            # Persist state
            data_dir = StarTools.get_data_dir("astrbot_plugin_whisper")
            _save_sessions_sync(self._sessions, data_dir)
            delay_min = delay // 60
            logger.info(
                f"[Whisper] 用户发送消息，将于 {delay_min} 分钟后 {trigger_time.strftime('%H:%M')} 进行下次检查。"
            )
            await self._save_session(session_id)

        # Don't block event propagation

    async def _send_message(self, session_id: str, decision: LLMDecision):
        """Send message with segmentation support.

        Follows proactive_chat reference plugin patterns for message sending
        and conversation archiving.
        """
        content = decision.content
        prompt = decision.prompt

        config = parse_config(self.raw_config, session_id)
        state = self._get_session(session_id)

        logger.debug(f"[Whisper] 发送主动消息至 {session_id}")
        logger.debug(
            f"[Whisper] 消息内容: {content[:100]}{'...' if len(content) > 100 else ''}"
        )

        # Segment if enabled
        if config.segment_enabled:
            segments = _segment_text(
                content,
                config.segment_threshold,
                config.segment_mode,
                config.segment_regex,
                config.segment_words,
            )
        else:
            segments = [content]

        # Send each segment (matching reference plugin: MessageChain([Plain(text=...)]))
        for idx, segment in enumerate(segments):
            try:
                chain = MessageChain([Plain(text=segment)])
                await self.context.send_message(session_id, chain)

                # Delay between segments (not after last)
                if idx < len(segments) - 1 and config.segment_delay_ms > 0:
                    await asyncio.sleep(config.segment_delay_ms / 1000)
            except Exception as e:
                logger.error(f"[Whisper] 消息发送失败: {e}")
                break

        # Update state
        state.unanswered_count += 1
        state.next_trigger_time = 0

        # Archive to conversation history using proper message objects
        # (matching reference plugin's _finalize_and_reschedule pattern)
        try:
            conv_id = await self.context.conversation_manager.get_curr_conversation_id(
                session_id
            )
            if conv_id:
                # Store the prompt as user message (like proactive_chat)
                user_msg_obj = UserMessageSegment(content=[TextPart(text=prompt)])
                assistant_msg_obj = AssistantMessageSegment(
                    content=[TextPart(text=content)]
                )
                await self.context.conversation_manager.add_message_pair(
                    cid=conv_id,
                    user_message=user_msg_obj,
                    assistant_message=assistant_msg_obj,
                )
        except Exception as e:
            logger.warning(f"[Whisper] 对话归档失败: {e}")

        delay = get_silence_trigger_delay(config)
        next_time = datetime.now() + timedelta(seconds=delay)
        delay_min = delay // 60
        logger.info(
            f"[Whisper] 消息发送完成，将于 {delay_min} 分钟后 {next_time.strftime('%H:%M')} 进行下次检查。"
        )
        state.next_trigger_time = time.time() + delay
        _schedule_check(self.scheduler, session_id, delay)
        data_dir = StarTools.get_data_dir("astrbot_plugin_whisper")
        _save_sessions_sync(self._sessions, data_dir)

    async def _get_filtered_history(
        self, session_id: str
    ) -> tuple[list, str, Optional[str]]:
        """Get filtered conversation history for LLM calls.

        Extracts history from conversation_manager, handles both str and list types,
        filters out tool_calls and tool/system/function roles, and gets persona.

        Args:
            session_id: Session ID

        Returns:
            Tuple of (filtered_history, persona_prompt, conv_id)
        """
        history_text = ""
        pure_history_messages = []
        persona_prompt = ""
        conv_id = None

        try:
            conv_id = await self.context.conversation_manager.get_curr_conversation_id(
                session_id
            )

            # If new session, try to create conversation
            if not conv_id:
                try:
                    conv_id = await self.context.conversation_manager.new_conversation(
                        session_id
                    )
                except Exception:
                    pass

            if conv_id:
                conversation = self.context.conversation_manager.get_conversation(
                    session_id, conv_id
                )
                # AstrBot v4.19.x: get_conversation() is async; keep compatibility.
                if inspect.isawaitable(conversation):
                    conversation = await conversation
                if conversation and conversation.history:
                    # Handle both string and list types (matching reference plugin)
                    try:
                        if isinstance(conversation.history, str):
                            import json as _json

                            pure_history_messages = _json.loads(conversation.history)
                        else:
                            pure_history_messages = conversation.history
                    except (json.JSONDecodeError, TypeError):
                        pure_history_messages = []

                    # Get persona — check conversation-bound first
                    if (
                        conversation
                        and hasattr(conversation, "persona_id")
                        and conversation.persona_id
                    ):
                        try:
                            persona = self.context.persona_manager.get_persona(
                                conversation.persona_id
                            )
                            if inspect.isawaitable(persona):
                                persona = await persona
                            if persona:
                                persona_prompt = persona.system_prompt
                        except Exception:
                            pass

        except Exception as e:
            logger.warning(f"[Whisper] 获取对话历史失败: {e}")

        # Fallback persona: use default if not found from conversation
        if not persona_prompt:
            try:
                default_persona = (
                    await self.context.persona_manager.get_default_persona_v3(
                        umo=session_id
                    )
                )
                if default_persona:
                    persona_prompt = default_persona.get("prompt", "")
            except Exception as e:
                logger.warning(f"[Whisper] 获取人设失败: {e}")

        # Filter out invalid messages that cause API errors:
        # Skip: tool, system, function roles
        # Skip: any message with tool_calls (both assistant with tool_calls and their tool responses)
        filtered_history = []
        config = parse_config(self.raw_config, session_id)
        messages = pure_history_messages[-config.max_history_messages :]

        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                tool_calls = msg.get("tool_calls", [])
                tool_call_id = msg.get("tool_call_id", "")
            else:
                role = getattr(msg, "role", "")
                tool_calls = getattr(msg, "tool_calls", [])
                tool_call_id = getattr(msg, "tool_call_id", "")

            # Skip tool/system/function roles
            if role in ("tool", "system", "function"):
                continue

            # Skip ANY message with tool_calls (both assistant and tool responses)
            if tool_calls:
                continue

            filtered_history.append(msg)

        logger.debug(
            f"[Whisper] 过滤后历史: {len(filtered_history)} 条消息 (来自 {len(pure_history_messages)})"
        )

        return filtered_history, persona_prompt, conv_id

    async def _execute_check(self, session_id: str):
        config = parse_config(self.raw_config, session_id)
        state = self._get_session(session_id)

        now_ts = time.time()
        can_send, guard_reason = should_send_proactive(config, state)
        if not can_send:
            if guard_reason == "quiet_hours":
                delay = get_quiet_hours_end_delay(config)
                if delay > 0:
                    state.next_trigger_time = now_ts + delay
                    _schedule_check(self.scheduler, session_id, delay)
                return
            return

        filtered_history, persona_prompt, _ = await self._get_filtered_history(
            session_id
        )
        # Get MCP context
        additional_context = await self.mcp_manager.get_combined_context(config)
        prompt = _build_proactive_prompt(config, state, additional_context)

        llm_response = None
        try:
            provider_id = await self.context.get_current_chat_provider_id(session_id)
            llm_response = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
                contexts=filtered_history,
                system_prompt=persona_prompt,
            )
        except Exception:
            try:
                provider = self.context.get_using_provider(umo=session_id)
                if provider:
                    llm_response = await provider.text_chat(
                        prompt=prompt,
                        contexts=filtered_history,
                        system_prompt=persona_prompt,
                    )
            except Exception:
                llm_response = None

        decision: Optional[LLMDecision]
        if llm_response and llm_response.completion_text:
            decision = _parse_content_response(llm_response.completion_text)
            decision.prompt = prompt
        else:
            decision = None

        # Log LLM decision
        if decision:
            logger.info(
                f"[Whisper] LLM 决策: should_send={decision.should_send}, "
                f"reason={decision.reason}"
            )

        if decision and decision.should_send is True and decision.content.strip():
            # Format suggestion from MCP services
            suggestion = self.mcp_manager.format_combined_suggestions(decision)
            if suggestion and config.spotify_suggest_enabled:
                decision.content += suggestion
            await self._send_message(session_id, decision)
            state.backoff_level = 0
            return

        # Backoff scheduling (should_send=False or parse failed)
        base = config.silence_trigger_minutes
        cap = config.timeout_max
        delay_minutes = compute_backoff_minutes(
            base_minutes=base, level=state.backoff_level, cap_minutes=cap
        )
        state.backoff_level += 1
        delay_seconds = apply_jitter(delay_minutes * 60)
        state.next_trigger_time = now_ts + delay_seconds
        _schedule_check(self.scheduler, session_id, delay_seconds)
        next_time = datetime.now() + timedelta(seconds=delay_seconds)
        logger.info(
            f"[Whisper] 不发送主动消息，将于 {delay_minutes} 分钟后 {next_time.strftime('%H:%M')} 进行下次检查。"
        )

    # ===== Commands =====

    @filter.command("whisper")
    async def cmd_whisper(self, event: AstrMessageEvent):
        """查看 Whisper 状态"""
        session_id = event.unified_msg_origin
        state = self._get_session(session_id)
        config = parse_config(self.raw_config, session_id)

        msg = _format_status_message(config, state)
        yield event.plain_result(msg)

    @filter.command("whisper_on")
    async def cmd_whisper_on(self, event: AstrMessageEvent):
        """启用 Whisper"""
        session_id = event.unified_msg_origin
        state = self._get_session(session_id)
        state.enabled = True
        state.last_message_time = time.time()

        config = parse_config(self.raw_config, session_id)
        delay = get_silence_trigger_delay(config)
        state.next_trigger_time = time.time() + delay
        _schedule_check(self.scheduler, session_id, delay)
        # Persist state
        data_dir = StarTools.get_data_dir("astrbot_plugin_whisper")
        _save_sessions_sync(self._sessions, data_dir)
        yield event.plain_result("✅ Whisper 已启用")

    @filter.command("whisper_off")
    async def cmd_whisper_off(self, event: AstrMessageEvent):
        """禁用 Whisper"""
        session_id = event.unified_msg_origin
        state = self._get_session(session_id)
        state.enabled = False

        if self.scheduler:
            _cancel_all_session_jobs(self.scheduler, session_id)

        await self._save_session(session_id)
        yield event.plain_result("✅ Whisper 已禁用")
