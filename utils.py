# Whisper Plugin - Utility Functions

import json
import random
import re
import time
from datetime import datetime, timedelta
from datetime import time as datetime_time
from typing import Optional

# Import dataclasses from models module
from .models import SessionState, LLMDecision, WhisperConfig


# ===== Prompt Template Constants =====


DEFAULT_PROMPT_TEMPLATE = """你正在与一个朋友私聊。
现在是 {{current_time}}，对方已经有一段时间没有回复了（你已主动发送了 {{unanswered_count}} 条消息未得到回复）。

请判断你是否应该主动发送一条消息。
考虑以下因素：
1. 对话的上下文和氛围
2. 上次对话的内容是否需要跟进
3. 你已经发送了多少条未回复的消息（避免骚扰）
4. 当前时间是否合适发消息

请以 JSON 格式回复：
{"should_send": true/false, "content": "你要发送的消息内容", "reason": "你做出这个判断的原因", "spotify_action": {"type": "recommend", "query": "..."}}

规则：
- 只输出 JSON，不要有任何其他文字。
- 如果 should_send=false，content 必须是空字符串。
- 如果 should_send=true，content 必须是非空字符串。
- 如果你在上下文中看到用户正在听某首歌或者使用某个功能，且你在回复文字中想给出相关的动作建议（如：切歌、播放某个特定类型/专辑歌曲...），你可以可选地在 JSON 中附加 "spotify_action": {"type": "recommend", "query": "..."}"""


# ===== Time and Delay Functions =====


def is_quiet_hours(config: WhisperConfig) -> bool:
    """
    Check if current time is within quiet hours.

    Args:
        config: WhisperConfig with quiet hours settings

    Returns:
        True if within quiet hours, False otherwise
    """
    if not config.quiet_hours_enabled:
        return False

    now = datetime.now()
    current_time = now.time()

    start_str = config.quiet_hours_start
    end_str = config.quiet_hours_end

    start_hour, start_minute = map(int, start_str.split(":"))
    end_hour, end_minute = map(int, end_str.split(":"))

    start_time = datetime_time(start_hour, start_minute)
    end_time = datetime_time(end_hour, end_minute)

    # Handle cross-midnight case (e.g., 23:00-08:00)
    if start_time <= end_time:
        # Normal case: start and end on same day
        return start_time <= current_time <= end_time
    else:
        # Cross-midnight case: either after start OR before end
        return current_time >= start_time or current_time <= end_time


def get_quiet_hours_end_delay(config: WhisperConfig) -> int:
    """
    Calculate seconds until quiet hours end.

    Args:
        config: WhisperConfig with quiet hours settings

    Returns:
        Seconds until quiet hours end, or 0 if not in quiet hours
    """
    if not config.quiet_hours_enabled or not is_quiet_hours(config):
        return 0

    now = datetime.now()
    current_time = now.time()

    end_str = config.quiet_hours_end
    end_hour, end_minute = map(int, end_str.split(":"))
    end_time = datetime_time(end_hour, end_minute)

    # Handle cross-midnight case (e.g., 23:00-08:00)
    start_str = config.quiet_hours_start
    start_hour, start_minute = map(int, start_str.split(":"))
    start_time = datetime_time(start_hour, start_minute)

    if start_time > end_time:
        # Cross-midnight: quiet hours end is tomorrow
        if current_time >= start_time:
            # After midnight, next day's end
            tomorrow = now + timedelta(days=1)
            end_dt = tomorrow.replace(
                hour=end_hour, minute=end_minute, second=0, microsecond=0
            )
        else:
            # Before midnight, today
            end_dt = now.replace(
                hour=end_hour, minute=end_minute, second=0, microsecond=0
            )
    else:
        # Same day
        end_dt = now.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)
        if current_time > end_time:
            # Already passed end time today, next day
            end_dt += timedelta(days=1)

    delay = (end_dt - now).total_seconds()
    return max(int(delay), 60)  # At least 60 seconds


def get_silence_trigger_delay(config: WhisperConfig) -> int:
    """
    Get fixed silence trigger delay (converted to seconds).

    Args:
        config: WhisperConfig with timeout settings

    Returns:
        Fixed delay in seconds based on silence_trigger_minutes
    """
    return config.silence_trigger_minutes * 60


def should_send_proactive(
    config: WhisperConfig, state: SessionState
) -> tuple[bool, str]:
    """
    Check if proactive message should be sent.

    Args:
        config: WhisperConfig with settings
        state: Current session state

    Returns:
        Tuple of (can_send, reason_if_blocked)
        reason is: "disabled", "quiet_hours", "max_consecutive", or "" (can send)
    """
    if not config.enable:
        return False, "disabled"

    if is_quiet_hours(config):
        return False, "quiet_hours"

    if state.unanswered_count >= config.max_consecutive:
        return False, "max_consecutive"

    return True, ""


# ===== Prompt Builders =====


def replace_prompt_placeholders(template: str, state: SessionState, config=None) -> str:
    """
    Replace placeholders in prompt template.

    Args:
        template: Prompt template with {{current_time}} and {{unanswered_count}} placeholders
        state: Current session state
        config: WhisperConfig (optional) for timeout_max

    Returns:
        Template with placeholders replaced
    """
    now = datetime.now()
    current_time_str = now.strftime("%Y-%m-%d %H:%M")

    result = template.replace("{{current_time}}", current_time_str)
    result = result.replace("{{unanswered_count}}", str(state.unanswered_count))

    # Add timeout_max if config is provided
    if config:
        result = result.replace("{{timeout_max}}", str(config.timeout_max))
    else:
        result = result.replace("{{timeout_max}}", "30")

    return result


def _build_proactive_prompt(
    config: WhisperConfig, state: SessionState, additional_context: str = ""
) -> str:
    """
    Build the prompt for LLM decision.

    Note: Conversation history is passed to LLM via 'contexts' parameter,
    not embedded in the prompt text. This keeps the prompt clean and avoids
    redundant information (RAG memory already provides context).

    Args:
        config: WhisperConfig
        state: SessionState
        additional_context: Additional external state context to append (e.g., Spotify status)

    Returns:
        Formatted prompt string
    """
    template = config.proactive_prompt or DEFAULT_PROMPT_TEMPLATE
    prompt = replace_prompt_placeholders(template, state, config)

    # History is passed via contexts, no need to include in prompt text
    # (matching proactive_chat behavior)

    # Append additional_context if provided and not empty
    if additional_context:
        prompt += f"\n\n[额外外部状态上下文]\n{additional_context}"

    return prompt


# ===== LLM Response Parsers =====


def _parse_content_response(raw_text: str) -> LLMDecision:
    """
    Parse Phase 2 LLM response for content generation.

    Args:
        raw_text: Raw text response from LLM

    Returns:
        LLMDecision object with content and reason
    """
    if not raw_text or not raw_text.strip():
        return LLMDecision(reason="empty_response")

    # Try direct JSON parse
    try:
        data = json.loads(raw_text)
        should_send = data.get("should_send")
        if not isinstance(should_send, bool):
            should_send = None

        content = str(data.get("content", ""))
        reason = str(data.get("reason", ""))
        spotify_action = (
            data.get("spotify_action")
            if isinstance(data.get("spotify_action"), dict)
            else None
        )
        if should_send is None:
            reason = "missing_should_send"
        elif should_send is True and not content:
            reason = "empty_content"
        return LLMDecision(
            should_send=should_send,
            content=content,
            reason=reason,
            spotify_action=spotify_action,
        )
    except json.JSONDecodeError:
        pass

    # Try regex extraction for JSON object with content field
    json_pattern = r'\{[^{}]*"content"[^{}]*\}'
    match = re.search(json_pattern, raw_text)
    if match:
        try:
            data = json.loads(match.group(0))
            should_send = data.get("should_send")
            if not isinstance(should_send, bool):
                should_send = None
            content = str(data.get("content", ""))
            reason = str(data.get("reason", ""))
            spotify_action = (
                data.get("spotify_action")
                if isinstance(data.get("spotify_action"), dict)
                else None
            )
            if should_send is None:
                reason = "missing_should_send"
            elif should_send is True and not content:
                reason = "empty_content"
            return LLMDecision(
                should_send=should_send,
                content=content,
                reason=reason,
                spotify_action=spotify_action,
            )
        except:
            pass

    # Last resort: try to match any JSON-like object
    fallback_pattern = r"\{[\s\S]*?\}"
    match = re.search(fallback_pattern, raw_text)
    if match:
        try:
            data = json.loads(match.group(0))
            should_send = data.get("should_send")
            if not isinstance(should_send, bool):
                should_send = None
            content = str(data.get("content", ""))
            reason = str(data.get("reason", ""))
            spotify_action = (
                data.get("spotify_action")
                if isinstance(data.get("spotify_action"), dict)
                else None
            )
            if should_send is None:
                reason = "missing_should_send"
            elif should_send is True and not content:
                reason = "empty_content"
            return LLMDecision(
                should_send=should_send,
                content=content,
                reason=reason,
                spotify_action=spotify_action,
            )
        except:
            pass

    # All parsing failed
    return LLMDecision(reason="json_parse_failed")


# ===== Text Processing =====


def compute_backoff_minutes(base_minutes: int, level: int, cap_minutes: int) -> int:
    try:
        base = int(base_minutes)
        lvl = int(level)
        cap = int(cap_minutes)
    except (TypeError, ValueError):
        return 1

    if base <= 0:
        base = 1
    if lvl < 0:
        lvl = 0
    if cap <= 0:
        cap = base

    minutes = base * (2**lvl)
    return min(minutes, cap)


def apply_jitter(seconds: int, ratio: float = 0.1) -> int:
    try:
        sec = int(seconds)
    except (TypeError, ValueError):
        sec = 1

    if sec <= 0:
        sec = 1

    r = ratio
    if r < 0:
        r = 0
    if r > 0.9:
        r = 0.9

    low = max(int(sec * (1 - r)), 1)
    high = max(int(sec * (1 + r)), 1)
    if high < low:
        high = low
    return random.randint(low, high)


def _segment_text(
    text: str,
    threshold: int = 150,
    mode: str = "regex",
    regex_pattern: str = r"([。？！~…\n])",
    split_words: str = "。！？～…\n",
) -> list[str]:
    if not text:
        return []

    if len(text) <= threshold:
        return [text]

    segments = []
    max_len = threshold

    if mode == "regex":
        import re

        parts = re.split(regex_pattern, text)
    else:
        parts = []
        current = ""
        for char in text:
            current += char
            if char in split_words:
                parts.append(current)
                current = ""
        if current:
            parts.append(current)

    rejoined = []
    current = ""
    for p in parts:
        current += p
        if p and p[-1] in "。？！~…\n":
            rejoined.append(current)
            current = ""
    if current:
        rejoined.append(current)

    current = ""
    for part in rejoined:
        if len(current) + len(part) <= max_len:
            current += part
        else:
            if current:
                segments.append(current)
            while len(part) > max_len:
                segments.append(part[:max_len])
                part = part[max_len:]
            current = part

    if current:
        segments.append(current)

    return segments if segments else [text]


# ===== Status Formatting =====


def _format_status_message(config, state) -> str:
    """Format status message for user."""
    enabled_text = "已启用" if state.enabled else "已禁用"
    timeout_range = f"{config.silence_trigger_minutes}-{config.timeout_max} 分钟"
    next_check = "未计划" if state.next_trigger_time == 0 else "已计划"

    return f"""=== Whisper 状态 ===
当前状态: {enabled_text}
沉默超时: {timeout_range}
未回复计数: {state.unanswered_count}/{config.max_consecutive}
安静时段: {"已启用 " + config.quiet_hours_start + "-" + config.quiet_hours_end if config.quiet_hours_enabled else "已禁用"}"""
