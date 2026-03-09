# Whisper Plugin - Data Models and Configuration

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict


@dataclass
class SessionState:
    """Each session's state."""

    session_id: str  # unified_msg_origin
    last_message_time: float  # Timestamp of last user message
    unanswered_count: int = 0  # Consecutive unanswered proactive message count
    next_trigger_time: float = 0.0  # Next scheduled trigger time
    enabled: bool = True  # Whether this session is enabled
    self_id: str = ""  # Bot's own ID

    backoff_level: int = 0
    last_check_time: float = 0.0


@dataclass
class LLMDecision:
    """Phase 2 LLM content generation result."""

    should_send: Optional[bool] = None
    content: str = ""  # Message content
    reason: str = ""  # Generation reason
    prompt: str = ""  # The prompt sent to LLM (for archiving)
    spotify_action: Optional[dict] = None


@dataclass
class WhisperConfig:
    """Runtime configuration."""

    enable: bool = True
    silence_trigger_minutes: int = 5
    timeout_max: int = 30
    max_consecutive: int = 3
    quiet_hours_enabled: bool = True
    quiet_hours_start: str = "23:00"
    quiet_hours_end: str = "08:00"
    max_history_messages: int = 20
    # Segment settings
    segment_enabled: bool = True
    segment_threshold: int = 150  # If text > threshold, don't segment
    segment_mode: str = "regex"  # "regex" or "words"
    segment_regex: str = r".*?[。？！~…\n]+|.+$"
    segment_words: str = "。！？～…\n"
    segment_delay_ms: int = 1500
    proactive_prompt: str = ""
    mcp_enabled: bool = False
    mcp_services: list = field(default_factory=list)
    spotify_context_enabled: bool = False
    spotify_suggest_enabled: bool = False
    spotify_mcp_command: str = "node data/mcp_servers/spotify-mcp-server/build/index.js"
    llm_provider_id: str = ""


def parse_config(raw_config: dict, session_id: Optional[str] = None) -> WhisperConfig:
    """
    Merge global config with session-level overrides.

    Args:
        raw_config: Raw configuration dictionary from plugin config
        session_id: Session ID for session-specific overrides

    Returns:
        Merged WhisperConfig
    """
    # Backward compatibility: support old timeout_min key
    silence_trigger_minutes = raw_config.get("silence_trigger_minutes")
    if silence_trigger_minutes is None and "timeout_min" in raw_config:
        # Use old timeout_min value if silence_trigger_minutes not set
        silence_trigger_minutes = raw_config.get("timeout_min", 5)
    elif silence_trigger_minutes is None:
        silence_trigger_minutes = 5

    # Build WhisperConfig from global settings
    config = WhisperConfig(
        enable=raw_config.get("enable", True),
        silence_trigger_minutes=silence_trigger_minutes,
        timeout_max=raw_config.get("timeout_max", 30),
        max_consecutive=raw_config.get("max_consecutive", 3),
        quiet_hours_enabled=raw_config.get("quiet_hours_enabled", True),
        quiet_hours_start=raw_config.get("quiet_hours_start", "23:00"),
        quiet_hours_end=raw_config.get("quiet_hours_end", "08:00"),
        max_history_messages=raw_config.get("max_history_messages", 20),
        segment_enabled=raw_config.get("segment_enabled", True),
        segment_threshold=raw_config.get("segment_threshold", 150),
        segment_mode=raw_config.get("segment_mode", "regex"),
        segment_regex=raw_config.get("segment_regex", r".*?[。？！~…\n]+|.+$"),
        segment_words=raw_config.get("segment_words", "。！？～…\n"),
        segment_delay_ms=raw_config.get("segment_delay_ms", 1500),
        proactive_prompt=raw_config.get("proactive_prompt", ""),
        mcp_enabled=raw_config.get("mcp_enabled", False),
        mcp_services=raw_config.get("mcp_services", []),
        spotify_context_enabled=raw_config.get("spotify_context_enabled", False),
        spotify_suggest_enabled=raw_config.get("spotify_suggest_enabled", False),
        spotify_mcp_command=raw_config.get(
            "spotify_mcp_command",
            "node data/mcp_servers/spotify-mcp-server/build/index.js",
        ),
        llm_provider_id=raw_config.get("llm_provider_id", ""),
    )

    # Apply session-level overrides if available
    if session_id:
        session_configs = raw_config.get("session_configs", {})
        session_override = session_configs.get(session_id, {})
        if session_override:
            # Override with session-specific values
            for key in session_override:
                if hasattr(config, key):
                    setattr(config, key, session_override[key])

    return config
