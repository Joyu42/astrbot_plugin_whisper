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
    pending_phase: str = "predict"  # Phase-aware persistence: "predict" or "send"

    backoff_level: int = 0
    last_check_time: float = 0.0


@dataclass
class LLMDecision:
    """Phase 2 LLM content generation result."""

    should_send: Optional[bool] = None
    content: str = ""  # Message content
    reason: str = ""  # Generation reason
    prompt: str = ""  # The prompt sent to LLM (for archiving)


@dataclass
class PredictionResult:
    """Phase 1 LLM prediction result."""

    minutes: int = 0  # LLM predicted minutes
    reason: str = ""  # Prediction reason (optional, for logging)
    prompt: str = ""  # The prompt sent to LLM (for archiving)
    valid: bool = True  # Whether parsing was successful


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
    segment_enabled: bool = True
    segment_max_length: int = 100
    segment_delay_ms: int = 1500
    proactive_prompt: str = ""
    prediction_prompt: str = ""


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
        segment_max_length=raw_config.get("segment_max_length", 100),
        segment_delay_ms=raw_config.get("segment_delay_ms", 1500),
        proactive_prompt=raw_config.get("proactive_prompt", ""),
        prediction_prompt=raw_config.get("prediction_prompt", ""),
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
