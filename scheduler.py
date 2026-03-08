# Whisper Plugin - Scheduler and Persistence Module

import json
import os
import asyncio
from dataclasses import asdict, fields
from datetime import datetime, timedelta
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger

from astrbot.api import logger
from .models import SessionState


# ===== APScheduler Functions =====


def _init_scheduler(plugin_instance):
    """Initialize and start the scheduler."""
    import zoneinfo

    tz = None
    # Try to get timezone from AstrBot global config
    try:
        tz_key = plugin_instance.context.get_config().get("timezone")
        if tz_key:
            tz = zoneinfo.ZoneInfo(tz_key)
    except Exception:
        pass

    # Fallback: use UTC to avoid Docker 'local' timezone issues
    if tz is None:
        try:
            tz = zoneinfo.ZoneInfo("UTC")
        except Exception:
            import datetime as _dt

            tz = _dt.timezone.utc

    scheduler = AsyncIOScheduler(timezone=tz)
    if not scheduler.running:
        scheduler.start()
    return scheduler


def _cancel_all_session_jobs(scheduler, session_id: str):
    """Cancel whisper_check job for a session."""
    try:
        scheduler.remove_job(f"whisper_check_{session_id}")
    except:
        pass


def _cancel_all_checks(scheduler):
    """Cancel all scheduled whisper check jobs."""
    for job in scheduler.get_jobs():
        if job.id.startswith("whisper_check_"):
            try:
                scheduler.remove_job(job.id)
            except:
                pass


def _schedule_check(scheduler, session_id: str, delay_seconds: int):
    _cancel_check_job(scheduler, session_id)
    run_date = datetime.now() + timedelta(seconds=delay_seconds)
    scheduler.add_job(
        func=_on_check_timeout_wrapper,
        trigger=DateTrigger(run_date=run_date),
        args=[session_id],
        id=f"whisper_check_{session_id}",
        replace_existing=True,
    )


def _cancel_check_job(scheduler, session_id: str):
    try:
        scheduler.remove_job(f"whisper_check_{session_id}")
    except:
        pass


# Global reference to plugin instance for scheduler callbacks
_plugin_instance: Optional["WhisperPlugin"] = None


def _set_plugin_instance(plugin: "WhisperPlugin"):
    """Set the global plugin instance for scheduler callbacks."""
    global _plugin_instance
    _plugin_instance = plugin


async def _on_check_timeout_wrapper(session_id: str):
    if _plugin_instance:
        await _plugin_instance._execute_check(session_id)
    else:
        logger.warning(f"[Whisper] 未找到会话 {session_id} 的插件实例")


# ===== Session Persistence IO =====


def _get_data_file_path(data_dir: Optional[str] = None) -> str:
    """
    Get the path to session data file.

    Args:
        data_dir: Plugin data directory path. If None, uses current directory.

    Returns:
        Full path to session_data.json
    """
    if data_dir is None:
        data_dir = "."
    return os.path.join(data_dir, "session_data.json")


def _load_sessions_sync(data_dir: Optional[str] = None) -> dict:
    """
    Load sessions from JSON file (synchronous).

    Args:
        data_dir: Plugin data directory path

    Returns:
        Dictionary of session_id -> SessionState
    """
    file_path = _get_data_file_path(data_dir)

    if not os.path.exists(file_path):
        return {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        allowed_keys = {f.name for f in fields(SessionState)}

        # Convert dict back to SessionState
        sessions = {}
        for session_id, state_dict in data.items():
            if isinstance(state_dict, dict):
                filtered = {k: v for k, v in state_dict.items() if k in allowed_keys}
                sessions[session_id] = SessionState(**filtered)
        return sessions
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"[Whisper] 加载会话数据失败: {e}")
        return {}


def _save_sessions_sync(sessions: dict, data_dir: Optional[str] = None):
    """
    Save sessions to JSON file (synchronous).

    Args:
        sessions: Dictionary of session_id -> SessionState
        data_dir: Plugin data directory path
    """
    file_path = _get_data_file_path(data_dir)

    # Convert SessionState to dict
    data = {}
    for session_id, state in sessions.items():
        data[session_id] = asdict(state)

    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"[Whisper] 保存会话数据失败: {e}")
