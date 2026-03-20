"""MCP Manager for Whisper plugin."""

import asyncio
import shlex
from typing import Optional

from astrbot.api import logger
from .models import LLMDecision, WhisperConfig
from .mcp_service import MCPService
from .spotify_service import SpotifyMCPService


class MCPManager:
    """Manager for MCP services."""

    def __init__(self) -> None:
        """Initialize the MCP manager."""
        self._services: dict[str, MCPService] = {}

    @staticmethod
    def _service_name(entry: object) -> str:
        if isinstance(entry, str):
            return entry.strip().lower()
        if isinstance(entry, dict):
            name = entry.get("name") or entry.get("service")
            if isinstance(name, str):
                return name.strip().lower()
        return ""

    async def load_services(self, config: WhisperConfig) -> None:
        """Load and start MCP services based on configuration.

        Args:
            config: The WhisperConfig containing MCP service settings.
        """
        if not config.mcp_enabled:
            return

        for entry in config.mcp_services:
            service_name = self._service_name(entry)
            if service_name == "spotify":
                command = shlex.split(config.spotify_mcp_command)
                service = SpotifyMCPService(command)
                self._services["spotify"] = service
                await service.start()

    async def stop_all(self) -> None:
        """Stop all MCP services."""
        for service in self._services.values():
            await service.stop()
        self._services.clear()

    async def get_combined_context(self, config: WhisperConfig) -> str:
        """Get combined context from all enabled MCP services.

        Args:
            config: The WhisperConfig containing context settings.

        Returns:
            str: Combined context from all enabled services, or empty string if none.
        """
        contexts = []

        if config.spotify_context_enabled and "spotify" in self._services:
            try:
                logger.info("[Whisper] 正在获取 Spotify MCP 上下文...")
                context = await asyncio.wait_for(
                    self._services["spotify"].get_context(), timeout=3.0
                )
                if context:
                    contexts.append(context)
                    logger.info(f"[Whisper] Spotify MCP 上下文获取成功: {context}")
                else:
                    logger.info("[Whisper] Spotify 未在播放，无上下文")
            except asyncio.TimeoutError:
                logger.warning("[Whisper] Spotify MCP 获取超时")
            except Exception as e:
                logger.warning(f"[Whisper] Spotify MCP 获取失败: {e}")

        return "".join(contexts)

    def format_combined_suggestions(self, decision: LLMDecision) -> str:
        """Format combined suggestions from MCP services.

        Args:
            decision: The LLMDecision containing action details.

        Returns:
            str: Combined suggestions string.
        """
        if decision.spotify_action is None:
            return ""

        if "spotify" not in self._services:
            return ""

        return self._services["spotify"].format_suggestion(decision.spotify_action)
