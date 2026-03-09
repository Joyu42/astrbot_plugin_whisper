"""Spotify MCP Service implementation for Whisper plugin."""

import traceback
from typing import TYPE_CHECKING, Any

from astrbot.api import logger
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


class SpotifyMCPService:
    """Spotify MCP service implementation."""

    def __init__(self, command: list[str]) -> None:
        """Initialize the Spotify MCP service.

        Args:
            command: The command to launch the MCP server.
        """
        self.command = command
        self._cmd = command[0]
        self._args = command[1:]

    async def start(self) -> None:
        """Start the MCP service. No-op for local context manager approach."""
        pass

    async def stop(self) -> None:
        """Stop the MCP service. No-op for local context manager approach."""
        pass

    async def get_context(self) -> str:
        """Get the current context from Spotify.

        Returns:
            str: The current playing context as a string, or empty string if not playing.
        """
        import os

        # 使用 AstrBot 工作目录作为 cwd，确保 MCP server 能找到相对路径的配置文件
        astrbot_cwd = os.getcwd()
        server_params = StdioServerParameters(
            command=self._cmd, args=self._args, cwd=astrbot_cwd
        )
        logger.debug(f"[Whisper] 启动 Spotify MCP: {self.command}")
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool("getNowPlaying", {})
                    # Check if the result indicates something is playing
                    # The result structure depends on the MCP server implementation
                    if hasattr(result, "content"):
                        for content in result.content:
                            if hasattr(content, "text") and content.text:
                                import json

                                try:
                                    data = json.loads(content.text)
                                except (json.JSONDecodeError, ValueError):
                                    logger.debug(
                                        f"[Whisper] Spotify MCP 返回非 JSON 文本: {content.text[:200]}"
                                    )
                                    continue
                                logger.debug(
                                    f"[Whisper] Spotify MCP 返回原始数据: {data}"
                                )
                                if data.get("is_playing"):
                                    name = data.get("name", "Unknown")
                                    artist = data.get("artist", "Unknown")
                                    device = data.get("device", "Unknown")
                                    context_str = f"用户当前正在听 Spotify: {name} - {artist} (在设备 {device} 上)"
                                    logger.info(
                                        f"[Whisper] Spotify 解析成功: {name} - {artist}"
                                    )
                                    return context_str
                    logger.info("[Whisper] Spotify 当前未在播放")
                    return ""
        except Exception as e:
            logger.warning(
                f"[Whisper] Spotify MCP 调用失败: {e}\n{traceback.format_exc()}"
            )
            return ""

    def format_suggestion(self, action: dict[str, Any]) -> str:
        """Format a Spotify action into a human-readable string.

        Args:
            action: A dictionary containing action details with 'type' and 'query' keys.

        Returns:
            str: The formatted suggestion string, or empty string for invalid actions.
        """
        action_type = action.get("type")
        if action_type != "recommend":
            return ""

        query = action.get("query")
        if not query:
            return ""

        return f"\n[Spotify 推荐建议: {query}]"
