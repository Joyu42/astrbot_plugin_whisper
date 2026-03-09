"""Spotify MCP Service implementation for Whisper plugin."""

import re
import traceback
from typing import TYPE_CHECKING, Any

from astrbot.api import logger
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def _parse_markdown_now_playing(text: str) -> dict | None:
    """解析 Spotify MCP 返回的 Markdown 格式文本。

    例如:
        # Currently Playing
        **Track**: "片っぽ"
        **Artist**: eill
        **Device**: EdgeMac (Computer)

    Returns:
        解析后的字典 {"track": ..., "artist": ..., "device": ...}，
        如果文本不包含播放信息则返回 None。
    """
    if "Currently Playing" not in text and "Track" not in text:
        return None

    result = {}
    # 匹配 **Key**: Value 或 **Key**: "Value" 格式
    for match in re.finditer(r'\*\*(\w+)\*\*\s*:\s*"?([^"\n]+)"?', text):
        key = match.group(1).strip().lower()
        value = match.group(2).strip()
        result[key] = value

    if not result.get("track") and not result.get("artist"):
        return None

    return result


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
                                text = content.text.strip()
                                import json

                                # 尝试 JSON 解析
                                try:
                                    data = json.loads(text)
                                    logger.debug(
                                        f"[Whisper] Spotify MCP 返回 JSON 数据: {data}"
                                    )
                                    if data.get("is_playing"):
                                        name = data.get("name", "Unknown")
                                        artist = data.get("artist", "Unknown")
                                        device = data.get("device", "Unknown")
                                        context_str = f"用户当前正在听 Spotify: {name} - {artist} (在设备 {device} 上)"
                                        logger.info(
                                            f"[Whisper] Spotify 解析成功 (JSON): {name} - {artist}"
                                        )
                                        return context_str
                                    continue
                                except (json.JSONDecodeError, ValueError):
                                    pass

                                # 回退: 解析 Markdown 纯文本格式
                                # 例如: **Track**: "片っぽ"\n**Artist**: eill
                                parsed = _parse_markdown_now_playing(text)
                                if parsed:
                                    name = parsed.get("track", "Unknown")
                                    artist = parsed.get("artist", "Unknown")
                                    device = parsed.get("device", "Unknown")
                                    context_str = f"用户当前正在听 Spotify: {name} - {artist} (在设备 {device} 上)"
                                    logger.info(
                                        f"[Whisper] Spotify 解析成功 (Markdown): {name} - {artist}"
                                    )
                                    return context_str

                                logger.debug(
                                    f"[Whisper] Spotify MCP 返回无法解析的文本: {text[:200]}"
                                )
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
