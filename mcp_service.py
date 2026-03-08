"""MCP Service abstract base class for Whisper plugin."""

from abc import ABC, abstractmethod
from typing import Any


class MCPService(ABC):
    """Abstract base class for MCP service implementations."""

    @abstractmethod
    async def start(self) -> None:
        """Start the MCP service."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the MCP service."""
        ...

    @abstractmethod
    async def get_context(self) -> str:
        """Get the current context from the MCP service.

        Returns:
            str: The current context as a string.
        """
        ...

    @abstractmethod
    def format_suggestion(self, action: dict[str, Any]) -> str:
        """Format an action/suggestion into a human-readable string.

        Args:
            action: A dictionary containing action details.

        Returns:
            str: The formatted suggestion string.
        """
        ...
