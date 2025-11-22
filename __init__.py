"""
Universal Provider Layer (UPL)

Framework-agnostic abstraction for 12 LLM providers.
Supports dynamic model fetching, prompt caching, and extended reasoning.
"""

__version__ = "1.2.0"
__author__ = "Owen - Aethyrion"

# Import main components for easy access
from .providers import (
    registry,
    ProviderConfig,
    ProviderType,
    ChatRequest,
    ChatMessage,
    ChatResponse,
    ModelInfo,
    ProviderHealth,
)

__all__ = [
    "registry",
    "ProviderConfig",
    "ProviderType",
    "ChatRequest",
    "ChatMessage",
    "ChatResponse",
    "ModelInfo",
    "ProviderHealth",
]
