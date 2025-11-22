"""
Universal Provider Abstraction Layer for Aethyrion Coder projects.

Works with AC1 (kosong), AC2 (Google ADK), AC3 (AutoGen), AC4 (PydanticAI).

Usage:
    from providers import registry, ProviderConfig, ProviderType, ChatRequest, ChatMessage
    
    configs = {
        'groq': ProviderConfig(
            name='groq',
            type=ProviderType.CLOUD_FREE,
            api_key='your-key'
        )
    }
    
    await registry.initialize(configs)
    provider = await registry.get_best_provider()
    response = await provider.chat(request)
"""

from .types import (
    ProviderType,
    ModelInfo,
    ProviderConfig,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ProviderHealth,
)

from .base import BaseProvider
from .registry import registry, ProviderRegistry

__all__ = [
    # Types
    'ProviderType',
    'ModelInfo',
    'ProviderConfig',
    'ChatMessage',
    'ChatRequest',
    'ChatResponse',
    'ProviderHealth',
    # Core classes
    'BaseProvider',
    'ProviderRegistry',
    # Global instance
    'registry',
]
