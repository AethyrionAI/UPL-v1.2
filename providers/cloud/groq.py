"""
Groq cloud provider implementation.
Free tier: 14,400 requests/day
"""

import os
from typing import List, AsyncIterator, Optional
import time
from groq import AsyncGroq
from groq.types.chat import ChatCompletion

from ..base import BaseProvider
from ..types import (
    ModelInfo,
    ProviderConfig,
    ChatRequest,
    ChatResponse,
    ProviderHealth,
    ProviderType,
)


class GroqProvider(BaseProvider):
    """
    Groq cloud provider.
    
    Features:
    - Free tier: 14,400 requests/day
    - Very fast inference (500-1000 tokens/sec)
    - OpenAI-compatible API
    - Models: Llama 3.3 70B, Llama 3.1, Mixtral, etc.
    
    Usage:
        provider = GroqProvider(config)
        response = await provider.chat(request)
    """
    
    # Hardcoded fallback models (used if API fetch fails)
    FALLBACK_MODELS = [
        ModelInfo(
            id='llama-3.3-70b-versatile',
            name='Llama 3.3 70B Versatile',
            provider='groq',
            context_window=131072,
            max_output=8192,
            is_free=True,
            speed=560,
            supports_streaming=True,
            supports_vision=False,
            supports_function_calling=True,
            description='Best for code generation (560 T/s, 131K context)'
        ),
        ModelInfo(
            id='llama-3.1-70b-versatile',
            name='Llama 3.1 70B Versatile',
            provider='groq',
            context_window=131072,
            max_output=8192,
            is_free=True,
            speed=500,
            supports_streaming=True,
            supports_vision=False,
            supports_function_calling=True,
            description='Excellent quality (500 T/s, 131K context)'
        ),
        ModelInfo(
            id='llama-3.1-8b-instant',
            name='Llama 3.1 8B Instant',
            provider='groq',
            context_window=131072,
            max_output=8192,
            is_free=True,
            speed=1000,
            supports_streaming=True,
            supports_vision=False,
            supports_function_calling=True,
            description='Fast & efficient (1000 T/s, 131K context)'
        ),
        ModelInfo(
            id='mixtral-8x7b-32768',
            name='Mixtral 8x7B',
            provider='groq',
            context_window=32768,
            max_output=32768,
            is_free=True,
            speed=400,
            supports_streaming=True,
            supports_vision=False,
            supports_function_calling=True,
            description='MoE architecture (400 T/s, 32K context)'
        ),
    ]
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = AsyncGroq(api_key=config.api_key) if config.api_key else None
    
    def get_default_model(self) -> str:
        """Get recommended default model"""
        return 'llama-3.3-70b-versatile'
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Send chat request to Groq.
        
        Args:
            request: Chat request
            
        Returns:
            ChatResponse with content and metadata
            
        Raises:
            Exception: If API call fails
        """
        if not self.client:
            raise RuntimeError("Groq client not initialized. API key required.")
        
        start_time = time.time()
        
        # Inject project context if present (universal approach)
        messages = request.messages_as_dicts()
        messages = self._inject_project_context(messages, request.metadata)
        
        try:
            completion: ChatCompletion = await self.client.chat.completions.create(
                model=request.model,
                messages=messages,  # ← Use modified messages
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=False,
                stop=request.stop,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            return ChatResponse(
                content=completion.choices[0].message.content or "",
                model=completion.model,
                provider="groq",
                tokens_used=completion.usage.total_tokens if completion.usage else None,
                latency_ms=latency_ms,
                finish_reason=completion.choices[0].finish_reason,
            )
            
        except Exception as e:
            raise Exception(f"Groq API error: {str(e)}")
    
    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """
        Stream chat responses from Groq.
        
        Args:
            request: Chat request with stream=True
            
        Yields:
            String tokens as they arrive
            
        Raises:
            Exception: If API call fails
        """
        if not self.client:
            raise RuntimeError("Groq client not initialized. API key required.")
        
        try:
            stream = await self.client.chat.completions.create(
                model=request.model,
                messages=request.messages_as_dicts(),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
                stop=request.stop,
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            raise Exception(f"Groq streaming error: {str(e)}")
    
    async def list_models(self) -> List[ModelInfo]:
        """
        Fetch available models from Groq API.
        Falls back to hardcoded list if API call fails.
        
        Returns:
            List of ModelInfo objects
        """
        if not self.client:
            print("⚠️  Groq: No API key, using fallback models")
            return self.FALLBACK_MODELS
        
        try:
            # Fetch models from Groq API
            response = await self.client.models.list()
            
            models = []
            for model in response.data:
                # Only include active models
                if not model.active:
                    continue
                
                models.append(ModelInfo(
                    id=model.id,
                    name=model.id.replace('-', ' ').title(),
                    provider='groq',
                    context_window=model.context_window,
                    max_output=8192,  # Groq doesn't expose this in API
                    is_free=True,
                    speed=None,  # Speed varies, not exposed in API
                    supports_streaming=True,
                    supports_vision=False,
                    supports_function_calling=True,
                ))
            
            # Sort by context window (larger first)
            models.sort(key=lambda m: m.context_window, reverse=True)
            
            return models if models else self.FALLBACK_MODELS
            
        except Exception as e:
            print(f"⚠️  Groq: Failed to fetch models ({e}), using fallback")
            return self.FALLBACK_MODELS
    
    async def health_check(self) -> ProviderHealth:
        """
        Check if Groq is accessible.
        
        Returns:
            ProviderHealth with status
        """
        if not self.client:
            return ProviderHealth(
                provider="groq",
                healthy=False,
                error="No API key configured"
            )
        
        try:
            start_time = time.time()
            models = await self.list_models()
            latency_ms = (time.time() - start_time) * 1000
            
            return ProviderHealth(
                provider="groq",
                healthy=True,
                latency_ms=latency_ms,
                models_available=len(models)
            )
            
        except Exception as e:
            return ProviderHealth(
                provider="groq",
                healthy=False,
                error=str(e)
            )
    
    def to_autogen_config(self, model: str) -> dict:
        """
        Convert to AutoGen config format.
        
        Args:
            model: Model ID to use
            
        Returns:
            Config dict for AutoGen
        """
        return {
            "model": model,
            "api_key": self.config.api_key,
            "base_url": "https://api.groq.com/openai/v1",
            "api_type": "openai",  # Groq uses OpenAI-compatible API
        }
