"""
Moonshot (Kimi) paid provider implementation.
"""

import os
from typing import List, AsyncIterator, Optional
import time
import httpx

from ..base import BaseProvider
from ..types import (
    ModelInfo,
    ProviderConfig,
    ChatRequest,
    ChatResponse,
    ProviderHealth,
    ProviderType,
)


class MoonshotProvider(BaseProvider):
    """
    Moonshot (Kimi) provider.
    
    Features:
    - High-quality Chinese/English responses
    - Long context windows (up to 1M!)
    - Competitive pricing
    - OpenAI-compatible API
    
    Pricing:
    - K2 models: Very competitive
    - Similar to OpenAI pricing structure
    
    API Key:
    - Get from: https://platform.moonshot.ai/
    - Set env: KIMI_API_KEY or MOONSHOT_API_KEY
    
    Usage:
        provider = MoonshotProvider(config)
        response = await provider.chat(request)
    """
    
    BASE_URL = "https://api.moonshot.ai/v1"
    
    # Updated fallback models (November 2025)
    FALLBACK_MODELS = [
        # Kimi K2 models (newest)
        ModelInfo(
            id='kimi-k2-thinking',
            name='Kimi K2 Thinking (1M)',
            provider='moonshot',
            context_window=1000000,
            max_output=32000,
            is_free=False,
            speed=None,
            supports_streaming=True,
            supports_vision=False,
            supports_function_calling=True,
            description='1M context window (recommended)'
        ),
        ModelInfo(
            id='kimi-k2-turbo-preview',
            name='Kimi K2 Turbo (256K)',
            provider='moonshot',
            context_window=256000,
            max_output=32000,
            is_free=False,
            speed=None,
            supports_streaming=True,
            supports_vision=False,
            supports_function_calling=True,
            description='Fast K2 model'
        ),
        ModelInfo(
            id='kimi-k2-0905-preview',
            name='Kimi K2 0905 (256K)',
            provider='moonshot',
            context_window=256000,
            max_output=32000,
            is_free=False,
            speed=None,
            supports_streaming=True,
            supports_vision=False,
            supports_function_calling=True,
            description='K2 September release'
        ),
        ModelInfo(
            id='kimi-k2-0711-preview',
            name='Kimi K2 0711 (128K)',
            provider='moonshot',
            context_window=128000,
            max_output=32000,
            is_free=False,
            speed=None,
            supports_streaming=True,
            supports_vision=False,
            supports_function_calling=True,
            description='K2 July release'
        ),
        # Moonshot V1 models (older)
        ModelInfo(
            id='moonshot-v1-128k',
            name='Moonshot V1 (128K)',
            provider='moonshot',
            context_window=128000,
            max_output=8192,
            is_free=False,
            speed=None,
            supports_streaming=True,
            supports_vision=False,
            supports_function_calling=True,
            description='V1 generation model'
        ),
        ModelInfo(
            id='moonshot-v1-32k',
            name='Moonshot V1 (32K)',
            provider='moonshot',
            context_window=32000,
            max_output=8192,
            is_free=False,
            speed=None,
            supports_streaming=True,
            supports_vision=False,
            supports_function_calling=True,
            description='V1 32K context'
        ),
        ModelInfo(
            id='moonshot-v1-8k',
            name='Moonshot V1 (8K)',
            provider='moonshot',
            context_window=8000,
            max_output=8192,
            is_free=False,
            speed=None,
            supports_streaming=True,
            supports_vision=False,
            supports_function_calling=True,
            description='V1 8K context'
        ),
    ]
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        # Try multiple env vars (both KIMI and MOONSHOT)
        self.api_key = config.api_key or os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY")
        self.base_url = config.base_url or self.BASE_URL
        
        if not self.api_key:
            raise ValueError("Moonshot API key required. Set KIMI_API_KEY or MOONSHOT_API_KEY env var.")
    
    def get_default_model(self) -> str:
        """Get recommended default model"""
        return 'kimi-k2-thinking'
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Send chat request to Moonshot.
        
        Args:
            request: Chat request
            
        Returns:
            ChatResponse with content and metadata
            
        Raises:
            Exception: If API call fails
        """
        start_time = time.time()
        
        # Inject project context if present (universal approach)
        messages = request.messages_as_dicts()
        messages = self._inject_project_context(messages, request.metadata)
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": request.model,
                        "messages": messages,  # ← Use modified messages
                        "temperature": request.temperature,
                        "max_tokens": request.max_tokens,
                        "stream": False,
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                latency_ms = (time.time() - start_time) * 1000
                
                content = data["choices"][0]["message"]["content"] or ""
                
                # Get token usage
                usage = data.get("usage", {})
                tokens_used = usage.get("total_tokens")
                
                return ChatResponse(
                    content=content,
                    model=data.get("model", request.model),
                    provider="moonshot",
                    tokens_used=tokens_used,
                    latency_ms=latency_ms,
                    finish_reason=data["choices"][0].get("finish_reason"),
                )
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    raise Exception("Invalid Moonshot API key. Check your KIMI_API_KEY.")
                elif e.response.status_code == 429:
                    raise Exception("Rate limit exceeded. Please wait and try again.")
                else:
                    raise Exception(f"Moonshot API HTTP error: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                raise Exception(f"Moonshot API error: {str(e)}")
    
    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """
        Stream chat responses from Moonshot.
        
        Args:
            request: Chat request with stream=True
            
        Yields:
            String tokens as they arrive
            
        Raises:
            Exception: If API call fails
        """
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": request.model,
                        "messages": request.messages_as_dicts(),
                        "temperature": request.temperature,
                        "max_tokens": request.max_tokens,
                        "stream": True,
                    }
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        
                        data_str = line[6:]  # Remove "data: " prefix
                        
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            import json
                            data = json.loads(data_str)
                            
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content")
                                
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
                            
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    raise Exception("Invalid Moonshot API key.")
                elif e.response.status_code == 429:
                    raise Exception("Rate limit exceeded.")
                else:
                    raise Exception(f"Moonshot streaming error: {e.response.status_code}")
            except Exception as e:
                raise Exception(f"Moonshot streaming error: {str(e)}")
    
    async def list_models(self) -> List[ModelInfo]:
        """
        Fetch current Moonshot models from API (OpenAI-compatible).
        Falls back to hardcoded list if API call fails.
        
        Returns:
            List of ModelInfo objects
        """
        try:
            # Try OpenAI-compatible models endpoint
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    models = []
                    
                    for model_data in data.get("data", []):
                        model_id = model_data.get("id")
                        
                        # Extract context window from ID
                        context_window = self._extract_context_window(model_id)
                        
                        models.append(ModelInfo(
                            id=model_id,
                            name=self._format_model_name(model_id),
                            provider='moonshot',
                            context_window=context_window,
                            max_output=32000 if 'k2' in model_id else 8192,
                            is_free=False,
                            speed=None,
                            supports_streaming=True,
                            supports_vision=False,
                            supports_function_calling=True,
                            description=self._format_model_name(model_id)
                        ))
                    
                    if models:
                        return models
        
        except Exception as e:
            # If API call fails, fall back to hardcoded list
            pass
        
        # Fallback to hardcoded list
        return self.FALLBACK_MODELS
    
    def _format_model_name(self, id: str) -> str:
        """Format model ID into readable name"""
        # Kimi K2 models (newest)
        if id == 'kimi-k2-thinking':
            return 'Kimi K2 Thinking (1M)'
        if id == 'kimi-k2-turbo-preview':
            return 'Kimi K2 Turbo (256K)'
        if id == 'kimi-k2-0905-preview':
            return 'Kimi K2 0905 (256K)'
        if id == 'kimi-k2-0711-preview':
            return 'Kimi K2 0711 (128K)'
        if 'kimi-k2' in id:
            context = self._extract_context_window(id) // 1000
            return f'Kimi K2 ({context}K)'
        
        # Moonshot V1 models (older)
        if id == 'moonshot-v1-128k':
            return 'Moonshot V1 (128K)'
        if id == 'moonshot-v1-32k':
            return 'Moonshot V1 (32K)'
        if id == 'moonshot-v1-8k':
            return 'Moonshot V1 (8K)'
        
        return id
    
    def _extract_context_window(self, id: str) -> int:
        """Extract context window size from model ID"""
        # Kimi K2 Thinking has 1M context
        if id == 'kimi-k2-thinking':
            return 1000000
        # Other K2 models have 256K by default
        if 'kimi-k2' in id:
            return 256000
        # V1 models by context size
        if '128k' in id:
            return 128000
        if '32k' in id:
            return 32000
        if '8k' in id:
            return 8000
        # Default to 128K
        return 128000
    
    async def health_check(self) -> ProviderHealth:
        """
        Check if Moonshot API is accessible.
        
        Returns:
            ProviderHealth with status
        """
        if not self.api_key:
            return ProviderHealth(
                provider="moonshot",
                healthy=False,
                error="No Moonshot API key configured"
            )
        
        try:
            start_time = time.time()
            
            # Quick health check with minimal token usage
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.get_default_model(),
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 10,
                    }
                )
                response.raise_for_status()
            
            latency_ms = (time.time() - start_time) * 1000
            
            return ProviderHealth(
                provider="moonshot",
                healthy=True,
                latency_ms=latency_ms,
                models_available=len(self.FALLBACK_MODELS)
            )
            
        except Exception as e:
            return ProviderHealth(
                provider="moonshot",
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
            "api_key": self.api_key,
            "base_url": self.base_url,
            "api_type": "openai",  # Moonshot uses OpenAI-compatible API
            "provider": "moonshot",
        }
