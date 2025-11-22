"""
Cerebras cloud provider implementation.
Free tier: 14,400 requests/day
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


class CerebrasProvider(BaseProvider):
    """
    Cerebras cloud provider.
    
    Features:
    - Free tier: 14,400 requests/day
    - Ultra-fast inference (faster than Groq)
    - 128K context window
    - OpenAI-compatible API
    - Models: Llama 3.1, Llama 3.3
    
    Usage:
        provider = CerebrasProvider(config)
        response = await provider.chat(request)
    """
    
    BASE_URL = "https://api.cerebras.ai/v1"
    
    # Hardcoded fallback models (used if API fetch fails)
    FALLBACK_MODELS = [
        ModelInfo(
            id='llama3.1-8b',
            name='Llama 3.1 8B',
            provider='cerebras',
            context_window=131072,  # 128K
            max_output=8192,
            is_free=True,
            speed=None,  # Ultra fast but not specified
            supports_streaming=True,
            supports_vision=False,
            supports_function_calling=True,
            description='Fast & efficient (128K context)'
        ),
        ModelInfo(
            id='llama3.1-70b',
            name='Llama 3.1 70B',
            provider='cerebras',
            context_window=131072,  # 128K
            max_output=8192,
            is_free=True,
            speed=None,
            supports_streaming=True,
            supports_vision=False,
            supports_function_calling=True,
            description='Best quality (128K context)'
        ),
        ModelInfo(
            id='llama3.3-70b',
            name='Llama 3.3 70B',
            provider='cerebras',
            context_window=131072,  # 128K
            max_output=8192,
            is_free=True,
            speed=None,
            supports_streaming=True,
            supports_vision=False,
            supports_function_calling=True,
            description='Latest model (128K context)'
        ),
    ]
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.api_key = config.api_key
        self.base_url = config.base_url or self.BASE_URL
    
    def get_default_model(self) -> str:
        """Get recommended default model"""
        return 'llama3.3-70b'
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Send chat request to Cerebras.
        
        Args:
            request: Chat request
            
        Returns:
            ChatResponse with content and metadata
            
        Raises:
            Exception: If API call fails
        """
        if not self.api_key:
            raise RuntimeError("Cerebras API key required.")
        
        start_time = time.time()
        
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
                        "messages": request.messages_as_dicts(),
                        "temperature": request.temperature,
                        "max_tokens": request.max_tokens,
                        "stream": False,
                        "stop": request.stop,
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                latency_ms = (time.time() - start_time) * 1000
                
                return ChatResponse(
                    content=data["choices"][0]["message"]["content"] or "",
                    model=data["model"],
                    provider="cerebras",
                    tokens_used=data.get("usage", {}).get("total_tokens"),
                    latency_ms=latency_ms,
                    finish_reason=data["choices"][0].get("finish_reason"),
                )
                
            except httpx.HTTPStatusError as e:
                raise Exception(f"Cerebras API HTTP error: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                raise Exception(f"Cerebras API error: {str(e)}")
    
    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """
        Stream chat responses from Cerebras.
        
        Args:
            request: Chat request with stream=True
            
        Yields:
            String tokens as they arrive
            
        Raises:
            Exception: If API call fails
        """
        if not self.api_key:
            raise RuntimeError("Cerebras API key required.")
        
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
                        "stop": request.stop,
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
                raise Exception(f"Cerebras streaming HTTP error: {e.response.status_code}")
            except Exception as e:
                raise Exception(f"Cerebras streaming error: {str(e)}")
    
    async def list_models(self) -> List[ModelInfo]:
        """
        Fetch available models from Cerebras API.
        Falls back to hardcoded list if API call fails.
        
        Returns:
            List of ModelInfo objects
        """
        if not self.api_key:
            print("⚠️  Cerebras: No API key, using fallback models")
            return self.FALLBACK_MODELS
        
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                models = []
                for model in data.get("data", []):
                    # Parse model info
                    model_id = model.get("id", "")
                    
                    # Skip non-chat models
                    if not model_id or "embed" in model_id.lower():
                        continue
                    
                    models.append(ModelInfo(
                        id=model_id,
                        name=model_id.replace('-', ' ').replace('_', ' ').title(),
                        provider='cerebras',
                        context_window=131072,  # Cerebras default 128K
                        max_output=8192,
                        is_free=True,
                        speed=None,
                        supports_streaming=True,
                        supports_vision=False,
                        supports_function_calling=True,
                    ))
                
                # Sort by name
                models.sort(key=lambda m: m.id)
                
                return models if models else self.FALLBACK_MODELS
                
            except Exception as e:
                print(f"⚠️  Cerebras: Failed to fetch models ({e}), using fallback")
                return self.FALLBACK_MODELS
    
    async def health_check(self) -> ProviderHealth:
        """
        Check if Cerebras is accessible.
        
        Returns:
            ProviderHealth with status
        """
        if not self.api_key:
            return ProviderHealth(
                provider="cerebras",
                healthy=False,
                error="No API key configured"
            )
        
        try:
            start_time = time.time()
            models = await self.list_models()
            latency_ms = (time.time() - start_time) * 1000
            
            return ProviderHealth(
                provider="cerebras",
                healthy=True,
                latency_ms=latency_ms,
                models_available=len(models)
            )
            
        except Exception as e:
            return ProviderHealth(
                provider="cerebras",
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
            "api_type": "openai",  # Cerebras uses OpenAI-compatible API
        }
