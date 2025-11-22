"""
LM Studio local provider implementation.
Runs locally, no API key required
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


class LMStudioProvider(BaseProvider):
    """
    LM Studio local provider.
    
    Features:
    - GUI-based local inference
    - OpenAI-compatible API
    - No API key required
    - Default port: 1234
    
    Installation:
    - Download from: https://lmstudio.ai
    - Start server in LM Studio
    - Load a model
    
    Usage:
        provider = LMStudioProvider(config)
        response = await provider.chat(request)
    """
    
    DEFAULT_URL = "http://localhost:1234/v1"
    
    # Hardcoded fallback models
    FALLBACK_MODELS = [
        ModelInfo(
            id='local-model',
            name='Local Model (LM Studio)',
            provider='lmstudio',
            context_window=8000,
            max_output=8192,
            is_free=True,
            speed=None,
            supports_streaming=True,
            supports_vision=False,
            supports_function_calling=False,
            description='Currently loaded model'
        ),
    ]
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or self.DEFAULT_URL
    
    def get_default_model(self) -> str:
        """Get recommended default model"""
        return 'local-model'
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Send chat request to LM Studio.
        
        Args:
            request: Chat request
            
        Returns:
            ChatResponse with content and metadata
            
        Raises:
            Exception: If API call fails
        """
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": request.model,
                        "messages": request.messages_as_dicts(),
                        "temperature": request.temperature,
                        "max_tokens": request.max_tokens,
                        "stream": False,
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                latency_ms = (time.time() - start_time) * 1000
                
                content = data["choices"][0]["message"]["content"] or ""
                
                return ChatResponse(
                    content=content,
                    model=data.get("model", request.model),
                    provider="lmstudio",
                    tokens_used=data.get("usage", {}).get("total_tokens"),
                    latency_ms=latency_ms,
                    finish_reason=data["choices"][0].get("finish_reason"),
                )
                
            except httpx.ConnectError:
                raise Exception("Failed to connect to LM Studio. Make sure it's running on http://localhost:1234")
            except httpx.HTTPStatusError as e:
                raise Exception(f"LM Studio API HTTP error: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                raise Exception(f"LM Studio API error: {str(e)}")
    
    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """
        Stream chat responses from LM Studio.
        
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
                            
            except httpx.ConnectError:
                raise Exception("Failed to connect to LM Studio. Make sure it's running.")
            except Exception as e:
                raise Exception(f"LM Studio streaming error: {str(e)}")
    
    async def list_models(self) -> List[ModelInfo]:
        """
        Fetch loaded models from LM Studio.
        Falls back to hardcoded list if LM Studio not running.
        
        Returns:
            List of ModelInfo objects
        """
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                response = await client.get(f"{self.base_url}/models")
                response.raise_for_status()
                data = response.json()
                
                models = []
                for model in data.get("data", []):
                    model_id = model.get("id", "")
                    if not model_id:
                        continue
                    
                    models.append(ModelInfo(
                        id=model_id,
                        name=model_id,
                        provider='lmstudio',
                        context_window=8000,  # Default
                        max_output=8192,
                        is_free=True,
                        speed=None,
                        supports_streaming=True,
                        supports_vision=False,
                        supports_function_calling=False,
                    ))
                
                return models if models else self.FALLBACK_MODELS
                
            except Exception as e:
                print(f"⚠️  LM Studio: Not running or no models loaded ({e}), using fallback")
                return self.FALLBACK_MODELS
    
    async def health_check(self) -> ProviderHealth:
        """
        Check if LM Studio is running.
        
        Returns:
            ProviderHealth with status
        """
        try:
            start_time = time.time()
            models = await self.list_models()
            latency_ms = (time.time() - start_time) * 1000
            
            return ProviderHealth(
                provider="lmstudio",
                healthy=True,
                latency_ms=latency_ms,
                models_available=len(models)
            )
            
        except Exception as e:
            return ProviderHealth(
                provider="lmstudio",
                healthy=False,
                error="LM Studio not running. Install from https://lmstudio.ai and start server"
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
            "base_url": self.base_url,
            "api_type": "openai",  # LM Studio uses OpenAI-compatible API
            "provider": "lmstudio",
        }
