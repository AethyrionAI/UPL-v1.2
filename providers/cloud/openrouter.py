"""
OpenRouter cloud provider implementation.
Free models available (aggregate of multiple providers)
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


class OpenRouterProvider(BaseProvider):
    """
    OpenRouter cloud provider.
    
    Features:
    - Aggregates multiple LLM providers
    - Many free models available
    - OpenAI-compatible API
    - Pay-as-you-go or free tier models
    - Models: Llama, Mistral, Gemini, Claude, etc.
    
    Usage:
        provider = OpenRouterProvider(config)
        response = await provider.chat(request)
    """
    
    BASE_URL = "https://openrouter.ai/api/v1"
    
    # Hardcoded fallback models (free models as of Nov 2024)
    FALLBACK_MODELS = [
        ModelInfo(
            id='openai/gpt-oss-20b:free',
            name='OpenAI GPT OSS 20B (Free)',
            provider='openrouter',
            context_window=131072,
            max_output=8192,
            is_free=True,
            speed=None,
            supports_streaming=True,
            supports_vision=False,
            supports_function_calling=True,
            description='Free GPT OSS 20B (recommended)'
        ),
        ModelInfo(
            id='z-ai/glm-4.5-air:free',
            name='Z.AI GLM 4.5 Air (Free)',
            provider='openrouter',
            context_window=131072,
            max_output=8192,
            is_free=True,
            speed=None,
            supports_streaming=True,
            supports_vision=False,
            supports_function_calling=True,
            description='Free Z.AI GLM'
        ),
        ModelInfo(
            id='google/gemma-3n-e2b-it:free',
            name='Google Gemma 3n 2B (Free)',
            provider='openrouter',
            context_window=8192,
            max_output=4096,
            is_free=True,
            speed=None,
            supports_streaming=True,
            supports_vision=False,
            supports_function_calling=True,
            description='Free Google Gemma'
        ),
    ]
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.api_key = config.api_key
        self.base_url = config.base_url or self.BASE_URL
    
    def get_default_model(self) -> str:
        """Get recommended default model"""
        return 'openai/gpt-oss-20b:free'
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Send chat request to OpenRouter.
        
        Args:
            request: Chat request
            
        Returns:
            ChatResponse with content and metadata
            
        Raises:
            Exception: If API call fails
        """
        if not self.api_key:
            raise RuntimeError("OpenRouter API key required.")
        
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://aethyrion.com",  # Optional but recommended
                        "X-Title": "AethyrionCoder AC3",  # Optional but recommended
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
                    provider="openrouter",
                    tokens_used=data.get("usage", {}).get("total_tokens"),
                    latency_ms=latency_ms,
                    finish_reason=data["choices"][0].get("finish_reason"),
                )
                
            except httpx.HTTPStatusError as e:
                raise Exception(f"OpenRouter API HTTP error: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                raise Exception(f"OpenRouter API error: {str(e)}")
    
    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """
        Stream chat responses from OpenRouter.
        
        Args:
            request: Chat request with stream=True
            
        Yields:
            String tokens as they arrive
            
        Raises:
            Exception: If API call fails
        """
        if not self.api_key:
            raise RuntimeError("OpenRouter API key required.")
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://aethyrion.com",
                        "X-Title": "AethyrionCoder AC3",
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
                raise Exception(f"OpenRouter streaming HTTP error: {e.response.status_code}")
            except Exception as e:
                raise Exception(f"OpenRouter streaming error: {str(e)}")
    
    async def list_models(self) -> List[ModelInfo]:
        """
        Fetch available models from OpenRouter API.
        Filters for free models only.
        Falls back to hardcoded list if API call fails.
        
        Returns:
            List of ModelInfo objects (free models only)
        """
        if not self.api_key:
            print("⚠️  OpenRouter: No API key, using fallback models")
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
                    model_id = model.get("id", "")
                    
                    # Filter for free models only
                    pricing = model.get("pricing", {})
                    prompt_price = float(pricing.get("prompt", "0"))
                    completion_price = float(pricing.get("completion", "0"))
                    
                    is_free = (prompt_price == 0 and completion_price == 0) or ":free" in model_id
                    
                    if not is_free:
                        continue
                    
                    # Parse context window
                    context_window = model.get("context_length", 32768)
                    
                    models.append(ModelInfo(
                        id=model_id,
                        name=model.get("name", model_id),
                        provider='openrouter',
                        context_window=context_window,
                        max_output=8192,
                        is_free=True,
                        speed=None,
                        supports_streaming=True,
                        supports_vision='vision' in model_id.lower(),
                        supports_function_calling=True,
                    ))
                
                # Sort by context window (larger first)
                models.sort(key=lambda m: m.context_window, reverse=True)
                
                return models if models else self.FALLBACK_MODELS
                
            except Exception as e:
                print(f"⚠️  OpenRouter: Failed to fetch models ({e}), using fallback")
                return self.FALLBACK_MODELS
    
    async def health_check(self) -> ProviderHealth:
        """
        Check if OpenRouter is accessible.
        
        Returns:
            ProviderHealth with status
        """
        if not self.api_key:
            return ProviderHealth(
                provider="openrouter",
                healthy=False,
                error="No API key configured"
            )
        
        try:
            start_time = time.time()
            models = await self.list_models()
            latency_ms = (time.time() - start_time) * 1000
            
            return ProviderHealth(
                provider="openrouter",
                healthy=True,
                latency_ms=latency_ms,
                models_available=len(models)
            )
            
        except Exception as e:
            return ProviderHealth(
                provider="openrouter",
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
            "api_type": "openai",  # OpenRouter uses OpenAI-compatible API
        }
