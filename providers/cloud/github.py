"""
GitHub Models cloud provider implementation.
Free tier available for GitHub users
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


class GitHubModelsProvider(BaseProvider):
    """
    GitHub Models provider.
    
    Features:
    - Free tier for GitHub users
    - Multiple popular models
    - OpenAI-compatible API
    - Models: GPT-4, Llama, Mistral, Phi, etc.
    
    Usage:
        provider = GitHubModelsProvider(config)
        response = await provider.chat(request)
    """
    
    BASE_URL = "https://models.github.ai/inference"
    
    # Hardcoded fallback models (from Helix's current list)
    FALLBACK_MODELS = [
        ModelInfo(
            id='openai/gpt-4o-mini',
            name='OpenAI GPT-4o mini 💚',
            provider='github',
            context_window=131072,
            max_output=16384,
            is_free=True,
            speed=None,
            supports_streaming=True,
            supports_vision=True,
            supports_function_calling=True,
            description='Fast GPT-4 variant (150 req/day)'
        ),
        ModelInfo(
            id='microsoft/phi-4',
            name='Phi-4 💚',
            provider='github',
            context_window=16384,
            max_output=8192,
            is_free=True,
            speed=None,
            supports_streaming=True,
            supports_vision=False,
            supports_function_calling=True,
            description='Microsoft Phi-4 (150 req/day)'
        ),
        ModelInfo(
            id='meta/llama-3.3-70b-instruct',
            name='Llama-3.3-70B-Instruct 🔥',
            provider='github',
            context_window=128000,
            max_output=8192,
            is_free=True,
            speed=None,
            supports_streaming=True,
            supports_vision=False,
            supports_function_calling=True,
            description='Meta Llama 3.3 70B (50 req/day)'
        ),
    ]
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.api_key = config.api_key  # GitHub PAT token
        self.base_url = config.base_url or self.BASE_URL
    
    def get_default_model(self) -> str:
        """Get recommended default model"""
        return 'openai/gpt-4o-mini'
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Send chat request to GitHub Models.
        
        Args:
            request: Chat request
            
        Returns:
            ChatResponse with content and metadata
            
        Raises:
            Exception: If API call fails
        """
        if not self.api_key:
            raise RuntimeError("GitHub token required.")
        
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
                    provider="github",
                    tokens_used=data.get("usage", {}).get("total_tokens"),
                    latency_ms=latency_ms,
                    finish_reason=data["choices"][0].get("finish_reason"),
                )
                
            except httpx.HTTPStatusError as e:
                raise Exception(f"GitHub Models API HTTP error: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                raise Exception(f"GitHub Models API error: {str(e)}")
    
    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """
        Stream chat responses from GitHub Models.
        
        Args:
            request: Chat request with stream=True
            
        Yields:
            String tokens as they arrive
            
        Raises:
            Exception: If API call fails
        """
        if not self.api_key:
            raise RuntimeError("GitHub token required.")
        
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
                raise Exception(f"GitHub Models streaming HTTP error: {e.response.status_code}")
            except Exception as e:
                raise Exception(f"GitHub Models streaming error: {str(e)}")
    
    async def list_models(self) -> List[ModelInfo]:
        """
        Fetch available models from GitHub Models API.
        Falls back to hardcoded list if API call fails.
        
        Returns:
            List of ModelInfo objects
        """
        if not self.api_key:
            print("⚠️  GitHub Models: No token, using fallback models")
            return self.FALLBACK_MODELS
        
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                response = await client.get(
                    "https://models.github.ai/catalog/models",  # CATALOG endpoint!
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Accept": "application/vnd.github+json",
                        "X-GitHub-Api-Version": "2022-11-28",
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                models = []
                
                # GitHub returns a list directly
                model_list = data if isinstance(data, list) else []
                
                # Filter for text generation models
                for model in model_list:
                    output_modalities = model.get("supported_output_modalities", [])
                    tags = model.get("tags", [])
                    
                    # Must output text
                    if "text" not in output_modalities:
                        continue
                    
                    # Exclude pure embedding models
                    if len(output_modalities) == 1 and output_modalities[0] == "embeddings":
                        continue
                    if "rag" in tags and output_modalities[0] == "embeddings":
                        continue
                    
                    model_id = model.get("id", "")
                    if not model_id:
                        continue
                    
                    # Get tier info
                    tier = model.get("rate_limit_tier", "unknown")
                    tier_emoji = ""
                    if tier == "low":
                        tier_emoji = "💚"  # 150 req/day
                    elif tier == "high":
                        tier_emoji = "🔥"  # 50 req/day
                    elif tier == "custom":
                        tier_emoji = "⚡"  # 8-30 req/day
                    
                    # Parse capabilities
                    limits = model.get("limits", {})
                    context_window = limits.get("max_input_tokens", 128000)
                    
                    models.append(ModelInfo(
                        id=model_id,
                        name=f"{model.get('name', model_id)} {tier_emoji}".strip(),
                        provider='github',
                        context_window=context_window,
                        max_output=8192,
                        is_free=True,
                        speed=None,
                        supports_streaming=True,
                        supports_vision="vision" in model.get("capabilities", []),
                        supports_function_calling="function_calling" in model.get("capabilities", []),
                    ))
                
                # Sort by tier priority (Low > High > Custom)
                tier_priority = {"💚": 1, "🔥": 2, "⚡": 3}
                def get_tier_priority(m: ModelInfo) -> int:
                    for emoji, priority in tier_priority.items():
                        if emoji in m.name:
                            return priority
                    return 99
                
                models.sort(key=lambda m: (get_tier_priority(m), m.name))
                
                return models if models else self.FALLBACK_MODELS
                
            except Exception as e:
                print(f"⚠️  GitHub Models: Failed to fetch models ({e}), using fallback")
                return self.FALLBACK_MODELS
    
    async def health_check(self) -> ProviderHealth:
        """
        Check if GitHub Models is accessible.
        
        Returns:
            ProviderHealth with status
        """
        if not self.api_key:
            return ProviderHealth(
                provider="github",
                healthy=False,
                error="No GitHub token configured"
            )
        
        try:
            start_time = time.time()
            models = await self.list_models()
            latency_ms = (time.time() - start_time) * 1000
            
            return ProviderHealth(
                provider="github",
                healthy=True,
                latency_ms=latency_ms,
                models_available=len(models)
            )
            
        except Exception as e:
            return ProviderHealth(
                provider="github",
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
            "api_type": "openai",  # GitHub Models uses OpenAI-compatible API
        }
