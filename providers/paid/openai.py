"""
OpenAI paid provider implementation.
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


class OpenAIProvider(BaseProvider):
    """
    OpenAI provider.
    
    Features:
    - GPT-4, GPT-3.5, and other models
    - Function calling support
    - Vision support (GPT-4V)
    - Industry standard
    
    Pricing:
    - GPT-4o: $2.50/$10 per million tokens (input/output)
    - GPT-4o-mini: $0.15/$0.60 per million tokens
    - GPT-3.5: $0.50/$1.50 per million tokens
    
    API Key:
    - Get from: https://platform.openai.com/api-keys
    - Set env: OPENAI_API_KEY
    
    Usage:
        provider = OpenAIProvider(config)
        response = await provider.chat(request)
    """
    
    BASE_URL = "https://api.openai.com/v1"
    
    # Hardcoded models
    FALLBACK_MODELS = [
        # Thinking models first
        ModelInfo(
            id='o1',
            name='O1 🧠',
            provider='openai',
            context_window=200000,
            max_output=100000,
            is_free=False,
            speed=None,
            supports_streaming=False,  # o1 doesn't support streaming
            supports_vision=False,
            supports_function_calling=True,
            supports_thinking=True,
            description='Reasoning model • Thinking Mode (automatic)'
        ),
        ModelInfo(
            id='o1-mini',
            name='O1 Mini 🧠',
            provider='openai',
            context_window=128000,
            max_output=65000,
            is_free=False,
            speed=None,
            supports_streaming=False,
            supports_vision=False,
            supports_function_calling=True,
            supports_thinking=True,
            description='Faster reasoning • Thinking Mode (automatic)'
        ),
        ModelInfo(
            id='o1-preview',
            name='O1 Preview 🧠',
            provider='openai',
            context_window=128000,
            max_output=32000,
            is_free=False,
            speed=None,
            supports_streaming=False,
            supports_vision=False,
            supports_function_calling=True,
            supports_thinking=True,
            description='Preview reasoning model • Thinking Mode'
        ),
        # Standard GPT models
        ModelInfo(
            id='gpt-4o',
            name='GPT-4o ⚡',
            provider='openai',
            context_window=128000,
            max_output=16384,
            is_free=False,
            speed=None,
            supports_streaming=True,
            supports_vision=True,
            supports_function_calling=True,
            supports_thinking=False,
            description='Latest GPT-4 (recommended)'
        ),
        ModelInfo(
            id='gpt-4o-mini',
            name='GPT-4o Mini ⚡',
            provider='openai',
            context_window=128000,
            max_output=16384,
            is_free=False,
            speed=None,
            supports_streaming=True,
            supports_vision=True,
            supports_function_calling=True,
            supports_thinking=False,
            description='Fast and cheap GPT-4'
        ),
        ModelInfo(
            id='gpt-4-turbo',
            name='GPT-4 Turbo 🚀',
            provider='openai',
            context_window=128000,
            max_output=4096,
            is_free=False,
            speed=None,
            supports_streaming=True,
            supports_vision=True,
            supports_function_calling=True,
            supports_thinking=False,
            description='Previous generation'
        ),
        ModelInfo(
            id='gpt-3.5-turbo',
            name='GPT-3.5 Turbo 🚀',
            provider='openai',
            context_window=16385,
            max_output=4096,
            is_free=False,
            speed=None,
            supports_streaming=True,
            supports_vision=False,
            supports_function_calling=True,
            supports_thinking=False,
            description='Cheap and fast'
        ),
    ]
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = config.base_url or self.BASE_URL
        
        if not self.api_key:
            raise ValueError("OpenAI API key required. Get one from https://platform.openai.com/api-keys")
    
    def get_default_model(self) -> str:
        """Get recommended default model"""
        return 'gpt-4o-mini'
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Send chat request to OpenAI.
        
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
                    provider="openai",
                    tokens_used=tokens_used,
                    latency_ms=latency_ms,
                    finish_reason=data["choices"][0].get("finish_reason"),
                )
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    raise Exception("Invalid OpenAI API key. Check your OPENAI_API_KEY.")
                elif e.response.status_code == 429:
                    raise Exception("Rate limit exceeded. Please wait and try again.")
                elif e.response.status_code == 402:
                    raise Exception("Insufficient credits. Add credits at https://platform.openai.com/account/billing")
                else:
                    raise Exception(f"OpenAI API HTTP error: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                raise Exception(f"OpenAI API error: {str(e)}")
    
    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """
        Stream chat responses from OpenAI.
        
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
                    raise Exception("Invalid OpenAI API key.")
                elif e.response.status_code == 429:
                    raise Exception("Rate limit exceeded.")
                else:
                    raise Exception(f"OpenAI streaming error: {e.response.status_code}")
            except Exception as e:
                raise Exception(f"OpenAI streaming error: {str(e)}")
    
    async def list_models(self) -> List[ModelInfo]:
        """
        Fetch available models from OpenAI.
        Falls back to hardcoded list if API call fails.
        
        Returns:
            List of ModelInfo objects
        """
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                models = []
                for model_data in data.get("data", []):
                    model_id = model_data.get("id", "")

                    # Exclude non-chat models (blacklist approach)
                    if any(x in model_id for x in [
                        "embedding",     # Text embeddings
                        "moderation",    # Moderation API
                        "whisper",       # Audio transcription
                        "tts",           # Text-to-speech
                        "dall-e",        # Image generation
                        "babbage",       # Old completion models
                        "davinci",       # Old completion models
                        "ada",           # Old completion models
                        "curie",         # Old completion models
                        "transcribe",    # Transcription
                        "realtime",      # Realtime audio
                        "audio",         # Audio models
                        "image",         # Image generation
                        "search",        # Search API models
                    ]):
                        continue

                    # Only include models owned by OpenAI
                    owned_by = model_data.get("owned_by", "")
                    if owned_by and owned_by != "system" and not owned_by.startswith("openai"):
                        continue

                    # Detect context window
                    if "o1" in model_id or "o3" in model_id or "o5" in model_id:
                        context_window = 200000
                    elif "gpt-5" in model_id:
                        context_window = 200000
                    elif "gpt-4" in model_id:
                        context_window = 128000
                    elif "gpt-3.5" in model_id:
                        context_window = 16385
                    else:
                        context_window = 8192

                    # Detect max output
                    if "o1" in model_id:
                        max_output = 100000 if model_id == "o1" else 65000
                    elif "gpt-5" in model_id:
                        max_output = 32000
                    else:
                        max_output = 16384

                    # Detect thinking support
                    supports_thinking = any(x in model_id for x in [
                        "o1", "o3", "o4", "o5", "reasoning"
                    ])

                    # Add emoji indicators
                    if supports_thinking:
                        emoji = " 🧠"
                    elif "gpt-5" in model_id:
                        emoji = " ✨"
                    elif "gpt-4o" in model_id:
                        emoji = " ⚡"
                    elif "turbo" in model_id:
                        emoji = " 🚀"
                    else:
                        emoji = ""

                    models.append(ModelInfo(
                        id=model_id,
                        name=f"{model_id.replace('gpt-', 'GPT-').upper()}{emoji}",
                        provider='openai',
                        context_window=context_window,
                        max_output=max_output,
                        is_free=False,
                        supports_streaming=not supports_thinking,  # o1 doesn't support streaming
                        supports_vision="vision" in model_id.lower() or "gpt-4" in model_id,
                        supports_function_calling=True,
                        supports_thinking=supports_thinking,
                        description=model_data.get("id", "OpenAI model")
                    ))

                if models:
                    # Sort: thinking models first, then by ID
                    models.sort(key=lambda m: (not m.supports_thinking, m.id))
                
                return models if models else self.FALLBACK_MODELS
                
        except Exception as e:
            print(f"⚠️  OpenAI: Failed to fetch models ({e}), using fallback")
            return self.FALLBACK_MODELS
    
    async def health_check(self) -> ProviderHealth:
        """
        Check if OpenAI API is accessible.
        
        Returns:
            ProviderHealth with status
        """
        if not self.api_key:
            return ProviderHealth(
                provider="openai",
                healthy=False,
                error="No OpenAI API key configured"
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
                        "max_tokens": 5,
                    }
                )
                response.raise_for_status()
            
            latency_ms = (time.time() - start_time) * 1000
            
            return ProviderHealth(
                provider="openai",
                healthy=True,
                latency_ms=latency_ms,
                models_available=len(self.FALLBACK_MODELS)
            )
            
        except Exception as e:
            return ProviderHealth(
                provider="openai",
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
            "api_type": "openai",
            "provider": "openai",
        }
