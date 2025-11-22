"""
llama.cpp local provider implementation.
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


class LlamaCppProvider(BaseProvider):
    """
    llama.cpp local provider.
    
    Features:
    - Raw binary inference (fastest local option)
    - No API key required
    - Simple HTTP API
    - Default port: 8080
    
    Installation:
    - Clone: git clone https://github.com/ggerganov/llama.cpp
    - Build: make
    - Run: ./server -m models/your-model.gguf -c 2048
    
    Usage:
        provider = LlamaCppProvider(config)
        response = await provider.chat(request)
    """
    
    DEFAULT_URL = "http://localhost:8080"
    
    # Hardcoded fallback models
    FALLBACK_MODELS = [
        ModelInfo(
            id='local-model',
            name='Local Model (llama.cpp)',
            provider='llamacpp',
            context_window=2048,
            max_output=2048,
            is_free=True,
            speed=None,
            supports_streaming=True,
            supports_vision=False,
            supports_function_calling=False,
            description='Currently loaded GGUF model'
        ),
    ]
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or self.DEFAULT_URL
    
    def get_default_model(self) -> str:
        """Get recommended default model"""
        return 'local-model'
    
    def _convert_messages_to_prompt(self, messages: list) -> str:
        """
        Convert chat messages to llama.cpp prompt format.
        llama.cpp uses simple prompt, not chat messages.
        """
        prompt_parts = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                prompt_parts.append(content)
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        # Add "Assistant:" at the end to prompt for response
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Send chat request to llama.cpp.
        
        Args:
            request: Chat request
            
        Returns:
            ChatResponse with content and metadata
            
        Raises:
            Exception: If API call fails
        """
        start_time = time.time()
        
        # Convert messages to prompt
        prompt = self._convert_messages_to_prompt(request.messages_as_dicts())
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/completion",
                    headers={
                        "Content-Type": "application/json",
                    },
                    json={
                        "prompt": prompt,
                        "temperature": request.temperature,
                        "n_predict": request.max_tokens,
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                latency_ms = (time.time() - start_time) * 1000
                
                content = data.get("content", "")
                
                return ChatResponse(
                    content=content.strip(),
                    model=request.model,
                    provider="llamacpp",
                    tokens_used=None,  # llama.cpp doesn't always return token count
                    latency_ms=latency_ms,
                    finish_reason=None,
                )
                
            except httpx.ConnectError:
                raise Exception("Failed to connect to llama.cpp. Make sure server is running on http://localhost:8080")
            except httpx.HTTPStatusError as e:
                raise Exception(f"llama.cpp API HTTP error: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                raise Exception(f"llama.cpp API error: {str(e)}")
    
    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """
        Stream chat responses from llama.cpp.
        
        Args:
            request: Chat request with stream=True
            
        Yields:
            String tokens as they arrive
            
        Raises:
            Exception: If API call fails
        """
        # Convert messages to prompt
        prompt = self._convert_messages_to_prompt(request.messages_as_dicts())
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/completion",
                    headers={
                        "Content-Type": "application/json",
                    },
                    json={
                        "prompt": prompt,
                        "temperature": request.temperature,
                        "n_predict": request.max_tokens,
                        "stream": True,
                    }
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        
                        data_str = line[6:]  # Remove "data: " prefix
                        
                        try:
                            import json
                            data = json.loads(data_str)
                            
                            if "content" in data:
                                text = data["content"]
                                if text:
                                    yield text
                        except json.JSONDecodeError:
                            continue
                            
            except httpx.ConnectError:
                raise Exception("Failed to connect to llama.cpp. Make sure server is running.")
            except Exception as e:
                raise Exception(f"llama.cpp streaming error: {str(e)}")
    
    async def list_models(self) -> List[ModelInfo]:
        """
        Return hardcoded model (llama.cpp serves one model at a time).
        
        Returns:
            List of ModelInfo objects
        """
        # llama.cpp only serves one model at a time
        return self.FALLBACK_MODELS
    
    async def health_check(self) -> ProviderHealth:
        """
        Check if llama.cpp server is running.
        
        Returns:
            ProviderHealth with status
        """
        try:
            start_time = time.time()
            
            # Quick health check
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/health")
                response.raise_for_status()
            
            latency_ms = (time.time() - start_time) * 1000
            
            return ProviderHealth(
                provider="llamacpp",
                healthy=True,
                latency_ms=latency_ms,
                models_available=1
            )
            
        except Exception as e:
            return ProviderHealth(
                provider="llamacpp",
                healthy=False,
                error="llama.cpp not running. Run: ./server -m models/your-model.gguf"
            )
    
    def to_autogen_config(self, model: str) -> dict:
        """
        Convert to AutoGen config format.
        
        Note: AutoGen doesn't natively support llama.cpp, so this returns
        a custom config that would need an adapter.
        
        Args:
            model: Model ID to use
            
        Returns:
            Config dict (custom format)
        """
        return {
            "model": model,
            "base_url": self.base_url,
            "api_type": "llamacpp",
            "provider": "llamacpp",
        }
