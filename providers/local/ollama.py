"""
Ollama local provider implementation.
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


class OllamaProvider(BaseProvider):
    """
    Ollama local provider.
    
    Features:
    - Runs locally on your machine
    - No API key required
    - All open-source models
    - Simple REST API
    - Default port: 11434
    
    Installation:
    - Download from: https://ollama.com
    - Run: ollama serve
    - Pull models: ollama pull qwen2.5-coder:latest
    
    Usage:
        provider = OllamaProvider(config)
        response = await provider.chat(request)
    """
    
    DEFAULT_URL = "http://localhost:11434"
    
    # Hardcoded fallback models (popular coding models)
    FALLBACK_MODELS = [
        ModelInfo(
            id='qwen2.5-coder:latest',
            name='Qwen 2.5 Coder',
            provider='ollama',
            context_window=32768,
            max_output=8192,
            is_free=True,
            speed=None,
            supports_streaming=True,
            supports_vision=False,
            supports_function_calling=False,
            description='Best for coding (recommended)'
        ),
        ModelInfo(
            id='llama3.2:latest',
            name='Llama 3.2',
            provider='ollama',
            context_window=131072,
            max_output=8192,
            is_free=True,
            speed=None,
            supports_streaming=True,
            supports_vision=False,
            supports_function_calling=False,
            description='Latest Llama model'
        ),
        ModelInfo(
            id='codellama:latest',
            name='Code Llama',
            provider='ollama',
            context_window=16000,
            max_output=8192,
            is_free=True,
            speed=None,
            supports_streaming=True,
            supports_vision=False,
            supports_function_calling=False,
            description='Meta code specialist'
        ),
    ]
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or self.DEFAULT_URL
    
    def get_default_model(self) -> str:
        """Get recommended default model"""
        return 'qwen2.5-coder:latest'
    
    def _convert_messages_to_prompt(self, messages: list) -> str:
        """
        Convert chat messages to Ollama prompt format.
        Ollama uses simple prompt, not chat messages.
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
        Send chat request to Ollama.
        
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
                    f"{self.base_url}/api/generate",
                    headers={
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": request.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": request.temperature,
                            "num_predict": request.max_tokens,
                        }
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                latency_ms = (time.time() - start_time) * 1000
                
                content = data.get("response", "")
                
                return ChatResponse(
                    content=content.strip(),
                    model=request.model,
                    provider="ollama",
                    tokens_used=None,  # Ollama doesn't return token count in this format
                    latency_ms=latency_ms,
                    finish_reason=None,
                )
                
            except httpx.HTTPStatusError as e:
                raise Exception(f"Ollama API HTTP error: {e.response.status_code} - {e.response.text}")
            except httpx.ConnectError:
                raise Exception("Failed to connect to Ollama. Make sure it's running on http://localhost:11434")
            except Exception as e:
                raise Exception(f"Ollama API error: {str(e)}")
    
    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """
        Stream chat responses from Ollama.
        
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
                    f"{self.base_url}/api/generate",
                    headers={
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": request.model,
                        "prompt": prompt,
                        "stream": True,
                        "options": {
                            "temperature": request.temperature,
                            "num_predict": request.max_tokens,
                        }
                    }
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        
                        try:
                            import json
                            data = json.loads(line)
                            
                            if "response" in data:
                                text = data["response"]
                                if text:
                                    yield text
                        except json.JSONDecodeError:
                            continue
                            
            except httpx.ConnectError:
                raise Exception("Failed to connect to Ollama. Make sure it's running.")
            except Exception as e:
                raise Exception(f"Ollama streaming error: {str(e)}")
    
    async def list_models(self) -> List[ModelInfo]:
        """
        Fetch installed models from Ollama.
        Falls back to hardcoded list if Ollama not running.
        
        Returns:
            List of ModelInfo objects
        """
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                
                models = []
                for model in data.get("models", []):
                    model_name = model.get("name", "")
                    if not model_name:
                        continue
                    
                    models.append(ModelInfo(
                        id=model_name,
                        name=model_name,
                        provider='ollama',
                        context_window=32768,  # Default, varies by model
                        max_output=8192,
                        is_free=True,
                        speed=None,
                        supports_streaming=True,
                        supports_vision=False,
                        supports_function_calling=False,
                    ))
                
                # Sort by name
                models.sort(key=lambda m: m.id)
                
                return models if models else self.FALLBACK_MODELS
                
            except Exception as e:
                print(f"⚠️  Ollama: Not running or not accessible ({e}), using fallback")
                return self.FALLBACK_MODELS
    
    async def health_check(self) -> ProviderHealth:
        """
        Check if Ollama is running.
        
        Returns:
            ProviderHealth with status
        """
        try:
            start_time = time.time()
            models = await self.list_models()
            latency_ms = (time.time() - start_time) * 1000
            
            return ProviderHealth(
                provider="ollama",
                healthy=True,
                latency_ms=latency_ms,
                models_available=len(models)
            )
            
        except Exception as e:
            return ProviderHealth(
                provider="ollama",
                healthy=False,
                error="Ollama not running. Install from https://ollama.com and run 'ollama serve'"
            )
    
    def to_autogen_config(self, model: str) -> dict:
        """
        Convert to AutoGen config format.
        
        Note: AutoGen doesn't natively support Ollama, so this returns
        a custom config that would need an adapter.
        
        Args:
            model: Model ID to use
            
        Returns:
            Config dict (custom format)
        """
        return {
            "model": model,
            "base_url": self.base_url,
            "api_type": "ollama",
            "provider": "ollama",
        }
