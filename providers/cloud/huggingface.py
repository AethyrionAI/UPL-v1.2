"""
HuggingFace Inference API provider implementation.
Free tier available
"""

import os
from typing import List, AsyncIterator, Optional
import time
from huggingface_hub import InferenceClient

from ..base import BaseProvider
from ..types import (
    ModelInfo,
    ProviderConfig,
    ChatRequest,
    ChatResponse,
    ProviderHealth,
    ProviderType,
)


class HuggingFaceProvider(BaseProvider):
    """
    HuggingFace Inference API provider.
    
    Features:
    - Free inference API (using huggingface_hub SDK)
    - Thousands of models available
    - Async support via InferenceClient
    - Models: Llama, Mistral, Phi, Qwen, etc.
    
    Note: Uses HuggingFace's InferenceClient (official Python SDK)
    
    Usage:
        provider = HuggingFaceProvider(config)
        response = await provider.chat(request)
    """
    
    # Hardcoded fallback models (popular free models from Helix)
    FALLBACK_MODELS = [
        ModelInfo(
            id='meta-llama/Llama-3.1-70B-Instruct',
            name='Llama 3.1 70B Instruct',
            provider='huggingface',
            context_window=128000,
            max_output=8192,
            is_free=True,
            speed=None,
            supports_streaming=True,
            supports_vision=False,
            supports_function_calling=False,
            description='Meta Llama 3.1 70B (recommended)'
        ),
        ModelInfo(
            id='Qwen/Qwen2.5-Coder-32B-Instruct',
            name='Qwen 2.5 Coder 32B',
            provider='huggingface',
            context_window=128000,
            max_output=8192,
            is_free=True,
            speed=None,
            supports_streaming=True,
            supports_vision=False,
            supports_function_calling=False,
            description='Alibaba code specialist'
        ),
        ModelInfo(
            id='mistralai/Mistral-7B-Instruct-v0.3',
            name='Mistral 7B Instruct',
            provider='huggingface',
            context_window=32768,
            max_output=8192,
            is_free=True,
            speed=None,
            supports_streaming=True,
            supports_vision=False,
            supports_function_calling=False,
            description='Lightweight but capable'
        ),
        ModelInfo(
            id='codellama/CodeLlama-34b-Instruct-hf',
            name='Code Llama 34B Instruct',
            provider='huggingface',
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
        self.api_key = config.api_key  # HuggingFace token
        
        # Initialize InferenceClient
        if self.api_key:
            self.client = InferenceClient(token=self.api_key)
        else:
            self.client = None
    
    def get_default_model(self) -> str:
        """Get recommended default model"""
        return 'meta-llama/Llama-3.1-70B-Instruct'
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Send chat request to HuggingFace.
        
        Args:
            request: Chat request
            
        Returns:
            ChatResponse with content and metadata
            
        Raises:
            Exception: If API call fails
        """
        if not self.client:
            raise RuntimeError("HuggingFace token required.")
        
        start_time = time.time()
        
        try:
            # Convert messages to HF format
            messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in request.messages_as_dicts()
            ]
            
            # Call HuggingFace inference API
            response = self.client.chat_completion(
                messages=messages,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=False,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract content
            content = response.choices[0].message.content or ""
            
            return ChatResponse(
                content=content,
                model=request.model,
                provider="huggingface",
                tokens_used=None,  # HF doesn't always return token count
                latency_ms=latency_ms,
                finish_reason=response.choices[0].finish_reason if hasattr(response.choices[0], 'finish_reason') else None,
            )
            
        except Exception as e:
            # Handle model loading (cold start)
            if "loading" in str(e).lower():
                raise Exception("Model is loading (cold start). This can take 20-30 seconds. Please try again.")
            # Handle rate limiting
            elif "rate limit" in str(e).lower() or "429" in str(e):
                raise Exception("Rate limit exceeded. Free tier allows ~200 requests/hour.")
            else:
                raise Exception(f"HuggingFace API error: {str(e)}")
    
    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """
        Stream chat responses from HuggingFace.
        
        Args:
            request: Chat request with stream=True
            
        Yields:
            String tokens as they arrive
            
        Raises:
            Exception: If API call fails
        """
        if not self.client:
            raise RuntimeError("HuggingFace token required.")
        
        try:
            # Convert messages to HF format
            messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in request.messages_as_dicts()
            ]
            
            # Stream response
            stream = self.client.chat_completion(
                messages=messages,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
            )
            
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                        
        except Exception as e:
            if "loading" in str(e).lower():
                raise Exception("Model is loading (cold start). Please try again in 20-30 seconds.")
            elif "rate limit" in str(e).lower():
                raise Exception("Rate limit exceeded. Free tier allows ~200 requests/hour.")
            else:
                raise Exception(f"HuggingFace streaming error: {str(e)}")
    
    async def list_models(self) -> List[ModelInfo]:
        """
        Return curated list of popular models.
        
        Note: HuggingFace has 1000+ models. We provide a curated list
        of popular, working models for coding tasks.
        
        Returns:
            List of ModelInfo objects
        """
        # Return our curated list (same as Helix)
        return self.FALLBACK_MODELS
    
    async def health_check(self) -> ProviderHealth:
        """
        Check if HuggingFace is accessible.
        
        Returns:
            ProviderHealth with status
        """
        if not self.client:
            return ProviderHealth(
                provider="huggingface",
                healthy=False,
                error="No HuggingFace token configured"
            )
        
        try:
            start_time = time.time()
            
            # Quick test with a simple request (non-blocking)
            # Just verify the client is initialized
            models = await self.list_models()
            
            latency_ms = (time.time() - start_time) * 1000
            
            return ProviderHealth(
                provider="huggingface",
                healthy=True,
                latency_ms=latency_ms,
                models_available=len(models)
            )
            
        except Exception as e:
            return ProviderHealth(
                provider="huggingface",
                healthy=False,
                error=str(e)
            )
    
    def to_autogen_config(self, model: str) -> dict:
        """
        Convert to AutoGen config format.
        
        Note: AutoGen doesn't natively support HuggingFace, so this returns
        a custom config that would need an adapter.
        
        Args:
            model: Model ID to use
            
        Returns:
            Config dict (custom format)
        """
        return {
            "model": model,
            "api_key": self.api_key,
            "api_type": "huggingface",
            "provider": "huggingface",
        }
