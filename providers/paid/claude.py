"""
Claude (Anthropic) paid provider implementation.
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


class ClaudeProvider(BaseProvider):
    """
    Claude (Anthropic) provider.
    
    Features:
    - High-quality responses
    - Long context windows (200K+)
    - Function calling support
    - Prompt caching
    
    Pricing:
    - Sonnet: $3/$15 per million tokens (input/output)
    - Opus: $15/$75 per million tokens
    
    API Key:
    - Get from: https://console.anthropic.com/
    - Set env: ANTHROPIC_API_KEY
    
    Usage:
        provider = ClaudeProvider(config)
        response = await provider.chat(request)
    """
    
    BASE_URL = "https://api.anthropic.com/v1"
    API_VERSION = "2023-06-01"
    
    # Updated fallback models (November 2025)
    FALLBACK_MODELS = [
        ModelInfo(
            id='claude-sonnet-4-5-20250929',
            name='Claude Sonnet 4.5',
            provider='claude',
            context_window=200000,
            max_output=8192,
            is_free=False,
            speed=None,
            supports_streaming=True,
            supports_vision=True,
            supports_function_calling=True,
            supports_thinking=True,
            description='Best for code generation (recommended) • Thinking Mode'
        ),
        ModelInfo(
            id='claude-haiku-4-5-20251001',
            name='Claude Haiku 4.5',
            provider='claude',
            context_window=200000,
            max_output=8192,
            is_free=False,
            speed=None,
            supports_streaming=True,
            supports_vision=True,
            supports_function_calling=True,
            supports_thinking=True,
            description='Fastest & cheapest - better than Haiku 4 • Thinking Mode'
        ),
        ModelInfo(
            id='claude-opus-4-1-20250805',
            name='Claude Opus 4.1',
            provider='claude',
            context_window=200000,
            max_output=8192,
            is_free=False,
            speed=None,
            supports_streaming=True,
            supports_vision=True,
            supports_function_calling=True,
            supports_thinking=True,
            description='Most powerful - complex decisions • Thinking Mode'
        ),
        ModelInfo(
            id='claude-sonnet-4-20250514',
            name='Claude Sonnet 4',
            provider='claude',
            context_window=200000,
            max_output=8192,
            is_free=False,
            speed=None,
            supports_streaming=True,
            supports_vision=True,
            supports_function_calling=True,
            supports_thinking=True,
            description='Previous version - still excellent • Thinking Mode'
        ),
        ModelInfo(
            id='claude-opus-4-20250514',
            name='Claude Opus 4',
            provider='claude',
            context_window=200000,
            max_output=8192,
            is_free=False,
            speed=None,
            supports_streaming=True,
            supports_vision=True,
            supports_function_calling=True,
            supports_thinking=True,
            description='Previous premium version • Thinking Mode'
        ),
        ModelInfo(
            id='claude-3-7-sonnet-20250219',
            name='Claude Sonnet 3.7',
            provider='claude',
            context_window=200000,
            max_output=8192,
            is_free=False,
            speed=None,
            supports_streaming=True,
            supports_vision=True,
            supports_function_calling=True,
            description='Claude 3.7 generation'
        ),
        ModelInfo(
            id='claude-3-5-haiku-20241022',
            name='Claude Haiku 3.5',
            provider='claude',
            context_window=200000,
            max_output=8192,
            is_free=False,
            speed=None,
            supports_streaming=True,
            supports_vision=True,
            supports_function_calling=True,
            description='Fast Claude 3.5 model'
        ),
        ModelInfo(
            id='claude-3-haiku-20240307',
            name='Claude Haiku 3',
            provider='claude',
            context_window=200000,
            max_output=4096,
            is_free=False,
            speed=None,
            supports_streaming=True,
            supports_vision=False,
            supports_function_calling=True,
            description='Original Claude 3 Haiku'
        ),
    ]
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.api_key = config.api_key
        self.base_url = config.base_url or self.BASE_URL
        
        if not self.api_key:
            raise ValueError("Claude API key required. Get one from https://console.anthropic.com/")
    
    def get_default_model(self) -> str:
        """Get recommended default model"""
        return 'claude-sonnet-4-5-20250929'
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Send chat request to Claude with prompt caching support.
        
        Args:
            request: Chat request
            
        Returns:
            ChatResponse with content and metadata
            
        Raises:
            Exception: If API call fails
        """
        start_time = time.time()
        
        # Convert messages for Claude format
        messages = []
        system_prompt = None
        system_blocks = []
        
        for msg in request.messages_as_dicts():
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Check for project context in request metadata
        project_context = None
        if hasattr(request, 'metadata') and request.metadata:
            project_context = request.metadata.get('project_context')
        
        # Build system blocks with caching for project files
        if project_context:
            from core.project_context import format_project_cache_blocks
            # Add cached project file blocks
            system_blocks.extend(format_project_cache_blocks(project_context))
        
        # Add regular system prompt (not cached - this changes more often)
        if system_prompt:
            system_blocks.append({
                "type": "text",
                "text": system_prompt
            })
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            try:
                payload = {
                    "model": request.model,
                    "messages": messages,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                }

                # Use system blocks if we have project context, otherwise simple string
                if system_blocks:
                    payload["system"] = system_blocks
                elif system_prompt:
                    payload["system"] = system_prompt

                # Enable thinking mode for Claude 4.x models
                if hasattr(request, 'thinking_mode') and request.thinking_mode:
                    # Support all Claude 4.x models (4.5, 4.1, 4.0)
                    if any(x in request.model for x in ['claude-sonnet-4', 'claude-opus-4', 'claude-haiku-4']):
                        payload["thinking"] = {
                            "type": "enabled",
                            "budget_tokens": 10000  # Allow up to 10K tokens for thinking
                        }
                        # CRITICAL: Claude requires temperature=1.0 when thinking is enabled!
                        payload["temperature"] = 1.0

                response = await client.post(
                    f"{self.base_url}/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": self.API_VERSION,
                        "content-type": "application/json",
                    },
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                
                latency_ms = (time.time() - start_time) * 1000
                
                # Extract text content from blocks
                content = ""
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        content += block.get("text", "")
                
                # Get token usage
                usage = data.get("usage", {})
                tokens_used = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                
                return ChatResponse(
                    content=content,
                    model=data.get("model", request.model),
                    provider="claude",
                    tokens_used=tokens_used,
                    latency_ms=latency_ms,
                    finish_reason=data.get("stop_reason"),
                )
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    raise Exception("Invalid Claude API key. Check your ANTHROPIC_API_KEY.")
                elif e.response.status_code == 429:
                    raise Exception("Rate limit exceeded. Please wait and try again.")
                elif e.response.status_code == 529:
                    raise Exception("Claude API is temporarily overloaded. Please try again.")
                else:
                    raise Exception(f"Claude API HTTP error: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                raise Exception(f"Claude API error: {str(e)}")
    
    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """
        Stream chat responses from Claude.
        
        Args:
            request: Chat request with stream=True
            
        Yields:
            String tokens as they arrive
            
        Raises:
            Exception: If API call fails
        """
        # Convert messages for Claude format
        messages = []
        system_prompt = None
        
        for msg in request.messages_as_dicts():
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            try:
                payload = {
                    "model": request.model,
                    "messages": messages,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "stream": True,
                }

                if system_prompt:
                    payload["system"] = system_prompt

                # Enable thinking mode for Claude 4.x models
                if hasattr(request, 'thinking_mode') and request.thinking_mode:
                    # Support all Claude 4.x models (4.5, 4.1, 4.0)
                    if any(x in request.model for x in ['claude-sonnet-4', 'claude-opus-4', 'claude-haiku-4']):
                        payload["thinking"] = {
                            "type": "enabled",
                            "budget_tokens": 10000  # Allow up to 10K tokens for thinking
                        }
                        # CRITICAL: Claude requires temperature=1.0 when thinking is enabled!
                        payload["temperature"] = 1.0

                async with client.stream(
                    "POST",
                    f"{self.base_url}/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": self.API_VERSION,
                        "content-type": "application/json",
                    },
                    json=payload
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        
                        data_str = line[6:]  # Remove "data: " prefix
                        
                        try:
                            import json
                            data = json.loads(data_str)
                            
                            # Handle different event types
                            event_type = data.get("type")
                            
                            if event_type == "content_block_delta":
                                delta = data.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    text = delta.get("text", "")
                                    if text:
                                        yield text
                        except json.JSONDecodeError:
                            continue
                            
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    raise Exception("Invalid Claude API key.")
                elif e.response.status_code == 429:
                    raise Exception("Rate limit exceeded.")
                else:
                    raise Exception(f"Claude streaming error: {e.response.status_code}")
            except Exception as e:
                raise Exception(f"Claude streaming error: {str(e)}")
    
    async def list_models(self) -> List[ModelInfo]:
        """
        Fetch current Claude models from Anthropic API (NEW!).
        Falls back to hardcoded list if API call fails.
        
        Returns:
            List of ModelInfo objects
        """
        try:
            # Try to fetch dynamic models from Anthropic's models API
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": self.API_VERSION,
                    },
                    params={"limit": 100}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    models = []
                    
                    for model_data in data.get("data", []):
                        model_id = model_data.get("id")
                        # Detect thinking support for Claude 4.x models
                        supports_thinking = any(x in model_id for x in ['claude-sonnet-4', 'claude-opus-4', 'claude-haiku-4'])

                        # Convert API response to ModelInfo
                        models.append(ModelInfo(
                            id=model_id,
                            name=model_data.get("display_name", model_id),
                            provider='claude',
                            context_window=200000,  # All Claude models have 200K
                            max_output=8192,  # Standard max output
                            is_free=False,
                            speed=None,
                            supports_streaming=True,
                            supports_vision=True,
                            supports_function_calling=True,
                            supports_thinking=supports_thinking,
                            description=f"{model_data.get('display_name', 'Claude model')}{' • Thinking Mode' if supports_thinking else ''}"
                        ))
                    
                    if models:
                        return models
        
        except Exception as e:
            # If API call fails, fall back to hardcoded list
            pass
        
        # Fallback to hardcoded list
        return self.FALLBACK_MODELS
    
    async def health_check(self) -> ProviderHealth:
        """
        Check if Claude API is accessible.
        
        Returns:
            ProviderHealth with status
        """
        if not self.api_key:
            return ProviderHealth(
                provider="claude",
                healthy=False,
                error="No Claude API key configured"
            )
        
        try:
            start_time = time.time()
            
            # Quick health check with minimal token usage
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    f"{self.base_url}/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": self.API_VERSION,
                        "content-type": "application/json",
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
                provider="claude",
                healthy=True,
                latency_ms=latency_ms,
                models_available=len(self.FALLBACK_MODELS)
            )
            
        except Exception as e:
            return ProviderHealth(
                provider="claude",
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
            "api_type": "anthropic",
            "provider": "claude",
        }
