"""
Google Gemini provider implementation.
Free tier: 1,500 requests/day
"""

import os
from typing import List, AsyncIterator, Optional
import time
import google.generativeai as genai

from ..base import BaseProvider
from ..types import (
    ModelInfo,
    ProviderConfig,
    ChatRequest,
    ChatResponse,
    ProviderHealth,
    ProviderType,
)


class GeminiProvider(BaseProvider):
    """
    Google Gemini provider.
    
    Features:
    - Free tier: 1,500 requests/day
    - Fast inference
    - Large context windows (up to 1M tokens!)
    - Multimodal support (text, images, video, audio)
    - Models: Gemini 2.0 Flash, Gemini 1.5 Pro, etc.
    
    Usage:
        provider = GeminiProvider(config)
        response = await provider.chat(request)
    """
    
    # Hardcoded fallback models (used if API fetch fails)
    FALLBACK_MODELS = [
        # Gemini 2.5 models (ALL have thinking!)
        ModelInfo(
            id='gemini-2.5-flash',
            name='Gemini 2.5 Flash ⚡',
            provider='gemini',
            context_window=1048576,  # 1M tokens
            max_output=8192,
            is_free=True,
            speed=None,
            supports_streaming=True,
            supports_vision=True,
            supports_function_calling=True,
            supports_thinking=True,
            description='Free! 1M context • Thinking Mode'
        ),
        ModelInfo(
            id='gemini-2.5-pro',
            name='Gemini 2.5 Pro 🏆',
            provider='gemini',
            context_window=1048576,  # 1M tokens
            max_output=8192,
            is_free=True,
            speed=None,
            supports_streaming=True,
            supports_vision=True,
            supports_function_calling=True,
            supports_thinking=True,
            description='Free! 1M context • Thinking Mode'
        ),
        # Gemini 1.5 models (no thinking)
        ModelInfo(
            id='gemini-1.5-flash',
            name='Gemini 1.5 Flash',
            provider='gemini',
            context_window=1000000,  # 1M tokens
            max_output=8192,
            is_free=True,
            speed=None,
            supports_streaming=True,
            supports_vision=True,
            supports_function_calling=True,
            supports_thinking=False,
            description='Free! 1M context'
        ),
    ]
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.api_key = config.api_key
        
        # Configure the SDK
        if self.api_key:
            genai.configure(api_key=self.api_key)
    
    def get_default_model(self) -> str:
        """Get recommended default model"""
        return 'gemini-2.5-flash'
    
    def _convert_messages_to_gemini_format(self, messages: list) -> list:
        """
        Convert our universal message format to Gemini's format.
        
        Gemini uses: [{"role": "user"/"model", "parts": ["text"]}]
        We use: [{"role": "user"/"assistant"/"system", "content": "text"}]
        """
        gemini_messages = []
        
        for msg in messages:
            # Map our roles to Gemini's roles
            role = msg["role"]
            if role == "assistant":
                role = "model"  # Gemini uses "model" not "assistant"
            elif role == "system":
                # Gemini doesn't have system role - prepend to first user message
                continue
            
            gemini_messages.append({
                "role": role,
                "parts": [msg["content"]]
            })
        
        # Prepend system message to first user message if exists
        system_content = None
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
                break
        
        if system_content and gemini_messages:
            # Find first user message and prepend system content
            for i, msg in enumerate(gemini_messages):
                if msg["role"] == "user":
                    gemini_messages[i]["parts"][0] = f"{system_content}\n\n{msg['parts'][0]}"
                    break
        
        return gemini_messages
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Send chat request to Gemini.
        
        Args:
            request: Chat request
            
        Returns:
            ChatResponse with content and metadata
            
        Raises:
            Exception: If API call fails
        """
        if not self.api_key:
            raise RuntimeError("Gemini API key required.")
        
        start_time = time.time()
        
        try:
            # Create model instance
            model = genai.GenerativeModel(request.model)
            
            # Convert messages to Gemini format
            gemini_messages = self._convert_messages_to_gemini_format(
                request.messages_as_dicts()
            )

            # Prepare generation config
            gen_config = genai.types.GenerationConfig(
                temperature=request.temperature,
                max_output_tokens=request.max_tokens,
                stop_sequences=request.stop,
            )

            # Enable thinking mode for Gemini 2.5 models
            thinking_config = None
            if hasattr(request, 'thinking_mode') and request.thinking_mode:
                # Check if this is a 2.5 model (ALL 2.5 models support thinking!)
                if '2.5' in request.model or request.model.startswith('gemini-2.5'):
                    # Use medium thinking budget (8192 tokens)
                    thinking_config = genai.types.ThinkingConfig(
                        thinking_budget_tokens=8192
                    )

            # Generate response with or without thinking config
            if thinking_config:
                response = await model.generate_content_async(
                    gemini_messages,
                    generation_config=gen_config,
                    thinking_config=thinking_config,
                )
            else:
                response = await model.generate_content_async(
                    gemini_messages,
                    generation_config=gen_config,
                )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract content
            content = ""
            if response.candidates:
                content = response.candidates[0].content.parts[0].text
            
            # Calculate tokens (Gemini provides this in usage_metadata)
            tokens_used = None
            if hasattr(response, 'usage_metadata'):
                tokens_used = (
                    response.usage_metadata.prompt_token_count +
                    response.usage_metadata.candidates_token_count
                )
            
            return ChatResponse(
                content=content,
                model=request.model,
                provider="gemini",
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                finish_reason=str(response.candidates[0].finish_reason) if response.candidates else None,
            )
            
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")
    
    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """
        Stream chat responses from Gemini.
        
        Args:
            request: Chat request with stream=True
            
        Yields:
            String tokens as they arrive
            
        Raises:
            Exception: If API call fails
        """
        if not self.api_key:
            raise RuntimeError("Gemini API key required.")
        
        try:
            # Create model instance
            model = genai.GenerativeModel(request.model)
            
            # Convert messages to Gemini format
            gemini_messages = self._convert_messages_to_gemini_format(
                request.messages_as_dicts()
            )

            # Prepare generation config
            gen_config = genai.types.GenerationConfig(
                temperature=request.temperature,
                max_output_tokens=request.max_tokens,
                stop_sequences=request.stop,
            )

            # Enable thinking mode for Gemini 2.5 models
            thinking_config = None
            if hasattr(request, 'thinking_mode') and request.thinking_mode:
                # Check if this is a 2.5 model (ALL 2.5 models support thinking!)
                if '2.5' in request.model or request.model.startswith('gemini-2.5'):
                    # Use medium thinking budget (8192 tokens)
                    thinking_config = genai.types.ThinkingConfig(
                        thinking_budget_tokens=8192
                    )

            # Stream response with or without thinking config
            if thinking_config:
                response = await model.generate_content_async(
                    gemini_messages,
                    generation_config=gen_config,
                    thinking_config=thinking_config,
                    stream=True,
                )
            else:
                response = await model.generate_content_async(
                    gemini_messages,
                    generation_config=gen_config,
                    stream=True,
                )
            
            async for chunk in response:
                if chunk.candidates and chunk.candidates[0].content.parts:
                    text = chunk.candidates[0].content.parts[0].text
                    if text:
                        yield text
                        
        except Exception as e:
            raise Exception(f"Gemini streaming error: {str(e)}")
    
    async def list_models(self) -> List[ModelInfo]:
        """
        Fetch available models from Gemini API.
        Falls back to hardcoded list if API call fails.
        
        Returns:
            List of ModelInfo objects
        """
        if not self.api_key:
            print("⚠️  Gemini: No API key, using fallback models")
            return self.FALLBACK_MODELS
        
        try:
            # List all available models
            models = []
            for model in genai.list_models():
                # Only include generative models (not embedding, etc.)
                if 'generateContent' not in model.supported_generation_methods:
                    continue
                
                # Parse context window
                context_window = model.input_token_limit if hasattr(model, 'input_token_limit') else 32768
                max_output = model.output_token_limit if hasattr(model, 'output_token_limit') else 8192

                model_id = model.name.replace('models/', '')  # Remove 'models/' prefix

                # ALL Gemini 2.5 series models support thinking!
                # Per Google docs: "Thinking features are supported on all the 2.5 series models"
                supports_thinking = (
                    model_id.startswith('gemini-2.5') or
                    '2.5' in model_id or
                    'thinking' in model_id.lower()
                )

                # Add emoji for thinking models
                emoji = " ⚡" if supports_thinking else ""
                display_name = model.display_name if hasattr(model, 'display_name') else model.name

                # Update description for thinking models
                description = 'Free! 1M context • Thinking Mode' if supports_thinking and context_window >= 1000000 else 'Free tier'

                models.append(ModelInfo(
                    id=model_id,
                    name=f"{display_name}{emoji}",
                    provider='gemini',
                    context_window=context_window,
                    max_output=max_output,
                    is_free=True,
                    speed=None,
                    supports_streaming=True,
                    supports_vision='vision' in model.name.lower(),
                    supports_function_calling=True,
                    supports_thinking=supports_thinking,
                    description=description
                ))
            
            # Sort by context window (larger first)
            models.sort(key=lambda m: m.context_window, reverse=True)
            
            return models if models else self.FALLBACK_MODELS
            
        except Exception as e:
            print(f"⚠️  Gemini: Failed to fetch models ({e}), using fallback")
            return self.FALLBACK_MODELS
    
    async def health_check(self) -> ProviderHealth:
        """
        Check if Gemini is accessible.
        
        Returns:
            ProviderHealth with status
        """
        if not self.api_key:
            return ProviderHealth(
                provider="gemini",
                healthy=False,
                error="No API key configured"
            )
        
        try:
            start_time = time.time()
            models = await self.list_models()
            latency_ms = (time.time() - start_time) * 1000
            
            return ProviderHealth(
                provider="gemini",
                healthy=True,
                latency_ms=latency_ms,
                models_available=len(models)
            )
            
        except Exception as e:
            return ProviderHealth(
                provider="gemini",
                healthy=False,
                error=str(e)
            )
    
    def to_autogen_config(self, model: str) -> dict:
        """
        Convert to AutoGen config format.
        
        Note: AutoGen doesn't natively support Gemini, so this returns
        a config that could be used with a custom adapter.
        
        Args:
            model: Model ID to use
            
        Returns:
            Config dict (custom format)
        """
        return {
            "model": model,
            "api_key": self.api_key,
            "api_type": "google",
            "provider": "gemini",
        }
