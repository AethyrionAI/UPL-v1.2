"""
Base provider interface that all providers must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, AsyncIterator, Optional
import time
from .types import (
    ModelInfo, 
    ProviderConfig, 
    ChatRequest, 
    ChatResponse,
    ProviderHealth,
    ProviderType
)


class BaseProvider(ABC):
    """
    Base interface all providers must implement.
    
    Usage:
        provider = SomeProvider(config)
        response = await provider.chat(request)
    """
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.name = config.name
        self.type = config.type
    
    # Abstract methods - must be implemented by all providers
    
    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Send chat request and get response.
        
        Args:
            request: Chat request with messages and parameters
            
        Returns:
            ChatResponse with content and metadata
            
        Raises:
            Exception: If API call fails
        """
        pass
    
    @abstractmethod
    async def chat_stream(
        self, 
        request: ChatRequest
    ) -> AsyncIterator[str]:
        """
        Stream chat responses token by token.
        
        Args:
            request: Chat request with messages and parameters
            
        Yields:
            String tokens as they arrive
            
        Raises:
            Exception: If API call fails
        """
        pass
    
    @abstractmethod
    async def list_models(self) -> List[ModelInfo]:
        """
        Get list of available models from provider.
        Should fetch dynamically when possible, fallback to hardcoded list.
        
        Returns:
            List of ModelInfo objects
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> ProviderHealth:
        """
        Check if provider is accessible and healthy.
        Should verify API key, server status, etc.
        
        Returns:
            ProviderHealth with status and metadata
        """
        pass
    
    @abstractmethod
    def get_default_model(self) -> str:
        """
        Get recommended default model for provider.
        
        Returns:
            Model ID string
        """
        pass
    
    # Helper methods - implemented in base class
    
    def is_free(self) -> bool:
        """Check if provider is free tier"""
        return self.type == ProviderType.CLOUD_FREE
    
    def is_local(self) -> bool:
        """Check if provider runs locally"""
        return self.type == ProviderType.LOCAL
    
    def requires_api_key(self) -> bool:
        """Check if provider needs API key"""
        return not self.is_local()
    
    def has_api_key(self) -> bool:
        """Check if API key is configured"""
        return bool(self.config.api_key)
    
    # Framework-specific adapters
    
    def to_autogen_config(self, model: str) -> Dict:
        """
        Convert to AutoGen config_list format.
        Override if provider needs special handling.
        
        Args:
            model: Model ID to use
            
        Returns:
            Config dict for AutoGen agents
        """
        return {
            "model": model,
            "api_key": self.config.api_key,
            "base_url": self.config.base_url,
            "api_type": "openai",  # Most use OpenAI-compatible API
        }
    
    def to_kosong_config(self, model: str) -> Dict:
        """
        Convert to AC1's kosong format.
        Override if needed.
        
        Args:
            model: Model ID to use
            
        Returns:
            Config dict for kosong
        """
        return {
            "provider": self.name,
            "model": model,
            "api_key": self.config.api_key,
            "base_url": self.config.base_url,
        }
    
    def to_litellm_config(self, model: str) -> str:
        """
        Convert to LiteLLM format (for AC2/ADK).
        Format: "provider/model"
        
        Args:
            model: Model ID to use
            
        Returns:
            LiteLLM model string
        """
        return f"{self.name}/{model}"
    
    # Utility methods
    
    async def _measure_latency(self, func):
        """
        Measure latency of an async function call.
        
        Args:
            func: Async function to measure
            
        Returns:
            Tuple of (result, latency_ms)
        """
        start = time.time()
        result = await func()
        latency_ms = (time.time() - start) * 1000
        return result, latency_ms
    
    def __str__(self) -> str:
        """Human-readable provider info"""
        return f"{self.name} ({self.type.value})"
    
    def __repr__(self) -> str:
        """Developer-friendly provider info"""
        return f"<{self.__class__.__name__} name={self.name} type={self.type.value}>"
    
    # Project context injection (universal approach)
    
    def _inject_project_context(
        self, 
        messages: List[Dict],
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Inject project files into messages as system prompt.
        
        This is the universal approach that works for all providers.
        Providers with caching (like Claude) can override this method
        to use their custom caching implementation.
        
        Args:
            messages: List of message dicts with role/content
            metadata: Request metadata containing project_context
            
        Returns:
            Modified messages list with project files injected
        """
        # Skip if no metadata or no project context
        if not metadata:
            return messages
        
        project_context = metadata.get('project_context')
        if not project_context:
            return messages
        
        # Import here to avoid circular dependency
        try:
            from core.project_context import format_project_context
        except ImportError:
            # If we can't import, just return original messages
            return messages
        
        # Format project files as plain text
        files_text = format_project_context(project_context, include_summary=True)
        
        # Find existing system message or create new one
        system_msg = None
        other_msgs = []
        
        for msg in messages:
            if msg.get('role') == 'system':
                system_msg = msg
            else:
                other_msgs.append(msg)
        
        # Prepend project files to system message
        if system_msg:
            # Add files before existing system prompt
            system_msg['content'] = f"{files_text}\n\n{system_msg['content']}"
        else:
            # Create new system message with just files
            system_msg = {'role': 'system', 'content': files_text}
        
        # Return with system message first
        return [system_msg] + other_msgs
