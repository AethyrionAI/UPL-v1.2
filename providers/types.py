"""
Universal types for provider abstraction layer.
Shared across all ACx projects (AC1, AC2, AC3, AC4).
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Literal
from enum import Enum


class ProviderType(Enum):
    """Provider categories"""
    CLOUD_FREE = "cloud_free"
    LOCAL = "local"
    PAID = "paid"


@dataclass
class ModelInfo:
    """Universal model information"""
    id: str                              # Unique model ID
    name: str                            # Display name
    provider: str                        # Provider name
    context_window: int                  # Max context tokens
    max_output: int                      # Max output tokens
    is_free: bool                        # Free tier available
    speed: Optional[float] = None        # Tokens/sec (if known)
    supports_streaming: bool = True      # Streaming support
    supports_vision: bool = False        # Image input support
    supports_function_calling: bool = True  # Tool/function calls
    supports_thinking: bool = False      # Extended reasoning/thinking mode
    description: Optional[str] = None    # Model description
    
    def __str__(self) -> str:
        """Human-readable model info"""
        speed_info = f" ({self.speed} T/s)" if self.speed else ""
        return f"{self.name}{speed_info} - {self.context_window:,} tokens"


@dataclass
class ProviderConfig:
    """Provider configuration"""
    name: str
    type: ProviderType
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 120
    max_retries: int = 3
    rate_limit: Optional[int] = None     # Requests per minute


@dataclass
class ChatMessage:
    """Universal chat message format"""
    role: Literal["system", "user", "assistant"]
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for API calls"""
        return {"role": self.role, "content": self.content}


@dataclass
class ChatRequest:
    """Universal chat request"""
    messages: List[ChatMessage]
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = False
    stop: Optional[List[str]] = None
    thinking_mode: bool = False          # Enable extended reasoning
    metadata: Optional[Dict[str, Any]] = None  # Additional context (e.g., project files)
    
    def messages_as_dicts(self) -> List[Dict[str, str]]:
        """Convert messages to list of dicts for API calls"""
        return [msg.to_dict() for msg in self.messages]


@dataclass
class ChatResponse:
    """Universal chat response"""
    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None
    finish_reason: Optional[str] = None
    
    def __str__(self) -> str:
        """Human-readable response info"""
        tokens_info = f" ({self.tokens_used} tokens)" if self.tokens_used else ""
        latency_info = f" in {self.latency_ms}ms" if self.latency_ms else ""
        return f"[{self.provider}/{self.model}]{tokens_info}{latency_info}"


@dataclass
class ProviderHealth:
    """Provider health status"""
    provider: str
    healthy: bool
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    models_available: int = 0
    
    def __str__(self) -> str:
        """Human-readable health status"""
        if self.healthy:
            return f"✅ {self.provider}: {self.models_available} models available"
        else:
            return f"❌ {self.provider}: {self.error}"
