"""
Provider registry for managing all providers.
Handles initialization, selection, and fallback chains.
"""

from typing import Dict, List, Optional
import asyncio
from .base import BaseProvider
from .types import ProviderType, ProviderConfig, ProviderHealth


class ProviderRegistry:
    """
    Central registry for all providers.
    
    Usage:
        registry = ProviderRegistry()
        await registry.initialize(configs)
        provider = await registry.get_best_provider()
        response = await provider.chat(request)
    """
    
    def __init__(self):
        self._providers: Dict[str, BaseProvider] = {}
        self._initialized = False
    
    async def initialize(self, configs: Dict[str, ProviderConfig]):
        """
        Initialize all providers with their configs.
        
        Args:
            configs: Dict mapping provider names to their configs
        """
        # Import provider classes here to avoid circular imports
        # Free cloud providers
        from .cloud.groq import GroqProvider
        from .cloud.cerebras import CerebrasProvider
        from .cloud.gemini import GeminiProvider
        from .cloud.openrouter import OpenRouterProvider
        from .cloud.github import GitHubModelsProvider
        from .cloud.huggingface import HuggingFaceProvider
        
        # Local providers
        from .local.ollama import OllamaProvider
        from .local.lmstudio import LMStudioProvider
        from .local.llamacpp import LlamaCppProvider
        
        # Paid providers
        from .paid.openai import OpenAIProvider
        from .paid.claude import ClaudeProvider
        from .paid.moonshot import MoonshotProvider
        
        provider_classes = {
            # Free cloud (6)
            'groq': GroqProvider,
            'cerebras': CerebrasProvider,
            'gemini': GeminiProvider,
            'openrouter': OpenRouterProvider,
            'github': GitHubModelsProvider,
            'huggingface': HuggingFaceProvider,
            # Local (3)
            'ollama': OllamaProvider,
            'lmstudio': LMStudioProvider,
            'llamacpp': LlamaCppProvider,
            # Paid (3)
            'openai': OpenAIProvider,
            'claude': ClaudeProvider,
            'moonshot': MoonshotProvider,
        }
        
        for name, config in configs.items():
            if name in provider_classes:
                try:
                    self._providers[name] = provider_classes[name](config)
                except Exception as e:
                    print(f"⚠️  Failed to initialize {name}: {e}")
        
        self._initialized = True
        print(f"✅ Initialized {len(self._providers)} providers")
    
    async def health_check_all(self) -> Dict[str, ProviderHealth]:
        """
        Run health checks on all providers.
        
        Returns:
            Dict mapping provider names to their health status
        """
        if not self._initialized:
            raise RuntimeError("Registry not initialized. Call initialize() first.")
        
        health_checks = {}
        tasks = []
        
        for name, provider in self._providers.items():
            tasks.append(provider.health_check())
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for (name, provider), result in zip(self._providers.items(), results):
            if isinstance(result, Exception):
                health_checks[name] = ProviderHealth(
                    provider=name,
                    healthy=False,
                    error=str(result)
                )
            else:
                health_checks[name] = result
        
        return health_checks
    
    async def get_available_providers(
        self,
        prefer_free: bool = True,
        include_local: bool = True,
        include_paid: bool = False
    ) -> List[BaseProvider]:
        """
        Get list of available (healthy) providers.
        
        Args:
            prefer_free: Prioritize free cloud providers
            include_local: Include local providers
            include_paid: Include paid providers
            
        Returns:
            List of healthy providers sorted by priority
        """
        if not self._initialized:
            raise RuntimeError("Registry not initialized. Call initialize() first.")
        
        available = []
        health_checks = await self.health_check_all()
        
        for name, provider in self._providers.items():
            health = health_checks.get(name)
            if not health or not health.healthy:
                continue
            
            # Check type filters
            if provider.type == ProviderType.CLOUD_FREE and not prefer_free:
                continue
            if provider.type == ProviderType.LOCAL and not include_local:
                continue
            if provider.type == ProviderType.PAID and not include_paid:
                continue
            
            available.append(provider)
        
        # Sort by priority: free > local > paid
        def priority(p: BaseProvider) -> int:
            if p.type == ProviderType.CLOUD_FREE:
                return 0
            elif p.type == ProviderType.LOCAL:
                return 1
            else:
                return 2
        
        return sorted(available, key=priority)
    
    async def get_best_provider(
        self,
        prefer_free: bool = True,
        prefer_local: bool = False
    ) -> Optional[BaseProvider]:
        """
        Get the best available provider based on preferences.
        
        Priority:
        1. Free cloud (if prefer_free=True)
        2. Local (if prefer_local=True)
        3. Paid (as fallback)
        
        Args:
            prefer_free: Prioritize free providers
            prefer_local: Prioritize local providers
            
        Returns:
            Best available provider or None
        """
        providers = await self.get_available_providers(
            prefer_free=prefer_free,
            include_local=True,
            include_paid=True
        )
        
        if not providers:
            return None
        
        if prefer_local:
            local = [p for p in providers if p.is_local()]
            if local:
                return local[0]
        
        if prefer_free:
            free = [p for p in providers if p.is_free()]
            if free:
                return free[0]
        
        return providers[0]
    
    def get_provider(self, name: str) -> Optional[BaseProvider]:
        """
        Get specific provider by name.
        
        Args:
            name: Provider name
            
        Returns:
            Provider instance or None if not found
        """
        return self._providers.get(name)
    
    async def create_fallback_chain(
        self,
        primary_providers: List[str],
        include_paid_fallback: bool = True
    ) -> List[BaseProvider]:
        """
        Create a fallback chain of providers.
        
        Example:
            ['groq', 'cerebras', 'gemini'] + paid fallbacks
        
        Args:
            primary_providers: List of provider names to try first
            include_paid_fallback: Add paid providers as fallback
            
        Returns:
            List of healthy providers in fallback order
        """
        chain = []
        health_checks = await self.health_check_all()
        
        # Add primary providers
        for name in primary_providers:
            provider = self.get_provider(name)
            health = health_checks.get(name)
            if provider and health and health.healthy:
                chain.append(provider)
        
        # Add paid fallbacks if requested
        if include_paid_fallback:
            for name in ['openai', 'claude', 'moonshot']:
                provider = self.get_provider(name)
                health = health_checks.get(name)
                if provider and health and health.healthy:
                    chain.append(provider)
        
        return chain
    
    async def get_autogen_config_list(
        self,
        providers: Optional[List[str]] = None,
        include_paid_fallback: bool = True
    ) -> List[Dict]:
        """
        Build AutoGen-compatible config_list with fallback chain.
        
        Args:
            providers: Specific providers to use (None = all free)
            include_paid_fallback: Add paid providers as fallback
            
        Returns:
            List of config dicts for AutoGen agents
        """
        if providers is None:
            # Get all free cloud providers
            available = await self.get_available_providers(
                prefer_free=True,
                include_local=False,
                include_paid=False
            )
            providers = [p.name for p in available]
        
        # Build fallback chain
        chain = await self.create_fallback_chain(
            providers,
            include_paid_fallback
        )
        
        # Convert to AutoGen configs
        configs = []
        for provider in chain:
            # Get default model for provider
            models = await provider.list_models()
            if models:
                # Use first model (providers should return recommended first)
                model = models[0]
                configs.append(provider.to_autogen_config(model.id))
        
        return configs
    
    def list_providers(self) -> List[str]:
        """
        Get list of all initialized provider names.
        
        Returns:
            List of provider names
        """
        return list(self._providers.keys())
    
    def __str__(self) -> str:
        """Human-readable registry info"""
        return f"ProviderRegistry({len(self._providers)} providers)"
    
    def __repr__(self) -> str:
        """Developer-friendly registry info"""
        provider_names = ', '.join(self._providers.keys())
        return f"<ProviderRegistry providers=[{provider_names}]>"


# Global registry instance
registry = ProviderRegistry()
