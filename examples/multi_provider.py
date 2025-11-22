"""
Multi-provider UPL usage example
Shows how to use multiple providers with fallback chains
"""

import asyncio
import os
from dotenv import load_dotenv
from providers import registry, ProviderConfig, ChatRequest, ChatMessage, ProviderType


async def main():
    load_dotenv()
    
    print("=" * 80)
    print("UPL MULTI-PROVIDER EXAMPLE")
    print("=" * 80)
    print()
    
    # Initialize multiple providers
    configs = {
        'groq': ProviderConfig(
            name='groq',
            type=ProviderType.CLOUD_FREE,
            api_key=os.getenv('GROQ_API_KEY')
        ),
        'cerebras': ProviderConfig(
            name='cerebras',
            type=ProviderType.CLOUD_FREE,
            api_key=os.getenv('CEREBRAS_API_KEY')
        ),
        'openai': ProviderConfig(
            name='openai',
            type=ProviderType.PAID,
            api_key=os.getenv('OPENAI_API_KEY')
        ),
    }
    
    print("📦 Initializing providers...")
    await registry.initialize(configs)
    print()
    
    # Run health checks
    print("🏥 Running health checks...")
    health_checks = await registry.health_check_all()
    for name, health in health_checks.items():
        if health.healthy:
            print(f"   ✅ {name}: {health.models_available} models ({health.latency_ms:.0f}ms)")
        else:
            print(f"   ❌ {name}: {health.error}")
    print()
    
    # Test each provider
    request = ChatRequest(
        model="",  # Will be set per provider
        messages=[
            ChatMessage(role='user', content='Say hello from UPL in 5 words!')
        ],
        temperature=0.7,
        max_tokens=50
    )
    
    print("💬 Testing providers...")
    for name in ['groq', 'cerebras', 'openai']:
        provider = registry.get_provider(name)
        if not provider:
            print(f"   ⏭️  {name}: Not configured")
            continue
        
        health = health_checks.get(name)
        if not health or not health.healthy:
            print(f"   ⏭️  {name}: Unhealthy")
            continue
        
        try:
            request.model = provider.get_default_model()
            response = await provider.chat(request)
            print(f"   ✅ {name}: {response.content}")
        except Exception as e:
            print(f"   ❌ {name}: {str(e)[:50]}")
    
    print()
    
    # Demonstrate fallback chain
    print("🔗 Creating fallback chain...")
    chain = await registry.create_fallback_chain(
        primary_providers=['groq', 'cerebras'],
        include_paid_fallback=True
    )
    
    print(f"   Chain: {' → '.join([p.name for p in chain])}")
    print()
    
    print("=" * 80)
    print("✅ UPL MULTI-PROVIDER EXAMPLE COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    asyncio.run(main())
