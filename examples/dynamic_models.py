"""
Example: Dynamic Model Fetching
Purpose: Showcase auto-fetching models from provider APIs
Provider: Claude or Moonshot (both support dynamic fetching)

UPL automatically fetches the latest models from provider APIs, ensuring
you always have access to new releases without manual updates.
"""

import asyncio
import os
from dotenv import load_dotenv
from providers import registry, ProviderConfig, ProviderType


async def main():
    """Fetch and display current models from Claude API"""
    
    load_dotenv()
    
    # Initialize Claude provider
    config = ProviderConfig(
        name='claude',
        type=ProviderType.PAID,
        api_key=os.getenv('ANTHROPIC_API_KEY')
    )
    
    await registry.initialize({'claude': config})
    provider = registry.get_provider('claude')
    
    print("🔄 Fetching latest models from Anthropic API...\n")
    
    # This makes a live API call to /v1/models
    models = await provider.list_models()
    
    print(f"Found {len(models)} models:\n")
    
    for model in models[:5]:  # Show top 5
        # Display model information
        thinking = "🧠 Thinking" if model.supports_thinking else ""
        vision = "👁️ Vision" if model.supports_vision else ""
        features = f"{thinking} {vision}".strip()
        
        print(f"  • {model.name}")
        print(f"    ID: {model.id}")
        print(f"    Context: {model.context_window:,} tokens")
        print(f"    Features: {features or 'Standard'}")
        print(f"    {model.description}\n")
    
    print("💡 These models are fetched fresh from the API - no hardcoded lists!")
    print("   If Anthropic releases a new model tomorrow, UPL sees it automatically.")


if __name__ == '__main__':
    asyncio.run(main())
