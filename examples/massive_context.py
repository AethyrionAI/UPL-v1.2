"""
Example: Massive Context Windows
Purpose: Showcase Moonshot's 1M token context window
Provider: Moonshot (Kimi)

Context Window Comparison:
  • Moonshot K2 Thinking: 1,000,000 tokens (entire codebases!)
  • Claude Sonnet 4.5:      200,000 tokens (5x smaller)
  • GPT-4 Turbo:            128,000 tokens (8x smaller)
  • Most other models:       32,000 tokens (31x smaller)
"""

import asyncio
import os
from dotenv import load_dotenv
from providers import registry, ProviderConfig, ChatRequest, ChatMessage, ProviderType


async def main():
    """Demonstrate massive context with Moonshot"""
    
    load_dotenv()
    
    # Initialize Moonshot provider
    config = ProviderConfig(
        name='moonshot',
        type=ProviderType.PAID,
        api_key=os.getenv('KIMI_API_KEY')  # Or MOONSHOT_API_KEY
    )
    
    await registry.initialize({'moonshot': config})
    provider = registry.get_provider('moonshot')
    
    # Simulate a massive codebase (in real use, this could be hundreds of files)
    large_context = """
    # Imagine this is an entire repository:
    # - 100+ Python files
    # - Complete API documentation
    # - Database schemas
    # - Configuration files
    # All injected into context at once!
    """ * 1000  # Simulate very large context
    
    print("📚 Sending request with massive context...\n")
    print(f"Context size: {len(large_context):,} characters")
    print("(In a real scenario, this could be your entire codebase)\n")
    
    # Create request with large context
    request = ChatRequest(
        model='kimi-k2-thinking',  # 1M token context!
        messages=[
            ChatMessage(
                role='user', 
                content=f"{large_context}\n\nAnalyze this codebase and suggest improvements."
            )
        ],
        max_tokens=500
    )
    
    try:
        response = await provider.chat(request)
        print(f"Response:\n{response.content}\n")
        print(f"{response}")
        print("\n💡 This would overflow most other models!")
        print("   Moonshot's 1M context = fit entire repos in a single request")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    asyncio.run(main())
