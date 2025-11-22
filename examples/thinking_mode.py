"""
Example: Thinking Mode
Purpose: Showcase Claude 4.x extended reasoning
Provider: Claude (Anthropic)

Claude 4.x models can "think" before responding, using up to 10,000 tokens
of internal reasoning. This leads to better answers for complex problems.
"""

import asyncio
import os
from dotenv import load_dotenv
from providers import registry, ProviderConfig, ChatRequest, ChatMessage, ProviderType


async def main():
    """Demonstrate extended reasoning with Claude 4.x"""
    
    load_dotenv()
    
    # Initialize Claude provider
    config = ProviderConfig(
        name='claude',
        type=ProviderType.PAID,
        api_key=os.getenv('ANTHROPIC_API_KEY')
    )
    
    await registry.initialize({'claude': config})
    provider = registry.get_provider('claude')
    
    # Ask a complex question that benefits from "thinking"
    complex_question = """
    Design a distributed system for processing 1M transactions/second
    with <50ms latency, considering:
    - Fault tolerance
    - Geographic distribution
    - Cost optimization
    - Scalability to 10M TPS
    """
    
    print("🧠 Sending complex request with thinking mode enabled...\n")
    
    # Create request with thinking_mode=True
    request = ChatRequest(
        model='claude-sonnet-4-5-20250929',  # Works with any Claude 4.x model
        messages=[
            ChatMessage(role='user', content=complex_question)
        ],
        thinking_mode=True,  # ← Enable extended reasoning
        max_tokens=1000
    )
    
    response = await provider.chat(request)
    
    print(f"Response:\n{response.content}\n")
    print(f"{response}")
    print("\n💡 Claude used up to 10K tokens to think through this problem!")


if __name__ == '__main__':
    asyncio.run(main())
