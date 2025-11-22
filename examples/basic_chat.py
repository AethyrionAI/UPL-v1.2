"""
Example: Basic Chat
Purpose: Getting started - simplest possible UPL usage
Provider: Groq (free tier)
"""

import asyncio
import os
from dotenv import load_dotenv
from providers import registry, ProviderConfig, ChatRequest, ChatMessage, ProviderType


async def main():
    """Send a simple chat request using Groq (free tier)"""
    
    # Load API keys from .env file
    load_dotenv()
    
    # Initialize Groq provider (free tier, no credit card required!)
    config = ProviderConfig(
        name='groq',
        type=ProviderType.CLOUD_FREE,
        api_key=os.getenv('GROQ_API_KEY')  # Get free key at https://console.groq.com
    )
    
    # Initialize registry with our provider
    await registry.initialize({'groq': config})
    
    # Get the provider instance
    provider = registry.get_provider('groq')
    
    # Create a chat request
    request = ChatRequest(
        model=provider.get_default_model(),  # Uses recommended model
        messages=[
            ChatMessage(role='user', content='Explain UPL in one sentence')
        ],
        temperature=0.7,
        max_tokens=100
    )
    
    # Send the request and get response
    response = await provider.chat(request)
    
    # Print the response
    print(f"\n{response.content}")
    print(f"\n{response}")  # Shows provider/model/tokens/latency


if __name__ == '__main__':
    asyncio.run(main())
