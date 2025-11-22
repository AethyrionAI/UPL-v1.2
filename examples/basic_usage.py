"""
Basic UPL usage example
"""

import asyncio
import os
from dotenv import load_dotenv
from providers import registry, ProviderConfig, ChatRequest, ChatMessage, ProviderType


async def main():
    load_dotenv()
    
    # Initialize Groq provider
    config = ProviderConfig(
        name='groq',
        type=ProviderType.CLOUD_FREE,
        api_key=os.getenv('GROQ_API_KEY')
    )
    
    # Initialize registry with configs dict
    await registry.initialize({
        'groq': config
    })
    
    # Get provider
    provider = registry.get_provider('groq')
    
    # Send chat request
    request = ChatRequest(
        model=provider.get_default_model(),
        messages=[
            ChatMessage(role='user', content='Hello from UPL!')
        ],
        temperature=0.7,
        max_tokens=100
    )
    
    response = await provider.chat(request)
    print(f"Response: {response.content}")


if __name__ == '__main__':
    asyncio.run(main())
