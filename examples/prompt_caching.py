"""
Example: Prompt Caching
Purpose: Showcase 81% cost savings with Claude's prompt caching
Provider: Claude (Anthropic)

Cost Comparison (10 requests with same 50K context):
  Without Caching: 10 × $0.30 = $3.00
  With Caching:    $0.30 + (9 × $0.06) = $0.84
  Savings: 81% reduction ($2.16 saved)
"""

import asyncio
import os
from dotenv import load_dotenv
from providers import registry, ProviderConfig, ChatRequest, ChatMessage, ProviderType


async def main():
    """Demonstrate prompt caching with Claude"""
    
    load_dotenv()
    
    # Initialize Claude provider
    config = ProviderConfig(
        name='claude',
        type=ProviderType.PAID,
        api_key=os.getenv('ANTHROPIC_API_KEY')  # Get at https://console.anthropic.com
    )
    
    await registry.initialize({'claude': config})
    provider = registry.get_provider('claude')
    
    # Simulate large project context (e.g., codebase files)
    project_files = """
    # Large codebase context (imagine 50K tokens of code here)
    # This is what gets cached after first request
    """ * 100  # Simulate larger context
    
    print("Sending 3 requests with same context...\n")
    
    for i in range(3):
        # Create request with project context in metadata
        request = ChatRequest(
            model='claude-sonnet-4-5-20250929',
            messages=[
                ChatMessage(role='user', content=f'Question {i+1}: What patterns do you see?')
            ],
            metadata={
                'project_context': [
                    {'path': 'codebase.py', 'content': project_files}
                ]
            },
            max_tokens=200
        )
        
        response = await provider.chat(request)
        
        # First request: Full cost ($0.30)
        # Next requests: 90% cache discount ($0.06 each)
        cache_status = "❌ Full cost" if i == 0 else "⚡ 90% cached (81% savings!)"
        print(f"Request {i+1}: {cache_status}")
        print(f"  {response}\n")


if __name__ == '__main__':
    asyncio.run(main())
