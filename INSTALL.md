r'''# UPL Installation Guide

## Quick Install

```bash
# Navigate to UPL directory
cd O:\AethyrionCoder\UPL

# Install in development mode (recommended)
pip install -e .
```

## Verify Installation

```bash
python -c "from providers import registry; print('✅ UPL installed!')"
```

---

## Basic Usage

```python
import asyncio
import os
from dotenv import load_dotenv
from providers import registry, ProviderConfig, ChatRequest, ChatMessage, ProviderType

async def main():
    load_dotenv()
    
    # Initialize provider(s)
    configs = {
        'groq': ProviderConfig(
            name='groq',
            type=ProviderType.CLOUD_FREE,
            api_key=os.getenv('GROQ_API_KEY')
        )
    }
    
    await registry.initialize(configs)
    
    # Get provider
    provider = registry.get_provider('groq')
    
    # Send request
    request = ChatRequest(
        model=provider.get_default_model(),
        messages=[ChatMessage(role='user', content='Hello!')],
        temperature=0.7,
        max_tokens=100
    )
    
    response = await provider.chat(request)
    print(response.content)

asyncio.run(main())
```

---

## Usage in Other Projects

### AC1 (Custom Agent)
```python
# In AC1's requirements.txt
-e O:/AethyrionCoder/UPL

# In AC1 code
from providers import registry, ProviderConfig, ProviderType
import os

# Initialize
configs = {
    'groq': ProviderConfig(
        name='groq',
        type=ProviderType.CLOUD_FREE,
        api_key=os.getenv('GROQ_API_KEY')
    )
}
await registry.initialize(configs)
provider = registry.get_provider('groq')
```

### AC2 (Google ADK)
```python
# In AC2's requirements.txt
-e O:/AethyrionCoder/UPL

# In AC2 code - get AutoGen config
from providers import registry

await registry.initialize(configs)
config_list = await registry.get_autogen_config_list()
```

### AC3 (Microsoft AutoGen)
```python
# In AC3's requirements.txt
-e O:/AethyrionCoder/UPL

# In AC3 code
from providers import registry

await registry.initialize(configs)
config_list = await registry.get_autogen_config_list(
    providers=['groq', 'cerebras'],
    include_paid_fallback=True
)
```

### AC4 (PydanticAI)
```python
# In AC4's requirements.txt
-e O:/AethyrionCoder/UPL

# In AC4 code
from providers import registry

await registry.initialize(configs)
provider = registry.get_provider('groq')
config = provider.to_pydantic_config('llama-3.3-70b')
```

### AC5 (OpenCoder)
```python
# Install UPL in AC5's Python environment
pip install -e O:/AethyrionCoder/UPL

# Use in Python scripts called by TypeScript
from providers import registry
```

---

## Update UPL

```bash
cd O:\AethyrionCoder\UPL
git pull  # If using version control
# Changes automatically available to all projects using -e install
```

---

## Examples

### Run Basic Example
```bash
cd O:\AethyrionCoder\UPL
python examples\basic_usage.py
```

### Run Multi-Provider Example
```bash
cd O:\AethyrionCoder\UPL
python examples\multi_provider.py
```

---

## Common Patterns

### Initialize Multiple Providers
```python
configs = {
    'groq': ProviderConfig(name='groq', type=ProviderType.CLOUD_FREE, api_key=...),
    'openai': ProviderConfig(name='openai', type=ProviderType.PAID, api_key=...),
    'ollama': ProviderConfig(name='ollama', type=ProviderType.LOCAL, base_url='http://localhost:11434')
}
await registry.initialize(configs)
```

### Health Check All Providers
```python
health_checks = await registry.health_check_all()
for name, health in health_checks.items():
    if health.healthy:
        print(f"✅ {name}: {health.models_available} models")
```

### Create Fallback Chain
```python
chain = await registry.create_fallback_chain(
    primary_providers=['groq', 'cerebras'],
    include_paid_fallback=True
)

# Try providers in order
for provider in chain:
    try:
        response = await provider.chat(request)
        break  # Success!
    except Exception as e:
        continue  # Try next provider
```

### Get Best Available Provider
```python
provider = await registry.get_best_provider(
    prefer_free=True,
    prefer_local=False
)
```

---

## Troubleshooting

### Import Error
```bash
# Reinstall in editable mode
cd O:\AethyrionCoder\UPL
pip install -e .
```

### Missing Dependencies
```bash
cd O:\AethyrionCoder\UPL
pip install -r requirements.txt
```

### API Key Issues
```bash
# Check .env file exists
dir .env

# Copy from AC3 if needed
copy O:\AethyrionCoder\AC3\.env O:\AethyrionCoder\UPL\.env
```

---

**Last Updated:** November 10, 2025
'''