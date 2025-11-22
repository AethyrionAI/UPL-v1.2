# 🌐 UPL - Universal Provider Layer

**One API. 228+ Models. Zero Lock-in.**

Switch between OpenAI, Anthropic, Groq, Cerebras, Gemini, and 7 more providers with a single line of code. No rewrites. No vendor lock-in. Ever.

```python
# Same code, different provider - that's it.
from providers import registry, ProviderConfig, ChatRequest, ChatMessage

config = ProviderConfig(name='claude', api_key='your_key_here')
await registry.initialize_provider('claude', config)
provider = registry.get_provider('claude')

request = ChatRequest(
    model='claude-sonnet-4-5-20250929',
    messages=[ChatMessage(role='user', content='Explain async/await')]
)

response = await provider.chat(request)
print(response.content)
```

---

## 🚨 Status: Internal Infrastructure (Public for Learning)

UPL powers [Nova](https://github.com/AethyrionAI/Nova), [Helix](https://github.com/AethyrionAI/Helix), and [Forge](https://github.com/AethyrionAI/Forge). We're publishing it to **pay forward the inspiration** from [kosong](https://github.com/MoonshotAI/kosong).

**✅ Use it. Fork it. Learn from it.**  
**❌ Don't expect support.** We update UPL when our tools need it, not for external use cases.

---

## ⚡ What Makes UPL Different

### 🔄 Dynamic Model Discovery
Providers auto-fetch the latest models from their APIs. No hardcoded lists. No manual updates.

```python
models = await provider.list_models()  # Always current
# Claude Sonnet 4.6 drops tomorrow? You'll have it automatically.
```

### 💰 81% Cost Reduction (Prompt Caching)
Cache project context across requests. First request pays full price, next 9 are 81% cheaper.

```python
request = ChatRequest(
    model='claude-sonnet-4-5-20250929',
    messages=[ChatMessage(role='user', content='Review this code')],
    metadata={'project_context': project_files}  # Auto-cached!
)
# First request: $3.00
# Next 9 requests: $0.57 total
# Savings: $24.30 per 100 requests
```

### 🧠 Extended Reasoning (Claude 4.x)
10,000 token thinking budget for complex problems. Better code analysis. Smarter responses.

```python
request = ChatRequest(
    model='claude-sonnet-4-5-20250929',
    messages=[ChatMessage(role='user', content='Design a distributed system')],
    thinking_mode=True  # Claude thinks before responding
)
```

### 📚 1M Context Window (Moonshot)
Kimi K2 Thinking handles **1 million tokens**. That's 5x Claude, 8x GPT-4. Entire codebases in one prompt.

```python
config = ProviderConfig(name='moonshot', api_key='your_key_here')
await registry.initialize_provider('moonshot', config)
provider = registry.get_provider('moonshot')

# Feed it your entire monorepo - it can handle it
```

---

## 🎯 What It Powers

| Tool | Use Case | UPL Features |
|------|----------|--------------|
| **[Nova](https://github.com/AethyrionAI/Nova)** | AI coding assistant | 228+ models, prompt caching, thinking mode |
| **[Helix](https://github.com/AethyrionAI/Helix)** | React component generator | Multi-provider support, auto-failover |
| **[Forge](https://github.com/AethyrionAI/Forge)** | Game development AI | 1M context, extended reasoning |
| **Prism** | ServiceNow assistant | Custom model routing |

---

## 📊 The Numbers

- **228+ Models** across 12 providers
- **30,300+ Free Requests/Day** from 6 free cloud providers
- **81% Cost Savings** with prompt caching
- **1M Token Context** with Moonshot K2
- **Zero Vendor Lock-in** - switch providers anytime

---

## 🚀 Quick Start

### Installation

```bash
# Clone this repo
git clone https://github.com/AethyrionAI/UPL.git
cd UPL

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys
```

### Basic Usage

```python
import asyncio
from providers import registry, ProviderConfig, ChatRequest, ChatMessage

async def main():
    # Initialize provider
    config = ProviderConfig(
        name='groq',
        api_key='your_groq_api_key_here'
    )
    
    await registry.initialize_provider('groq', config)
    provider = registry.get_provider('groq')
    
    # Send request
    request = ChatRequest(
        model='llama-3.3-70b-versatile',
        messages=[ChatMessage(role='user', content='Hello!')]
    )
    
    response = await provider.chat(request)
    print(response.content)

if __name__ == "__main__":
    asyncio.run(main())
```

See `/examples` for advanced usage (caching, streaming, thinking mode).

---

## 🔌 All 12 Providers

### Free Cloud (30,300+ req/day)
- **Groq** - 20 models, 14,400 req/day, 500-1000 T/s
- **Cerebras** - 7 models, 14,400 req/day, blazing fast
- **Gemini** - 41 models, 1,500 req/day
- **OpenRouter** - 47 models, free tier
- **GitHub Models** - 42 models, free tier
- **HuggingFace** - 4 models, free inference

### Paid (Latest Models)
- **OpenAI** - GPT-4, GPT-3.5, DALL-E
- **Claude** - **Prompt caching**, thinking mode, 200K context
- **Moonshot** - **1M context**, Kimi K2 lineup

### Local (Unlimited)
- **Ollama** - Run any model locally
- **LM Studio** - Custom fine-tuned models
- **llama.cpp** - Ultra-fast local inference

[Full provider list →](docs/PROVIDERS.md)

---

## 💡 Why UPL Exists

### The Problem
Every AI provider has a different API. Switching providers means rewriting your entire integration.

```python
# Without UPL: Rewrite everything
openai_client = OpenAI(api_key=...)
anthropic_client = Anthropic(api_key=...)
groq_client = Groq(api_key=...)
# 3 different APIs, 3 different response formats, 3 different error handlers
```

### The Solution
UPL abstracts provider differences into one unified interface.

```python
# With UPL: Change one line
config = ProviderConfig(name='openai', api_key=...)  # or 'claude', or 'groq', or...
await registry.initialize_provider('openai', config)
provider = registry.get_provider('openai')

response = await provider.chat(request)
# Same code, any provider
```

---

## 📚 Documentation

- **[CHANGELOG.md](CHANGELOG.md)** - Version history and changes
- **[TESTING.md](TESTING.md)** - Comprehensive testing guide
- **[INSTALL.md](INSTALL.md)** - Installation instructions
- **[docs/PROVIDERS.md](docs/PROVIDERS.md)** - Full provider and model list
- **[docs/INSPIRATION.md](docs/INSPIRATION.md)** - The kosong origin story

---

## 🏗️ Architecture

```
UPL/
├── providers/
│   ├── __init__.py           # Public API
│   ├── types.py              # Universal types
│   ├── base.py               # Base provider interface
│   ├── registry.py           # Provider registry
│   │
│   ├── cloud/                # 6 free cloud providers
│   │   ├── groq.py
│   │   ├── cerebras.py
│   │   ├── gemini.py
│   │   ├── openrouter.py
│   │   ├── github.py
│   │   └── huggingface.py
│   │
│   ├── local/                # 3 local providers
│   │   ├── ollama.py
│   │   ├── lmstudio.py
│   │   └── llamacpp.py
│   │
│   └── paid/                 # 3 paid providers
│       ├── openai.py
│       ├── claude.py         # Dynamic fetch, caching, thinking
│       └── moonshot.py       # Dynamic fetch, 1M context
│
├── examples/                 # Usage examples
├── docs/                     # Documentation
├── requirements.txt
├── .env.example
└── README.md
```

---

## 📈 Version History

- **v1.2.0** (Nov 2025) - Dynamic models, prompt caching, thinking mode, 1M context
- **v1.0.0** (Oct 2025) - Initial (internal) release with 12 providers, 211 models

See [CHANGELOG.md](CHANGELOG.md) for detailed history.

---

## 🙏 The Origin Story

We discovered [Moonshot's kosong provider](https://github.com/MoonshotAI/kosong) and thought *"this pattern is brilliant, but we can take it further."*

Kosong showed us multi-provider abstraction. We added:
- Dynamic model fetching (no hardcoded lists)
- Prompt caching (81% cost savings)
- Extended reasoning (thinking mode)
- 1M context windows
- 228+ models across 12 providers

Then we built an entire ecosystem on top of it: Nova, Helix, Forge.

**Open source compounds.** Kosong inspired UPL. Maybe UPL will inspire you.

Full story: [docs/INSPIRATION.md](docs/INSPIRATION.md)

---

## 📄 License

MIT © 2025 Aethyrion

---

**Part of the [Aethyrion ecosystem](https://aethyrion.org)** - free developer tools, forever.
