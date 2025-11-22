# Supported Providers

UPL supports **228+ AI models** across **12 providers**. Here's the complete breakdown:

---

## Provider Overview

| Category | Providers | Models | Free Tier |
|----------|-----------|--------|-----------|
| **Free Cloud** | 6 | 161 | 30,300+ req/day |
| **Paid** | 3 | 67+ | Pay per use |
| **Local** | 3 | Unlimited | Free (self-hosted) |
| **Total** | 12 | 228+ | - |

---

## Free Cloud Providers

### Groq
**20 models • 14,400 requests/day • 500-1000 tokens/sec**

The speed demon of free providers. Exceptional inference speed on open-source models.

**Models:**
- Llama 3.3 70B Versatile
- Llama 3.2 90B Vision Preview
- Llama 3.2 11B Vision Preview
- Llama 3.2 3B Preview
- Llama 3.2 1B Preview
- Llama 3.1 70B Versatile
- Llama 3.1 8B Instant
- Mixtral 8x7B
- Gemma 2 9B
- And more...

**Rate Limits:**
- 14,400 requests/day (free tier)
- 30 requests/minute
- Ultra-fast inference

**Best For:** Speed-critical applications, real-time chat, rapid prototyping

---

### Cerebras
**7 models • 14,400 requests/day • Very fast**

Fast inference on Llama models with generous free tier.

**Models:**
- Llama 3.3 70B
- Llama 3.1 70B
- Llama 3.1 8B
- And more...

**Rate Limits:**
- 14,400 requests/day (free tier)
- Fast inference speed

**Best For:** High-volume free usage, Llama model preference

---

### Google Gemini
**41 models • 1,500 requests/day • Free tier**

Google's multimodal AI models with strong reasoning capabilities.

**Models:**
- Gemini 2.0 Flash Experimental
- Gemini 1.5 Pro
- Gemini 1.5 Flash
- Gemini 1.0 Pro
- And 37 more variants...

**Rate Limits:**
- 1,500 requests/day (free tier)
- 15 requests/minute

**Best For:** Multimodal tasks, vision + text, Google ecosystem integration

---

### OpenRouter
**47 models • Free tier available**

Aggregator providing access to multiple providers through one API.

**Models:**
- Access to OpenAI, Anthropic, Meta, Google, and more
- Models from 10+ different providers
- Constantly updated with new releases

**Rate Limits:**
- Free tier available
- Varies by model

**Best For:** Model variety, trying different providers, unified billing

---

### GitHub Models
**42 models • Free tier available**

GitHub's model marketplace with access to leading AI models.

**Models:**
- GPT-4o, GPT-4o Mini
- Claude 3.5 Sonnet
- Llama models
- Mistral models
- And more...

**Rate Limits:**
- Free tier for GitHub users
- Rate limits vary by model

**Best For:** GitHub integration, development workflows

---

### HuggingFace
**4 models • Free inference API**

Community-driven AI platform with free inference on select models.

**Models:**
- Meta-Llama models
- Mistral models
- Community fine-tunes

**Rate Limits:**
- Free inference API
- Rate limits apply

**Best For:** Community models, experimentation, research

---

## Paid Providers

### OpenAI
**Pay per use • Industry leader**

The gold standard for commercial AI applications.

**Models:**
- **GPT-4o** - Most capable, multimodal
- **GPT-4o Mini** - Fast and affordable
- **GPT-4 Turbo** - Previous generation flagship
- **GPT-3.5 Turbo** - Fast and cheap
- **DALL-E 3** - Image generation
- **Whisper** - Speech recognition

**Context Windows:**
- GPT-4o: 128K tokens
- GPT-3.5: 16K tokens

**Pricing:**
- GPT-4o: $2.50 per 1M input tokens
- GPT-4o Mini: $0.15 per 1M input tokens
- GPT-3.5 Turbo: $0.50 per 1M input tokens

**Best For:** Production applications, highest quality responses, multimodal tasks

---

### Anthropic Claude
**Pay per use • Dynamic model fetching • Prompt caching**

Best-in-class for code, reasoning, and long context. UPL supports full prompt caching.

**Models (November 2025):**
- **Claude Sonnet 4.5** - Recommended for code (200K context)
- **Claude Haiku 4.5** - Fastest & cheapest (200K context)
- **Claude Opus 4.1** - Most powerful (200K context)
- **Claude Sonnet 4** - Previous flagship (200K context)
- **Claude Opus 4** - Previous most powerful (200K context)
- **Claude Sonnet 3.7** - Legacy (200K context)
- **Claude Haiku 3.5** - Legacy fast model (200K context)
- **Claude Haiku 3** - Most affordable legacy (200K context)

**Special Features:**
- ⭐ **Dynamic model fetching** - Auto-discovers new models
- ⭐ **Prompt caching** - 81% cost reduction on repeated context
- ⭐ **Extended reasoning** - Thinking mode for Claude 4.x (10K token budget)
- All models: 200K context window

**Pricing (with caching):**
- First request: Full price
- Cached context: 90% discount
- Example: $3.00 → $0.57 for 10 requests (81% savings)

**Best For:** Code generation, complex reasoning, large codebases, cost optimization

---

### Moonshot (Kimi)
**Pay per use • Dynamic model fetching • 1M context**

Chinese provider with massive context windows. Perfect for entire codebases.

**Models (K2 Lineup):**
- **Kimi K2 Thinking** - 1M context! Long-context reasoning
- **Kimi K2 Turbo** - 256K context, fast inference
- **Kimi K2 0905** - 256K context
- **Kimi K2 0711** - 128K context
- Moonshot V1 models (128K/32K/8K)

**Special Features:**
- ⭐ **1 million token context** (K2 Thinking model)
- ⭐ **Dynamic model fetching** - Auto-discovers new models
- 5x larger context than Claude (200K)
- 8x larger context than GPT-4 (128K)

**Pricing:**
- Competitive rates
- International access via api.moonshot.ai

**Best For:** Massive context requirements, entire repositories, long documents

---

## Local Providers

### Ollama
**Unlimited usage • Self-hosted • Easy setup**

Most popular local LLM runtime. Download and run any model locally.

**Supported Models:**
- Any model from Ollama library
- Llama 3.3, 3.2, 3.1
- Mistral, Mixtral
- Gemma, Phi, Qwen
- Custom fine-tuned models

**Setup:**
- Install Ollama: https://ollama.ai
- Pull models: `ollama pull llama3.3`
- Runs on: localhost:11434

**Hardware Requirements:**
- 8GB RAM minimum
- 16GB+ recommended
- GPU optional (faster inference)

**Best For:** Privacy, offline usage, unlimited requests, custom models

---

### LM Studio
**Unlimited usage • Self-hosted • GUI interface**

User-friendly desktop app for running local LLMs with a nice UI.

**Supported Models:**
- HuggingFace models
- GGUF format models
- Llama, Mistral, Phi, and more

**Setup:**
- Install LM Studio: https://lmstudio.ai
- Download models via GUI
- Runs on: localhost:1234

**Features:**
- Easy GUI for model management
- GPU acceleration support
- Chat interface included

**Best For:** Non-technical users, GUI preference, model experimentation

---

### llama.cpp
**Unlimited usage • Self-hosted • Maximum performance**

Raw C++ implementation for maximum inference speed. For advanced users.

**Supported Models:**
- Any GGUF format model
- Quantized models for efficiency

**Setup:**
- Build from source: https://github.com/ggerganov/llama.cpp
- Run server: `./server -m model.gguf`
- Runs on: localhost:8080

**Features:**
- Fastest inference (C++ native)
- Lowest memory usage (quantization)
- Metal/CUDA/OpenCL support

**Best For:** Maximum performance, minimal resources, advanced users

---

## Provider Selection Guide

### Need Speed?
1. **Groq** (500-1000 T/s, free)
2. **Cerebras** (very fast, free)
3. **llama.cpp** (local, optimized)

### Need Cost Savings?
1. **Groq** (14,400 free req/day)
2. **Cerebras** (14,400 free req/day)
3. **Claude with caching** (81% savings)

### Need Quality?
1. **Claude Sonnet 4.5** (best for code)
2. **Claude Opus 4.1** (most powerful)
3. **GPT-4o** (multimodal leader)

### Need Privacy?
1. **Ollama** (fully local)
2. **LM Studio** (fully local)
3. **llama.cpp** (fully local)

### Need Variety?
1. **OpenRouter** (47 models)
2. **Gemini** (41 models)
3. **GitHub Models** (42 models)

### Need Massive Context?
1. **Moonshot K2 Thinking** (1M tokens!)
2. **Claude** (200K tokens)
3. **GPT-4o** (128K tokens)

---

## Dynamic Model Fetching

UPL automatically fetches the latest models from provider APIs where supported:

**Supported:**
- ✅ **Claude** - Fetches from Anthropic `/models` endpoint
- ✅ **Moonshot** - Fetches from OpenAI-compatible endpoint

**Coming Soon:**
- ⏳ OpenAI, Gemini, others as APIs become available

**Fallback:**
- All providers have hardcoded lists as backup
- Models still work if API fetch fails

This means **new models appear automatically** - no code changes needed!

---

## Testing Provider Availability

```python
from providers import registry

# Check if a provider is available
await registry.initialize_provider('groq', config)
provider = registry.get_provider('groq')

health = await provider.health_check()
if health['status'] == 'healthy':
    print(f"✅ {health['provider']} is available")
    print(f"   Models: {health['models_available']}")
else:
    print(f"❌ {health['provider']} is unavailable")
```

---

## Adding Your Own Provider

UPL's architecture makes adding providers straightforward:

1. Inherit from `BaseProvider`
2. Implement required methods
3. Add to provider registry
4. Done!

See existing providers in `providers/cloud/`, `providers/paid/`, or `providers/local/` for examples.

**Note:** We don't accept PRs for new providers, but forking and extending is encouraged!

---

**Last Updated:** November 22, 2025  
**Total Models:** 228+  
**Total Providers:** 12
