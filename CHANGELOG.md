# Changelog

All notable changes to the Universal Provider Layer (UPL) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.2.0] - 2025-11-22

### Added

#### Dynamic Model Fetching System ⭐
- **Claude Provider**: Dynamic model fetching from Anthropic's `/models` API endpoint
  - Automatically discovers latest Claude models
  - Falls back to hardcoded list if API fails
  - File: `providers/paid/claude.py`
  
- **Moonshot Provider**: Dynamic model fetching from OpenAI-compatible `/models` endpoint
  - Automatically discovers latest Kimi K2 models
  - Falls back to hardcoded list if API fails
  - File: `providers/paid/moonshot.py`

#### Extended Reasoning Support ⭐
- **Thinking Mode**: Added support for Claude 4.x extended reasoning
  - New `supports_thinking` field in `ModelInfo` type
  - New `thinking_mode` parameter in `ChatRequest` type
  - Automatic detection for Claude Sonnet 4.x, Opus 4.x, and Haiku 4.x models
  - Files: `providers/types.py`, `providers/paid/claude.py`

#### Prompt Caching Support ⭐
- **Claude Provider**: Full prompt caching implementation
  - Support for project file context caching
  - System message blocks with cache control
  - 81% cost reduction on repeated context (90% cache discount)
  - Compatible with `metadata.project_context` in `ChatRequest`
  - File: `providers/paid/claude.py`

- **Universal Project Context**: All providers support project file injection
  - New `metadata` field in `ChatRequest` for additional context
  - Providers inject project files into system messages
  - Files: `providers/paid/claude.py`, `providers/paid/moonshot.py`, `providers/paid/openai.py`

### Changed

#### Updated Model Lists
- **Claude Provider**: Updated to November 2025 model lineup
  - Added Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`) - Recommended
  - Added Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) - Fastest & cheapest
  - Added Claude Opus 4.1 (`claude-opus-4-1-20250805`) - Most powerful
  - Added Claude Sonnet 4 (`claude-sonnet-4-20250514`)
  - Added Claude Opus 4 (`claude-opus-4-20250514`)
  - Added Claude Sonnet 3.7 (`claude-3-7-sonnet-20250219`)
  - Added Claude Haiku 3.5 (`claude-3-5-haiku-20241022`)
  - Kept Claude Haiku 3 (`claude-3-haiku-20240307`)
  - All models: 200K context window
  - **New Default**: `claude-sonnet-4-5-20250929`

- **Moonshot Provider**: Updated to K2 model lineup
  - Added **Kimi K2 Thinking** (`kimi-k2-thinking`) - 1M context! ⭐
  - Added Kimi K2 Turbo (`kimi-k2-turbo-preview`) - 256K context
  - Added Kimi K2 0905 (`kimi-k2-0905-preview`) - 256K context
  - Added Kimi K2 0711 (`kimi-k2-0711-preview`) - 128K context
  - Kept Moonshot V1 models (128K/32K/8K)
  - Fixed base URL to international: `https://api.moonshot.ai/v1`
  - **New Default**: `kimi-k2-thinking`

#### Model Count
- **Total Models**: 211 → **228+ models** (17+ new models added)
- **Providers**: Still 12 providers (6 free cloud + 3 local + 3 paid)

### Fixed
- **Moonshot Provider**: Corrected base URL from `.cn` to `.ai` for international access
- **Claude Provider**: Added proper thinking mode temperature handling (requires 1.0)

---

## [1.0.0] - 2025-11-10

### Initial Release

#### Features
- **12 LLM Providers**: 6 free cloud, 3 local, 3 paid
- **211 Models**: Across 11 working providers
- **30,300+ Free Requests/Day**: From free cloud providers
- **Framework Agnostic**: Works with any Python framework
- **Smart Fallback Chains**: Auto-failover on errors
- **Zero Vendor Lock-in**: Switch providers anytime

#### Providers
**Free Cloud (6):**
- Groq - 20 models, 14,400 req/day
- Cerebras - 7 models, 14,400 req/day
- Gemini - 41 models, 1,500 req/day
- OpenRouter - 47 models, free tier
- GitHub Models - 42 models, free tier
- HuggingFace - 4 models, free inference

**Paid (3):**
- OpenAI - GPT-4, GPT-3.5
- Claude - Anthropic models
- Moonshot - Kimi models

**Local (3):**
- Ollama - Unlimited usage
- LM Studio - Unlimited usage
- llama.cpp - Unlimited usage

#### Core Components
- **Base Provider Interface**: Standard interface for all providers
- **Provider Registry**: Centralized provider management
- **Universal Types**: Shared types across all providers
- **Health Check System**: Provider availability monitoring
- **Streaming Support**: Real-time response streaming
- **Rate Limiting**: Request throttling support

---

## Version Format
- **Major.Minor.Patch** (e.g., 1.2.0)
- **Major**: Breaking changes to API
- **Minor**: New features (backward compatible)
- **Patch**: Bug fixes only

---

**Legend:**
- ⭐ = Major new feature
- 🔧 = Enhancement
- 🐛 = Bug fix
- 📝 = Documentation
- ⚡ = Performance improvement
