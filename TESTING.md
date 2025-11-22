# AC3 Testing Guide

**Last Updated:** November 10, 2025

This guide covers testing all 12 providers in the AC3 Universal Provider Layer.

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Test Scripts Overview](#test-scripts-overview)
3. [Testing Free Cloud Providers](#testing-free-cloud-providers)
4. [Testing Paid Providers](#testing-paid-providers)
5. [Testing Local Providers](#testing-local-providers)
6. [Comprehensive Testing](#comprehensive-testing)
7. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Prerequisites

1. **Install Dependencies**
   ```bash
   # Windows
   INSTALL_DEPENDENCIES.bat
   
   # Linux/Mac
   pip install -r requirements.txt
   ```

2. **Configure API Keys**
   - Copy `.env.example` to `.env`
   - Add your API keys (see [API Keys Setup](#api-keys-setup))

3. **Run Tests**
   ```bash
   # Test all providers
   python scripts/test_all_12_providers.py
   
   # Or use batch file (Windows)
   RUN_ALL_PROVIDERS_TEST.bat
   ```

---

## 📂 Test Scripts Overview

| Script | Purpose | Providers Tested |
|--------|---------|------------------|
| `test_groq.py` | Test Groq | Groq |
| `test_cerebras.py` | Test Cerebras | Cerebras |
| `test_gemini.py` | Test Gemini | Gemini |
| `test_openai.py` | Test OpenAI | OpenAI |
| `test_all_providers.py` | Test free cloud | All 6 free cloud |
| `test_all_12_providers.py` | **Test everything** | All 12 providers |
| `list_models.py` | List available models | All configured |

---

## ☁️ Testing Free Cloud Providers

### Why Test Free Cloud First?

- ✅ **30,300+ free requests/day** - Generous quotas
- ✅ **161 models** - Wide variety
- ✅ **No cost** - Perfect for development
- ✅ **Fast** - Groq/Cerebras are extremely fast

### 1. Get API Keys

#### Groq (14,400 req/day, 500-1000 T/s)
1. Visit: https://console.groq.com
2. Sign up (free)
3. Go to API Keys section
4. Create new key
5. Add to `.env`: `GROQ_API_KEY=your_key_here`

#### Cerebras (14,400 req/day, very fast)
1. Visit: https://cloud.cerebras.ai
2. Sign up (free)
3. Go to API Keys section
4. Create new key
5. Add to `.env`: `CEREBRAS_API_KEY=your_key_here`

#### Gemini (1,500 req/day)
1. Visit: https://aistudio.google.com
2. Sign in with Google
3. Get API key
4. Add to `.env`: `GEMINI_API_KEY=your_key_here`

#### OpenRouter (Free models available)
1. Visit: https://openrouter.ai
2. Sign up
3. Get API key
4. Add to `.env`: `OPENROUTER_API_KEY=your_key_here`

#### GitHub Models (Free tier)
1. Visit: https://github.com/marketplace/models
2. Generate Personal Access Token
3. Add to `.env`: `GITHUB_TOKEN=your_token_here`

#### HuggingFace (Free inference)
1. Visit: https://huggingface.co/settings/tokens
2. Create new token
3. Add to `.env`: `HUGGINGFACE_TOKEN=your_token_here`

### 2. Test Individual Providers

```bash
# Test Groq (fastest)
python scripts/test_groq.py

# Expected output:
# ======================================================================
# TESTING GROQ PROVIDER
# ======================================================================
# ✅ Provider initialized: groq
#    Default model: llama-3.3-70b-versatile
# 
# 🏥 Running health check...
#    Healthy: True
#    Latency: 142ms
#    Models: 20
# 
# 📋 Fetching available models...
#    Found 20 models:
#       - llama-3.3-70b-versatile (ctx: 32,768)
#       - llama-3.1-70b-versatile (ctx: 128,000)
#       ... and 18 more
# 
# 💬 Testing chat completion...
# ✅ Groq Response:
#    Model: llama-3.3-70b-versatile
#    Latency: 234ms
#    Tokens: 45
#    Response: Hello from AC3!
# ======================================================================
```

```bash
# Test Cerebras
python scripts/test_cerebras.py

# Test Gemini
python scripts/test_gemini.py
```

### 3. Test All Free Cloud Providers

```bash
python scripts/test_all_providers.py

# Expected output:
# ======================================================================
# TESTING FREE CLOUD PROVIDERS (6 providers)
# ======================================================================
# 
# ✅ groq: 20 models (142ms)
# ✅ cerebras: 7 models (478ms)
# ✅ gemini: 41 models (1.7s)
# ✅ openrouter: 47 models (2.2s)
# ✅ github: 42 models (1.3s)
# ✅ huggingface: 4 models (1.9s)
# 
# Total: 161 models, 30,300+ free requests/day
# ======================================================================
```

---

## 💳 Testing Paid Providers

### When to Use Paid Providers?

- ✅ **Premium quality** - GPT-4, Claude Opus
- ✅ **Reliable** - Guaranteed uptime
- ✅ **Fallback** - When free providers hit limits
- ⚠️ **Cost** - Pay per token

### 1. OpenAI (Recommended)

#### Get API Key
1. Visit: https://platform.openai.com
2. Sign up
3. Add credits to account
4. Generate API key
5. Add to `.env`: `OPENAI_API_KEY=your_key_here`

#### Test OpenAI
```bash
python scripts/test_openai.py

# Or use batch file
RUN_OPENAI_TEST.bat

# Expected output:
# ======================================================================
# TESTING OPENAI PROVIDER
# ======================================================================
# ✅ Provider initialized: openai
#    Default model: gpt-4o-mini
# 
# 🏥 Running health check...
#    Healthy: True
#    Latency: 623ms
#    Models: 4
# 
# 📋 Fetching available models...
#    Found 4 models:
#       - gpt-4o (ctx: 128,000)
#       - gpt-4o-mini (ctx: 128,000)
#       - gpt-4-turbo (ctx: 128,000)
#       - gpt-3.5-turbo (ctx: 16,385)
# 
# 💬 Testing chat completion...
# ✅ OpenAI Response:
#    Model: gpt-4o-mini
#    Latency: 847ms
#    Tokens: 12
#    Response: Hello from AC3!
# ======================================================================
```

### 2. Claude (Anthropic)

#### Get API Key
1. Visit: https://console.anthropic.com
2. Sign up
3. Add credits
4. Generate API key
5. Add to `.env`: `ANTHROPIC_API_KEY=your_key_here`

#### Test Claude
```bash
# Currently tested via test_all_12_providers.py
python scripts/test_all_12_providers.py
```

### 3. Moonshot (Kimi)

#### Get API Key
1. Visit: https://platform.moonshot.cn
2. Sign up (may need Chinese account)
3. Generate API key
4. Add to `.env`: `MOONSHOT_API_KEY=your_key_here`

#### Test Moonshot
```bash
# Currently tested via test_all_12_providers.py
python scripts/test_all_12_providers.py
```

---

## 🏠 Testing Local Providers

### Why Use Local Providers?

- ✅ **Privacy** - Data never leaves your machine
- ✅ **Unlimited** - No rate limits or costs
- ✅ **Offline** - Works without internet
- ⚠️ **Setup required** - Need to install and run services

### 1. Ollama (Recommended for beginners)

#### Install Ollama
```bash
# Windows/Mac/Linux
# Download from: https://ollama.com
```

#### Download a Model
```bash
# Popular models
ollama pull qwen2.5-coder:7b    # Code-focused (4GB)
ollama pull llama3.2:latest     # General purpose (2GB)
ollama pull deepseek-r1:8b      # Reasoning (5GB)
```

#### Start Ollama Server
```bash
ollama serve
# Runs on http://localhost:11434
```

#### Test Ollama
```bash
# In another terminal
python scripts/test_ollama.py

# Or via comprehensive test
python scripts/test_all_12_providers.py
```

### 2. LM Studio (GUI-based)

#### Install LM Studio
1. Download from: https://lmstudio.ai
2. Install application
3. Download a model via GUI (e.g., Llama 3.2, Qwen 2.5)
4. Click "Start Server" button
5. Server runs on http://localhost:1234

#### Test LM Studio
```bash
python scripts/test_all_12_providers.py
```

### 3. llama.cpp (Advanced)

#### Install llama.cpp
```bash
# Clone repo
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build
cmake -B build
cmake --build build --config Release

# Download a GGUF model
# e.g., from https://huggingface.co/models?search=gguf
```

#### Start Server
```bash
./build/bin/server -m path/to/model.gguf --port 8080
```

#### Test llama.cpp
```bash
python scripts/test_all_12_providers.py
```

---

## 🎯 Comprehensive Testing

### Test All 12 Providers at Once

```bash
python scripts/test_all_12_providers.py

# Or use batch file
RUN_ALL_PROVIDERS_TEST.bat
```

### Expected Output

```
================================================================================
AC3 UNIVERSAL PROVIDER LAYER - COMPREHENSIVE TEST
Testing ALL 12 Providers: 6 Free Cloud + 3 Local + 3 Paid
================================================================================

================================================================================
PART 1: FREE CLOUD PROVIDERS (6 providers)
================================================================================

🔑 Checking API keys...
   ✅ groq            - Key found (14,400 req/day)
   ✅ cerebras        - Key found (14,400 req/day)
   ✅ gemini          - Key found (1,500 req/day)
   ✅ openrouter      - Key found (Free models available)
   ✅ github          - Key found (Free tier)
   ✅ huggingface     - Key found (Free inference)

================================================================================
PART 2: PAID PROVIDERS (3 providers)
================================================================================

🔑 Checking API keys...
   ✅ openai          - Key found (Pay per use)
   ✅ claude          - Key found (Pay per use)
   ✅ moonshot        - Key found (Pay per use)

================================================================================
PART 3: LOCAL PROVIDERS (3 providers)
================================================================================

🔌 Checking local services...
   ⏳ ollama          - Trying Port 11434...
   ⏳ lmstudio        - Trying Port 1234...
   ⏳ llamacpp        - Trying Port 8080...

================================================================================
INITIALIZING PROVIDERS & RUNNING HEALTH CHECKS
================================================================================

📦 Initializing 12 providers...

🏥 Running health checks on all providers...

   ✅ groq            -  20 models (142ms)
   ✅ cerebras        -   7 models (478ms)
   ✅ gemini          -  41 models (1734ms)
   ✅ openrouter      -  47 models (2156ms)
   ✅ github          -  42 models (1345ms)
   ✅ huggingface     -   4 models (1923ms)
   ✅ openai          -   4 models (623ms)
   ✅ claude          -   3 models (891ms)
   ✅ moonshot        -   3 models (734ms)
   ❌ ollama          - Connection refused (service not running)
   ❌ lmstudio        - Connection refused (service not running)
   ❌ llamacpp        - Connection refused (service not running)

================================================================================
TESTING CHAT COMPLETION (Sample from each category)
================================================================================

💬 Testing Free Cloud Provider: groq
   ✅ Success!
   Model: llama-3.3-70b-versatile
   Latency: 234.5ms
   Tokens: 45
   Response: Hello from AC3!

💬 Testing Paid Provider: openai
   ✅ Success!
   Model: gpt-4o-mini
   Latency: 847.2ms
   Tokens: 12
   Response: Hello from AC3!

⏭️  Local (ollama): Unhealthy/unavailable

================================================================================
COMPREHENSIVE SUMMARY
================================================================================

📊 FREE CLOUD PROVIDERS:
   Status: 6/6 healthy
   Models: 161 available
   Quota: 30,300 confirmed free requests/day

💳 PAID PROVIDERS:
   Status: 3/3 healthy
   Models: 10 available

🏠 LOCAL PROVIDERS:
   Status: 0/3 running
   Note: No local services detected (install Ollama/LM Studio/llama.cpp)

================================================================================
OVERALL RESULTS
================================================================================
✅ Healthy Providers: 9/12
📚 Total Models: 171
🆓 Free Cloud: 6/6
💳 Paid: 3/3
🏠 Local: 0/3

🎯 AC3 Universal Provider Layer: OPERATIONAL
   You have access to 171 models across 9 providers!
================================================================================
```

---

## 🐛 Troubleshooting

### API Key Issues

#### "No API key configured"
```bash
# Check .env file exists
ls .env

# Check key is set (don't share output!)
cat .env | grep API_KEY

# Regenerate .env from template
cp .env.example .env
# Then add your keys
```

#### "Invalid API key"
- Verify key is correct (no extra spaces/newlines)
- Check key hasn't expired
- Verify account has credits (for paid providers)
- Regenerate key from provider dashboard

### Connection Issues

#### "Connection refused" (Local providers)
```bash
# For Ollama
ollama serve

# For LM Studio
# Open LM Studio app → Click "Start Server"

# For llama.cpp
cd llama.cpp
./build/bin/server -m path/to/model.gguf
```

#### "Timeout" errors
```bash
# Increase timeout in .env
PROVIDER_TIMEOUT=60  # Default is 30s
```

### Rate Limit Issues

#### "Rate limit exceeded"
```bash
# Use fallback chain
python scripts/test_all_12_providers.py

# The system will try next provider automatically
```

#### Check remaining quota
```bash
# Most providers have dashboards showing usage
# Groq: https://console.groq.com/usage
# Cerebras: https://cloud.cerebras.ai/usage
# etc.
```

### Model Issues

#### "Model not found"
```bash
# List available models
python scripts/list_models.py

# Check provider's model list
python -c "
from providers import registry
models = await registry.get_provider('groq').list_models()
for m in models:
    print(m.id)
"
```

#### Local model not loaded (Ollama)
```bash
# Download model first
ollama pull qwen2.5-coder:7b

# List downloaded models
ollama list
```

### Import Errors

#### "Module not found"
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python version
python --version
# Need Python 3.11+
```

#### "Cannot import name X"
```bash
# Check you're in correct directory
cd O:\AethyrionCoder\AC3

# Add to Python path
export PYTHONPATH="${PYTHONPATH}:."
```

---

## 📊 Test Results Tracking

### Create Test Log

```bash
# Run and save results
python scripts/test_all_12_providers.py > test_results_$(date +%Y%m%d).txt

# Compare with previous
diff test_results_20251110.txt test_results_20251111.txt
```

### Performance Benchmarks

Expected latencies:
- **Groq**: 100-300ms (⚡ fastest)
- **Cerebras**: 300-600ms (⚡ very fast)
- **Gemini**: 1-3s (🟢 good)
- **OpenRouter**: 1-3s (🟡 varies by model)
- **GitHub**: 1-2s (🟢 good)
- **HuggingFace**: 2-5s (🟡 slower, free)
- **OpenAI**: 500-1500ms (🟢 good)
- **Claude**: 800-2000ms (🟢 good)
- **Moonshot**: 500-1500ms (🟢 good)
- **Ollama**: 100-2000ms (depends on local GPU)
- **LM Studio**: 200-3000ms (depends on local GPU)
- **llama.cpp**: 500-5000ms (depends on CPU/GPU)

---

## ✅ Test Checklist

Before considering testing complete:

### Free Cloud Providers
- [ ] Groq: Health check passes
- [ ] Groq: Can list models
- [ ] Groq: Chat completion works
- [ ] Cerebras: Health check passes
- [ ] Cerebras: Can list models
- [ ] Cerebras: Chat completion works
- [ ] Gemini: Health check passes
- [ ] Gemini: Can list models
- [ ] Gemini: Chat completion works
- [ ] OpenRouter: Health check passes
- [ ] OpenRouter: Can list models
- [ ] OpenRouter: Chat completion works
- [ ] GitHub: Health check passes
- [ ] GitHub: Can list models
- [ ] GitHub: Chat completion works
- [ ] HuggingFace: Health check passes
- [ ] HuggingFace: Can list models
- [ ] HuggingFace: Chat completion works

### Paid Providers
- [ ] OpenAI: Health check passes
- [ ] OpenAI: Can list models
- [ ] OpenAI: Chat completion works
- [ ] Claude: Health check passes (optional)
- [ ] Claude: Can list models (optional)
- [ ] Claude: Chat completion works (optional)
- [ ] Moonshot: Health check passes (optional)
- [ ] Moonshot: Can list models (optional)
- [ ] Moonshot: Chat completion works (optional)

### Local Providers (Optional)
- [ ] Ollama: Installed and running
- [ ] Ollama: Model downloaded
- [ ] Ollama: Health check passes
- [ ] Ollama: Chat completion works
- [ ] LM Studio: Installed (optional)
- [ ] llama.cpp: Installed (optional)

### System Tests
- [ ] All 12 providers initialize without errors
- [ ] Health checks complete in < 30s
- [ ] Fallback chains work
- [ ] Streaming works
- [ ] Error handling works

---

## 🎓 Best Practices

### Development Testing
1. **Start with free cloud** - Get 6 providers working first
2. **Add OpenAI** - Most reliable paid fallback
3. **Add local** - For privacy/unlimited usage
4. **Test fallback** - Ensure auto-failover works

### Production Testing
1. **Health checks** - Run before each session
2. **Monitor quotas** - Track free tier usage
3. **Load testing** - Test under concurrent load
4. **Error scenarios** - Test with invalid keys, rate limits

### CI/CD Testing
```yaml
# Example GitHub Actions
- name: Test AC3 Providers
  run: |
    pip install -r requirements.txt
    python scripts/test_all_12_providers.py
  env:
    GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
    # ... other keys
```

---

## 📚 Additional Resources

- **README.md** - Project overview and usage examples
- **STATUS.md** - Current implementation status
- **CURRENT_INSTRUCTIONS.md** - Project principles and roadmap
- **Provider docs** - See links in README.md

---

**Last Updated:** November 10, 2025  
**Status:** 🟢 All test scripts complete  
**Next:** Run tests, verify all providers work
