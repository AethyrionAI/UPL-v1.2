# The kosong Inspiration

## How We Discovered Multi-Provider Abstraction

In late 2025, while researching AI provider integrations for what would become Nova (our AI coding assistant), we stumbled across Moonshot's [kosong](https://github.com/MoonshotAI/kosong) library.

At the time, we were frustrated with vendor lock-in. Every AI provider had a different API:
- OpenAI used one format
- Anthropic used another
- Groq had their own approach
- Local providers like Ollama were completely different

Switching providers meant rewriting entire integrations. It was painful, time-consuming, and kept us locked into whatever provider we started with.

Then we found kosong.

## What kosong Showed Us

Kosong demonstrated something crucial: **you don't have to be locked into one AI provider**. You can abstract the differences and give users choice.

The core insight was elegant:
```python
# One interface, multiple providers
provider = get_provider('openai')  # or 'anthropic', or 'groq'
response = provider.chat(message)
```

Instead of learning 5+ different APIs, you learn one. Instead of rewriting code to switch providers, you change one line.

It was brilliant in its simplicity.

## The Limitations We Saw

Kosong was perfect for what it needed to be: a lightweight abstraction for a few key providers. But as we started building Nova, we realized we needed more:

**Coverage:** Kosong supported 3-4 providers. We wanted 10+.

**Models:** Kosong had dozens of models. We wanted hundreds.

**Features:** We needed streaming, error normalization, rate limiting, and more.

**Maintainability:** Hardcoded model lists meant constant updates. We wanted automatic discovery.

**Optimization:** We needed cost-saving features like prompt caching and smart routing.

Kosong gave us the pattern. We needed to expand it.

## Building UPL

We took kosong's core insight - provider abstraction as a first-class pattern - and rebuilt it from the ground up for our needs.

### Version 1.0 (November 2025)
**The Foundation**

Our first production version powered by the kosong inspiration:
- **12 providers** (6 free, 3 paid, 3 local)
- **211 models** across the ecosystem
- **30,300+ free requests/day** from free providers
- **Smart fallback chains** for reliability
- **Zero vendor lock-in** - switch providers anytime

UPL v1.0 became the backbone of our entire development toolkit. It powered Helix (our React component generator) and gave us the confidence to build more ambitious tools.

### Version 1.2 (November 2025)
**The Evolution**

As we built Nova, we hit new challenges that pushed UPL further:

**Challenge 1: Manual Model Updates**
- Every time Claude or Moonshot released new models, we had to update hardcoded lists
- Solution: **Dynamic model fetching** - providers auto-fetch from their APIs
- Result: New models appear automatically, zero maintenance

**Challenge 2: Expensive Context**
- Nova needed to pass entire codebases to Claude
- 100KB of project files × 10 requests = $3.00
- Solution: **Prompt caching** - cache context across requests
- Result: 81% cost reduction ($3.00 → $0.57)

**Challenge 3: Complex Reasoning**
- Some coding tasks needed deeper analysis
- Solution: **Extended reasoning mode** - Claude 4.x thinking budget
- Result: Better architecture decisions, smarter code generation

**Challenge 4: Massive Codebases**
- Some projects exceeded Claude's 200K context limit
- Solution: **1M context with Moonshot** - 5x larger than Claude
- Result: Entire monorepos fit in one prompt

Each challenge pushed UPL beyond what kosong originally showed us. But the core pattern - one interface, any provider - remained unchanged.

## The Ecosystem We Built

With UPL as the foundation, we built an entire suite of free developer tools:

**Nova** - AI coding assistant with 228+ models
- Uses: Dynamic fetching, prompt caching, thinking mode
- Impact: Professional-grade coding assistant, completely free

**Helix** - React component generator
- Uses: Multi-provider support, auto-failover
- Impact: Free alternative to v0.dev

**Forge** - Game development AI (in development)
- Uses: 1M context, extended reasoning
- Impact: Build games with AI assistance

**Prism** - ServiceNow assistant (closed beta)
- Uses: Custom model routing
- Impact: Free alternative to ServiceNow's paid AI tools

None of these would exist without UPL. And UPL wouldn't exist without kosong's inspiration.

## Paying It Forward

Kosong was open source. The Moonshot team didn't have to share their work, but they did. That generosity enabled us to build something better.

We're publishing UPL for the same reason. Not because we want to support it as a product, but because **open source compounds**.

Someone will see UPL and build something even better. Maybe they'll abstract away the differences between 50 providers instead of 12. Maybe they'll add features we never imagined. Maybe they'll build an entire ecosystem that makes ours look quaint.

That's the dream.

## The Pattern That Matters

The specific code in kosong wasn't revolutionary. The specific code in UPL isn't either. What matters is the **pattern**:

**Stop accepting vendor lock-in.**

You don't have to commit to one AI provider. You don't have to rewrite your code when a better model comes out. You don't have to choose between quality and cost.

Build an abstraction layer. Make switching providers trivial. Give yourself options.

Kosong showed us this pattern. We refined it. Now we're showing you.

What will you build?

## Technical Evolution

### What Kosong Had
- Provider abstraction pattern
- Basic chat completion
- Error handling
- A few key providers

### What UPL Added
- **12 providers** (vs kosong's 3-4)
- **228+ models** (vs kosong's dozens)
- **Dynamic model fetching** (vs hardcoded lists)
- **Prompt caching** (vs full-price every time)
- **Extended reasoning** (vs basic completion)
- **1M context** (vs standard limits)
- **Streaming support** (vs request-response only)
- **Smart fallback chains** (vs single provider)
- **Rate limiting** (vs hope for the best)

### What You Might Add
- 50+ providers?
- Automatic cost optimization?
- Model performance benchmarking?
- Multi-model ensemble responses?
- Your idea here?

## Thank You

To the Moonshot team for kosong. You opened a door we didn't know existed.

To everyone reading this: if UPL helps you build something cool, pay it forward. Share your work. Inspire the next person.

That's how we make better tools.

---

## Links

- **kosong (Original Inspiration):** https://github.com/MoonshotAI/kosong
- **UPL (This Project):** https://github.com/AethyrionAI/UPL
- **Nova (Powered by UPL):** https://github.com/AethyrionAI/Nova
- **Helix (Powered by UPL):** https://github.com/AethyrionAI/Helix
- **Aethyrion Ecosystem:** https://aethyrion.org

---

*"We build too many walls and not enough bridges." - Isaac Newton*

*"If I have seen further, it is by standing on the shoulders of giants." - Also Isaac Newton*

---

**Written:** November 2025  
**Author:** Owen, Aethyrion  
**License:** MIT
