# UPL Version 1.2.0 - Update Summary

**Date:** November 22, 2025  
**Version:** 1.0.0 → 1.2.0  
**Status:** ✅ Complete

---

## 📝 Files Updated

1. ✅ **setup.py** - Version bumped to 1.2.0
2. ✅ **__init__.py** - Version constant updated
3. ✅ **README.md** - Complete rewrite with new features
4. ✅ **CHANGELOG.md** - New file with version history

---

## 🎯 Major Changes

### Dynamic Model Fetching ⭐
- Claude: Fetches from Anthropic `/models` API
- Moonshot: Fetches from OpenAI-compatible API
- 228+ models (up from 211)

### Prompt Caching ⭐
- Claude only
- 81% cost reduction
- $3.00 → $0.57 per 10 requests

### Extended Reasoning ⭐
- Claude 4.x thinking mode
- 10,000 token budget

### Updated Models
- Claude Sonnet 4.5, Haiku 4.5, Opus 4.1
- Kimi K2 Thinking (1M context!)

---

## ✅ Verification

```bash
cd O:\Nova\UPL
python -c "import providers; print(providers.__version__)"
# Output: 1.2.0
```

---

**All documentation updated and version bumped successfully!**
