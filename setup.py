"""
Universal Provider Layer (UPL) - Setup
"""

from setuptools import setup, find_packages

setup(
    name="aethyrion-upl",
    version="1.2.0",
    description="Universal Provider Layer for LLM providers with dynamic model fetching, prompt caching, and thinking mode support",
    author="Owen - Aethyrion",
    packages=find_packages(),
    install_requires=[
        "python-dotenv>=1.0.0",
        "httpx>=0.24.0",
        "groq>=0.9.0",
        "google-generativeai>=0.8.0",
        "huggingface-hub>=0.20.0",
        "openai>=1.0.0",
        "anthropic>=0.18.0",
    ],
    python_requires=">=3.11",
)
