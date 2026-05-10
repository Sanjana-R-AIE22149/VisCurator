#!/usr/bin/env python3
"""Run this to check if VisCurator's environment is correctly configured."""
import os, sys
from pathlib import Path

print("=== VisCurator Environment Check ===\n")

# Check .env
env_file = Path("backend/.env")
print(f".env file: {'EXISTS' if env_file.exists() else 'MISSING — copy .env.example to .env'}")

if env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
    except ImportError:
        print("⚠️  dotenv not installed, skipping .env loading")

# Check API key
key = os.getenv("NVIDIA_API_KEY", "")
if not key or key == "your_key_here":
    print("❌ NVIDIA_API_KEY: NOT SET or is placeholder")
    print("   → Get one free at https://build.nvidia.com")
else:
    print(f"✓  NVIDIA_API_KEY: set ({key[:8]}...)")

# Check packages
packages = ["datasets", "torch", "albumentations", "cv2", "imagehash", 
            "PIL", "aiohttp", "openai", "fastapi", "uvicorn"]
print("\nPackages:")
missing = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f"  ✓  {pkg}")
    except ImportError:
        print(f"  ❌ {pkg} — run: pip install -r backend/requirements.txt")
        missing.append(pkg)

# Test NIM connection
print("\nTesting NIM connection...")
if key and key != "your_key_here":
    try:
        import asyncio
        from openai import AsyncOpenAI
        async def _test():
            client = AsyncOpenAI(
                api_key=key, 
                base_url="https://integrate.api.nvidia.com/v1"
            )
            resp = await client.chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=[{"role": "user", "content": "Reply with only the word: OK"}],
                max_tokens=5,
            )
            return resp.choices[0].message.content
        result = asyncio.run(_test())
        print(f"  ✓  NIM responded: {result!r}")
    except Exception as e:
        print(f"  ❌ NIM connection failed: {e}")
else:
    print("  ⚠️  Skipped — no API key")

# Test HuggingFace
print("\nTesting HuggingFace datasets API...")
try:
    import asyncio, aiohttp
    async def _test_hf():
        async with aiohttp.ClientSession() as s:
            async with s.get("https://huggingface.co/api/datasets?search=mnist&limit=1") as r:
                return r.status, await r.json()
    status, data = asyncio.run(_test_hf())
    print(f"  ✓  HuggingFace API: HTTP {status}, got {len(data)} result(s)")
except Exception as e:
    print(f"  ❌ HuggingFace API failed: {e}")

print("\n=== Check complete ===")
if missing:
    print(f"\nFix missing packages with:")
    print(f"  pip install {' '.join(missing)}")
