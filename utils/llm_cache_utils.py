# plan to consolidate all LLM cache utilities (retrieval, inspection, clearing, exporting) into one focused file.
from datetime import datetime, timezone
import os
import json
import hashlib
from pathlib import Path
from typing import Callable, Tuple
import streamlit as st
import requests
from utils.llm_helpers import call_openrouter_chat

### How to Use It In your app
# ---
# from llm_cache_utils import get_or_generate_llm_output
# 
# [Comment] Example use inside a runbook generator
# output = get_or_generate_llm_output(
#    prompt=combined_prompt,
#    generate_fn=lambda: call_mistral_or_openai(prompt=combined_prompt)
# )

def hash_block_content(block: str) -> str:
    """Generate a SHA-256 hash for a given prompt string."""
    return hashlib.sha256(block.encode("utf-8")).hexdigest()

def get_or_generate_llm_output(prompt: str, generate_fn: Callable = None) -> str:
    """
    Retrieves a cached LLM output for the given prompt if available,
    otherwise calls the provided function (or call_openrouter_chat) and caches the result.

    Adds model name, timestamp, and debug indicators to the cache file.
    """
    cache_dir = Path("llm_cache")
    cache_dir.mkdir(exist_ok=True)
    hash_key = hash_block_content(prompt)
    cache_file = cache_dir / f"{hash_key}.json"

    if cache_file.exists():
        with open(cache_file, "r") as f:
            data = json.load(f)
            if st.session_state.get("enable_debug_mode"):
                st.markdown("### 🧠 Cached LLM Response Used")
                st.write(f"📄 Cache File: `{cache_file.name}`")
                st.json({"model": data.get("model", "unknown"), "timestamp": data.get("timestamp", "unknown")})
            return data.get("output", "")

    # If no custom function provided, fall back to default OpenRouter call
    if generate_fn is None:
        generate_fn = lambda: call_openrouter_chat(prompt)

    output = generate_fn()

    if output:
        metadata = {
            "prompt": prompt,
            "output": output,
            "model": st.session_state.get("llm_model", "unknown"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        with open(cache_file, "w") as f:
            json.dump(metadata, f, indent=2)

        if st.session_state.get("enable_debug_mode"):
            st.markdown("### 🧠 New LLM Response Cached")
            st.write(f"📄 Cache File: `{cache_file.name}`")
            st.json({"model": metadata["model"], "timestamp": metadata["timestamp"]})

    return output

# --- Provider Fallback Cache Utilities ---

PROVIDER_CACHE_DIR = "provider_cache"
USER_FALLBACK_DIR = "data/provider_fallbacks"
os.makedirs(PROVIDER_CACHE_DIR, exist_ok=True)
os.makedirs(USER_FALLBACK_DIR, exist_ok=True)

def get_provider_cache_path(utility: str, city: str, zip_code: str) -> str:
    key = f"{utility}_{city.lower().strip()}_{zip_code}"
    hashed = hashlib.sha256(key.encode()).hexdigest()
    return os.path.join(PROVIDER_CACHE_DIR, f"{utility}_{hashed}.json")

def get_user_fallback_path(city: str, zip_code: str, utility_key: str) -> str:
    city_slug = city.lower().replace(" ", "_")
    return os.path.join(USER_FALLBACK_DIR, f"{utility_key}_{city_slug}_{zip_code}.json")

def load_json_file(path: str) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def parse_timestamp(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return datetime.min

def get_best_provider_data(utility_key: str, city: str, zip_code: str) -> Tuple[dict, str]:
    """
    Chooses the freshest available provider data (user_fallback > cache > none).
    Returns:
        - provider_data (dict)
        - source_label (str): one of "user_fallback", "cache", "empty"
    """
    user_path = get_user_fallback_path(city, zip_code, utility_key)
    cache_path = get_provider_cache_path(utility_key, city, zip_code)

    user_data = load_json_file(user_path)
    cache_data = load_json_file(cache_path)

    user_ts = parse_timestamp(user_data.get("timestamp", ""))
    cache_ts = parse_timestamp(cache_data.get("timestamp", ""))

    if user_ts > cache_ts:
        return user_data, "user_fallback"
    elif cache_data:
        return cache_data, "cache"
    else:
        return {}, "empty"