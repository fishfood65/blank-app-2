# plan to consolidate all LLM cache utilities (retrieval, inspection, clearing, exporting) into one focused file.
from datetime import datetime
import os
import json
import hashlib
from pathlib import Path
from typing import Callable
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
            "timestamp": datetime.utcnow().isoformat()
        }
        with open(cache_file, "w") as f:
            json.dump(metadata, f, indent=2)

        if st.session_state.get("enable_debug_mode"):
            st.markdown("### 🧠 New LLM Response Cached")
            st.write(f"📄 Cache File: `{cache_file.name}`")
            st.json({"model": metadata["model"], "timestamp": metadata["timestamp"]})

    return output