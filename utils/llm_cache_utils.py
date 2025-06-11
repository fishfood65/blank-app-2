# plan to consolidate all LLM cache utilities (retrieval, inspection, clearing, exporting) into one focused file.
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
    cache_dir = Path("llm_cache")
    cache_dir.mkdir(exist_ok=True)
    hash_key = hash_block_content(prompt)
    cache_file = cache_dir / f"{hash_key}.json"

    if cache_file.exists():
        with open(cache_file, "r") as f:
            data = json.load(f)
            return data.get("output", "")

    # If no generate_fn passed, use default
    if generate_fn is None:
        generate_fn = lambda: call_openrouter_chat(prompt)

    output = generate_fn()

    if output:
        with open(cache_file, "w") as f:
            json.dump({"prompt": prompt, "output": output}, f)

    return output