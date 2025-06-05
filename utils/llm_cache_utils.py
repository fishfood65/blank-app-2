# plan to consolidate all LLM cache utilities (retrieval, inspection, clearing, exporting) into one focused file.
import os
import json
import hashlib
from pathlib import Path
from typing import Callable
import streamlit as st
import requests

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

