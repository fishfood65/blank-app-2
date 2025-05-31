import os
import json
import hashlib
from pathlib import Path
from typing import Callable

def generate_prompt_hash(prompt: str) -> str:
    """Generate a SHA-256 hash for a given prompt string."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

def save_llm_output(prompt: str, content: str, cache_dir: str = "cached_llm_outputs") -> str:
    """
    Saves the LLM output to a JSON file named by the hash of the prompt.
    Returns the hash key used for storage.
    """
    key = generate_prompt_hash(prompt)
    path = Path(cache_dir)
    path.mkdir(exist_ok=True)
    with open(path / f"{key}.json", "w", encoding="utf-8") as f:
        json.dump({"prompt": prompt, "output": content}, f, indent=2)
    return key

def load_llm_output(prompt: str, cache_dir: str = "cached_llm_outputs") -> str | None:
    """
    Loads a cached LLM output if it exists, using the hash of the prompt.
    Returns the output string, or None if not found.
    """
    key = generate_prompt_hash(prompt)
    filepath = Path(cache_dir) / f"{key}.json"
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f).get("output")
    return None

def is_llm_output_cached(prompt: str, cache_dir: str = "cached_llm_outputs") -> bool:
    """Checks whether the output for a given prompt is already cached."""
    key = generate_prompt_hash(prompt)
    return (Path(cache_dir) / f"{key}.json").exists()

def get_or_generate_llm_output(prompt: str, generate_fn: Callable[[], str], cache_dir: str = "cached_llm_outputs") -> str:
    """
    High-level utility to get LLM output from cache or generate it if missing.
    - `prompt`: the LLM prompt string
    - `generate_fn`: a callable that returns the output if cache miss
    - `cache_dir`: folder where results are cached
    """
    cached = load_llm_output(prompt, cache_dir=cache_dir)
    if cached is not None:
        return cached

    result = generate_fn()
    save_llm_output(prompt, result, cache_dir=cache_dir)
    return result
