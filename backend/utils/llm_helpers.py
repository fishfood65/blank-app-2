import os
import requests
from datetime import datetime
from typing import Optional
from pathlib import Path
import hashlib
import json

# Caching directory
CACHE_DIR = Path("provider_cache")
CACHE_DIR.mkdir(exist_ok=True)
USAGE_LOG_PATH = Path("provider_cache/llm_usage_log.json")


def estimate_token_count(text: str, model: str = "openai/gpt-4o:online") -> int:
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return len(text.split())  # crude fallback


def log_llm_usage(entry: dict):
    try:
        if USAGE_LOG_PATH.exists():
            existing = json.loads(USAGE_LOG_PATH.read_text())
        else:
            existing = []
        existing.append(entry)
        USAGE_LOG_PATH.write_text(json.dumps(existing, indent=2))
    except Exception as e:
        print(f"⚠️ Failed to write usage log: {e}")


def call_openrouter_chat(prompt: str, model: Optional[str] = None, debug: bool = False) -> str:
    """
    Calls OpenRouter API with the given prompt and returns only the assistant's response text.
    Logs token usage and errors to disk.
    """
    api_key = os.getenv("OPENROUTER_TOKEN")
    referer = os.getenv("OPENROUTER_REFERER", "https://example.com")
    timeout = 15  # seconds

    if not api_key:
        raise ValueError("❌ Missing OpenRouter API key.")

    if not prompt.strip():
        raise ValueError("❌ Prompt is empty.")

    base_model = model or os.getenv("DEFAULT_LLM_MODEL", "openai/gpt-4o")
    model_name = base_model if ":online" in base_model else base_model + ":online"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": referer,
        "Content-Type": "application/json",
        "X-Title": "UtilityProviderLookup"
    }

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,
        "temperature": 0.5
    }

    if debug:
        print("\n📤 OpenRouter Payload:")
        print(json.dumps(payload, indent=2))

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        response_json = response.json()
        content = response_json["choices"][0]["message"]["content"]

        if not content:
            raise ValueError("❌ LLM response contained no usable content.")

        usage_data = response_json.get("usage", {})
        prompt_tokens = usage_data.get("prompt_tokens") or estimate_token_count(prompt, model=model_name)
        response_tokens = usage_data.get("completion_tokens") or estimate_token_count(content, model=model_name)

        usage_entry = {
            "model": model_name,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": prompt_tokens + response_tokens,
            "timestamp": datetime.now().isoformat(),
            "prompt_preview": prompt[:200]
        }

        log_llm_usage(usage_entry)

        if debug:
            print("\n✅ LLM Response:")
            print(content)
            print("\n📊 Token Usage:")
            print(json.dumps(usage_entry, indent=2))

        return content

    except requests.Timeout:
        raise TimeoutError(f"❌ OpenRouter request timed out after {timeout} seconds.")

    except requests.RequestException as e:
        print(f"❌ OpenRouter API error: {e}")
        if hasattr(e, "response") and e.response is not None:
            print("\n📄 API Error Response:")
            print(e.response.text)
        raise


def get_or_generate_llm_output(prompt: str, cache_key: str) -> str:
    hash_key = hashlib.sha256(cache_key.encode()).hexdigest()
    path = CACHE_DIR / f"{hash_key}.txt"

    if path.exists():
        return path.read_text()

    result = call_openrouter_chat(prompt)
    path.write_text(result)
    return result
