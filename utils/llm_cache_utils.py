# plan to consolidate all LLM cache utilities (retrieval, inspection, clearing, exporting) into one focused file.
from datetime import datetime, timezone
import os
import json
import hashlib
from pathlib import Path
from typing import Callable, Tuple
import streamlit as st
import requests
from utils.data_helpers import normalize_provider_fields, parse_utility_block
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

def remove_user_fallback_file(city: str, zip_code: str, utility_key: str):
    """Deletes the fallback file for a given utility/location if it exists."""
    path = get_user_fallback_path(city, zip_code, utility_key)
    if os.path.exists(path):
        os.remove(path)

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
    
def is_usable_provider_response(data: dict, placeholder: str = "⚠️ Not Available") -> bool:
    """
    Returns True if at least 2 core fields are non-empty and not common placeholders.
    """
    core_fields = ["name", "description", "contact_phone", "contact_address"]
    bad_values = {placeholder.lower(), "not available", "n/a", "none", ""}

    valid_fields = [
        v for k, v in data.items()
        if k in core_fields and isinstance(v, str) and v.strip().lower() not in bad_values
    ]

    return len(valid_fields) >= 2

def get_best_provider_data(utility: str, city: str, zip_code: str) -> Tuple[dict, str, dict]:
    """
    Attempts to load provider data in this order:
    1. User fallback (editable overrides)
    2. Cached parsed data from disk
    3. Empty {}
    Returns:
        - provider_data (dict)
        - source_label (str): one of "user_fallback", "cache", "empty"
        - raw_data: raw dict (for debugging/logging)

    """
    fallback_path = f"data/provider_fallbacks/{utility}_{city.lower().strip()}_{zip_code}.json"
    cache_path = get_provider_cache_path(utility, city, zip_code)

    user_data = {}
    cache_data = {}

    # 🥇 Try user fallback first
    if os.path.exists(fallback_path):
        try:
            with open(fallback_path, "r", encoding="utf-8") as f:
                user_data = json.load(f)
        except Exception as e:
            print(f"❌ Failed to load fallback for `{utility}`: {e}")

    # 🥈 Try cache next
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                if isinstance(raw, str):
                    raw = parse_utility_block(raw)
                cache_data = raw
        except Exception as e:
            print(f"❌ Failed to load cache for `{utility}`: {e}")

    # 🧪 Build debug log
    decision_log = {
    "utility": utility,
    "city": city,
    "zip": zip_code,
    "user_data_keys": list(user_data.keys()),
    "cache_data_keys": list(cache_data.keys()),
    "user_ts": user_data.get("timestamp"),
    "cache_ts": cache_data.get("timestamp"),
    }

    # ✅ Use user fallback if it has meaningful data
    if has_meaningful_data(user_data):
        decision_log["decision"] = "user_fallback"
        st.session_state.setdefault("provider_debug_log", {})[utility] = decision_log
        return normalize_provider_fields(user_data), "user_fallback", user_data

    # 🧮 Use cache if it has meaningful data (and newer or equal timestamp)
    if has_meaningful_data(cache_data):
        user_ts = parse_timestamp(user_data.get("timestamp", ""))
        cache_ts = parse_timestamp(cache_data.get("timestamp", ""))
        decision_log["user_ts_parsed"] = str(user_ts)
        decision_log["cache_ts_parsed"] = str(cache_ts)

        if cache_ts >= user_ts:
            decision_log["decision"] = "cache"
            st.session_state.setdefault("provider_debug_log", {})[utility] = decision_log
            return normalize_provider_fields(cache_data), "cache", cache_data
        
        decision_log["decision"] = "user_fallback (stale)"

    # 🧨 Nothing usable
    decision_log["decision"] = "none"
    st.session_state.setdefault("provider_debug_log", {})[utility] = decision_log

    return {}, "none", {}

def has_meaningful_data(d: dict) -> bool:
    """
    Returns True if any of the important fields have valid, non-placeholder content.
    """
    if not isinstance(d, dict):
        return False

    ignore_values = {"", "⚠️ not available", "not available", "n/a", "none"}
    keys = ["name", "contact_phone", "contact_address", "contact_website"]

    return any(
        str(d.get(k, "")).strip().lower() not in ignore_values for k in keys
    )


def save_user_fallback_data(city: str, zip_code: str, utility_key: str, data: dict):
    """
    Save user-edited provider data in a flat format to the fallback file.
    Includes metadata such as city, zip, utility type, and update timestamp.
    """

    # Compose payload with flat structure
    payload = {
        "city": city,
        "zip_code": zip_code,
        "utility": utility_key,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    payload.update(data)

    # Ensure fallback directory exists
    file_path = get_user_fallback_path(city, zip_code, utility_key)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as f:
        json.dump(payload, f, indent=2)

def ensure_datetime_strings(data: dict) -> dict:
    """
    Recursively converts datetime objects to ISO strings.
    """
    from datetime import datetime

    def convert(value):
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        return value

    return convert(data)

def save_provider_update_to_disk(city: str, zip_code: str, utility: str, data: dict):
    """
    Saves corrected or validated provider info to a JSON file for fallback reuse.
    Flattens structure so contact fields are top-level (not nested under "data").
    """
    fallback_dir = "data/provider_fallbacks"
    os.makedirs(fallback_dir, exist_ok=True)

    # Use city_zip_utility as filename
    safe_city = city.lower().replace(" ", "_")
    filename = f"{utility}_{safe_city}_{zip_code}.json"
    filepath = os.path.join(fallback_dir, filename)

    # Flattened payload
    payload = {
        "city": city,
        "zip_code": zip_code,
        "utility": utility,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **data  # injects all contact fields at top level
    }

    safe_payload = ensure_datetime_strings(payload)

    with open(filepath, "w") as f:
        json.dump(safe_payload, f, indent=2)

    if st.session_state.get("enable_debug_mode"):
        st.info(f"💾 Saved fallback: `{filename}`")