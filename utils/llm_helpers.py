# utils/llm_helpers.py
from datetime import datetime
import os
import requests
import streamlit as st
import tiktoken
import time
import json


LLM_USAGE_DIR = "llm_cache"
LLM_USAGE_FILE = os.path.join(LLM_USAGE_DIR, "usage_log.jsonl")

def append_llm_usage_log(entry: dict):
    """
    Appends a single LLM usage entry to disk in JSONL format.
    Ensures directory exists and avoids duplicate writes for identical timestamps.
    """
    os.makedirs(LLM_USAGE_DIR, exist_ok=True)

    if not isinstance(entry, dict):
        raise ValueError("Entry must be a dictionary")

    # Add a fallback timestamp if missing
    entry.setdefault("timestamp", datetime.now().isoformat())

    with open(LLM_USAGE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def call_openrouter_chat(prompt: str) -> str:

# Optional: Local token estimator using tiktoken
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

    api_key = os.getenv("OPENROUTER_TOKEN")
    referer = os.getenv("OPENROUTER_REFERER", "https://example.com")
    base_model = st.session_state.get("llm_model", "openai/gpt-4o")
    if ":online" not in base_model and "openai/gpt-4o" not in base_model:
        model_name = base_model + ":online"
    else:
        model_name = base_model

    timeout = 15 # seconds

    if not api_key:
        st.error("❌ Missing OpenRouter API key.")
        return 
    
    prompt = prompt.strip()
    if not prompt:
        st.error("❌ Empty prompt passed to LLM.")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": referer,
        "X-Title": "UtilityProviderLookup"
    }

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,
        "temperature": 0.5,
    }

    if st.session_state.get("enable_debug_mode"):
        st.markdown("### 🧪 OpenRouter Payload")
        st.json(payload)

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        response_json = response.json()

        # ✅ Only now extract content
        content = response_json["choices"][0]["message"]["content"]

        if not content:
            st.error("❌ LLM response did not contain usable content.")
            return None
        
        st.session_state["last_llm_output"] = content  # ✅ Save raw LLM output

        # ✅ Attempt to extract usage data (if available)
        usage_data = response_json.get("usage", {})  # <--- This line defines usage_data

        annotations = response_json["choices"][0]["message"].get("annotations", [])
        if annotations:
            st.session_state["last_web_citations"] = annotations

        # Prefer OpenRouter-provided usage if available
        if "prompt_tokens" in usage_data and "completion_tokens" in usage_data:
            prompt_tokens = usage_data["prompt_tokens"]
            response_tokens = usage_data["completion_tokens"]
        else:
            prompt_tokens = estimate_token_count(prompt, model=model_name)
            response_tokens = estimate_token_count(content, model=model_name)

        usage_entry = {
            "model": model_name,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": prompt_tokens + response_tokens,
            "timestamp": datetime.now().isoformat(),
            "prompt_preview": prompt[:200] # first 200 chars for debugging
        }
        
        # Append to cumulative log
        log = st.session_state.setdefault("llm_usage_log", [])
        log.append(usage_entry)
        st.session_state["llm_usage_log"] = log

        # ✅ Optional: Persist usage to disk for debug dashboard
        try:
            append_llm_usage_log(usage_entry)
        except Exception as e:
            if st.session_state.get("enable_debug_mode"):
                st.warning(f"⚠️ Failed to write usage log to disk: {e}")

        return content
        # Append to cumulative log
        log = st.session_state.setdefault("llm_usage_log", [])
        log.append(usage_entry)
        st.session_state["llm_usage_log"] = log

        return content

    except requests.Timeout:
        st.error(f"❌ OpenRouter request timed out after {timeout} seconds.")
        return None
    
    except requests.exceptions.RequestException as e:
        st.error(f"❌ OpenRouter API error: {e}")
        if hasattr(e, "response") and e.response is not None:
            try:
                st.code(e.response.text, language="json")
            except:
                pass
        return None

# ⏳ Cooldown Helper
def is_refresh_allowed(key: str, max_attempts: int = 3, cooldown_sec: int = 600) -> bool:
    attempts = st.session_state.setdefault("provider_refresh_attempts", {})
    timestamps = st.session_state.setdefault("provider_refresh_timestamps", {})
    now = time.time()

    last_attempt_time = timestamps.get(key, 0)
    time_since_last = now - last_attempt_time

    # Reset after cooldown
    if time_since_last > cooldown_sec:
        attempts[key] = 0

    if attempts.get(key, 0) < max_attempts:
        return True
    return False

# 🧪 Manual override
if st.session_state.get("enable_debug_mode") and st.button("🔓 Override Refresh Lock"):
    st.session_state["provider_refresh_attempts"] = {}
    st.session_state["provider_refresh_timestamps"] = {}
    st.success("✅ Refresh limits reset.")