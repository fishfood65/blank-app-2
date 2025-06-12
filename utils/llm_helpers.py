# utils/llm_helpers.py
import os
import requests
import streamlit as st

def call_openrouter_chat(prompt: str) -> str:
    import os
    import requests
    import streamlit as st

    api_key = os.getenv("OPENROUTER_TOKEN")
    referer = os.getenv("OPENROUTER_REFERER", "https://example.com")
    model_name = st.session_state.get("llm_model", "claude-3-haiku")

    if not api_key:
        st.error("❌ Missing OpenRouter API key.")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": referer,
        "X-Title": "UtilityProviderLookup"
    }

    # ✅ Ensure prompt is non-empty and properly stripped
    prompt = prompt.strip()
    if not prompt:
        st.error("❌ Empty prompt passed to LLM.")
        return None

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
        return response.json()["choices"][0]["message"]["content"]

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

