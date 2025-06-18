import os
import json
import hashlib
from typing import Dict

# Adjust as needed
CACHE_DIR = os.getenv("PROVIDER_CACHE_DIR", "./cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def get_provider_cache_path(utility: str, city: str, zip_code: str) -> str:
    key = f"{utility}_{city.lower().strip()}_{zip_code}"
    hashed = hashlib.sha256(key.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{utility}_{hashed}.json")

def load_provider_from_cache(utility: str, city: str, zip_code: str) -> dict:
    path = get_provider_cache_path(utility, city, zip_code)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, dict) or not data:
                    st.warning(f"⚠️ Cache file for `{utility}` exists but is empty or malformed:\n{path}")
                else:
                    st.info(f"📂 Loaded cache for `{utility}` with keys: {list(data.keys())}")
                return data
            except Exception as e:
                st.error(f"❌ Failed to load provider cache for `{utility}`: {e}")
                return {}
    else:
        st.warning(f"📭 No cache file found for `{utility}` at path:\n{path}")
    return {}


def save_provider_to_cache(utility: str, city: str, zip_code: str, content: dict):
    path = get_provider_cache_path(utility, city, zip_code)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2)
