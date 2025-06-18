import os
from datetime import datetime
from backend.utils.cache_helpers import get_provider_cache_path

DEFAULT_PROVIDER_MAP = {
    "electricity": {
        "name": "Pacific Gas and Electric Company (PG&E)",
        "description": "PG&E provides electric service to much of Northern California.",
        "contact_phone": "1-800-743-5000",
        "contact_address": "77 Beale St, San Francisco, CA 94105",
        "contact_website": "https://www.pge.com"
    },
    "natural_gas": {
        "name": "Pacific Gas and Electric Company (PG&E)",
        "description": "PG&E handles natural gas delivery in many areas of California.",
        "contact_phone": "1-800-743-5000",
        "contact_address": "77 Beale St, San Francisco, CA 94105",
        "contact_website": "https://www.pge.com"
    },
    "water": {
        "name": "San Jose Water",
        "description": "Provides water utility services to San Jose and surrounding communities.",
        "contact_phone": "408-279-7900",
        "contact_address": "110 W Taylor St, San Jose, CA 95110",
        "contact_website": "https://www.sjwater.com"
    },
    "internet": {
        "name": "Comcast Xfinity",
        "description": "Xfinity offers internet and cable services nationwide.",
        "contact_phone": "1-800-XFINITY",
        "contact_address": "1701 JFK Blvd, Philadelphia, PA 19103",
        "contact_website": "https://www.xfinity.com"
    }
}

def apply_provider_overrides(
    parsed: dict,
    user_override: dict,
    fields_to_override: list = ["contact_phone", "contact_address", "contact_website", "description"]
) -> tuple[dict, bool, bool]:
    """
    Applies user-confirmed overrides to a parsed provider block.
    
    Args:
        parsed (dict): The raw parsed LLM response (can be empty).
        user_override (dict): The user's confirmed update from disk/session.
        fields_to_override (list): Fields eligible for override (excluding 'name').

    Returns:
        Tuple[dict, bool, bool]: (updated parsed block, name_changed flag, needs_refresh flag)
    """
    if not user_override or not isinstance(user_override, dict):
        return parsed, False, False

    updated = parsed.copy()
    name_before = parsed.get("name", "").strip()
    name_after = user_override.get("name", "").strip()

    name_changed = name_before and name_after and (name_before != name_after)
    needs_refresh = False

    if not name_changed:
        # ✅ Use stored name + apply overrides
        if name_after:
            updated["name"] = name_after
        for field in fields_to_override:
            user_val = user_override.get(field, "").strip()
            if user_val:
                updated[field] = user_val
    else:
        # 🛠 Name was changed → assume rest of LLM block may be stale
        updated["name"] = name_after
        for field in fields_to_override:
            existing_val = parsed.get(field, "").strip()
            if not existing_val:
                needs_refresh = True  # Field is missing after name change
            updated.pop(field, None)  # Remove stale value

    return updated, name_changed, needs_refresh

def get_best_provider_data(
    utility: str,
    city: str,
    zip_code: str,
    debug_mode: bool = False
) -> tuple[dict, str, dict]:
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

    if os.path.exists(fallback_path):
        try:
            with open(fallback_path, "r", encoding="utf-8") as f:
                user_data = json.load(f)
        except Exception as e:
            if debug_mode:
                print(f"❌ Failed to load fallback for `{utility}`: {e}")

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                if isinstance(raw, str):
                    raw = parse_utility_block(raw)
                cache_data = raw
        except Exception as e:
            if debug_mode:
                print(f"❌ Failed to load cache for `{utility}`: {e}")

    # Debug trace
    if debug_mode:
        print(f"🔍 {utility} – user_keys: {list(user_data.keys())}, cache_keys: {list(cache_data.keys())}")

    if has_meaningful_data(user_data):
        return normalize_provider_fields(user_data), "user_fallback", user_data

    if has_meaningful_data(cache_data):
        user_ts = parse_timestamp(user_data.get("timestamp", ""))
        cache_ts = parse_timestamp(cache_data.get("timestamp", ""))

        if debug_mode:
            print(f"📊 Timestamps → user: {user_ts}, cache: {cache_ts}")

        if cache_ts >= user_ts:
            return normalize_provider_fields(cache_data), "cache", cache_data

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


def parse_timestamp(ts: str) -> datetime:
    """
    Parses ISO-formatted timestamp string to datetime object.
    Returns datetime.min on failure.
    """
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return datetime.min
