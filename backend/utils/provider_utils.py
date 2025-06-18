#backend/utils/provider_utils.py
import re
from datetime import datetime, timezone
from typing import Dict, Optional
from backend.utils.prompt_templates import generate_single_provider_prompt, parse_utility_block, normalize_provider_fields, is_usable_provider_response
from backend.utils.llm_helpers import call_openrouter_chat, get_or_generate_llm_output
from backend.utils.provider_fallbacks import DEFAULT_PROVIDER_MAP, apply_provider_overrides, get_best_provider_data
from backend.utils.cache_helpers import save_provider_to_cache  # ✅ Newly added
from config.constants import UTILITY_KEYS

def fetch_utility_providers_backend(
    city: str,
    zip_code: str,
    internet_hint: str,
    user_corrections: Optional[Dict] = None,
    force_refresh_map: Optional[Dict[str, bool]] = None,
    debug_mode: bool = False
) -> Dict[str, Dict]:
    """
    FastAPI-compatible version of the utility provider resolver.
    Returns structured provider data for each utility.
    """
    results = {}
    if force_refresh_map is None:
        force_refresh_map = {}
    if user_corrections is None:
        user_corrections = {}

    for utility in UTILITY_KEYS:
        label = utility.replace("_", " ").title()
        force_refresh = force_refresh_map.get(utility, False)

        # Step 1: Attempt cached or fallback data
        provider_data, source, user_data = get_best_provider_data(utility, city, zip_code)

        if provider_data and is_usable_provider_response(provider_data) and not force_refresh:
            provider_data["source"] = source
            results[utility] = provider_data
            continue

        # Step 2: Generate prompt and call LLM
        prompt = generate_single_provider_prompt(utility, city, zip_code, internet_hint)
        cache_key = f"{utility}|{city}|{zip_code}|{internet_hint}"

        try:
            raw_response = get_or_generate_llm_output(prompt, cache_key)
            if not raw_response:
                raise ValueError("Empty response from LLM")
            parsed = normalize_provider_fields(parse_utility_block(raw_response))
        except Exception as e:
            if debug_mode:
                print(f"[❌] LLM call failed for {label}: {e}")
            parsed = {}

        # Step 3: Retry if name/description is missing
        if not parsed.get("name") or not parsed.get("description"):
            try:
                retry_response = call_openrouter_chat(prompt)
                if not retry_response:
                    raise ValueError("Empty retry response from LLM")
                parsed = normalize_provider_fields(parse_utility_block(retry_response))
            except Exception as e:
                if debug_mode:
                    print(f"[⚠️] Retry failed for {label}: {e}")
                parsed = {}

        # Step 4: Apply static fallback if still incomplete
        fallback = DEFAULT_PROVIDER_MAP.get(utility, {})
        missing_critical = any(not parsed.get(field, "").strip() for field in ["name", "description", "contact_phone", "contact_address"])
        if missing_critical and fallback:
            for field, val in fallback.items():
                if not parsed.get(field):
                    parsed[field] = val
            parsed["source"] = "fallback"

        # Step 5: Apply user override corrections
        if utility in user_corrections:
            parsed, name_changed, needs_refresh = apply_provider_overrides(parsed, user_corrections[utility])

            if debug_mode and (name_changed or needs_refresh):
                print(f"[🔁] User override triggered: name_changed={name_changed}, needs_refresh={needs_refresh}")

            if needs_refresh:
                retry_prompt = generate_single_provider_prompt(utility, city, zip_code, internet_hint)
                try:
                    retry_response = call_openrouter_chat(retry_prompt)
                    if retry_response:
                        parsed = normalize_provider_fields(parse_utility_block(retry_response))
                        parsed["source"] = "llm"
                except Exception as e:
                    if debug_mode:
                        print(f"[❌] Retry after override failed for {utility}: {e}")

        # Step 6: Final tagging and timestamp
        parsed["timestamp"] = datetime.now(timezone.utc).isoformat()
        parsed["source"] = parsed.get("source", "llm")

        # Step 7: Save fallback to disk if needed
        if parsed["source"] == "fallback":
            save_provider_to_cache(utility, city, zip_code, parsed)

        results[utility] = parsed

    return results
