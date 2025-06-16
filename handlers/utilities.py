### Leve 1 - Home
#from utils.prompt_block_utils import generate_all_prompt_blocks
from typing import Dict, Optional
import markdown_it
import streamlit as st
import re
import io
from docx import Document
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
import re
import time
import io
import hashlib
import uuid
import json
from pathlib import Path
from utils.preview_helpers import get_active_section_label, display_provider_contact_info
from utils.data_helpers import (
    normalize_provider_fields,
    register_provider_input,
    register_task_input, 
    get_answer, 
    extract_and_log_providers,
    parse_utility_block,
    register_input_only,
    apply_provider_overrides,
    get_provider_display_name
)
#from utils.runbook_generator_helpers import generate_docx_from_prompt_blocks, maybe_render_download, maybe_generate_runbook
from utils.debug_utils import debug_all_sections_input_capture_with_summary, reset_all_session_state
from prompts.templates import generate_single_provider_prompt, wrap_with_claude_style_formatting, generate_corrected_provider_prompt
from utils.common_helpers import get_schedule_placeholder_mapping
from utils.llm_cache_utils import get_or_generate_llm_output
from utils.llm_helpers import call_openrouter_chat
from utils.docx_helpers import export_provider_docx, format_provider_markdown, render_runbook_section_output
from utils.event_logger import log_event
from utils.llm_helpers import is_refresh_allowed
from utils.provider_fallbacks import DEFAULT_PROVIDER_MAP

CACHE_DIR = "provider_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Generate the AI prompt ---
# Load from environment (default) or user input
api_key = os.getenv("OPENROUTER_TOKEN") or st.text_input("Enter your OpenRouter API key:", type="password")
referer = os.getenv("OPENROUTER_REFERER", "https://example.com")
model_name = "openai/gpt-4o:online"  # You can make this dynamic if needed

# Show success/error message
if api_key:
    st.success("✅ OpenRouter API key loaded.")
else:
    st.error("❌ OpenRouter API key is not set.")

# --- Helper functions (top of the file) ---
def get_utilities_inputs(section: str):
    """
    Collects home-related inputs and registers them with task metadata.

    Args:
        section (str): The section name under which to log inputs.

    Returns:
        Tuple[str, str, str]: city, zip_code, and internet_provider
    """
    city = register_task_input(
        label="City",
        input_fn=st.text_input,
        section=section,
        value=st.session_state.get("City", ""),
        key="City",  # Ensures value stored in st.session_state["City"]
        required=True,
        area="home",
        shared=True
    )

    zip_code = register_task_input(
        label="ZIP Code",
        input_fn=st.text_input,
        section=section,
        value=st.session_state.get("ZIP Code", ""),
        key="ZIP Code",
        required=True,
        area="home",
        shared=True
    )

    internet_provider = register_task_input(
        label="Internet Provider",
        input_fn=st.text_input,
        section=section,
        value=st.session_state.get("Internet Provider", ""),
        key="Internet Provider",
        required=False,
        area="home",
        shared=True
    )

    if not internet_provider or internet_provider.lower() in ["⚠️ not provided", "n/a", ""]:
        st.warning("⚠️ No Internet Provider given — will let LLM try to infer it.")

    # Optional: mirror into top-level keys (but not required for get_answer)
    st.session_state["home_city"] = city
    st.session_state["home_zip_code"] = zip_code
    st.session_state["home_internet_provider"] = internet_provider

    return city, zip_code, internet_provider

def query_utility_providers(section: str, test_mode: bool = False) -> dict:
    """
    Queries utility providers using city and ZIP from the specified section.

    Args:
        section (str): The section name where city and ZIP inputs were stored.
        test_mode (bool): If True, return hardcoded results for testing.

    Returns:
        dict: Structured provider metadata, with keys like electricity, water, gas, etc.

    """
    city = get_answer(key="City", section=section, verbose=True)
    zip_code = get_answer(key="ZIP Code", section=section, verbose=True)
    internet = get_answer(key="Internet Provider", section=section, verbose = True)

    if not city or not zip_code:
        st.warning("⚠️ Missing City or ZIP Code. Cannot query providers.")
        return {}

    # Normalize "⚠️ Not provided"
    if internet and internet.strip().lower() in ["", "⚠️ not provided", "n/a"]:
        internet = ""  # Let LLM fill it in

    if test_mode:
        return {
            "electricity": "Austin Energy",
            "natural_gas": "Atmos Energy",
            "water": "Austin Water",
            "internet": "Comcast"
        }

    # Build and send the prompt
    prompt = generate_single_provider_prompt(city, zip_code,internet)
    wrapped_prompt = wrap_with_claude_style_formatting(prompt)

    try:
        content = get_or_generate_llm_output(prompt, generate_fn=lambda: call_openrouter_chat(wrapped_prompt))
        if not content:
            st.warning("⚠️ No content returned from LLM.")
            return {}
    except Exception as e:
        st.error(f"❌ Error querying OpenRouter: {str(e)}")
        return {}

    return extract_and_log_providers(content, section=section)

def get_provider_cache_path(utility: str, city: str, zip_code: str) -> str:
    key = f"{utility}_{city.lower().strip()}_{zip_code}"
    hashed = hashlib.sha256(key.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{utility}_{hashed}.json")

def load_provider_from_cache(utility: str, city: str, zip_code: str) -> str:
    path = get_provider_cache_path(utility, city, zip_code)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def save_provider_to_cache(utility: str, city: str, zip_code: str, content: str):
    path = get_provider_cache_path(utility, city, zip_code)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def auto_reset_refresh_status(utility: str, cooldown_minutes: int = 10):
    """
    Resets the refresh attempt count if cooldown has passed.
    """
    now = time.time()
    timestamps = st.session_state.setdefault("provider_refresh_timestamps", {})
    attempts = st.session_state.setdefault("provider_refresh_attempts", {})

    last_ts = timestamps.get(utility)
    if last_ts is not None and now - last_ts >= cooldown_minutes * 60:
            attempts[utility] = 0
            # ✅ Do NOT zero out the timestamp — it's needed to track future refresh cooldowns
            if st.session_state.get("enable_debug_mode"):
                st.info(f"🔄 Cooldown expired for {utility}. Resetting attempts.")
            log_event(
                event_type="cooldown_reset",
                data={"utility": utility, "section": "utilities"},
                tag="cooldown"
            )


def get_remaining_cooldown(utility: str, cooldown_minutes: int = 10) -> int:
    """
    Returns remaining cooldown in minutes for a given utility.
    """
    now = time.time()
    last_ts = st.session_state.get("provider_refresh_timestamps", {}).get(utility)
    if last_ts is None:
        return 0
    elapsed = now - last_ts
    remaining = max(0, cooldown_minutes * 60 - elapsed)
    return int(remaining // 60)

PROVIDER_CORRECTIONS_PATH = "provider_corrections.json" 

def load_all_user_provider_corrections() -> Dict[str, dict]:
    """
    Loads all saved user corrections to utility provider metadata from disk.

    Returns:
        Dict[str, dict]: A dictionary keyed by utility type ("electricity", etc.)
                         with corresponding provider field overrides.
    """
    if not os.path.exists(PROVIDER_CORRECTIONS_PATH):
        return {}

    try:
        with open(PROVIDER_CORRECTIONS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception as e:
        import streamlit as st
        if st.session_state.get("enable_debug_mode"):
            st.error(f"⚠️ Failed to load provider corrections: {e}")
    return {}

def load_user_corrections(utility: str) -> dict:
    all_corrections = load_all_user_provider_corrections()
    return all_corrections.get(utility, {})

# --- First-time provider fetch (uncorrected) ---
def fetch_utility_providers(section: str, force_refresh_map: Optional[dict] = None):
    """
    Queries the LLM one utility at a time and stores structured results in session_state.
    Uses disk + session cache to avoid redundant calls.
    Logs all extracted data into input_data with shared visibility.
    """
    user_corrections = load_all_user_provider_corrections()

    if force_refresh_map is None:
        force_refresh_map = {}
        
    # 🔍 Look up core location fields
    city = get_answer(key="City", section=section, verbose=True)
    zip_code = get_answer(key="ZIP Code", section=section, verbose=True)
    internet_input = get_answer(key="Internet Provider", section=section, verbose=True)

    utilities = ["electricity", "natural_gas", "water", "internet"]
    results = {}

    for utility in utilities:
        label = utility.replace("_", " ").title()
        force_refresh = force_refresh_map.get(utility, False)

        query_id = f"{utility}|{city}|{zip_code}|{internet_input}"
        query_hash = hashlib.sha256(query_id.encode()).hexdigest()
        session_cache_key = f"cached_llm_response_{utility}_{query_hash}"

        # Auto-reset refresh attempts if cooldown passed
        auto_reset_refresh_status(utility)

        raw_response = "" # ✅ Initialize
        
        # ✅ Try disk cache first
        if not force_refresh:
            raw_response = load_provider_from_cache(utility, city, zip_code)
            if raw_response and st.session_state.get("enable_debug_mode"):
                st.info(f"📁 Using disk cache for `{label}`")
       
       # ✅ If not on disk, check session
        if not raw_response:
            raw_response = st.session_state.get(session_cache_key)
            if st.session_state.get("enable_debug_mode"):
                st.info(f"♻️ Using session cache for `{utility}`")
        
        # ✅ If no cache, call LLM
        if not raw_response:
            prompt = generate_single_provider_prompt(utility, city, zip_code, internet_input)
            try:
                with st.spinner(f"🔍 Fetching {label} provider..."):
                    raw_response = call_openrouter_chat(prompt)
                    # Cache to session and disk
                    st.session_state[session_cache_key] = raw_response
                    save_provider_to_cache(utility, city, zip_code, raw_response)
                log_event("llm_output_raw", {"utility": utility, "prompt": prompt, "response": raw_response, "section": section})
            except Exception as e:
                st.error(f"❌ LLM call failed for {label}: {e}")
                log_event("llm_error", {"utility": utility, "error": str(e), "section": section})
                raw_response = ""
                continue

        # Parse + normalize + log
        parsed = parse_utility_block(raw_response)
        parsed = normalize_provider_fields(parsed)

        # ✅ Step 1: Retry LLM call if core metadata is missing
        if not parsed.get("name") or not parsed.get("description"):
            if st.session_state.get("enable_debug_mode"):
                st.warning(f"♻️ Auto-refreshing `{label}` due to missing fields")
            # Retry logic
            prompt = generate_single_provider_prompt(utility, city, zip_code, internet_input)
            try:
                raw_response = call_openrouter_chat(prompt)
                save_provider_to_cache(utility, city, zip_code, raw_response)
                parsed = parse_utility_block(raw_response)
                log_event("llm_retry_success", {"utility": utility, "section": section})
            except Exception as e:
                st.error(f"❌ Retry failed for {label}: {e}")
                log_event("llm_retry_failed", {"utility": utility, "error": str(e), "section": section})

        # ✅ Step 4: Fallback if LLM retry still fails or metadata is incomplete
        fallback_used = False
        fallback = DEFAULT_PROVIDER_MAP.get(utility, {})

        # Define critical fields that must not be empty
        critical_fields = ["name", "description", "contact_phone", "contact_address"]

        missing_critical = any(not parsed.get(field, "").strip() for field in critical_fields)

        if missing_critical and fallback:
            # Fill in any missing fields from fallback
            for field, val in fallback.items():
                if not parsed.get(field):
                    parsed[field] = val
            fallback_used = True
            log_event(
                event_type="fallback_to_static_provider",
                data={"utility": utility, "missing_fields": critical_fields, "section": section},
                tag="failover"
            )

        # 🛡️ Mark fallback-incomplete providers for forced refresh on next run
        if fallback_used:
            st.session_state.setdefault("force_refresh_map", {})[utility] = True
            if st.session_state.get("enable_debug_mode"):
                st.info(f"⚠️ `{label}` used fallback but is still missing fields — auto-refresh flagged.")

        # Apply overrides
        if utility in user_corrections:
            parsed, name_changed, needs_refresh = apply_provider_overrides(parsed, user_corrections[utility])
            log_event("provider_patch_applied", {"utility": utility, "name_changed": name_changed, "refresh_flagged": needs_refresh, "section": section})

        parsed["timestamp"] = datetime.now(timezone.utc).isoformat()
        parsed["source"] = parsed.get("source", "llm")

        if parsed.get("source") == "fallback":
            save_provider_update_to_disk(city, zip_code, utility, parsed)  # Save static fallback

        # Debug output
        if st.session_state.get("enable_debug_mode"):
            st.markdown(f"#### 🧾 Raw Response: {label}")
            st.code(raw_response)
            st.markdown(f"#### 📦 Parsed Fields: {label}")
            st.json(parsed)

        # ✅ Register inputs
        name = parsed.get("name", "").strip()
        if name:
            st.session_state[f"{utility}_provider"] = name
            register_input_only(
                f"{label} Provider", name,
                section=section, area="home", shared=True
            )

            prefix = f"{label} ({name})"
            register_input_only(f"{prefix} Description", parsed.get("description", ""), section=section, area="home", shared=True)
            register_input_only(f"{prefix} Contact Phone", parsed.get("contact_phone", ""), section=section, area="home", shared=True)
            register_input_only(f"{prefix} Contact Website", parsed.get("contact_website", ""), section=section, area="home", shared=True)
            #register_input_only(f"{prefix} Contact Email", parsed.get("contact_email", ""), section=section, area="home", shared=True)
            register_input_only(f"{prefix} Contact Address", parsed.get("contact_address", ""), section=section, area="home", shared=True)
            register_input_only(f"{prefix} Emergency Steps", parsed.get("emergency_steps", ""), section=section, area="home", shared=True)

        results[utility] = parsed

    # ✅ Store results to session
    st.session_state["utility_providers"] = results
    st.session_state["utility_provider_metadata"] = results

    # ✅ Auto-reset force_refresh_map so it doesn't persist across calls
    if "force_refresh_map" in st.session_state:
        for utility in utilities:
            st.session_state["force_refresh_map"][utility] = False

    return results

def ensure_datetime_strings(obj):
    """Recursively convert datetime objects to ISO strings."""
    if isinstance(obj, dict):
        return {k: ensure_datetime_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_datetime_strings(i) for i in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj

def save_provider_update_to_disk(city: str, zip_code: str, utility: str, data: dict):
    """
    Saves corrected or validated provider info to a JSON file for fallback reuse.
    """
    fallback_dir = "data/provider_fallbacks"
    os.makedirs(fallback_dir, exist_ok=True)

    # Use city_zip_utility as filename
    safe_city = city.lower().replace(" ", "_")
    filename = f"{safe_city}_{zip_code}_{utility}.json"
    filepath = os.path.join(fallback_dir, filename)

    payload = {
        "city": city,
        "zip_code": zip_code,
        "utility": utility,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }

    safe_payload = ensure_datetime_strings(payload)

    with open(filepath, "w") as f:
        json.dump(safe_payload, f, indent=2)

    if st.session_state.get("enable_debug_mode"):
        st.info(f"💾 Saved fallback: `{filename}`")

# utility_provider_ui.py

# --- Editor logic (simplified UI) ---
def render_provider_editor_table_view(utility_key: str, provider_data: dict, section: str = "utilities", simplified_mode: bool = True):
    label = utility_key.replace("_", " ").title()

    # 🔣 Field mapping (UI → data key)
    required_field_map = {
        "Name": "name",
        "Phone": "contact_phone",
        "Website": "contact_website"
    }

    required_ui_labels = list(required_field_map.keys())
    placeholder_vals = {"", "⚠️ not available", "not available", "n/a", "unknown"}

    # Divider and title
    st.markdown("---")
    utility_emojis = {
        "electricity": "⚡",
        "natural_gas": "🔥",
        "water": "💧",
        "internet": "🌐"
    }
    emoji = utility_emojis.get(utility_key, "🔧")
    st.markdown(f"## {emoji} {label} Provider")

    # Get location context
    city = get_answer("City", section)
    zip_code = get_answer("ZIP Code", section)
    disabled = st.session_state.get("utilities_locked", False)

    # Inject AI values into corrected_utility_providers if not already present
    llm_data = st.session_state.get("utility_providers", {}).get(utility_key)
    if llm_data:
        corrected = st.session_state.setdefault("corrected_utility_providers", {})
        current_entry = corrected.setdefault(utility_key, {})
        if not current_entry:
            for ui_label, data_key in required_field_map.items():
                val = llm_data.get(data_key)
                if val:
                    current_entry[data_key] = val
                    current_entry[f"{data_key}_source"] = "ai"
            corrected[utility_key] = current_entry

    # Current provider data (from corrected)
    corrected = st.session_state.get("corrected_utility_providers", {})
    current_entry = corrected.get(utility_key, {})

    name = provider_data.get("name", "")
    phone = provider_data.get("contact_phone", "")
    address = provider_data.get("contact_address", "")
    website = provider_data.get("contact_website", "")
    description = provider_data.get("description", "")
    emergency = provider_data.get("emergency_steps", "")
    tips = provider_data.get("non_emergency_tips", "")

    # Pre-select fields with missing values
    preselected_fields = []
    if not name: preselected_fields.append("Name")
    if not phone: preselected_fields.append("Phone")
    if not address: preselected_fields.append("Address")
    if not website: preselected_fields.append("Website")

    # Check if any prior user correction was entered for this utility
    input_notes = st.session_state.get("provider_input_notes", {}).get(utility_key, {})
    user_requested_fields = input_notes.get("fields", [])

    # Full width layout
    st.markdown("#### ✏️ What would you like the AI to improve?")

    fields_to_update = st.multiselect(
        "Select fields to improve:",
        options=["Name", "Phone", "Address", "Website"],
        default=preselected_fields,
        help="Choose anything that is missing or incorrect.",
        key=f"{section}_{utility_key}_fields"
    )

    notes = st.text_area(
        "Optional clarification (helps the AI respond better)",
        placeholder=(
            "e.g. 'Correct name is AT&T Fiber', "
            "'Phone is missing, please include one', "
            "'Find the nearest Xfinity store in San Jose, CA 95148'"
        ),
        key=f"{section}_{utility_key}_notes"
    )

    # Save user input for later use in LLM prompt
    st.session_state.setdefault("provider_input_notes", {})
    st.session_state["provider_input_notes"][utility_key] = {
        "fields": fields_to_update,
        "notes": notes.strip(),
        "city": city,
        "zip": zip_code,
    }

    if st.button(f"🔁 Update {label}", key=f"{section}_{utility_key}_update_btn", disabled=disabled):
        st.session_state.setdefault("force_refresh_map", {})[utility_key] = True
        st.success(f"{label} will be re-queried.")

    def has_llm_provided_min_required_data(llm_data: dict, required_fields: list = ["name", "contact_phone", "contact_address"]) -> bool:
        for field in required_fields:
            value = llm_data.get(field, "").strip().lower()
            if not value or "⚠️" in value or "not available" in value:
                return False
        return True

    # Show AI response if available
    llm_data = st.session_state.get("utility_providers", {}).get(utility_key)
    if llm_data:
        st.markdown("#### ✅ Suggested Info (from AI)")
        st.markdown(f"""
        - **Name**: {llm_data.get("name", "—")}
        - **Phone**: {llm_data.get("contact_phone", "—")}
        - **Address**: {llm_data.get("contact_address", "—")}
        - **Website**: {llm_data.get("contact_website", "—")}
        """)

        if has_llm_provided_min_required_data(llm_data):
            st.success("✅ The AI provided all required fields. No update may be necessary.")
                # ✅ Inform user but allow manual override
            # ✅ Accept All AI Values button only if data is complete
            if st.button(f"✅ Accept All Values", key=f"{section}_{utility_key}_accept_ai_btn", disabled=disabled):
                for ui_label, data_key in required_field_map.items():
                    val = llm_data.get(data_key)
                    if val:
                        current_entry[data_key] = val
                        current_entry[f"{data_key}_source"] = "ai"

                corrected[utility_key] = current_entry
                register_provider_input(label, current_entry.get("name", ""), section)
                save_provider_update_to_disk(city, zip_code, utility_key, current_entry)
                log_event("provider_ai_accepted", {"utility": utility_key, "section": section}, tag="ai_accept")
                st.success(f"✅ All available AI values saved for {label} provider.")
        else:
            st.warning("⚠️ Some fields from the AI response are still missing. Consider retrying or updating manually.")

    # Merge selected fields into corrected_utility_providers
    corrected = st.session_state.setdefault("corrected_utility_providers", {})
    current_entry = corrected.setdefault(utility_key, {})

    # Merge user-selected fields
    for field in fields_to_update:
        if field == "Name":
            current_entry["name"] = llm_data.get("name", "")
            current_entry["name_source"] = "user"
        elif field == "Phone":
            current_entry["contact_phone"] = llm_data.get("contact_phone", "")
            current_entry["name_source"] = "user"
        elif field == "Address":
            current_entry["contact_address"] = llm_data.get("contact_address", "")
            current_entry["name_source"] = "user"
        elif field == "Website":
            current_entry["contact_website"] = llm_data.get("contact_website", "")
            current_entry["name_source"] = "user"

    # Auto-fill skipped fields if AI provided usable data
    for ui_label, data_key in required_field_map.items():
        if not current_entry.get(data_key) and llm_data.get(data_key):
            current_entry[data_key] = llm_data[data_key]
            current_entry[f"{data_key}_source"] = "user"

    # Save updated data
    corrected[utility_key] = current_entry
    register_provider_input(label, current_entry.get("name", ""), section)
    save_provider_update_to_disk(city, zip_code, utility_key, current_entry)
    log_event("provider_saved_to_disk", {"utility": utility_key, "section": section}, tag="correction")

    # Final status message (only show if update button was clicked)
    if st.session_state.get(f"{section}_{utility_key}_update_btn_clicked", False):
        missing_fields = [
            f.title().replace("_", " ") for f in required_field_map.values()
            if not current_entry.get(f) or "⚠️" in current_entry.get(f).lower() or "not available" in current_entry.get(f).lower()
        ]
        if missing_fields:
            st.warning(
                f"📌 Partial update saved for {label} provider. Still missing required field(s): {', '.join(missing_fields)}."
            )
        else:
            st.success(f"✅ Saved complete info for {label} provider.")

    # Debug block (optional)
    if st.session_state.get("enable_debug_mode"):
        st.markdown("### 🧪 Final Provider Entry")
        st.json(current_entry)

# --- Automatically run LLM updates if queued ---
def is_usable_provider_response(data: dict, placeholder: str = "⚠️ Not Available") -> bool:
    """
    Returns True if at least 2 core fields are non-empty and not placeholders.
    """
    core_fields = ["name", "description", "contact_phone", "contact_address"]
    valid_fields = [
        v for k, v in data.items()
        if k in core_fields and isinstance(v, str) and v.strip() != "" and v.strip() != placeholder
    ]
    return len(valid_fields) >= 2

def sanitize_provider_fields(data: dict, placeholder: str = "⚠️ Not Available") -> dict:
    return {
        k: ("" if isinstance(v, str) and placeholder in v else v)
        for k, v in data.items()
    }

def sanitize_value(val: str) -> Optional[str]:
    return None if isinstance(val, str) and "⚠️" in val else val

# --- Handle queued provider updates (batched) ---
def handle_queued_provider_updates(section: str = "utilities"):
    import re

    force_refresh_map = st.session_state.get("force_refresh_map", {})
    if not force_refresh_map:
        return

    city = get_answer("City", section)
    zip_code = get_answer("ZIP Code", section)
    internet = get_answer("Internet Provider", section)

    prompts = []
    utility_keys = []

    for utility_key, should_refresh in force_refresh_map.items():
        if not should_refresh:
            continue
        
        label = utility_key.replace("_", " ").title()
        st.info(f"🔁 Updating: {label}")

        provider_data = st.session_state.get("utility_providers", {}).get(utility_key, {})
        corrected = st.session_state.get("corrected_utility_providers", {}).get(utility_key, {})

        auto_reset_refresh_status(utility_key)
        st.session_state["provider_refresh_attempts"].setdefault(utility_key, 0)
        st.session_state["provider_refresh_attempts"][utility_key] += 1
        st.session_state["provider_refresh_timestamps"][utility_key] = time.time()

        requested_fields = corrected.get("requested_fields", [])
        notes = corrected.get("input_notes", "")
        name = corrected.get("name", "")
        phone = corrected.get("contact_phone", "")

        name_correction_phrases = ["name is wrong", "name is incorrect", "correct name is", "should be called"]
        for phrase in name_correction_phrases:
            if phrase in notes.lower() and "Name" not in requested_fields:
                requested_fields.append("Name")
                break

        phone_correction_phrases = ["phone is missing", "missing phone", "correct phone is", "update phone to"]
        for phrase in phone_correction_phrases:
            if phrase in notes.lower() and "Phone" not in requested_fields:
                requested_fields.append("Phone")
                break

        # Fuzzy match correction notes
        name_match = re.search(r"(?:correct name is|should be called)[:\s]+(.+?)([\.;\n]|$)", notes, re.IGNORECASE)
        phone_match = re.search(r"(?:correct phone is|update phone to)[:\s]+([\d\-\(\)\s+]+)", notes, re.IGNORECASE)
        extracted_name = name_match.group(1).strip() if name_match else name
        extracted_phone = phone_match.group(1).strip() if phone_match else phone

        # Sanitize values before sending to the LLM
        # Apply before prompt generation
        name = sanitize_value(name)
        phone = sanitize_value(phone)

        # ✅ Include all corrected fields into notes context
        # This ensures prompt logic can access updated name/phone/address directly
        if "corrected_fields" not in st.session_state["provider_input_notes"][utility_key]:
            st.session_state["provider_input_notes"][utility_key]["corrected_fields"] = {}

        for field in ["name", "contact_phone", "contact_address", "contact_website"]:
            val = corrected.get(field)
            if val:
                st.session_state["provider_input_notes"][utility_key]["corrected_fields"][field] = val

        prompt = generate_corrected_provider_prompt(
            utility_key=utility_key,
            city=city,
            zip_code=zip_code,
            user_name=extracted_name,
            user_phone=extracted_phone,
            fields=requested_fields,
            notes=notes
        )

        prompts.append((utility_key, prompt))
        utility_keys.append(utility_key)

    if not prompts:
        st.info("ℹ️ No LLM calls made. No providers were queued for update.")
        return

    st.info(f"🔁 Updating: {', '.join([k.replace('_', ' ').title() for k in utility_keys])}")

    if st.session_state.get("enable_debug_mode"):
        st.markdown("### ⚙️ Batched LLM Prompts")
        for utility_key, prompt in prompts:
            st.markdown(f"#### `{utility_key}`")
            st.code(prompt)

    try:
        with st.spinner("🔍 Contacting LLM for multiple providers..."):
            combined_prompt = "\n\n".join(p[1] for p in prompts)
            combined_response = call_openrouter_chat(combined_prompt)

        if st.session_state.get("enable_debug_mode"):
            st.markdown("### 📬 Raw Combined LLM Response")
            st.code(combined_response)
            st.session_state["raw_combined_llm_response"] = combined_response

        response_blocks = combined_response.split("## ")
        found_utilities = set()

        def strip_emojis(text):
            return re.sub(r"[\U00010000-\U0010ffff\u2600-\u26FF\u2700-\u27BF\u1F300-\u1F64F\u1F680-\u1F6FF]+", "", text)

        for block in response_blocks:
            block_full = "## " + block.strip()
            matched = False
            for utility_key in utility_keys:
                label_name = utility_key.replace("_", " ").lower()
                if strip_emojis(label_name) in strip_emojis(block.lower()):
                    matched = True
                    parsed = parse_utility_block(block_full)
                    normalized = normalize_provider_fields(parsed)
                    minimal_data = {
                        k: normalized.get(k)
                        for k in [
                            "name", "contact_phone", "contact_address", "contact_website",
                            "description", "emergency_steps", "non_emergency_tips"
                        ]
                        if k in normalized
                    }
                    minimal_data["timestamp"] = datetime.now(timezone.utc).isoformat()
                    minimal_data["source"] = "llm_batch"

                    # ✅ Optional: sanitize "⚠️ Not Available"
                    minimal_data = sanitize_provider_fields(minimal_data)

                    # ✅ Skip saving if all fields are still unusable
                    if not is_usable_provider_response(minimal_data):
                        if st.session_state.get("enable_debug_mode"):
                            st.warning(f"⚠️ Skipping update for `{utility_key}` due to unusable LLM response.")
                        continue

                    st.session_state.setdefault("corrected_utility_providers", {})[utility_key] = minimal_data
                    st.session_state.setdefault("utility_providers", {})[utility_key] = minimal_data
                    save_provider_update_to_disk(city, zip_code, utility_key, minimal_data)
                    DEFAULT_PROVIDER_MAP[utility_key] = minimal_data

                    log_event(
                        event_type="provider_refreshed",
                        data={"utility": utility_key, "source": "llm_batch", "section": section},
                        tag="refresh"
                    )

                    st.success(f"✅ Updated info for: {label}")
                    if st.session_state.get("enable_debug_mode"):
                        st.markdown(f"### 🧩 Parsed `{utility_key}` Block")
                        st.json(parsed)
                        st.markdown(f"### 🧽 Normalized `{utility_key}` Block")
                        st.json(minimal_data)

                    found_utilities.add(utility_key)
                    break

            if not matched:
                if st.session_state.get("enable_debug_mode"):
                    st.warning(f"⚠️ Unmatched block: {block[:60]}...")

        if found_utilities:
            st.success(f"✅ Updated info for: {', '.join(found_utilities)}")

    except Exception as e:
            st.error(f"❌ Failed to update {label}: {e}")
            log_event("llm_error", {"utility": utility_key, "error": str(e), "section": section})

    # Always reset all refresh triggers
    for key in force_refresh_map:
        force_refresh_map[key] = False

# --- Helper to retrieve confirmed provider data ---
def get_confirmed_provider(utility_key: str) -> dict:
    # ✅ If finalized confirmed provider exists, always show it
    confirmed = st.session_state.get("confirmed_utility_providers", {}).get(utility_key)
    if confirmed:
        return confirmed

    # ✅ Use corrected version ONLY if it has real LLM output (e.g., a name and phone)
    corrected = st.session_state.get("corrected_utility_providers", {}).get(utility_key, {})
    real_keys = {
        "name", "contact_phone", "contact_address", "contact_website",
        "description", "emergency_steps", "non_emergency_tips"
    }
    has_real_data = any(k in corrected and corrected[k] and "⚠️" not in str(corrected[k]) for k in real_keys)
    if has_real_data:
        return corrected

    # ✅ Fallback to LLM-provided or default static map (original unedited output)
    return st.session_state.get("utility_providers", {}).get(utility_key, {})


# --- Helper to check if all providers are confirmed ---
def all_providers_confirmed(required_keys=["electricity", "natural_gas", "water", "internet"]) -> bool:
    confirmed = st.session_state.get("confirmed_utility_providers", {})
    for key in required_keys:
        if not confirmed.get(key, {}).get("confirmed", False):
            return False
    return True

# --- Table wrapper ---
def render_provider_table(utility_key: str, section: str = "utilities"):
    label = utility_key.replace("_", " ").title()
    icon = {"electricity": "⚡", "natural_gas": "🔥", "water": "💧", "internet": "🌐"}.get(utility_key, "🔌")
    provider_data = get_confirmed_provider(utility_key)

    # 🧾 Section heading for each provider
    st.markdown(f"## {icon} {label} Provider")
    
    # 🔄 Display table with two columns: contact info + editor
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📋 Retrieved Info")
        display_provider_contact_info(
            provider_data,
            visible_fields=["name", "contact_phone", "contact_address", "contact_website"])

    with col2:
        st.markdown("### ✏️ Review & Edit")
        render_provider_editor_table_view(utility_key, provider_data, section=section)

# --- Admin override tools ---
def render_admin_refresh_override(section: str = "utilities"):
    st.markdown("---")
    with st.expander("🔧 Developer Tools: Refresh Limit Overrides", expanded=False):
        attempts = st.session_state.get("provider_refresh_attempts", {})
        cooldowns = {k: get_remaining_cooldown(k) for k in attempts}

        if cooldowns:
            st.markdown("### ⏱️ Current Cooldowns")
            for utility, time_left in cooldowns.items():
                st.markdown(f"- **{utility.title()}**: `{time_left} minutes remaining`")

        if st.button("🔪 Override Refresh Limits Now"):
            for utility in attempts:
                st.session_state["provider_refresh_attempts"][utility] = 0
                st.session_state["provider_refresh_timestamps"][utility] = 0

            log_event("override_refresh_limits", {"utilities_reset": list(attempts.keys()), "section": section}, tag="dev_override")
            st.success("🔁 All refresh attempt limits have been reset.")


# --- Section jump UI ---
def render_section_jump_buttons():
    st.markdown("### 🔽 Jump to a Provider Section")
    col1, col2, col3, col4 = st.columns(4)
    if col1.button("⚡ Electricity"): st.session_state["scroll_target"] = "electricity"
    if col2.button("🔥 Natural Gas"): st.session_state["scroll_target"] = "natural_gas"
    if col3.button("💧 Water"): st.session_state["scroll_target"] = "water"
    if col4.button("🌐 Internet"): st.session_state["scroll_target"] = "internet"

# --- Unified table view controller ---
def render_all_provider_tables(section: str = "utilities"):
    handle_queued_provider_updates(section)

    st.markdown("## 📇 Review and Edit Utility Providers")
    for utility_key in ["electricity", "natural_gas", "water", "internet"]:
        render_provider_table(utility_key, section=section)

    st.markdown("---")
    st.markdown("### ✅ Finalize All Provider Info")

    if st.session_state.get("enable_debug_mode"):
        render_admin_refresh_override(section=section)

    if st.button("✅ Confirm All Utility Info"):
        required_utilities = ["electricity", "natural_gas", "water", "internet"]
        required_fields = ["name", "contact_phone", "contact_address"]

        missing_fields = {}
        corrected = st.session_state.get("corrected_utility_providers", {})

        for key in required_utilities:
            provider = corrected.get(key, {})
            missing = [field for field in required_fields if not provider.get(field)]
            if missing:
                missing_fields[key] = missing

        if missing_fields:
            st.warning("⚠️ Missing required info:")
            for utility, fields in missing_fields.items():
                st.markdown(f"- **{utility.title()}**: missing {', '.join(fields)}")
        else:
            st.session_state["confirmed_utility_providers"] = corrected
            st.session_state["utility_info_locked"] = True
            st.success("🔐 Utility provider info confirmed and saved.")

            if st.session_state.get("enable_debug_mode"):
                st.markdown("### 🧪 Debug: Saved Providers")
                st.write("🔌 Session Provider Data:", st.session_state.get("utility_providers"))
                st.write("🔌 Electricity:", st.session_state.get("electricity_provider"))
                st.write("🔥 Natural Gas:", st.session_state.get("natural_gas_provider"))
                st.write("💧 Water:", st.session_state.get("water_provider"))

# --- Main Function Start ---
def utilities():
    section = "utilities"
    generate_key = f"generate_runbook_{section}"  # Define it early

    st.subheader("Let's gather some information. Please enter your details:")

    # Step 1: Input collection
    city, zip_code, internet_provider = get_utilities_inputs(section)

    if st.session_state.get("enable_debug_mode"):
        st.markdown("### 🧪 Debug: get_utilities_inputs ")
        st.write("City:", city)
        st.write("ZIP Code:", zip_code)
        st.write("Internet Provider:", internet_provider)

    # Step 2: Fetch utility providers
    if st.button("Find My Utility Providers"):
        with st.spinner("Querying providers from OpenRouter..."):
            force_refresh_map = st.session_state.get("force_refresh_map", {})
            fetch_utility_providers(section=section, force_refresh_map=force_refresh_map)
            st.session_state["show_provider_corrections"] = True
            st.success("Providers stored in session state!")

    if st.session_state.get("enable_debug_mode"):
        st.markdown("### 🧪 Debug: Find Utility Providers")
        st.write("🔌 Session Provider Data:", st.session_state.get("utility_providers"))
        st.write("🔌 Electricity:", st.session_state.get("electricity_provider"))
        st.write("🔥 Natural Gas:", st.session_state.get("natural_gas_provider"))
        st.write("💧 Water:", st.session_state.get("water_provider"))
        st.write("🌐 Internet:", st.session_state.get("internet_provider"))
        st.write("🤖 Using Model:", st.session_state.get("llm_model", "openai/gpt-4o:online"))
        if "llm_usage_log" in st.session_state:
            latest = st.session_state["llm_usage_log"][-1]
            st.markdown("### 📊 LLM Token Usage")
            st.json(latest)

        if "last_web_citations" in st.session_state:
            st.markdown("### 🌐 Web Search Citations (OpenRouter)")
            for cite in st.session_state["last_web_citations"]:
                url = cite["url_citation"].get("url", "")
                domain = url.split("/")[2] if "://" in url else url
                st.markdown(f"- [{domain}]({url})")

    # ✅ Auto-jump to Step 6 after rerun
    if st.session_state.pop("trigger_refresh_and_preview", False):
        st.session_state["utility_info_locked"] = True


    with st.expander("🧪 Test Corrected Provider Prompt"):
        test_prompt = generate_corrected_provider_prompt(
            utility_key="electicity",
            city="San Jose",
            zip_code="95148",
            user_name="Pacific Gas & Electric Company PG&E",
            fields=["Name", "Phone", "Address"],
            notes=""
        )

        st.markdown("### 📤 Prompt to Send")
        st.code(test_prompt, language="markdown")

        if st.button("🧠 Send Prompt to LLM"):
            with st.spinner("📡 Sending to LLM..."):
                try:
                    response = call_openrouter_chat(test_prompt)
                    st.success("✅ Response received!")
                    st.markdown("### 📬 LLM Response")
                    st.markdown(response)
                except Exception as e:
                    st.error(f"❌ LLM call failed: {e}")


    # Step 3-5: Display and confirm LLM provider results and reward
    if st.session_state.get("show_provider_corrections"):
        render_all_provider_tables(section=section)

    # Step 6: 🎉 Reward + Download runbook (only shown after confirmation)
    if st.session_state.get("utility_info_locked"):
        st.subheader("📄 Download Utility Provider Runbook")

        providers = st.session_state.get("utility_providers", {})
        if not providers:
            st.info("⚠️ No utility provider data available.")
            return

        markdown_str = format_provider_markdown(providers)
        docx_bytes = export_provider_docx(providers)

        if docx_bytes and markdown_str:
            st.session_state["utility_docx"] = docx_bytes
            st.session_state["utility_markdown"] = markdown_str

        render_runbook_section_output(
            markdown_str=st.session_state.get("utility_markdown"),
            docx_bytes_io=st.session_state.get("utility_docx"),
            title="Utility Providers",
            filename_prefix="utility_providers",
            expand_preview=False
        )



                        
