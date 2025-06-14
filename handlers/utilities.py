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
from datetime import datetime, timedelta
import re
import time
import io
import hashlib
import uuid
import json
from pathlib import Path
from utils.preview_helpers import get_active_section_label
from utils.data_helpers import (
    normalize_provider_fields,
    register_provider_input,
    register_task_input, 
    get_answer, 
    extract_and_log_providers,
    parse_utility_block,
    register_input_only,
    apply_provider_overrides
)
#from utils.runbook_generator_helpers import generate_docx_from_prompt_blocks, maybe_render_download, maybe_generate_runbook
from utils.debug_utils import debug_all_sections_input_capture_with_summary, reset_all_session_state
from prompts.templates import generate_single_provider_prompt, wrap_with_claude_style_formatting
from utils.common_helpers import get_schedule_placeholder_mapping
from utils.llm_cache_utils import get_or_generate_llm_output
from utils.llm_helpers import call_openrouter_chat
from utils.preview_helpers import render_provider_contacts
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
    if last_ts is not None:
        if now - last_ts >= cooldown_minutes * 60:
            attempts[utility] = 0
            timestamps[utility] = 0
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

        if force_refresh:
            if not is_refresh_allowed(utility):
                cooldown = get_remaining_cooldown(utility)
                st.warning(f"⛔ Refresh limit reached for {label}. Try again later.  ⏳ Try again in {cooldown} minutes.")
                log_event(
                    event_type="refresh_blocked",
                    data={
                        "utility": utility,
                        "label": label,
                        "attempts": st.session_state["provider_refresh_attempts"].get(utility, 0),
                        "cooldown_remaining_min": cooldown,
                        "section": section
                    },
                    tag="cooldown"
                )
                continue

            st.session_state["provider_refresh_attempts"].setdefault(utility, 0)
            st.session_state["provider_refresh_attempts"][utility] += 1
            st.session_state["provider_refresh_timestamps"][utility] = time.time()

            log_event(
                event_type="refresh_triggered",
                data={
                    "utility": utility,
                    "label": label,
                    "attempt_num": st.session_state["provider_refresh_attempts"][utility],
                    "section": section
                },
                tag="refresh"
            )

        raw_response = ""  # ✅ Initialize

        # ✅ Try disk cache first
        if not force_refresh:
            raw_response = load_provider_from_cache(utility, city, zip_code)
            if raw_response and st.session_state.get("enable_debug_mode"):
                st.info(f"📁 Using disk cache for `{label}`")

        # ✅ If not on disk, check session
        if not raw_response and session_cache_key in st.session_state:
            raw_response = st.session_state[session_cache_key]
            if st.session_state.get("enable_debug_mode"):
                st.info(f"♻️ Using session cache for `{utility}`")

        # ✅ If no cache, call LLM
        if not raw_response:
            prompt = generate_single_provider_prompt(utility, city, zip_code, internet_input)

            if st.session_state.get("enable_debug_mode"):
                st.markdown(f"### ⚙️ Calling LLM for `{label}`")
                st.code(prompt)

            try:
                with st.spinner(f"🔍 Looking up {label} provider..."):
                    raw_response = call_openrouter_chat(prompt)
                    # Cache to session and disk
                    st.session_state[session_cache_key] = raw_response
                    save_provider_to_cache(utility, city, zip_code, raw_response)
            except Exception as e:
                st.error(f"❌ Error querying {label}: {e}")
                raw_response = ""

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
            except Exception as e:
                st.error(f"❌ Retry failed for {label}: {e}")

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
            if "force_refresh_map" not in st.session_state:
                st.session_state["force_refresh_map"] = {}
            st.session_state["force_refresh_map"][utility] = True
            
        if fallback_used and st.session_state.get("enable_debug_mode"):
            st.info(f"⚠️ `{label}` used fallback but is still missing fields — auto-refresh flagged.")


        # ✅ Inject user corrections from disk (if available)
        if utility in user_corrections:
            parsed, name_changed, needs_refresh = apply_provider_overrides(parsed, user_corrections[utility])
            
            log_event(
                event_type="provider_patch_applied",
                data={
                    "utility": utility, 
                    "section": section, 
                    "source": "user_disk",
                    "name_changed": name_changed,
                    "refresh_flagged": needs_refresh
                    },
                tag="correction"
            )

        # 🔧 Pull prior values if available — otherwise use safe defaults
        prior = st.session_state.get("utility_providers", {}).get(utility, {})
        
        # 🔄 Merge emergency + non-emergency info (deduplicating if needed)
        merged = prior.copy()
        # Emergency steps are usually a single field
        merged["emergency_steps"] = parsed.get("emergency_steps", prior.get("emergency_steps", ""))
        
        # Deduplicate non-emergency tips
        tips_parsed = parsed.get("non_emergency_tips", "").strip()
        tips_prior = prior.get("non_emergency_tips", "").strip()
        if tips_parsed and tips_parsed != tips_prior:
            merged["non_emergency_tips"] = tips_parsed
        else:
            merged["non_emergency_tips"] = tips_prior or tips_parsed

        # Contact & metadata
        for field in ["name", "description", "contact_phone", "contact_address", "contact_website"]:
            merged[field] = parsed.get(field, prior.get(field, ""))

        # 🔎 Validation and logging for fallback metadata fields
        fallback_fields = ["name", "description", "contact_phone", "contact_address", "contact_website"]
        for field in fallback_fields:
            parsed_val = parsed.get(field, "").strip()
            prior_val = prior.get(field, "").strip()
            merged_val = merged.get(field, "").strip()

            # If current parsed field is empty but prior value is used, it's a fallback
            if not parsed_val and prior_val:
                if st.session_state.get("enable_debug_mode"):
                    st.warning(f"⚠️ Using fallback for `{field}` from prior session for `{label}`")
                log_event(
                    event_type="metadata_fallback_used",
                    data={
                        "utility": utility,
                        "label": label,
                        "field": field,
                        "fallback_value": prior_val,
                        "section": section
                    },
                    tag="fallback"
                )

        results[utility] = merged

        if parsed.get("source") == "fallback":
            save_provider_update_to_disk(utility, parsed)  # Save static fallback

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

    # ✅ Store results to session
    st.session_state["utility_providers"] = results
    st.session_state["utility_provider_metadata"] = results

    # ✅ Auto-reset force_refresh_map so it doesn't persist across calls
    if "force_refresh_map" in st.session_state:
        for utility in utilities:
            st.session_state["force_refresh_map"][utility] = False

    return results

def save_provider_update_to_disk(city: str, zip_code: str, utility: str, data: dict):
    """
    Saves corrected or validated provider info to a JSON file for fallback reuse.

    Args:
        city (str): City name (e.g., "San Jose")
        zip_code (str): ZIP code (e.g., "95148")
        utility (str): Utility type ("electricity", "water", etc.)
        data (dict): Provider info (name, description, contact details, etc.)
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
        "updated_at": datetime.utcnow().isoformat(),
        "data": data,
    }

    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2)

    if "enable_debug_mode" in st.session_state and st.session_state["enable_debug_mode"]:
        st.info(f"💾 Saved fallback: `{filename}`")

def render_provider_correction_and_refresh(section: str = "utilities"):
    # Step 1: Ensure refresh flags exist
    if "force_refresh_map" not in st.session_state:
        st.session_state["force_refresh_map"] = {
            "electricity": False,
            "natural_gas": False,
            "water": False,
            "internet": False,
        }
    # ✅ Step 2: Ensure editable provider state is initialized
    if "corrected_utility_providers" not in st.session_state:
        st.session_state["corrected_utility_providers"] = st.session_state.get("utility_providers", {}).copy()

    # Step 3: Display editable correction form and collect updates
    current_results = st.session_state.get("utility_providers", {})
    updated = get_corrected_providers(current_results, section=section)

    # Step 4: Check which utilities need a refresh
    force_refresh_map = st.session_state["force_refresh_map"]
    queued = [k for k, v in force_refresh_map.items() if v]

    if queued:
        st.markdown("### 🔄 Queued for Update:")
        st.write(", ".join(label.replace("_", " ").title() for label in queued))

        if st.button("♻️ Update Queued Providers Now"):
            refreshed = fetch_utility_providers(section=section, force_refresh_map=force_refresh_map)

            # Merge new emergency/non-emergency info back into the provider blocks
            for utility in queued:
                updated_block = updated.get(utility) or st.session_state.get("utility_providers", {}).get(utility, {}).copy()
                updated_block["emergency_steps"] = refreshed.get(utility, {}).get("emergency_steps", updated_block.get("emergency_steps", ""))
                updated_block["non_emergency_tips"] = refreshed.get(utility, {}).get("non_emergency_tips", updated_block.get("non_emergency_tips", ""))
                force_refresh_map[utility] = False  # Reset flag
                updated[utility] = updated_block  # Store merged block

                if st.session_state.get("enable_debug_mode"):
                    st.markdown(f"### 🔍 Merged Updated Block for {utility}")
                    st.json(updated_block)

        st.success("✅ Updated provider info successfully. Please review below.")


    with st.expander("🔧 Debug / Developer Overrides", expanded=False):
        if st.button("🧪 Override Refresh Limit"):
            for utility in st.session_state.get("provider_refresh_attempts", {}):
                st.session_state["provider_refresh_attempts"][utility] = 0
                st.session_state["provider_refresh_timestamps"][utility] = 0
            log_event(
                event_type="override_refresh_limits",
                data={
                    "utilities_reset": list(st.session_state.get("provider_refresh_attempts", {}).keys()),
                    "section": section
                },
                tag="dev_override"
            )
            st.success("🔁 Refresh attempt limits reset.")
    
    # Step 5: Re-render the updated form using refreshed data
    st.session_state["corrected_utility_providers"] = updated
    updated = get_corrected_providers(updated, section=section)  # <- Re-render with updated info

def get_corrected_providers(results: dict, section: str) -> dict:
    """
    Allows user to review and optionally correct provider data.
    Emergency Steps are view-only.
    Prevents overwrites unless explicitly confirmed.
    Displays refresh cooldown state.
    """
    updated = {}
    label_to_key = {
        "Electricity": "electricity",
        "Natural Gas": "natural_gas",
        "Water": "water",
        "Internet": "internet"
    }
    # Retrieve city and zip from session
    city = get_answer(key="City", section=section)
    zip_code = get_answer(key="ZIP Code", section=section)

    for utility, current in results.items():
        label = utility.replace("_", " ").title()
        name = current.get("name", "")
        phone = current.get("contact_phone", "")
        website = current.get("contact_website", "")
        #email = current.get("contact_email", "")
        address = current.get("contact_address", "")
        description = current.get("description", "")
        emergency = current.get("emergency_steps", "")
        tips = current.get("non_emergency_tips", "")

        # Cooldown tracking
        auto_reset_refresh_status(utility)
        attempts = st.session_state.get("provider_refresh_attempts", {}).get(utility, 0)
        cooldown = get_remaining_cooldown(utility)
        cooldown_active = attempts >= 3 and cooldown > 0

        with st.expander(f"🔧 Validate or Update {label} Provider", expanded=False):
            st.markdown(f"### 🛠️ {label} Provider")
            st.markdown(f"**Current Provider**: {name or '⚠️ Not Available'}")

            st.markdown(f"⏱️ Update Attempts: `{attempts}`  |  Cooldown: `{cooldown} min remaining`")
            st.markdown("🛈 Use **Update** to retrieve missing info or correct outdated details. Use **Confirm** after reviewing or editing provider details to save them..")

            if cooldown_active:
                st.warning(f"⛔ You must wait before updating this provider again.")

            # 🔁 Update Button
            update_clicked = st.button(f"🔁 Update {label}", key=f"update_{utility}")
            if update_clicked:
                if cooldown_active:
                    st.warning(f"⏳ Cannot update {label} yet. Please wait.")
                else:
                    st.session_state.setdefault("force_refresh_map", {})[utility] = True
                    st.success(f"{label} provider will be re-queried with updated info.")

            # ✅ Visual feedback if update was triggered earlier
            if st.session_state.get("force_refresh_map", {}).get(utility):
                st.info("🔄 Update requested. Changes will apply after refresh.")

            # Editable fields
            
            # Name
            name_check = st.checkbox(f"✏️ Correct Provider Name ({name})", value=False, key=f"{utility}_name_check")
            name_input = st.text_input(f"{label} Provider Name", value=name, disabled=not name_check, key=f"{utility}_name")

            # Phone            
            phone_check = st.checkbox(f"✏️ Correct Phone", value=False, key=f"{utility}_phone_check")
            phone_input = st.text_input("Phone", value=phone, disabled=not phone_check, key=f"{utility}_phone")

            # Email
            #correct_email = st.checkbox(f"✏️ Correct Email", value=False, key=f"{key}_email_check")
            #email_input = st.text_input("Email", value=email, disabled=not correct_email, key=f"{key}_email")

            # Address
            address_check = st.checkbox(f"✏️ Correct Address", value=False, key=f"{utility}_address_check")
            address_input = st.text_area("Address", value=address, disabled=not address_check, key=f"{utility}_address")

            # Website
            website_check = st.checkbox(f"✏️ Correct Website", value=False, key=f"{utility}_website_check")
            website_input = st.text_input("Website", value=website, disabled=not website_check, key=f"{utility}_website")

            # Description
            desc_check = st.checkbox(f"✏️ Correct Description", value=False, key=f"{utility}_desc_check")
            desc_input = st.text_area("Provider Description", value=description, disabled=not desc_check, key=f"{utility}_desc")

            # Read-only fields
            st.markdown(f"🚨 **Emergency Steps (Read-Only):**\n{emergency or '—'}")
            if tips and tips != "⚠️ Not Available":
                st.markdown(f"💡 **Non-Emergency Tips:**\n{tips}")

            # ✅ Require confirmation before applying
            confirmed_key = f"{utility}_confirmed"
            confirmed = st.checkbox("✅ I have reviewed and confirmed this provider info", key=confirmed_key)

            # Determine final values 
            final_name = name_input if name_check else name
            final_block = {
                "name": final_name, 
                "contact_phone": phone_input if phone_check else phone,
                #"contact_email": email_input if correct_email else email,
                "contact_address": address_input if address_check else address,
                "contact_website": website_input if website_check else website,
                "description": desc_input if desc_check else description,
                "emergency_steps": emergency,  # do not change
                "non_emergency_tips": tips,
                "confirmed": confirmed
            }

            updated[utility] = final_block

            # ✅ Update session state for each corrected field
            if name_check:
                st.session_state[f"{utility}_provider"] = final_name
                register_provider_input(label, final_name, section)

            # Auto-flag update if name changed AND metadata fields are missing
            name_changed = name_check and name_input != name
            meta_missing = not (phone or address or website)
            if name_changed and meta_missing:
                st.session_state.setdefault("force_refresh_map", {})[utility] = True
                st.info("🔁 Auto-update triggered due to name change and missing info.")
                if st.session_state.get("enable_debug_mode"):
                    st.warning(f"⚠️ Name changed + metadata missing → update triggered for {label}")
           
            # ✅ Save correction to disk only if confirmed
            if confirmed:
                save_provider_update_to_disk(city, zip_code, utility, final_block)
                log_event(
                    event_type="provider_saved_to_disk",
                    data={"utility": utility, "section": section},
                    tag="correction"
                )
                # ✅ Update session provider entry immediately
                st.session_state.setdefault["utility_providers", {}][utility] = final_block
                st.session_state.setdefault["utility_provider_metadata",{}][utility] = final_block

                if st.session_state.get("enable_debug_mode"):
                    st.markdown(f"### 🧩 Updated Entry for `{label}`")
                    st.json(final_block)

    return updated

def update_session_state_with_providers(updated):
    st.session_state["utility_providers"] = updated
    for key, value in updated.items():
        st.session_state[f"{key}_provider"] = value
        if st.session_state.get("enable_debug_mode"):
            st.markdown("### 🧪 Debug: update_session_state_with_providers")
            st.write("🔌 Session Provider Data:", st.session_state.get("utility_providers"))
            st.write("🔌 Electricity:", st.session_state.get("electricity_provider"))
            st.write("🔥 Natural Gas:", st.session_state.get("natural_gas_provider"))
            st.write("💧 Water:", st.session_state.get("water_provider"))

# --- Main Function Start ---
def utilities():
    section = "utilities"
    generate_key = f"generate_runbook_{section}"  # Define it early

    #st.markdown(f"### Currently Viewing: {get_active_section_label(section)}")
    #switch_section(section)

    st.subheader("Let's gather some information. Please enter your details:")
   
# Step 1: Input collection
    # ✅ Call the function at runtime
    city, zip_code, internet_provider = get_utilities_inputs(section)

    if st.session_state.get("enable_debug_mode"): # DEBUG: get_utilities_inputs()
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

# Step 3: Display resulting LLM retrieved Utility Providers
    if st.session_state.get("show_provider_corrections"):
        st.markdown("### 📇 Retrieved Utility Providers")
        # 🧱 Visual contact display (read-only)
        render_provider_contacts(section=section)
        # ✅ Step 2: Allow corrections + refresh
        render_provider_correction_and_refresh(section=section)

# Step 4: Save Utility Providers (with validation)
        if st.button("✅ Confirm All Utility Info"):
            required_utilities = ["electricity", "natural_gas", "water", "internet"]
            required_fields = ["name", "contact_phone", "contact_address"] #Removed "contact_email"

            missing_fields = {}

            for key in required_utilities:
                provider = st.session_state.get("utility_providers", {}).get(key, {})
                missing = [field for field in required_fields if not provider.get(field)]
                if missing:
                    missing_fields[key] = missing

            if missing_fields:
                st.warning("⚠️ Missing required info:")
                for utility, fields in missing_fields.items():
                    st.markdown(f"- **{utility.title()}**: missing {', '.join(fields)}")
            else:
                corrected = st.session_state.get("corrected_utility_providers", {})
                st.session_state["confirmed_utility_providers"] = corrected
                st.session_state["utility_info_locked"] = True
                st.success("🔒 Utility provider info confirmed and saved.")

                # ✅ Show output in debug mode
                if st.session_state.get("enable_debug_mode"):
                    st.markdown("### 🧪 Debug: Saved Providers")
                    st.write("🔌 Session Provider Data:", st.session_state.get("utility_providers"))
                    st.write("🔌 Electricity:", st.session_state.get("electricity_provider"))
                    st.write("🔥 Natural Gas:", st.session_state.get("natural_gas_provider"))
                    st.write("💧 Water:", st.session_state.get("water_provider"))

    # Step 5: Return Reward
                if st.session_state.get("utility_info_locked"):
                    st.subheader("🎉 Reward")
                    st.markdown("You've successfully confirmed all your utility providers! ✅")

    # Step 6: Display Raw LLM Output + Allow Download
                    providers = st.session_state.get("utility_providers", {})
                    if not providers:
                        st.info("⚠️ No utility provider data available.")
                    else:
                        markdown_str = format_provider_markdown(providers)
                        docx_bytes = export_provider_docx(providers)

                        if docx_bytes and markdown_str:
                            st.session_state["utility_docx"] = docx_bytes
                            st.session_state["utility_markdown"] = markdown_str

                        render_runbook_section_output(
                            markdown_str=st.session_state.get("utility_markdown"),
                            docx_bytes_io=st.session_state.get("utility_docx") ,
                            title="Utility Providers",
                            filename_prefix="utility_providers",
                            expand_preview=False #Optional: set True open by default
                        )


                        
