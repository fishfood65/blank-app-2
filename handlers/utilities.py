### Leve 1 - Home
#from utils.prompt_block_utils import generate_all_prompt_blocks
from typing import Optional
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
from utils.preview_helpers import get_active_section_label
from utils.data_helpers import (
    normalize_provider_fields,
    register_provider_input,
    register_task_input, 
    get_answer, 
    extract_and_log_providers,
    parse_utility_block,
    register_input_only,
)
#from utils.runbook_generator_helpers import generate_docx_from_prompt_blocks, maybe_render_download, maybe_generate_runbook
from utils.debug_utils import debug_all_sections_input_capture_with_summary, reset_all_session_state
from prompts.templates import generate_single_provider_prompt, wrap_with_claude_style_formatting
from utils.common_helpers import get_schedule_placeholder_mapping
from utils.llm_cache_utils import get_or_generate_llm_output
from utils.llm_helpers import call_openrouter_chat
from utils.preview_helpers import render_provider_contacts, render_provider_correction_and_refresh
from utils.docx_helpers import export_provider_docx, format_provider_markdown, render_runbook_section_output
from event_logger import log_event
from llm_helpers import is_refresh_allowed

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

def fetch_utility_providers(section: str, force_refresh_map: Optional[dict] = None):
    """
    Queries the LLM one utility at a time and stores structured results in session_state.
    Uses disk + session cache to avoid redundant calls.
    Logs all extracted data into input_data with shared visibility.
    """
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
                st.warning(f"⛔ Refresh limit reached for {label}. Try again later.")
                cooldown = get_remaining_cooldown(utility)
                st.info(f"⏳ Try again in {cooldown} minutes.")
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
        prior = st.session_state.get("utility_providers", {}).get(utility, {})
        merged = prior.copy()
        merged["emergency_steps"] = parsed.get("emergency_steps", prior.get("emergency_steps", ""))
        merged["non_emergency_tips"] = parsed.get("non_emergency_tips", prior.get("non_emergency_tips", ""))
        results[utility] = merged

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

def render_provider_correction_and_refresh(section: str = "utilities"):
    if "force_refresh_map" not in st.session_state:
        st.session_state["force_refresh_map"] = {
            "electricity": False,
            "natural_gas": False,
            "water": False,
            "internet": False,
        }

    current_results = st.session_state.get("utility_providers", {})
    updated = get_corrected_providers(current_results, section=section)

    force_refresh_map = st.session_state["force_refresh_map"]
    queued = [k for k, v in force_refresh_map.items() if v]

    if queued:
        st.markdown("### 🔄 Queued for Refresh:")
        st.write(", ".join(label.replace("_", " ").title() for label in queued))

        if st.button("♻️ Refresh Queued Providers Now"):
            refreshed = fetch_utility_providers(section=section, force_refresh_map=force_refresh_map)
            for utility in queued:
                updated_block = updated.get(utility, {})
                updated_block["emergency_steps"] = refreshed.get(utility, {}).get("emergency_steps", updated_block.get("emergency_steps", ""))
                updated_block["non_emergency_tips"] = refreshed.get(utility, {}).get("non_emergency_tips", updated_block.get("non_emergency_tips", ""))
                updated[utility] = updated_block
                force_refresh_map[utility] = False
            st.success("✅ Refreshed successfully. Please confirm any changes.")

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
            st.success("🔁 Refresh counters reset.")

    st.session_state["corrected_utility_providers"] = updated

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

    for label, key in label_to_key.items():
        current = results.get(key, {})
        name = current.get("name", "")
        phone = current.get("contact_phone", "")
        website = current.get("contact_website", "")
        #email = current.get("contact_email", "")
        address = current.get("contact_address", "")
        description = current.get("description", "")
        emergency = current.get("emergency_steps", "")
        tips = current.get("non_emergency_tips", "")

        # Cooldown tracking
        auto_reset_refresh_status(key)
        attempts = st.session_state.get("provider_refresh_attempts", {}).get(key, 0)
        cooldown = get_remaining_cooldown(key)
        cooldown_active = attempts >= 3 and cooldown > 0

        with st.expander(f"🔧 Validate or Update {label} Provider", expanded=False):
            st.markdown(f"### 🛠️ {label} Provider")

            st.markdown(f"⏱️ Refresh Attempts: `{attempts}`  |  Cooldown: `{cooldown} min remaining`")
            if cooldown_active:
                st.warning(f"⛔ You must wait before refreshing this provider again.")

            # 🔁 Refresh Button
            refresh_clicked = st.button(f"🔁 Refresh {label}", key=f"refresh_{key}")
            if refresh_clicked:
                if "force_refresh_map" not in st.session_state:
                    st.session_state["force_refresh_map"] = {}
                if cooldown_active:
                    st.warning(f"⏳ Cannot refresh {label} yet. Try again later.")
                else:
                    st.session_state["force_refresh_map"][key] = True
                    st.success(f"{label} provider will be re-queried.")

            # Editable fields
            
            # Name
            correct_name = st.checkbox(f"✏️ Correct Provider Name ({name})", value=False, key=f"{key}_name_check")
            name_input = st.text_input(f"{label} Provider Name", value=name, disabled=not correct_name, key=f"{key}_name")

            # Phone            
            correct_phone = st.checkbox(f"✏️ Correct Phone", value=False, key=f"{key}_phone_check")
            phone_input = st.text_input("Phone", value=phone, disabled=not correct_phone, key=f"{key}_phone")

            # Email
            #correct_email = st.checkbox(f"✏️ Correct Email", value=False, key=f"{key}_email_check")
            #email_input = st.text_input("Email", value=email, disabled=not correct_email, key=f"{key}_email")

            # Address
            correct_address = st.checkbox(f"✏️ Correct Address", value=False, key=f"{key}_address_check")
            address_input = st.text_area("Address", value=address, disabled=not correct_address, key=f"{key}_address")

            # Website
            correct_website = st.checkbox(f"✏️ Correct Website", value=False, key=f"{key}_website_check")
            website_input = st.text_input("Website", value=website, disabled=not correct_website, key=f"{key}_website")

            # Description
            correct_description = st.checkbox(f"✏️ Correct Description", value=False, key=f"{key}_desc_check")
            desc_input = st.text_area("Provider Description", value=description, disabled=not correct_description, key=f"{key}_desc")

            # Read-only fields
            st.markdown(f"🚨 **Emergency Steps (Read-Only):**\n{emergency or '—'}")
            if tips and tips != "⚠️ Not Available":
                st.markdown(f"💡 **Non-Emergency Tips:**\n{tips}")

            # ✅ Require confirmation before applying
            confirmed_key = f"{key}_confirmed"
            confirmed = st.checkbox("✅ I have reviewed and confirmed this provider info", key=confirmed_key)

            # Assemble updated block
            updated[key] = {
                "name": name_input if correct_name else name,
                "contact_phone": phone_input if correct_phone else phone,
                #"contact_email": email_input if correct_email else email,
                "contact_address": address_input if correct_address else address,
                "contact_website": website_input if correct_website else website,
                "description": desc_input if correct_description else description,
                "emergency_steps": emergency,  # do not change
                "non_emergency_tips": tips,
                "confirmed": confirmed
            }

            # ✅ Update session state for each corrected field
            if correct_name:
                register_provider_input(label, name_input, section)
                st.session_state[f"{key}_provider"] = name_input

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


                        
