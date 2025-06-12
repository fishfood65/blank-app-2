### Leve 1 - Home
from utils.prompt_block_utils import generate_all_prompt_blocks
import streamlit as st
import re
from mistralai import Mistral, UserMessage, SystemMessage
import os
import pandas as pd
from datetime import datetime, timedelta
import re
import time
import io
import uuid
import json
from utils.preview_helpers import get_active_section_label
from utils.data_helpers import (
    register_task_input, 
    get_answer, 
    check_missing_utility_inputs,
    extract_and_log_providers
)
from utils.runbook_generator_helpers import generate_docx_from_prompt_blocks, maybe_render_download, maybe_generate_runbook
from utils.debug_utils import debug_all_sections_input_capture_with_summary, reset_all_session_state
from prompts.templates import utility_provider_lookup_prompt, wrap_with_claude_style_formatting
from utils.common_helpers import get_schedule_placeholder_mapping
from utils.llm_cache_utils import get_or_generate_llm_output
from utils.llm_helpers import call_openrouter_chat

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
        task_type="Location Info",
        required=True
    )

    zip_code = register_task_input(
        label="ZIP Code",
        input_fn=st.text_input,
        section=section,
        value=st.session_state.get("ZIP Code", ""),
        key="ZIP Code",
        task_type="Location Info",
        required=True
    )

    internet_provider = register_task_input(
        label="Internet Provider",
        input_fn=st.text_input,
        section=section,
        value=st.session_state.get("Internet Provider", ""),
        key="Internet Provider",
        task_type="Utilities",
        required=False
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
    internet = get_answer("Internet Provider", section, verbose = True)

    if not city or not zip_code:
        st.warning("⚠️ Missing City or ZIP Code. Cannot query providers.")
        return {}

    # Normalize "⚠️ Not provided"
    if internet and internet.strip().lower() in ["", "⚠️ not provided", "n/a"]:
        internet = ""  # Let LLM fill it in

    if st.session_state.get("enable_debug_mode"):
        st.markdown("### 🧪 Debug: Utility Query Inputs fetching from get_answer(), data saved by register_task")
        st.write("📍 City:", city)
        st.write("📮 ZIP Code:", zip_code)
        st.write("🌐 Internet Provider:", internet)

    if test_mode:
        return {
            "electricity": "Austin Energy",
            "natural_gas": "Atmos Energy",
            "water": "Austin Water",
            "internet": "Comcast"
        }

    # Build and send the prompt
    prompt = utility_provider_lookup_prompt(city, zip_code,internet)
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

def register_provider_input(label: str, value: str, section: str):
    if not label or not isinstance(label, str):
        raise ValueError("Provider label must be a non-empty string.")

    if not isinstance(value, str) or value.strip().lower() in ["", "not found", "n/a"]:
        return

    # Register as structured input data
    input_data = st.session_state.setdefault("input_data", {})
    section_data = input_data.setdefault(section, [])

    # Remove duplicates first
    section_data = [entry for entry in section_data if entry["question"] != f"{label} Provider"]
    section_data.append({
        "question": f"{label} Provider",
        "answer": value.strip()
    })
    input_data[section] = section_data

    # Track in task_inputs
    task_row = {
        "question": f"{label} Provider",
        "answer": value.strip(),
        "category": section,
        "section": section,
        "area": "home",
        "task_type": "info",
        "is_freq": False
    }
    st.session_state.setdefault("task_inputs", []).append(task_row)

def get_corrected_providers(results: dict) -> dict:
        updated = {}
        label_to_key = {
            "Electricity": "electricity",
            "Natural Gas": "natural_gas",
            "Water": "water"
        }

        for label in label_to_key:
            key = label_to_key[label]
            current_value = results.get(key, "")
            correct_flag = st.checkbox(f"Correct {label} Provider", value=False)
            corrected = st.text_input(
                f"{label} Provider",
                value=current_value,
                disabled=not correct_flag
            )
            if correct_flag and corrected != current_value:
                register_provider_input(label, corrected)
                st.session_state[f"{key}_provider"] = corrected
            updated[key] = corrected if correct_flag else current_value

        return updated

def fetch_utility_providers(section: str):
    results = query_utility_providers(section=section)
    st.session_state["utility_providers"] = results

    # ✅ Save individual providers into session_state for easier access downstream
    st.session_state["electricity_provider"] = results.get("electricity", "")
    st.session_state["natural_gas_provider"] = results.get("natural_gas", "")
    st.session_state["water_provider"] = results.get("water", "")
    st.session_state["internet_provider"] = results.get("internet", "")

    if st.session_state.get("enable_debug_mode"):
        st.markdown("### 🧪 Debug: query_utility_providers")
        st.write("🔌 Session Provider Data:", st.session_state.get("utility_providers"))
        st.write("🔌 Electricity:", st.session_state.get("electricity_provider"))
        st.write("🔥 Natural Gas:", st.session_state.get("natural_gas_provider"))
        st.write("💧 Water:", st.session_state.get("water_provider"))
        st.write("🌐 Internet:", st.session_state.get("internet_provider"))
        st.write("🤖 Using Model:", st.session_state.get("llm_model", "claude-3-haiku"))
    return results

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
            fetch_utility_providers(section=section)
            st.session_state["show_provider_corrections"] = True
            st.success("Providers stored in session state!")
  
    if st.session_state.get("enable_debug_mode"):
        st.markdown("### 🧪 Debug: Find Utility Providers")
        st.write("🔌 Session Provider Data:", st.session_state.get("utility_providers"))
        st.write("🔌 Electricity:", st.session_state.get("electricity_provider"))
        st.write("🔥 Natural Gas:", st.session_state.get("natural_gas_provider"))
        st.write("💧 Water:", st.session_state.get("water_provider"))

# Step 3: Display resulting LLM retrieved Utility Providers
    if st.session_state.get("show_provider_corrections"):
        current_results = st.session_state.get("utility_providers", {
            "electricity": "",
            "natural_gas": "",
            "water": ""
        })
        updated = get_corrected_providers(current_results)

        if st.session_state.get("enable_debug_mode", False):
            debug_all_sections_input_capture_with_summary(["home", "emergency_kit"])

# Step 4: Save Utility Providers (with validation)
        if st.button("💾 Save Utility Providers"):
            missing = check_missing_utility_inputs()
            if missing:
                st.warning(f"⚠️ Missing required fields: {', '.join(missing)}")
            else:
                update_session_state_with_providers(updated)
                st.session_state["utility_providers_saved"] = True
                st.success("✅ Utility providers updated!")
                st.subheader("🎉 Reward")

                # ✅ Show output in debug mode
                if st.session_state.get("enable_debug_mode"):
                    st.markdown("### 🧪 Debug: Saved Providers")
                    st.write("🔌 Session Provider Data:", st.session_state.get("utility_providers"))
                    st.write("🔌 Electricity:", st.session_state.get("electricity_provider"))
                    st.write("🔥 Natural Gas:", st.session_state.get("natural_gas_provider"))
                    st.write("💧 Water:", st.session_state.get("water_provider"))
    
# Step 5: Generate prompt blocks if Utility Providers are saved
    if st.session_state.get("utility_providers_saved"):
        blocks = generate_all_prompt_blocks(section)
            # ✅ Automatically generate prompt blocks once providers are saved
        st.session_state[f"{section}_runbook_blocks"] = blocks
        st.subheader("🎉 Reward")
        
        if st.session_state.get("enable_debug_mode"):
            st.markdown("### 🧾 Prompt Preview")
            for block in blocks:
                st.code(block, language="markdown")
        #Step 2: Generate DOCX
        include_priority = st.session_state.get("include_priority", True) # Ensure default for include_priority

        def generate_utilities_docx():
            blocks = generate_all_prompt_blocks(section)
            st.session_state[f"{section}_runbook_blocks"] = blocks  # ✅ Store for debug
            return generate_docx_from_prompt_blocks(
                section=section,
                blocks=blocks,
                schedule_sources=get_schedule_placeholder_mapping(),
                include_heading=False,
                include_priority=include_priority,
                use_llm=True,
                api_key=os.getenv("MISTRAL_TOKEN"),
                doc_heading="🔌 Utilities Emergency Runbook",
                debug=st.session_state.get("enable_debug_mode", False),
            )

        maybe_generate_runbook(
            section=section,
            generator_fn=generate_utilities_docx,
            doc_heading="🔌 Utilities Emergency Runbook",
            filename="utilities_emergency_runbook",
            button_label="📥 Generate Runbook"
        )
        
