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
    extract_providers_from_text, 
    check_missing_utility_inputs
)
from utils.runbook_generator_helpers import generate_docx_from_prompt_blocks, maybe_render_download
from utils.debug_utils import debug_all_sections_input_capture_with_summary, clear_all_session_data
from prompts.templates import utility_provider_lookup_prompt

# --- Generate the AI prompt ---
api_key = os.getenv("MISTRAL_TOKEN")
client = Mistral(api_key=api_key)

if not api_key:
    api_key = st.text_input("Enter your Mistral API key:", type="password")

if api_key:
    st.success("API key successfully loaded.")
else:
   st.error("API key is not set.")

# --- Helper functions (top of the file) ---
def get_home_inputs(section: str):
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
        task_type="Location Info"
    )

    zip_code = register_task_input(
        label="ZIP Code",
        input_fn=st.text_input,
        section=section,
        value=st.session_state.get("ZIP Code", ""),
        key="ZIP Code",
        task_type="Location Info"
    )

    internet_provider = register_task_input(
        label="Internet Provider",
        input_fn=st.text_input,
        section=section,
        value=st.session_state.get("Internet Provider", ""),
        key="Internet Provider",
        task_type="Utilities"
    )

    # Optional: mirror into top-level keys (but not required for get_answer)
    st.session_state["city"] = city
    st.session_state["zip_code"] = zip_code
    st.session_state["internet_provider"] = internet_provider

    return city, zip_code, internet_provider

def query_utility_providers(section: str, test_mode: bool = False) -> dict:
    """
    Queries utility providers using city and ZIP from the specified section.

    Args:
        section (str): The section name where city and ZIP inputs were stored.
        test_mode (bool): If True, return hardcoded results for testing.

    Returns:
        dict: A dictionary containing utility provider names.
    """
    city = get_answer(key="City", section=section, verbose=True)
    zip_code = get_answer(key="ZIP Code", section=section, verbose=True)

    if st.session_state.get("enable_debug_mode"):
        st.markdown("### 🧪 Debug: Utility Query Inputs fetching from get_answer(), data saved by register_task")
        st.write("📍 City:", city)
        st.write("📮 ZIP Code:", zip_code)

    if test_mode:
        return {
            "electricity": "Austin Energy",
            "natural_gas": "Atmos Energy",
            "water": "Austin Water"
        }

    prompt = utility_provider_lookup_prompt(city, zip_code)

    try:
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[UserMessage(content=prompt)],
            max_tokens=1500,
            temperature=0.5,
        )
        content = response.choices[0].message.content
    except Exception as e:
        st.error(f"Error querying Mistral API: {str(e)}")
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

def extract_and_log_providers(content: str, section: str) -> dict:
    """
    Extracts provider names and logs them using structured metadata.
    """
    providers = extract_providers_from_text(content)

    register_provider_input("Electricity", providers["electricity"], section=section)
    register_provider_input("Natural Gas", providers["natural_gas"], section=section)
    register_provider_input("Water", providers["water"], section=section)

    # Store for retrieval
    st.session_state["utility_providers"] = providers
    for key, val in providers.items():
        st.session_state[f"{key}_provider"] = val

    return providers

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
    if st.session_state.get("enable_debug_mode"):
        st.markdown("### 🧪 Debug: query_utility_providers")
        st.write("🔌 Session Provider Data:", st.session_state.get("utility_providers"))
        st.write("🔌 Electricity:", st.session_state.get("electricity_provider"))
        st.write("🔥 Natural Gas:", st.session_state.get("natural_gas_provider"))
        st.write("💧 Water:", st.session_state.get("water_provider"))
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
def home():
    section = "home"
    generate_key = f"generate_runbook_{section}"  # Define it early

    #st.markdown(f"### Currently Viewing: {get_active_section_label(section)}")
    #switch_section(section)

    if st.button("🧼 Reset All Data"):
        clear_all_session_data()

    st.subheader("Let's gather some information. Please enter your details:")
   
# Step 1: Input collection
    # ✅ Call the function at runtime
    city, zip_code, internet_provider = get_home_inputs(section)

    if st.session_state.get("enable_debug_mode"): # DEBUG: get_home_inputs()
        st.markdown("### 🧪 Debug: get_home_inputs ")
        st.write("City:", city)
        st.write("ZIP Code:", zip_code)
        st.write("Internet Provider:", internet_provider)

# Step 2: Fetch utility providers

    if st.button("Find My Utility Providers"):
        with st.spinner("Fetching providers from Mistral..."):
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
        st.write("Render button for:", generate_key)
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
        
        if st.session_state.get("enable_debug_mode"):
            st.markdown("### 🧾 Prompt Preview")
            for block in blocks:
                st.code(block, language="markdown")
        #Step 2: Generate DOCX
        include_priority = st.session_state.get("include_priority", True) # Ensure default for include_priority

        def generate_kit_docx():
            blocks = generate_all_prompt_blocks(section)
            st.session_state[f"{section}_runbook_blocks"] = blocks  # ✅ Store for debug
            return generate_docx_from_prompt_blocks(
                section=section,
                blocks=blocks,
                schedule_sources=get_schedule_placeholder_mapping(),
                include_heading=True,
                include_priority=include_priority,
                use_llm=False,
                api_key=os.getenv("MISTRAL_TOKEN"),
                doc_heading="📬 Mail and 🗑️ Trash Runbook",
                debug=st.session_state.get("enable_debug_mode", False),
            )

        maybe_generate_runbook(
            section=section,
            generator_fn=generate_kit_docx,
            doc_heading="📬 Mail and 🗑️ Trash Runbook",
            filename="utilities_emergency_kit.docx",
            button_label="📥 Generate Runbook"
        )
        
        
        # Step 6: Generate the DOCX and Markdown
        st.subheader("🎉 Reward")

                
            # After generation
            st.session_state[generate_key] = False

        if st.session_state.get(generate_key):
            #st.info("⚙️ Calling generate_docx_from_prompt_blocks...")
            buffer, markdown_text = generate_docx_from_prompt_blocks(
                section=section,
                blocks=blocks,
                insert_main_heading=True,
                use_llm=bool(True),
                api_key=os.getenv("MISTRAL_TOKEN"),
                doc_heading="🏠 Utilities Emergency Runbook",
                debug=False,
                #include_priority=include_priority
            )
            if st.session_state.get("enable_debug_mode"):
                st.write("📋 Blocks sent to DOCX generator:", blocks)
                #st.write("📝 Markdown Text:", markdown_text)
                st.write("📁 DOCX Buffer:", buffer)
                st.write("🧪 Buffer type:", type(buffer))
                st.write("🧪 Buffer size:", buffer.getbuffer().nbytes if isinstance(buffer, io.BytesIO) else "Invalid")

            # Cache results in session state
            st.session_state[f"{section}_runbook_text"] = markdown_text
            st.session_state[f"{section}_runbook_buffer"] = buffer
            st.session_state[f"{section}_runbook_ready"] = True

# Step 6: Show download options (only if generation succeeded)
    if st.session_state.get(f"{section}_runbook_ready"):
        st.success("✅ Runbook Ready!")
        maybe_render_download(section=section, filename="utilities_emergency.docx")
        st.session_state["level_progress"][section] = True
    else:
        st.info("ℹ️ Click the button above to generate your runbook.")
