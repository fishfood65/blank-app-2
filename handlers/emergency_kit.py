### Level 2 - Emergency Kit
from utils.prompt_block_utils import generate_all_prompt_blocks
import streamlit as st
import re
from mistralai import Mistral, UserMessage, SystemMessage
from dotenv import load_dotenv
import os
import pandas as pd
from datetime import datetime, timedelta
from docx import Document
from docx.text.run import Run
import re
import time
from PIL import Image
import io
import uuid
import json
from docx.shared import Inches
from utils.preview_helpers import get_active_section_label
from utils.data_helpers import register_task_input, get_answer, extract_providers_from_text, check_missing_utility_inputs 
from utils.debug_utils import debug_all_sections_input_capture_with_summary, clear_all_session_data, debug_single_get_answer
from utils.runbook_generator_helpers import generate_docx_from_prompt_blocks, maybe_render_download, maybe_generate_runbook
from prompts.templates import utility_provider_lookup_prompt
from utils.common_helpers import get_schedule_placeholder_mapping

# --- Generate the AI prompt ---
api_key = os.getenv("MISTRAL_TOKEN")
client = Mistral(api_key=api_key)

if not api_key:
    api_key = st.text_input("Enter your Mistral API key:", type="password")

if api_key:
    st.success("API key successfully loaded.")
else:
   st.error("API key is not set.")

# --- Constants ---
KIT_ITEMS = [
    "🔦 Flashlights and extra batteries",
    "🩹 First aid kit",
    "🥫 Non-perishable food and bottled water",
    "💊 Medications and personal hygiene items",
    "📂 Important documents (insurance, identification)",
    "📻 Battery-powered or hand-crank radio",
    "🎺 Whistle (for signaling)",
    "😷 Dust masks (for air filtration)",
    "🗺️ Local maps and contact lists",
    "🧯 Fire extinguisher"
]
# --- Helper functions (top of the file) ---

def homeowner_kit_stock(section="emergency_kit"):
    selected = []

    with st.form(key="emergency_kit_form"):
        st.write("Select all emergency supplies you currently have:")

        for start in range(0, len(KIT_ITEMS), 4):
            chunk = KIT_ITEMS[start : start + 4]
            cols = st.columns(len(chunk))

            for idx, item in enumerate(chunk):
                safe_key = "kit_" + re.sub(r'[^a-z0-9_]', '', item.lower().replace(' ', '_'))
                has_item = register_task_input(
                    label=item,
                    input_fn=cols[idx].checkbox,
                    section=section,
                    task_type="Emergency Supply",
                    is_freq=False,
                    key=safe_key,
                    value=st.session_state.get(safe_key, False)
                )
                if has_item:
                    selected.append(item)

        submitted = st.form_submit_button("Submit")

    if submitted:
        missing = [item for item in KIT_ITEMS if item not in selected]
        if missing:
            st.warning("⚠️ Consider adding the following items to your emergency kit:")
            for item in missing:
                st.write(f"- {item}")

    return selected, [item for item in KIT_ITEMS if item not in selected]

def emergency_kit(section="emergency_kit"):
    st.subheader("🧰 Emergency Kit Setup")

    # 1. Kit ownership status
    emergency_kit_status = register_task_input(
        label="Do you have an Emergency Kit?",
        input_fn=st.radio,
        section=section,
        options=["Yes","No"],
        index=0,
        metadata={"is_task": False, "frequency_field": False},
        key="emergency_kit_status",
        value=st.session_state.get("emergency_kit_status", "")
    )

    if emergency_kit_status == 'Yes':
        st.success('Great—you already have a kit!', icon=":material/medical_services:")
    else:
        st.warning("⚠️ Let's build your emergency kit with what you have.")

    # 2. Kit location
    emergency_kit_location = register_task_input(
        label="Where is (or where will) the Emergency Kit be located?",
        input_fn=st.text_area,
        section=section,
        value=st.session_state.get("emergency_kit_location", ""),
        placeholder="e.g., hall closet, garage bin",
        key="emergency_kit_location",
        metadata={"is_task": False, "frequency_field": False}
    )

    # 3. Core stock selector (uses capture_input internally with task metadata)
    selected_items, not_selected_items = homeowner_kit_stock()
    st.session_state['homeowner_kit_stock'] = selected_items
    st.session_state["not_selected_items"] = not_selected_items

    # 4. Custom additions
    additional = register_task_input(
        label="Add any additional emergency kit items not in the list above (comma-separated):",
        input_fn=st.text_input,
        section=section,
        value=st.session_state.get("additional_kit_items", ""),
        key="additional_kit_items",
        metadata={"is_task": False, "frequency_field": False}
    )

# --- Main Function Start ---

def emergency_kit_utilities():
    section = "emergency_kit"
    generate_key = f"generate_runbook_{section}"  # Define it early
    
    # 🧪 Optional: Reset controls for testing
    if st.checkbox("🧪 Reset Emergency Kit Session State"):
        emergency_keys = [
            "generated_prompt", "runbook_buffer", "runbook_text", "user_confirmation", "homeowner_kit_stock",
            "additional_kit_items", "not_selected_items", f"{section}_runbook_text", f"{section}_runbook_buffer", 
            f"{section}_runbook_ready"
        ]
        for key in emergency_keys:
            st.session_state.pop(key, None)
        st.success("🔄 Level 2 session state reset.")
        st.stop()  # 🔁 prevent rest of UI from running this frame

    #st.markdown(f"### Currently Viewing: {get_active_section_label(section)}")
    #switch_section("emergency_kit")

    # Step 1: Input collection
    emergency_kit()

    if st.session_state.get("enable_debug_mode", False):
        st.markdown("### 🧪 Session State Debug")
        st.write("emergency_kit_status:", st.session_state.get("emergency_kit_status"))
        st.write("emergency_kit_location:", st.session_state.get("emergency_kit_location"))
        st.write("additional_kit_items:", st.session_state.get("additional_kit_items"))
        st.markdown("### 🧪 Task Inputs")
        st.json(st.session_state.get("task_inputs", []))
        st.write(debug_single_get_answer(key="emergency_kit_status", section="emergency_kit"))
        st.write(debug_single_get_answer(key="emergency_kit_location", section="emergency_kit"))
        st.write(debug_single_get_answer(key="additional_kit_items", section="emergency_kit"))
        debug_all_sections_input_capture_with_summary(["emergency_kit"])

    missing = check_missing_utility_inputs()
    if missing:
        st.warning(f"⚠️ Missing required fields: {', '.join(missing)}")
        if st.button("🔙 Back to Level 1"):
            st.session_state["go_home"] = True
            st.rerun()
        return # ⬅️ Early exit
    
    # ✅ Automatically generate prompt blocks once providers are saved
    blocks = generate_all_prompt_blocks(section)
    st.success("✅ All required utility inputs are complete.")
    st.session_state["utility_providers_saved"] = True

    #Debug preview
    if st.session_state.get("enable_debug_mode"):
        preview_blocks = generate_all_prompt_blocks(section)
        st.markdown("### 🧾 Prompt Preview")
        for block in preview_blocks:
            st.code(block, language="markdown")
        st.markdown("### 🧪 get_answer() Results")
        st.write(debug_single_get_answer(key="emergency_kit_status", section="emergency_kit"))
        st.write(debug_single_get_answer(key="emergency_kit_location", section="emergency_kit"))
        st.write(debug_single_get_answer(key="additional_kit_items", section="emergency_kit"))

    #Step 2: Generate DOCX

    def generate_kit_docx():
        blocks = generate_all_prompt_blocks(section)
        #st.success("✅ All required utility inputs are complete.")
        #st.session_state["utility_providers_saved"] = True
        return generate_docx_from_prompt_blocks(
            section=section,
            blocks=blocks, 
            schedule_sources=get_schedule_placeholder_mapping(),   
            include_heading=True,
            use_llm=True,
            api_key=os.getenv("MISTRAL_TOKEN"),
            doc_heading="⛑️ Utilities & Emergency Kit Runbook ",
            debug=st.session_state.get("enable_debug_mode", False)
            #include_priority=include_priority
        )

    maybe_generate_runbook(
        section=section,
        generator_fn=generate_kit_docx,
        doc_heading="⛑️ Utilities & Emergency Kit Runbook",
        filename="utilities_emergency_kit.docx",
        button_label="📥 Generate Runbook"
    )

    