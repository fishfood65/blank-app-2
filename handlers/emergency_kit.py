### Level 2 - Emergency Kit
from config.sections import SECTION_METADATA
from utils.docx_helpers import generate_emergency_utilities_kit_docx, generate_emergency_utilities_kit_markdown, render_runbook_section_output
from utils.prompt_block_utils import generate_all_prompt_blocks
import streamlit as st
import re
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
    load_kit_items_config,
    make_safe_key,
    normalize_item,
    register_task_input, 
    get_answer, 
    fuzzy_match_item
)
from utils.debug_utils import (
    debug_all_sections_input_capture_with_summary, 
    debug_single_get_answer
)
from utils.runbook_generator_helpers import (
    generate_docx_from_prompt_blocks, 
    maybe_render_download, 
    maybe_generate_runbook
)
from utils.common_helpers import get_schedule_placeholder_mapping

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

def homeowner_kit_stock(section="emergency_kit"):
    selected = []
    kit_config = load_kit_items_config()
    KIT_ITEMS = kit_config.get("recommended_items", [])

    # Safe defaults to avoid UnboundLocalError
    additional_input = ""
    additional_normalized = []
    auto_matched = []
    unmatched = []

    normalized_kit = {normalize_item(item): item for item in KIT_ITEMS}

    with st.form(key="emergency_kit_form"):
        st.write("Select all emergency supplies you currently have:")

        cols = st.columns(4)  # Always 4 columns

        for idx, item in enumerate(KIT_ITEMS):
            col = cols[idx % 4]
            # 🧼 Clean key generation
            safe_key = "kit_" + re.sub(r'[^a-z0-9]', '_', item.lower())
            safe_key = re.sub(r'_+', '_', safe_key).strip('_')

            register_task_input(
                label=item,
                input_fn=col.checkbox,
                section=section,
                task_type="Emergency Supply",
                area="home",
                is_freq=False,
                key=safe_key,
                value=st.session_state.get(safe_key, False)
            )
            if st.session_state.get(safe_key):
                selected.append(item)

        # Additional items input
        st.markdown("**Optional: Add any emergency kit items not listed above (comma-separated):**")
        additional_input = st.text_input(
            label="Additional Items",
            key="additional_kit_items",
            value=st.session_state.get("additional_kit_items", ""),
            placeholder="e.g., gloves, blanket, walkie talkies"
        )
        # Use `additional_input` directly from here on — don't write back into session_state

        submitted = st.form_submit_button("Submit")

    if submitted:
        # Normalize kit items
        normalized_kit = {normalize_item(k): k for k in KIT_ITEMS}

        # Parse and normalize user-provided additional items
        additional_raw = [item.strip() for item in additional_input.split(",") if item.strip()]
        additional_normalized = [normalize_item(x) for x in additional_raw]

        auto_matched = []
        unmatched = []

        for norm, raw_item in zip(additional_normalized, additional_raw):
            if norm in normalized_kit:
                matched_item = normalized_kit[norm]
                if matched_item not in selected:
                    selected.append(matched_item)
                    auto_matched.append(matched_item)
            else:
                # 🔍 Fuzzy match if no exact normalized match
                fuzzy = fuzzy_match_item(raw_item, KIT_ITEMS, cutoff=0.75)
                if fuzzy and fuzzy not in selected:
                    selected.append(fuzzy)
                    auto_matched.append(fuzzy)

                    # Log auto-matched item
                    safe_key = "kit_" + make_safe_key(fuzzy)
                    register_task_input(
                        label=matched_item,
                        input_fn=lambda label, **kwargs: True,
                        section=section,
                        task_type="Emergency Supply",
                        area="home",
                        is_freq=False,
                        key=safe_key,
                        value=True
                    )
                else:
                    unmatched.append(raw_item)

        if auto_matched:
            st.success(f"✅ Matched and included additional items: {', '.join(auto_matched)}")
            st.session_state["matched_additional_items"] = auto_matched

        if unmatched:
            st.session_state["unmatched_additional_items"] = unmatched
            st.info("🆕 Custom items (not in kit list):")
            for item in unmatched:
                st.write(f"- ❓ {item}")

            if st.button("🔁 Retry Matching", key="retry_kit_matching"):
                unmatched = st.session_state.get("unmatched_additional_items", [])
                matched_items = []

                for raw_item in unmatched:
                    fuzzy = fuzzy_match_item(raw_item, KIT_ITEMS, cutoff=0.75)
                    if fuzzy and fuzzy not in st.session_state.get("homeowner_kit_stock", []):
                        st.session_state["homeowner_kit_stock"].append(fuzzy)
                        matched_items.append(fuzzy)

                        safe_key = "kit_" + re.sub(r'[^a-z0-9]', '_', fuzzy.lower())
                        safe_key = re.sub(r'_+', '_', safe_key).strip('_')
                        st.session_state[safe_key] = True  # Mark checkbox as selected

                        register_task_input(
                            label=fuzzy,
                            input_fn=lambda label, **kwargs: True,
                            section="emergency_kit",
                            task_type="Emergency Supply",
                            area="home",
                            is_freq=False,
                            key=safe_key,
                            value=True
                        )

                # Clear unmatched list if everything is matched
                still_unmatched = [i for i in unmatched if i not in matched_items]
                st.session_state["matched_additional_items"] = matched_items
                if still_unmatched:
                    st.session_state["unmatched_additional_items"] = still_unmatched
                    st.info(f"ℹ️ Still unmatched: {', '.join(still_unmatched)}")
                else:
                    st.session_state.pop("unmatched_additional_items", None)
                    st.success("✅ All unmatched items successfully re-matched!")
                
                    # ✅ Force rerun to refresh checkboxes/UI
                    st.rerun()

            # ✅ Debug block goes here
    if st.session_state.get("enable_debug_mode"):
        st.markdown("### 🧪 Emergency Kit Debug Info")
        st.json({
            "Inventory Summary": {
                "Selected Items": selected,
                "Missing Items": [item for item in KIT_ITEMS if item not in selected],
            },
            "Additional Items": {
                "Raw Input": additional_input,
                "Normalized": additional_normalized,
                "Auto-Matched": auto_matched,
                "Unmatched": unmatched,
                "Retry Pool": st.session_state.get("unmatched_additional_items", [])
            },
            "Checkbox State": {
                "Keys": [k for k in st.session_state if k.startswith("kit_")]
            }
        })

    return selected, [item for item in KIT_ITEMS if item not in selected]

def emergency_kit(section="emergency_kit"):
    st.subheader("🧰 Emergency Kit Setup")

    # 1. Kit ownership status
    emergency_kit_status = register_task_input(
        label="Do you have an Emergency Kit?",
        input_fn=st.radio,
        section=section,
        area="home",
        task_type="Emergency Supply",
        is_freq=False,
        key="emergency_kit_status",
        value=st.session_state.get("emergency_kit_status", ""),
        options=["Yes","No"],
        index=0,
        shared=True,
        required=True
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
        area="home",
        task_type="Emergency Supply",
        is_freq=False,
        key="emergency_kit_location",
        value=st.session_state.get("emergency_kit_location", ""),
        placeholder="e.g., hall closet, garage bin",
        shared=True
    )

    # 3. Core stock selector (uses capture_input internally with task metadata)
    selected_items, not_selected_items = homeowner_kit_stock()
    st.session_state['homeowner_kit_stock'] = selected_items
    st.session_state["not_selected_items"] = not_selected_items
    
    # 💡 Save fuzzy matches if present
    if "auto_matched_kit_items" in st.session_state:
        st.session_state["matched_additional_items"] = st.session_state["auto_matched_kit_items"]

    # 💡 Save unmatched inputs (may be needed for retry button)
    if "unmatched_kit_items" in st.session_state:
        st.session_state["unmatched_additional_items"] = st.session_state["unmatched_kit_items"]

    # Debug
    if st.session_state.get("enable_debug_mode"):
        st.markdown("### 🧪 Emergency Kit Debug Info")

        st.markdown("**✅ Selected Items:**")
        st.write(st.session_state.get("homeowner_kit_stock", []))

        st.markdown("**❌ Missing Standard Items:**")
        st.write(st.session_state.get("not_selected_items", []))

        st.markdown("**🔁 Auto-Matched Additional Items:**")
        st.write(st.session_state.get("matched_additional_items", []))

        st.markdown("**❓ Unmatched Additional Items:**")
        st.write(st.session_state.get("unmatched_additional_items", []))

        st.markdown("**📦 Raw Additional Input:**")
        st.write(st.session_state.get("additional_kit_items", ""))

        st.markdown("**📌 All Checkbox Keys:**")
        st.write([k for k in st.session_state if k.startswith("kit_")])


# --- Main Function Start ---

def emergency_kit_utilities():
    section = "emergency_kit"
    generate_key = f"generate_runbook_{section}"  # Define it early
    
    # 🧪 Optional: Reset controls for testing
    if st.checkbox("🧪 Reset Emergency Kit Session State"):
        emergency_keys = [
            "generated_prompt", "runbook_buffer", "runbook_text", "user_confirmation", "homeowner_kit_stock",
            "additional_kit_items", "not_selected_items", "matched_additional_items", "unmatched_additional_items",
            f"{section}_runbook_text", f"{section}_runbook_buffer", f"{section}_runbook_ready"
        ]
        for key in emergency_keys:
            st.session_state.pop(key, None)
        st.success("🔄 Level 2 session state reset.")
        st.stop()  # 🔁 prevent rest of UI from running this frame

    #st.markdown(f"### Currently Viewing: {get_active_section_label(section)}")
    #switch_section("emergency_kit")

    # Step 1: Input collection
    emergency_kit()

    # Debug Section
    if st.session_state.get("enable_debug_mode"):
        st.markdown("### 🧪 Session Debug")
        st.json({
            "emergency_kit_status": st.session_state.get("emergency_kit_status"),
            "emergency_kit_location": st.session_state.get("emergency_kit_location"),
            "homeowner_kit_stock": st.session_state.get("homeowner_kit_stock"),
            "corrected_utility_providers": list(st.session_state.get("corrected_utility_providers", {}).keys()),
        })

    #Step 2: Check for Valid Data and Return Runbook
    has_status = st.session_state.get("emergency_kit_status") in ["Yes", "No"]
    has_location = bool(st.session_state.get("emergency_kit_location"))
    has_selected_items = bool(st.session_state.get("homeowner_kit_stock"))
    has_utilities = bool(st.session_state.get("corrected_utility_providers"))

    ready_to_generate = has_status and has_location and has_selected_items and has_utilities

    st.subheader("🧪 Provider Debug")
    st.json(st.session_state.get("corrected_utility_providers", {}))

    if ready_to_generate:
        st.subheader("🎉 Reward")
        # Level 2 Complete - for Progress
        st.session_state.setdefault("level_progress", {})
        st.session_state["level_progress"]["emergency_kit"] = True

        markdown = generate_emergency_utilities_kit_markdown()
        docx_bytes = generate_emergency_utilities_kit_docx()
        render_runbook_section_output(
            markdown_str=markdown,
            docx_bytes_io=docx_bytes,
            title="Emergency Utilities + Kit Runbook",
            filename_prefix="emergency_utilities_kit",
            expand_preview=True,
        )
    else:
        st.warning("⚠️ Runbook not ready. Please resolve the missing items below:")

        if not has_status or not has_location or not has_selected_items:
            st.info(
                "✅ Complete the **Emergency Kit** form above:\n"
                "- Do you have a kit?\n"
                "- Where is it?\n"
                "- What’s inside?"
            )

        if not has_utilities:
            st.warning(
                "⚠️ Utility provider details are missing.\n\n"
                "Please go to the **🏠 Home Setup → Utility Providers** section to look up and confirm your providers."
            )
            
            if SECTION_METADATA.get("utilities"):
                if st.button("🔁 Go to Utility Providers", help="Navigate to Home Setup → Utility Providers"):
                    st.session_state["active_section"] = "utilities"
                    st.rerun()





    
   

    
