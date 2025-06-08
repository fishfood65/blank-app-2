### Level 3 - Mail Handing
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
from utils.data_helpers import register_task_input, get_answer, extract_providers_from_text, check_missing_utility_inputs, select_runbook_date_range, sanitize_label, sanitize, section_has_valid_input, check_runbook_dates_confirmed, check_for_date_change
from utils.debug_utils import debug_all_sections_input_capture_with_summary, clear_all_session_data, debug_single_get_answer
from utils.runbook_generator_helpers import generate_docx_from_prompt_blocks, maybe_render_download, maybe_generate_runbook, render_runbook_preview_inline, display_user_friendly_schedule_table
from utils.common_helpers import get_schedule_utils, debug_saved_schedule_dfs, get_schedule_placeholder_mapping, merge_all_schedule_dfs
from utils.task_schedule_utils_updated import extract_and_schedule_all_tasks, extract_unscheduled_tasks_from_inputs_with_category, display_enriched_task_preview, save_task_schedules_by_type, load_label_map, normalize_label
from utils.debug_utils import log_extracted_tasks_debug, debug_schedule_df_presence
from prompts.templates import utility_provider_lookup_prompt
from config.section_router import get_handler
from old.old_code import maybe_generate_prompt, render_prompt_preview

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
def mail(section):
    # 🔁 Lock toggle
    if st.session_state.get("mail_locked", False):
        if st.button("🔓 Unlock Mail Form"):
            st.session_state["mail_locked"] = False
    else:
        if st.button("🔒 Lock Mail Form"):
            st.session_state["mail_locked"] = True

    # Status indicator
    status = "🔒 Locked" if st.session_state.get("mail_locked") else "✅ Editable"
    st.markdown(f"**Status:** {status}")

    disabled = st.session_state.get("mail_locked", False)

    # Form wrapper
    with st.form("mail_form"):
        register_task_input(
            label= "📍 Mailbox Location", 
            input_fn = st.text_area,
            section=section,
            task_type="Mail Handling",
            placeholder="E.g., 'At the end of the driveway...'", 
            disabled=disabled
        )

        register_task_input(
            label= "🔑 Mailbox Key (Optional)", 
            input_fn= st.text_area,
            section=section,
            task_type="Mail Handling",
            placeholder="E.g., 'On key hook...'", 
            disabled=disabled
        )

        register_task_input(
            label= "📆 Mail Pick-Up Schedule", 
            input_fn= st.text_area,
            section=section,
            task_type="Mail Handling",
            is_freq=True,
            placeholder="E.g., 'Mondays and Thursdays'", 
            disabled=disabled
        )

        register_task_input(
            label="📥 What to Do with the Mail after pickup?", 
            input_fn= st.text_area,
            section=section,
            task_type="Mail Handling", 
            placeholder="E.g., 'Place in kitchen tray'", 
            disabled=disabled
        )

        register_task_input(
            label= "🚚 Pick up oversized packages at", 
            input_fn=st.text_area,
            section=section,
            task_type="Mail Handling", 
            placeholder="E.g., 'Put packages inside entryway closet'", 
            disabled=disabled
        )
        register_task_input(
            label="📦 Place packages after pickup", 
            input_fn=st.text_area,
            section=section,
            task_type="Mail Handling",
            placeholder="E.g., 'Put packages inside entryway closet'", 
            disabled=disabled
        )

        submitted = st.form_submit_button("✅ Save Mail Info")

    if submitted:
        st.success("📬 Mail info saved!")
        # ✅ Auto-lock the form after submission
        st.session_state["mail_locked"] = True
        st.rerun()# 🔁 force rerun to refresh status display
        if st.session_state.get("enable_debug_mode"):
            st.json(st.session_state.get("task_inputs", {}).get(section, {}))

def register_radio_input(label, section, task_type, options, key=None, disabled=False, index=0):
    """
    Renders a radio input and stores its value and metadata into session_state.
    Compatible with list-style task_inputs and input_data structures.

    Args:
        label (str): Input label shown to user
        section (str): Section name, e.g., "mail_trash"
        task_type (str): e.g., "Outdoor Trash"
        options (tuple): Radio options
        key (str): Optional key override
        disabled (bool): Whether input is disabled
        index (int): Default selected index
    Returns:
        str: selected option
    """
    if key is None:
        key = f"{section}_{sanitize(label)}"

    value = st.radio(label, options=options, index=index, key=key, disabled=disabled)
    timestamp = datetime.now().isoformat()

    task_row = {
        "question": label,
        "answer": value,
        "category": section,
        "section": section,
        "area": "home",
        "task_type": task_type,
        "is_freq": False,
        "timestamp": timestamp,
        "key": key,
    }

    # --- task_inputs
    task_inputs = st.session_state.setdefault("task_inputs", [])
    task_inputs = [row for row in task_inputs if row.get("key") != key]
    task_inputs.append(task_row)
    st.session_state["task_inputs"] = task_inputs

    # --- input_data
    section_data = st.session_state.setdefault("input_data", {}).setdefault(section, [])
    section_data = [row for row in section_data if row.get("key") != key]
    section_data.append({
        "question": label,
        "answer": value,
        "section": section,
        "key": key,
    })
    st.session_state["input_data"][section] = section_data

    return value

def trash(section):
    # 🔁 Lock toggle
    if st.session_state.get("trash_locked", False):
        if st.button("🔓 Unlock Trash Form"):
            st.session_state["trash_locked"] = False
    else:
        if st.button("🔒 Lock Trash Form"):
            st.session_state["trash_locked"] = True

    # Status indicator
    status = "🔒 Locked" if st.session_state.get("trash_locked") else "✅ Editable"
    st.markdown(f"**Status:** {status}")

    disabled = st.session_state.get("trash_locked", False)

    # Outside the form — capture Single-Family Home selection
    label = "🏠 Is a Single-family home?"
    radio_key = f"{section}_{sanitize(label)}"
    task_type = "Outdoor Trash"

    radio_response = register_radio_input(
        label=label,
        section=section,
        task_type=task_type,
        options=("Yes", "No"),
        key=radio_key,
        disabled=disabled
    )
    is_single_fmaily = radio_response == "Yes"

    # ─── Conditionally show single-family block ──────────────
    if is_single_fmaily:
        st.subheader("🛻 Additional Instructions for Single-Family Homes")
        register_task_input("🛻 When and where should garbage, recycling, and compost bins be placed for pickup?", st.text_area, section, task_type="Outdoor Trash", is_freq=True, placeholder="E.g. Curb Monday night", disabled=disabled)
        register_task_input("🗑️ When and where should bins be brought back in?", st.text_area, section, task_type="Outdoor Trash", is_freq=True, placeholder="E.g. Garage Tuesday night", disabled=disabled)

    # Now enter the form block
    with st.form("trash_form"):

        # ─── Indoor Trash Disposal ────────────────────────────────
        st.subheader("🏠 Indoor Trash & Recycling")
        register_task_input("🧴 Kitchen Garbage Bin", st.text_area, section, task_type="Indoor Trash", is_freq=True, placeholder="E.g. Bin is under sink...", disabled=disabled)
        register_task_input("♻️ Indoor Recycling Bin(s)", st.text_area, section, task_type="Indoor Trash", is_freq=True, placeholder="E.g. Bins are under sink...", disabled=disabled)
        register_task_input("🧃 Indoor Compost or Green Waste", st.text_area, section, task_type="Indoor Trash", is_freq=True, placeholder="E.g. Bin is yellow and under sink...", disabled=disabled)
        register_task_input("🧼 Bathroom Trash Bin", st.text_area, section, task_type="Indoor Trash", is_freq=True, placeholder="E.g. Empty Mondays...", disabled=disabled)
        register_task_input("🪑 Other Room Trash Bins", st.text_area, section, task_type="Indoor Trash", is_freq=True, placeholder="E.g. Empty weekly before trash day...", disabled=disabled)

        st.markdown("---")

        # ─── Outdoor Trash Pickup ────────────────────────────────
        st.subheader("🚛 Garbage & Recycling Pickup Details")
        register_task_input("📍 Bin Storage Location", st.text_area, section, task_type="Outdoor Trash", placeholder="E.g., side yard", disabled=disabled)
        register_task_input("🏷️ How are bins marked?", st.text_area, section, task_type="Outdoor Trash", placeholder="E.g., blue = recycling", disabled=disabled)
        register_task_input("📋 What to know before recycling or composting", st.text_area, section, task_type="Outdoor Trash", placeholder="E.g., rinse containers", disabled=disabled)

        st.markdown("---")

        # ─── Waste Management Contact ─────────────────────────────
        st.subheader("📇 Waste Management Contact Info")
        register_task_input("🏢 Waste Management Company Name", st.text_input, section, task_type="Outdoor Trash", disabled=disabled)
        register_task_input("📞 Contact Phone Number", st.text_input, section, task_type="Outdoor Trash", disabled=disabled)
        register_task_input("📝 When to Contact", st.text_area, section, task_type="Outdoor Trash", disabled=disabled)

        # ✅ Submit button must be *inside* the form
        submitted = st.form_submit_button("✅ Save Trash Info")

    if submitted:
        st.success("📬 Trash info saved!")
        # Add the radio response into task_inputs
        st.session_state["task_inputs"] = [
            row for row in st.session_state.get("task_inputs", [])
            if row.get("key") != radio_key
        ]

        st.session_state["task_inputs"].append({
            "question": label,
            "answer": is_single_fmaily,
            "category": section,
            "section": section,
            "area": "home",
            "task_type": task_type,
            "is_freq": False,
            "key": radio_key
        })
                # ✅ Auto-lock the form after submission
        st.session_state["trash_locked"] = True
        st.rerun()# 🔁 force rerun to refresh status display
        if st.session_state.get("enable_debug_mode"):
            st.json(st.session_state.get("task_inputs", {}).get(section, {}))

# --- Main Function Start ---

def mail_trash():
    section = "mail_trash"
    generate_key = f"generate_runbook_{section}"  # Define it early

    # Optional reset
    skip_rerun = st.checkbox("⚠️ Skip rerun (debug only)", value=False)
    if st.checkbox("🔪 Reset Mail and Trash Session State"):
        keys_to_clear = [
            "generated_prompt",
            "runbook_buffer",
            "runbook_text",
            "user_confirmation",
            "task_inputs",
            "combined_home_schedule_df",
            "mail_schedule_df",
            "trash_schedule_df",
            "mail_schedule_markdown",
            "trash_flat_schedule_md",
            "home_schedule_markdown",
            "mail_locked",
            "trash_locked",
            "indoor_trash_schedule_df",
            "mail_handling_schedule_df",
            "outdoor_trash_schedule_df",
            "input_data",
            "autosaved_json",
            "interaction_log",
            "grouped_mail_task",
            "grouped_trash_schedule"
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        st.success("🔄 Level 3 session state reset. Inputs and data cleared.")
            # ⛔ Skip rerun if debugging
        if not skip_rerun:
            st.rerun()

    # 🔍 Debugging output
    with st.expander("🧠 Session State (After Reset)"):
        st.json({k: str(v) for k, v in st.session_state.items()})

    section = st.session_state.get("section", "home")

    st.markdown(f"### Currently Viewing: {get_active_section_label(section)}")


# --- Sections Broken out by "--" ---

    # 📬 Mail
    from handlers.mail_trash import mail
    with st.expander("📬 Mail Handling", expanded=True):
        mail(section)
# ----------------------------------
    # 🗑️ Trash
    from handlers.mail_trash import trash
    with st.expander("🗑️ Trash Handling", expanded=False):
        trash(section)
# ----------------------------------
    #st.write("🧪 input_data[mail_trash]:", st.session_state.get("input_data", {}).get("mail_trash"))
    #st.write("🧪 task_inputs[mail_trash]:", [
    #t for t in st.session_state.get("task_inputs", []) if t.get("section") == "mail_trash"
    #])

    if section_has_valid_input("mail_trash", min_entries=8):
        st.subheader("🧐 Customize, Review and Reward")

    # # Step 1: Get valid dates from user
        check_for_date_change()  
        choice, start_date, end_date, valid_dates = select_runbook_date_range()

        if start_date and end_date:
            # Step 2: Save date info
            st.session_state.update({
                "start_date": start_date,
                "end_date": end_date,
                "valid_dates": valid_dates
            })
            st.markdown("---")
            # Step 3: Load scheduling utilities
            utils = get_schedule_utils()

            # Step 4: Extract raw tasks from session and input
            df = extract_unscheduled_tasks_from_inputs_with_category()
            if st.session_state.get("enable_debug_mode", False):
                debug_summary = log_extracted_tasks_debug(df, section="mail_trash")

            # Step 5: Schedule tasks (for your only section)
            schedule_df = extract_and_schedule_all_tasks(
                valid_dates=valid_dates,
                utils=utils,
                df=df,  # 👈 use the extracted tasks
                section="mail_trash",
                output_key="mail_trash_schedule_df"  # 👈 optional, also stored to session
            )

            # Step 6: Optional preview toggle
            refresh_preview = st.checkbox("🔄 Refresh Preview", value=True)

            # Step 7: Schedule and merge all available *_schedule_df entries
            if st.session_state.get("enable_debug_mode"):
                debug_schedule_df_presence() # optional log
            
            combined_df = merge_all_schedule_dfs(
                valid_dates=valid_dates,
                utils=utils,
                output_key="combined_home_schedule_df",  # optional deduplication protection
                deduplicate=True,
                annotate_source=True,
                group_keys={
                "trash_schedule_df": ["indoor_trash_schedule_df", "outdoor_trash_schedule_df"]
            }
            )

            # Step 8: Save merged version to session
            st.session_state["combined_home_schedule_df"] = combined_df

            # Step 9: Show user preview
            if refresh_preview and combined_df is not None:
                st.subheader("📆 Review & Update Scheduled Tasks:")
                display_enriched_task_preview(combined_df)

            # Step 10: Save split schedules for mail/trash/etc
            save_task_schedules_by_type(combined_df)

            display_user_friendly_schedule_table(
                df=combined_df,
                heading_text="🧹 Task Schedule",
                show_heading=True,             # Optional, defaults to True
                show_legend=True,              # ✅ Enables emoji legend in expander
                enable_task_filter=True        # ✅ Enables collapsible task type filter
            )

            # Step 14: Redundant update (keep for consistency)
            st.session_state.update({
                "combined_home_schedule_df": combined_df
            })
    st.markdown("---")

    if st.button("🔄 Reset Runbook Prerequisites (for testing)"):
        for key in ["mail_locked", "trash_locked", "runbook_dates_confirmed"]:
            st.session_state.pop(key, None)
        st.success("🧹 Reset complete. You can re-test lock and confirm flow.")
        st.stop()

    # ✅ Automatically generate prompt blocks once providers are saved
    if (
        st.session_state.get("mail_locked") is True and
        st.session_state.get("trash_locked") is True and
        st.session_state.get("runbook_dates_confirmed") is True
    ):
        blocks = generate_all_prompt_blocks(section)
        st.session_state[f"{section}_runbook_blocks"] = blocks

        #Step 2: Generate DOCX
        include_priority = st.session_state["include_priority"] # Ensure default for include_priority

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