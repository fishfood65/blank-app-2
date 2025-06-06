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
from utils.data_helpers import register_task_input, get_answer, extract_providers_from_text, check_missing_utility_inputs, select_runbook_date_range, render_lock_toggle
from utils.debug_utils import debug_all_sections_input_capture_with_summary, clear_all_session_data, debug_single_get_answer
from utils.runbook_generator_helpers import generate_docx_from_prompt_blocks, maybe_render_download, maybe_generate_runbook
from utils.common_helpers import get_schedule_utils
from prompts.templates import utility_provider_lookup_prompt
from config.section_router import get_handler

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
def mail(section="mail"):
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
        if st.session_state.get("enable_debug_mode"):
            st.json(st.session_state.get("task_inputs", {}).get(section, {}))


def trash(section="trash"):
    # 🔁 Lock toggle
    if st.session_state.get("trash_locked", False):
        if st.button("🔓 Unlock Mail Form"):
            st.session_state["trash_locked"] = False
    else:
        if st.button("🔒 Lock Trash Form"):
            st.session_state["trash_locked"] = True

    # Status indicator
    status = "🔒 Locked" if st.session_state.get("trash_locked") else "✅ Editable"
    st.markdown(f"**Status:** {status}")

    disabled = st.session_state.get("trash_locked", False)
    
    # Form wrapper
    with st.form("trash_form"):
        # ─── Indoor Trash Disposal ────────────────────────────────
        st.write("Indoor Garbage Disposal (Kitchen, Recycling, Compost, Bath, Other)")
        register_task_input(
            label="🧴 Kitchen Garbage Bin", 
            input_fn=st.text_area,
            section=section,
            task_type="Indoor Trash", 
            is_freq=True,
            placeholder="E.g. Bin is under sink. Empty when full and on Monday nights before trash pickup day. Replacement bags under sink.",
            disabled=disabled
        )

        register_task_input(
            label="♻️ Indoor Recycling Bin(s)", 
            input_fn=st.text_area,
            section=section,
            task_type="Indoor Trash", 
            is_freq=True,
            placeholder="E.g. Bins are under sink. The one labeled Paper is for paper products.  The one labled contaniers is for plastic and glass containers. Empty when full and on Monday nights before trash pickup day. Don't forget to rinse containers.",
            disabled=disabled
        )

        register_task_input(
            label="🧃 Indoor Compost or Green Waste", 
            input_fn=st.text_area,
            section=section,
            task_type="Indoor Trash", 
            is_freq=True,
            placeholder="E.g. Bin is under sink and yellow in color. Empty when full and on Monday nights before trash pickup day. No plastic bags.",
            disabled=disabled
        )

        register_task_input(
            label="🧼 Bathroom Trash Bin", 
            input_fn=st.text_area,
            section=section,
            task_type="Indoor Trash", 
            is_freq=True,
            placeholder="E.g. Empty on Mondays before trash day. Replacment bags are under bathroom sink.",
            disabled=disabled
        )

        register_task_input(
            label="🪑 Other Room Trash Bins", 
            input_fn=st.text_area,
            section=section,
            task_type="Indoor Trash",
            is_freq=True,
            placeholder="E.g. Empty bedroom and office bins weekly on Mondays before trash day. Replacment bags are under bathroom sink.",
            disabled=disabled
        )

    # ─── Outdoor Trash Disposal ────────────────────────────────
        st.write("Garbage & Recycling Pickup Details")
        register_task_input(
            label="Where are the trash, recycling, and compost bins stored outside?", 
            input_fn=st.text_area,
            section=section, 
            task_type="Outdoor Trash",
            placeholder="E.g., 'Bins are kept in the side yard.' or 'Bins are located at the left corner of the complex'",
            disabled=disabled
        )
        register_task_input(
            label="How are the outdoor bins marked?", 
            input_fn=st.text_area, 
            section=section, 
            task_type="Outdoor Trash",  
            placeholder="E.g., 'Black bin with green lid is for compost.  Black bin with black lid is for garbage.  Grey bin with blue lid is for plastic and glass container recycling. Grey bin with yellow lid is for paper recycling.'",
            disabled=disabled 
        )
        register_task_input(
            label="Stuff to know before putting recycling or compost in the bins?", 
            input_fn=st.text_area,
            section=section,  
            task_type="Outdoor Trash", 
            placeholder="E.g., 'Separate the recycling into paper, containers (plastic, glass). Compost goes into the compost bin.'",
            disabled=disabled
        )

    # ─── Single-Family Disposal Area ─────────────────────────────────
        st.write("Single-family homes only")
        single_family_disposal = register_task_input(
            label="Is a Single-family home?",
            input_fn=st.checkbox,
            section=section,
            task_type="Outdoor Trash",
            disabled=disabled
        )
        if single_family_disposal and not disabled:
            register_task_input(
                label="When and where should garbage, recycling, and compost bins be placed for pickup?", 
                input_fn=st.text_area,
                section=section,  
                task_type="Outdoor Trash", 
                is_freq=True, 
                placeholder="E.g., 'Please wheel out all the bins from the garage to the end of the driveway on Monday night before garbage day.'",
                disabled=disabled, 
                )
            register_task_input(
                label="When and where should garbage, recycling, and compost bins be brought back in after pickup?", 
                input_fn=st.text_area,
                section=section, 
                task_type="Outdoor Trash", 
                is_freq=True, 
                placeholder="E.g.,'Please please pick up the bins from the driveway and return them to the garage on Tuesday night after garbage day.'",
                disabled=disabled, 
                )
    # ─── Waste Management Contact ─────────────────────────────
        st.write("Waste Management Contact Info")
        register_task_input(
            label="Waste Management Company Name", 
            input_fn=st.text_input, 
            section=section,
            task_type="Outdoor Trash", 
            disabled=disabled,
            )
        register_task_input(
            label="Contact Phone Number", 
            input_fn=st.text_input, 
            section=section,
            task_type="Outdoor Trash", 
            disabled=disabled,
            )
        register_task_input(
            label="When to Contact", 
            input_fn=st.text_area, 
            section=section,
            task_type="Outdoor Trash", 
            disabled=disabled,
            )
        submitted = st.form_submit_button("✅ Save Trash Info")

    if submitted:
        st.success("📬 Trash info saved!")
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
            st.experimental_rerun()

    # 🔍 Debugging output
    with st.expander("🧠 Session State (After Reset)"):
        st.json({k: str(v) for k, v in st.session_state.items()})

    section = st.session_state.get("section", "home")

    st.markdown(f"### Currently Viewing: {get_active_section_label(section)}")


# --- Use Expanders to Break out Groups ---

    # 📬 Mail
    from handlers.mail_trash import mail
    with st.expander("📬 Mail Handling", expanded=True):
        mail()

    # 🗑️ Trash
    from handlers.mail_trash import trash
    with st.expander("🗑️ Trash Handling", expanded=False):
        trash()

    # 🧐 Review
    st.markdown("### 🧐 Review & Reward")
    # Your placeholder or preview/export function here
    # e.g., render_schedule_preview(), render_prompt_preview(), etc.