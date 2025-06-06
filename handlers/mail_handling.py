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
from utils.data_helpers import register_task_input, get_answer, extract_providers_from_text, check_missing_utility_inputs, render_lock_toggle
from utils.debug_utils import debug_all_sections_input_capture_with_summary, clear_all_session_data, debug_single_get_answer
from utils.runbook_generator_helpers import generate_docx_from_prompt_blocks, maybe_render_download, maybe_generate_runbook
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
def mail(section="mail"):
    st.subheader("📥 Mail & Package Instructions")

    # 🔒 Lock/unlock toggle
    render_lock_toggle(section="mail", session_key="mail_locked", label="Mail Info")

    # Determine whether inputs are editable
    disabled = st.session_state.get("mail_locked", False)

    with st.expander("📋 Details", expanded=True):
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

def trash_handling(section="trash_handling"):
    st.subheader("🗑️ Trash & Recycling Instructions")

    # 🔒 Lock/unlock toggle
    render_lock_toggle(section="trash_handling", session_key="trash_locked", label="Trash Info")

    # Determine whether inputs are editable
    disabled = st.session_state.get("trash_locked", False)
    
    # ─── Indoor Trash Disposal ────────────────────────────────
    with st.expander("Indoor Garbage Disposal (Kitchen, Recycling, Compost, Bath, Other)", expanded=True):
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
    with st.expander("Garbage & Recycling Pickup Details", expanded=True):
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
    with st.expander("Single-family homes only", expanded=True):
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
    with st.expander("Waste Management Contact Info", expanded=True):
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


# --- Main Function Start ---