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
from utils.data_helpers import register_task_input, get_answer, extract_providers_from_text, check_missing_utility_inputs, select_runbook_date_range, sanitize_label, sanitize, section_has_valid_input
from utils.debug_utils import debug_all_sections_input_capture_with_summary, clear_all_session_data, debug_single_get_answer
from utils.runbook_generator_helpers import generate_docx_from_prompt_blocks, maybe_render_download, maybe_generate_runbook, render_runbook_preview_inline
from utils.common_helpers import get_schedule_utils, debug_saved_schedule_dfs
from utils.task_schedule_utils_updated import extract_and_schedule_all_tasks, extract_unscheduled_tasks_from_inputs_with_category, display_enriched_task_preview, save_task_schedules_by_type, load_label_map, normalize_label
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
        mail(section)

    # 🗑️ Trash
    from handlers.mail_trash import trash
    with st.expander("🗑️ Trash Handling", expanded=False):
        trash(section)

    if section_has_valid_input("mail_trash", min_entries=5):
        st.subheader("🧐 Customize, Review and Reward")

    #if st.session_state.get("enable_debug_mode"):
    #    debug_saved_schedule_dfs()

        choice, start_date, end_date, valid_dates = select_runbook_date_range()

        if start_date and end_date:
            st.session_state.update({
                "start_date": start_date,
                "end_date": end_date,
                "valid_dates": valid_dates
            })

            utils = get_schedule_utils()

            df = extract_unscheduled_tasks_from_inputs_with_category()
        # st.write("📦 Session task inputs:", st.session_state.get("task_inputs", []))
            #st.write("🧾 Extracted Raw Task DataFrame:", df)

            # 🆕 Universal scheduler
            combined_df = None
            refresh_preview = st.checkbox("🔄 Refresh Preview", value=True)

            if refresh_preview:
                combined_df = extract_and_schedule_all_tasks(valid_dates, utils)

                if combined_df is not None:
                    dupes = combined_df[combined_df.duplicated(subset=["Date", "Day", "clean_task", "task_type"], keep=False)]
                    st.write("🔁 Possible duplicate tasks scheduled:", dupes)

                    preview_df = combined_df.drop_duplicates(subset=["Date", "Day", "clean_task", "task_type"])
                    st.write("📆 Preview Scheduled Tasks:")
                    display_enriched_task_preview(combined_df)

                    save_task_schedules_by_type(combined_df)
                    st.write("📆 Scheduled Tasks:", combined_df)

                    st.session_state["combined_home_schedule_df"] = combined_df

            st.session_state.update({
            #    "mail_schedule_df": mail_df,
            #    "trash_schedule_df": trash_df,
                "combined_home_schedule_df": combined_df,
            #    "mail_schedule_markdown": generate_flat_home_schedule_markdown(mail_df),
            #    "trash_flat_schedule_md": generate_flat_home_schedule_markdown(trash_df),
            #    "home_schedule_markdown": generate_flat_home_schedule_markdown(combined_df),
            })

            if st.session_state.get("enable_debug_mode"):
                st.markdown("## 🔍 Task Preview: Raw vs Enriched")

                raw_df = extract_unscheduled_tasks_from_inputs_with_category()
                combined_df = st.session_state.get("combined_home_schedule_df")

                LABEL_MAP = load_label_map()
                if not df.empty:
                    st.markdown("### 🔍 Label → Clean Task Mappings")
                    for label in df["question"].dropna().unique():
                        norm_label = normalize_label(label)
                        cleaned = LABEL_MAP.get(norm_label, "⚠️ No match in LABEL_MAP")
                        st.text(f"Label: '{label}' → Normalized: '{norm_label}' → Cleaned: '{cleaned}'")

                else:
                    st.warning("⚠️ No raw tasks available to preview.")

                if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
                    st.markdown("### 📝 Raw Task Inputs")
                    st.dataframe(raw_df)

                    if isinstance(combined_df, pd.DataFrame) and not combined_df.empty:
                        st.markdown("### ✨ Enriched & Scheduled Tasks")
                        st.dataframe(combined_df)

                        st.markdown("### 🔁 Matched Task Diffs")
                        for i, row in raw_df.iterrows():
                            raw_q = str(row.get("question", "")).strip()
                            raw_a = str(row.get("answer", "")).strip()

                            matches = combined_df[
                                combined_df["question"].astype(str).str.strip() == raw_q
                            ]

                            if not matches.empty:
                                enriched_sample = matches.iloc[0]

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(f"**Raw Task {i+1}:**")
                                    st.write(f"**Question:** {raw_q}")
                                    st.write(f"**Answer:** {raw_a}")

                                with col2:
                                    st.markdown("**🪄 Enriched View**")
                                    st.write(f"**Clean Task:** {enriched_sample.get('clean_task', '')}")
                                    st.write(f"**Formatted Answer:** {enriched_sample.get('formatted_answer', '')}")
                                    st.write(f"**Inferred Days:** {enriched_sample.get('inferred_days', '')}")
                            else:
                                st.warning(f"⚠️ No enriched match for: `{raw_q}`")
                    else:
                        st.warning("⚠️ No enriched schedule found.")
                else:
                    st.warning("⚠️ No raw task inputs available.")
    