from utils.common_helpers import (
    extract_all_trash_tasks_grouped, 
    extract_grouped_mail_task, 
    generate_flat_home_schedule_markdown,
    switch_section,
    get_schedule_placeholder_mapping,
    debug_saved_schedule_dfs
)
from prompts.templates import(
    utility_provider_lookup_prompt
)
from utils.data_helpers import (
    capture_input, 
    flatten_answers_to_dict, 
    get_answer, 
    extract_and_log_providers, 
    log_provider_result, 
    preview_input_data, 
    export_input_data_as_csv, 
    render_lock_toggle,
    daterange,
    get_filtered_dates,
    select_runbook_date_range,
    register_task_input,
    extract_providers_from_text,
    check_missing_utility_inputs
)
from utils.runbook_generator_helpers import (
    maybe_render_download,
    generate_docx_from_prompt_blocks,
)
from utils.task_schedule_utils_updated import (
    extract_and_schedule_all_tasks,
    get_schedule_utils,
    generate_flat_home_schedule_markdown,
    save_task_schedules_by_type,
    extract_unscheduled_tasks_from_inputs_with_category,
    load_label_map,
    normalize_label
)
from utils.preview_helpers import (
    display_enriched_task_preview, 
    edit_button_redirect,
    get_active_section_label
)
from config.sections import (
    SECTION_METADATA, 
    LLM_SECTIONS,
    check_home_progress
)
from config.section_router import get_handler, critical_documents_flow
from utils.prompt_block_utils import generate_all_prompt_blocks
import streamlit as st
import re
from mistralai import Mistral, UserMessage, SystemMessage
import os
import pandas as pd
from datetime import datetime, timedelta
import re
import time
from PIL import Image
import io
import uuid
import json

# Only include sections that have a numeric level
LEVEL_LABELS = {
    key: meta["label"]
    for key, meta in SECTION_METADATA.items()
    if isinstance(meta.get("level"), int)
}

# Reverse lookup: label → key
LABEL_TO_KEY = {label: key for key, label in LEVEL_LABELS.items()}

# Tuple of labels in level order (for UI display)
levels = tuple(
    LEVEL_LABELS[k] for k in sorted(LEVEL_LABELS, key=lambda x: SECTION_METADATA[x]["level"])
)

st.write("# Welcome to Home Hero Academy! 👋")

st.markdown(
    """
    ### Your Mission
    Accept a series of challenges testing your knowledge about your home to empower you and your deputies to become a heroic guardian for your home and its precious contents
    """
    )
st.markdown(
    """
    ### Start your Training!
    """
    )

# Main entry point of the app

def main():
    # 🛑 Redirect to a specific page if triggered elsewhere (e.g., edit buttons)
    if "current_page" in st.session_state:
        section = st.session_state["current_page"]
        del st.session_state["current_page"]
    else:
        # 🧭 Initialize default section based on sidebar selection
        selected_label = st.sidebar.radio(
            "Choose a Level:",
            list(LABEL_TO_KEY.keys()),
            key="selected_level"
        )
        section = LABEL_TO_KEY[selected_label]
        st.session_state["section"] = section

    # 📦 Progress tracking
    st.markdown("#### 🧭 Progress")
    if "level_progress" not in st.session_state:
        st.session_state["level_progress"] = {key: False for key in LEVEL_LABELS}
    percent_complete, completed_levels = check_home_progress(st.session_state["level_progress"])
    total_levels = len(st.session_state["level_progress"])
    num_completed = len(completed_levels)
    friendly_labels = [LEVEL_LABELS.get(level, level) for level in completed_levels]

    st.progress(percent_complete)
    st.markdown(
        f"🏆 **Completed {num_completed} out of {total_levels} levels** - {', '.join(friendly_labels) if friendly_labels else 'None'}"
    )

    # ⬇️ Optional export for debugging or preview
    export_input_data_as_csv()

    # ✅ Set runtime flags (e.g., for prompt generation)
    st.session_state["include_priority"] = True

    # 🧭 Route to the selected handler
    handler = get_handler(section)
    if callable(handler):
        handler()
    else:
        # 🔁 Fallback if section is invalid or handler is missing
        st.warning(f"Section `{section}` not found. Redirecting to Home.")
        st.session_state["section"] = "home"
        st.rerun()

    #else:
        #if st.session_state.get("enable_debug_mode"):
        #    st.warning("⚠️ Section handler not defined.")

    # Limit access to Level 1 until completed
    #if not st.session_state["level_progress"]["home"]:
    #    available_levels = [LEVEL_LABELS["home"]]
    #else:
    #    available_levels = levels 

    # 🔧 TEMP DEV MODE OVERRIDE
    dev_mode = True  # Set to False to re-enable lock

    if dev_mode:
        available_levels = levels
    else:
        if not st.session_state["level_progress"]["home"]:
            available_levels = [LEVEL_LABELS["home"]]
        else:
            available_levels = levels

    #if "task_inputs" in st.session_state:
     #   st.session_state["task_inputs"] = []

    #if st.session_state.get("enable_debug_mode"): # DEBUG
    #    st.markdown("### 🧪 Debug: Section Key")
    #    st.write("🔍 Current section key:", section)

    if "current_page" in st.session_state:
        nav_page = st.session_state.pop("current_page")
        st.switch_page(f"{nav_page}.py")  # or use your page navigation logic
        st.stop()  # ✅ Prevent executing the rest of this page after switching

    # === Levels 5 and Bonus ===
    if st.session_state.get("enable_debug_mode"): # DEBUG
        st.write("📦 Available sections in metadata:", list(SECTION_METADATA.keys()))
        st.write("🎯 Current section from session:", section)

###### Main Functions that comprise of the Levels

def render_prompt_preview(missing: list, section: str = "home"):
    confirmed = st.session_state.get(f"{section}_user_confirmation", False)

    with st.expander("🧠 AI Prompt Preview (Optional)", expanded=True):
        if missing:
            st.warning(f"⚠️ Cannot generate prompt. Missing: {', '.join(missing)}")
            return

        if not confirmed:
            st.info("☕️ Please check the box to confirm AI prompt generation.")
            return

        prompt = st.session_state.get("generated_prompt", "")
        prompt_blocks = st.session_state.get("prompt_blocks", [])
        schedule_md = st.session_state.get("home_schedule_markdown", "_No schedule available._")

        if not prompt:
            st.warning("⚠️ Prompt not generated yet.")
            return

        # Build combined preview by inserting schedule into the final prompt block
        if prompt_blocks:
            full_preview = "\n\n".join(prompt_blocks)
            full_preview = full_preview.replace("<<INSERT_SCHEDULE_TABLE>>", schedule_md)
        else:
            full_preview = prompt.replace("<<INSERT_SCHEDULE_TABLE>>", schedule_md)

        # Display the formatted full prompt preview
        st.markdown(full_preview)
        if section == "mail_trash_handling":
            st.markdown("### 📋 Schedule Preview")
            st.markdown(st.session_state.get("home_schedule_markdown", "_No schedule available._"))
        st.success("✅ Prompt ready! This is what will be sent to the LLM.")


##### Level 3 - Mail Handling and Trash

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

def mail_trash_handling():
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
    # Optionally validate against SECTION_METADATA
    if section not in SECTION_METADATA:
        section = "home"  # or raise a warning/log it

    st.markdown(f"### Currently Viewing: {get_active_section_label(section)}")
    switch_section("mail_trash_handling")

# --- Use subsection labels ---
    subsections = SECTION_METADATA[section].get("subsections", {})
    mail_label = subsections.get("mail", "📬 Mail")
    trash_label = subsections.get("trash_handling", "🗑️ Trash")

    tab1, tab2, tab3 = st.tabs([mail_label, trash_label, "🧐 Review & Reward"])

    with tab1:
        mail(section="mail")

    with tab2:
        trash_handling(section="trash_handling")

    with tab3:
        st.subheader("🎁 Customize, Review and Reward")

        if st.session_state.get("enable_debug_mode"):
            debug_saved_schedule_dfs()

        choice, start_date, end_date, valid_dates = select_runbook_date_range()

        if start_date and end_date:
            st.session_state.update({
                "start_date": start_date,
                "end_date": end_date,
                "valid_dates": valid_dates
            })

            utils = get_schedule_utils()

            df = extract_unscheduled_tasks_from_inputs_with_category()
            st.write("📦 Session task inputs:", st.session_state.get("task_inputs", []))
            st.write("🧾 Extracted Raw Task DataFrame:", df)

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

            # Step 7: Confirm and maybe generate prompt
            st.subheader("👍 Review and Approve")

            # Set and track confirmation state
            confirm_key = f"confirm_ai_prompt_{section}"
            user_confirmation = st.checkbox("✅ Confirm AI Prompt", key=confirm_key)
            st.session_state[f"{section}_user_confirmation"] = user_confirmation

            # Generate prompts if confirmed
            if user_confirmation:
                combined_prompt, prompt_blocks = maybe_generate_prompt(section=section)

                if st.session_state.get("enable_debug_mode"):
                    st.markdown("### 🧾 Prompt Preview")
                    st.code(combined_prompt or "", language="markdown")

            # Step 8: Prompt preview + runbook
            missing = check_missing_utility_inputs()
            render_prompt_preview(missing, section=section)

            # Step 9: Optionally generate runbook if inputs are valid and confirmed
            st.subheader("🎉 Reward")
            if not missing and st.session_state.get("generated_prompt"):
                maybe_generate_runbook(section=section)
                # Level 3 Complete - for Progress
                st.session_state["level_progress"]["mail_trash_handling"] = True
      
##### Level 4 - Home Security and Services

def home_security():
    section = "Home Security"
    st.write("💝 Security-Conscious")

    # 🔒 Lock/unlock toggle
    render_lock_toggle(session_key="home_security_locked", label="Home Security Info")
    disabled = st.session_state.get("home_security_locked", False)

    with st.expander("Home Security System and Privacy Info", expanded=True):
        st.markdown("##### Home Security and Privacy Info")

        security_used = st.checkbox(
            "Are you privacy (i.e. your Internet Wifi has a password) and home security conscious?",
            key="home_security_used"
        )

        if not security_used:
            st.info("✅ No privacy or security info needed.")
            return  # Skip the rest if unchecked

        # Only shown if security_used is True
        register_task_input(
            label="Where is the Wi-Fi name and password kept?", 
            input_fn=st.text_input,
            section=section,
            task_type="Internet Wifi", 
            disabled=disabled,
        )
        register_task_input(
            label="How is Wi-Fi info shared with guests?", 
            input_fn=st.text_input,
            section=section, 
            task_type="Internet Wifi",
            disabled=disabled
        )
        register_task_input(
            label="Security Company Name", 
            input_fn=st.text_input,
            section=section, 
            task_type="Security System",
            disabled=disabled,
        )
        register_task_input(
            label="Security Company Phone Number", 
            input_fn=st.text_input,
            section=section, 
            task_type="Security System",
            disabled=disabled
        )
        register_task_input(
            label="How is the alarm system turned on and off?", 
            input_fn=st.text_area,
            section=section, 
            task_type="Security System",
            disabled=disabled,
        )
        register_task_input(
            label="What should be done if the alarm goes off or triggers a security alert?", 
            input_fn=st.text_area,
            section=section, 
            task_type="Security System",
            disabled=disabled,
        )
        register_task_input(
            label="Where are indoor security cameras or motion sensors, if any?", 
            input_fn=st.text_area,
            section=section, 
            task_type="Security System",
            disabled=disabled,
        )
        register_task_input(
            label="What's the emergency entry method for the home (spare key, lockbox, or smart lock)? Where is it located, and how is it used?", 
            input_fn=st.text_area,
            section=section, 
            task_type="Security System",
            disabled=disabled,
        )
        register_task_input(
            label="Is there a home phone? If so, what should be done if it stops working?", 
            input_fn=st.text_area,
            section=section, 
            task_type="Home Phone",
            disabled=disabled,
        )

def convenience_seeker():
    section = "Regular Help Around the House (Contacts + Instructions)"
    st.write("🧼 Regular Help Around the House (Contacts + Instructions)")

    render_lock_toggle(session_key="convenience_seeker_locked", label="Household Services")
    disabled = st.session_state.get("convenience_seeker_locked", False)

    with st.expander("Regular Help Around the House (Contacts + Instructions)", expanded=True):
        st.markdown("##### Home Services")
        services = ["Cleaning", "Gardening/Landscape", "Pool Maintenance"]

        # Track selected services as a task
        selected_services = register_task_input(
            label="Which of these regular services do you use?",
            input_fn=st.multiselect,
            section=section,
            task_type="Service",
            options=services,
            default=[],
            key="selected_services",
            disabled=disabled
        )
    
        for service in selected_services:
            nested_section = f"{section}.{service}"
            st.subheader(f"🔧 {service} Service Info")

            questions = [
                ("Company Name", st.text_input),
                ("Company Phone Number", st.text_input),
                ("How do home service providers access the home?", st.text_input),
                ("How should service providers be let in, and what should be done after they finish?", st.text_area),
                ("How can approved service providers be identified or confirmed?", st.text_area),
            ]

            for label, input_fn in questions:
                register_task_input(
                    label=label,
                    input_fn=input_fn,
                    section=nested_section,
                    task_type="Service",
                    disabled=disabled
                )

            if service in ["Cleaning", "Gardening/Landscape", "Pool Maintenance"]:
                register_task_input(
                    label="What’s the usual timing for home service visits?",
                    input_fn=st.text_area,
                    section=nested_section,
                    task_type="Service",
                    is_freq=True,
                    disabled=disabled
                )

def rent_own():
    section = "Rent or Own"
    st.write("🏠 Home Ownership Status")

    # Radio input: Rent or Own
    housing_status = st.radio(
        "Do you rent or own your home?",
        options=["Rent", "Own"],
        index=0,
        key="housing_status"
    )
    capture_input("Housing Status", lambda label, **kwargs: housing_status, section)

    if housing_status == "Rent":
        st.subheader("🏢 Property Management Info")
        rent_section = f"{section}.Property Management"
        register_task_input("Company Name", st.text_input, section=rent_section, task_type="Property Management")
        register_task_input("Company Phone Number", st.text_input, section=rent_section, task_type="Property Management")
        register_task_input("Company Email", st.text_input, section=rent_section, task_type="Property Management")
        register_task_input("When to Contact", st.text_area, section=rent_section, task_type="Property Management")

    elif housing_status == "Own":
        st.subheader("🧰 Homeowner Contacts")
        services = ["Handyman", "Electrician", "Exterminator", "Plumber", "HOA"]
        selected_services = st.multiselect(
            "Which service contacts are applicable?",
            options=services,
            default=[],
            key="homeowner_contacts_options"
        )

        for role in selected_services:
            role_section = f"{section}.{role}"
            st.subheader(f"🔧 {role}")
            register_task_input("Name", st.text_input, section=role_section, task_type=role)
            register_task_input("Phone Number", st.text_input, section=role_section, task_type=role)
            register_task_input("When to Contact", st.text_area, section=role_section, task_type=role)

        if "HOA" in selected_services:
            st.subheader("🏘️ HOA / Property Management")
            hoa_section = f"{section}.HOA"
            register_task_input("Company Name", st.text_input, section=hoa_section, task_type="HOA")
            register_task_input("Phone Number", st.text_input, section=hoa_section, task_type="HOA")
            register_task_input("Email", st.text_input, section=hoa_section, task_type="HOA")
            register_task_input("When to Contact", st.text_area, section=hoa_section, task_type="HOA")

def security_convenience_ownership():
    # 🔪 Optional: Reset controls for testing
    if st.checkbox("🔪 Reset Security and Home Services Session State"):
        for key in ["generated_prompt", "runbook_buffer", "runbook_text", "user_confirmation"]:
            st.session_state.pop(key, None)
        st.success("🔄 Level 4 session state reset.")
        st.stop()  # 🔁 prevent rest of UI from running this frame

    section = st.session_state.get("section", "home")
    switch_section("home_security")

    st.subheader("Level 4: Home Security, Privacy, Quality-Orientation, and Support")

    # Step 1: User Input
    st.write("🗑️ [DEBUG] input_data:", st.session_state.get("input_data", {}))
    home_security()
    convenience_seeker()
    rent_own()
    security_home_services_runbook()

def security_home_services_runbook():
    if st.session_state.get("enable_debug_mode"):
        debug_saved_schedule_dfs()
    
    section = st.session_state.get("section", "home")
    
    # Step 1: Let user select the date range
    choice, start_date, end_date, valid_dates = select_runbook_date_range()

    # st.write("🗓️ valid_dates selected:", valid_dates)

    if start_date and end_date:
        st.session_state["start_date"] = start_date
        st.session_state["end_date"] = end_date
        st.session_state["valid_dates"] = valid_dates

        # Build once and reuse
        utils = get_schedule_utils()

        # Step 2: Extract grouped schedule summaries for mail and trash
        mail_task = extract_grouped_mail_task(valid_dates)
        st.markdown("### 📬 Debug: Raw `mail_task` Output")
        st.write(mail_task)
        trash_df = extract_all_trash_tasks_grouped(valid_dates, utils)

        # Step 3: Convert mail task to DataFrame
        #st.write("📬 mail_task raw output:", mail_task)
        #st.write("🗑️ trash_df raw output (pre-check):", trash_df)
        mail_df = pd.DataFrame(mail_task) if mail_task else pd.DataFrame(
            columns=["Task", "Category", "Area", "Source", "Tag", "Date", "Day"]
        )
        st.session_state["mail_schedule_df"] = mail_df

        # Step 4: Combine into one unified DataFrame
        combined_schedule_df= pd.concat([mail_df, trash_df], ignore_index=True)

        # This merges the structured outputs (from mail and trash task extraction functions) into a single pandas DataFrame. It's used to:
        #   Provide a master table of all scheduled tasks.
        #   Be passed into:
        #   - The markdown generator for preview display.
        #   - The DOCX generator for inserting an actual table.
        #   - "combined_home_schedule_df" for 
        #       - add_table_from_schedule() for generating table in DOCX
        #💡 Why it's needed:
        #   You want both mail and trash tasks aligned by date, tagged, and grouped for rendering and prompt generation.

        # Step 4a: Create individual DataFrame and generate flat markdown for rendering or prompt
        if "Date" not in mail_df.columns:
            st.error("❌ Mail schedule missing 'Date' column.")
            st.write("📊 mail_schedule_df:", mail_df)
            mail_flat_schedule_md = "_Schedule data incomplete or malformed._"
        # Generate flat markdown for rendering or prompt
        else:
            mail_flat_schedule_md = generate_flat_home_schedule_markdown(mail_df)

        if "Date" not in trash_df.columns:
            st.error("❌ Trash schedule missing 'Date' column.")
            st.write("📊 trash_schedule_df:", trash_df)
            trash_flat_schedule_md = "_Schedule data incomplete or malformed._"
        
        # Step 5: Generate flat markdown for rendering or prompt
        else:
            trash_flat_schedule_md = generate_flat_home_schedule_markdown(trash_df)

        if "Date" not in combined_schedule_df.columns:
            st.error("❌ Combined schedule missing 'Date' column.")
            st.write("📊 combined_schedule_df:", combined_schedule_df)
            flat_schedule_md = "_Schedule data incomplete or malformed._"
        # Step 5: Generate flat markdown for rendering or prompt
        else:
            flat_schedule_md = generate_flat_home_schedule_markdown(combined_schedule_df)
        # Used in:
        #   - Prompt preview (render_prompt_preview) so users can visually confirm the schedule.
        #   - Prompt string via .replace("<<INSERT_SCHEDULE_TABLE>>", flat_schedule_md) — so the LLM sees a human-readable version of the schedule when generating the runbook. 
        #   - "home_schedule_markdown" 
        #       - render_prompt_preview() for prompt preview and prompt substitution

        # Step 6: Store all data into session
        st.session_state.update({
            "grouped_mail_task": mail_task,
            "grouped_trash_schedule": trash_df,
            "combined_home_schedule_df": combined_schedule_df,
            "home_schedule_markdown": flat_schedule_md,
            "mail_schedule_df": mail_df,
            "mail_schedule_markdown": mail_flat_schedule_md,
            "trash_schedule_df": trash_df,
            "trash_flat_schedule_md": trash_flat_schedule_md
        })
        mail_df = st.session_state.get("mail_schedule_df")

        if isinstance(mail_df, pd.DataFrame):
            if not mail_df.empty:
                st.write("📬 Mail Schedule DataFrame:")
                st.dataframe(mail_df)
            else:
                st.warning("⚠️ mail_schedule_df is registered but empty.")
        else:
            st.error("❌ mail_schedule_df is not a valid DataFrame or not found.")
        
        mail_flat_schedule_md = generate_flat_home_schedule_markdown(mail_df)
        st.session_state["mail_schedule_markdown"] = mail_flat_schedule_md


        # 🧪 Preview what’s going into the prompt -- Debug
        #st.markdown("### 🧪 Flat Schedule Markdown Preview")
        # st.code(flat_schedule_md, language="markdown")

        #st.write(f"🧪 show what is saved in 'grouped_mail_task': {mail_task}")
        #st.write(f"🧪 show what is saved in 'grouped_trash_schedule': {trash_df}")
        #st.write(f"🧪 show what is saved in 'combined_home_schedule_df': {combined_schedule_df}")
        #st.write(f"🧪 show what is saved in 'home_schedule_markdown': {flat_schedule_md}")


        # Step 7: Confirm and maybe generate prompt
        st.subheader("👍 Review and Approve")

        # Set and track confirmation state
        confirm_key = f"confirm_ai_prompt_{section}"
        user_confirmation = st.checkbox("✅ Confirm AI Prompt", key=confirm_key)
        st.session_state[f"{section}_user_confirmation"] = user_confirmation

        # Generate prompts if confirmed
        if user_confirmation:
            combined_prompt, prompt_blocks = maybe_generate_prompt(section=section)

            if st.session_state.get("enable_debug_mode"):
                st.markdown("### 🧾 Prompt Preview")
                st.code(combined_prompt or "", language="markdown")

        # Step 8: Prompt preview + runbook
        missing = check_missing_utility_inputs()
        render_prompt_preview(missing, section=section)

        # Step 9: Optionally generate runbook if inputs are valid and confirmed
        st.subheader("🎉 Reward")
        if not missing and st.session_state.get("generated_prompt"):
            maybe_generate_runbook(section=section)
            # Level 3 Complete - for Progress
            st.session_state["level_progress"]["mail_trash_handling"] = True

##### Level 5 - Emergency Kit Critical Documents

def emergency_kit_critical_documents():
    # Define categories and the corresponding documents
    documents = {
        'Identification Documents': [
            'Government-issued ID (Driver’s license, state ID)',
            'Social Security Card',
            'Birth Certificates',
            'Marriage/Divorce Certificates',
            'Citizenship/Immigration Documents',
            'Passport'
        ],
        'Health and Medical Documents': [
            'Health Insurance Cards',
            'Prescription Medications List',
            'Vaccination Records',
            'Emergency Medical Information',
            'Medical Power of Attorney'
        ],
        'Financial Documents': [
            'Bank Account Information',
            'Credit Cards/Debit Cards',
            'Checkbook',
            'Tax Returns (Last Year’s)',
            'Insurance Policies (Auto, Health, Home, Life, etc.)',
            'Investment Documents'
        ],
        'Homeownership or Rental Documents': [
            'Deed or Lease Agreement',
            'Mortgage or Rent Payment Records',
            'Home Insurance Policy'
        ],
         'Legal Documents': [
            'Will or Living Will',
            'Power of Attorney',
            'Property Title and Vehicle Titles',
            'Child Custody or Adoption Papers'
        ],
        'Emergency Contact Information': [
            'Contact List',
            'Emergency Plan'
        ],
        'Travel Documents': [
            'Travel Itinerary'
        ],
        'Educational Documents': [
            'School Records',
            'Diplomas and Degrees',
            'Certificates and Licenses'
        ],
        'Significant Documents': [
            'Pet Records',
            'Photos of Important Belongings',
            'Bankruptcy or Legal Filings'
        ]

    }

    # Initialize session state
    if "selected_documents" not in st.session_state:
        st.session_state.selected_documents = {}

    # 1) Category picker
    selected_category = st.selectbox(
        'Select a document category to view:',
        options=list(documents.keys()),
        key="selected_category"
    )

    # 2) Docs multiselect for that category
    if selected_category:
        st.write(f'You selected **{selected_category}**')

        # 2) Action buttons placed before the multiselect
        col1, col2 = st.columns(2)
        if col1.button('Add more categories', key="btn_add_more"):
            st.info("Pick another category above.")
        if col2.button('Finalize and Save All Selections', key="btn_finalize"):
            st.session_state.finalized = True

        # 3) Multi-select segmented control via horizontal checkboxes
        options = documents[selected_category]
        default = st.session_state.selected_documents.get(selected_category, [])
        cols = st.columns(len(options))
        new_picks = []
        for idx, opt in enumerate(options):
            # each checkbox lives in its own column
            checked = cols[idx].checkbox(
                opt,
                value=(opt in default),
                key=f"chk_{selected_category}_{idx}"
            )
            if checked:
                new_picks.append(opt)

        # save back
        st.session_state.selected_documents[selected_category] = new_picks

    # 5) If finalized, show all
    if st.session_state.get("finalized", False):
        st.header("✅ All Your Selections")
        for cat, docs in st.session_state.selected_documents.items():
            st.subheader(cat)
            for d in docs:
                st.write(f"• {d}")

def review_selected_documents():
    saved = st.session_state.get("selected_documents", {})
    if not saved:
        st.warning("No selections to review.")
        return

    st.header("📋 Review Selections")
    for cat, docs in saved.items():
        st.write(f"**{cat}:** {', '.join(docs)}")

    all_docs = [d for docs in saved.values() for d in docs]
    st.multiselect(
        "Tweak your list:",
        options=all_docs,
        default=all_docs,
        key="tweaked_docs"
    )
    if st.button("Save Tweaks", key="btn_save_tweaks"):
        st.success("Tweaks saved!")

def collect_document_details():
    selected = st.session_state.get("selected_documents", {})
    if not selected:
        st.warning("No documents selected. Go pick some first!")
        return

    # Initialize storage-confirmed flag
    if "storage_confirmed" not in st.session_state:
        st.session_state.storage_confirmed = False

    st.header("🗂 Document Access & Storage Details")

    PHYSICAL_STORAGE_OPTIONS = [
        "Canister","Closet","Drawer","Filing Cabinet","Handbag","Safe",
        "Safety Deposit Box","Storage Unit","Wallet","With Attorney", "With Financial Advisor/Accountant", "Other physical location"
    ]
    DIGITAL_STORAGE_OPTIONS = [
        "Computer/Tablet","Phone","USB flash drive","External hard drive",
        "Cloud storage (Google Drive, Dropbox, etc.)","Password Manager", "Mobile Application(s)", "Other digital location"
    ]

    # --- Step 0: Pick which storage types you use ---
    use_physical = st.checkbox("I use physical storage for my documents", key="use_physical")
    if use_physical:
        st.multiselect(
            "Select all physical storage locations you use:",
            options=PHYSICAL_STORAGE_OPTIONS,
            default=st.session_state.get("global_physical_storage", []),
            key="global_physical_storage"
        )

    use_digital = st.checkbox("I use digital storage for my documents", key="use_digital")
    if use_digital:
        st.multiselect(
            "Select all digital storage locations you use:",
            options=DIGITAL_STORAGE_OPTIONS,
            default=st.session_state.get("global_digital_storage", []),
            key="global_digital_storage"
        )

    # --- Step 0b: Confirm storage setups ---
    if st.button("Confirm storage types & locations", key="btn_confirm_storage"):
        errors = []
        if use_physical and not st.session_state.get("global_physical_storage"):
            errors.append("• select at least one physical storage location")
        if use_digital and not st.session_state.get("global_digital_storage"):
            errors.append("• select at least one digital storage location")
        if errors:
            st.error("Please:\n" + "\n".join(errors))
        else:
            st.session_state.storage_confirmed = True

    if not st.session_state.storage_confirmed:
        st.info("After selecting storage types & locations above, click **Confirm** to assign documents.")
        return

    # --- Step 1: Assign each document to chosen locations ---
    if "document_details" not in st.session_state:
        st.session_state.document_details = {}

    st.markdown("### Assign each document to one or more storage locations")
    all_assigned = True
    missing = []

    for category, docs in selected.items():
        if not docs:
            continue

        # Wrap the entire category in an expander
        with st.expander(category, expanded=False):
            for doc in docs:
                details = st.session_state.document_details.setdefault(doc, {})

                # Build the options from global storage lists
                options = []
                if use_physical:
                    options += st.session_state["global_physical_storage"]
                if use_digital:
                    options += st.session_state["global_digital_storage"]

                st.markdown(f"📄 **{doc}** — assign storage:")

                # Horizontal checkboxes, 4 per row
                picked = []
                for start in range(0, len(options), 4):
                    chunk = options[start : start + 4]
                    cols = st.columns(len(chunk))
                    for idx, opt in enumerate(chunk):
                        was = details.get("assigned_storage", [])
                        checked = cols[idx].checkbox(
                            opt,
                            value=(opt in was),
                            key=f"assign_{doc}_chk_{start+idx}"
                        )
                        if checked:
                            picked.append(opt)

                details["assigned_storage"] = picked

                if not picked:
                    all_assigned = False
                    missing.append(doc)

    # --- Step 2: Enforce that every document got assigned ---
    if not all_assigned:
        st.error("Please assign storage for all documents:")
        st.write(", ".join(missing))
        return

    # --- Step 3: Final save button ---
    if st.button("Save all document details", key="btn_save_details"):
        st.success("✅ All document details saved!")

    # Step 4: Ask storage-location questions
    st.header("🔍 Storage Location Details")

    # PHYSICAL STORAGE
    for storage in st.session_state.get("global_physical_storage", []):
        # normalize key name
        key_base = storage.lower().replace(" ", "_")
        with st.expander(f"{storage} Details", expanded=False):
            if storage == "Safety Deposit Box":
                st.text_input(
                    "Branch name & address:",
                    key=f"{key_base}_branch_address"
                )
                st.text_area(
                    "Emergency authorization required to retrieve contents:",
                    key=f"{key_base}_authorization"
                )

            elif storage == "Safe":
                st.text_input(
                    "Designated safe location (building/room/area):",
                    key=f"{key_base}_location"
                )
                st.text_area(
                    "Emergency steps & credentials needed to open safe:",
                    key=f"{key_base}_access_steps"
                )

            elif storage == "Storage Unit":
                st.text_input(
                    "Business name & address of unit:",
                    key=f"{key_base}_business_address"
                )
                st.text_area(
                    "Emergency authorization required for unit access:",
                    key=f"{key_base}_authorization"
                )

            elif storage == "With Attorney":
                st.text_area(
                    "Emergency contact method and proof of authorization needed:",
                    key=f"{key_base}_attorney_instructions"
                )

            elif storage == "Canister":
                st.text_input(
                    "Primary Canister location (building/room/cabinet/shelf):",
                    key=f"{key_base}_location"
                )
                st.text_area(
                    "Emergency steps & credentials needed to open canister:",
                    key=f"{key_base}_access_steps"
                )
                st.text_area(
                    "If secondary canisters are used , list each canister name & its location & contents:",
                    key=f"{key_base}_contents"
                )

            elif storage == "Drawer":
                st.text_input(
                    "Primary Drawer location (building/room/cabinet):",
                    key=f"{key_base}_location"
                )
                st.text_area(
                    "Emergency steps & credentials needed to open drawer:",
                    key=f"{key_base}_access_steps"
                )
                st.text_area(
                    "If secondary drawers are used , list each drawer name & its location & contents:",
                    key=f"{key_base}_contents"
                )

            elif storage == "Filing Cabinet":
                st.text_input(
                    "Primary Filing cabinet location (building/room/identifier):",
                    key=f"{key_base}_location"
                )
                st.text_area(
                    "Emergency steps & credentials needed to open cabinet:",
                    key=f"{key_base}_access_steps"
                )
                st.text_area(
                    "If secondary filing cabinets are used, list each cabinet name & its location & contents:",
                    key=f"{key_base}_contents"
                )

            elif storage == "Wallet":
                st.text_input(
                    "Wallet location (building/room/drawer/closet/bag):",
                    key=f"{key_base}_location"
                )
                st.text_area(
                    "Emergency steps & credentials to retrieve wallet:",
                    key=f"{key_base}_access_steps"
                )
                st.text_area(
                    "If secondary wallets are used, list each & its location & contents:",
                    key=f"{key_base}_contents"
                )

            elif storage == "Handbag":
                st.text_input(
                    "Primary Handbag location (building/room/drawer/closet):",
                    key=f"{key_base}_location"
                )
                st.text_area(
                    "Emergency steps & credentials to retrieve handbag:",
                    key=f"{key_base}_access_steps"
                )
                st.text_area(
                    "If secondary handbags are used, list its location & contents:",
                    key=f"{key_base}_contents"
                )

            elif storage == "Other physical location":
                st.text_input(
                    "Other location description (building/room/address):",
                    key=f"{key_base}_location"
                )
                st.text_area(
                    "Emergency steps & credentials to access this location:",
                    key=f"{key_base}_access_steps"
                )
                st.text_area(
                    "If multiple, list each location & its contents:",
                    key=f"{key_base}_contents"
                )

    # DIGITAL STORAGE
    for storage in st.session_state.get("global_digital_storage", []):
        key_base = storage.lower().replace(" ", "_").replace("/", "_")
        with st.expander(f"{storage} Details", expanded=False):
            if storage in ["Computer/Tablet", "Phone"]:
                st.text_input(
                    "Designated place (room/surface/storage)primary device:",
                    key=f"{key_base}_location"
                )
                st.text_area(
                    "Emergency steps & credentials to access device:",
                    key=f"{key_base}_access_steps"
                )
                st.text_area(
                    "If secondary devices exists, list each, its location and contents:",
                    key=f"{key_base}_contents"
                )

            elif storage in ["USB flash drive", "External hard drive"]:
                st.text_input(
                    "Designated place (room/surface/storage) for primary device:",
                    key=f"{key_base}_location"
                )
                st.text_area(
                    "Emergency steps & credentials to access drive:",
                    key=f"{key_base}_access_steps"
                )
                st.text_area(
                    "If secondary devices exists, list each & its location and contents:",
                    key=f"{key_base}_contents"
                )

            elif storage == "Cloud storage (Google Drive, Dropbox, etc.)":
                st.text_input(
                    "Primary Cloud platform name & link:",
                    key=f"{key_base}_platform"
                )
                st.text_area(
                    "Emergency steps & credentials to access account:",
                    key=f"{key_base}_access_steps"
                )
                st.text_area(
                    "If secondary platforms, list each & its link and contents:",
                    key=f"{key_base}_contents"
                )

            elif storage == "Password Manager":
                st.text_input(
                    "Password manager name:",
                    key=f"{key_base}_platform"
                )
                st.text_area(
                    "Emergency steps & credentials to access vault:",
                    key=f"{key_base}_access_steps"
                )
                st.text_area(
                    "If multiple vaults, list each & its contents:",
                    key=f"{key_base}_contents"
                )

            elif storage == "Mobile Application(s)":
                # New Mobile Application questions
                st.text_area(
                    "If multiple mobile applications are used, name each and note what is stored in each:",
                    key=f"{key_base}_apps_and_contents"
                )
                st.text_area(
                    "In an emergency, what steps and credentials are required for someone else to access the mobile application accounts holding key documents?:",
                    key=f"{key_base}_access_steps"
                )

    # Merge all storage‐location keys into document_details
    for doc, details in st.session_state.document_details.items():
        # for every storage the user selected
        for storage in st.session_state.get("global_physical_storage", []) \
                       + st.session_state.get("global_digital_storage", []):
            key_base = storage.lower().replace(" ", "_").replace("/", "_")
            # list every suffix you might have used
            for suffix in [
                "branch_address", "authorization",
                "location", "business_address",
                "attorney_instructions",
                "access_steps", "contents",
                "apps_and_contents", "platform"
            ]:
                full_key = f"{key_base}_{suffix}"
                if full_key in st.session_state:
                    # copy it into the per‐doc details dict
                    details[full_key] = st.session_state[full_key]

    # Final Save
    if st.button("Save all document & storage details", key="btn_save_all"):
        st.success("✅ All details saved!")


def generate_kit_tab():
    section = st.session_state.get("section", "home")
    switch_section("emergency_kit_critical_documents")

    """Renders the Generate Kit UI and uses generate_runbook_from_prompt to run the LLM and export."""
    st.header("📦 Generate Emergency Document Kit")

    # 1) Build and show the prompt (optional—you can hide this if you don't want the user to see it)
    prompt = emergency_kit_document_prompt()
    with st.expander("Preview LLM prompt", expanded=False):
        st.code(prompt, language="markdown")

    # 2) Ask the user to confirm before sending
    st.checkbox("✅ I confirm this prompt is correct", key="user_confirmation")

    # 3) Delegate to your reusable runbook function
    generate_runbook_from_prompt(
        prompt=prompt,
        api_key=os.getenv("MISTRAL_TOKEN"),
        button_text="Generate Emergency Kit Runbook",
        doc_heading="Emergency Document Kit",
        doc_filename="emergency_document_kit.docx"
    )

##### Bonus - Additional Instructions for Guest/House Sitters

def bonus_level():
    st.write("🎁 Bonus Level")

    # ─── Initialize session_state keys ────────────────────────
    st.session_state.setdefault('bonus_info', {})
    st.session_state.setdefault('prompt_emergency', None)
    st.session_state.setdefault('prompt_bonus', None)
    st.session_state.setdefault('prompt_mail_trash', None)
    st.session_state.setdefault('prompt_home_caretaker', None)

    # Confirmation flag for generation
    st.session_state.setdefault('bonus_generate_confirm', False)

    # Ensure progress flags exist
    #st.session_state.progress.setdefault("level_2_completed", False)
    #st.session_state.progress.setdefault("level_3_completed", False)
    #st.session_state.progress.setdefault("level_4_completed", False)

    # ─── Create two tabs ──────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "1️⃣ Bonus Input",
        "2️⃣ Generate Runbook",
        "Debug"
    ])

    # ─── Tab 1: Collect Bonus Inputs ─────────────────────────
    with tab1:
        info = st.session_state.bonus_info

        with st.expander("🏠 Home Maintenance", expanded=True):
            info['maintenance_tasks'] = st.text_area(
                "List regular home maintenance tasks (e.g., changing bulbs, checking smoke detectors, cleaning filters):",
                value=info.get('maintenance_tasks', '')
            )
            info['appliance_instructions'] = st.text_area(
                "Instructions for operating/maintaining major appliances and systems:",
                value=info.get('appliance_instructions', '')
            )

        with st.expander("📋 Home Rules & Preferences", expanded=False):
            info['house_rules'] = st.text_area(
                "Guest/house­sitter rules or preferences:",
                value=info.get('house_rules', '')
            )
            info['cultural_practices'] = st.text_area(
                "Cultural/religious practices guests should be aware of:",
                value=info.get('cultural_practices', '')
            )

        with st.expander("🧹 Housekeeping & Cleaning", expanded=False):
            info['housekeeping_instructions'] = st.text_area(
                "Basic housekeeping/cleaning routines and supply locations:",
                value=info.get('housekeeping_instructions', '')
            )
            info['cleaning_preferences'] = st.text_area(
                "Specific cleaning preferences or routines:",
                value=info.get('cleaning_preferences', '')
            )

        with st.expander("🎮 Entertainment & Technology", expanded=False):
            info['entertainment_info'] = st.text_area(
                "How to operate entertainment systems and streaming services:",
                value=info.get('entertainment_info', '')
            )
            info['device_instructions'] = st.text_area(
                "Instructions for using/charging personal devices:",
                value=info.get('device_instructions', '')
            )

        if st.button("💾 Save Bonus Info"):
            st.success("✅ Bonus level information saved!")

    # ─── Tab 2: Generate Runbook ─────────────────────────────
        with tab2:
            section = st.session_state.get("section", "home")
            switch_section("bonus_level")

            st.subheader("Select and Generate Your Runbook")

            # 1) Must have at least one Bonus input
            bonus_info = st.session_state.bonus_info
            if not any(v and str(v).strip() for v in bonus_info.values()):
                st.error("🔒 Please complete at least one Bonus section in Tab 1 before proceeding.")
                return

            # 2) Mission choices
            missions = [
                "Bonus Level Mission",
                "Mail & Trash + Bonus Mission",
                "Full Runbook Mission"
            ]
            choice = st.radio("Which runbook would you like to generate?", options=missions, key="bonus_runbook_choice")

            # 3) Confirmation checkbox
            confirmed = st.checkbox(
                "✅ Confirm AI Prompt",
                value=st.session_state.get("user_confirmation", False),
                key="user_confirmation"
            )
            if not confirmed:
                st.info("Please confirm to preview and generate your runbook.")
                return

            # 4) Now that user is ready, enforce prerequisites
            if choice == missions[0] and not st.session_state.progress["level_2_completed"]:
                st.warning("🔒 Complete Level 2 before generating the Bonus Level runbook.")
                return
            if choice == missions[1] and not st.session_state.progress["level_3_completed"]:
                st.warning("🔒 Complete Level 3 before generating the Mail & Trash + Bonus runbook.")
                return
            if choice == missions[2] and not st.session_state.progress["level_4_completed"]:
                st.warning("🔒 Complete Level 4 before generating the Full runbook.")
                return

            # 5) Build all prompts
            st.session_state.prompt_emergency      = emergency_kit_utilities_runbook_prompt()
            st.session_state.prompt_bonus          = bonus_level_runbook_prompt()
            st.session_state.prompt_mail_trash     = mail_trash_runbook_prompt()
            st.session_state.prompt_home_caretaker = home_caretaker_runbook_prompt()

            # 6) Assemble the exact prompts, labels, and filenames
            if choice == missions[0]:
                prompts     = [st.session_state.prompt_emergency, st.session_state.prompt_bonus]
                labels      = ["🆘 Emergency + Utilities Prompt", "🎁 Bonus Level Prompt"]
                button_text = "Complete Bonus Level Mission"
                doc_heading = "Home Emergency Runbook with Bonus Level"
                doc_file    = "home_runbook_with_bonus.docx"
            elif choice == missions[1]:
                prompts     = [st.session_state.prompt_emergency, st.session_state.prompt_mail_trash, st.session_state.prompt_bonus]
                labels      = ["🆘 Emergency + Utilities Prompt", "📫 Mail & Trash Prompt", "🎁 Bonus Level Prompt"]
                button_text = "Complete Mail & Trash + Bonus Mission"
                doc_heading = "Emergency + Mail & Trash Runbook with Bonus"
                doc_file    = "runbook_mail_trash_bonus.docx"
            else:
                prompts     = [
                    st.session_state.prompt_emergency,
                    st.session_state.prompt_mail_trash,
                    st.session_state.prompt_home_caretaker,
                    st.session_state.prompt_bonus
                ]
                labels      = [
                    "🆘 Emergency + Utilities Prompt",
                    "📫 Mail & Trash Prompt",
                    "💝 Home Services Prompt",
                    "🎁 Bonus Level Prompt"
                ]
                button_text = "Complete Full Mission"
                doc_heading = "Complete Emergency Runbook with Bonus and Services"
                doc_file    = "runbook_full_mission.docx"

            # 7) Preview selected prompts
            for lbl, p in zip(labels, prompts):
                st.markdown(f"#### {lbl}")
                st.code(p, language="markdown")

            # 8) Generate runbook button
            #generate_runbook_from_multiple_prompts(
            #    prompts=prompts,
            #    api_key=os.getenv("MISTRAL_TOKEN"),
            #    button_text=button_text,
            #    doc_heading=doc_heading,
            #    doc_filename=doc_file
            #)
    # ─── Tab 1: Collect Bonus Inputs ─────────────────────────
        with tab3:
            def show_all_section_names():
                input_data = st.session_state.get("input_data", {})
                section_names = list(input_data.keys())

                if section_names:
                    st.markdown("### 📂 Sections Used in App")
                    for name in section_names:
                        st.markdown(f"- **{name}**")
                else:
                    st.info("ℹ️ No sections recorded yet.")
            
            with st.write("🧭 Debug Info"):
                show_all_section_names()

### Call App Functions
if __name__ == "__main__":
    main()
