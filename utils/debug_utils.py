import streamlit as st
import json
from typing import List
from .data_helpers import sanitize, get_answer, sanitize_label
from config.sections import SECTION_METADATA
from .runbook_generator_helpers import runbook_preview_dispatcher

DEFAULT_COMMON_SECTIONS = set(SECTION_METADATA.keys())

def debug_all_sections_input_capture_with_summary(sections: List[str]):
    """
    Main debug utility to show a summary of all captured inputs and lookup results for multiple sections.
    """
    st.header("🧪 All Sections Debug Panel")

    for section in sections:
        with st.expander(f"🧠 Debug Section: `{section}`", expanded=False):
            debug_task_input_capture_with_answers_tabs(section)

    if st.button("📤 Export Full Debug Data"):
        export_data = {
            "task_inputs": st.session_state.get("task_inputs", []),
            "input_data": st.session_state.get("input_data", {}),
            "session_state": dict(st.session_state),
        }
        st.download_button(
            "⬇️ Download JSON",
            data=json.dumps(export_data, indent=2),
            file_name="debug_snapshot.json",
            mime="application/json"
        )

def debug_task_input_capture_with_answers_tabs(section: str):
    """
    Renders a tabbed interface to debug inputs, lookups, and session state for a given section.
    """
    entries = st.session_state.get("task_inputs", [])
    input_data = st.session_state.get("input_data", {}).get(section, [])
    st.markdown(f"## 🔍 Debug for Section: `{section}`")
    tabs = st.tabs(["🧾 Input Records", "📬 get_answer() Results", "📖 Runbook Preview", "🧠 Session State"])

    with tabs[0]:
        st.subheader("📌 task_inputs")
        st.dataframe([e for e in entries if e.get("section") == section])

        st.subheader("📎 input_data")
        st.dataframe(input_data)

    with tabs[1]:
        st.subheader("🔍 get_answer() Lookup Results")
        for record in input_data:
            raw_label = record.get("question", "")
            raw_key = record.get("key", "")
            val = get_answer(key=raw_label, section=section, verbose=True)
            sanitized_label = sanitize_label(raw_label)
            sanitized_key = sanitize_label(raw_key)
            st.markdown(f"""
            - **Label**: `{raw_label}`
            - **Key**: `{raw_key}`
            - **Sanitized Label**: `{sanitized_label}`
            - **Sanitized Key**: `{sanitized_key}`
            - **get_answer() Result**: `{val}`
            """)

    with tabs[2]:
        runbook_text = st.session_state.get(f"{section}_runbook_text", "")
        runbook_preview_dispatcher(
            section=section,
            runbook_text=runbook_text,
            mode="debug",
            show_schedule_snapshot=True
        )

    with tabs[3]:
        st.subheader("🧠 Raw `st.session_state`")
        st.json(dict(st.session_state))

def debug_single_get_answer(section: str, key: str):
    st.markdown(f"### 🧪 Debug `get_answer(section='{section}', key='{key}')`")

    task_inputs = st.session_state.get("task_inputs", [])
    st.markdown(f"**🔍 Total `task_inputs`:** {len(task_inputs)}")

    if not task_inputs:
        st.error("❌ No task inputs found.")
        return

    sanitized_key = sanitize_label(key)
    st.markdown(f"**🔑 Sanitized key:** `{sanitized_key}`")

    # Step 1: Filter by section
    matching_section = [entry for entry in task_inputs if entry.get("section") == section]
    if not matching_section:
        st.warning(f"⚠️ No entries found for section `{section}`.")
        return

    st.markdown(f"**📂 Entries in section `{section}`:**")
    st.json(matching_section)

    # Step 2: Try to find match by sanitized question or key
    for entry in matching_section:
        question_raw = entry.get("question", "")
        key_raw= entry.get("key", "")
        question_sanitized = sanitize_label(question_raw)
        key_sanitized = sanitize_label(key_raw)

        if question_sanitized == sanitized_key or key_sanitized == sanitized_key:
            st.success("✅ Exact match found:")
            st.write(f"🧾 Label: {question_raw}")
            st.write(f"🗝️ Key: {key_raw}")
            st.write(f"📦 Value returned: `{entry.get('answer')}`")
            return

    # Step 3: Try partial match
    partial_matches = [
        entry for entry in matching_section 
        if sanitized_key in sanitize_label(entry.get("question", ""))
        or sanitized_key in sanitize_label(entry.get("key", ""))
    ]
    if partial_matches:
        st.warning("⚠️ No exact match, but found partial label matches:")
        st.json(partial_matches)
        return

    st.error(f"❌ No label match for key `{sanitized_key}` in section `{section}`.")

def clear_all_session_data():
    """
    Clears all Streamlit session state data, including input_data, task_inputs,
    and any dynamically stored keys. Use this to reset the app state completely.
    """
    known_keys_to_clear = [
        "input_data",
        "task_inputs",
        "combined_home_schedule_df",
        "mail_schedule_df",
        "trash_schedule_df",
        "homeowner_kit_stock",
        "not_selected_items",
        "utility_providers",
        "generated_prompt",
        "runbook_text",
        "runbook_buffer",
        "user_confirmation",
        "runbook_date_range",
        # Add any additional known keys here
    ]

    # Clear all known keys explicitly
    for key in known_keys_to_clear:
        st.session_state.pop(key, None)

    # Clear any remaining keys dynamically
    for key in list(st.session_state.keys()):
        st.session_state.pop(key, None)

    st.success("🧹 All session state has been cleared.")

