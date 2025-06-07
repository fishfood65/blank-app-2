import pandas as pd
import streamlit as st
import json
from typing import List, Optional
from .data_helpers import sanitize, get_answer, sanitize_label
from config.sections import SECTION_METADATA
from .runbook_generator_helpers import runbook_preview_dispatcher
from .common_helpers import get_schedule_placeholder_mapping
from utils.task_schedule_utils_updated import extract_and_schedule_all_tasks, extract_unscheduled_tasks_from_inputs_with_category, display_enriched_task_preview, save_task_schedules_by_type, load_label_map, normalize_label

DEFAULT_COMMON_SECTIONS = set(SECTION_METADATA.keys())

def debug_all_sections_input_capture_with_summary(sections: List[str]):
    """
    Main debug utility to show a summary of all captured inputs and lookup results for multiple sections.
    """
    st.header("🧪 All Sections Debug Panel")

    for section in sections:
        if not isinstance(section, str):
            st.error(f"❌ Invalid section key: expected string, got {type(section)} - {section}")
            continue
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
    tabs = st.tabs(["🧾 Input Records", "📬 get_answer() Results", "📖 Runbook Preview", "🧠 Session State", "🧪 Enriched Diff","📆 Scheduled Tasks"])

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

    with tabs[4]:  # ✅ Enrichment Diff tab
        render_enrichment_debug_view(section)
    
    with tabs[5]:  # 📆 Scheduled Tasks tab
        combined_df = st.session_state.get("combined_home_schedule_df")
        render_schedule_debug_info(combined_df, section="combined_home_schedule_df")

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

def render_enrichment_debug_view(section: str):
    """
    Renders debug info comparing raw vs enriched task scheduling.
    Includes label normalization and task diffs.
    """
    st.markdown("## 🔍 Task Preview: Raw vs Enriched")

    raw_df = extract_unscheduled_tasks_from_inputs_with_category()
    combined_df = st.session_state.get("combined_home_schedule_df")
    label_map = load_label_map()

    if not raw_df.empty:
        st.markdown("### 🔍 Label → Clean Task Mappings")
        for label in raw_df["question"].dropna().unique():
            norm_label = normalize_label(label)
            cleaned = label_map.get(norm_label, "⚠️ No match in LABEL_MAP")
            st.text(f"Label: '{label}' → Normalized: '{norm_label}' → Cleaned: '{cleaned}'")
    else:
        st.warning("⚠️ No raw tasks available to preview.")

    if not raw_df.empty:
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

def render_schedule_debug_info(
    df: Optional[pd.DataFrame] = None,
    section: str = "combined_home_schedule_df"
):
    """
    Render debugging details for a scheduled task DataFrame.
    Supports optional explicit input or fallback to st.session_state[section].

    Includes:
    - Duplicate detection
    - Full raw table
    - Placeholder mapping
    - Grouped summary by SourceKey and metrics

    Usage:
    - Inside debug tabs:
    -- render_schedule_debug_info(section="combined_home_schedule_df")
    - For test injection:
    -- render_schedule_debug_info(df=some_filtered_df)
    """
    if df is None:
        df = st.session_state.get(section)

    if df is None or df.empty:
        st.warning(f"⚠️ No data found in `{section}`.")
        return

    # 🔁 Show duplicate tasks
    dupes = df[df.duplicated(subset=["Date", "Day", "clean_task", "task_type"], keep=False)]
    if not dupes.empty:
        st.markdown("### 🔁 Possible Duplicate Tasks")
        st.dataframe(dupes)

    # 🧠 Raw schedule output
    st.markdown(f"### 🧠 Raw `{section}` Schedule")
    st.dataframe(df)

    # 🧩 Placeholder → DataFrame mapping
    st.subheader("🧩 Placeholder → Schedule Mapping")
    mapping = get_schedule_placeholder_mapping()
    st.json(mapping)

    # 📆 Scheduled Tasks Output
    st.subheader("📆 Scheduled Tasks")
    st.write(df)

    # 📊 Grouped summary
    if "SourceKey" in df.columns:
        st.markdown("### 📊 Task Summary by `SourceKey`")
        summary = (
            df.groupby("SourceKey")
              .agg(
                  task_count=("clean_task", "count"),
                  unique_dates=("Date", pd.Series.nunique),
                  unique_types=("task_type", pd.Series.nunique)
              )
              .reset_index()
              .sort_values(by="task_count", ascending=False)
        )
        st.dataframe(summary)



