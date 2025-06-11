import pandas as pd
import streamlit as st
import json
from typing import List, Optional
from collections import Counter
from .data_helpers import sanitize, get_answer, sanitize_label
from config.sections import SECTION_METADATA
from .runbook_generator_helpers import runbook_preview_dispatcher
from .common_helpers import get_schedule_placeholder_mapping, get_schedule_utils
from utils.task_schedule_utils_updated import extract_and_schedule_all_tasks, extract_unscheduled_tasks_from_inputs_with_category, save_task_schedules_by_type, load_label_map, normalize_label, infer_relevant_days_from_text, format_answer_as_bullets, humanize_task
from utils.debug_scheduler_input import debug_schedule_task_input
from utils.preview_helpers import display_enriched_task_preview

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
            data=json.dumps(export_data, indent=2, default=str),
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
    tabs = st.tabs([
        "🧾 Input Records", 
        "📬 get_answer() Results", 
        "📖 Runbook Preview", 
        "🧠 Session State", 
        "🧪 Enriched Diff",
        "📆 Scheduled Tasks", 
        "🧱 Raw Extracted Tasks", 
        "📦 Raw _schedule_df Snapshots",
        "🔍 Merge Debug Tools",
        "🔗 Task vs Schedule Trace"
        ])

    with tabs[0]:
        st.subheader("📌 task_inputs")
        st.dataframe([e for e in entries if e.get("section") == section])

        st.subheader("📎 input_data")
        st.dataframe(input_data)
        
        from utils.debug_utils import log_section_input_debug
        counts = log_section_input_debug(section, min_entries=8)
        st.write("🔍 Returned counts object:", counts)

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

        debug_blocks = st.session_state.get(f"{section}_debug_blocks", [])
        debug_md_output = st.session_state.get(f"{section}_debug_markdown_output", [])

        if debug_blocks or debug_md_output:
            st.markdown("---")
            st.markdown("### 🧪 Prompt Debug")

            with st.markdown("🔹 Raw Prompt Blocks"):
                for i, block in enumerate(debug_blocks):
                    st.markdown(f"**Block {i+1}**")
                    blocks = st.session_state.get(f"{section}_runbook_blocks", [])
                    if not blocks:
                        st.warning("⚠️ No prompt blocks found in session_state.")
                    else:
                        for i, block in enumerate(blocks):
                            st.markdown(f"**Block {i+1}**")
                            st.code(block, language="markdown")


            with st.markdown("🔹 Markdown Output"):
                for i, md in enumerate(debug_md_output):
                    st.markdown(f"**Markdown Block {i+1}**")
                    st.code(md, language="markdown")
        st.subheader("📖 Prompt Blocks")
        blocks = st.session_state.get(f"{section}_debug_blocks", [])
        if blocks:
            for i, block in enumerate(blocks):
                st.code(block, language="markdown")
        else:
            st.warning("⚠️ No prompt blocks found in session_state.")

        st.subheader("🧾 Markdown Output")
        markdown_output = st.session_state.get(f"{section}_debug_markdown_output", [])
        if markdown_output:
            for i, md in enumerate(markdown_output):
                st.markdown(md)
        else:
            st.warning("⚠️ No markdown output found in session_state.")

    with tabs[3]:
        st.subheader("🧠 Raw `st.session_state`")
        st.json(dict(st.session_state))

    with tabs[4]:  # ✅ Enrichment Diff tab
        render_enrichment_debug_view(section)
    
    with tabs[5]:  # 📆 Scheduled Tasks tab
        st.subheader("📆 Combined Schedule Debug")

        extracted_df = extract_unscheduled_tasks_from_inputs_with_category()
        enriched_rows = []
        skipped_rows = []

        valid_dates = st.session_state.get("valid_dates", [])
        utils = get_schedule_utils()
        label_map = load_label_map()

        # Enrich & track skipped
        for row in extracted_df.to_dict("records"):
            answer = row.get("answer", "") or ""
            row["inferred_days"] = infer_relevant_days_from_text(answer)
            row["formatted_answer"] = format_answer_as_bullets(answer)
            row["clean_task"] = humanize_task(row, include_days=False, label_map=label_map)

            if not row["clean_task"] or not row["inferred_days"]:
                reason = []
                if not row["clean_task"]:
                    reason.append("❌ Missing clean_task")
                if not row["inferred_days"]:
                    reason.append("❌ No inferred_days")
                row["skip_reason"] = ", ".join(reason)
                skipped_rows.append(row)
            else:
                enriched_rows.append(row)

        # Show scheduling input debug
        debug_schedule_task_input(enriched_rows, valid_dates)

        # Show skipped tasks
        if skipped_rows:
            st.markdown("### ⚠️ Skipped Tasks")
            st.warning(f"{len(skipped_rows)} tasks were skipped and not scheduled:")
            skipped_df = pd.DataFrame(skipped_rows)
            st.dataframe(skipped_df[["question", "answer", "clean_task", "inferred_days", "skip_reason"]])
        else:
            st.success("✅ No skipped tasks. All enrichments passed!")

        # Show final scheduled output
        combined_df = st.session_state.get("combined_home_schedule_df")
        render_schedule_debug_info(combined_df, section="combined_home_schedule_df")

    with tabs[6]:  # 🧱 Raw Extracted Tasks
        st.subheader("🧱 Raw Extracted Tasks (from task_inputs)")
        df = extract_unscheduled_tasks_from_inputs_with_category()
        debug_summary = log_extracted_tasks_debug(df, section=section)

        st.markdown("### 📊 Summary Stats Returned")
        st.json(debug_summary)

        if not df.empty:
            st.download_button(
                "📥 Download Extracted Tasks as CSV",
                data=df.to_csv(index=False),
                file_name=f"{section}_extracted_tasks.csv",
                mime="text/csv"
            )
    with tabs[7]:  # 📦 Raw _schedule_df Snapshots
        st.subheader("📦 Raw *_schedule_df DataFrames")

        # 🔍 Show placeholder mappings
        placeholder_map = get_schedule_placeholder_mapping()

        if placeholder_map:
            st.markdown("### 🔗 Placeholder Mapping")
            st.json(placeholder_map)
        else:
            st.warning("⚠️ No placeholder mappings found.")

        # 📋 Show all *_schedule_df DataFrames
        keys = [k for k in st.session_state if k.endswith("_schedule_df")]
        if not keys:
            st.warning("⚠️ No *_schedule_df keys found.")
        else:
            for key in sorted(keys):
                df = st.session_state.get(key)
                if isinstance(df, pd.DataFrame):
                    # Find matching placeholder, if any
                    matching_placeholder = next((p for p, v in placeholder_map.items() if v == key), None)
                    if matching_placeholder:
                        st.markdown(f"#### 🔹 `{key}` → {len(df)} rows  \n📌 Placeholder: `{matching_placeholder}`")
                    else:
                        st.markdown(f"#### 🔹 `{key}` → {len(df)} rows")

                    st.dataframe(df)
                else:
                    st.error(f"❌ `{key}` is not a DataFrame")

    
    with tabs[8]:  # 🔍 Merge Debug Tools
        st.subheader("🔍 Merge Context Debug")

        from utils.debug_utils import debug_schedule_df_presence, debug_session_keys

        st.markdown("#### ✅ Detected *_schedule_df presence:")
        debug_schedule_df_presence()

        st.markdown("#### 🧠 Full Session Keys + Types")
        debug_session_keys()
    
    with tabs[9]:  # 🔗 Task vs Schedule Trace
        st.subheader("🔗 Task Inputs vs Scheduled Tasks")

        task_inputs = st.session_state.get("task_inputs", [])
        section_tasks = [t for t in task_inputs if t.get("section") == section]
        task_df = pd.DataFrame(section_tasks)

        schedule_dfs = {
            k: v for k, v in st.session_state.items()
            if k.endswith("_schedule_df") and isinstance(v, pd.DataFrame)
        }

        # Combine all scheduled rows
        if schedule_dfs:
            scheduled_df = pd.concat(schedule_dfs.values(), ignore_index=True)
        else:
            scheduled_df = pd.DataFrame()

        if not task_df.empty and "question" in task_df.columns and "clean_task" in scheduled_df.columns:
            task_df["match_key"] = task_df["question"].str.lower().str.strip()
            scheduled_df["match_key"] = scheduled_df["clean_task"].str.lower().str.strip()

            merged = task_df.merge(
                scheduled_df[["match_key", "Date", "task_type", "SourceKey"]],
                on="match_key",
                how="left",
                indicator=True
            )

            st.markdown("### 🔍 Trace Comparison")
            st.dataframe(merged)

            missing = merged[merged["_merge"] == "left_only"]
            if not missing.empty:
                st.warning(f"⚠️ {len(missing)} task_inputs were NOT scheduled")
                st.dataframe(missing)
            else:
                st.success("✅ All task_inputs appear in scheduled outputs.")
        else:
            st.warning("⚠️ Missing required columns ('question' or 'clean_task') for comparison.")

            st.markdown("#### 🧾 Available columns in task_df")
            st.write(task_df.columns.tolist())
            
            st.markdown("#### 🧾 Available columns in scheduled_df")
            st.write(scheduled_df.columns.tolist())


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
        key_raw = entry.get("key", "")
        question_sanitized = sanitize_label(question_raw)
        key_sanitized = sanitize_label(key_raw)

        if question_sanitized == sanitized_key or key_sanitized == sanitized_key:
            st.success("✅ Exact match found:")
            st.write(f"🧾 **Label**: `{question_raw}`")
            st.write(f"🗝️ **Key**: `{key_raw}`")
            st.write(f"📦 **Value returned**: `{entry.get('answer')}`")
            if "task_id" in entry:
                st.code(f"🆔 task_id: {entry['task_id']}")
            return

    # Step 3: Try partial match
    partial_matches = [
        entry for entry in matching_section 
        if sanitized_key in sanitize_label(entry.get("question", ""))
        or sanitized_key in sanitize_label(entry.get("key", ""))
    ]
    if partial_matches:
        st.warning("⚠️ No exact match, but found partial label matches:")
        for match in partial_matches:
            st.markdown("---")
            st.write(f"🧾 **Label**: `{match.get('question')}`")
            st.write(f"🗝️ **Key**: `{match.get('key')}`")
            st.write(f"📦 **Answer**: `{match.get('answer')}`")
            if "task_id" in match:
                st.code(f"🆔 task_id: {match['task_id']}")
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

def log_extracted_tasks_debug(df: pd.DataFrame, section: str = None) -> dict:
    """
    Visualize and summarize the output from extract_unscheduled_tasks_from_inputs_with_category().

    Args:
        df (pd.DataFrame): The extracted task dataframe.
        section (str, optional): If provided, filters tasks to just that section.

    Returns:
        dict: Summary statistics.
    """
    if df.empty:
        st.warning("⚠️ No tasks extracted.")
        return {"total_tasks": 0}

    # Optional section filter
    if section:
        df = df[df["section"] == section]
        if df.empty:
            st.warning(f"⚠️ No tasks found for section `{section}`.")
            return {"total_tasks": 0, "filtered": True, "section": section}

    # Optional sort
    if "timestamp" in df.columns:
        df = df.sort_values(by="timestamp", ascending=False)
    elif "section" in df.columns:
        df = df.sort_values(by="section")

    # Display raw task table
    st.markdown(f"### 📋 Extracted Tasks{' for `' + section + '`' if section else ''}")
    st.dataframe(df)

    # Summary stats
    missing_keys = df["key"].apply(lambda k: not str(k).strip()).sum() if "key" in df.columns else None
    missing_task_ids = df["task_id"].apply(lambda x: not str(x).strip()).sum() if "task_id" in df.columns else None

    summary = {
        "total_tasks": len(df),
        "distinct_sections": df["section"].nunique() if "section" in df.columns else None,
        "distinct_task_types": df["task_type"].nunique() if "task_type" in df.columns else None,
        "empty_answers": df["answer"].apply(lambda a: not str(a).strip()).sum() if "answer" in df.columns else None,
        "missing_keys": missing_keys,
        "missing_task_ids": missing_task_ids,
        "section": section,
        "filtered": bool(section),
    }

    st.markdown(f"### 🧮 Task Extraction Summary{' for `' + section + '`' if section else ''}")
    for k, v in summary.items():
        st.markdown(f"- **{k.replace('_', ' ').title()}**: `{v}`")

    # Optional breakdown by type
    if "task_type" in df.columns:
        st.markdown("### 📂 Task Type Breakdown")
        st.dataframe(
            df["task_type"].value_counts()
              .rename("count")
              .reset_index()
              .rename(columns={"index": "task_type"})
        )

    # Optional preview of bad rows
    if all(col in df.columns for col in ["key", "task_id", "question", "answer"]):
        st.markdown("### ⚠️ Problematic Task Rows (Missing Key / Task ID / Question / Answer)")
        problem_rows = df[
            df["key"].apply(lambda k: not str(k).strip()) |
            df["task_id"].apply(lambda t: not str(t).strip()) |
            df["question"].apply(lambda q: not str(q).strip()) |
            df["answer"].apply(lambda a: not str(a).strip())
        ]
        if not problem_rows.empty:
            st.dataframe(problem_rows)
        else:
            st.success("✅ All tasks have key, task_id, question, and answer populated.")

    return summary

def render_enrichment_debug_view(section: str):
    """
    Renders debug info comparing raw vs enriched task scheduling.
    Includes label normalization, label map keys, and color-coded task diffs.
    """
    st.markdown("## 🔍 Task Preview: Raw vs Enriched")

    raw_df = extract_unscheduled_tasks_from_inputs_with_category()
    combined_df = st.session_state.get("combined_home_schedule_df")
    label_map = load_label_map()

    if raw_df.empty:
        st.warning("⚠️ No raw tasks available to preview.")
        return

    st.markdown("### 🔍 Label → Clean Task Mappings")
    for label in raw_df["question"].dropna().unique():
        norm_label = normalize_label(label)
        cleaned = label_map.get(norm_label, "⚠️ No match in LABEL_MAP")
        st.text(f"Label: '{label}' → Normalized: '{norm_label}' → Cleaned: '{cleaned}'")

    # 🔎 Expandable view of all LABEL_MAP keys
    st.subheader("🧩 Label Map Keys")
    st.write(sorted(label_map.keys()))

    st.markdown("### 📝 Raw Task Inputs")
    st.dataframe(raw_df)

    if not isinstance(combined_df, pd.DataFrame) or combined_df.empty:
        st.warning("⚠️ No enriched schedule found.")
        return

    st.markdown("### ✨ Enriched & Scheduled Tasks")
    st.dataframe(combined_df)

    st.markdown("### 🔁 Matched Task Diffs")

    for i, row in raw_df.iterrows():
        raw_id = row.get("task_id")
        raw_q = str(row.get("question", "")).strip()
        raw_a = str(row.get("answer", "")).strip()
        raw_key = row.get("key")

        # 🔑 Match by task_id first
        if raw_id:
            matches = combined_df[combined_df["task_id"] == raw_id]
        elif raw_key:
            matches = combined_df[combined_df["key"] == raw_key]
        else:
            matches = combined_df[
                (combined_df["question"].astype(str).str.strip() == raw_q) &
                (combined_df["answer"].astype(str).str.strip() == raw_a)
            ]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**📝 Raw Task {i+1}:**")
            st.write(f"**Question:** {raw_q}")
            st.write(f"**Answer:** {raw_a}")
            st.write(f"**Key:** {raw_key}")
            st.write(f"**task_id:** `{raw_id}`")

        with col2:
            if not matches.empty:
                enriched_sample = matches.iloc[0]
                clean_task = enriched_sample.get("clean_task", "").strip()
                status_emoji = "✅" if clean_task and clean_task != "⚠️" else "⚠️"

                st.markdown(f"**{status_emoji} Enriched View**")
                st.write(f"**Clean Task:** {clean_task or '⚠️ Missing'}")
                st.write(f"**Formatted Answer:** {enriched_sample.get('formatted_answer', '')}")
                st.write(f"**Inferred Days:** {enriched_sample.get('inferred_days', '')}")
                st.write(f"**Frequency Tag:** {enriched_sample.get('FrequencyTag', '')}")
                st.write(f"**task_id:** `{enriched_sample.get('task_id', '')}`")
            else:
                st.markdown("**⚠️ No enriched match found**")
                st.write(f"Missing match for task_id: `{raw_id}` or fallback key: `{raw_key}`")


def log_section_input_debug(section: str, min_entries: int = 1) -> dict:
    """
    Displays debugging stats for inputs collected from a given section,
    and returns a dictionary of counts for deeper introspection.

    Returns:
        dict: {
            "count_regular": int,
            "count_tasks": int,
            "total": int,
            "meets_threshold": bool
        }
    """
    def is_valid_answer(ans: str) -> bool:
        stripped = str(ans or "").strip().lower()
        return stripped not in ["", "⚠️ not provided", "none", "n/a", "na"]

    input_data = st.session_state.get("input_data", {}).get(section, [])
    count_regular = sum(1 for entry in input_data if is_valid_answer(entry.get("answer")))

    task_inputs = st.session_state.get("task_inputs", [])
    count_tasks = sum(
        1 for entry in task_inputs
        if entry.get("section") == section and is_valid_answer(entry.get("answer"))
    )

    total = count_regular + count_tasks
    meets_threshold = total >= min_entries

    # Debug UI output
    st.markdown(f"""### 🧪 Debug: `{section}` input summary
- Regular inputs (`input_data`): **{count_regular}**
- Task inputs (`task_inputs`): **{count_tasks}**
- ✅ Total counted inputs: **{total}**
- Minimum required: **{min_entries}**
- ✅ Meets threshold: **{meets_threshold}**

> ⚠️ Skips answers like "⚠️ not provided", "none", or "n/a"
""")

    if st.session_state.get("enable_debug_mode"):
        st.markdown("#### Raw Regular Inputs")
        st.json(input_data)

        st.markdown("#### Raw Task Inputs")
        st.json([entry for entry in task_inputs if entry.get("section") == section])

    return {
        "count_regular": count_regular,
        "count_tasks": count_tasks,
        "total": total,
        "meets_threshold": meets_threshold
    }

def log_extracted_tasks_debug(df: pd.DataFrame, section: str = None) -> dict:
    """
    Visualize and summarize the output from extract_unscheduled_tasks_from_inputs_with_category().

    Args:
        df (pd.DataFrame): The extracted task dataframe.
        section (str, optional): If provided, filters tasks to just that section.

    Returns:
        dict: Summary statistics.
    """
    if df.empty:
        st.warning("⚠️ No tasks extracted.")
        return {"total_tasks": 0}

    # Optional section filter
    if section:
        df = df[df["section"] == section]
        if df.empty:
            st.warning(f"⚠️ No tasks found for section `{section}`.")
            return {"total_tasks": 0, "filtered": True, "section": section}

    # Optional sort
    if "timestamp" in df.columns:
        df = df.sort_values(by="timestamp", ascending=False)
    elif "section" in df.columns:
        df = df.sort_values(by="section")

    # Display raw task table
    st.markdown(f"### 📋 Extracted Tasks{' for `' + section + '`' if section else ''}")
    st.dataframe(df)

    # Summary stats
    missing_keys = df["key"].apply(lambda k: not str(k).strip()).sum() if "key" in df.columns else None
    missing_task_ids = df["task_id"].apply(lambda x: not str(x).strip()).sum() if "task_id" in df.columns else None

    summary = {
        "total_tasks": len(df),
        "distinct_sections": df["section"].nunique() if "section" in df.columns else None,
        "distinct_task_types": df["task_type"].nunique() if "task_type" in df.columns else None,
        "empty_answers": df["answer"].apply(lambda a: not str(a).strip()).sum() if "answer" in df.columns else None,
        "missing_keys": missing_keys,
        "missing_task_ids": missing_task_ids,
        "section": section,
        "filtered": bool(section),
    }

    st.markdown(f"### 🧮 Task Extraction Summary{' for `' + section + '`' if section else ''}")
    for k, v in summary.items():
        st.markdown(f"- **{k.replace('_', ' ').title()}**: `{v}`")

    # Optional breakdown by type
    if "task_type" in df.columns:
        st.markdown("### 📂 Task Type Breakdown")

def debug_schedule_df_presence():
    schedule_keys = sorted([k for k in st.session_state if k.endswith("_schedule_df")])
    if not schedule_keys:
        st.warning("❌ No *_schedule_df keys found.")
        return

    for key in schedule_keys:
        df = st.session_state.get(key)
        if isinstance(df, pd.DataFrame):
            if df.empty:
                st.warning(f"⚠️ `{key}` is an empty DataFrame.")
            else:
                st.success(f"✅ `{key}`: {len(df)} rows")
        else:
            st.error(f"🚫 `{key}` is not a DataFrame.")

def debug_session_keys():
    st.markdown("### 🧵 Session Keys Summary")
    for k in sorted(st.session_state.keys()):
        v = st.session_state[k]
        length_str = f"(len={len(v)})" if hasattr(v, "__len__") else ""
        st.write(f"- `{k}` → `{type(v).__name__}` {length_str}")
        if isinstance(v, dict) and len(v) > 0:
            st.json({kk: str(type(v[kk])) for kk in list(v.keys())[:5]})

def log_all_schedule_dfs_from_session(show_dataframes=True):
    st.markdown("### 📦 All *_schedule_df in session_state")
    keys = sorted([k for k in st.session_state if k.endswith("_schedule_df")])
    if not keys:
        st.warning("⚠️ No *_schedule_df found.")
        return []

    total_rows = 0
    for k in keys:
        df = st.session_state.get(k)
        if isinstance(df, pd.DataFrame):
            st.markdown(f"#### 🔹 `{k}` → {len(df)} rows")
            total_rows += len(df)
            if df.empty:
                st.warning(f"⚠️ `{k}` is empty.")
            elif show_dataframes:
                st.dataframe(df)
        else:
            st.error(f"🚫 `{k}` is not a DataFrame.")
    st.success(f"✅ Total scheduled tasks across all: **{total_rows}**")
    return keys

from collections import Counter

def debug_schedule_task_input(tasks: list, valid_dates: list):
    st.markdown("### 🧪 Task Scheduling Input Debug")

    if not tasks:
        st.warning("⚠️ No tasks passed to scheduler.")
        return

    st.markdown(f"- Total tasks passed: `{len(tasks)}`")
    st.markdown(f"- Valid dates: `{[d.isoformat() for d in valid_dates]}`")

    # Show first few tasks with focused display
    st.markdown("#### 📋 Task Preview (first 3)")
    for i, task in enumerate(tasks[:3]):
        st.markdown(f"**Task {i+1}** — ID: `{task.get('task_id', '❌ missing')}`")
        st.write({k: task.get(k) for k in ["question", "answer", "clean_task", "task_type", "task_id"]})

    # Highlight tasks with missing critical fields
    st.markdown("#### ⚠️ Tasks Missing Required Fields")
    required_fields = ["question", "answer", "clean_task", "task_id"]
    bad_tasks = [
        t for t in tasks if not all(str(t.get(f, "")).strip() for f in required_fields)
    ]

    if bad_tasks:
        st.warning(f"⚠️ {len(bad_tasks)} tasks have missing fields: {', '.join(required_fields)}")
        for i, bt in enumerate(bad_tasks):
            st.markdown(f"**❌ Incomplete Task {i+1}**")
            st.write({k: bt.get(k) for k in required_fields})
    else:
        st.success("✅ All tasks have required fields.")

    # Detect duplicate task_ids
    task_id_counts = Counter(t.get("task_id") for t in tasks if t.get("task_id"))
    dupes = [tid for tid, count in task_id_counts.items() if count > 1]

    if dupes:
        st.markdown("### ⚠️ Duplicate `task_id`s Detected")
        for tid in dupes:
            st.markdown(f"**task_id:** `{tid}`")
            for t in [t for t in tasks if t.get("task_id") == tid]:
                st.write({k: t.get(k) for k in ["question", "answer", "clean_task", "task_type", "task_id"]})
    else:
        st.success("✅ All `task_id`s are unique.")

def render_schedule_debug_info(
    df: Optional[pd.DataFrame] = None,
    section: str = "combined_home_schedule_df"
):
    """
    Render debugging details for a scheduled task DataFrame.
    Includes duplicate detection, schedule inspection, placeholder mapping, and group summaries.
    """
    if df is None:
        df = st.session_state.get(section)

    if df is None or df.empty:
        st.warning(f"⚠️ No data found in `{section}`.")
        return

    # 🔁 Duplicate detection
    required_cols = ["Date", "Day", "clean_task", "task_type"]
    if all(col in df.columns for col in required_cols):
        dupes = df[df.duplicated(subset=required_cols, keep=False)]
        if not dupes.empty:
            st.markdown("### 🔁 Possible Duplicate Tasks")
            st.dataframe(dupes)
    else:
        st.info("ℹ️ Duplicate detection skipped due to missing columns.")

    # 📆 Sort for easier reading
    if "Date" in df.columns:
        df = df.sort_values(by="Date")

    st.markdown(f"### 🧠 Raw `{section}` Schedule")
    st.dataframe(df)

    st.subheader("🧩 Placeholder → Schedule Mapping")
    mapping = get_schedule_placeholder_mapping()
    st.json(mapping)

    st.subheader("📆 Scheduled Tasks")
    st.write(df)

    if "SourceKey" in df.columns and df["SourceKey"].notna().any():
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
    else:
        st.info("ℹ️ No valid `SourceKey` values found for summary.")

    if "task_type" in df.columns:
        st.markdown("### 📂 Task Type Breakdown")
        st.dataframe(
            df["task_type"].value_counts()
              .rename("count")
              .reset_index()
              .rename(columns={"index": "task_type"})
        )

def run_batch_meaningful_content_test(path="debug_llm_blocks.json"):
    with open(path, "r", encoding="utf-8") as f:
        blocks = json.load(f)

    results = []
    for i, block in enumerate(blocks, start=1):
        meaningful = is_content_meaningful(block)
        results.append((i, meaningful, block.strip()[:80]))  # show a preview

    print("🧪 Batch Test Results")
    for i, meaningful, preview in results:
        status = "✅ Meaningful" if meaningful else "❌ Trivial"
        print(f"Block {i:02d}: {status} — {preview!r}")

    num_trivial = sum(1 for _, m, _ in results if not m)
    num_total = len(results)
    print(f"\nSummary: {num_total - num_trivial} meaningful / {num_total} total blocks")
