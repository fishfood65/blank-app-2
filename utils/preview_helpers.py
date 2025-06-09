# UI-related helpers
import streamlit as st
import re
import pandas as pd
from config.sections import SECTION_METADATA
from config.section_router import get_page_map
import uuid
import hashlib
import json

def get_active_section_label(section_key: str) -> str:
    """
    Returns the human-friendly label for a given section key.
    Falls back to the section key itself if label is missing.
    """
    return SECTION_METADATA.get(section_key, {}).get("label", section_key)


def display_enriched_task_preview(combined_df: pd.DataFrame, section: str = "home"):
    """
    Show enriched & scheduled tasks grouped by (question, answer, clean_task),
    with inline edit buttons for each.
    """
    #st.markdown(f"## 📋 Enriched & Scheduled Task Preview ({get_active_section_label(section)})")

    if combined_df.empty:
        st.info("⚠️ No enriched tasks to preview.")
        return

    # 🧼 Clean and filter valid tasks
    df = combined_df.copy()
    df["question"] = df["question"].fillna("").astype(str).str.strip()
    df["clean_task"] = df["clean_task"].fillna("").astype(str).str.strip()
    df = df[(df["question"] != "") & (df["clean_task"] != "")]
    if df.empty:
        st.info("⚠️ No valid enriched tasks to preview.")
        return

    _rendered_keys = set()
    page_map = get_page_map()

    # ✅ Group by question/answer/clean_task for clarity
    grouped = df.groupby(["question", "answer", "clean_task"], dropna=False)
    for i, ((question, answer, clean_task), group_df) in enumerate(grouped):
        with st.expander(f"📝 {clean_task}", expanded=False):
            st.markdown(f"**Question**: {question}")
            st.markdown(f"**Answer**: {answer}")
            st.markdown(f"**Task Summary**: {clean_task}")
            st.markdown("**Scheduled Occurrences:**")
            st.dataframe(
                group_df[["Date", "Day", "task_type"]].sort_values("Date").reset_index(drop=True)
            )
            edit_button_redirect(row=group_df.iloc[0], i=i, page_map=page_map, _rendered_keys=_rendered_keys)


def sanitize_key(text: str) -> str:
    """Removes emojis/special chars and spaces for use as a Streamlit key."""
    return re.sub(r'[^a-zA-Z0-9_]', '', text.replace(" ", "_"))


def edit_button_redirect(row, _rendered_keys, i=None, page_map=None, fallback_section: str = "home"):
    """
    Displays a row-level edit button. Sets session_state['current_page'] on click.

    Args:
        row: Single task row (Series or dict)
        _rendered_keys: Set tracking previously rendered buttons
        i: Optional unique row index
        page_map: Section/task_type → page map
        fallback_section: Default section if missing
    """
    if page_map is None:
        from section_router import get_page_map
        page_map = get_page_map()

    if not isinstance(row, dict):
        row = row.to_dict()

    label = str(row.get("question", "")).strip()
    clean_task = str(row.get("clean_task", "")).strip()

    # 🚫 Skip if this is a group/header row or has suspicious label content
    if (
        not label
        or not clean_task
        or "—" in label  # e.g., "mail_trash — Monday Jun 09"
        or re.match(r"^\s*[\w\-]+ — \w+", label)  # matches similar group headings
    ):
        return


    section = row.get("section") or fallback_section
    task_type = row.get("task_type", "generic")

    # 🔐 Create a consistent key from row contents
    row_fingerprint = json.dumps(row, sort_keys=True, default=str)
    hash_part = hashlib.sha1(row_fingerprint.encode()).hexdigest()[:12]
    key_suffix = f"{hash_part}_{i or 'na'}"

    if key_suffix in _rendered_keys:
        return  # 🚫 Prevent duplicate key

    _rendered_keys.add(key_suffix)
    button_key = f"edit_{key_suffix}"

    # 🧼 Optional: Sanitize label for readability
    short_label = label if len(label) <= 80 else label[:77] + "..."

    if st.button(f"✏️ Edit '{short_label}'", key=button_key):
        st.session_state["edit_mode"] = {
            "row_index": i,
            "section": section,
            "task_type": task_type,
        }

        # 🔁 Redirect to mapped page
        dest_page = (
            page_map.get(section)
            or page_map.get(task_type)
            or page_map.get(fallback_section)
            or "01_Home.py"
        )
        st.switch_page(dest_page)


def list_saved_llm_outputs():
    """Display stored LLM-generated responses and metadata keys."""
    keys = [k for k in st.session_state if k.startswith("llm_response_") or k.startswith("llm_metadata_")]
    if not keys:
        st.info("No LLM outputs have been stored yet.")
        return

    st.markdown("### 💾 Stored LLM Outputs")
    for key in sorted(keys):
        st.markdown(f"- `{key}`")
