# UI-related helpers
import streamlit as st
import re
import pandas as pd
from config.sections import SECTION_METADATA

def get_active_section_label(section_key: str) -> str:
    """
    Returns the human-friendly label for a given section key.
    Falls back to the section key itself if label is missing.
    """
    return SECTION_METADATA.get(section_key, {}).get("label", section_key)


def display_enriched_task_preview(combined_df: pd.DataFrame, section: str = "home"):
    """
    Show grouped enriched & scheduled task preview with edit buttons.
    Groups by (question, answer, clean_task) and lists associated scheduled dates.

    Args:
        combined_df: The enriched task dataframe
        section: Used for fallback if row['section'] is missing
    """
    st.markdown(f"## 📋 Enriched & Scheduled Task Preview ({get_active_section_label(section)})")

    if combined_df.empty:
        st.info("⚠️ No enriched tasks to preview.")
        return

    grouped = combined_df.groupby(["question", "answer", "clean_task"], dropna=False)

    for (question, answer, clean_task), group in grouped:
        with st.expander(f"📝 {clean_task}", expanded=False):
            st.markdown(f"**Question**: {question}")
            st.markdown(f"**Answer**: {answer}")
            st.markdown(f"**Task Summary**: {clean_task}")

            st.markdown("**Scheduled Occurrences:**")
            st.dataframe(
                group[["Date", "Day", "task_type"]].sort_values("Date").reset_index(drop=True)
            )

            # Optional: Show metadata for debugging
            # st.write(group[["section", "task_type", "inferred_days", "formatted_answer"]])

            edit_button_redirect(group.iloc[0], fallback_section=section)


def sanitize_key(text: str) -> str:
    """Removes emojis/special chars and spaces for use as a Streamlit key."""
    return re.sub(r'[^a-zA-Z0-9_]', '', text.replace(" ", "_"))


def edit_button_redirect(row, key_prefix="edit", i=None, page_map=None, fallback_section: str = "home"):
    """
    Displays a row-level edit button. Sets session_state['current_page'] on click.

    Args:
        row (dict or pd.Series): A row from enriched_df.
        key_prefix (str): Streamlit key prefix to avoid duplicate keys.
        i (int): Optional unique row index fallback.
        page_map (dict): Optional mapping from section/task_type to filename.
        fallback_section (str): Used if section/task_type cannot be resolved from row.
    """
    label = str(row.get("question", "Edit"))
    task_type = str(row.get("task_type", "")).lower()
    section = str(row.get("section", "")).lower() or fallback_section

    destination = section or task_type or fallback_section
    if page_map and destination in page_map:
        destination = page_map[destination]

    # Ensure unique and clean Streamlit key
    safe_label = sanitize_key(label)
    safe_date = str(row.get("Date", ""))
    suffix = f"_{i}" if i is not None else ""
    button_key = f"{key_prefix}_{safe_label}_{safe_date}{suffix}"

    if st.button(f"✏️ Edit '{label}'", key=button_key):
        st.session_state["current_page"] = destination
        st.experimental_rerun()


def list_saved_llm_outputs():
    """Display stored LLM-generated responses and metadata keys."""
    keys = [k for k in st.session_state if k.startswith("llm_response_") or k.startswith("llm_metadata_")]
    if not keys:
        st.info("No LLM outputs have been stored yet.")
        return

    st.markdown("### 💾 Stored LLM Outputs")
    for key in sorted(keys):
        st.markdown(f"- `{key}`")

