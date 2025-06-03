import streamlit as st
import re
import pandas as pd

def display_enriched_task_preview(combined_df: pd.DataFrame):
    """
    Show grouped enriched & scheduled task preview with edit buttons.
    Groups by (question, answer, clean_task) and lists associated scheduled dates.
    """
    st.markdown("## 📋 Enriched & Scheduled Task Preview")

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

            # Row-level edit (just use first row as representative)
            from utils.preview_helpers import edit_button_redirect  # if needed

            edit_button_redirect(group.iloc[0])  # safely pulls a representative row

def sanitize_key(text: str) -> str:
    """Removes emojis/special chars and spaces for use as a Streamlit key."""
    return re.sub(r'[^a-zA-Z0-9_]', '', text.replace(" ", "_"))

def edit_button_redirect(row, key_prefix="edit", i=None, page_map=None):
    """
    Displays a row-level edit button. Sets session_state['current_page'] on click.

    Args:
        row (dict or pd.Series): A row from enriched_df.
        key_prefix (str): Streamlit key prefix to avoid duplicate keys.
        i (int): Optional unique row index fallback.
        page_map (dict): Optional mapping from section/task_type to filename.
    """
    label = str(row.get("question", "Edit"))
    task_type = str(row.get("task_type", "")).lower()
    section = str(row.get("section", "")).lower()

    destination = section or task_type or "home"

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