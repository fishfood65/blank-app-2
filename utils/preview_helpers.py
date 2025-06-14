# UI-related helpers
import streamlit as st
import re
import pandas as pd
from config.sections import SECTION_METADATA 
from config.section_router import get_page_map, get_question_page_map, QUESTION_PAGE_MAP, PAGE_MAP
import uuid
import hashlib
import json

def get_active_section_label(section_key: str) -> str:
    """
    Returns the human-friendly label for a given section key.
    Falls back to the section key itself if label is missing.
    """
    return SECTION_METADATA.get(section_key, {}).get("label", section_key)


def display_enriched_task_preview(combined_df: pd.DataFrame, section: str = "utilities"):
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


def edit_button_redirect(row, _rendered_keys, i=None, page_map=None, fallback_section: str = "utilities"):
    """
    Displays a row-level edit button. Sets session_state['current_page'] on click.

    Args:
        row: Single task row (Series or dict)
        _rendered_keys: Set tracking previously rendered buttons
        i: Optional unique row index
        page_map: Section/task_type → page map
        fallback_section: Default section if missing
    """
    if not isinstance(row, dict):
        row = row.to_dict()

    label = str(row.get("question", "")).strip()
    clean_task = str(row.get("clean_task", "")).strip()
    
    # 🚫 Skip group/heading rows
    if not label or not clean_task:
        return

    section = row.get("section") or fallback_section
    task_type = normalize_for_map(row.get("task_type", "generic"))
    question = normalize_for_map(str(row.get("question", "")))

    # 🔐 Create a consistent key from row contents
    row_fingerprint = json.dumps(row, sort_keys=True, default=str)
    hash_part = hashlib.sha1(row_fingerprint.encode()).hexdigest()[:12]
    key_suffix = f"{hash_part}_{i or 'na'}"

    if key_suffix in _rendered_keys:
        return  # 🚫 Prevent duplicate key

    _rendered_keys.add(key_suffix)
    button_key = f"edit_{key_suffix}"

    # 🔁 Load routing maps
    page_map = page_map or get_page_map()
    question_page_map = get_question_page_map()

    # 🧭 Routing resolution
    dest_page = (
        QUESTION_PAGE_MAP.get((task_type, question)) or
        PAGE_MAP.get(section) or
        PAGE_MAP.get(task_type) or
        PAGE_MAP.get(fallback_section) or
        "01_Utilities.py"
    )

    if st.button(f"✏️ Edit '{label}'", key=button_key):
        st.session_state["edit_mode"] = {
            "row_index": i,
            "section": section,
            "task_type": task_type,
        }
        st.switch_page(f"/{dest_page}")


def list_saved_llm_outputs():
    """Display stored LLM-generated responses and metadata keys."""
    keys = [k for k in st.session_state if k.startswith("llm_response_") or k.startswith("llm_metadata_")]
    if not keys:
        st.info("No LLM outputs have been stored yet.")
        return

    st.markdown("### 💾 Stored LLM Outputs")
    for key in sorted(keys):
        st.markdown(f"- `{key}`")

def normalize_question(text):
    """Simplify question string to a mapping-friendly key."""
    return text.lower().replace(" ", "_").strip()

def normalize_for_map(text):
    """Normalize text for consistent dictionary keys."""
    text = text.lower().strip()
    text = re.sub(r'\(.*?\)', '', text)  # remove parentheses and contents
    text = re.sub(r'[^a-z0-9_]+', '_', text)  # replace non-alphanumeric with underscore
    return text.strip('_')  # remove leading/trailing underscores

def render_provider_contacts(section: str = "utilities"):
    """
    Renders a visual contact card layout for each utility provider in session state.
    """

    providers = st.session_state.get("utility_providers", {})
    st.markdown("### 🐞 Debug: raw provider data")
    st.json(providers)

    if not providers:
        st.info("No utility provider metadata found.")
        return

    st.markdown("## 🔌 Utility Provider Contact Info")

    icons = {
        "electricity": "⚡",
        "natural_gas": "🔥",
        "water": "💧",
        "internet": "🌐"
    }

    for utility_key in ["electricity", "natural_gas", "water", "internet"]:
        info = providers.get(utility_key, {})
        name = info.get("name", "").strip()
        if not name:
            st.warning(f"{icon} {label} provider name not found. This will be refreshed silently.")
            continue

        label = utility_key.replace("_", " ").title()
        icon = icons.get(utility_key, "🔌")

        st.markdown(f"### {icon} {label}: {name}")

        with st.expander(f"📇 View {name} Contact Info", expanded=False):
            st.markdown(f"**📄 Description:** {info.get('description', '—')}")
            st.markdown(f"**📞 Phone:** {info.get('contact_phone', '—')}")
            #st.markdown(f"**📧 Email:** {info.get('contact_email', '—')}")
            st.markdown(f"**🏢 Address:** {info.get('contact_address', '—')}")
            st.markdown(f"**🌐 Website:** {info.get('contact_website', '—')}")
            st.markdown(f"**🚨 Emergency Steps:**  \n{info.get('emergency_steps', '—')}")

            non_emergency = info.get("non_emergency_tips", "").strip()
            if non_emergency and non_emergency != "⚠️ Not Available":
                st.markdown(f"**💡 Non-Emergency Tips:**  \n{non_emergency}")

def debug_render_provider_contacts(section: str = "utilities"):
    """
    Renders a visual contact card layout for each utility provider in session state.
    """

    providers = st.session_state.get("utility_providers", {})
    if not providers:
        st.info("No utility provider metadata found.")
        return

    st.markdown("## 🔌 Utility Provider Contact Info")

    for utility_key, info in providers.items():
        name = info.get("name", "").strip()
        if not name:
            continue

        st.markdown(f"### 🛠️ {utility_key.replace('_', ' ').title()}: {name}")

        with st.subheader(f"📇 View {name} Contact Info"):
            st.markdown(f"**📄 Description:** {info.get('description', '—')}")
            st.markdown(f"**📞 Phone:** {info.get('contact_phone', '—')}")
            #st.markdown(f"**📧 Email:** {info.get('contact_email', '—')}")
            st.markdown(f"**🏢 Address:** {info.get('contact_address', '—')}")
            st.markdown(f"**🌐 Website:** {info.get('contact_website', '—')}")
            st.markdown(f"**🚨 Emergency Steps:** {info.get('emergency_steps', '—')}")

            non_emergency = info.get("non_emergency_tips", "").strip()
            if non_emergency and non_emergency != "⚠️ Not Available":
                st.markdown(f"**💡 Non-Emergency Tips:**  \n{non_emergency}")

def render_saved_section(label, md_key, docx_key, file_prefix):
    markdown = st.session_state.get(md_key)
    docx_bytes = st.session_state.get(docx_key)

    # Determine status icon
    if markdown or docx_bytes:
        icon = "🟢"
        status = "Available"
    else:
        icon = "🔴"
        status = "Not Available"

    st.markdown(f"### {icon} {label} ({status})")

    # Render markdown if present
    if markdown:
        st.markdown(markdown, unsafe_allow_html=True)
    else:
        st.info("⚠️ Markdown not available for this section.")

    # Offer DOCX download if present
    if docx_bytes:
        st.download_button(
            label=f"📄 Download {file_prefix}.docx",
            data=docx_bytes,
            file_name=f"{file_prefix}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
