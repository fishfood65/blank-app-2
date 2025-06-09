from .sections import SECTION_METADATA, LLM_SECTIONS
import streamlit as st

def critical_documents_flow(
    emergency_kit_critical_documents,
    review_selected_documents,
    collect_document_details,
    generate_kit_tab
):
    st.subheader("💼 Level 5 Critical Documents")
    tabs = st.tabs([
        "📝 Select Documents",
        "📋 Review Selections",
        "🗂 Document Details",
        "📦 Generate Kit"
    ])
    with tabs[0]:
        st.markdown("### Step 1: Pick Critical Documents")
        emergency_kit_critical_documents()
    with tabs[1]:
        st.markdown("### Step 2: Review Your Picks")
        review_selected_documents()
    with tabs[2]:
        st.markdown("### Step 3: Fill in Document Details")
        collect_document_details()
    with tabs[3]:
        generate_kit_tab()

def get_handler(section_key: str, subsection_key: str | None = None):
    if section_key == "home":
        from handlers.home import home
        return home

    elif section_key == "emergency_kit":
        from handlers.emergency_kit import emergency_kit_utilities
        return emergency_kit_utilities

    elif section_key == "home_security":
        from home_app2 import home_security
        return home_security

    elif section_key == "emergency_kit_critical_documents":
        from home_app2 import (
            emergency_kit_critical_documents,
            review_selected_documents,
            collect_document_details,
            generate_kit_tab,
        )
        return emergency_kit_critical_documents

    elif section_key == "mail_trash":
        from handlers.mail_trash import mail_trash
        return mail_trash  # fallback to combined handler

    elif section_key == "bonus_level":
        from home_app2 import bonus_level
        return bonus_level

    return None

PAGE_MAP = {
    "home": "01_Home",
    "mail_trash": "02_Mail_Trash",
    "emergency_kit": "03_Emergency_Kit",
    "bonus_level": "04_Bonus_Level",
    "critical_documents": "05_Critical_Documents",
    "pets_dog": "06_Dogs",
    "pets_cat": "07_Cats",
    "runbook": "Runbook_Generator",

    # Task type overrides
    "trash": "02_Mail_Trash",
    "mail": "02_Mail_Trash",
}


QUESTION_PAGE_MAP = {
    # Format: (task_type, normalized_question): page_name
    "indoor_trash": {
        "kitchen_garbage_bin": "02_Mail_Trash.py",
        "indoor_recycling_bins": "02_Mail_Trash.py",
        "indoor_compost_or_green_waste": "02_Mail_Trash.py",
        "bathroom_trash_bin": "02_Mail_Trash.py",
        "other_room_trash_bins": "02_Mail_Trash.py",
    },
    "outdoor_trash": {
        "when_and_where_should_garbage_recycling_and_compost_bins_be_placed_for_pickup": "02_Mail_Trash.py",
        "when_and_where_should_bins_be_brought_back_in": "02_Mail_Trash.py",
    },
    "mail_handling": {
        "mail_pick_up_schedule": "02_Mail_Trash.py",
    },
}

def get_page_map():
    """Returns a mapping of section/task_type to valid Streamlit handler key."""
    return PAGE_MAP

def get_question_page_map():
    return QUESTION_PAGE_MAP

