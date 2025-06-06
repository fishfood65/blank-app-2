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

