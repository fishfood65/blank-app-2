import streamlit as st
from utils.prompt_block_utils import generate_all_prompt_blocks
from datetime import datetime
from typing import List
import os
from config.sections import SECTION_METADATA, LLM_SECTIONS
from utils.runbook_generator_helpers import (
    generate_docx_from_prompt_blocks,
    maybe_render_download,
)
from utils.preview_helpers import render_saved_section
from config import SAVED_SECTIONS
import pandas as pd

## Next Steps
### need to add validation logic to ensure data is present to generate custom prompt
### need to add pets to this
### need to add support for valid dates
### need to check if sections can be 

def render_saved_section_status_table(sections):
    table_data = []

    for section in sections:
        label = section["label"]
        icon = section.get("icon", "📄")
        md = st.session_state.get(section["md_key"])
        docx = st.session_state.get(section["docx_key"])
        
        status = "🟢 Available" if md or docx else "🔴 Missing"
        table_data.append({
            "Section": f"{icon} {label}",
            "Status": status,
            "Markdown": "✅" if md else "❌",
            "DOCX": "✅" if docx else "❌",
        })

    df = pd.DataFrame(table_data)
    st.markdown("### 🧾 Saved Sections Overview")
    st.table(df)


st.subheader("📄 View Previously Generated Sections")

render_saved_section_status_table(SAVED_SECTIONS)

tab1, tab2 = st.tabs(["🧾 Placeholder", "📄 View Previously Generated Sections"])

    # Add more here if needed:
    # render_saved_section("Emergency Kit", "emergency_markdown", "emergency_docx", "emergency_kit")

# -- Tab 1: Previously Generated Output --
with tab2:
    render_saved_section(
        label="Utility Providers",
        md_key="utility_markdown",
        docx_key="utility_docx",
        file_prefix="utility_providers"
    )

    # Add more here if needed:
    # render_saved_section("Emergency Kit", "emergency_markdown", "emergency_docx", "emergency_kit")













################## old code

def multi_section_runbook():
    st.header("📘 Multi-Section Emergency Runbook Generator")

    # Step 1: Section Selection from metadata
    section_options = [k for k, v in SECTION_METADATA.items() if v.get("enabled", True)]
    section_labels = [SECTION_METADATA[k]["label"] for k in section_options]
    label_to_key = {SECTION_METADATA[k]["label"]: k for k in section_options}

    selected_labels = st.multiselect("Select sections to include:", section_labels)
    selected_sections = [label_to_key[label] for label in selected_labels]

    # Step 2: Collect Prompt Blocks
    if selected_sections:
        llm_blocks = []
        non_llm_blocks = []

        for section in selected_sections:
            blocks = generate_all_prompt_blocks(section)
            if section in LLM_SECTIONS:
                llm_blocks.extend(blocks)
            else:
                non_llm_blocks.extend(blocks)

        all_blocks = llm_blocks + non_llm_blocks

        # Step 3: Preview blocks if debug mode
        if st.session_state.get("enable_debug_mode"):
            st.markdown("### 🧾 Prompt Preview")
            for i, block in enumerate(all_blocks):
                st.code(f"[{i + 1}] {block}", language="markdown")

        # Step 4: Generate Runbook
        generate_key = "generate_runbook"
        if st.button("📥 Generate Runbook"):
            st.session_state[generate_key] = True

        if st.session_state.get(generate_key):
            st.info("⚙️ Calling generate_docx_from_prompt_blocks...")
            buffer, markdown_text = generate_docx_from_prompt_blocks(
                blocks=all_blocks,
                use_llm=True,
                api_key=os.getenv("MISTRAL_TOKEN"),
                doc_heading="🏠 Multi-Section Emergency Runbook",
                debug=st.session_state.get("enable_debug_mode", False),
            )

            # Cache output in session state
            st.session_state["runbook_text"] = markdown_text
            st.session_state["runbook_buffer"] = buffer
            st.session_state["runbook_ready"] = True

    # Step 5: Show preview and download link
    if st.session_state.get("runbook_ready"):
        st.success("✅ Runbook Ready!")
        if maybe_render_download(section="runbook", filename="multi_section_runbook.docx"):
            # Optional level tracking
            if "level_progress" not in st.session_state:
                st.session_state["level_progress"] = {}
            st.session_state["level_progress"]["runbook"] = True
    else:
        st.info("ℹ️ Click the button above to generate your runbook.")
