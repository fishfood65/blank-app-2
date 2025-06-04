import streamlit as st
from utils.prompt_block_utils import generate_all_prompt_blocks
from datetime import datetime
from typing import List
from config.sections import SECTION_METADATA, LLM_SECTIONS
from utils.runbook_generator_helpers import (
    generate_docx_from_prompt_blocks,
    maybe_render_download,
)
import os
## Next Steps
### need to add validation logic to ensure data is present to generate custom prompt
### need to add pets to this
### need to add support for valid dates
### need to check if sections can be aliased

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
