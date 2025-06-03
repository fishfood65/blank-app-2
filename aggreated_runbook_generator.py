import streamlit as st
from utils.prompt_block_utils import generate_all_prompt_blocks
from utils.runbook_generator_helpers import generate_docx_from_prompt_blocks, maybe_render_download
from datetime import datetime

st.title("📘 Runbook Builder")

LLM_SECTIONS = {"home", "emergency_kit"}

# Step 1: Section Selection
selected_sections = st.multiselect(
    "Select sections to include in the runbook",
    [
        "home",
        "mail_trash_handling",
        "home_security",
        "emergency_kit",
        "emergency_kit_critical_documents",
        "bonus_level"
    ]
)

if selected_sections:
    # Step 2: Split into LLM and non-LLM blocks
    llm_blocks = []
    non_llm_blocks = []
    for section in selected_sections:
        blocks = generate_all_prompt_blocks(section)
        if section in LLM_SECTIONS:
            llm_blocks.extend(blocks)
        else:
            non_llm_blocks.extend(blocks)

    all_blocks = llm_blocks + non_llm_blocks

    # Step 3: Generate the DOCX and Markdown
    if st.button("📥 Generate Runbook"):
        buffer, markdown_text = generate_docx_from_prompt_blocks(
            blocks=all_blocks,
            use_llm=bool(llm_blocks),
            api_key=st.secrets.get("MISTRAL_TOKEN"),
            doc_heading="🏠 Household Runbook",
            debug=False
        )

        # Cache results in session state
        st.session_state["runbook_text"] = markdown_text
        st.session_state["runbook_buffer"] = buffer

    # Step 4: Render preview/download UI
    if "runbook_text" in st.session_state and "runbook_buffer" in st.session_state:
        maybe_render_download(section="runbook", filename="household_runbook.docx")

else:
    st.info("Select at least one section to generate your runbook.")
