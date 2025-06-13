import streamlit as st
from utils.debug_utils import debug_task_input_capture_with_answers_tabs
from config.sections import SECTION_METADATA

st.set_page_config(page_title="🪪 Debug Tools", layout="wide")

st.title("🧠 Internal Debug Dashboard")

st.markdown("""
Use this dashboard to inspect input records, runbook generation, LLM outputs, scheduling steps, and task traceability.
""")

# ✅ Dynamically fetch all sections from metadata
all_sections = list(SECTION_METADATA.keys())
default_section = "home" if "home" in all_sections else all_sections[0]

selected_section = st.selectbox("📂 Select a section to debug:", all_sections, index=all_sections.index(default_section))

st.markdown("---")

# 🔍 Invoke the main debug tab suite
debug_task_input_capture_with_answers_tabs(selected_section)

