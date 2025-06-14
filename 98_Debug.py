import streamlit as st
# ✅ MUST BE FIRST Streamlit call

from utils.debug_utils import debug_task_input_capture_with_answers_tabs, debug_all_sections_input_capture_with_summary, render_llm_debug_info
from config.sections import SECTION_METADATA

st.title("🧠 Internal Debug Dashboard")

st.markdown("""
Use this dashboard to inspect input records, runbook generation, LLM outputs, scheduling steps, and task traceability.
""")
st.markdown("---")
render_llm_debug_info()

# ✅ Dynamically fetch all sections from metadata
all_sections = list(SECTION_METADATA.keys())
default_section = "home" if "home" in all_sections else all_sections[0]

selected_section = st.selectbox("📂 Select a section to debug:", all_sections, index=all_sections.index(default_section))

st.markdown("---")

# 🔍 Invoke the main debug tab suite
debug_task_input_capture_with_answers_tabs(selected_section)

# 📋 Optional: Show global summary (toggle preferred)
if st.checkbox("🧾 Show Full Input Summary"):
    debug_all_sections_input_capture_with_summary(
        ["home", "utilities", "emergency_kit", "mail_trash"]
    )
