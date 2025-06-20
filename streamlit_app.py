import streamlit as st
from utils.debug_utils import debug_all_sections_input_capture_with_summary, clear_llm_cache, reset_all_session_state, render_llm_debug_info
import os

# ------------------------------------------------------------------
# 1️⃣  Register pages (order here = order in sidebar)
# ------------------------------------------------------------------
pages = [
    st.Page("starting_page.py",        title="Main",         icon="📍"),
    #st.Page("Dog_Bingo_Game.py", title="Dog Game",  icon="🐶"),
    #st.Page("Cat_Bingo_Game.py",     title="Cat Game",      icon="🐱"),
    #st.Page("Pet_Guide_Generator.py",     title="Pet Guide",      icon="📋"),
    st.Page("pet_app_updated.py",     title="Pet Game",      icon="📋"),
    st.Page("home_app2.py", title="Home Guide Playground", icon="🏡"),
    #st.Page("home_app_05_23_modified.py",     title="Home Guide",      icon="🏡"),
    st.Page("aggregated_runbook_generator.py", title="Customize Runbook", icon="📖"),
    st.Page("mychatapp.py",     title="Chat Q&A",      icon="🗣️"),
    st.Page("98_Debug.py", title="Debug Dashboard", icon="🧠")
]

# ------------------------------------------------------------------
# 2️⃣  Display navigation and run the chosen page
# ------------------------------------------------------------------
pg = st.navigation(pages)     # shows the sidebar selector
st.set_page_config(page_title="My Streamlit Suite", page_icon="🚀", layout="wide")
pg.run()                      # execute the selected page

st.sidebar.markdown("🔍 [Go to Debug Dashboard](#/98_Debug)")

with st.sidebar.expander("⚙️ Developer Options", expanded=False):
    st.checkbox("🐞 Enable Debug Mode", key="enable_debug_mode", value=False)
    st.checkbox("📆 Show Schedule Snapshot in Preview", key="show_schedule_snapshot", value=False)
    st.selectbox("🤖 LLM Model", ["openai/gpt-4o:online", "anthropic/claude-3-haiku", "openai/gpt-3.5-turbo", "mistralai/mistral-7b-instruct"], key="llm_model")
    if st.button("🧹 Clear LLM Cache"):
        clear_llm_cache()
    if st.button("🔄 Reset All App State"):
        reset_all_session_state()


# ✅ Then the debug check happens — safely reads the key
if st.session_state.get("enable_debug_mode"):
    st.markdown("### 🧠 LLM Debug Info")
    render_llm_debug_info()



