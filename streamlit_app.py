import streamlit as st
from utils.debug_utils import debug_all_sections_input_capture_with_summary
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
    st.Page("mychatapp.py",     title="Chat Q&A",      icon="🗣️")
]

# ------------------------------------------------------------------
# 2️⃣  Display navigation and run the chosen page
# ------------------------------------------------------------------
pg = st.navigation(pages)     # shows the sidebar selector
st.set_page_config(page_title="My Streamlit Suite", page_icon="🚀", layout="wide")
pg.run()                      # execute the selected page

with st.sidebar.expander("⚙️ Developer Options", expanded=False):
    st.checkbox("🐞 Enable Debug Mode", key="enable_debug_mode", value=False)
    st.checkbox("📆 Show Schedule Snapshot in Preview", key="show_schedule_snapshot", value=False)
    st.selectbox("🤖 LLM Model", ["anthropic/claude-3-haiku", "openai/gpt-3.5-turbo", "mistralai/mistral-7b-instruct"], key="llm_model")

# ✅ Then the debug check happens — safely reads the key
if st.session_state.get("enable_debug_mode"):
    debug_all_sections_input_capture_with_summary(["home", "emergency_kit", "mail_trash"])
    st.markdown("---")
    st.markdown("### 🧠 LLM Debug Info")
    
    api_key_present = bool(os.getenv("OPENROUTER_TOKEN"))
    referer_value = os.getenv("OPENROUTER_REFERER", "❌ Not set")
    model_name = "anthropic/claude-3-haiku"  # Or make this dynamic if needed

    st.write("🔑 **API Key Loaded:**", "✅ Yes" if api_key_present else "❌ No")
    st.write("🌐 **Referer Header:**", referer_value)
    st.write("🤖 **Model in Use:**", model_name)


