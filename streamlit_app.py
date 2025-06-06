import streamlit as st

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