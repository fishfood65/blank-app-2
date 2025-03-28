import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to TBM POC! 👋")

st.sidebar.success("Select a link above.")

st.markdown(
    """
    TBM POC is a to be named POC 

    ### Want to try it out? Start by picking out a pet game.
    - [Cat Game](https://blank-app-9veokruhyi5.streamlit.app/Cat_Bingo_Game)
    - [Dog Game](https://blank-app-9veokruhyi5.streamlit.app/Dog_Bingo_Game)
    ### Next take the output and go below
    - [Pet Guide Generator](https://blank-app-9veokruhyi5.streamlit.app/Pet_Guide_Generator)
"""
)
