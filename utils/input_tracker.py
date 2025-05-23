
import streamlit as st
import json
from datetime import datetime

def init_section(section_name):
    """Initialize a new section to group questions and answers."""
    if "input_data" not in st.session_state:
        st.session_state["input_data"] = {}
    if section_name not in st.session_state["input_data"]:
        st.session_state["input_data"][section_name] = []

def capture_input(label, input_fn, section_name, *args, **kwargs):
    """
    Displays the input using input_fn and stores label, value, and timestamp under a section.
    Returns the input value.
    """
    init_section(section_name)
    value = input_fn(label, *args, **kwargs)
    entry = {
        "question": label,
        "answer": value,
        "timestamp": datetime.now().isoformat()
    }
    st.session_state["input_data"][section_name].append(entry)
    return value

def export_input_data_as_json(file_name="input_data.json"):
    """
    Export the collected input data as a downloadable JSON file.
    """
    if "input_data" in st.session_state:
        json_data = json.dumps(st.session_state["input_data"], indent=2)
        st.download_button(
            label="📥 Download Input Data as JSON",
            data=json_data,
            file_name=file_name,
            mime="application/json"
        )
