import streamlit as st
import json
import csv
import io
from datetime import datetime
from docx import Document
import pandas as pd
import plotly.express as px
from uuid import uuid4

def get_or_create_session_id():
    """Assigns a persistent unique ID per session"""
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid4())
    return st.session_state["session_id"]

def set_user_id(user_id):
    """Lets you manually tag a user (e.g., via login form or input"""
    st.session_state["user_id"] = user_id

def init_section(section_name):
    """Initialize a new section to group questions and answers."""
    if "input_data" not in st.session_state:
        st.session_state["input_data"] = {}
    if section_name not in st.session_state["input_data"]:
        st.session_state["input_data"][section_name] = []

def log_interaction(action_type, label, value, section_name):
    """Log a user interaction for troubleshooting or auditing."""
    if "interaction_log" not in st.session_state:
        st.session_state["interaction_log"] = []
    st.session_state["interaction_log"].append({
        "timestamp": datetime.now().isoformat(),
        "action": action_type,
        "question": label,
        "answer": value,
        "section": section_name
    })

def capture_input(label, input_fn, section_name, *args, **kwargs):
    """Displays input using input_fn and stores label, value, timestamp under a section."""
    init_section(section_name)
    value = input_fn(label, *args, **kwargs)
    entry = {
        "question": label,
        "answer": value,
        "timestamp": datetime.now().isoformat()
    }
    st.session_state["input_data"][section_name].append(entry)
    log_interaction("input", label, value, section_name)
    autosave_input_data()
    return value

def preview_input_data():
    """Display a summary of all collected input data."""
    if "input_data" not in st.session_state or not st.session_state["input_data"]:
        st.warning("No input data collected yet.")
        return
    st.markdown("### 📝 Input Summary")
    for section, entries in st.session_state["input_data"].items():
        st.markdown(f"#### 📁 {section}")
        for item in entries:
            st.markdown(f"- **{item['question']}**: {item['answer']} _(at {item['timestamp']})_")

def preview_interaction_log():
    """Display a log of all user interactions."""
    if "interaction_log" not in st.session_state or not st.session_state["interaction_log"]:
        st.info("No interactions logged yet.")
        return
    st.markdown("### 🧾 Interaction History")
    for log in st.session_state["interaction_log"]:
        st.markdown(f"- [{log['timestamp']}] {log['action'].capitalize()} — **{log['question']}** → `{log['answer']}` _(Section: {log['section']})_")

def autosave_input_data():
    """Automatically save data to session file (local or cloud placeholder)."""
    if "input_data" in st.session_state:
        st.session_state["autosaved_json"] = json.dumps(st.session_state["input_data"], indent=2)
        # Placeholder: save_to_cloud(st.session_state["autosaved_json"])

def export_input_data_as_json(file_name="input_data.json"):
    """Export the collected input data as JSON."""
    if "input_data" in st.session_state:
        json_data = json.dumps(st.session_state["input_data"], indent=2)
        st.download_button(
            label="📥 Download as JSON",
            data=json_data,
            file_name=file_name,
            mime="application/json"
        )

def export_input_data_as_csv(file_name="input_data.csv"):
    """Export the collected input data as CSV."""
    if "input_data" in st.session_state:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Section", "Question", "Answer", "Timestamp"])
        for section, entries in st.session_state["input_data"].items():
            for item in entries:
                writer.writerow([section, item["question"], item["answer"], item["timestamp"]])
        st.download_button(
            label="📄 Download as CSV",
            data=output.getvalue(),
            file_name=file_name,
            mime="text/csv"
        )

def export_input_data_as_docx(file_name="input_data.docx"):
    """Export the collected input data as DOCX."""
    if "input_data" in st.session_state:
        doc = Document()
        doc.add_heading("Collected Input Summary", level=1)
        for section, entries in st.session_state["input_data"].items():
            doc.add_heading(section, level=2)
            for item in entries:
                doc.add_paragraph(f"{item['question']}: {item['answer']} ({item['timestamp']})")
        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        st.download_button(
            label="📝 Download as DOCX",
            data=buffer,
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

# Placeholder for future enhancement: Cloud saving logic
# def save_to_cloud(json_data):
#     """Save the JSON data to a cloud storage provider."""
#     pass