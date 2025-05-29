import pytest
from utils.input_tracker import capture_input, get_answer, render_lock_toggle
import streamlit as st
from datetime import datetime

@pytest.fixture(autouse=True)
def setup_session_state(monkeypatch):
    monkeypatch.setattr(st, "session_state", {
        "input_data": {},
        "session_id": "test-session"
    })

def test_capture_input_adds_data(monkeypatch):
    # Setup fake Streamlit state
    import types
    import utils.input_tracker as it  # Adjust import path to match your project
    fake_st = types.SimpleNamespace()
    fake_st.session_state = {
        "input_data": {},
        "section": "Test Section",
        "session_id": "abc123"
    }

    # Replace st with fake_st in the module under test
    monkeypatch.setattr(it, "st", fake_st)
    monkeypatch.setattr(it, "log_interaction", lambda *args, **kwargs: None)
    monkeypatch.setattr(it, "autosave_input_data", lambda: None)
    monkeypatch.setattr(it, "init_section", lambda section: None)

    # Fake input function
    def fake_input(label, *args, **kwargs):
        return "test answer"

    # Run capture_input
    value = it.capture_input("Test Question", fake_input, section="Test Section")

    # Assertions
    assert value == "test answer"
    assert "Test Section" in fake_st.session_state["input_data"]
    record = fake_st.session_state["input_data"]["Test Section"][0]
    assert record["question"] == "Test Question"
    assert record["answer"] == "test answer"
    assert record["required"] is False

def test_get_answer_finds_latest_answer():
    st.session_state["input_data"] = {
        "Test Section": [
            {"question": "What?", "answer": "First"},
            {"question": "What?", "answer": "Second"}
        ]
    }
    assert get_answer("What?", section="Test Section") == "Second"

def test_render_lock_toggle(monkeypatch):
    monkeypatch.setattr(st, "toggle", lambda label, value: not value)
    monkeypatch.setattr(st, "success", lambda x: x)
    monkeypatch.setattr(st, "info", lambda x: x)

    st.session_state.clear()
    render_lock_toggle("test_lock", "Test Section")
    assert "test_lock" in st.session_state
    assert isinstance(st.session_state["test_lock"], bool)
