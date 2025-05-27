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
    st.session_state["input_data"]["Test Section"] = []
    def fake_input(label, *args, **kwargs):
        return "test answer"
    value = capture_input("Test Question", fake_input, "Test Section")
    assert value == "test answer"
    assert st.session_state["input_data"]["Test Section"][0]["question"] == "Test Question"
    assert st.session_state["input_data"]["Test Section"][0]["answer"] == "test answer"

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
