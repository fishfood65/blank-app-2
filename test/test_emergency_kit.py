import pytest
import streamlit as st
from datetime import datetime, timedelta
import builtins
import contextlib
from home_app_05_23_modified import emergency_kit_utilities
from prompts.prompts_home import emergency_kit_utilities_runbook_prompt

@pytest.fixture(autouse=True)
def patch_streamlit_functions(monkeypatch):
    """Monkeypatch Streamlit functions and session state for testing emergency_kit_utilities."""

    # Core UI mocks
    monkeypatch.setattr(st, "success", lambda msg, icon=None: None)
    monkeypatch.setattr(st, "warning", lambda msg: None)
    monkeypatch.setattr(st, "write", lambda *args, **kwargs: None)
    monkeypatch.setattr(st, "header", lambda *args, **kwargs: None)
    monkeypatch.setattr(st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(st, "code", lambda *args, **kwargs: None)
    monkeypatch.setattr(st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(st, "download_button", lambda *args, **kwargs: None)

    # Form inputs
    monkeypatch.setattr(st, "radio", lambda *args, **kwargs: "Yes")
    monkeypatch.setattr(st, "text_area", lambda *args, **kwargs: "Garage Shelf")
    monkeypatch.setattr(st, "text_input", lambda *args, **kwargs: "Matches, rope, gloves")
    monkeypatch.setattr(st, "checkbox", lambda *args, **kwargs: True)
    monkeypatch.setattr(st, "columns", lambda n: [st for _ in range(n)])
    monkeypatch.setattr(st, "form", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(st, "form_submit_button", lambda *args, **kwargs: True)
    monkeypatch.setattr(st, "expander", lambda *args, **kwargs: contextlib.nullcontext())

    # Reset session state
    st.session_state.clear()
    st.session_state.update({
        "input_data": {"Emergency Kit": []},
        "level_progress": {},
        "confirm_ai_prompt_emergency_kit": True,
        "session_id": "test-session"
    })

    yield

def test_emergency_kit_utilities_runs(monkeypatch):
    import home_app_05_23_modified
    print("Before patch:", home_app_05_23_modified.emergency_kit_utilities_runbook_prompt)

    monkeypatch.setattr("home_app_05_23_modified.emergency_kit_utilities_runbook_prompt", lambda: "Mocked Emergency Prompt")
    print("After patch:", home_app_05_23_modified.emergency_kit_utilities_runbook_prompt)

    # Run the utility
    emergency_kit_utilities()

    # Assertions
    assert "generated_prompt" in st.session_state
    assert st.session_state["generated_prompt"] == "Mocked Emergency Prompt"

    assert "homeowner_kit_stock" in st.session_state
    assert isinstance(st.session_state["homeowner_kit_stock"], list)

    assert "not_selected_items" in st.session_state
    assert isinstance(st.session_state["not_selected_items"], list)
