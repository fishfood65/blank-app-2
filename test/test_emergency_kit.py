import types
import sys
import pytest
import builtins
import streamlit as st
from datetime import datetime

@pytest.fixture(autouse=True)
def mock_streamlit(monkeypatch):
    """Setup a fake Streamlit session and patch key Streamlit functions."""
    fake_st = types.SimpleNamespace()
    fake_st.session_state = {
        "input_data": {
            "Emergency Kit": []
        },
        "level_progress": {}
    }

    # Fake streamlit st.write / st.success / st.warning
    fake_st.write = lambda *args, **kwargs: None
    fake_st.success = lambda *args, **kwargs: None
    fake_st.warning = lambda *args, **kwargs: None
    fake_st.header = lambda *args, **kwargs: None
    fake_st.radio = lambda *args, **kwargs: "Yes"
    fake_st.text_area = lambda *args, **kwargs: "Garage Shelf"
    fake_st.text_input = lambda *args, **kwargs: "Matches, rope, gloves"
    fake_st.checkbox = lambda *args, **kwargs: True
    fake_st.columns = lambda n: [fake_st for _ in range(n)]
    fake_st.form = lambda *args, **kwargs: builtins.__import__('contextlib').nullcontext()
    fake_st.form_submit_button = lambda *args, **kwargs: True
    fake_st.expander = lambda *args, **kwargs: builtins.__import__('contextlib').nullcontext()
    fake_st.button = lambda *args, **kwargs: False
    fake_st.download_button = lambda *args, **kwargs: None
    fake_st.code = lambda *args, **kwargs: None
    fake_st.stop = lambda: None

    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    monkeypatch.setitem(st.__dict__, "session_state", fake_st.session_state)

    yield  # teardown handled automatically

def test_emergency_kit_utilities_runs(monkeypatch):
    from utils.utils_home_helpers import emergency_kit_utilities

    # Monkeypatch the prompt generator to return a static string
    monkeypatch.setattr("utils.utils_home_helpers.emergency_kit_utilities_runbook_prompt", lambda: "Mocked Emergency Prompt")

    # Run the function
    emergency_kit_utilities()

    # Assertions
    assert "generated_prompt" in st.session_state
    assert isinstance(st.session_state["generated_prompt"], str)
    assert "homeowner_kit_stock" in st.session_state
    assert isinstance(st.session_state["homeowner_kit_stock"], list)
    assert "not_selected_items" in st.session_state
    assert isinstance(st.session_state["not_selected_items"], list)
