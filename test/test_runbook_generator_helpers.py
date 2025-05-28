import pytest
from utils.runbook_generator_helpers import (
    preview_runbook_output, # test included
    generate_docx_from_split_prompts, # test included
    generate_docx_from_text,
    maybe_generate_prompt,
    render_prompt_preview,
    maybe_render_download,
    maybe_generate_runbook
    )
from utils.common_helpers import get_schedule_utils
from docx import Document
import tempfile
import os
import io
from contextlib import contextmanager

def test_preview_runbook_output_does_not_crash(monkeypatch):
    import streamlit as st
    monkeypatch.setattr(st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(st, "expander", lambda label, expanded=True: None)
    monkeypatch.setattr(st, "markdown", lambda x: x)
    monkeypatch.setattr(st, "warning", lambda x: x)
    
    # Just verify it doesn't raise
    preview_runbook_output("### Heading\n- Item")

class FakeChat:
    def complete(self, model, messages, max_tokens, temperature):
        print("✅ FakeChat.complete called with message:", messages[0].content)  # Debug
        class Message:
            def __init__(self):
                self.content = "Mocked response content"
        class Choice:
            def __init__(self):
                self.message = Message()
        class Response:
            def __init__(self):
                self.choices = [Choice()]
        return Response()

class FakeClient:
    def __init__(self, api_key):
        print("✅ FakeClient created with api_key:", api_key)  # Debug
        self.api_key = api_key
        self.chat = FakeChat()

@contextmanager
def fake_spinner(msg):
    print("SPINNER:", msg)
    yield

def test_generate_docx_from_split_prompts(monkeypatch):
    import streamlit as st
    import utils.runbook_generator_helpers as rh

    # Fix context manager error
    monkeypatch.setattr(st, "spinner", fake_spinner)
    monkeypatch.setattr(st, "error", lambda *args, **kwargs: print("STREAMLIT ERROR:", args[0]))
    monkeypatch.setattr("utils.runbook_generator_helpers.Mistral", lambda api_key: FakeClient(api_key))

    prompts = ["This is prompt 1.", "This is prompt 2."]
    section_titles = ["Section 1", "Section 2"]
    api_key = "fake-key"

    buffer, full_text = rh.generate_docx_from_split_prompts(
        prompts,
        api_key,
        section_titles=section_titles,
        doc_heading="Test Runbook"
    )

    print("📄 FULL TEXT RETURNED:", full_text)
    assert isinstance(buffer, io.BytesIO)
    assert "Mocked response content" in full_text

def test_get_schedule_utils_keys_and_types():
    utils = get_schedule_utils()
    expected_keys = [
        "emoji_tags",
        "weekday_to_int",
        "extract_week_interval",
        "extract_weekday_mentions",
        "extract_dates",
        "normalize_date",
        "determine_frequency_tag"
    ]
    for key in expected_keys:
        assert key in utils
    assert callable(utils["extract_week_interval"])
    assert isinstance(utils["emoji_tags"], dict)

