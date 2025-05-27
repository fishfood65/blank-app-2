import pytest
from utils.runbook_generator_helpers import preview_run_book_output, generate_docx_from_split_prompts, get_schedule_utils
from docx import Document
import tempfile
import os

def test_preview_run_book_output_returns_cleaned_string():
    text = "### Heading\n\n- Item 1\n- Item 2"
    result = preview_run_book_output(text)
    assert isinstance(result, str)
    assert "Heading" in result
    assert "- Item 1" in result

def test_generate_docx_from_split_prompts_creates_file():
    emergency = "### Emergency\n- Call 911"
    mail_trash = "### Mail\n- Pick up mail"
    runbook = "### House\n- Clean floors"
    doc_heading = "Test Runbook"

    with tempfile.TemporaryDirectory() as tmpdirname:
        doc_filename = os.path.join(tmpdirname, "test_output.docx")
        generate_docx_from_split_prompts(
            emergency,
            mail_trash,
            runbook,
            doc_heading,
            doc_filename
        )
        assert os.path.exists(doc_filename)
        doc = Document(doc_filename)
        assert any("Emergency" in p.text or "Mail" in p.text or "House" in p.text for p in doc.paragraphs)

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

