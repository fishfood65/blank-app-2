import pytest
import streamlit as st
import sys
import types
from datetime import date, datetime, timedelta
from utils.utils_home_helpers import extract_grouped_mail_task, check_home_progress

@pytest.fixture(autouse=True)
def setup_mail_input(monkeypatch):
    monkeypatch.setattr(st, "session_state", {
        "input_data": {
            "Mail & Packages": [
                {"question": "📍 Mailbox Location", "answer": "Front gate"},
                {"question": "📆 Mail Pick-Up Schedule", "answer": "Every Monday"},
                {"question": "📥 What to Do with the Mail", "answer": "Sort and leave on table"},
                {"question": "📦 Packages", "answer": "Place inside door"},
                {"question": "🔑 Mailbox Key (Optional)", "answer": "Key is hanging near the door"}
            ]
        }
    })

def test_extract_grouped_mail_task_returns_dict():
    valid_dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(30)]
    result = extract_grouped_mail_task(valid_dates)
    assert isinstance(result, dict)
    assert "Task" in result
    assert "Tag" in result
    assert "Category" in result
    assert "Source" in result

def test_check_home_progress_returns_percentage_and_completed():
    progress_dict = {
        "Level 1": True,
        "Level 2": False,
        "Level 3": True,
        "Level 4": True
    }
    percent, completed = check_home_progress(progress_dict)
    assert percent == 75
    assert completed == ["Level 1", "Level 3", "Level 4"]

def test_extract_grouped_mail_task_with_missing_fields(monkeypatch):
    monkeypatch.setattr(st, "session_state", {
        "input_data": {
            "Mail & Packages": [
                {"question": "📍 Mailbox Location", "answer": ""},
                {"question": "📆 Mail Pick-Up Schedule", "answer": ""},
                {"question": "📥 What to Do with the Mail", "answer": ""}
            ]
        }
    })
    valid_dates = [date(2024, 1, 1)]
    result = extract_grouped_mail_task(valid_dates)
    assert result is None

def test_extract_all_trash_tasks_grouped_with_mocked_streamlit(monkeypatch):
    today = datetime.today().date()
    weekday_name = today.strftime("%A")

    # Create a fake streamlit session_state with correct keys
    fake_st = types.SimpleNamespace()
    fake_st.session_state = {
        "input_data": {
            "Trash Handling": [
                {
                    "question": "Kitchen Trash Bin Location, Emptying Schedule and Replacement Trash Bags",
                    "answer": f"Empty every {weekday_name}. Bags under sink."
                },
                {
                    "question": "Garbage Pickup Day",
                    "answer": f"Put bins out every week on {weekday_name}."
                },
                {
                    "question": "Bathroom Trash Bin Emptying Schedule and Replacement Trash Bags",
                    "answer": "Monthly pickup on the 1st."
                }
            ]
        }
    }

    # Monkeypatch streamlit to use this fake session state
    import utils.utils_home_helpers as uh
    monkeypatch.setattr(uh, "st", fake_st)

    valid_dates = [today + timedelta(days=i) for i in range(14)]
    df = uh.extract_all_trash_tasks_grouped(valid_dates)

    print("Extracted rows:", df.to_dict(orient="records"))
    assert not df.empty

def test_check_home_progress_empty_dict():
    percent, completed = check_home_progress({})
    assert percent == 0
    assert completed == []

def test_check_home_progress_all_false():
    progress_dict = {
        "Level 1": False,
        "Level 2": False
    }
    percent, completed = check_home_progress(progress_dict)
    assert percent == 0
    assert completed == []

def test_check_home_progress_all_true():
    progress_dict = {
        "Level 1": True,
        "Level 2": True
    }
    percent, completed = check_home_progress(progress_dict)
    assert percent == 100
    assert completed == ["Level 1", "Level 2"]

