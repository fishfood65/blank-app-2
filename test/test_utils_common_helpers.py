import unittest
import pytest
import streamlit as st
import sys
import types
from datetime import date, datetime, timedelta
import pandas as pd
from utils.common_helpers import (
    extract_grouped_mail_task, 
    check_home_progress, 
    extract_all_trash_tasks_grouped, 
    generate_flat_home_schedule_markdown,
    get_schedule_utils
    )

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

class TestCommonHelpers(unittest.TestCase):

    def setUp(self):
        self.schedule_df = pd.DataFrame([
            {
                "Date": "2025-05-01",
                "Task": "Take out kitchen trash",
                "Source": "Trash Handling",
                "Tag": "kitchen",
                "Category": "home"
            },
            {
                "Date": "2025-05-02",
                "Task": "Recycle paper",
                "Source": "Trash Handling",
                "Tag": "recycling",
                "Category": "home"
            }
        ])
        self.schedule_df["Date"] = pd.to_datetime(self.schedule_df["Date"])

    def test_extract_all_trash_tasks_grouped(self):
        grouped = extract_all_trash_tasks_grouped(self.schedule_df)
        self.assertIsInstance(grouped, pd.DataFrame)
        self.assertIn("Task", grouped.columns)
        self.assertGreaterEqual(len(grouped), 1)

    def test_generate_flat_home_schedule_markdown(self):
        markdown = generate_flat_home_schedule_markdown(self.schedule_df)
        self.assertIsInstance(markdown, str)
        self.assertIn("📅 2025-05-01", markdown)
        self.assertIn("Take out kitchen trash", markdown)

    def test_get_schedule_utils(self):
        df, tags, sources = get_schedule_utils(self.schedule_df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(tags, list)
        self.assertIsInstance(sources, list)
        self.assertIn("kitchen", tags)
        self.assertIn("Trash Handling", sources)

def test_generate_flat_home_schedule_markdown():
    schedule_df = pd.DataFrame([
        {"Date": "2024-01-01", "Task": "Clean kitchen", "Source": "user"},
        {"Date": "2024-01-02", "Task": "Check fire alarm", "Source": "auto"}
    ])
    result = generate_flat_home_schedule_markdown(schedule_df)
    assert "Clean kitchen" in result
    assert "Check fire alarm" in result
    assert "2024-01-01" in result
    assert "2024-01-02" in result

def test_get_schedule_utils():
    from datetime import datetime
    st.session_state.clear()
    st.session_state["input_data"] = {
        "Home Basics": [
            {"question": "Daily Tasks", "answer": "Check mail"},
            {"question": "Weekly Tasks", "answer": "Take out trash"},
        ]
    }
    df, notes = get_schedule_utils()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert any("Check mail" in task for task in df["Task"])
    assert notes is None or isinstance(notes, str)
    
if __name__ == "__main__":
    unittest.main()
