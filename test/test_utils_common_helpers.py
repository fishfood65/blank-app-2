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
import utils.common_helpers as uh

@pytest.fixture(autouse=True)
def setup_mail_input(monkeypatch: pytest.MonkeyPatch):
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

def test_extract_grouped_mail_task_with_missing_fields(monkeypatch: pytest.MonkeyPatch):
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

def test_extract_all_trash_tasks_grouped_with_mocked_streamlit(monkeypatch: pytest.MonkeyPatch):
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
    # Provide dummy utils object
    dummy_utils = {
        "emoji_tags": { "[Weekly]": "🔁", "[Monthly]": "📅", "[Daily]": "🗓️", "[One-Time]": "📌" },
        "weekday_to_int": {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
            "Friday": 4, "Saturday": 5, "Sunday": 6
        },
        "extract_week_interval": lambda x: None,
        "extract_weekday_mentions": lambda x: [weekday_name] if weekday_name in x else [],
        "extract_dates": lambda x: [],
        "normalize_date": lambda x: None,
    }

    # Monkeypatch streamlit and get_schedule_utils
    monkeypatch.setattr(uh, "st", fake_st)
    monkeypatch.setattr(uh, "get_schedule_utils", lambda: dummy_utils)

    valid_dates = [today + timedelta(days=i) for i in range(14)]
    df = uh.extract_all_trash_tasks_grouped(valid_dates,dummy_utils)

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

        dummy_utils = {
        "emoji_tags": { "[Weekly]": "🔁", "[Monthly]": "📅", "[Daily]": "🗓️", "[One-Time]": "📌" },
        "weekday_to_int": {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6},
        "extract_week_interval": lambda x: None,
        "extract_weekday_mentions": lambda x: [weekday_name] if weekday_name in x else [],
        "extract_dates": lambda x: [],
        "normalize_date": lambda x: None,
        }

        today = datetime.today().date()
        weekday_name = today.strftime("%A")

        st.session_state["input_data"] = {
                "Trash Handling": [
                    {
                        "question": "Kitchen Trash Bin Location, Emptying Schedule and Replacement Trash Bags",
                        "answer": f"Empty every {weekday_name} and again next {weekday_name}. Bags under sink."
                    },
                    {
                        "question": "Garbage Pickup Day",
                        "answer": f"Put bins out every week on {weekday_name}."
                    },
                    {
                        "question": "Bathroom Trash Bin Emptying Schedule and Replacement Trash Bags",
                        "answer": "Take out on the 1st and 15th of every month."
                    }
                ]
            }

        valid_dates = [today + timedelta(days=i) for i in range(14)]
        df = extract_all_trash_tasks_grouped(valid_dates, utils=dummy_utils)

        print("🧪 Extracted rows:", df.to_dict(orient="records"))

        self.assertFalse(df.empty, "❌ No trash tasks were extracted")
        self.assertIn("Task", df.columns)
        self.assertIn("Date", df.columns)

    def test_generate_flat_home_schedule_markdown(self):
        markdown = generate_flat_home_schedule_markdown(self.schedule_df)
        self.assertIsInstance(markdown, str)
        self.assertIn("📅 Thursday, 2025-05-01", markdown)

    def test_get_schedule_utils(self):
        result = get_schedule_utils()
        self.assertIn("emoji_tags", result)
        self.assertIn("weekday_to_int", result)
        self.assertIn("extract_week_interval", result)
        self.assertIn("extract_weekday_mentions", result)
        self.assertIn("extract_dates", result)
        self.assertIn("normalize_date", result)
        self.assertIn("determine_frequency_tag", result)

        # Basic type checks
        self.assertIsInstance(result["emoji_tags"], dict)
        self.assertIsInstance(result["weekday_to_int"], dict)
        self.assertTrue(callable(result["extract_week_interval"]))
        self.assertTrue(callable(result["extract_weekday_mentions"]))
        self.assertTrue(callable(result["extract_dates"]))
        self.assertTrue(callable(result["normalize_date"]))
        self.assertTrue(callable(result["determine_frequency_tag"]))

        # Sample check of interval function
        self.assertEqual(result["extract_week_interval"]("Take out trash every 2 weeks"), 2)
        self.assertEqual(result["extract_week_interval"]("This is biweekly."), 2)
        self.assertEqual(result["extract_week_interval"]("Do this every week"), 1)

def test_generate_flat_home_schedule_markdown():
    schedule_df = pd.DataFrame([
        {
            "Date": "2024-01-01",
            "Task": "Clean kitchen",
            "Tag": "cleaning",
            "Source": "user",
            "Category": "home"
        },
        {
            "Date": "2024-01-02",
            "Task": "Check fire alarm",
            "Tag": "safety",
            "Source": "auto",
            "Category": "home"
        }
    ])
    result = generate_flat_home_schedule_markdown(schedule_df)

    assert isinstance(result, str)
    assert "**📅" in result  # Markdown section headers
    assert "| Task" in result  # Markdown table
    assert "Clean kitchen" in result
    assert "Check fire alarm" in result

if __name__ == "__main__":
    unittest.main()
