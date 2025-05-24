import streamlit as st
import json
import csv
import io
from datetime import datetime, timedelta, date
from docx import Document
import pandas as pd
import plotly.express as px
from uuid import uuid4
import re
from collections import defaultdict


def check_home_progress(progress_dict):
    """
    Checks overall progress across all home levels.
    Returns a completion percentage and list of completed levels.
    """
    total_levels = len(progress_dict)
    completed = [k for k, v in progress_dict.items() if v]
    percent_complete = int((len(completed) / total_levels) * 100)
    return percent_complete, completed

def extract_grouped_mail_task(valid_dates):
    import re
    import pandas as pd

    emoji_tags = {
        "[Daily]": "🔁 Daily",
        "[Weekly]": "📅 Weekly",
        "[One-Time]": "🗓️ One-Time",
        "[X-Weeks]": "↔️ Every X Weeks",
        "[Monthly]": "📆 Monthly"
    }

    weekday_to_int = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
        "Friday": 4, "Saturday": 5, "Sunday": 6
    }

    def extract_week_interval(text):
        match = re.search(r"every (\d+) week", text.lower())
        if match:
            return int(match.group(1))
        if "biweekly" in text.lower():
            return 2
        return None

    def extract_weekday_mentions(text):
        return [day for day in weekday_to_int if day.lower() in text.lower()]

    def extract_dates(text):
        patterns = [
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?",
            r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?"
        ]
        matches = []
        for pattern in patterns:
            matches += re.findall(pattern, text, flags=re.IGNORECASE)
        return matches

    def normalize_date(date_str):
        try:
            return pd.to_datetime(date_str, errors="coerce").date()
        except:
            return None

    input_data = st.session_state.get("input_data", {})
    mail_entries = input_data.get("Mail & Packages", [])
    label_map = {entry["question"]: entry["answer"] for entry in mail_entries}

    mailbox_location = label_map.get("📍 Mailbox Location", "").strip()
    mailbox_key = label_map.get("🔑 Mailbox Key (Optional)", "").strip()
    pick_up_schedule = label_map.get("📆 Mail Pick-Up Schedule", "").strip()
    mail_handling = label_map.get("📥 What to Do with the Mail", "").strip()
    package_handling = label_map.get("📦 Packages", "").strip()

    if not (mailbox_location and pick_up_schedule and mail_handling):
        return None

    # ⏳ Determine scheduling tag
    interval = extract_week_interval(pick_up_schedule)
    weekday_mentions = extract_weekday_mentions(pick_up_schedule)
    tag = ""

    # 1. Specific dates
    for ds in extract_dates(pick_up_schedule):
        parsed_date = normalize_date(ds)
        if parsed_date and parsed_date in valid_dates:
            tag = emoji_tags["[One-Time]"]
            break

    # 2. Every X weeks
    if interval and not tag:
        tag = f"↔️ Every {interval} Weeks"

    # 3. Weekly
    if weekday_mentions and not interval:
        tag = emoji_tags["[Weekly]"]

    # 4. Monthly
    if not tag and "monthly" in pick_up_schedule.lower():
        tag = emoji_tags["[Monthly]"]

    # 5. Fallback
    if not tag:
        tag = emoji_tags["[Daily]"]

    # 📬 Compose task
    task_lines = [
        f"{tag} 📬 Mail should be picked up **{pick_up_schedule}**.",
        f"Mailbox is located at: {mailbox_location}."
    ]

    if mailbox_key:
        task_lines.append(f"Mailbox key info: {mailbox_key}.")

    task_lines.append(f"Instructions for mail: {mail_handling}")

    if package_handling:
        task_lines.append(f"📦 Package instructions: {package_handling}")

    return {
        "Task": "\n".join(task_lines).strip(),
        "Category": "Mail",
        "Area": "home",
        "Source": "Mail & Packages",
        "Tag": tag
    }

def extract_all_trash_tasks_grouped(valid_dates):
    """
    One-stop function: extracts indoor + outdoor trash tasks,
    parses schedule metadata, and returns a unified DataFrame.
    """
    emoji_tags = {
        "[Daily]": "🔁 Daily",
        "[Weekly]": "📅 Weekly",
        "[One-Time]": "🗓️ One-Time",
        "[X-Weeks]": "↔️ Every X Weeks",
        "[Monthly]": "📆 Monthly"
    }

    weekday_to_int = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
        "Friday": 4, "Saturday": 5, "Sunday": 6
    }

    def extract_week_interval(text):
        match = re.search(r"every (\d+) week", text.lower())
        if match:
            return int(match.group(1))
        if "biweekly" in text.lower():
            return 2
        return None

    def extract_weekday_mentions(text):
        return [day for day in weekday_to_int if day.lower() in text.lower()]

    def extract_dates(text):
        patterns = [
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?",
            r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?"
        ]
        matches = []
        for pattern in patterns:
            matches += re.findall(pattern, text, flags=re.IGNORECASE)
        return matches

    def normalize_date(date_str):
        try:
            return pd.to_datetime(date_str, errors="coerce").date()
        except:
            return None

    input_data = st.session_state.get("input_data", {})
    trash_entries = input_data.get("Trash Handling", [])
    label_map = {entry["question"]: entry["answer"] for entry in trash_entries}

    indoor_task_labels = [
        "Kitchen Trash Bin Location, Emptying Schedule and Replacement Trash Bags",
        "Bathroom Trash Bin Emptying Schedule and Replacement Trash Bags",
        "Other Room Trash Bin Emptying Schedule and Replacement Trash Bags",
        "Recycling Trash Bin Location and Emptying Schedule (if available) and Sorting Instructions"
    ]

    indoor_rows = []

    def add_task_row(date_obj, label, answer, tag):
        return {
            "Date": str(date_obj),
            "Day": date_obj.strftime("%A"),
            "Task": f"{label} – {answer}",
            "Tag": tag,
            "Category": "Trash Handling",
            "Area": "home",
            "Source": "Trash Handling"
        }

    # === Indoor Task Extraction ===
    for label in indoor_task_labels:
        answer = str(label_map.get(label, "")).strip()
        if not answer:
            continue

        interval = extract_week_interval(answer)
        weekday_mentions = extract_weekday_mentions(answer)
        task_already_scheduled = False

        for ds in extract_dates(answer):
            parsed_date = normalize_date(ds)
            if parsed_date and parsed_date in valid_dates:
                indoor_rows.append(add_task_row(parsed_date, label, answer, emoji_tags["[One-Time]"]))
                task_already_scheduled = True

        if interval:
            if weekday_mentions:
                for wd in weekday_mentions:
                    weekday_int = weekday_to_int[wd]
                    matching_dates = [d for d in valid_dates if d.weekday() == weekday_int]
                    if matching_dates:
                        base_date = matching_dates[0]
                        for d in matching_dates:
                            if (d - base_date).days % (interval * 7) == 0:
                                indoor_rows.append(add_task_row(d, label, answer, f"↔️ Every {interval} Weeks"))
                                task_already_scheduled = True
            else:
                base_date = valid_dates[0]
                for d in valid_dates:
                    if (d - base_date).days % (interval * 7) == 0:
                        indoor_rows.append(add_task_row(d, label, answer, f"↔️ Every {interval} Weeks"))
                        task_already_scheduled = True

        if weekday_mentions and not interval:
            for wd in weekday_mentions:
                weekday_idx = weekday_to_int[wd]
                for d in valid_dates:
                    if d.weekday() == weekday_idx:
                        indoor_rows.append(add_task_row(d, label, answer, emoji_tags["[Weekly]"]))
                        task_already_scheduled = True

        if "monthly" in answer.lower():
            base_date = pd.to_datetime(valid_dates[0])
            current_date = base_date
            while current_date.date() <= valid_dates[-1]:
                if current_date.date() in valid_dates:
                    indoor_rows.append(add_task_row(current_date.date(), label, answer, emoji_tags["[Monthly]"]))
                    task_already_scheduled = True
                current_date += pd.DateOffset(months=1)

        if not task_already_scheduled:
            for d in valid_dates:
                indoor_rows.append(add_task_row(d, label, answer, emoji_tags["[Daily]"]))

    # === Outdoor Task Extraction ===
    def extract_outdoor_trash_tasks_with_schedule():
        outdoor_rows = []
        for label in [
            "Instructions for Placing and Returning Outdoor Bins",
            "What the Outdoor Trash Bins Look Like",
            "Specific Location or Instructions for Outdoor Bins",
            "Garbage Pickup Day",
            "Recycling Pickup Day"
        ]:
            answer = str(label_map.get(label, "")).strip()
            if not answer:
                continue

            interval = extract_week_interval(answer)
            weekday_mentions = extract_weekday_mentions(answer)
            task_already_scheduled = False
            task_text = f"{label} – {answer}"

            for ds in extract_dates(answer):
                parsed_date = normalize_date(ds)
                if parsed_date and parsed_date in valid_dates:
                    outdoor_rows.append(add_task_row(parsed_date, label, answer, emoji_tags["[One-Time]"]))
                    task_already_scheduled = True

            if interval:
                if weekday_mentions:
                    for wd in weekday_mentions:
                        weekday_int = weekday_to_int[wd]
                        matching_dates = [d for d in valid_dates if d.weekday() == weekday_int]
                        if matching_dates:
                            base_date = matching_dates[0]
                            for d in matching_dates:
                                if (d - base_date).days % (interval * 7) == 0:
                                    outdoor_rows.append(add_task_row(d, label, answer, f"↔️ Every {interval} Weeks"))
                                    task_already_scheduled = True
                else:
                    base_date = valid_dates[0]
                    for d in valid_dates:
                        if (d - base_date).days % (interval * 7) == 0:
                            outdoor_rows.append(add_task_row(d, label, answer, f"↔️ Every {interval} Weeks"))
                            task_already_scheduled = True

            if weekday_mentions and not interval:
                for wd in weekday_mentions:
                    weekday_idx = weekday_to_int[wd]
                    for d in valid_dates:
                        if d.weekday() == weekday_idx:
                            outdoor_rows.append(add_task_row(d, label, answer, emoji_tags["[Weekly]"]))
                            task_already_scheduled = True

            if "monthly" in answer.lower():
                base_date = pd.to_datetime(valid_dates[0])
                current_date = base_date
                while current_date.date() <= valid_dates[-1]:
                    if current_date.date() in valid_dates:
                        outdoor_rows.append(add_task_row(current_date.date(), label, answer, emoji_tags["[Monthly]"]))
                        task_already_scheduled = True
                    current_date += pd.DateOffset(months=1)

            if not task_already_scheduled:
                for d in valid_dates:
                    outdoor_rows.append(add_task_row(d, label, answer, emoji_tags["[Daily]"]))

        return outdoor_rows

    outdoor_rows = extract_outdoor_trash_tasks_with_schedule()

    combined_df = pd.DataFrame(indoor_rows + outdoor_rows)
    return combined_df.sort_values(by=["Date", "Day", "Category", "Tag", "Task"]).reset_index(drop=True)

def generate_flat_home_schedule_markdown(schedule_df):
    """
    Generate a unified markdown schedule grouped by Date → Tasks.
    Sets Category to 'home' and adds uploaded image file names next to matching tasks.

    Args:
        schedule_df (pd.DataFrame): Combined schedule (trash + mail).
    Returns:
        str: Markdown string with image file names next to tasks.
    """
    import pandas as pd

    if schedule_df.empty:
        return "_No schedule data available._"

    # Normalize and sort
    schedule_df["Date"] = pd.to_datetime(schedule_df["Date"], errors="coerce")
    schedule_df = schedule_df.sort_values(by=["Date", "Source", "Tag", "Task"])
    schedule_df["Day"] = schedule_df["Date"].dt.strftime("%A")
    schedule_df["DateStr"] = schedule_df["Date"].dt.strftime("%Y-%m-%d")

    # Set category to 'home'
    schedule_df["Category"] = "home"

    # Prepare image label-to-filename map
    image_map = {}
    if "trash_images" in st.session_state:
        for label, image_data in st.session_state["trash_images"].items():
            if image_data:
                # Use uploader file name if available
                uploader_key = f"{label.replace(' Image', '')}_upload"
                file = st.session_state.get(uploader_key)
                if file and hasattr(file, "name"):
                    image_map[label.replace(" Image", "").lower()] = file.name
                else:
                    image_map[label.replace(" Image", "").lower()] = "📷 Uploaded"

    # Modify tasks with matching image references
    def append_image_filename(row):
        for label, filename in image_map.items():
            if label in row["Task"].lower():
                return f'{row["Task"]}  \n📷 Image: {filename}'
        return row["Task"]

    schedule_df["Task"] = schedule_df.apply(append_image_filename, axis=1)

    # Generate markdown by date
    sections = []
    for date, group in schedule_df.groupby("Date"):
        day = date.strftime("%A")
        date_str = date.strftime("%Y-%m-%d")
        header = f"**📅 {day}, {date_str}**"

        table = group[["Task", "Tag", "Category", "Source"]].to_markdown(index=False)
        sections.append(f"{header}\n{table}\n")

    return "\n".join(sections).strip()
