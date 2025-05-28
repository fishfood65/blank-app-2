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
from docx.shared import Inches
from PIL import Image
import io
from .runbook_generator_helpers import get_schedule_utils

def check_home_progress(progress_dict):
    """
    Checks overall progress across all home levels.
    Returns a completion percentage and list of completed levels.
    """
    total_levels = len(progress_dict)
    completed = [k for k, v in progress_dict.items() if v]
    if total_levels == 0:
        return 0, []
    percent_complete = int((len(completed) / total_levels) * 100)
    return percent_complete, completed

def extract_grouped_mail_task(valid_dates):
    utils = get_schedule_utils()

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

    # Use shared utils
    interval = utils["extract_week_interval"](pick_up_schedule)
    weekday_mentions = utils["extract_weekday_mentions"](pick_up_schedule)
    tag = ""

    for ds in utils["extract_dates"](pick_up_schedule):
        parsed_date = utils["normalize_date"](ds)
        if parsed_date and parsed_date in valid_dates:
            tag = utils["emoji_tags"]["[One-Time]"]
            break

    if interval and not tag:
        tag = f"↔️ Every {interval} Weeks"
    if weekday_mentions and not interval:
        tag = utils["emoji_tags"]["[Weekly]"]
    if not tag and "monthly" in pick_up_schedule.lower():
        tag = utils["emoji_tags"]["[Monthly]"]
    if not tag:
        tag = utils["emoji_tags"]["[Daily]"]

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

def extract_all_trash_tasks_grouped(valid_dates, utils):
    utils = get_schedule_utils()

    input_data = st.session_state.get("input_data", {})
    trash_entries = input_data.get("Trash Handling", [])
    label_map = {entry["question"]: entry["answer"] for entry in trash_entries}

    emoji_tags = utils["emoji_tags"]
    weekday_to_int = utils["weekday_to_int"]
    extract_week_interval = utils["extract_week_interval"]
    extract_weekday_mentions = utils["extract_weekday_mentions"]
    extract_dates = utils["extract_dates"]
    normalize_date = utils["normalize_date"]

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

    def schedule_task(label, answer):
        rows = []
        interval = extract_week_interval(answer)
        weekday_mentions = extract_weekday_mentions(answer)

        task_already_scheduled = False

        for ds in extract_dates(answer):
            parsed_date = normalize_date(ds)
            if parsed_date and parsed_date in valid_dates:
                rows.append(add_task_row(parsed_date, label, answer, emoji_tags["[One-Time]"]))
                task_already_scheduled = True

        if interval:
            base_date = valid_dates[0]
            for d in valid_dates:
                if (d - base_date).days % (interval * 7) == 0:
                    rows.append(add_task_row(d, label, answer, f"↔️ Every {interval} Weeks"))
                    task_already_scheduled = True

        if weekday_mentions and not interval:
            for wd in weekday_mentions:
                weekday_idx = weekday_to_int.get(wd)
                for d in valid_dates:
                    if d.weekday() == weekday_idx:
                        rows.append(add_task_row(d, label, answer, emoji_tags["[Weekly]"]))
                        task_already_scheduled = True

        if "monthly" in answer.lower():
            current_date = pd.to_datetime(valid_dates[0])
            while current_date.date() <= valid_dates[-1]:
                if current_date.date() in valid_dates:
                    rows.append(add_task_row(current_date.date(), label, answer, emoji_tags["[Monthly]"]))
                    task_already_scheduled = True
                current_date += pd.DateOffset(months=1)

        if not task_already_scheduled and not weekday_mentions and not extract_dates(answer):
            for d in valid_dates:
                rows.append(add_task_row(d, label, answer, emoji_tags["[Daily]"]))
        return rows

    indoor_task_labels = [
        "Kitchen Trash Bin Location, Emptying Schedule and Replacement Trash Bags",
        "Bathroom Trash Bin Emptying Schedule and Replacement Trash Bags",
        "Other Room Trash Bin Emptying Schedule and Replacement Trash Bags",
        "Recycling Trash Bin Location and Emptying Schedule (if available) and Sorting Instructions"
    ]

    indoor_rows = []
    for label in indoor_task_labels:
        answer = str(label_map.get(label, "")).strip()
        if answer:
            indoor_rows.extend(schedule_task(label, answer))

    outdoor_rows = []

    # Garbage/Recycling combined scheduling
    pickup_related_labels = {
        "Garbage Pickup Day": [
            "Instructions for Placing and Returning Outdoor Bins",
            "What the Outdoor Trash Bins Look Like",
            "Specific Location or Instructions for Outdoor Bins"
        ],
        "Recycling Pickup Day": [
            "Instructions for Placing and Returning Outdoor Bins",
            "What the Outdoor Trash Bins Look Like",
            "Specific Location or Instructions for Outdoor Bins"
        ]
    }

    garbage_day = label_map.get("Garbage Pickup Day", "").strip()
    recycling_day = label_map.get("Recycling Pickup Day", "").strip()

    weekday_mentions_garbage = extract_weekday_mentions(garbage_day)
    weekday_mentions_recycling = extract_weekday_mentions(recycling_day)

    # Find shared days
    shared_days = set(weekday_mentions_garbage).intersection(weekday_mentions_recycling)

    scheduled_days = set()

    for pickup_type, anchor_labels in pickup_related_labels.items():
        pickup_day = label_map.get(pickup_type, "").strip()
        weekday_mentions = extract_weekday_mentions(pickup_day)

        for wd in weekday_mentions:
            weekday_idx = weekday_to_int.get(wd)
            for date in valid_dates:
                if date.weekday() == weekday_idx and wd not in scheduled_days:
                    tag = "♻️ Shared Pickup Instructions" if wd in shared_days else emoji_tags["[Weekly]"]
                    for anchor_label in anchor_labels:
                        answer = label_map.get(anchor_label, "").strip()
                        if answer:
                            outdoor_rows.append(add_task_row(
                                date, f"{pickup_type} - {anchor_label}", answer, tag
                            ))
                    scheduled_days.add(wd)

    # Include rows for Garbage and Recycling Day labels themselves
    for pickup_label in ["Garbage Pickup Day", "Recycling Pickup Day"]:
        answer = str(label_map.get(pickup_label, "")).strip()
        if answer:
            outdoor_rows.extend(schedule_task(pickup_label, answer))

    combined_df = pd.DataFrame(indoor_rows + outdoor_rows)
    if not combined_df.empty:
        return combined_df.sort_values(by=["Date", "Day", "Category", "Tag", "Task"]).reset_index(drop=True)
    else:
        return combined_df

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

def add_home_schedule_to_docx(doc, schedule_df):
    """
    Adds a grouped home schedule (Source → Date) to DOCX with embedded images inside task cells.

    Args:
        doc (Document): python-docx Document.
        schedule_df (pd.DataFrame): DataFrame with Task, Tag, Date, Source.
    """

    if schedule_df.empty:
        return

    # Set all categories to 'home'
    schedule_df["Category"] = "home"
    schedule_df["Date"] = pd.to_datetime(schedule_df["Date"], errors="coerce")
    schedule_df = schedule_df.sort_values(by=["Source", "Date", "Tag", "Task"])

    # Build image map from session
    image_map = {}
    if "trash_images" in st.session_state:
        for label, img_bytes in st.session_state["trash_images"].items():
            if img_bytes:
                keyword = label.replace(" Image", "").strip().lower()
                image_map[keyword] = img_bytes

    # Begin DOCX layout
    doc.add_page_break()
    doc.add_heading("📆 Home Maintenance Schedule", level=1)

    for source, source_group in schedule_df.groupby("Source"):
        doc.add_heading(f"🗂️ {source}", level=2)

        for date, date_group in source_group.groupby("Date"):
            day = date.strftime("%A")
            date_str = date.strftime("%Y-%m-%d")
            doc.add_heading(f"{day}, {date_str}", level=3)

            table = doc.add_table(rows=1, cols=3)
            table.style = "Table Grid"
            hdr_cells = table.rows[0].cells
            hdr_cells[0].text = "Task"
            hdr_cells[1].text = "Tag"
            hdr_cells[2].text = "Category"

            for _, row in date_group.iterrows():
                task_text = str(row["Task"])
                tag = str(row["Tag"])
                category = str(row["Category"])
                task_lower = task_text.lower()

                cells = table.add_row().cells

                # Write text first
                paragraph = cells[0].paragraphs[0]
                run = paragraph.add_run(task_text)

                # Match image to task
                for keyword, image_bytes in image_map.items():
                    if keyword in task_lower:
                        try:
                            image_stream = io.BytesIO(image_bytes)
                            image = Image.open(image_stream)
                            image.thumbnail((500, 500))  # Resize if needed

                            resized_stream = io.BytesIO()
                            image.save(resized_stream, format="PNG")
                            resized_stream.seek(0)

                            paragraph.add_run().add_picture(resized_stream, width=Inches(2.5))
                        except Exception as e:
                            paragraph.add_run(f"\n⚠️ Failed to embed image: {e}")
                        break  # One image per task

                cells[1].text = tag
                cells[2].text = category

        doc.add_paragraph("")  # spacing
