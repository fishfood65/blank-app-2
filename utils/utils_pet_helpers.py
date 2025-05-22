import streamlit as st
from mistralai import Mistral, UserMessage, SystemMessage
import csv
import io
from datetime import datetime, timedelta
from docx import Document
from collections import defaultdict
import json
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import os
import pandas as pd
import re
from utils.data_access import (
    get_saved_pets_by_species,
)

def check_bingo(answers):
    for i in range(7):
        if all(answers[i][j] for j in range(7)) or all(answers[j][i] for j in range(7)):
            return True
    if all(answers[i][i] for i in range(7)) or all(answers[i][6 - i] for i in range(7)):
        return True
    return False

def flatten_answers_to_dict(questions, answers):
    return {
        questions[row * 7 + col]['label']: answers[row][col]
        for row in range(7) for col in range(7)
        if answers[row][col].strip()
    }

def get_pet_name(answers):
    try:
        return answers[0][0].strip() or "UnnamedPet"
    except:
        return "UnnamedPet"

def convert_to_csv(data_list):
    if not data_list:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=data_list[0].keys())
    writer.writeheader()
    writer.writerows(data_list)
    return output.getvalue()

def get_updated_exclusion_keywords():
    return [
        "contact", "phone", "email", "sitter", "walker",
        "favorite", "toys", "play styles", "fear", "behavioral",
        "socialization", "appearance", "trainer", "photo", "description",
        "walking equipment", "walk behavior", "food brand/type your dog eats"
    ]

def should_exclude_label(label):
    label_lower = label.lower()
    return any(kw in label_lower for kw in get_updated_exclusion_keywords())

def export_all_pets_to_docx(saved_pets, species, all_question_metadata):
    doc = Document()
    doc.add_heading(f"{species.capitalize()} Care Report", 0)
    for i, pet in enumerate(saved_pets, 1):
        name = pet.get("🐕 Pet Name", f"{species.capitalize()} #{i}")
        doc.add_heading(f"{i}. {name}", level=1)
        categories = {}
        for question, answer in pet.items():
            for q in all_question_metadata[species]:
                if q["label"] == question:
                    cat = q["category"]
                    categories.setdefault(cat, []).append((q["label"], answer))
                    break
        for cat, qas in categories.items():
            doc.add_heading(cat, level=2)
            for label, answer in qas:
                doc.add_paragraph(f"{label}: {answer}")
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf

def load_pet_for_edit(index, state_prefix, questions):
    selected = st.session_state[f"saved_{state_prefix}s"][index]
    new_answers = [['' for _ in range(7)] for _ in range(7)]
    for i in range(49):
        row, col = divmod(i, 7)
        label = questions[i]['label']
        new_answers[row][col] = selected.get(label, "")
    st.session_state[f"answers_{state_prefix}"] = new_answers
    st.session_state[f"editing_index_{state_prefix}"] = index
    st.rerun()

def extract_pet_scheduled_tasks_with_intervals(questions, saved_pets, valid_dates):
    """
    Extracts pet care tasks based on valid_dates, labels, and user answers.
    Supports daily, weekly, every X weeks, monthly, and one-time tasks.
    Returns a long-form DataFrame and a list of warnings.
    """
    schedule_rows = []
    warnings = []
    added_tasks = set()  # Prevent duplicates

    recurring_keywords = {
        "feeding": ["feed", "feeding schedule", "food", "treats"],
        "medication": ["medication", "pill", "dose", "schedule"],
        "walk": ["walk", "walking", "exercise"],
        "grooming": ["brush", "bathing", "grooming", "nail", "teeth", "ear cleaning"],
        "play": ["play", "toys", "activities"],
        "litter": ["litter", "waste", "box"],
        "water": ["water", "bowl"]
    }

    one_time_keywords = ["pill", "appointment", "check-up", "vaccination", "vet", "visit"]

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

    def label_matches(label, keywords):
        return any(kw in label.lower() for kw in keywords)

    def extract_dates(text):
        patterns = [
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?",
            r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?"
        ]
        matches = []
        for pattern in patterns:
            matches += re.findall(pattern, text, flags=re.IGNORECASE)
        return matches

    def extract_weekday_mentions(text):
        return [day for day in weekday_to_int if day.lower() in text.lower()]

    def normalize_date(date_str):
        try:
            return pd.to_datetime(date_str, errors="coerce").date()
        except:
            return None

    def extract_week_interval(text):
        match = re.search(r"every (\d+) week", text.lower())
        if match:
            return int(match.group(1))
        if "biweekly" in text.lower():
            return 2
        return None

    def extract_monthly_recurrence(text):
        return (
            "monthly" in text.lower() or
            "every month" in text.lower() or
            re.search(r"every \d+ month", text.lower())
        )

    def get_pet_display_name(pet_dict):
        for k, v in pet_dict.items():
            if "name" in k.lower() and isinstance(v, str) and v.strip():
                return v.strip()
        return "Unnamed Pet"

    def add_task(day, pet, task, tag):
        key = (day, pet, task)
        if key not in added_tasks:
            schedule_rows.append({"Day": day, "Pet": pet, "Task": task, "Tag": tag})
            added_tasks.add(key)

    for pet in saved_pets:
        pet_name = get_pet_display_name(pet)

        for q in questions:
            label = q["label"]
            answer = pet.get(label, "").strip()
            if not answer or should_exclude_label(label):
                continue

            weekday_mentions = extract_weekday_mentions(answer)
            interval = extract_week_interval(answer)
            task_text = f"{label} – {answer}"

            # 1. Daily recurring ONLY if no interval or weekday explicitly mentioned
            if (
                any(label_matches(label, kws) for kws in recurring_keywords.values()) and
                not extract_week_interval(answer) and
                not extract_weekday_mentions(answer) and
                not extract_monthly_recurrence(answer)
            ):
                for d in valid_dates:
                    add_task(d.strftime('%A'), pet_name, task_text, emoji_tags["[Daily]"])

            # 2. Weekly (specific weekdays)
            for wd in weekday_mentions:
                weekday_int = weekday_to_int[wd]
                for d in valid_dates:
                    if d.weekday() == weekday_int:
                        add_task(d.strftime('%A'), pet_name, task_text, emoji_tags["[Weekly]"])

            # 3. Every X weeks
            if interval:
                if weekday_mentions:
                    for wd in weekday_mentions:
                        weekday_int = weekday_to_int[wd]
                        matching_dates = [d for d in valid_dates if d.weekday() == weekday_int]
                        if not matching_dates:
                            continue
                        base_date = matching_dates[0]
                        for d in matching_dates:
                            if (d - base_date).days % (interval * 7) == 0:
                                add_task(d.strftime('%A'), pet_name, task_text, f"↔️ Every {interval} Weeks")
                else:
                    base_date = valid_dates[0]
                    for d in valid_dates:
                        if (d - base_date).days % (interval * 7) == 0:
                            add_task(d.strftime('%A'), pet_name, task_text, f"↔️ Every {interval} Weeks")
                    warnings.append(
                        f"⚠️ '{label}' mentions every {interval} weeks but no weekday. Defaulted to {base_date.strftime('%A')}."
                    )

            # 4. Monthly
            if extract_monthly_recurrence(answer) and not weekday_mentions:
                base_date = pd.to_datetime(valid_dates[0])
                current_date = base_date
                while current_date.date() <= valid_dates[-1]:
                    if current_date.date() in valid_dates:
                        add_task(current_date.strftime('%A'), pet_name, task_text, emoji_tags["[Monthly]"])
                    current_date += pd.DateOffset(months=1)
                warnings.append(
                    f"⚠️ '{label}' mentions monthly frequency with no weekday. Scheduled starting {base_date.strftime('%Y-%m-%d')} monthly."
                )

            # 5. One-time events
            if label_matches(label, one_time_keywords):
                for ds in extract_dates(answer):
                    parsed_date = normalize_date(ds)
                    if parsed_date and parsed_date in valid_dates:
                        add_task(parsed_date.strftime('%A'), pet_name, task_text, emoji_tags["[One-Time]"])

    df = pd.DataFrame(schedule_rows)
    df = df.sort_values(by=["Day", "Tag", "Pet"]).reset_index(drop=True)
    return df, warnings

def get_pet_display_name(pet_dict):
    """Try to retrieve a pet's name from any key that looks like 'Name'."""
    for k in pet_dict:
        if "name" in k.lower():
            val = pet_dict[k].strip()
            if val:
                return val
    return "Unnamed Pet"
