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

def extract_pet_scheduled_tasks(questions, saved_pets, valid_dates):
    """
    Extracts pet care tasks based on valid_dates, labels, and user answers.
    Returns a long-form DataFrame with Day, Pet, Task, Tag.
    """

    schedule_rows = []

    # Recurring task keyword categories
    recurring_keywords = {
        "feeding": ["feed", "feeding schedule", "food", "treats"],
        "medication": ["medication", "pill", "dose", "schedule"],
        "walk": ["walk", "walking", "exercise"],
        "grooming": ["brush", "bathing", "grooming", "nail", "teeth", "ear cleaning"],
        "play": ["play", "toys", "activities"],
        "litter": ["litter", "waste", "box"],
        "water": ["water", "bowl"]
    }

    # One-time task keywords
    one_time_keywords = ["pill", "appointment", "check-up", "vaccination", "vet", "visit"]

    def label_matches(label, keywords):
        return any(kw in label.lower() for kw in keywords)

    def extract_dates(text):
        # Extract natural language dates like "June 5", "6/5/24"
        patterns = [
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?",
            r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?"
        ]
        matches = []
        for pattern in patterns:
            matches += re.findall(pattern, text, flags=re.IGNORECASE)
        return matches

    def extract_weekday_mentions(text):
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        return [day for day in weekdays if day.lower() in text.lower()]

    def normalize_date(date_str):
        try:
            return pd.to_datetime(date_str, errors="coerce").date()
        except:
            return None

    emoji_tags = {
        "[Daily]": "🔁 Daily",
        "[Weekly]": "📅 Weekly",
        "[One-Time]": "🗓️ One-Time",
        "[Assumed-Daily]": "🔄 Assumed Daily"
    }

    for pet in saved_pets:
        pet_name = pet.get("🐕 Pet Name", "Unnamed Pet")
        for q in questions:
            label = q["label"]
            answer = pet.get(label, "").strip()
            if not answer:
                continue

            base_task = f"{pet_name}: {label} – {answer}"

            # 1. Daily (recurring tasks)
            for keywords in recurring_keywords.values():
                if label_matches(label, keywords):
                    for date_obj in valid_dates:
                        schedule_rows.append({
                            "Day": date_obj.strftime('%A'),
                            "Pet": pet_name,
                            "Task": f"{label} – {answer}",
                            "Tag": emoji_tags["[Daily]"]
                        })
                    break

            # 2. Weekly (mentions like "every Monday")
            for weekday in extract_weekday_mentions(answer):
                schedule_rows.append({
                    "Day": weekday,
                    "Pet": pet_name,
                    "Task": f"{label} – {answer}",
                    "Tag": emoji_tags["[Weekly]"]
                })

            # 3. One-Time tasks with explicit dates
            if label_matches(label, one_time_keywords):
                for ds in extract_dates(answer):
                    parsed_date = normalize_date(ds)
                    if parsed_date and parsed_date in valid_dates:
                        schedule_rows.append({
                            "Day": parsed_date.strftime('%A'),
                            "Pet": pet_name,
                            "Task": f"{label} – {answer}",
                            "Tag": emoji_tags["[One-Time]"]
                        })

    df = pd.DataFrame(schedule_rows)
    df = df.sort_values(by=["Day", "Tag", "Pet"]).reset_index(drop=True)
    return df