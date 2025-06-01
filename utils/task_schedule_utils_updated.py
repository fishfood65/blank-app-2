# task_schedule_utils.py

from collections import defaultdict
import pandas as pd
import re

def generate_category_keywords_from_labels(input_data):
    keyword_freq = defaultdict(int)
    for section, entries in input_data.items():
        for entry in entries:
            label = entry.get("question", "")
            words = re.findall(r"\b\w{4,}\b", label.lower())
            for word in words:
                keyword_freq[word] += 1
    candidate_keywords = {k: v for k, v in keyword_freq.items() if v >= 2}
    sorted_keywords = sorted(candidate_keywords.items(), key=lambda x: -x[1])
    return sorted_keywords

def filter_tasks(df, category=None, tag=None):
    if df.empty:
        return df
    if category:
        df = df[df["Category"] == category]
    if tag:
        df = df[df["Tag"] == tag]
    return df

def group_tasks_by_tag(df):
    if df.empty:
        return {}
    return dict(tuple(df.groupby("Tag")))

def group_tasks_by_category(df):
    if df.empty:
        return {}
    return dict(tuple(df.groupby("Category")))

def schedule_tasks_from_templates(tasks, valid_dates, utils):
    scheduled = []
    for task in tasks:
        label = task.get("question", "")
        answer = str(task.get("answer", "")).strip()
        schedule_fn = utils.get("schedule_task_from_answer")
        if label and answer and schedule_fn:
            scheduled.extend(schedule_fn(label, answer, valid_dates))
    return pd.DataFrame(scheduled)

def extract_unscheduled_tasks_from_inputs(section, label_task_fn, valid_dates, utils):
    import streamlit as st
    input_data = st.session_state.get("input_data", {})
    section_data = input_data.get(section, [])
    rows = []
    for item in section_data:
        label = item.get("question")
        answer = item.get("answer")
        if label and answer:
            task_rows = label_task_fn(label, answer, valid_dates, utils)
            rows.extend(task_rows)
    return pd.DataFrame(rows)

def generate_combined_schedule(valid_dates, utils, sections=["Trash Handling", "Mail", "Quality-Oriented Household Services", "Rent or Own"]):
    combined_rows = []
    for section in sections:
        extractor_fn = utils.get(f"extractor_{section.replace(' ', '_').lower()}")
        if extractor_fn:
            df = extractor_fn(valid_dates, utils)
            if not df.empty:
                combined_rows.append(df)
    return pd.concat(combined_rows, ignore_index=True) if combined_rows else pd.DataFrame()

def extract_unscheduled_tasks_from_inputs_with_category(section_key="input_data") -> pd.DataFrame:
    """
    Extracts all captured tasks from input_data across all sections and infers category.
    Returns a flat DataFrame with 'question', 'answer', 'section', and inferred 'category'.
    """
    all_rows = []
    input_data = st.session_state.get(section_key, {})

    for section, entries in input_data.items():
        if isinstance(entries, dict):  # Nested format
            for group_name, group_fields in entries.items():
                for label, value in group_fields.items():
                    all_rows.append({
                        "question": label,
                        "answer": value,
                        "section": section,
                        "subcategory": group_name,
                        "category": section
                    })
        elif isinstance(entries, list):  # Flat format
            for item in entries:
                label = item.get("question")
                value = item.get("answer")
                all_rows.append({
                    "question": label,
                    "answer": value,
                    "section": section,
                    "subcategory": None,
                    "category": section
                })

    return pd.DataFrame(all_rows)
