# task_schedule_utils.py

from collections import defaultdict
import pandas as pd
import re
from typing import List, Tuple, Optional, Dict
import streamlit as st
import json
import os
import spacy
from utils.debug_scheduler_input import debug_schedule_task_input

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

def schedule_tasks_from_templates(
    tasks: list,
    valid_dates: list,
    utils: dict,
    debug: bool = False
) -> pd.DataFrame:
    """
    Given a list of task dicts and valid_dates, produce a scheduled DataFrame using known frequency/dates.
    Returns a DataFrame with full enrichment metadata including task_id and trace keys.
    """
    if not isinstance(tasks, list):
        raise ValueError("Expected list of task dicts for 'tasks'")
    if not isinstance(valid_dates, list) or not valid_dates:
        raise ValueError("valid_dates must be a non-empty list of datetime.date objects")

    emoji_tags = utils["emoji_tags"]
    weekday_to_int = utils["weekday_to_int"]
    extract_week_interval = utils["extract_week_interval"]
    extract_weekday_mentions = utils["extract_weekday_mentions"]
    extract_dates = utils["extract_dates"]
    normalize_date = utils["normalize_date"]

    all_rows = []

    for task in tasks:
        if not isinstance(task, dict):
            continue

        label = task.get("question", "")
        answer = str(task.get("answer", "")).strip()
        category = task.get("category", "Uncategorized")
        section = task.get("section", "")
        area = task.get("area", "home")
        task_type = task.get("task_type")
        task_key = task.get("key", label)
        task_id = task.get("task_id")
        source_key = task.get("SourceKey", f"{section}::{label}")
        section_label = task.get("SectionLabel", section)
        base_row = {
            "Category": category,
            "Source": section or category,
            "Area": area,
            "task_type": task_type,
            "question": label,
            "answer": answer,
            "clean_task": task.get("clean_task", ""),
            "formatted_answer": task.get("formatted_answer", ""),
            "inferred_days": task.get("inferred_days", []),
            "key": task_key,
            "task_id": task_id,
            "SourceKey": source_key,
            "SectionLabel": section_label,
        }
        # ✅ Preserve metadata fields like expanded_task_id and inferred_day
        extra_keys = ["expanded_task_id", "inferred_day", "task_id", "formatted_answer", "clean_task", "SourceKey", "SectionLabel", "area"]
        for k in extra_keys:
            if k in task:
                base_row[k] = task[k]

        if not label or not answer:
            continue

        rows = []
        seen_keys = set()

        # 1. Explicit Dates
        for ds in extract_dates(answer):
            d = normalize_date(ds)
            if d and d in valid_dates:
                row_key = (str(d), task_key)
                if row_key not in seen_keys:
                    seen_keys.add(row_key)
                    rows.append({**base_row, **{
                        "Date": str(d),
                        "Day": d.strftime("%A"),
                        "Task": f"{label} – {answer}",
                        "Tag": emoji_tags.get("[Explicit]", "📅 One-time"),
                    }})
                    if debug:
                        st.write(f"📅 [Explicit Date] {label} → {d}")

        # 2. Repeating Intervals
        interval = extract_week_interval(answer)
        if interval:
            base_date = valid_dates[0]
            for d in valid_dates:
                if (d - base_date).days % (interval * 7) == 0:
                    row_key = (str(d), task_key)
                    if row_key not in seen_keys:
                        seen_keys.add(row_key)
                        rows.append({**base_row, **{
                            "Date": str(d),
                            "Day": d.strftime("%A"),
                            "Task": f"{label} – {answer}",
                            "Tag": emoji_tags.get("[X-Weeks]", f"↔️ Every {interval} Weeks"),
                        }})
                        if debug:
                            st.write(f"🔁 [Interval] {label} → {d}")

        # 3. Weekday Mentions
        weekday_mentions = extract_weekday_mentions(answer)
        if weekday_mentions and not interval:
            for wd in weekday_mentions:
                weekday_idx = weekday_to_int.get(wd)
                for d in valid_dates:
                    if d.weekday() == weekday_idx:
                        row_key = (str(d), task_key)
                        if row_key not in seen_keys:
                            seen_keys.add(row_key)
                            rows.append({**base_row, **{
                                "Date": str(d),
                                "Day": d.strftime("%A"),
                                "Task": f"{label} – {answer}",
                                "Tag": emoji_tags.get("[Weekly]", "🔁 Weekly"),
                            }})
                            if debug:
                                st.write(f"📆 [Weekday Mention] {label} → {d}")

        # 4. Monthly fallback
        if "monthly" in answer.lower():
            current_date = pd.to_datetime(valid_dates[0])
            while current_date.date() <= valid_dates[-1]:
                d = current_date.date()
                if d in valid_dates:
                    row_key = (str(d), task_key)
                    if row_key not in seen_keys:
                        seen_keys.add(row_key)
                        rows.append({**base_row, **{
                            "Date": str(d),
                            "Day": d.strftime("%A"),
                            "Task": f"{label} – {answer}",
                            "Tag": emoji_tags.get("[Monthly]", "📅 Monthly"),
                        }})
                        if debug:
                            st.write(f"📆 [Monthly] {label} → {d}")
                current_date += pd.DateOffset(months=1)

        all_rows.extend(rows)

    return pd.DataFrame(all_rows)


def extract_unscheduled_tasks_from_inputs(section, label_task_fn, valid_dates, utils):
    import streamlit as st
    input_data = st.session_state.get("input_data", {})
    section_data = input_data.get(section, [])

    if not isinstance(section_data, list):
        st.warning(f"⚠️ Unexpected input format for section `{section}`.")
        return pd.DataFrame()

    rows = []
    for item in section_data:
        label = item.get("question")
        answer = item.get("answer", "")
        if isinstance(answer, bool):
            answer = "Yes" if answer else "No"
        answer = str(answer).strip()

        if label and answer.lower() not in ["", "⚠️ not provided", "none"]:
            task_rows = label_task_fn(label, answer, valid_dates, utils)
            for row in task_rows:
                row["SourceKey"] = f"{section}::{label}"
                row["SectionLabel"] = section
            rows.extend(task_rows)

    return pd.DataFrame(rows)

def generate_combined_schedule(valid_dates, utils, sections=["Trash Handling", "Mail", "Quality-Oriented Household Services", "Rent or Own"]):
    combined_rows = []

    for section in sections:
        extractor_key = f"extractor_{section.replace(' ', '_').lower()}"
        extractor_fn = utils.get(extractor_key)

        if callable(extractor_fn):
            df = extractor_fn(valid_dates, utils)
            if not df.empty:
                st.success(f"✅ Extracted {len(df)} tasks from `{section}`")
                combined_rows.append(df)
            else:
                st.info(f"ℹ️ No tasks found in `{section}`")
        else:
            st.warning(f"⚠️ No extractor function found for `{section}`")

    combined_df = pd.concat(combined_rows, ignore_index=True) if combined_rows else pd.DataFrame()

    # Optional: sort final schedule
    if not combined_df.empty and "Date" in combined_df.columns:
        combined_df["Date"] = pd.to_datetime(combined_df["Date"], errors="coerce")
        combined_df = combined_df.sort_values(by="Date")

    return combined_df

def extract_unscheduled_tasks_from_inputs_with_category(section_key="task_inputs") -> pd.DataFrame:
    """
    Extracts structured tasks from session_state['task_inputs'] list.
    Each row contains task metadata like is_freq, task_type, etc.
    Ensures required columns, cleans values, and tags with SourceKey, SectionLabel, and task_id.
    """
    task_inputs = st.session_state.get(section_key, [])

    if not isinstance(task_inputs, list):
        st.warning(f"⚠️ Unexpected data format in `{section_key}`.")
        return pd.DataFrame()

    df = pd.DataFrame(task_inputs)

    # ✅ Ensure required columns
    required_cols = [
        "question", "answer", "category", "section", "area",
        "task_type", "is_freq", "key", "timestamp"
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    # ✅ Drop rows missing both question and answer
    df = df.dropna(subset=["question", "answer"], how="all")

    # ✅ Normalize text fields
    for col in ["question", "answer", "task_type", "section", "key"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # ✅ Filter out invalid answers
    df = df[df["answer"].str.lower().isin(["", "⚠️ not provided", "none"]) == False]

    # 🔖 Add SourceKey for traceability
    df["SourceKey"] = df["section"].fillna("Unknown") + "::" + df["question"].fillna("")

    # 🌝 Add SectionLabel
    df["SectionLabel"] = df["section"].fillna("Unknown")

    # 🆔 Generate stable task_id from content only
    def generate_task_id(row):
        base = f"{row.get('question', '')}|{row.get('answer', '')}|{row.get('section', '')}"
        return hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]

    df["task_id"] = df.apply(
        lambda r: r["task_id"] if "task_id" in r and pd.notna(r["task_id"]) else generate_task_id(r),
        axis=1
    )

    return df

def export_schedule_to_markdown(schedule_df: pd.DataFrame, image_base_url: Optional[str] = None) -> str:
    """
    Converts a schedule DataFrame to markdown format with optional image URLs.

    Args:
        schedule_df (pd.DataFrame): Schedule DataFrame.
        image_base_url (str, optional): Base URL to prepend to image filenames, if any.

    Returns:
        str: Markdown string.
    """
    if schedule_df.empty:
        return "_No schedule data available._"

    schedule_df["Date"] = pd.to_datetime(schedule_df["Date"], errors="coerce")
    schedule_df = schedule_df.sort_values(by=["Date", "Source", "Tag", "Task"])
    schedule_df["Day"] = schedule_df["Date"].dt.strftime("%A")
    schedule_df["DateStr"] = schedule_df["Date"].dt.strftime("%Y-%m-%d")

    grouped = schedule_df.groupby("Date")
    lines = []

    for date, group in grouped:
        day_str = date.strftime("%A, %Y-%m-%d")
        lines.append(f"### 📅 {day_str}\n")
        lines.append("| Task | Image |")
        lines.append("|------|--------|")

        for _, row in group.iterrows():
            task = row["Task"]
            image_md = ""

            # Attempt to extract image reference from task text (if tagged)
            if "📷 Image:" in task:
                task, filename = task.split("📷 Image:")
                filename = filename.strip()
                if image_base_url:
                    image_md = f"![{filename}]({image_base_url.rstrip('/')}/{filename})"
                else:
                    image_md = f"![{filename}]({filename})"

            lines.append(f"| {task.strip()} | {image_md} |")

        lines.append("")  # blank line for spacing

    return "\n".join(lines)

def get_schedule_utils():
    """Shared date parsing, tag generation, and frequency utilities."""
    print("💡 get_schedule_utils() CALLED")
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
        text = text.lower()
        print("🧪 Checking interval in text:", text)

        # ✅ Fix: use \d+ correctly, not \\d+
        if "every week" in text and not re.search(r"every \d+ week", text):
            print("✅ Detected 'every week' → returning 1")
            return 1

        match = re.search(r"every (\d+) week", text)
        if match:
            print(f"✅ Detected 'every {match.group(1)} weeks'")
            return int(match.group(1))

        if "biweekly" in text:
            return 2

        return None

    def extract_weekday_mentions(text):
        text = text.lower()
        weekdays = []
        if "weekend" in text:
            weekdays.extend(["Saturday", "Sunday"])
        if "weekday" in text:
            weekdays.extend(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
        for day in weekday_to_int:
            if day.lower() in text:
                weekdays.append(day)
        return sorted(set(weekdays))

    def extract_dates(text):
        patterns = [
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?",
            r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?"
        ]
        matches = []
        for pattern in patterns:
            matches += re.findall(pattern, text, flags=re.IGNORECASE)
        return matches
    
    def extract_annual_dates(text):
        """
        Extract annual recurrence patterns like 'every July 4' or 'every 7/4'
        and return a list of (month, day) tuples.
        """
        results = []

        # Match "every July 4" or "each July 4"
        month_day_pattern = re.findall(r"(?:every|each)?\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2})", text, re.IGNORECASE)
        for month, day in month_day_pattern:
            try:
                parsed_date = pd.to_datetime(f"{month} {day}", format="%b %d")
                results.append((parsed_date.month, parsed_date.day))
            except:
                pass

        # Match numeric form: "every 7/4" or "on 07/04"
        numeric_pattern = re.findall(r"(?:every|on)?\s*(\d{1,2})/(\d{1,2})", text)
        for month, day in numeric_pattern:
            try:
                results.append((int(month), int(day)))
            except:
                pass

        return results

    def normalize_date(date_str):
        try:
            return pd.to_datetime(date_str, errors="coerce").date()
        except:
            return None

    def determine_frequency_tag(text, valid_dates):
        interval = extract_week_interval(text)
        weekday_mentions = extract_weekday_mentions(text)

        for ds in extract_dates(text):
            parsed_date = normalize_date(ds)
            if parsed_date and parsed_date in valid_dates:
                return emoji_tags["[One-Time]"]

        if interval:
            return f"↔️ Every {interval} Weeks"
        if weekday_mentions and not interval:
            return emoji_tags["[Weekly]"]
        if "monthly" in text.lower():
            return emoji_tags["[Monthly]"]
        return emoji_tags["[Daily]"]

    return {
        "emoji_tags": emoji_tags,
        "weekday_to_int": weekday_to_int,
        "extract_week_interval": extract_week_interval,
        "extract_weekday_mentions": extract_weekday_mentions,
        "extract_dates": extract_dates,
        "normalize_date": normalize_date,
        "determine_frequency_tag": determine_frequency_tag
    }
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

def infer_relevant_days_from_text(text: str) -> list:
    """
    Extracts relevant weekdays from a text string, handling both explicit weekdays and common aliases.

    Args:
        text (str): The input text to analyze.

    Returns:
        list: Sorted list of weekday names found in the text.
    """
    if not isinstance(text, str):
        return []

    text = text.lower()
    days_found = set()

    WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    WEEKDAY_ALIASES = {
        "daily": WEEKDAYS,
        "every day": WEEKDAYS,
        "each day": WEEKDAYS,
        "weekdays": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        "weekends": ["Saturday", "Sunday"],
        "mon": ["Monday"],
        "tue": ["Tuesday"],
        "wed": ["Wednesday"],
        "thu": ["Thursday"],
        "fri": ["Friday"],
        "sat": ["Saturday"],
        "sun": ["Sunday"],
    }

    # Match full weekday names
    for day in WEEKDAYS:
        if re.search(rf"\b{day.lower()}\b", text):
            days_found.add(day)

    # Match aliases
    for alias, mapped_days in WEEKDAY_ALIASES.items():
        if alias in text:
            days_found.update(mapped_days)

    return sorted(days_found)

import re

def extract_time_mentions(text):
    if not isinstance(text, str):
        return None

    time_phrases = []

    patterns = [
        r"\bby\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b",                  # by 8am / by 7:30 pm
        r"\bafter\s+(?:breakfast|lunch|dinner|sunset)\b",            # after dinner
        r"\bbefore\s+(?:breakfast|lunch|dinner|sunset)\b",           # before lunch
        r"\b(?:at|around)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b",       # at 9 / around 10:30 am
        r"\b(?:noon|midnight|morning|evening|night)\b",              # morning, noon
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        for match in matches:
            cleaned = match.strip()
            if cleaned.lower() not in [p.lower() for p in time_phrases]:
                time_phrases.append(cleaned)

    return ", ".join(time_phrases) if time_phrases else None


def format_answer_as_bullets(answer: str) -> str:
    """
    Converts multiline answer text into Markdown-style bullet points.
    """
    if not isinstance(answer, str) or not answer.strip():
        return ""

    lines = [line.strip() for line in answer.strip().splitlines() if line.strip()]
    if len(lines) <= 1:
        return answer.strip()

    return "\n".join([f"- {line}" for line in lines])

def load_label_map():
    json_path = os.path.join(os.path.dirname(__file__), "task_label_map.json")
    print(f"📂 Trying to load label map from: {json_path}")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ File not found: {json_path}")

    with open(json_path, "r") as f:
        return json.load(f)

#LABEL_MAP = load_label_map()

def normalize_label(label: str) -> str:
    """Remove leading emojis/symbols and normalize spacing."""
    return re.sub(r"^[^\w]*(.*)", r"\1", label).strip()

nlp = spacy.load("en_core_web_sm")

def extract_key_phrase(answer: str, key: str, return_confidence: bool = False, multi_key: bool = False):
    """
    Extracts key phrases (days, timing, location, frequency, duration) from a string.
    Supports optional confidence scoring and multi-key extraction mode.
    """

    def conf(value, score=1.0):
        return (value, score) if return_confidence else value

    if not isinstance(answer, str):
        return {} if multi_key else conf("")

    answer = answer.strip()
    doc = nlp(answer)

    results = {}

    # Handle 'days'
    def extract_days():
        days = infer_relevant_days_from_text(answer)
        return conf(", ".join(days), 0.95 if days else 0.0)

    # Handle 'timing'
    def extract_timing():
        time = extract_time_mentions(answer)
        return conf(time, 0.9 if time else 0.0)

    # Handle 'location'
    def extract_location():
        for prep in ["in", "at", "on", "inside", "outside", "next to", "near", "behind", "beside", "by"]:
            match = re.search(rf"{prep}\s+([a-zA-Z\s,]+?)([.,;]|$)", answer, re.IGNORECASE)
            if match:
                return conf(match.group(0).strip(), 0.85)
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]
        return conf(noun_chunks[0], 0.6) if noun_chunks else conf("", 0.0)

    # Handle 'frequency'
    def extract_frequency():
        freq_terms = ["daily", "weekly", "biweekly", "monthly", "every", "each"]
        for token in doc:
            if token.text.lower() in freq_terms:
                return conf(token.sent.text.strip(), 0.9)
        match = re.search(r"every\s+\w+", answer, re.IGNORECASE)
        if match:
            return conf(match.group(0), 0.7)
        return conf("", 0.0)

    # Handle 'duration'
    def extract_duration():
        for ent in doc.ents:
            if ent.label_ == "DURATION":
                return conf(ent.text.strip(), 0.95)
        match = re.search(r"\b\d+\s+(minutes?|hours?|days?)\b", answer)
        if match:
            return conf(match.group(0), 0.7)
        return conf("", 0.0)

    # Key-to-function map
    extractors = {
        "days": extract_days,
        "timing": extract_timing,
        "location": extract_location,
        "frequency": extract_frequency,
        "duration": extract_duration,
    }

    if multi_key:
        for k, func in extractors.items():
            results[k] = func()
        return results

    # Single key mode
    return extractors.get(key, lambda: conf("", 0.0))()

def humanize_task(row: dict, label_map: dict, include_days: bool = False) -> str:
    if label_map is None:
        label_map = load_label_map()
    
    label = str(row.get("question", "")).strip()
    answer = str(row.get("answer", "")).strip()
    inferred_days = row.get("inferred_days", [])
    if not isinstance(inferred_days, list):
        inferred_days = []

    normalized = normalize_label(label)
    config = label_map.get(normalized)

    if not config:
        return f"{normalized} on {', '.join(inferred_days)}" if inferred_days else normalized

    if isinstance(config, str):
        return f"{config} on {', '.join(inferred_days)}" if inferred_days else config

    template = config.get("template")
    keys = config.get("keys", [])
    values = {}

    for key in keys:
        val = extract_key_phrase(answer, key)
        if not val and key == "days" and inferred_days:
            val = ", ".join(inferred_days)
        values[key] = val

    # Replace only known keys, skip others
    try:
        # Replace any missing keys with blank
        def safe_format(template_str, values_dict):
            return re.sub(r"\{(\w+)\}", lambda m: values_dict.get(m.group(1), ""), template_str)

        return safe_format(template, values).strip()
    except Exception as e:
        return f"{normalized} (⚠️ Template error)"
    
import hashlib
import pandas as pd
from typing import Optional

def enrich_task_dataframe(df: pd.DataFrame, label_map: Optional[dict] = None) -> pd.DataFrame:
    """
    Applies enrichment to a raw task DataFrame.
    Adds: clean_task, inferred_days, formatted_answer, and exploded inferred_day (if available).
    Ensures task_id is preserved or generated consistently.
    """
    if label_map is None:
        label_map = load_label_map()

    enriched_rows = []

    def ensure_task_id(row):
        """Generate SHA-1 based task_id if missing."""
        task_id = row.get("task_id", None)
        if task_id and str(task_id).strip():
            return task_id
        base = f"{row.get('question', '')}|{row.get('answer', '')}|{row.get('section', '')}|{row.get('timestamp', '')}"
        return hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]

    for _, row in df.iterrows():
        row = row.to_dict()
        row["task_id"] = ensure_task_id(row)
        row["inferred_days"] = infer_relevant_days_from_text(row.get("answer", ""))
        row["formatted_answer"] = row.get("answer", "")
        row["clean_task"] = humanize_task(row, label_map=label_map)

        # Explode inferred days
        if row["inferred_days"]:
            for day in row["inferred_days"]:
                enriched_rows.append({**row, "inferred_day": day})
        else:
            enriched_rows.append(row)

    enriched_df = pd.DataFrame(enriched_rows)

    # Optional: enforce column order or drop duplicates on task_id + inferred_day
    enriched_df = enriched_df.drop_duplicates(subset=["task_id", "inferred_day"], keep="first")

    return enriched_df

def extract_and_schedule_all_tasks(
    valid_dates: List,
    utils: Dict = None,
    df: Optional[pd.DataFrame] = None,
    section: str = None,
    output_key: Optional[str] = None,
    save_to_session: bool = True  # ✅ Explicit save flag
) -> pd.DataFrame:
    """
    Extracts and schedules tasks using enrichment and inference.
    Optionally stores results to session_state[output_key] if both output_key and save_to_session are provided.
    """
    if utils is None:
        utils = get_schedule_utils()

    if df is None:
        df = extract_unscheduled_tasks_from_inputs_with_category()

    if section:
        df = df[df["section"] == section]

    label_map = load_label_map()
    enriched_rows = []

    for row in df.to_dict(orient="records"):
        question = row.get("question", "")
        answer = row.get("answer", "")

        if isinstance(answer, bool):
            answer = "Yes" if answer else "No"
        elif answer is None:
            answer = ""

        row["task_id"] = row.get("task_id") or hashlib.sha1(
            f"{question}|{answer}|{row.get('section', '')}".encode("utf-8")
        ).hexdigest()[:10]

        row["inferred_days"] = infer_relevant_days_from_text(answer)
        row["formatted_answer"] = format_answer_as_bullets(answer)
        row["clean_task"] = humanize_task(row, include_days=False, label_map=label_map)

        if row["inferred_days"]:
            for day in row["inferred_days"]:
                new_row = row.copy()
                new_row["inferred_day"] = day
                expanded_base = f"{new_row['task_id']}|{day}"
                new_row["expanded_task_id"] = hashlib.sha1(expanded_base.encode("utf-8")).hexdigest()[:12]
                enriched_rows.append(new_row)
        else:
            row["inferred_day"] = None
            row["expanded_task_id"] = hashlib.sha1(f"{row['task_id']}|None".encode("utf-8")).hexdigest()[:12]
            enriched_rows.append(row)

    enriched_df = pd.DataFrame(enriched_rows)

    if st.session_state.get("enable_debug_mode", False):
        debug_schedule_task_input(enriched_df.to_dict(orient="records"), valid_dates)

    base_schedule = schedule_tasks_from_templates(
        tasks=enriched_df.to_dict(orient="records"),
        valid_dates=valid_dates,
        utils=utils
    )

    final_rows = []
    for scheduled_row in base_schedule.to_dict(orient="records"):
        matched = next(
            (e for e in enriched_rows
             if e.get("expanded_task_id") == scheduled_row.get("expanded_task_id")), None
        )
        combined = {**matched, **scheduled_row} if matched else scheduled_row
        final_rows.append(combined)

    final_df = pd.DataFrame(final_rows)

    if output_key and save_to_session:
        existing_df = st.session_state.get(output_key, pd.DataFrame())

        if not existing_df.empty:
            if "expanded_task_id" in final_df.columns and "expanded_task_id" in existing_df.columns:
                new_ids = final_df["expanded_task_id"].unique().tolist()
                existing_df = existing_df[~existing_df["expanded_task_id"].isin(new_ids)]

            if "task_id" in final_df.columns:
                prior_keys = final_df[["SourceKey", "key", "task_id"]].drop_duplicates()
                removed = []
                for _, row in prior_keys.iterrows():
                    src, key, task_id = row["SourceKey"], row["key"], row["task_id"]
                    mask = (
                        (existing_df["SourceKey"] == src) &
                        (existing_df["key"] == key) &
                        (existing_df["task_id"] != task_id)
                    )
                    removed.extend(existing_df[mask].to_dict("records"))
                    existing_df = existing_df[~mask]

                if removed and st.session_state.get("enable_debug_mode", False):
                    st.markdown("### 🗑️ Removed outdated tasks due to answer change")
                    for task in removed:
                        st.json(task, expanded=False)

        combined_df = pd.concat([existing_df, final_df], ignore_index=True)

        if "expanded_task_id" in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=["expanded_task_id"], keep="last")
        else:
            st.warning("⚠️ `expanded_task_id` column missing in combined_df, skipping deduplication.")

        st.session_state[output_key] = combined_df
        return combined_df

    else:
        # ✅ Pure return: do not write to session_state
        return final_df

def save_task_schedules_by_type(combined_df: pd.DataFrame):
    """
    Saves slices of combined_df into st.session_state keyed by task_type (in snake_case),
    and prepares them for DOCX placeholder insertion.

    Example:
        task_type="Security System" → "security_system_schedule_df"
        Will register placeholder: <<INSERT_SECURITY_SYSTEM_SCHEDULE_TABLE>>
    """
    if combined_df.empty or "task_type" not in combined_df.columns:
        st.warning("⚠️ No scheduled tasks to save by type.")
        return

    for task_type, group_df in combined_df.groupby("task_type"):
        if pd.isna(task_type) or group_df.empty:
            continue

        # Convert task_type to snake_case key and UPPER_CASE placeholder
        snake_key = re.sub(r"[^\w]+", "_", task_type.strip().lower()).strip("_")
        docx_key = f"{snake_key}_schedule_df"
        placeholder = f"<<INSERT_{snake_key.upper()}_SCHEDULE_TABLE>>"

        st.session_state[docx_key] = group_df.copy()

        if st.session_state.get("enable_debug_mode"):
            st.write(f"✅ Saved {len(group_df)} tasks to `{docx_key}` for placeholder `{placeholder}`")

def validate_inferred_day_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates the presence and consistency of 'inferred_day' and 'inferred_days' columns.

    Returns a DataFrame of problematic rows, or an empty DataFrame if all pass.
    """
    problems = []

    for idx, row in df.iterrows():
        inferred_days = row.get("inferred_days", [])
        inferred_day = row.get("inferred_day", None)

        # Case 1: inferred_day exists but not in inferred_days
        if inferred_day and isinstance(inferred_days, list) and inferred_day not in inferred_days:
            problems.append({
                "index": idx,
                "issue": "inferred_day not in inferred_days",
                "inferred_day": inferred_day,
                "inferred_days": inferred_days,
                "question": row.get("question"),
                "task": row.get("Task", "")
            })

        # Case 2: inferred_days exists but is not a list
        if inferred_days and not isinstance(inferred_days, list):
            problems.append({
                "index": idx,
                "issue": "inferred_days is not a list",
                "inferred_days": inferred_days,
                "question": row.get("question"),
                "task": row.get("Task", "")
            })

        # Case 3: inferred_day exists but is not a string
        if inferred_day and not isinstance(inferred_day, str):
            problems.append({
                "index": idx,
                "issue": "inferred_day is not a string",
                "inferred_day": inferred_day,
                "question": row.get("question"),
                "task": row.get("Task", "")
            })

    return pd.DataFrame(problems)


