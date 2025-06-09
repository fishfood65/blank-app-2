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
    Returns a DataFrame with full enrichment metadata.
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
    determine_frequency_tag = utils["determine_frequency_tag"]

    all_rows = []

    for task in tasks:
        if not isinstance(task, dict):
            continue

        label = task.get("question", "")
        answer = str(task.get("answer", "")).strip()
        category = task.get("category", "Uncategorized")
        section = task.get("section", "")
        area = task.get("area", "home")
        task_type = task.get("task_type", None)
        source = section or category

        if not label or not answer:
            continue

        rows = []  # buffer for deduplication
        seen_keys = set()

        # 1. One-time explicit dates
        for ds in extract_dates(answer):
            d = normalize_date(ds)
            if d and d in valid_dates:
                row_key = (str(d), label)
                if row_key not in seen_keys:
                    seen_keys.add(row_key)
                    rows.append({
                        "Date": str(d),
                        "Day": d.strftime("%A"),
                        "Task": f"{label} – {answer}",
                        "Tag": emoji_tags.get("[Explicit]", "📅 One-time"),
                        "Category": category,
                        "Source": source,
                        "Area": area,
                        "task_type": task_type,
                        "question": label,
                        "answer": answer,
                        "clean_task": task.get("clean_task", ""),
                        "formatted_answer": task.get("formatted_answer", ""),
                        "inferred_days": task.get("inferred_days", []),
                    })
                    if debug:
                        st.write(f"📅 [Explicit Date] {label} → {d}")

        # 2. Repeating intervals
        interval = extract_week_interval(answer)
        if interval:
            base_date = valid_dates[0]
            for d in valid_dates:
                if (d - base_date).days % (interval * 7) == 0:
                    row_key = (str(d), label)
                    if row_key not in seen_keys:
                        seen_keys.add(row_key)
                        rows.append({
                            "Date": str(d),
                            "Day": d.strftime("%A"),
                            "Task": f"{label} – {answer}",
                            "Tag": emoji_tags.get("[Weekly]", "🔁 Weekly"),
                            "Category": category,
                            "Source": source,
                            "Area": area,
                            "task_type": task_type,
                            "question": label,
                            "answer": answer,
                            "clean_task": task.get("clean_task", ""),
                            "formatted_answer": task.get("formatted_answer", ""),
                            "inferred_days": task.get("inferred_days", []),
                        })
                        if debug:
                            st.write(f"🔁 [Interval] {label} → {d}")

        # 3. Weekday mentions
        weekday_mentions = extract_weekday_mentions(answer)
        if weekday_mentions and not interval:
            for wd in weekday_mentions:
                weekday_idx = weekday_to_int.get(wd)
                for d in valid_dates:
                    if d.weekday() == weekday_idx:
                        row_key = (str(d), label)
                        if row_key not in seen_keys:
                            seen_keys.add(row_key)
                            rows.append({
                                "Date": str(d),
                                "Day": d.strftime("%A"),
                                "Task": f"{label} – {answer}",
                                "Tag": emoji_tags.get("[Weekly]", "🔁 Weekly"),
                                "Category": category,
                                "Source": source,
                                "Area": area,
                                "task_type": task_type,
                                "question": label,
                                "answer": answer,
                                "clean_task": task.get("clean_task", ""),
                                "formatted_answer": task.get("formatted_answer", ""),
                                "inferred_days": task.get("inferred_days", []),
                            })
                            if debug:
                                st.write(f"📆 [Weekday Mention] {label} → {d}")

        # 4. Monthly fallback
        if "monthly" in answer.lower():
            current_date = pd.to_datetime(valid_dates[0])
            while current_date.date() <= valid_dates[-1]:
                d = current_date.date()
                row_key = (str(d), label)
                if d in valid_dates and row_key not in seen_keys:
                    seen_keys.add(row_key)
                    rows.append({
                        "Date": str(d),
                        "Day": d.strftime("%A"),
                        "Task": f"{label} – {answer}",
                        "Tag": emoji_tags.get("[Monthly]", "📅 Monthly"),
                        "Category": category,
                        "Source": source,
                        "Area": area,
                        "task_type": task_type,
                        "question": label,
                        "answer": answer,
                        "clean_task": task.get("clean_task", ""),
                        "formatted_answer": task.get("formatted_answer", ""),
                        "inferred_days": task.get("inferred_days", []),
                    })
                    if debug:
                        st.write(f"📆 [Monthly] {label} → {d}")
                current_date += pd.DateOffset(months=1)

        # Push deduplicated task rows
        all_rows.extend(rows)

    return pd.DataFrame(all_rows)

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

def extract_unscheduled_tasks_from_inputs_with_category(section_key="task_inputs") -> pd.DataFrame:
    """
    Extracts structured tasks from session_state['task_inputs'] list.
    Each row already contains task metadata like is_freq, task_type, etc.
    """
    task_inputs = st.session_state.get(section_key, [])
    df = pd.DataFrame(task_inputs)

    # Ensure expected columns exist
    for col in ["question", "answer", "category", "section", "area", "task_type", "is_freq"]:
        if col not in df.columns:
            df[col] = None

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
    
def enrich_task_dataframe(df: pd.DataFrame, label_map: Optional[dict] = None) -> pd.DataFrame:
    """
    Applies enrichment to a raw task DataFrame.
    Adds: clean_task, inferred_days, formatted_answer, and exploded inferred_day (if available).
    """
    label_map = load_label_map()
    enriched_rows = []

    for _, row in df.iterrows():
        row = row.to_dict()
        row["inferred_days"] = infer_relevant_days_from_text(row.get("answer", ""))
        row["formatted_answer"] = row.get("answer", "")
        row["clean_task"] = humanize_task(row, label_map=label_map)

        if row["inferred_days"]:
            for day in row["inferred_days"]:
                enriched_rows.append({**row, "inferred_day": day})
        else:
            enriched_rows.append(row)

    return pd.DataFrame(enriched_rows)

def extract_and_schedule_all_tasks(
    valid_dates: List,
    utils: Dict = None,
    df: Optional[pd.DataFrame] = None,
    section: str = None,
    output_key: str = None
) -> pd.DataFrame:
    """
    Extracts and schedules tasks using enrichment and inference:
    - Applies human-readable labels
    - Infers day mentions
    - Formats answers
    - Defers scheduling if no day match is found
    Optionally filters by section and stores result in a specific session key.
    """
    if utils is None:
        utils = get_schedule_utils()

    # 🧪 If no df passed in, extract all
    if df is None:
        df = extract_unscheduled_tasks_from_inputs_with_category()

    label_map = load_label_map()
    raw_df = extract_unscheduled_tasks_from_inputs_with_category()

    # Optional section filtering
    if section:
        raw_df = raw_df[raw_df["section"] == section]

    enriched_rows = []
    for row in raw_df.to_dict(orient="records"):
        question = row.get("question", "")
        answer = row.get("answer", "")

        if isinstance(answer, bool):
            answer = "Yes" if answer else "No"
        elif answer is None:
            answer = ""

        row["inferred_days"] = infer_relevant_days_from_text(answer)
        row["formatted_answer"] = format_answer_as_bullets(answer)
        row["clean_task"] = humanize_task(row, include_days=False, label_map=label_map)

        enriched_rows.append(row)

    enriched_df = pd.DataFrame(enriched_rows)

    # Debug: visualize tasks being scheduled
    if st.session_state.get("enable_debug_mode", False):
        debug_schedule_task_input(enriched_df.to_dict(orient="records"), valid_dates)

    final_rows = []
    base_schedule = schedule_tasks_from_templates(
        tasks=enriched_df.to_dict(orient="records"),
        valid_dates=valid_dates,
        utils=utils
    )

    for scheduled_row in base_schedule.to_dict(orient="records"):
        matched = next(
            (e for e in enriched_rows
             if e["question"] == scheduled_row.get("question")
             and e["answer"] == scheduled_row.get("answer")), None
        )
        combined = {**scheduled_row, **matched} if matched else scheduled_row
        final_rows.append(combined)

    final_df = pd.DataFrame(final_rows)

    # ✅ Optionally store result in session
    if output_key:
        st.session_state[output_key] = final_df

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


