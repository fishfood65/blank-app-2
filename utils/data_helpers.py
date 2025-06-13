from typing import List, Optional, Tuple
import streamlit as st
import json
import csv
import io
from datetime import datetime, timedelta, date
import pandas as pd
import plotly.express as px
from uuid import uuid4
import re
from collections import defaultdict
from config.sections import SECTION_METADATA

DEFAULT_COMMON_SECTIONS = set(SECTION_METADATA.keys())

def get_or_create_session_id():
    """Assigns a persistent unique ID per session"""
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid4())
    return st.session_state["session_id"]

def set_user_id(user_id):
    """Lets you manually tag a user (e.g., via login form or input"""
    st.session_state["user_id"] = user_id

def init_section(section_name):
    """Initialize a new section to group questions and answers."""
    if "input_data" not in st.session_state:
        st.session_state["input_data"] = {}
    if section_name not in st.session_state["input_data"]:
        st.session_state["input_data"][section_name] = []

def log_interaction(action_type, label, value, section_name):
    """Log a user interaction for troubleshooting or auditing."""
    if "interaction_log" not in st.session_state:
        st.session_state["interaction_log"] = []
    st.session_state["interaction_log"].append({
        "timestamp": datetime.now().isoformat(),
        "action": action_type,
        "question": label,
        "answer": value,
        "section": section_name
    })

def capture_input(
    label: str,
    input_fn,
    section: str,
    *args,
    validate_fn=None,
    preprocess_fn=None,
    required: bool = False,
    autofill: bool = True,
    metadata: dict = None,
    **kwargs
):
    """
    Captures user input and records structured metadata in session state.

    Returns:
        The final processed input value.
    """
    # Use explicit key or generate one
    default_key = f"{section}_{sanitize_label(label)}"
    unique_key = kwargs.get("key", default_key)
    kwargs["key"] = unique_key

    # Optional: dynamic default value via callable
    value_fn = kwargs.pop("value_fn", None)
    if callable(value_fn):
        kwargs["value"] = value_fn()

    # Auto-fill value from get_answer()
    elif autofill:
        default_value = get_answer(key=label, section=section)
        if default_value is not None and "value" not in kwargs:
            kwargs["value"] = default_value
        
        # Prevent .value for incompatible widgets
        if hasattr(input_fn, "__name__") and input_fn.__name__ in [st.radio, st.selectbox, st.multiselect] and "value" in kwargs:
            kwargs.pop("value", None)
    
    # Optional: set default placeholder for clarity
    kwargs.setdefault("placeholder", f"Enter your {label.lower()}")

    # Render input widget
    value = input_fn(label, *args, **kwargs)

    # Optional transform
    if preprocess_fn:
        try:
            value = preprocess_fn(value)
        except Exception as e:
            st.error(f"❌ Error processing input: {e}")
            return None

    # Optional validation
    if validate_fn:
        try:
            if not validate_fn(value):
                st.warning(f"⚠️ Invalid value for '{label}'")
                return None
        except Exception as e:
            st.error(f"❌ Validation failed: {e}")
            return None

    init_section(section)

    entry = {
        "question": label,
        "answer": value,
        "key": unique_key,  # ✅ required for get_answer to match
        "timestamp": datetime.now().isoformat(),
        "input_type": getattr(input_fn, "__name__", str(input_fn)),
        "section": section,
        "session_id": st.session_state.get("session_id"),
        "required": required,
        "metadata": metadata or {},
    }

    st.session_state.setdefault("input_data", {}).setdefault(section, [])

    # ✅ Optional: overwrite previous entry for same key
    section_entries = st.session_state["input_data"][section]
    section_entries = [e for e in section_entries if e.get("key") != unique_key]
    section_entries.append(entry)
    st.session_state["input_data"][section] = section_entries
    
    try:
        log_interaction("input", label, value, section)
    except Exception as e:
        if st.session_state.get("enable_debug_mode"):
                st.error(f"❌ log_interaction failed: {e}")
    autosave_input_data()

    return value

def update_or_log_task(
    question: str,
    answer: str,
    section: str,
    task_type: str = None,
    is_freq: bool = False,
    key: Optional[str] = None,
    area: str = "home",
    canonical_map: Optional[dict] = None,
    overwrite_label: Optional[str] = None,
):
    """
    Logs a task input in session state, replacing any existing task with the same key.

    Args:
        question (str): The input label shown to the user.
        answer (str): The user's response.
        section (str): The logical section (e.g. "utilities", "trash").
        task_type (str): Logical category of task (e.g. "utilities").
        is_freq (bool): Whether the task has a frequency.
        key (str): Optional key override. Defaults to sanitized label or canonical.
        area (str): High-level category ("home", "pets", etc.)
        canonical_map (dict): Optional mapping from question → canonical key.
        overwrite_label (str): Optional new label to display (while keeping canonical key).
    """
    if not answer or str(answer).strip().lower() in ["", "⚠️ not provided", "n/a"]:
        return

    final_question = overwrite_label or question

    # Canonical or default key
    if canonical_map:
        canonical_key = canonical_map.get(question.lower(), key or sanitize_label(question))
    else:
        canonical_key = key or sanitize_label(question)

    new_entry = {
        "question": final_question,
        "answer": answer.strip(),
        "key": canonical_key,
        "category": section,
        "section": section,
        "area": area,
        "task_type": task_type,
        "is_freq": is_freq,
        "timestamp": datetime.now().isoformat()
    }

    task_inputs = st.session_state.setdefault("task_inputs", [])
    task_inputs = [t for t in task_inputs if t.get("key") != canonical_key]
    task_inputs.append(new_entry)
    st.session_state["task_inputs"] = task_inputs

# Wrapper for task-based inputs
def register_task_input(
    label,
    input_fn,
    section: str,
    is_freq: bool = False,
    task_type: str = None,
    key: str = None,  # ✅ Explicit key
    area: str = "home", # ✅ NEW: Promote `area` to argument
    *args,
    **kwargs
):
    """
    Registers a user input and stores it in session state for both tasks and non-tasks.
    Prevents duplicate task entries by overwriting old ones based on a unique key.
    Compatible with get_answer() via consistent key handling.

    Args:
        label (str): The input label (used for question and task description).
        input_fn: Streamlit input function (e.g., st.text_area).
        section (str): The section name (flat string, e.g., "emergency_kit").
        is_freq (bool): Whether this task is frequency-based.
        task_type (str): Optional logical type (e.g., 'mail', 'trash').
        key (str): Optional key to ensure matching across inputs and get_answer().
        *args, **kwargs: Passed through to capture_input().
    """
    # ✅ Use or generate consistent key
    key = key or f"{section}_{sanitize_label(label)}"
    kwargs["key"] = key

    # ✅ Add metadata (task control block)
    metadata = {
        "is_task": True,
        "task_label": label,
        "is_freq": is_freq,
        "area": area,
        "section": section,
        "task_type": task_type,
    }
    kwargs["metadata"] = metadata

    # ✅ Capture user input
    value = capture_input(label, input_fn, section=section, *args, **kwargs)

    if value not in (None, ""):
        timestamp = datetime.now().isoformat()

        # ✅ Delegate to canonical task logging function
        update_or_log_task(
            question=label,
            answer=value,
            section=section,
            task_type=task_type,
            is_freq=is_freq,
            key=key,
            area=area,
            canonical_map=kwargs.get("canonical_map"),
            overwrite_label=kwargs.get("overwrite_label")
        )

        # ✅ Input data block (mirrors task_inputs)
        input_data = st.session_state.setdefault("input_data", {}).setdefault(section, [])
        input_data = [i for i in input_data if i.get("key") != key]
        input_data.append({
            "question": label,
            "answer": value,
            "section": section,
            "key": key,
            "timestamp": timestamp,
        })
        st.session_state["input_data"][section] = input_data  # ✅ Always reassign

    return value

def register_input_only(
    label: str,
    value: str,
    section: str,
    key: str = None,
    metadata: dict = None,
    *,
    allow_empty: bool = False
):
    """
    Logs a non-task input entry in `input_data`, but skips `task_inputs`.

    Args:
        label (str): Display label or question.
        value (str): User or LLM-provided answer.
        section (str): Which section it belongs to.
        key (str, optional): Optional key for deduplication.
        metadata (dict, optional): Optional metadata dictionary.
        allow_empty (bool): If True, store even if value is empty.
    """
    if not allow_empty and (not value or str(value).strip() == ""):
        return

    key = key or f"{section}_{sanitize_label(label)}"
    timestamp = datetime.now().isoformat()

    input_entry = {
        "question": label,
        "answer": value.strip(),
        "section": section,
        "key": key,
        "timestamp": timestamp,
        "metadata": metadata or {},
    }

    section_entries = st.session_state.setdefault("input_data", {}).setdefault(section, [])
    section_entries = [e for e in section_entries if e.get("key") != key]
    section_entries.append(input_entry)
    st.session_state["input_data"][section] = section_entries

def sanitize(text):
    if not isinstance(text, str):
        print(f"[DEBUG] sanitize received invalid type: {text} ({type(text)})")
        return ""
    return (
        text.lower().strip()
        .replace(" ", "_")
        .replace("?", "")
        .replace(":", "")
        .replace("-", "_")
    )

def get_answer(
    *,
    key: str,
    section: str,
    nested_parent: str = None,
    nested_child: str = None,
    verbose: bool = False,
    common_sections: set = None
) -> str | None:
    """
    Retrieves the most recent answer for a given key in a section.
    Supports flat lookup, nested fallback, and optional verbose debugging.
    Nested_parent/nested_child is prioritized before task_inputs or input_data

    Args:
        section (str): Logical section name (e.g., "utilities", "emergency_kit").
        key (str): The question label or key to retrieve.
        nested_parent (str): Optional session_state parent (e.g., "input_data").
        nested_child (str): Optional nested dict under parent.
        verbose (bool): If True, shows debug output.
        common_sections (set): Optional override set of valid section names.

    Returns:
        str | None: Answer if found.
    """
    norm_key = sanitize(key)

    # Warn if arguments may be reversed
    common_set = common_sections or DEFAULT_COMMON_SECTIONS
    if section not in common_set and key in common_set:
        if verbose:
            st.warning(f"⚠️ `section='{section}'` may be reversed with `key='{key}'`")

    # Optional nested lookup (e.g., input_data["home"]["city"])
    if nested_parent and nested_child:
        nested_val = (
            st.session_state.get(nested_parent, {})
            .get(nested_child, {})
            .get(key)
        )
        if nested_val is not None:
            if verbose:
                st.write(f"✅ [Nested Fallback] Found in `{nested_parent} → {nested_child} → {key}`")
            return nested_val

    # 1️⃣ Look in st.session_state["task_inputs"]
    entries = st.session_state.get("task_inputs", [])
    for entry in entries:
        if entry.get("section") != section:
            continue
        label_raw = entry.get("question", "")
        key_raw = entry.get("key", "")

        if sanitize(label_raw) == norm_key or sanitize(key_raw) == norm_key:
            if verbose:
                st.write("🔍 Searching in `task_inputs`")
                st.write(f"✅ Match found — label: `{label_raw}`, key: `{key_raw}`")
            return entry.get("answer")

    # 2️⃣ Fallback to st.session_state["input_data"]
    input_data = st.session_state.get("input_data", {}).get(section, [])
    for entry in input_data:
        label_raw = entry.get("question", "")
        key_raw = entry.get("key", "")

        if sanitize(label_raw) == norm_key or sanitize(key_raw) == norm_key:
            if verbose:
                st.write("🔍 Searching in `input_data`")
                st.write(f"✅ Match found — label: `{label_raw}`, key: `{key_raw}`")
            return entry.get("answer")

    if verbose:
        st.warning(f"❌ No match found for key '{key}' in section '{section}'.")
    return None


def check_missing_utility_inputs(section="utilities"):
    """
    Checks for missing required fields for utility runbook generation.
    Returns a list of missing field labels.
    """
    missing = []

    if not get_answer(key="City", section=section):
        missing.append("City")
    if not get_answer(key="ZIP Code", section=section):
        missing.append("ZIP Code")
    if not get_answer(key="Internet Provider", section=section):
        missing.append("Internet Provider")

    return missing

def flatten_answers_to_dict(questions, answers):
    """
    Converts a flat list of questions and answers into a dictionary,
    filtering out empty answers.
    """
    return {
        question: answer
        for question, answer in zip(questions, answers)
        if str(answer).strip()
    }

def preview_input_data():
    """Display a summary of all collected input data."""
    if "input_data" not in st.session_state or not st.session_state["input_data"]:
        st.warning("No input data collected yet.")
        return
    st.markdown("### 📝 Input Summary")
    for section, entries in st.session_state["input_data"].items():
        st.markdown(f"#### 📁 {section}")
        for item in entries:
            st.markdown(f"- **{item['question']}**: {item['answer']} _(at {item['timestamp']})_")

def daterange(start, end):
    for n in range((end - start).days + 1):
        yield start + timedelta(n)

def get_filtered_dates(start_date, end_date, selected_days):
    if not selected_days:
        return []

    selected_days = [d.lower() for d in selected_days]
    all_dates = pd.date_range(start=start_date, end=end_date).to_list()

    return [
        d.date()  # ✅ Convert Timestamp to datetime.date
        for d in all_dates
        if d.strftime("%A").lower() in selected_days
    ]

def select_runbook_date_range():
    """
    Displays a compact runbook date selector with presets, date inputs, and confirmation.
    Returns: choice, start_date, end_date, valid_dates
    """
    st.subheader("🗓️ Customize Runbook Dates")

    today = datetime.now().date()
    default_end = today + timedelta(days=7)

    # 🔄 Reset confirmation if inputs have changed
    check_for_date_change()

    with st.form(key="confirm_dates_form"):
        # Row 1: Range Type + Dates
        col0, col1, col2 = st.columns([1.1, 1, 1])
        with col0:
            choice = st.radio("⏱️ Range", ["Pick Dates", "General"], key="date_choice", horizontal=True)
        with col1:
            start_date = st.date_input("📅 Start", today, key="start_date_input")
        with col2:
            end_date = st.date_input("📆 End", default_end, key="end_date_input")

        # Row 2: Full-with Pills
        selected_days = st.pills(
            label="📆 Select Days",
            options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            selection_mode="multi",
            key="selected_days",
            default=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]  # Optional sensible default
        )

        # Row 3: Submit button
        submitted = st.form_submit_button("✅ Confirm Runbook Dates")

    # Handle submission
    if submitted:
        if start_date >= end_date:
            st.error("⚠️ Start date must be before end date.")
        elif (end_date - start_date).days > 31:
            st.error("⚠️ Selected period must be no longer than 1 month.")
        else:
            valid_dates = get_filtered_dates(start_date, end_date, selected_days)
            st.session_state.update({
                "runbook_dates_confirmed": True,
                "start_date": start_date,
                "end_date": end_date,
                "valid_dates": valid_dates,
                #"include_priority": show_priority,
            })
            st.success(f"📆 Dates confirmed! {len(valid_dates)} valid days selected.")

    # Final return
    return (
        st.session_state.get("date_choice"),
        st.session_state.get("start_date"),
        st.session_state.get("end_date"),
        st.session_state.get("valid_dates", [])
    )


def check_runbook_dates_confirmed() -> bool:
    """
    Returns True if the user confirmed date selection for the given section.
    """
    return st.session_state.get("runbook_dates_confirmed", False)

def check_for_date_change():
    """
    Auto-reset session_state.set("runbook_dates_confirmed") to False
    if start date, end date, or refinement selection changes after confirmation.
    """

    confirmed = st.session_state.get("runbook_dates_confirmed", False)
    if not confirmed:
        return  # Nothing to do

    stored_start = st.session_state.get("start_date")
    stored_end = st.session_state.get("end_date")
    stored_refinement = st.session_state.get("refinement")

    current_start = st.session_state.get("start_date_input")
    current_end = st.session_state.get("end_date_input")
    current_refinement = st.session_state.get("date_refinement_icon")

    # Only apply this logic if using "Pick Dates"
    if st.session_state.get("date_choice") == "Pick Dates":
        if current_start and stored_start and current_start != stored_start:
            st.session_state["runbook_dates_confirmed"] = False
            st.warning("🔁 Start date changed — reconfirm needed.")
        elif current_end and stored_end and current_end != stored_end:
            st.session_state["runbook_dates_confirmed"] = False
            st.warning("🔁 End date changed — reconfirm needed.")
        elif current_refinement and stored_refinement:
            refinement_map = {
                "📆 All Days": "All Days",
                "🏢 Weekdays": "Weekdays Only",
                "🎉 Weekends": "Weekend Only"
            }
            current_clean = refinement_map.get(current_refinement)
            if current_clean != stored_refinement:
                st.session_state["runbook_dates_confirmed"] = False
                st.warning("🔁 Filter changed — reconfirm needed.")

def preview_interaction_log():
    """Display a log of all user interactions."""
    if "interaction_log" not in st.session_state or not st.session_state["interaction_log"]:
        st.info("No interactions logged yet.")
        return
    st.markdown("### 🧾 Interaction History")
    for log in st.session_state["interaction_log"]:
        st.markdown(f"- [{log['timestamp']}] {log['action'].capitalize()} — **{log['question']}** → `{log['answer']}` _(Section: {log['section']})_")

def custom_serializer(obj):
    # Handle datetime and date objects
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()  # ✅ Convert to ISO string like "2025-05-24"
    
    # If it's some other unsupported type, raise an error
    raise TypeError(f"Type {type(obj)} not serializable")

def autosave_input_data():
    """Automatically save data to session file (local or cloud placeholder)."""
    if "input_data" in st.session_state:
        st.session_state["autosaved_json"] = json.dumps(
        st.session_state["input_data"],
        indent=2,
        default=custom_serializer  # ✅ use your custom handler
        )
        # Placeholder: save_to_cloud(st.session_state["autosaved_json"])

def export_input_data_as_json(file_name="input_data.json"):
    """Export the collected input data as JSON."""
    if "input_data" in st.session_state:
        json_data = json.dumps(
            st.session_state["input_data"], 
            indent=2, 
            default=custom_serializer # ✅ Handles date/datetime
            )
        st.download_button(
            label="📥 Download as JSON",
            data=json_data,
            file_name=file_name,
            mime="application/json"
        )

def is_iso_date(string: str) -> bool:
    """Check if a string matches an ISO 8601 date format (YYYY-MM-DD)."""
    try:
        datetime.strptime(string, "%Y-%m-%d")
        return True
    except ValueError:
        return False

def restore_dates(obj):
    """Recursively convert ISO date strings into datetime.date objects."""
    if isinstance(obj, dict):
        return {k: restore_dates(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [restore_dates(item) for item in obj]
    elif isinstance(obj, str) and is_iso_date(obj):
        return datetime.strptime(obj, "%Y-%m-%d").date()
    return obj

def import_input_data_from_json(json_str: str):
    """
    Load JSON string, convert ISO-formatted dates back into date objects,
    and save into session state.
    """
    try:
        parsed = json.loads(json_str)
        restored = restore_dates(parsed)
        st.session_state["input_data"] = restored
        st.success("✅ Input data successfully imported and restored.")
    except Exception as e:
        st.error(f"❌ Failed to import input data: {e}")

# Streamlit code usage to re-import JSON data and automatically convert ISO date strings back into datetime.date objects.
#uploaded_file = st.file_uploader("Upload previously exported input JSON", type=["json"])
#if uploaded_file:
#    json_str = uploaded_file.read().decode("utf-8")
#    if st.button("📤 Import JSON Data"):
#       import_input_data_from_json(json_str)

def convert_to_csv(data_list):
    if not data_list:
        return ""

    # Collect all possible keys
    fieldnames = sorted({key for row in data_list for key in row})

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for row in data_list:
        row_copy = row.copy()
        for k, v in row_copy.items():
            if isinstance(v, dict):
                row_copy[k] = json.dumps(v)
        writer.writerow(row_copy)

    return output.getvalue()

def export_input_data_as_csv(file_name="input_data.csv"):
    """Wraps convert_to_csv() to export all input_data with section info."""
    full_list = []
    for section, entries in st.session_state.get("input_data", {}).items():
        for item in entries:
            item_copy = item.copy()
            item_copy["section"] = section
            full_list.append(item_copy)

    csv_data = convert_to_csv(full_list)

    st.download_button(
        label="📄 Download Your Data",
        data=csv_data,
        file_name=file_name,
        mime="text/csv"
    )

def render_lock_toggle(section: str, session_key: Optional[str] = None, label: Optional[str] = None):
    """
    Renders a simple toggle switch to lock/unlock inputs and updates session state.

    Args:
        section (str): The logical section name (e.g., "mail", "trash_handling").
        session_key (Optional[str]): Optional override for session state key.
        label (Optional[str]): Optional display label for the section.
    """
    key = session_key or f"{section}_locked"
    display_label = label or section.replace("_", " ").title()

    if key not in st.session_state:
        st.session_state[key] = False  # default unlocked

    is_locked = st.toggle(f"🔒 Lock {display_label}", value=st.session_state[key])
    st.session_state[key] = is_locked

    if is_locked:
        st.success(f"✅ {display_label} is saved and locked. Unlock to edit.")
    else:
        st.info(f"📝 You can now edit your {display_label.lower()}. Lock to save when finished.")

def interaction_dashboard():
    if "interaction_log" not in st.session_state or not st.session_state["interaction_log"]:
        st.info("No interaction data to visualize.")
        return

    df = pd.DataFrame(st.session_state["interaction_log"])
    st.markdown("### 📊 Interaction Dashboard")

    section_filter = st.selectbox("Filter by section", options=["All"] + sorted(df["section"].unique().tolist()))
    if section_filter != "All":
        df = df[df["section"] == section_filter]

    st.dataframe(df)

    st.markdown("#### 🔄 Inputs per Section")
    fig = px.histogram(df, x="section", color="action", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### ⏱ Interaction Timeline")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    timeline = px.scatter(df, x="timestamp", y="section", color="user_id", hover_data=["question", "answer"])
    st.plotly_chart(timeline, use_container_width=True)

def log_provider_result(label, value, section="Utility Providers"):
    """
    Logs a single provider result into input_data, with basic validation.
    - Skips empty, 'not found', or 'n/a' values
    - Raises an error for invalid labels
    """
    if not label or not isinstance(label, str):
        raise ValueError("Provider label must be a non-empty string.")

    if not isinstance(value, str) or value.strip().lower() in ["", "not found", "n/a"]:
        return  # Skip logging invalid or placeholder values

    if "input_data" not in st.session_state:
        st.session_state["input_data"] = {}
    if section not in st.session_state["input_data"]:
        st.session_state["input_data"][section] = []

    st.session_state["input_data"][section].append({
        "question": f"{label} Provider",
        "answer": value,
        "timestamp": datetime.now().isoformat()
    })

    if "interaction_log" not in st.session_state:
        st.session_state["interaction_log"] = []
    session_id = st.session_state.get("session_id", "anonymous")
    user_id = st.session_state.get("user_id", "anonymous")

    st.session_state["interaction_log"].append({
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "session_id": session_id,
        "action": "autolog",
        "question": f"{label} Provider",
        "answer": value,
        "section": section
    })

def extract_and_log_providers(content: str, section: str) -> dict:
    """
    Parses LLM output and logs structured provider metadata.

    Returns:
        dict: Structured provider info with name, description, contact, and emergency info.
    """

    def extract_block(keyword: str) -> str:
        pattern = rf"## .*{keyword}.*?(?=## |\Z)"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        return match.group(0).strip(" -") if match else ""

    def parse_block(block: str) -> dict:
        name_match = re.search(r"## [^\–\-]+[\–\-]\s*(.*)", block)
        description = re.search(r"\*\*Description:\*\* (.*)", block)
        phone = re.search(r"\*\*Phone:\*\* (.*)", block)
        website = re.search(r"\*\*Website:\*\* (.*)", block)
        email = re.search(r"\*\*Email:\*\* (.*)", block)
        address = re.search(r"\*\*Address:\*\* (.*)", block)
        emergency = re.search(r"\*\*Emergency Steps:\*\*(.*?)(\n\n|^## |\Z)", block, re.DOTALL)

        return {
            "name": name_match.group(1).strip() if name_match else "",
            "description": description.group(1).strip() if description else "",
            "contact_phone": phone.group(1).strip() if phone else "",
            "contact_website": website.group(1).strip() if website else "",
            "contact_email": email.group(1).strip() if email else "",
            "contact_address": address.group(1).strip() if address else "",
            "emergency_steps": emergency.group(1).strip() if emergency else "",
        }

    utility_keywords = {
        "electricity": "Electricity",
        "natural_gas": "Natural Gas",
        "water": "Water",
        "internet": "Internet",
    }

    structured_results = {}

    for key, label in utility_keywords.items():
        block = extract_block(label)

        if st.session_state.get("enable_debug_mode"):
            st.markdown(f"### 🧪 Debug Block for `{label}`")
            st.code(block or "⚠️ No block found", language="markdown")

        parsed = parse_block(block)
        structured_results[key] = parsed

        name = parsed.get("name", "")
        if name:
            st.session_state[f"{key}_provider"] = name
            register_input_only(f"{label} Provider", name, section=section)

            # ✅ Register extra fields
            register_input_only(f"{name} Description", parsed.get("description", ""), section=section)
            register_input_only(f"{name} Contact Phone", parsed.get("contact_phone", ""), section=section)
            register_input_only(f"{name} Contact Website", parsed.get("contact_website", ""), section=section)
            register_input_only(f"{name} Contact Email", parsed.get("contact_email", ""), section=section)
            register_input_only(f"{name} Contact Address", parsed.get("contact_address", ""), section=section)
            register_input_only(f"{name} Emergency Steps", parsed.get("emergency_steps", ""), section=section)

    # ✅ Save full dict for re-use
    st.session_state["utility_providers"] = structured_results
    st.session_state["utility_provider_metadata"] = structured_results

    return structured_results

def generate_category_keywords_from_labels(input_data):
    """Auto-generate category keywords based on recurring words in labels."""
    keyword_freq = defaultdict(int)

    for section, entries in input_data.items():
        for entry in entries:
            label = entry.get("question", "")
            words = re.findall(r"\b\w{4,}\b", label.lower())  # Filter short/noisy words
            for word in words:
                keyword_freq[word] += 1

    # Filter common task-related words only (tunable)
    candidate_keywords = {k: v for k, v in keyword_freq.items() if v >= 2}  # show those used more than once
    sorted_keywords = sorted(candidate_keywords.items(), key=lambda x: -x[1])

    return sorted_keywords

def sanitize_label(label: str) -> str:
    return label.lower().strip().replace(" ", "_").replace("?", "").replace(":", "")

def section_has_valid_input(section: str, min_entries: int = 1) -> bool:
    """
    Checks whether a section has enough inputs across input_data and task_inputs.
    Returns True/False but defers logging to a separate debug function.
    """
    input_data = st.session_state.get("input_data", {}).get(section, [])
    count_regular = sum(1 for entry in input_data if entry.get("answer", "").strip())

    task_inputs = st.session_state.get("task_inputs", [])
    count_tasks = sum(
        1 for entry in task_inputs
        if entry.get("section") == section and entry.get("answer", "").strip()
    )

    total = count_regular + count_tasks
    return total >= min_entries

def register_provider_input(label: str, value: str, section: str):
    if not label or not isinstance(label, str):
        raise ValueError("Provider label must be a non-empty string.")

    if not isinstance(value, str) or value.strip().lower() in ["", "not found", "n/a"]:
        return

    # Register as structured input data
    input_data = st.session_state.setdefault("input_data", {})
    section_data = input_data.setdefault(section, [])

    # Remove duplicates first
    section_data = [entry for entry in section_data if entry["question"] != f"{label} Provider"]
    section_data.append({
        "question": f"{label} Provider",
        "answer": value.strip()
    })
    input_data[section] = section_data

    # Track in task_inputs
    task_row = {
        "question": f"{label} Provider",
        "answer": value.strip(),
        "category": section,
        "section": section,
        "area": "home",
        "task_type": "info",
        "is_freq": False
    }
    st.session_state.setdefault("task_inputs", []).append(task_row)

def update_or_log_task(
    question: str,
    answer: str,
    section: str,
    task_type: str = None,
    is_freq: bool = False,
    key: Optional[str] = None,
    area: str = "home",
    canonical_map: Optional[dict] = None,
    overwrite_label: Optional[str] = None,
):
    """
    Logs a task input in session state, replacing any existing task with the same key.

    Args:
        question (str): The input label shown to the user.
        answer (str): The user's response.
        section (str): The logical section (e.g. "utilities", "trash").
        task_type (str): Logical category of task (e.g. "utilities").
        is_freq (bool): Whether the task has a frequency.
        key (str): Optional key override. Defaults to sanitized label or canonical.
        area (str): High-level category ("home", "pets", etc.)
        canonical_map (dict): Optional mapping from question → canonical key.
        overwrite_label (str): Optional new label to display (while keeping canonical key).
    """
    if not answer or str(answer).strip().lower() in ["", "⚠️ not provided", "n/a"]:
        return

    final_question = overwrite_label or question

    # Canonical or default key
    if canonical_map:
        canonical_key = canonical_map.get(question.lower(), key or sanitize_label(question))
    else:
        canonical_key = key or sanitize_label(question)

    new_entry = {
        "question": final_question,
        "answer": answer.strip(),
        "key": canonical_key,
        "category": section,
        "section": section,
        "area": area,
        "task_type": task_type,
        "is_freq": is_freq,
        "timestamp": datetime.now().isoformat()
    }

    task_inputs = st.session_state.setdefault("task_inputs", [])
    task_inputs = [t for t in task_inputs if t.get("key") != canonical_key]
    task_inputs.append(new_entry)
    st.session_state["task_inputs"] = task_inputs



# Placeholder for future enhancement: Cloud saving logic
# def save_to_cloud(json_data):
#     """Save the JSON data to a cloud storage provider."""
#     pass
