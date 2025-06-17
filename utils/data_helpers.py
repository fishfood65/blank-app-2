from typing import Any, Dict, List, Optional, Tuple
import spacy
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
from difflib import get_close_matches  # as fallback
from config.sections import SECTION_METADATA, get_section_meta
from utils.event_logger import log_event

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

def get_all_input_records(
    section: str = None,
    *,
    shared: bool = None,
    instance_id: str = None,
    area: str = None
) -> list[dict]:
    from config.sections import SECTION_METADATA

    all_records = []

    if area:
        sections_to_include = [
            s for s, meta in SECTION_METADATA.items()
            if meta.get("area") == area
        ]
    elif section:
        sections_to_include = [section]
    else:
        sections_to_include = list(SECTION_METADATA.keys())

    for sec in sections_to_include:
        task_inputs = st.session_state.get("task_inputs", [])
        input_data = st.session_state.get("input_data", {}).get(sec, [])

        for entry in task_inputs:
            if entry.get("section") != sec:
                continue
            if shared is not None and entry.get("shared") != shared:
                continue
            if instance_id is not None and entry.get("instance_id") != instance_id:
                continue
            all_records.append({**entry, "source": "task_inputs"})

        for entry in input_data:
            if shared is not None and entry.get("shared") != shared:
                continue
            if instance_id is not None and entry.get("instance_id") != instance_id:
                continue
            all_records.append({**entry, "source": "input_data"})

    return all_records

def validate_input_entry(entry: dict) -> bool:
    required_keys = ["question", "answer", "key", "section", "metadata"]
    for key in required_keys:
        if key not in entry:
            return False
    if "area" not in entry["metadata"]:
        return False
    return True

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
    # 🧼 Key generation
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
        
        # Safe handling for incompatible widgets
        widget_names = ["radio", "selectbox", "multiselect"]
        fn_name = getattr(input_fn, "__name__", "")
        if fn_name in widget_names and "value" in kwargs:
            kwargs.pop("value")
    
    # Only set placeholder for input widgets that support it
    widgets_with_placeholder = ["text_input", "text_area", "number_input"]
    fn_name = getattr(input_fn, "__name__", "")
    if fn_name in widgets_with_placeholder:
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

    # ✂️ Clean up input string
    value = safe_strip(value)

    # 🧠 Finalize metadata
    metadata = dict(metadata) if metadata else {}
    metadata.setdefault("section", section)
    metadata.setdefault("task_label", label)

    if st.session_state.get("enable_debug_mode") and metadata.get("category"):
        st.info(f"📂 Capturing input under category: {metadata['category']}")

    init_section(section)

    log_input_entry(
        label=label,
        value=value,
        section=section,
        key=unique_key,
        required=required,
        metadata=metadata,
        is_task=False  # Explicitly mark this as a non-task input
    )
        
    try:
        log_interaction("input", label, value, section)
    except Exception as e:
        if st.session_state.get("enable_debug_mode"):
            st.error(f"❌ log_interaction failed: {e}")
    autosave_input_data()

    return value

def log_input_entry(
    label: str,
    value: str,
    section: str,
    key: str = None,
    *,
    is_task: bool = False,
    is_freq: bool = False,
    task_type: str = None,
    area: str = None,
    instance_id: str = None,
    shared: bool = False,
    required: bool = False,
    metadata: dict = None
):
    """
    Central logging function for both task and non-task inputs.
    Adds standardized metadata, deduplicates by key, and stores in session state.
    """
    value = safe_strip(value)
    if value in [None, ""]:
        return

    key = key or f"{section}_{sanitize_label(label)}"
    metadata = dict(metadata) if metadata else {}

    # 🧠 Infer area from SECTION_METADATA if not provided
    if not area:
        area = SECTION_METADATA.get(section, {}).get("area", "general")

    # 🆕 Optional top-level category for easier filtering
    category = metadata.get("category")

    # ✨ Build structured input record
    record = {
        "question": label,
        "answer": value,
        "key": key,
        "section": section,
        "area": area,
        "task_type": task_type,
        "is_freq": is_freq,
        "timestamp": datetime.now().isoformat(),
        "shared": shared,
        "instance_id": instance_id,
        "required": required,
        "category": category,        # ✅ Added here for grouping
        "metadata": metadata,
    }

    # 📦 Store in session_state
    if is_task:
        task_inputs = st.session_state.setdefault("task_inputs", [])
        task_inputs = [t for t in task_inputs if t.get("key") != key]
        task_inputs.append(record)
        st.session_state["task_inputs"] = task_inputs
    else:
        section_entries = st.session_state.setdefault("input_data", {}).setdefault(section, [])
        section_entries = [e for e in section_entries if e.get("key") != key]
        section_entries.append(record)
        st.session_state["input_data"][section] = section_entries

    # 🧾 Optional: Log event
    log_event(
        event_type="task_logged" if is_task else "input_logged",
        data=record,
        tag="task" if is_task else "input"
    )

    return record

# Wrapper for task-based inputs
def register_task_input(
    label,
    input_fn,
    section: str,
    is_freq: bool = False,
    task_type: str = None,
    key: str = None,  # ✅ Explicit key
    area: str = None,  # Auto-filled from SECTION_METADATA if None
    instance_id: Optional[str] = None,
    shared: bool = False,
    required: bool = False,
    category: Optional[str] = None, # ✅ NEW: Optional third-level grouping
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
        instance_id (str, optional): For named entity inputs (e.g., pet name, profile ID)
        shared (bool): For inputs meant to apply across multiple entities
        area (str, optional): High-level area (home, pets, finances). Inferred if not given.
        category (str, optional): Subgrouping within section (e.g., "Feeding", "Vet Info")
        required (bool): Whether the input is required
        *args, **kwargs: Passed through to capture_input().
    """

    # ✅ Use or generate consistent key
    key = key or f"{section}_{sanitize_label(label)}"
    kwargs["key"] = key

    # 📦 Canonical metadata (auto-enriched from SECTION_METADATA)
    metadata = kwargs.pop("metadata", {})
    metadata.update({
        "is_task": True,
        "task_label": label,
        "is_freq": is_freq,
        "area": area,  # May be None; will be inferred by log_input_entry
        "section": section,
        "task_type": task_type,
        "shared": shared,
    })

    # ✅ Add optional category if provided
    if category:
        metadata["category"] = category

    # ✅ Capture user input widget
    value = capture_input(label, input_fn, section, *args, metadata=metadata, **kwargs)

    if value not in (None, ""):
        # 🧠 Save task in session state
        log_input_entry(
            label=label,
            value=value,
            section=section,
            key=key,
            is_task=True,
            is_freq=is_freq,
            task_type=task_type,
            area=area,
            instance_id=instance_id,
            shared=shared,
            required=required,
            metadata=metadata
        )

    return value

def register_input_only(
    label: str,
    value: str,
    section: str,
    key: str = None,
    metadata: dict = None,
    *,
    area: str = None,
    instance_id: str = None,
    shared: bool = False,
    required: bool = False,
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
    value_clean = safe_strip(value)
    
    if not allow_empty and (not value_clean or value_clean == ""):
        return

    return log_input_entry(
        label=label,
        value=value,
        section=section,
        key=key,
        is_task=False,
        area=area,
        instance_id=instance_id,
        shared=shared,
        required=required,
        metadata=metadata
    )

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

def _search_entries(
    entries: list[dict],
    norm_key: str,
    section: str = None,
    instance_id: str = None,
    area: str = None,
    shared: bool = None,
    required: bool = None,
    verbose: bool = False,
    category=None  # ✅ NEW
) -> str | None:
    """
    Search a list of structured entries for a matching key or label.

    Args:
        entries (list): List of input/task dict entries.
        norm_key (str): Sanitized search key.
        section (str): Optional section filter.
        instance_id (str): Optional per-entity filter (e.g. dog_1).
        area (str): Optional domain filter (e.g. home, pets).
        shared (bool): Optional filter for shared inputs.
        required (bool): Optional filter for required fields.
        verbose (bool): If True, prints matches and reasons.

    Returns:
        str | None: Matched answer if found.
    """
    for entry in entries:
        label_raw = entry.get("question", "")
        key_raw = entry.get("key", "")
        label_sanitized = sanitize(label_raw)
        key_sanitized = sanitize(key_raw)

        # ✅ Match on either sanitized question or key
        if norm_key not in [label_sanitized, key_sanitized]:
            continue

        if section and entry.get("section") != section:
            continue
        if instance_id and entry.get("instance_id") != instance_id:
            continue
        if area and entry.get("area") != area:
            continue
        if shared is not None and entry.get("shared") != shared:
            continue
        if required is not None and entry.get("required") != required:
            continue
        if category and entry.get("metadata", {}).get("category") != category:
            continue

        if verbose:
            st.write(f"✅ Match found in entry: {entry}")
        return entry.get("answer")

    if verbose:
        st.info(f"❌ No matching entry found for key='{norm_key}'")
    return None

def get_answer(
    key: str,
    section: str,
    instance_id: str = None,
    area: str = None,
    shared: bool = None,
    required: bool = None,
    fallback_to_global: bool = True,
    nested_parent: str = None,
    nested_child: str = None,
    verbose: bool = False,
    common_sections: set = None,
    category: str = None  # ✅ NEW: Optional category filter
) -> str| None:
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
        category (str): Optional filter to restrict results to a metadata category.

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
    task_inputs = st.session_state.get("task_inputs", [])
    answer = _search_entries(
        entries=task_inputs,
        norm_key=norm_key,
        section=section,
        instance_id=instance_id,
        area=area,
        shared=shared,
        required=required,
        verbose=verbose,
        category=category # ✅ NEW
    )
    if answer is not None:
        return answer

    # 2️⃣ Check input_data[section] safely
    raw_inputs = st.session_state.get("input_data", {}).get(section, [])
    if not isinstance(raw_inputs, list):
        if verbose:
            st.warning(f"⚠️ input_data['{section}'] expected a list but got {type(raw_inputs)} — skipping.")
        raw_inputs = []

    answer = _search_entries(
        entries=raw_inputs,
        norm_key=norm_key,
        section=section,
        instance_id=instance_id,
        area=area,
        shared=shared,
        required=required,
        verbose=verbose
    )

    if answer is not None:
        return answer

    # 3️⃣ Fallback to input_data["global"]
    if fallback_to_global:
        global_inputs = st.session_state.get("input_data", {}).get("global", [])
        if not isinstance(global_inputs, list):
            if verbose:
                st.warning("⚠️ input_data['global'] expected a list but got "
                           f"{type(global_inputs)} — skipping.")
            global_inputs = []

        answer = _search_entries(
            entries=global_inputs,
            norm_key=norm_key,
            section=section,
            instance_id=instance_id,
            area=area,
            shared=shared,
            required=required,
            verbose=verbose
        )
        if answer is not None:
            return answer

    if verbose:
        st.warning(f"❌ No match found for key '{key}' in section '{section}'.")
    return None

def apply_provider_overrides(
    parsed: dict,
    user_override: dict,
    fields_to_override: list = ["contact_phone", "contact_address", "contact_website", "description"]
) -> Tuple[dict, bool, bool]:
    """
    Applies user-confirmed overrides to a parsed provider block.
    
    Args:
        parsed (dict): The raw parsed LLM response (can be empty).
        user_override (dict): The user's confirmed update from disk/session.
        fields_to_override (list): Fields eligible for override (excluding 'name').

    Returns:
        Tuple[dict, bool, bool]: (updated parsed block, name_changed flag, needs_refresh flag)
    """
    if not user_override or not isinstance(user_override, dict):
        return parsed, False, False

    updated = parsed.copy()
    name_before = parsed.get("name", "").strip()
    name_after = user_override.get("name", "").strip()

    name_changed = name_before and name_after and (name_before != name_after)
    needs_refresh = False

    if not name_changed:
        # ✅ Use stored name + apply overrides
        if name_after:
            updated["name"] = name_after
        for field in fields_to_override:
            user_val = user_override.get(field, "").strip()
            if user_val:
                updated[field] = user_val
    else:
        # 🛠 Name was changed → assume rest of LLM block may be stale
        updated["name"] = name_after
        for field in fields_to_override:
            existing_val = parsed.get(field, "").strip()
            if not existing_val:
                needs_refresh = True  # Field is missing after name change
            updated.pop(field, None)  # Remove stale value

    return updated, name_changed, needs_refresh


def load_kit_items_config(config_path="config/recommended_kit_items.json"):
    with open(config_path) as f:
        return json.load(f)


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
    Lightweight wrapper for logging provider metadata using register_input_only().
    Skips empty/placeholder values, adds shared fallback visibility.
    """
    if not label or not isinstance(label, str):
        raise ValueError("Provider label must be a non-empty string.")
    
    value_clean = safe_strip(value)

    if not isinstance(value_clean, str) or value_clean().lower() in ["", "not found", "n/a"]:
        return  # Skip logging invalid or placeholder values

    register_input_only(
        label=f"{label} Provider",
        value=value,
        section=section,
        area="home",
        shared=True,
        required=False
    )

def clean_md_artifacts(text: str) -> str:
    """
    Cleans up common markdown formatting artifacts.
    Removes:
    - Asterisks `*`
    - Markdown links `[label](url)`
    - Extra whitespace
    """
    # Remove markdown-style links [label](url) → label
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Remove extra formatting characters
    text = re.sub(r"[()*_`]", "", text)
    return text.strip()


# ✅ Recommended unified parser
def parse_utility_block(block: str) -> dict:
    """
    Extracts structured fields from a markdown-formatted utility provider block,
    with markdown artifacts removed for all values.
    """
    def extract_and_clean(pattern: str, multiline: bool = False) -> str:
        flags = re.DOTALL if multiline else 0
        match = re.search(pattern, block, flags)
        return clean_md_artifacts(match.group(1).strip()) if match else ""

    return {
        "name": extract_and_clean(r"## [^\–\-]+[\–\-]\s*(.*)"),
        "description": extract_and_clean(r"\*\*Description:\*\* (.*)"),
        "contact_phone": extract_and_clean(r"\*\*Phone:\*\* (.*)"),
        "contact_website": extract_and_clean(r"\*\*Website:\*\* (.*)"),
        #"contact_email": extract_and_clean(r"\*\*Email:\*\* (.*)"),
        "contact_address": extract_and_clean(r"\*\*Address:\*\* (.*)"),
        "emergency_steps": extract_and_clean(
            r"\*\*Emergency Steps:\*\*\s*((?:.|\n)*?)(?=\n## |\Z)", multiline=True
        ) or "⚠️ Emergency steps not provided.",
        "non_emergency_tips": extract_and_clean(
            r"\*\*Non-Emergency Tips:\*\*\s*((?:.|\n)*?)(?=\n## |\Z)", multiline=True
        ) or ""
    }
def normalize_provider_fields(parsed: dict) -> dict:
    """
    Applies fallback replacement, deduplication, and formatting cleanup.
    """
    def normalize_val(val: str) -> str:
        if not val or val.strip().lower() in ["n/a", "not found", "⚠️ not available"]:
            return "⚠️ Not Available"
        return val.strip()

    for key in parsed:
        if isinstance(parsed[key], str):
            parsed[key] = normalize_val(parsed[key])

    return parsed

def extract_provider_blocks(content: str) -> dict:
    """
    Extracts raw markdown blocks for each utility type using more precise headings.
    """
    utility_keywords = {
        "electricity": "Electricity",
        "natural_gas": "Natural Gas",
        "water": "Water",
        "internet": "Internet",
    }

    blocks = {}
    for key, label in utility_keywords.items():
        # 🔧 More precise heading capture using non-greedy + stop-at-next-heading
        pattern = rf"## [^\n]*{re.escape(label)}[^\n]*\n(.*?)(?=\n## |\Z)"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            blocks[key] = f"## {label} – " + match.group(0).strip()
        else:
            blocks[key] = ""

    return blocks


def extract_and_log_providers(content: str, section: str) -> dict:
    """
    Parses LLM output and logs structured provider metadata into session state.
    Also registers each field for downstream usage and debugging.
    """
    blocks = extract_provider_blocks(content)
    structured_results = {}

    for key, block in blocks.items():
        label = key.replace("_", " ").title()

        if not block.strip():
            continue  # Skip if block is empty or not returned

        if st.session_state.get("enable_debug_mode"):
            st.markdown(f"### 🔎 Extracted Block for `{label}`")
            st.code(block if block.strip() else "⚠️ No block found", language="markdown")

        parsed = parse_utility_block(block)
        structured_results[key] = parsed

        if st.session_state.get("enable_debug_mode"):
            st.markdown(f"#### 🧪 Parsed Fields for `{label}`")
            st.json(parsed)

        name = parsed.get("name", "")
        if name:
            st.session_state[f"{key}_provider"] = name
            register_input_only(f"{label} Provider", name, section=section)

            # Use namespaced label to avoid overwrite collisions (e.g., PG&E for multiple)
            prefix = f"{label} ({name})"

            register_input_only(f"{prefix} Description", parsed.get("description", ""), section=section)
            register_input_only(f"{prefix} Contact Phone", parsed.get("contact_phone", ""), section=section)
            register_input_only(f"{prefix} Contact Website", parsed.get("contact_website", ""), section=section)
            #register_input_only(f"{prefix} Contact Email", parsed.get("contact_email", ""), section=section)
            register_input_only(f"{prefix} Contact Address", parsed.get("contact_address", ""), section=section)
            register_input_only(f"{prefix} Emergency Steps", parsed.get("emergency_steps", ""), section=section)

    # 🔁 Deduplicate emergency steps (if all are identical)
    all_steps = [p["emergency_steps"] for p in structured_results.values() if p["emergency_steps"]]
    if len(set(all_steps)) == 1 and all_steps:
        shared = all_steps[0]
        for provider in structured_results.values():
            provider["emergency_steps"] = f"{shared}\n\n_(Same for all utilities)_"

    # ✅ Save full structured result to session state
    st.session_state["utility_providers"] = structured_results
    st.session_state["utility_provider_metadata"] = structured_results

    # 🧪 Add this right before return
    if st.session_state.get("enable_debug_mode"):
        st.markdown("### 🧾 Provider Names Summary")
        names = {k: v.get("name", "—") for k, v in structured_results.items()}
        st.json(names)

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

def safe_strip(value):
    """Safely strip strings and leave non-strings untouched."""
    if isinstance(value, str):
        return value.strip()
    return value

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
    """
    Logs a provider result into both `input_data` and `task_inputs` with full metadata.
    Ensures no duplicates and consistent formatting for downstream functions like `get_answer()`.
    Automatically logs debug and analytics events.
    """
    if not label or not isinstance(label, str):
        raise ValueError("Provider label must be a non-empty string.")
    
    value_clean = safe_strip(value)

    # Skip placeholder or missing values
    
    if not isinstance(value_clean, str) or value_clean.strip().lower() in ["", "not found", "n/a", "⚠️ not available"]:
        return  # Skip placeholder or missing values

    question = f"{label} Provider"
    timestamp = datetime.now().isoformat()

    # ✅ Resolve area from section metadata
    area = SECTION_METADATA.get(section, {}).get("area", section)

    # ✅ 1. Update input_data
    input_data = st.session_state.setdefault("input_data", {})
    section_data = input_data.setdefault(section, [])

    # Remove existing question if it exists
    section_data = [entry for entry in section_data if entry.get("question") != question]
    section_data.append({
        "question": question,
        "answer": value,
        "timestamp": timestamp
    })
    input_data[section] = section_data

    # ✅ 2. Update task_inputs
    task_inputs = st.session_state.setdefault("task_inputs", [])
    task_inputs = [
        task for task in task_inputs
        if not (task.get("question") == question and task.get("section") == section)
    ]
    task_inputs.append({
        "question": question,
        "answer": value_clean,
        "key": sanitize_label(question),
        "section": section,
        "area": area,
        "category": section,
        "task_type": "info",
        "is_freq": False,
        "timestamp": timestamp
    })
    st.session_state["task_inputs"] = task_inputs

    # ✅ 3. Optional: update top-level key for direct lookup
    st.session_state[sanitize_label(question)] = value
    
    # ✅ 4. Debug display
    if st.session_state.get("enable_debug_mode"):
        st.info(f"✅ Registered provider input: **{label}** = `{value}`")

    # ✅ 5. Analytics/logging
    log_event(
        event_type="provider_input_registered",
        data={
            "question": question,
            "answer": value,
            "section": section,
            "area": area
        },
        tag="provider_update"
    )

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

# utils/provider_helpers.py

def get_provider_display_name(provider_data: dict, fallback_label: str = "Provider") -> str:
    """
    Safely return a display name for the provider using name, contact_name, or a fallback.
    This works across categories (utilities, medical, insurance, etc.).
    """
    if not provider_data or not isinstance(provider_data, dict):
        return fallback_label

    name = provider_data.get("name", "").strip()
    contact_name = provider_data.get("contact_name", "").strip()

    invalid = {"", "⚠️ not available", "not available", "n/a"}

    if name.lower() not in invalid:
        return name
    elif contact_name.lower() not in invalid:
        return contact_name
    return fallback_label

def load_kit_items_config(config_path="config/recommended_kit_items.json"):
    with open(config_path) as f:
        return json.load(f)

def normalize_item(text):
    """Normalize item string for fuzzy matching."""
    return re.sub(r'[^a-z0-9]', '', text.lower())

def fuzzy_match_item(user_item: str, valid_items: list, threshold: float = 0.82):
    """
    Attempts to match user-entered item to a known emergency kit item using spaCy similarity.
    Falls back to difflib.get_close_matches if no match is found.
    """

    nlp = spacy.load("en_core_web_sm")
    
    if not user_item or not valid_items:
        return None

    user_doc = nlp(user_item.lower())

    best_match = None
    best_score = 0.0

    for ref in valid_items:
        ref_doc = nlp(ref.lower())
        score = user_doc.similarity(ref_doc)
        if score > best_score:
            best_score = score
            best_match = ref

    if best_score >= threshold:
        return best_match

    # Fallback using string similarity
    fallback = get_close_matches(user_item, valid_items, n=1, cutoff=0.85)
    return fallback[0] if fallback else None

def make_safe_key(label: str) -> str:
    return re.sub(r'_+', '_', re.sub(r'[^a-z0-9]', '_', label.lower())).strip('_')

# Placeholder for future enhancement: Cloud saving logic
# def save_to_cloud(json_data):
#     """Save the JSON data to a cloud storage provider."""
#     pass
