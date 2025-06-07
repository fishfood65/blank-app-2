#shared logic across apps
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import streamlit as st
import re
from collections import defaultdict
from config.sections import SECTION_METADATA

def extract_all_trash_tasks_grouped(valid_dates, utils):
    utils = utils or get_schedule_utils()
    #st.write("🗑️ [DEBUG] extract_all_trash_tasks_grouped received:", valid_dates)
    #st.write("🗑️ [DEBUG] input_data:", st.session_state.get("input_data", {}))

    input_data = st.session_state.get("input_data", {})
    trash_entries = input_data.get("Trash Handling", []) or input_data.get("trash_handling")
    #st.write("🧪 trash_entries:", trash_entries) -- for Debug
    label_map = {entry["question"]: entry["answer"] for entry in trash_entries}
    #st.write("🧪 trash label_map keys:", list(label_map.keys())) -- for Debug

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

    for pickup_label in ["Garbage Pickup Day", "Recycling Pickup Day"]:
        answer = str(label_map.get(pickup_label, "")).strip()
        if answer:
            outdoor_rows.extend(schedule_task(pickup_label, answer))

    combined_df = pd.DataFrame(indoor_rows + outdoor_rows)
    if not combined_df.empty:
        return combined_df.sort_values(by=["Date", "Day", "Category", "Tag", "Task"]).reset_index(drop=True)
    else:
        return combined_df

def extract_quality_service_schedule():
    """
    Builds a schedule of recurring quality-oriented household services.
    Returns a list of dicts or a DataFrame for calendar rendering or export.
    """
    schedule = []
    data = st.session_state.get("input_data", {})
    root = data.get("Quality-Oriented Household Services", {})

    for service, fields in root.items():
        frequency = fields.get("Frequency")
        day = fields.get("Day of the Week")

        if frequency and day and day != "Not Specified":
            task = {
                "Task": f"{service} Service Visit",
                "Category": "Quality Service",
                "Source": "convenience_seeker",
                "Tag": fields.get("Company Name", ""),
                "Area": "home",
                "Frequency": frequency,
                "Day": day,
                "Notes": fields.get(f"Post-{service} Procedures", ""),
            }
            schedule.append(task)

    return pd.DataFrame(schedule) if schedule else pd.DataFrame(
        columns=["Task", "Category", "Source", "Tag", "Area", "Frequency", "Day", "Notes"]
    )

def extract_rent_own_service_schedule():
    """
    Builds a schedule of recurring quality-oriented household services.
    Returns a list of dicts or a DataFrame for calendar rendering or export.
    """
    schedule = []
    data = st.session_state.get("input_data", {})
    root = data.get("Rent or Own", {})

    for service, fields in root.items():
        frequency = fields.get("Frequency")
        day = fields.get("Day of the Week")

        if frequency and day and day != "Not Specified":
            task = {
                "Task": f"{service} Service Visit",
                "Category": "Quality Service",
                "Source": "convenience_seeker",
                "Tag": fields.get("Company Name", ""),
                "Area": "home",
                "Frequency": frequency,
                "Day": day,
                "Notes": fields.get(f"Post-{service} Procedures", ""),
            }
            schedule.append(task)

    return pd.DataFrame(schedule) if schedule else pd.DataFrame(
        columns=["Task", "Category", "Source", "Tag", "Area", "Frequency", "Day", "Notes"]
    )

def extract_grouped_mail_task(valid_dates):
    utils = get_schedule_utils()

    input_data = st.session_state.get("input_data", {})
    mail_entries = input_data.get("mail") or input_data.get("Mail & Packages", [])

    if not mail_entries:
        st.warning("📬 No mail entries found in session_state.")
        return []

    label_map = {entry["question"]: entry["answer"] for entry in mail_entries}

    mailbox_location = label_map.get("📍 Mailbox Location", "").strip()
    mailbox_key = label_map.get("🔑 Mailbox Key (Optional)", "").strip()
    pick_up_schedule = label_map.get("📆 Mail Pick-Up Schedule", "").strip()
    mail_handling = label_map.get("📥 What to Do with the Mail", "").strip()
    package_handling = label_map.get("📦 Packages", "").strip()

    if not (mailbox_location and pick_up_schedule and mail_handling):
        st.warning("📬 Missing required mail fields.")
        return []

    tag = utils["determine_frequency_tag"](pick_up_schedule, valid_dates)
    weekday_mentions = utils["extract_weekday_mentions"](pick_up_schedule)
    raw_dates = utils["extract_dates"](pick_up_schedule)

    one_time_tasks = defaultdict(list)
    repeating_tasks = defaultdict(list)

    # Core task lines (shared by all entries)
    base_lines = [
        f"📬 Mail should be picked up **{pick_up_schedule}**.",
        f"📍 Location: {mailbox_location}",
    ]
    if mailbox_key:
        base_lines.append(f"🔑 Key info: {mailbox_key}")
    base_lines.append(f"📥 Mail handling: {mail_handling}")
    if package_handling:
        base_lines.append(f"📦 Package instructions: {package_handling}")

    # One-time tasks
    for ds in raw_dates:
        parsed = utils["normalize_date"](ds)
        if parsed and parsed in valid_dates:
            one_time_tasks[parsed].extend(base_lines + ["📌 One-Time scheduled pickup."])

    for weekday in weekday_mentions:
            for d in valid_dates:
                if d.strftime("%A") == weekday:
                    repeating_tasks[d].append({
                        "Task": "Check mailbox",
                        "Category": "home",
                        "Area": "mail",
                        "Source": "user",
                        "Tag": tag,
                        "Date": d,
                        "Day": d.strftime("%A")
                    })

    # Merge into structured tasks
    structured_tasks = []

    def make_task(date, task_dicts, kind_tag):
    # Join the 'Task' strings from each dict
        task_lines = [d["Task"] for d in task_dicts if "Task" in d]
        return {
            "Task": "\n".join(task_lines).strip(),
            "Category": "home",
            "Area": "mail",
            "Source": "user",
            "Tag": kind_tag,
            "Date": date,
            "Day": date.strftime("%A")
        }

    for date, lines in sorted(one_time_tasks.items()):
        structured_tasks.append(make_task(date, lines, "📌 One-Time"))

    for date, lines in sorted(repeating_tasks.items()):
        structured_tasks.append(make_task(date, lines, tag or "🔁 Repeating"))

    #st.write("📬 [DEBUG] Structured Mail Tasks:", structured_tasks)
    return structured_tasks

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

def get_schedule_utils():
    """Shared date parsing, tag generation, and frequency utilities."""
    print("✨ get_schedule_utils() CALLED")
    import calendar
    emoji_tags = {
        "[Daily]": "🔁 Daily",
        "[Weekly]": "📅 Weekly",
        "[One-Time]": "🗓️ One-Time",
        "[X-Weeks]": "↔️ Every X Weeks",
        "[Monthly]": "📆 Monthly",
        "[Quarterly]": "🔢 Quarterly",
        "[Bi-Annually]": "🌐 Bi-Annually"
    }

    weekday_to_int = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
        "Friday": 4, "Saturday": 5, "Sunday": 6
    }

    def extract_week_interval(text):
        text = text.lower()
        print("🧪 Checking interval in text:", text)
        if "every week" in text and not re.search(r"every \\d+ week", text):
            print("✅ Detected 'every week' → returning 1")
            return 1
        match = re.search(r"every (\\d+) week", text)
        if match:
            print(f"✅ Detected 'every {match.group(1)} weeks'")
            return int(match.group(1))
        if "biweekly" in text:
            return 2
        return None

    def extract_weekday_mentions(text):
        return [day for day in weekday_to_int if day.lower() in text.lower()]

    def extract_dates(text):
        patterns = [
            r"\\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\\s+\\d{1,2}(?:,\\s*\\d{4})?",
            r"\\b\\d{1,2}/\\d{1,2}(?:/\\d{2,4})?"
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

    def extract_annual_dates(text):
        matches = []
        for match in re.findall(r"every (\w+ \d{1,2}|\d{1,2}/\d{1,2})", text.lower()):
            try:
                dt = pd.to_datetime(match, errors="coerce")
                if pd.notna(dt):
                    matches.append((dt.month, dt.day))
            except:
                pass
        return matches

    def extract_ordinal_patterns(text):
        ordinals = {
            "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
            "fourth": 4, "4th": 4, "fifth": 5, "5th": 5
        }
        patterns = []
        for ordinal, num in ordinals.items():
            for day in weekday_to_int:
                pattern = rf"{ordinal} {day.lower()}"
                if re.search(pattern, text.lower()):
                    patterns.append((num, day))
        return patterns

    def resolve_nth_weekday(year, month, weekday_name, n):
        weekday_num = weekday_to_int[weekday_name]
        count = 0
        for day in range(1, calendar.monthrange(year, month)[1] + 1):
            if datetime(year, month, day).weekday() == weekday_num:
                count += 1
                if count == n:
                    return datetime(year, month, day).date()
        return None

    def determine_frequency_tag(text, valid_dates):
        interval = extract_week_interval(text)
        weekday_mentions = extract_weekday_mentions(text)

        for ds in extract_dates(text):
            parsed_date = normalize_date(ds)
            if parsed_date and parsed_date in valid_dates:
                return emoji_tags["[One-Time]"]

        if "quarterly" in text.lower():
            return emoji_tags["[Quarterly]"]
        if "bi-annual" in text.lower() or "biannually" in text.lower():
            return emoji_tags["[Bi-Annually]"]
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
        "determine_frequency_tag": determine_frequency_tag,
        "extract_annual_dates": extract_annual_dates,
        "extract_ordinal_patterns": extract_ordinal_patterns,
        "resolve_nth_weekday": resolve_nth_weekday
    }


def get_schedule_placeholder_mapping() -> dict:
    """
    Scans session state for any keys ending in '_schedule_df' and builds a mapping
    of placeholder strings to their corresponding DataFrame keys.

    Example:
        If 'mail_schedule_df' exists in st.session_state,
        returns { '<<INSERT_MAIL_SCHEDULE_TABLE>>': 'mail_schedule_df' }

    Returns:
        Dict[str, str]: Mapping of placeholders to DataFrame keys.
    """
    mapping = {}
    for key in st.session_state:
        if key.endswith("_schedule_df") and isinstance(st.session_state[key], (pd.DataFrame, type(None))):
            placeholder_name = key.replace("_schedule_df", "").upper()
            placeholder = f"<<INSERT_{placeholder_name}_SCHEDULE_TABLE>>"
            mapping[placeholder] = key
    return mapping

def switch_section(section: str):
    last = st.session_state.get("active_section")
    if last and last != section:
        st.session_state[f"{last}_runbook_ready"] = False
    st.session_state["active_section"] = section

def debug_saved_schedule_dfs():
    """Prints all saved schedule DataFrames from session_state."""
    st.markdown("### 🧪 Saved Schedule DataFrames in Session")
    for key, val in st.session_state.items():
        if key.endswith("_schedule_df") and isinstance(val, pd.DataFrame):
            st.markdown(f"#### `{key}`")
            st.dataframe(val)

def merge_all_schedule_dfs(
    valid_dates: Optional[List[datetime.date]] = None,
    utils=None,
    output_key: str = "combined_home_schedule_df",
    exclude_keys: Optional[List[str]] = None,
    dedup_fields: Optional[List[str]] = None,
    deduplicate: bool = True,
    annotate_source: bool = True,
    save_to_session: bool = True,
    show_summary: bool = True,
    group_keys: Optional[Dict[str, List[str]]] = None  # New grouping param
) -> pd.DataFrame:
    if exclude_keys is None:
        exclude_keys = []
    if dedup_fields is None:
        dedup_fields = ["Date", "Day", "clean_task", "task_type"]
    if group_keys is None:
        group_keys = {
            "trash_schedule_df": ["indoor_trash_schedule_df", "outdoor_trash_schedule_df"]
        }

    all_keys = [
        k for k in st.session_state
        if k.endswith("_schedule_df") and isinstance(st.session_state[k], pd.DataFrame)
    ]

    seen_keys = set()
    keys_to_merge = []

    # Merge grouped schedule DFs
    for combined_key, parts in group_keys.items():
        dfs = []
        for key in parts:
            if key in seen_keys or key in exclude_keys:
                continue
            df = st.session_state.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = df.copy()
                df["SourceKey"] = key
                dfs.append(df)
                seen_keys.add(key)
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            st.session_state[combined_key] = combined_df
            keys_to_merge.append(combined_key)

    # Include any remaining schedule DFs not already grouped
    for key in all_keys:
        if key not in seen_keys and key != output_key and key not in exclude_keys:
            keys_to_merge.append(key)

    merged_frames = []
    for key in keys_to_merge:
        df = st.session_state.get(key)
        if df is not None and not df.empty:
            df_copy = df.copy()
            if annotate_source:
                df_copy["SourceKey"] = key

            if valid_dates is not None and "Date" in df_copy.columns:
                df_copy["Date"] = pd.to_datetime(df_copy["Date"], errors="coerce")
                df_copy = df_copy[df_copy["Date"].dt.date.isin(valid_dates)]

            if utils:
                for col in df_copy.columns:
                    if df_copy[col].dtype == object:
                        try:
                            df_copy[col] = df_copy[col].astype(str)
                        except Exception:
                            pass

                if "clean_task" in df_copy.columns:
                    df_copy["WeekdayMentions"] = df_copy["clean_task"].apply(
                        lambda x: utils["extract_weekday_mentions"](x) if isinstance(x, str) else []
                    )
                    df_copy["FrequencyTag"] = df_copy["clean_task"].apply(
                        lambda x: utils["determine_frequency_tag"](x, valid_dates or []) if isinstance(x, str) else ""
                    )
                    df_copy["DateMentions"] = df_copy["clean_task"].apply(
                        lambda x: utils["extract_dates"](x) if isinstance(x, str) else []
                    )
                    df_copy["WeekInterval"] = df_copy["clean_task"].apply(
                        lambda x: utils["extract_week_interval"](x) if isinstance(x, str) else None
                    )
                    df_copy["AnnualDates"] = df_copy["clean_task"].apply(
                        lambda x: utils["extract_annual_dates"](x) if isinstance(x, str) else []
                    )
                    df_copy["OrdinalPatterns"] = df_copy["clean_task"].apply(
                        lambda x: utils["extract_ordinal_patterns"](x) if isinstance(x, str) else []
                    )

            merged_frames.append(df_copy)

    if not merged_frames:
        st.warning("\u26a0\ufe0f No schedule DataFrames found to merge.")
        return pd.DataFrame()

    combined_df = pd.concat(merged_frames, ignore_index=True)

    if deduplicate and dedup_fields:
        before = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=dedup_fields)
        after = len(combined_df)
        if before > after:
            st.info(f"\U0001f9f9 Removed {before - after} duplicate task entries.")

    if show_summary and "SourceKey" in combined_df.columns and st.session_state.get("enable_debug_mode", False):
        st.subheader("\U0001f4ca Schedule Source Summary")
        summary_df = (
            combined_df.groupby("SourceKey")
            .agg(
                TaskCount=("clean_task", "count"),
                UniqueDates=("Date", lambda x: pd.to_datetime(x, errors="coerce").dt.date.nunique()),
                UniqueTaskTypes=("task_type", lambda x: x.nunique())
            )
            .reset_index()
        )
        st.dataframe(summary_df)

    if save_to_session:
        st.session_state[output_key] = combined_df

    return combined_df




