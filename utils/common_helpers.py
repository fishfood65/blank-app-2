#shared logic across apps
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import streamlit as st
import re
from collections import defaultdict
from config.sections import SECTION_METADATA
import calendar

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
    print("💡 get_schedule_utils() CALLED")

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

    def extract_ordinal_patterns(text):
        return re.findall(r"\b(first|second|third|fourth|last)\b", text.lower())

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
        results = []

        # Textual: "every July 4"
        month_day_pattern = re.findall(
            r"(?:every|each)?\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2})",
            text, re.IGNORECASE)
        for month, day in month_day_pattern:
            try:
                parsed_date = pd.to_datetime(f"{month} {day}", format="%b %d")
                results.append((parsed_date.month, parsed_date.day))
            except:
                pass

        # Numeric: "7/4"
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
        text = text.lower()

        # ⏱️ 1-time date detection
        for ds in extract_dates(text):
            parsed_date = normalize_date(ds)
            if parsed_date and parsed_date in valid_dates:
                return emoji_tags["[One-Time]"]

        # ↔️ Weekly interval
        interval = extract_week_interval(text)
        if interval:
            return emoji_tags.get("[X-Weeks]", "") + f" (Every {interval} Weeks)"

        # 📅 Weekly mention
        if extract_weekday_mentions(text):
            return emoji_tags["[Weekly]"]

        # 📆 Monthly mention
        if "monthly" in text or "every month" in text:
            return emoji_tags["[Monthly]"]

        # 🔢 Quarterly
        if "quarterly" in text or "every 3 months" in text:
            return emoji_tags["[Quarterly]"]

        # 🌐 Bi-Annually
        if "twice a year" in text or "every 6 months" in text or "bi-annually" in text:
            return emoji_tags["[Bi-Annually]"]

        return emoji_tags["[Daily]"]

    def resolve_nth_weekday(year, month, weekday_name, n):
        """
        Returns the date of the nth occurrence of a weekday in a given month/year.
        Example: 2nd Tuesday of July 2025 → datetime.date(2025, 7, 8)
        """
        weekday_num = weekday_to_int.get(weekday_name)
        if weekday_num is None:
            return None

        count = 0
        for day in range(1, calendar.monthrange(year, month)[1] + 1):
            date_obj = datetime(year, month, day)
            if date_obj.weekday() == weekday_num:
                count += 1
                if count == n:
                    return date_obj.date()
        return None

    return {
        "emoji_tags": emoji_tags,
        "weekday_to_int": weekday_to_int,
        "extract_week_interval": extract_week_interval,
        "extract_weekday_mentions": extract_weekday_mentions,
        "extract_ordinal_patterns": extract_ordinal_patterns,
        "extract_dates": extract_dates,
        "extract_annual_dates": extract_annual_dates,
        "normalize_date": normalize_date,
        "determine_frequency_tag": determine_frequency_tag,
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
    group_keys: Optional[Dict[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Merges all *_schedule_df DataFrames in session_state.
    Supports task_id-based deduplication and enrichment utilities.
    """
    import hashlib

    if exclude_keys is None:
        exclude_keys = []
    if group_keys is None:
        group_keys = {
            "trash_schedule_df": ["indoor_trash_schedule_df", "outdoor_trash_schedule_df"]
        }

    # Set deduplication fields
    if dedup_fields is None:
        dedup_fields = ["task_id", "Date", "clean_task", "task_type"]

    all_keys = [
        k for k in st.session_state
        if k.endswith("_schedule_df") and isinstance(st.session_state[k], pd.DataFrame)
    ]

    seen_keys = set()
    keys_to_merge = []

    # 🧩 Merge grouped schedule DataFrames
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

    # 📦 Add remaining ungrouped schedule DataFrames
    for key in all_keys:
        if key not in seen_keys and key != output_key and key not in exclude_keys:
            keys_to_merge.append(key)

    merged_frames = []
    for key in keys_to_merge:
        df = st.session_state.get(key)
        if df is not None and not df.empty:
            df_copy = df.copy()

            # ✅ Add SourceKey
            if annotate_source:
                df_copy["SourceKey"] = key

            # 🧠 Ensure task_id is present
            if "task_id" not in df_copy.columns or df_copy["task_id"].isnull().any():
                df_copy["task_id"] = df_copy.apply(lambda row: hashlib.sha1(
                    f"{row.get('question', '')}|{row.get('answer', '')}|{row.get('section', '')}".encode("utf-8")
                ).hexdigest()[:10], axis=1)

            # 📆 Filter by valid dates
            if valid_dates and "Date" in df_copy.columns:
                df_copy["Date"] = pd.to_datetime(df_copy["Date"], errors="coerce")
                df_copy = df_copy[df_copy["Date"].dt.date.isin(valid_dates)]

            # 🧠 Enrich using schedule utils
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
        st.warning("⚠️ No schedule DataFrames found to merge.")
        return pd.DataFrame()

    combined_df = pd.concat(merged_frames, ignore_index=True)

    # 🧼 Deduplicate
    if deduplicate and dedup_fields:
        before = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=dedup_fields)
        after = len(combined_df)
        if st.session_state.get("enable_debug_mode"):
            st.info(f"🧹 Removed {before - after} duplicate rows using: {dedup_fields}")

    # 🧮 Summary
    if show_summary and "SourceKey" in combined_df.columns and st.session_state.get("enable_debug_mode", False):
        st.subheader("📊 Schedule Source Summary")
        summary_df = (
            combined_df.groupby("SourceKey")
            .agg(
                TaskCount=("clean_task", "count"),
                UniqueDates=("Date", lambda x: pd.to_datetime(x, errors="coerce").dt.date.nunique()),
                UniqueTaskTypes=("task_type", lambda x: x.nunique()),
                UniqueTaskIDs=("task_id", lambda x: x.nunique())
            )
            .reset_index()
        )
        st.dataframe(summary_df)

        missing_ids = combined_df["task_id"].apply(lambda x: not str(x).strip()).sum()
        if missing_ids > 0:
            st.warning(f"⚠️ {missing_ids} scheduled tasks are missing a `task_id`.")

    # 💾 Save result
    if save_to_session:
        st.session_state[output_key] = combined_df

    return combined_df
