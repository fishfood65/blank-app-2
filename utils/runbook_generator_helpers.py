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
from typing import List, Tuple, Optional


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
        return [day for day in weekday_to_int if day.lower() in text.lower()]

    def extract_dates(text):
        patterns = [
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?",
            r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?"
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


def add_table_from_schedule(doc: Document, schedule_df: pd.DataFrame):
    if schedule_df.empty:
        doc.add_paragraph("_No schedule data available._")
        return

    schedule_df["Date"] = pd.to_datetime(schedule_df["Date"], errors="coerce")
    schedule_df = schedule_df.sort_values(by=["Date", "Source", "Tag", "Task"])
    schedule_df["Day"] = schedule_df["Date"].dt.strftime("%A")
    schedule_df["DateStr"] = schedule_df["Date"].dt.strftime("%Y-%m-%d")

    grouped = schedule_df.groupby("Date")

    for date, group in grouped:
        day_str = date.strftime("%A, %Y-%m-%d")
        doc.add_heading(f"📅 {day_str}", level=2)
        table = doc.add_table(rows=1, cols=2)
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'Task'
        hdr_cells[1].text = 'Image'

        for _, row in group.iterrows():
            task = row["Task"]
            image_path = ""
            for label in ["Outdoor Bin Image", "Recycling Bin Image"]:
                if label.lower().replace(" image", "") in task.lower():
                    uploaded = st.session_state.trash_images.get(label)
                    if uploaded:
                        image_path = uploaded
                        break

            row_cells = table.add_row().cells
            row_cells[0].text = task
            if image_path:
                paragraph = row_cells[1].paragraphs[0]
                run = paragraph.add_run()
                run.add_picture(image_path, width=Inches(1.5))
            else:
                row_cells[1].text = ""

        doc.add_paragraph("")

def generate_docx_from_split_prompts(
    prompts: List[str],
    api_key: str,
    *,
    section_titles: Optional[List[str]] = None,
    model: str = "mistral-small-latest",
    doc_heading: str = "Runbook",
    temperature: float = 0.5,
    max_tokens: int = 2048,
) -> Tuple[io.BytesIO, str]:

    combined_output = []
    for i, prompt in enumerate(prompts):
        if not prompt.strip():
            continue
        try:
            with st.spinner(f"📦 Processing Prompt {i + 1}..."):
                client = Mistral(api_key=api_key)
                completion = client.chat.complete(
                    model=model,
                    messages=[SystemMessage(content=prompt)],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                response_text = completion.choices[0].message.content
                title = section_titles[i].strip() if section_titles and i < len(section_titles) else ""
                section_label = f"### {title}" if title else ""
                output = f"{section_label}\n\n{response_text.strip()}" if section_label else response_text.strip()
                combined_output.append(output)
        except Exception as e:
            st.error(f"❌ Error processing prompt {i+1}: {e}")
            continue

    full_text = "\n\n".join(combined_output)
    doc = Document()
    doc.add_heading(doc_heading, 0)

    lines = full_text.splitlines()
    schedule_inserted = False
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line == "<<INSERT_SCHEDULE_TABLE>>":
            if not schedule_inserted:
                schedule_df = st.session_state.get("home_schedule_df", pd.DataFrame())
                add_table_from_schedule(doc, schedule_df)
                schedule_inserted = True
            continue

        # Headings
        if line.startswith("##### "):
            doc.add_heading(line[6:].strip(), level=4)
        elif line.startswith("#### "):
            doc.add_heading(line[5:].strip(), level=3)
        elif line.startswith("### "):
            doc.add_heading(line[4:].strip(), level=2)
        elif line.startswith("## "):
            doc.add_heading(line[3:].strip(), level=1)
        elif line.startswith("# "):
            doc.add_heading(line[2:].strip(), level=0)

        # Bullets
        elif line.startswith("- ") or line.startswith("* "):
            doc.add_paragraph(line[2:].strip(), style="List Bullet")

        # Numbered List
        elif re.match(r"^\d+\. ", line):
            doc.add_paragraph(re.sub(r"^\d+\. ", "", line), style="List Number")

        # Paragraph with inline markdown
        else:
            para = doc.add_paragraph()
            cursor = 0
            for match in re.finditer(r"(\*\*.*?\*\*)", line):
                start, end = match.span()
                if start > cursor:
                    para.add_run(line[cursor:start])
                bold_text = match.group(1)[2:-2]
                para.add_run(bold_text).bold = True
                cursor = end
            if cursor < len(line):
                para.add_run(line[cursor:])
            para.style.font.size = Pt(11)

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)

    return buffer, full_text

def preview_runbook_output(runbook_text: str, label: str = "📖 Preview Runbook"):
    """
    Shows an expandable markdown preview of the runbook text when a button is clicked.

    Args:
        runbook_text (str): The raw LLM-generated markdown-style text.
        label (str): Button label to trigger the preview.
    """
    if not runbook_text:
        st.warning("⚠️ No runbook content available to preview.")
        return

    if st.button(label):
        with st.expander("🧠 AI-Generated Runbook Preview", expanded=True):
            st.markdown(runbook_text)
