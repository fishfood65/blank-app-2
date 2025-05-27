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
        match = re.search(r"every (\d+) week", text.lower())
        if match:
            return int(match.group(1))
        if "biweekly" in text.lower():
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
    """
    Splits prompts into individual LLM calls to manage token limits,
    stitches together the results, and returns a formatted DOCX and combined response.

    Args:
        prompts: A list of prompt strings to process.
        api_key: Your Mistral API key.
        section_titles: Optional titles for each prompt section (must match prompt count if provided).
        model, doc_heading, temperature, max_tokens: LLM configuration and output formatting.

    Returns:
        Tuple of (DOCX file as BytesIO, concatenated LLM response text).
    """

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

                # Determine title if provided
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

    for line in full_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("# "):
            doc.add_heading(line[2:].strip(), level=1)
        elif line.startswith("## "):
            doc.add_heading(line[3:].strip(), level=2)
        elif line.startswith("### "):
            doc.add_heading(line[4:].strip(), level=3)
        elif line.startswith("- ") or line.startswith("* "):
            doc.add_paragraph(line[2:].strip(), style="List Bullet")
        elif line[:2].isdigit() and line[2:4] == ". ":
            doc.add_paragraph(line[4:].strip(), style="List Number")
        else:
            para = doc.add_paragraph(line)
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
