import streamlit as st
from mistralai import Mistral, UserMessage, SystemMessage
import csv
from io import BytesIO, StringIO
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
from typing import List, Tuple, Optional, Union
import tempfile
from .common_helpers import get_schedule_placeholder_mapping
from .prompt_block_utils import generate_all_prompt_blocks
from .task_schedule_utils_updated import export_schedule_to_markdown
import markdown

from docx import Document
import pandas as pd
from datetime import datetime

def add_table_from_schedule(doc: Document, schedule_df: pd.DataFrame, section: str):
    """
    Adds grouped schedule tables to the DOCX document by date.
    Image support has been removed.
    """
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

        table = doc.add_table(rows=1, cols=1)
        table.style = 'Light List'
        table.autofit = True

        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'Task'

        for _, row in group.iterrows():
            task = row["Task"]
            row_cells = table.add_row().cells
            row_cells[0].text = task

        doc.add_paragraph("")  # spacing between tables

def add_table_from_schedule_to_markdown(schedule_df: pd.DataFrame, section: str) -> str:
    """
    Convert a schedule DataFrame into grouped markdown tables by date,
    without images.
    """

    if schedule_df.empty:
        return "_No schedule data available._"

    # Format and sort
    schedule_df["Date"] = pd.to_datetime(schedule_df["Date"], errors="coerce")
    schedule_df = schedule_df.sort_values(by=["Date", "Source", "Tag", "Task"])
    schedule_df["Day"] = schedule_df["Date"].dt.strftime("%A")
    schedule_df["DateStr"] = schedule_df["Date"].dt.strftime("%Y-%m-%d")

    output_md = []
    grouped = schedule_df.groupby("Date")

    for date, group in grouped:
        day_str = date.strftime("%A, %Y-%m-%d")
        output_md.append(f"### 📅 {day_str}\n")
        output_md.append("| Task |\n|------|")

        for _, row in group.iterrows():
            task = row["Task"]
            output_md.append(f"| {task} |")

        output_md.append("")  # spacing

    return "\n".join(output_md)

def generate_docx_from_prompt_blocks(
    section: str,
    blocks: List[str],
    use_llm: bool = False,
    api_key: Optional[str] = None,
    doc_heading: str = "Runbook",
    model: str = "mistral-small-latest",
    debug: bool = False,
    insert_main_heading: bool = True, 
) -> Tuple[io.BytesIO, str]:
    from docx import Document
    from docx.shared import Pt

    doc = Document()
    if insert_main_heading:
        doc.add_heading(doc_heading, 0)
    markdown_output = []
    debug_warnings = []
    final_output = ""

    schedule_sources = get_schedule_placeholder_mapping()

    if use_llm:
        client = Mistral(api_key=api_key)
        for i, block in enumerate(blocks):
            if not block.strip():
                continue

            if debug:
                st.markdown(f"### 🧱 Prompt Block {i+1}")
                st.code(block, language="markdown")

            try:
                st.info("⚙️ Calling generate_docx_from_prompt_blocks...")
                completion = client.chat.complete(
                    model=model,
                    messages=[SystemMessage(content=block)],
                    max_tokens=2048,
                    temperature=0.5,
                )
                response_text = completion.choices[0].message.content.strip()
            except Exception as e:
                response_text = f"❌ LLM error: {e}"

            markdown_output.append(response_text)
        final_output = "\n\n".join(markdown_output)

    else:
        for block in blocks:
            if not block.strip():
                continue
            markdown_output.append(block)
        final_output = "\n\n".join(markdown_output)

    lines = final_output.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line in schedule_sources:
            df_key = schedule_sources[line]
            schedule_df = st.session_state.get(df_key)
            doc.add_paragraph("")
            if isinstance(schedule_df, pd.DataFrame) and not schedule_df.empty:
                add_table_from_schedule(doc, schedule_df, section=section)
            else:
                if debug:
                    section_hint = df_key.replace("_schedule_df", "").replace("_", " ").title()
                    debug_warnings.append(
                        f"⚠️ Skipping **{section_hint}** schedule placeholder `{line}` — no data found in [`st.session_state['{df_key}']`](#debug-{df_key})."
                    )
                continue
            doc.add_paragraph("")
            continue

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
        elif line.startswith("- ") or line.startswith("* "):
            doc.add_paragraph(line[2:].strip(), style="List Bullet")
        elif re.match(r"^\d+\.\s", line):
            doc.add_paragraph(re.sub(r"^\d+\.\s+", "", line), style="List Number")
        else:
            para = doc.add_paragraph()
            cursor = 0
            for match in re.finditer(r"(\*\*.*?\*\*)", line):
                start, end = match.span()
                if start > cursor:
                    para.add_run(line[cursor:start])
                para.add_run(match.group(1)[2:-2]).bold = True
                cursor = end
            if cursor < len(line):
                para.add_run(line[cursor:])
            para.style.font.size = Pt(11)

    doc.add_page_break()
    combined_schedule = st.session_state.get("combined_home_schedule_df")
    if isinstance(combined_schedule, pd.DataFrame) and not combined_schedule.empty:
        doc.add_heading("📆 Complete Schedule Summary", level=1)
        combined_schedule["Date"] = pd.to_datetime(combined_schedule["Date"], errors="coerce")
        combined_schedule = combined_schedule.sort_values(by=["Date", "Category", "Tag", "Task"])

        for date, group in combined_schedule.groupby("Date"):
            day_str = date.strftime("%A, %Y-%m-%d")
            doc.add_heading(f"📅 {day_str}", level=2)

            for category, cat_group in group.groupby("Category"):
                doc.add_heading(category, level=3)
                table = doc.add_table(rows=1, cols=1)
                table.style = 'Light List'
                table.autofit = True

                hdr_cells = table.rows[0].cells
                hdr_cells[0].text = 'Task'

                for _, row in cat_group.iterrows():
                    task = row["Task"]
                    row_cells = table.add_row().cells
                    row_cells[0].text = task

                doc.add_paragraph("")

        markdown_output.append("## 📆 Complete Schedule Summary")
        markdown_output.append(add_table_from_schedule_to_markdown(combined_schedule))
    else:
        if debug:
            debug_warnings.append(
                "⚠️ No schedule data found in [`combined_home_schedule_df`](#debug-combined_home_schedule_df). Skipping 📆 Complete Schedule Summary."
            )

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)

    if debug and debug_warnings:
        with st.expander("⚠️ Missing Schedules (Debug)", expanded=True):
            for msg in debug_warnings:
                st.markdown(f"- {msg}", unsafe_allow_html=True)

        for key in ["combined_home_schedule_df", "trash_schedule_df", "mail_schedule_df"]:
            if key in st.session_state:
                st.markdown(f"<a name='debug-{key}'></a>", unsafe_allow_html=True)
                st.markdown(f"### 🔎 Debug: `{key}`", unsafe_allow_html=True)
                st.dataframe(st.session_state[key])

    return buffer, "\n\n---\n\n".join(markdown_output).strip()

def preview_runbook_output(section: str, runbook_text: str, label: str = "📖 Preview Runbook"):
    """
    Shows an expandable markdown preview of the runbook text with optional section-specific schedule substitution.

    Args:
        section (str): Used to determine which schedule data (if any) to insert.
        runbook_text (str): The raw LLM-generated markdown-style text.
        label (str): Button label to trigger the preview.
    """
    if not runbook_text:
        st.warning("⚠️ No runbook content available to preview.")
        return

    # Optionally inject schedule table based on placeholder and section
    if "<<INSERT_SCHEDULE_TABLE>>" in runbook_text:
        df_key = "combined_home_schedule_df" if section == "home" else f"{section}_schedule_df"
        schedule_df = st.session_state.get(df_key)
        if isinstance(schedule_df, pd.DataFrame) and not schedule_df.empty:
            schedule_md = add_table_from_schedule_to_markdown(schedule_df)
            runbook_text = runbook_text.replace("<<INSERT_SCHEDULE_TABLE>>", schedule_md)

    with st.expander("🧠 AI-Generated Runbook Preview", expanded=True):
        st.markdown(runbook_text, unsafe_allow_html=True)

def maybe_render_download(section: str, filename: Optional[str] = None) -> bool:
    """
    Render preview and download buttons for a runbook generated for a specific section.

    Parameters:
    - section (str): The section key used for retrieving runbook content from session state.
    - filename (Optional[str]): Optional override for the DOCX filename.

    Returns:
    - bool: True if a download buffer was presented, False otherwise.
    """
    buffer_key = f"{section}_runbook_buffer"
    text_key = f"{section}_runbook_text"

    buffer = st.session_state.get(buffer_key)
    runbook_text = st.session_state.get(text_key)
    doc_heading = f"{section.replace('_', ' ').title()} Emergency Runbook"

    if not filename:
        filename = f"{section}_emergency_runbook.docx"

    st.subheader(f"📤 Export Options: {doc_heading}")

    col1, col2, col3, col4 = st.columns(4)
    # 🔍 Render preview and HTML export if runbook text is available
    with col1:
        if st.button("📖 Preview Runbook", key=f"Preview_Runbook_{section}"):
            preview_runbook_output(section, runbook_text)
    with col2:
        if runbook_text:
            html_output = f"<html><body><h1>{doc_heading}</h1>\n{markdown.markdown(runbook_text)}\n</body></html>"
            html_filename = filename.replace(".docx", ".html")
            st.download_button(
                label="🌐 Download as HTML",
                data=html_output.encode("utf-8"),
                file_name=html_filename,
                mime="text/html",
                key=f"HTML_Runbook_{section}"
            )
    with col3:
        if buffer:
            st.download_button(
                label="📄 Download DOCX",
                data=buffer,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key=f"DOCX_Runbook_{section}"
            )
        else:
            st.warning(f"⚠️ DOCX runbook buffer not found for `{section}`. Markdown/HTML export still available.")
    with col4:
        if st.button(f"♻️ Reset {section.title()} Runbook Cache", key=f"{section}_reset_preview"):
            for key in [buffer_key, text_key, f"{section}_runbook_ready"]:
                st.session_state.pop(key, None)
            st.success(f"🔄 Cleared session state for `{section}` runbook.")
            st.stop()

