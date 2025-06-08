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
from typing import Callable, List, Literal, Tuple, Optional, Union
import markdown as md
import tempfile
from .common_helpers import get_schedule_placeholder_mapping
from .prompt_block_utils import generate_all_prompt_blocks
from .task_schedule_utils_updated import export_schedule_to_markdown
import markdown
from config.constants import TASK_TYPE_EMOJI, SCHEDULE_HEADING_MAP, PRIORITY_ORDER

from docx import Document
import pandas as pd
from datetime import datetime

def display_user_friendly_schedule_table(
    df: pd.DataFrame,
    label_col: str = "clean_task",
    show_heading: bool = True,
    heading_text: str = "📆 Scheduled Tasks",
    show_legend: bool = True,
    enable_task_filter: bool = True
):
    """
    Displays a user-friendly task schedule table:
    - Sorts by date and priority
    - Adds emojis and task labels
    - Gracefully handles missing columns
    - Includes collapsible emoji legend and task type filter
    """

    if df is None or df.empty:
        st.warning("⚠️ No scheduled tasks to display.")
        return

    df = df.copy()

    # 🏷️ Add emoji and readable label
    if "task_type" in df.columns:
        df["emoji"] = df["task_type"].map(TASK_TYPE_EMOJI).fillna("❓")
        df["🏷️ Task Type"] = df["emoji"] + " " + df["task_type"]
    else:
        df["🏷️ Task Type"] = "❓ Unknown"

    # 🧮 Add priority
    if "task_type" in df.columns:
        priority_lookup = {task: i for i, task in enumerate(PRIORITY_ORDER, 1)}
        df["priority"] = df["task_type"].map(priority_lookup).fillna(len(PRIORITY_ORDER) + 1).astype(int)
    else:
        df["priority"] = len(PRIORITY_ORDER) + 1

    # 🧮 Sort by Date and priority
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values(by=["Date", "priority"])

    # 📋 Column label mapping
    display_cols = {}

    if "Date" in df.columns:
        display_cols["Date"] = "📅 Date"
    if "Day" in df.columns:
        display_cols["Day"] = "🗓️ Day"

    display_cols["🏷️ Task Type"] = "🏷️ Task Type"

    if label_col in df.columns:
        display_cols[label_col] = "✅ Task"
    else:
        df[label_col] = "⚠️ Missing"
        display_cols[label_col] = "✅ Task"

    user_friendly_df = df[list(display_cols.keys())].rename(columns=display_cols)

    # 🧾 Final display
    if show_heading:
        st.subheader(heading_text)

    st.dataframe(user_friendly_df, use_container_width=True)

    # 📖 Emoji legend in expander wrapped by columns
    col1, col2 = st.columns(2)
    with col1:
        if show_legend:
            with st.expander("🗂️ Show Emoji Legend", expanded=False):
                legend_lines = [f"{emoji} = {task}" for task, emoji in TASK_TYPE_EMOJI.items()]
                st.markdown("  \n".join(legend_lines))
    with col2:
        # ✅ Filter by task type if enabled
        if enable_task_filter and "task_type" in df.columns:
            with st.expander("🎛️ Filter by Task Type", expanded=False):
                unique_types = sorted(df["task_type"].dropna().unique())
                selected_types = st.multiselect("Show only these task types:", unique_types, default=unique_types)
                df = df[df["task_type"].isin(selected_types)]

def add_table_from_schedule(doc, schedule_df, section: str, include_priority: bool = True, include_heading: bool = False, heading_text: Optional[str] = None):
    from docx.shared import Inches
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    if schedule_df.empty:
        return

    schedule_df = schedule_df.copy()
    schedule_df["Date"] = pd.to_datetime(schedule_df["Date"], errors="coerce")
    schedule_df["Priority"] = schedule_df["task_type"].apply(
        lambda t: PRIORITY_ORDER.index(t) if t in PRIORITY_ORDER else len(PRIORITY_ORDER)
    )
    schedule_df["Task Type"] = schedule_df["task_type"].apply(
        lambda t: f"{TASK_TYPE_EMOJI.get(t, '')} {t}" if t else "❓"
    )

    schedule_df = schedule_df.sort_values(by=["Date", "Priority", "clean_task"])

    # 🔹 Add main schedule heading (level 2) once before any date blocks
    if include_heading and heading_text:
        doc.add_heading(heading_text, level=2)
        doc.add_paragraph("")  # space after heading

    # 🔹 Add each date block under heading level 3
    for date, group in schedule_df.groupby("Date"):
        day_str = date.strftime("%A, %Y-%m-%d")
        doc.add_heading(f"📅 {day_str}", level=3)

        table = doc.add_table(rows=1, cols=3 if include_priority else 2)
        table.style = "Table Grid"
        hdr_cells = table.rows[0].cells

        hdr_cells[0].text = "📝 Task"
        if include_priority:
            hdr_cells[1].text = "🔢 Priority"
            hdr_cells[2].text = "🔖 Type"
        else:
            hdr_cells[1].text = "🔖 Type"

        for _, row in group.iterrows():
            cells = table.add_row().cells
            cells[0].text = str(row.get("clean_task", ""))
            if include_priority:
                cells[1].text = str(row.get("Priority", ""))
                cells[2].text = str(row.get("Task Type", ""))
            else:
                cells[1].text = str(row.get("Task Type", ""))

        doc.add_paragraph("")

def add_table_from_schedule_to_markdown(schedule_df: pd.DataFrame, section: str, include_priority: bool = True, include_heading: bool = False, heading_text: Optional[str] = None) -> str:
    if schedule_df.empty:
        return "_No schedule data available._"

    schedule_df = schedule_df.copy()
    schedule_df["Date"] = pd.to_datetime(schedule_df["Date"], errors="coerce")
    schedule_df["Day"] = schedule_df["Date"].dt.strftime("%A")
    schedule_df["DateStr"] = schedule_df["Date"].dt.strftime("%Y-%m-%d")
    schedule_df["Priority"] = schedule_df["task_type"].apply(
        lambda t: PRIORITY_ORDER.index(t) if t in PRIORITY_ORDER else len(PRIORITY_ORDER)
    )
    schedule_df["Task Type"] = schedule_df["task_type"].apply(
        lambda t: f"{TASK_TYPE_EMOJI.get(t, '')} {t}" if t else "❓"
    )
    schedule_df = schedule_df.sort_values(by=["Date", "Priority", "clean_task"])

    output_md = []

    if include_heading and heading_text:
        output_md.append(f"## {heading_text}\n")

    for date, group in schedule_df.groupby("Date"):
        day_str = date.strftime("%A, %Y-%m-%d")
        output_md.append(f"### 📅 {day_str}\n")

        header_cols = ["📜 Task", "🔖 Type"]
        if include_priority:
            header_cols.insert(1, "🔢 Priority")
        output_md.append("| " + " | ".join(header_cols) + " |")
        output_md.append("|" + " --- |" * len(header_cols))

        for _, row in group.iterrows():
            values = [row.get("clean_task", ""), row.get("Task Type", "")]
            if include_priority:
                values.insert(1, str(row.get("Priority", "")))
            output_md.append("| " + " | ".join(values) + " |")

        output_md.append("")

    return "\n".join(output_md)

def add_table_from_schedule_to_html(schedule_df: pd.DataFrame, section: str, include_priority: bool = True, include_heading: bool = False, heading_text: Optional[str] = None) -> str:
    if schedule_df.empty:
        return "<p><em>No schedule data available.</em></p>"

    schedule_df = schedule_df.copy()
    schedule_df["Date"] = pd.to_datetime(schedule_df["Date"], errors="coerce")
    schedule_df["Day"] = schedule_df["Date"].dt.strftime("%A")
    schedule_df["DateStr"] = schedule_df["Date"].dt.strftime("%Y-%m-%d")
    schedule_df["Priority"] = schedule_df["task_type"].apply(
        lambda t: PRIORITY_ORDER.index(t) if t in PRIORITY_ORDER else len(PRIORITY_ORDER)
    )
    schedule_df["Task Type"] = schedule_df["task_type"].apply(
        lambda t: f"{TASK_TYPE_EMOJI.get(t, '')} {t}" if t else "❓"
    )
    schedule_df = schedule_df.sort_values(by=["Date", "Priority", "clean_task"])

    html_output = []

    if include_heading and heading_text:
        html_output.append(f"<h2>{heading_text}</h2>")

    for date, group in schedule_df.groupby("Date"):
        day_str = date.strftime("%A, %Y-%m-%d")
        html_output.append(f"<h3>📅 {day_str}</h3>")
        html_output.append("<table border='1' cellspacing='0' cellpadding='4' style='border-collapse: collapse;'>")

        columns = ["📜 Task", "🔖 Type"]
        if include_priority:
            columns.insert(1, "🔢 Priority")
        html_output.append("<thead><tr>" + "".join(f"<th>{col}</th>" for col in columns) + "</tr></thead>")

        html_output.append("<tbody>")
        for _, row in group.iterrows():
            cells = [row.get("clean_task", ""), row.get("Task Type", "")]
            if include_priority:
                cells.insert(1, str(row.get("Priority", "")))
            html_output.append("<tr>" + "".join(f"<td>{str(cell)}</td>" for cell in cells) + "</tr>")
        html_output.append("</tbody></table><br>")

    return "\n".join(html_output)

def add_table_from_schedule_to_html(schedule_df: pd.DataFrame, section: str, include_priority: bool = True, include_heading: bool = False, heading_text: Optional[str] = None) -> str:
    if schedule_df.empty:
        return "<p><em>No schedule data available.</em></p>"

    schedule_df = schedule_df.copy()
    schedule_df["Date"] = pd.to_datetime(schedule_df["Date"], errors="coerce")
    schedule_df["Day"] = schedule_df["Date"].dt.strftime("%A")
    schedule_df["DateStr"] = schedule_df["Date"].dt.strftime("%Y-%m-%d")
    schedule_df["Priority"] = schedule_df["task_type"].apply(
        lambda t: PRIORITY_ORDER.index(t) if t in PRIORITY_ORDER else len(PRIORITY_ORDER)
    )
    schedule_df["Task Type"] = schedule_df["task_type"].apply(
        lambda t: f"{TASK_TYPE_EMOJI.get(t, '')} {t}" if t else "❓"
    )
    schedule_df = schedule_df.sort_values(by=["Date", "Priority", "clean_task"])

    html_output = []

    if include_heading and heading_text:
        html_output.append(f"<h2>{heading_text}</h2>")

    for date, group in schedule_df.groupby("Date"):
        day_str = date.strftime("%A, %Y-%m-%d")
        html_output.append(f"<h3>📅 {day_str}</h3>")
        html_output.append("<table border='1' cellspacing='0' cellpadding='4' style='border-collapse: collapse;'>")

        columns = ["📜 Task", "🔖 Type"]
        if include_priority:
            columns.insert(1, "🔢 Priority")
        html_output.append("<thead><tr>" + "".join(f"<th>{col}</th>" for col in columns) + "</tr></thead>")

        html_output.append("<tbody>")
        for _, row in group.iterrows():
            cells = [row.get("clean_task", ""), row.get("Task Type", "")]
            if include_priority:
                cells.insert(1, str(row.get("Priority", "")))
            html_output.append("<tr>" + "".join(f"<td>{str(cell)}</td>" for cell in cells) + "</tr>")
        html_output.append("</tbody></table><br>")

    return "\n".join(html_output)

def generate_llm_responses(blocks: List[str], api_key: str, model: str, debug: bool) -> List[str]:
    client = Mistral(api_key=api_key)
    markdown_output = []

    for i, block in enumerate(blocks):
        if not block.strip():
            continue

        if debug:
            st.markdown(f"### 🧱 Prompt Block {i+1}")
            st.code(block, language="markdown")

        try:
            st.info("⚙️ Calling LLM...")
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

    return 


def generate_docx_from_prompt_blocks(
    blocks: list[str],
    section: str,
    doc_heading: str,
    schedule_sources: dict[str, str],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    use_llm: bool = False,
    debug: bool = False,
    include_priority: bool = True,
    insert_main_heading: bool = False,
    include_heading: bool = False,
) -> tuple[io.BytesIO, str, str]:
    import io
    import markdown
    import pandas as pd
    import re
    from docx import Document
    from docx.shared import Pt
    from config.constants import SCHEDULE_HEADING_MAP

    doc = Document()
    doc.add_heading(doc_heading, level=0)
    markdown_output = []
    debug_warnings = []

    # Step 1: LLM or Manual Markdown generation
    if use_llm:
        markdown_output = generate_llm_responses(blocks, api_key, model, debug)
    else:
        for i, block in enumerate(blocks):
            if not block.strip():
                if debug:
                    st.markdown(f"🔍 Skipping empty block {i+1}")
                continue
            markdown_output.append(block)

    # Step 2: Placeholder substitution for markdown/html (but preserve placeholders for DOCX)
    placeholder_map = get_schedule_placeholder_mapping()
    markdown_output_for_export = markdown_output.copy()
    for i, line in enumerate(markdown_output_for_export):
        placeholder = line.strip()
        if placeholder in placeholder_map:
            df_key = placeholder_map[placeholder]
            schedule_df = st.session_state.get(df_key)
            if isinstance(schedule_df, pd.DataFrame) and not schedule_df.empty:
                markdown_output_for_export[i] = add_table_from_schedule_to_markdown(schedule_df, section)
            else:
                markdown_output_for_export[i] = f"⚠️ No schedule data found for `{df_key}`"

    # Step 3: Generate final markdown and HTML
    markdown_text = "\n\n".join(markdown_output_for_export).strip()
    full_html_blocks = "\n".join([markdown.markdown(block) for block in markdown_output_for_export])

    combined_df = st.session_state.get("combined_home_schedule_df")
    if isinstance(combined_df, pd.DataFrame) and not combined_df.empty:
        full_html_blocks += "<h2>📆 Complete Schedule Summary</h2>\n"
        full_html_blocks += add_table_from_schedule_to_html(combined_df, section, include_priority=True)
    else:
        if debug:
            debug_warnings.append("⚠️ No schedule data found in [`combined_home_schedule_df`](#debug-combined_home_schedule_df). Skipping 📆 Complete Schedule Summary.")

    html_output = f"<html><body><h1>{doc_heading}</h1>\n{full_html_blocks}\n</body></html>"

    # Step 4: DOCX rendering
    for line in markdown_output:
        line = line.strip()
        if not line:
            continue

        if line in schedule_sources:
            df_key = schedule_sources[line]
            schedule_df = st.session_state.get(df_key)
            if isinstance(schedule_df, pd.DataFrame) and not schedule_df.empty:
                heading_text = SCHEDULE_HEADING_MAP.get(
                    line,
                    f"📆 {line.replace('<<INSERT_', '').replace('_SCHEDULE_TABLE>>', '').replace('_', ' ').title()} Schedule"
                )
                add_table_from_schedule(doc, schedule_df, section=section, include_heading=True, heading_text=heading_text)
            else:
                if debug:
                    section_hint = df_key.replace("_schedule_df", "").replace("_", " ").title()
                    debug_warnings.append(
                        f"⚠️ Skipping **{section_hint}** schedule placeholder `{line}` — no data found in [`st.session_state['{df_key}']`](#debug-{df_key})."
                    )
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

    # Step 5: Append final summary table (DOCX only)
    if isinstance(combined_df, pd.DataFrame) and not combined_df.empty:
        add_table_from_schedule(
            doc,
            combined_df,
            section=section,
            include_heading=True,
            heading_text="📆 Complete Schedule Summary"
        )
    doc.add_page_break()

    # Step 6: Finalize buffer
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)

    # Step 7: Debug rendering
    if debug and debug_warnings:
        with st.expander("⚠️ Missing Schedules (Debug)", expanded=True):
            for msg in debug_warnings:
                st.markdown(f"- {msg}", unsafe_allow_html=True)

        for key in ["combined_home_schedule_df", "trash_schedule_df", "mail_schedule_df"]:
            if key in st.session_state:
                st.markdown(f"<a name='debug-{key}'></a>", unsafe_allow_html=True)
                st.markdown(f"### 🔎 Debug: `{key}`", unsafe_allow_html=True)
                st.dataframe(st.session_state[key])

    return buffer, markdown_text, html_output




def preview_runbook_output(section: str, runbook_text: str, label: str = "📖 Preview Runbook"): ### depreciated ####
    """
    Debug/test-mode runbook preview with extra session context or metadata.

    Args:
        section (str): Used to determine which schedule data (if any) to insert.
        runbook_text (str): The raw LLM-generated markdown-style text.
        label (str): Button label to trigger the preview.
    """
    st.markdown(f"### {label} for `{section}`")

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

def runbook_preview_dispatcher(
    section: str,
    runbook_text: str,
    mode: Literal["debug", "inline"] = "inline",
    show_schedule_snapshot: bool = False
):
    """
    Renders an AI-generated runbook preview using tabs to avoid nested expander errors.
    Includes optional schedule preview in a second tab if enabled and available.

    Args:
        section (str): The section the runbook is for.
        runbook_text (str): The raw markdown-style text from the LLM.
        mode (str): Either "debug" or "inline" to determine the label.
        show_schedule_snapshot (bool): Whether to include a tab with schedule preview.
    """
    if not runbook_text:
        st.warning("⚠️ No runbook content available to preview.")
        return

    # Lookup potential schedule
    df_key = "combined_home_schedule_df" if section == "home" else f"{section}_schedule_df"
    schedule_df = st.session_state.get(df_key)
    schedule_available = isinstance(schedule_df, pd.DataFrame) and not schedule_df.empty

    # Inject schedule if requested and placeholder is present
    if "<<INSERT_SCHEDULE_TABLE>>" in runbook_text and schedule_available:
        schedule_md = add_table_from_schedule_to_markdown(schedule_df)
        runbook_text = runbook_text.replace("<<INSERT_SCHEDULE_TABLE>>", schedule_md)

    preview_label = "🧠 Runbook Preview (Debug)" if mode == "debug" else "📖 Runbook Preview"
    show_tabs = show_schedule_snapshot and schedule_available

    # Always use tabs to avoid nested expander errors
    if show_tabs:
        tabs = st.tabs([preview_label, "📆 Schedule Snapshot"])
        with tabs[0]:
            st.markdown(runbook_text, unsafe_allow_html=True)
        with tabs[1]:
            st.markdown("### 📆 Scheduled Tasks Snapshot")
            st.dataframe(schedule_df)
    else:
        tabs = st.tabs([preview_label])
        with tabs[0]:
            st.markdown(runbook_text, unsafe_allow_html=True)

def render_runbook_preview_inline(
    section: str,
    runbook_text: str,
    schedule_df: Optional[pd.DataFrame],
    heading: str,
    timestamp: str
):
    """
    Renders a preview of the runbook and an optional schedule snapshot using tabs.
    """
    tabs = st.tabs(["📖 Runbook Text", "📊 Scheduled Tasks"])

    # 📝 Runbook Markdown View
    with tabs[0]:
        st.markdown(f"### {heading}")
        st.markdown(f"_Generated on {timestamp}_")

        if "<<INSERT_SCHEDULE_TABLE>>" in runbook_text and isinstance(schedule_df, pd.DataFrame):
            schedule_md = add_table_from_schedule_to_markdown(schedule_df)
            runbook_text = runbook_text.replace("<<INSERT_SCHEDULE_TABLE>>", schedule_md)

        st.markdown(runbook_text, unsafe_allow_html=True)

    # 📅 Schedule DataFrame Snapshot
    with tabs[1]:
        if schedule_df is not None and not schedule_df.empty:
            st.dataframe(schedule_df)
        else:
            st.info("ℹ️ No scheduled task snapshot available.")

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
    html_key = f"{section}_runbook_html"
    preview_toggle_key = f"{section}_show_preview"

    buffer = st.session_state.get(buffer_key)
    runbook_text = st.session_state.get(text_key)
    html_output = st.session_state.get(html_key)
    doc_heading = doc_heading = st.session_state.get(f"{section}_doc_heading", f"{section.replace('_', ' ').title()} Runbook")
    st.session_state[f"{section}_doc_heading"] = doc_heading

    if not filename:
        filename = f"{section}_emergency_runbook.docx"

    st.subheader(f"📤 Export Options: {doc_heading}")

    # Row of buttons (columns)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📖 Preview Runbook", key=f"Preview_Runbook_{section}"):
            st.session_state[preview_toggle_key] = True

    with col2:
        if html_output:
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
            st.warning(f"⚠️ DOCX runbook buffer not found for {section}. Markdown/HTML export still available.")

    with col4:
        if st.button(f"♻️ Reset {section.title()} Runbook Cache", key=f"{section}_reset_preview"):
            for key in [buffer_key, text_key, html_key, f"{section}_runbook_ready", preview_toggle_key]:
                st.session_state.pop(key, None)
            st.success(f"🔄 Cleared session state for {section} runbook.")
            st.stop()

    # ✅ Full-width preview area outside the column row
    if runbook_text and st.session_state.get(preview_toggle_key):
        if "<<" in runbook_text and ">>" in runbook_text:
            st.warning("⚠️ Some schedule placeholders may not have been replaced in the runbook preview.")
        runbook_preview_dispatcher(
            section=section,
            runbook_text=runbook_text,
            mode="inline",
            show_schedule_snapshot=True
        )
        return True
    
    # ✅ Additional Debug Preview (always shows raw markdown if available)
    if runbook_text:
        with st.expander("📝 Raw Markdown Output (Debug)", expanded=False):
            st.code(runbook_text, language="markdown")

    return bool(buffer)

def maybe_generate_runbook(
    section: str,
    generator_fn: Callable[[], tuple[io.BytesIO, str, str]],
    *,
    doc_heading: str = "",
    button_label: str = "📥 Generate Runbook",
    filename: Optional[str] = None
):
    """
    Only generate runbook if requested by button and not already cached.

    Args:
        section (str): e.g., 'emergency_kit'
        generator_fn: returns (buffer, markdown_text, html_output)
        doc_heading (str): Optional heading for display.
        button_label (str): Button text.
        filename (str): Optional download filename.
    """
    generate_key = f"{section}_generate"
    ready_key = f"{section}_runbook_ready"
    buffer_key = f"{section}_runbook_buffer"
    text_key = f"{section}_runbook_text"

    st.subheader("🎉 Reward")
    if st.button("📤 Generate Runbook", key=generate_key):
        st.session_state[ready_key] = False
        st.rerun()

    if not st.session_state.get(ready_key):
        # Not ready yet, try generating
        buffer, markdown_text, html_output = generator_fn()
        if not markdown_text.strip():
            st.warning("⚠️ Runbook generation did not produce any content.")

        # Store in session
        st.session_state[buffer_key] = buffer
        st.session_state[text_key] = markdown_text
        st.session_state[f"{section}_runbook_html"] = html_output
        st.session_state[ready_key] = True

        if st.session_state.get("enable_debug_mode"):
            st.success(f"✅ Generated runbook for `{section}`")
            st.write("📋 Final Markdown Output:", markdown_text[:500])  # Preview
            st.write("📁 DOCX Buffer:", buffer)

    # Show download
    if st.session_state.get(ready_key):
        maybe_render_download(section=section, filename=filename or f"{section}_runbook.docx")
        st.session_state.setdefault("level_progress", {})[section] = True
    else:
        st.info("ℹ️ Click the button above to generate your runbook.")

