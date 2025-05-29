import streamlit as st
from mistralai import Mistral, UserMessage, SystemMessage
import csv
from io import BytesIO
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
import tempfile
from prompts.prompts_home import (
    utilities_emergency_runbook_prompt,
    emergency_kit_utilities_runbook_prompt,
    mail_trash_runbook_prompt,
    home_caretaker_runbook_prompt,
    emergency_kit_document_prompt,
    bonus_level_runbook_prompt
)

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
        table.style = 'Light List' #Optional: imporve tabel appearance
        table.autofit = True

        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'Task'
        hdr_cells[1].text = 'Image'

        for _, row in group.iterrows():
            task = row["Task"]
            image_path = ""

            # Look for matching uploaded images
            for label in ["Outdoor Bin Image", "Recycling Bin Image"]:
                if label.lower().replace(" image", "") in task.lower():
                    uploaded = st.session_state.trash_images.get(label)
                    if uploaded:
                        # Save uploaded file to temp location
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                            tmp.write(uploaded.read())
                            image_path = tmp.name
                        break

            row_cells = table.add_row().cells
            row_cells[0].text = task

            if image_path:
                try:
                    run = row_cells[1].paragraphs[0].add_run()
                    run.add_picture(image_path, width=Inches(1.5))
                    if st.session_state.get("enable_debug_mode"):
                        st.write(f"🖼️ Added image for task: {task}")
                except Exception as e:
                    row_cells[1].text = "⚠️ Image load failed"
                    if st.session_state.get("enable_debug_mode"):
                        st.write(f"❌ Error adding image for task '{task}': {e}")
            else:
                row_cells[1].text = ""

        doc.add_paragraph("")  # spacing between tables

def generate_docx_from_split_prompts(
    prompts: List[str],
    api_key: str,
    *,
    section_titles: Optional[List[str]] = None,
    model: str = "mistral-small-latest",
    doc_heading: str = "Runbook",
    temperature: float = 0.5,
    max_tokens: int = 2048,
    debug: bool = False  # Optional debug toggle
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

        if line == "<<INSERT_SCHEDULE_TABLE>>" and not schedule_inserted:
            schedule_df = st.session_state.get("home_schedule_df", pd.DataFrame())
            if debug:
                doc.add_paragraph("✅ [DEBUG] Schedule table inserted below.")
                print("📋 Found INSERT_SCHEDULE_TABLE marker.")
                print("📋 Schedule DataFrame passed to add_table_from_schedule:\n", schedule_df)
            add_table_from_schedule(doc, schedule_df)
            schedule_inserted = True  # Skip adding this line as plain text
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

        # Numbered list
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
                para.add_run(match.group(1)[2:-2]).bold = True
                cursor = end
            if cursor < len(line):
                para.add_run(line[cursor:])
            para.style.font.size = Pt(11)

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)

    return buffer, full_text

def generate_docx_from_text(text: str, doc_heading: str = "Runbook") -> BytesIO:
    doc = Document()
    doc.add_heading(doc_heading, 0)

    lines = text.splitlines()
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

        # Same heading, bullet, and markdown logic...
        # (copy from generate_docx_from_split_prompts)

    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer


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

def maybe_generate_prompt(section: str = "home", prompts: Optional[List[str]] = None) -> Tuple[Optional[str], List[str]]:
    """
    Generate a section-specific prompt and return both the final combined prompt string
    and a list of individual prompt fragments (if any).
    
    Parameters:
    - section: str — like "home", "pets", etc.
    - prompts: Optional[List[str]] — used to collect individual prompts

    Returns:
    - Tuple of (combined_prompt_string or None, list_of_prompt_strings)
    """
    confirm_key = f"confirm_ai_prompt_{section}"
    confirmed = st.session_state.get(confirm_key, False)

    st.write(f"🧪 maybe_generate_prompt() called for section: `{section}`")
    st.write(f"🧪 Confirmation checkbox state: {confirmed}")
    #st.write(f"🧪 Current input data for section:", st.session_state.get("input_data", {}).get(section, []))

    if not confirmed:
        st.session_state["generated_prompt"] = None
        return None, prompts or []

    if prompts is None:
        prompts = []

        # Merge inputs from related subsections if needed
    if section == "mail_trash_handling":
        merged_inputs = (
            st.session_state.get("input_data", {}).get("mail", []) +
            st.session_state.get("input_data", {}).get("trash_handling", [])
        )
        st.session_state["input_data"]["mail_trash_handling"] = merged_inputs  # 🔁 Save merged view
        st.write("📬 [DEBUG] Saved merged input_data['mail_trash_handling']:", merged_inputs)

    #if "mail_trash_handling" in st.session_state.get("input_data", {}):
    #    st.subheader("🧪 Debug: Merged Mail & Trash Inputs")
    #    st.json(st.session_state["input_data"]["mail_trash_handling"])
    #else:
    #    st.info("ℹ️ No merged mail_trash_handling input found in session_state.")

    #if st.sidebar.checkbox("🧪 Show `mail_trash_handling` Data"):
    #    merged_inputs = st.session_state.get("input_data", {}).get("mail_trash_handling", [])
    #    for i, entry in enumerate(merged_inputs):
     #       st.sidebar.markdown(f"**Entry {i + 1}**")
    #        for k, v in entry.items():
    #            st.sidebar.write(f"- `{k}`: {v}")

    # Section-specific logic
    if section == "home":
        prompts.append(utilities_emergency_runbook_prompt())
    elif section == "emergency_kit":
        prompts.append(emergency_kit_utilities_runbook_prompt())
    elif section == "mail_trash_handling":
        prompts.extend([
            emergency_kit_utilities_runbook_prompt(),
            mail_trash_runbook_prompt(debug_key="trash_info_debug_preview"),
        ])
        prompt = mail_trash_runbook_prompt(debug_key="trash_info_debug_run")
        #st.write("📬 [DEBUG] mail_trash_runbook_prompt returned:", prompt)
        prompts.append(prompt)

    elif section == "home_security":
        prompts.extend([
            emergency_kit_utilities_runbook_prompt(),
            mail_trash_runbook_prompt(),
            home_caretaker_runbook_prompt()
        ])
    elif section == "emergency_kit_critical_documents":
        prompts.append(emergency_kit_document_prompt())
    elif section == "bonus_level":
        prompts.append(bonus_level_runbook_prompt())
    else:
        prompts.append = (f"# ⚠️ No prompt available for section: {section}")

    # Combine into a single prompt string
    combined_prompt = "\n\n".join(prompts)

    # Save to session
    st.session_state["generated_prompt"] = combined_prompt
    st.write("🧪 [DEBUG] Combined Prompt:", st.session_state["generated_prompt"])

    return combined_prompt, prompts

def render_prompt_preview(missing: list, section: str = "home"):
    confirmed = st.session_state.get(f"{section}_user_confirmation", False)
    
    with st.expander("🧠 AI Prompt Preview (Optional)", expanded=True):
        if missing:
            st.warning(f"⚠️ Cannot generate prompt. Missing: {', '.join(missing)}")
        elif not confirmed:
            st.info("☕️ Please check the box to confirm AI prompt generation.")
        elif st.session_state.get("generated_prompt"):
            prompt = st.session_state["generated_prompt"] # added for debug
            #st.code(st.session_state["generated_prompt"], language="markdown") #-- commented out for debugging

            schedule_md = st.session_state.get("home_schedule_markdown", "_No schedule available._") # added for debug
            prompt_with_schedule = prompt.replace("<<INSERT_SCHEDULE_TABLE>>", schedule_md) # added for debug
            st.success("✅ Prompt ready! Now you can generate your runbook.")

            # Show in code block
            st.code(prompt_with_schedule, language="markdown") # added for debuging
            #st.success("✅ Prompt ready! Now you can generate your runbook.")

        else:
            st.warning("⚠️ Prompt not generated yet.")

def maybe_render_download(section: str = "home", filename: Optional[str] = None):
    """
    Renders download button and preview for generated runbook.
    
    Parameters:
    - section: str — used to generate default filename
    - filename: Optional[str] — custom file name for the DOCX download
    """
    buffer = st.session_state.get("runbook_buffer")
    runbook_text = st.session_state.get("runbook_text")

    if not filename:
        filename = f"{section}_emergency_runbook.docx"

    if buffer:
        st.download_button(
            label="📥 Download DOCX",
            data=buffer,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        st.success("✅ Runbook ready for download!")
        st.write("📋 [DEBUG] runbook_text preview:") # for debug
        st.code(runbook_text, language="markdown") # for debug

    if runbook_text:
        preview_runbook_output(runbook_text)

def maybe_generate_runbook(section: str = "home", doc_heading: Optional[str] = None):
    """
    Generate a runbook DOCX from the prompt stored in session state and render download.
    
    Parameters:
    - section: str — used to customize default document heading and filename
    - doc_heading: Optional[str] — override the heading shown in the document
    """
    prompt = st.session_state.get("generated_prompt", "")
    if not prompt:
        st.warning("⚠️ No prompt found in session. Cannot generate runbook.")
        return
    #st.write("📄 [DEBUG] Prompt to generate:", st.session_state.get("generated_prompt"))
    # Fix: Set doc_heading before button block
    if doc_heading is None:
        doc_heading = f"{section.replace('_', ' ').title()} Emergency Runbook"

    if st.button("📄 Generate Runbook Document"):
        schedule_md = st.session_state.get("home_schedule_markdown", "_No schedule available._")
        prompt_with_schedule = prompt.replace("<<INSERT_SCHEDULE_TABLE>>", schedule_md)

        # ✅ Debugging block: show prompt and schedule
        if st.session_state.get("enable_debug_mode"):
            st.write("📋 Raw Prompt with Schedule:", prompt_with_schedule)
            st.write("📊 Schedule DataFrame:", st.session_state.get("home_schedule_df"))

        try:
            buffer, _ = generate_docx_from_split_prompts(
                prompts=[prompt_with_schedule],
                api_key=os.getenv("MISTRAL_TOKEN"),
                doc_heading=doc_heading,
                debug=st.session_state.get("enable_debug_mode", False)# <-- Controlled by checkbox
            )

            # Replace in runbook_text too — since the markdown version is shown in preview
            # runbook_text = prompt_with_schedule

            # Save to session state
            st.session_state["runbook_buffer"] = buffer
            st.session_state["runbook_text"] = prompt_with_schedule
            st.session_state["runbook_ready"] = True  # ✅ Flag that buffer is ready
        except Exception as e:
            st.error(f"❌ Failed to generate runbook: {e}")
        # Always show preview/download if runbook is ready
    if st.session_state.get("runbook_ready"):
        maybe_render_download(section=section)
