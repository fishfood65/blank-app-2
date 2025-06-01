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
from typing import List, Tuple, Optional
import tempfile
from .common_helpers import get_schedule_placeholder_mapping
from .prompt_block_utils import generate_all_prompt_blocks
from task_schedule_utils_updated import export_schedule_to_markdown

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

def add_table_from_schedule_to_markdown(schedule_df: pd.DataFrame) -> str:
    """
    Converts a schedule DataFrame into a grouped markdown table by date.
    Each date section includes a table of tasks and optional image placeholders.

    Args:
        schedule_df (pd.DataFrame): The structured schedule with at least 'Date' and 'Task' columns.

    Returns:
        str: A markdown-formatted string of the grouped schedule.
    """
    if schedule_df.empty:
        return "_No schedule data available._"

    # Convert and sort
    schedule_df["Date"] = pd.to_datetime(schedule_df["Date"], errors="coerce")
    schedule_df = schedule_df.sort_values(by=["Date", "Source", "Tag", "Task"])
    schedule_df["Day"] = schedule_df["Date"].dt.strftime("%A")
    schedule_df["DateStr"] = schedule_df["Date"].dt.strftime("%Y-%m-%d")

    output = StringIO()
    for date, group in schedule_df.groupby("Date"):
        day_str = date.strftime("%A, %Y-%m-%d")
        output.write(f"### 📅 {day_str}\n\n")
        output.write("| Task | Image |\n")
        output.write("|------|-------|\n")
        for _, row in group.iterrows():
            task = row["Task"]
            # Image column remains blank or placeholder
            output.write(f"| {task} |  |\n")
        output.write("\n")

    return output.getvalue()

def generate_docx_from_split_prompts(
    prompts: List[str],
    api_key: str,
    *,
    section_titles: Optional[List[str]] = None,
    model: str = "mistral-small-latest",
    doc_heading: str = "Runbook",
    temperature: float = 0.5,
    max_tokens: int = 2048,
    debug: bool = False
) -> Tuple[io.BytesIO, str]:

    if not prompts or not isinstance(prompts, list):
        raise ValueError("🚫 prompts must be a non-empty list of strings.")

    prompts = [p.strip() for p in prompts if p.strip()]
    if not prompts:
        raise ValueError("🚫 All prompts were empty after stripping.")

    combined_output = []

    for i, prompt in enumerate(prompts):
        try:
            if debug:
                st.markdown(f"### 🧾 Prompt Block {i+1}")
                st.code(prompt, language="markdown")

            client = Mistral(api_key=api_key)
            completion = client.chat.complete(
                model=model,
                messages=[SystemMessage(content=prompt)],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            response_text = completion.choices[0].message.content

            if debug:
                st.markdown(f"### 🧾 Raw LLM Response [Block {i + 1}]")
                st.code(response_text, language="markdown")

            title = section_titles[i].strip() if section_titles and i < len(section_titles) else ""
            section_label = f"### {title}" if title else ""
            output_block = f"{section_label}\n\n{response_text.strip()}" if section_label else response_text.strip()
            combined_output.append(output_block)

        except Exception as e:
            st.error(f"❌ Error processing prompt {i + 1}: {e}")
            continue

    full_text = "\n\n".join(combined_output)
    if debug:
        st.markdown("### 🔍 Full Text Passed to DOCX Builder")
        st.code(full_text, language="markdown")

    doc = Document()
    doc.add_heading(doc_heading, 0)
    lines = full_text.splitlines()

    if debug:
        st.markdown("### 🧾 Lines Sent to DOCX")
        st.code("\n".join(lines), language="markdown")

    schedule_sources = get_schedule_placeholder_mapping()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Table placeholders
        if line in schedule_sources:
            df_key = schedule_sources[line]
            schedule_df = st.session_state.get(df_key)

            doc.add_paragraph("")  # spacing
            if isinstance(schedule_df, pd.DataFrame) and not schedule_df.empty:
                if debug:
                    doc.add_paragraph(f"✅ [DEBUG] Inserted table for {line}")
                add_table_from_schedule(doc, schedule_df)
            else:
                doc.add_paragraph("_No schedule available._")
                if debug:
                    st.warning(f"⚠️ No data found for placeholder: {line}")
            doc.add_paragraph("")
            continue

        # Markdown-to-Word formatting
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

def maybe_render_download(section: str = "home", filename: Optional[str] = None):
    """
    Renders download button and preview for generated runbook.
    
    Parameters:
    - section: str — used to generate default filename
    - filename: Optional[str] — custom file name for the DOCX download
    """
    buffer = st.session_state.get(f"{section}_runbook_buffer")
    runbook_text = st.session_state.get(f"{section}_runbook_text")

    if not filename:
        filename = f"{section}_emergency_runbook.docx"

    if runbook_text:
        preview_runbook_output(runbook_text)

    if buffer:
        st.download_button(
            label="📥 Download DOCX",
            data=buffer,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        st.success("✅ Runbook ready for download!")

def maybe_generate_prompt(section: str = "home") -> Tuple[Optional[str], List[str]]:
    """
    Generate a section-specific prompt and return both the final combined prompt string
    and a list of individual prompt fragments (if any).
    
    Returns:
    - combined_prompt (str or None)
    - flat_prompts (List[str]) — the actual prompt chunks for use with multi-part LLM generation
    """
    confirm_key = f"confirm_ai_prompt_{section}"
    confirmed = st.session_state.get(confirm_key, False)

    st.write(f"🧪 maybe_generate_prompt() called for section: `{section}`")
    st.write(f"🧪 Confirmation checkbox state: {confirmed}")

    if not confirmed:
        st.session_state["generated_prompt"] = None
        st.session_state["prompt_blocks"] = []
        return None, []

    # Handle merging of multi-input fields
    if section == "mail_trash_handling":
        merged_inputs = (
            st.session_state.get("input_data", {}).get("mail", []) +
            st.session_state.get("input_data", {}).get("trash_handling", [])
        )
        st.session_state["input_data"]["mail_trash_handling"] = merged_inputs
        st.write("📬 [DEBUG] Saved merged input_data['mail_trash_handling']:", merged_inputs)

    # ✅ Use centralized prompt block builder
    prompt_blocks = generate_all_prompt_blocks(section)

    # Flatten and combine for LLM preview and fallback
    flat_prompts = []
    for block in prompt_blocks:
        if isinstance(block, list):
            flat_prompts.extend(block)
        else:
            flat_prompts.append(block)

    combined_prompt = "\n\n".join(flat_prompts)

    # ✅ Store for downstream use
    st.session_state["generated_prompt"] = combined_prompt
    st.session_state["prompt_blocks"] = flat_prompts

    return combined_prompt, flat_prompts

def maybe_generate_runbook(section: str = "home", doc_heading: Optional[str] = None):
    """
    Generate a runbook DOCX from the prompt blocks generated by section and render download.

    Parameters:
    - section: str — used to customize default document heading and filename
    - doc_heading: Optional[str] — override the heading shown in the document
    """
    schedule_placeholder = "<<INSERT_SCHEDULE_TABLE>>"

    # Always generate fresh blocks from source
    prompt_blocks = generate_all_prompt_blocks(section)
    combined_prompt = "\n\n".join(prompt_blocks)

    # Fallback: If no blocks generated, try session fallback
    if not prompt_blocks and st.session_state.get("generated_prompt"):
        prompt_blocks = [st.session_state["generated_prompt"]]
        if st.session_state.get("enable_debug_mode"):
            st.info("⚠️ No prompt blocks returned — using combined prompt from session.")

    # Clean: Remove any empty or whitespace-only blocks
    prompt_blocks = [b for b in prompt_blocks if b.strip()]
    if not prompt_blocks:
        st.warning("⚠️ No valid prompt blocks available. Cannot generate runbook.")
        return

    # Set default heading if needed
    if doc_heading is None:
        doc_heading = f"{section.replace('_', ' ').title()} Emergency Runbook"

    # Button logic to trigger LLM + DOCX generation
    button_key = f"generate_runbook_button_{section}"
    generate_triggered = st.button("📄 Click Me to Generate Runbook", key=button_key)

    if generate_triggered:
        final_prompt_blocks = prompt_blocks.copy()
        st.write("🧩 Placeholder Mapping:", get_schedule_placeholder_mapping())

        # ✅ Schedule placeholder (if applicable)
        #if section == "mail_trash_handling":
        #    if all(schedule_placeholder not in block for block in final_prompt_blocks):
        #        final_prompt_blocks.append(f"## 📆 Mail & Trash Pickup Schedule\n\n{schedule_placeholder}")

        if st.session_state.get("enable_debug_mode"):
            st.markdown("### 🧾 Prompt Blocks Being Sent to LLM")
            for i, block in enumerate(final_prompt_blocks):
                st.code(f"[Block {i + 1}]\n{block}", language="markdown")

        try:
            buffer, llm_output = generate_docx_from_split_prompts(
                prompts=final_prompt_blocks,
                api_key=os.getenv("MISTRAL_TOKEN"),
                doc_heading=doc_heading,
                debug=st.session_state.get("enable_debug_mode", False)
            )

            # Store result under section scope
            st.session_state[f"{section}_runbook_buffer"] = buffer
            st.session_state[f"{section}_runbook_text"] = llm_output
            st.session_state[f"{section}_runbook_ready"] = True

            if st.session_state.get("enable_debug_mode"):
                st.markdown("### 📥 LLM Response (Combined)")
                st.code(llm_output, language="markdown")

        except Exception as e:
            st.error(f"❌ Failed to generate runbook: {e}")
            st.session_state[f"{section}_runbook_ready"] = False

    # Show download link if ready
    if st.session_state.get(f"{section}_runbook_ready"):
        st.markdown("___")
        st.write("⏲️ Runbook Ready")
        maybe_render_download(section=section)

def preview_runbook_output(runbook_text: str, label: str = "📖 Preview Runbook"):
    """
    Shows an expandable markdown preview of the runbook text when a button is clicked.

    Args:
        runbook_text (str): The raw LLM-generated markdown-style text.
        label (str): Button label to trigger the preview.
    """
    # use the following code with home_app_05_23_modified.py
    #if not runbook_text:
    #    st.warning("⚠️ No runbook content available to preview.")
    #    return

    #if st.button(label):
    #    with st.expander("🧠 AI-Generated Runbook Preview", expanded=True):
    #        st.markdown(runbook_text)
    if not runbook_text:
        st.warning("⚠️ No runbook content available to preview.")
        return

    if st.button(label):
        with st.expander("🧠 AI-Generated Runbook Preview", expanded=True):
            # Main LLM output
            st.markdown(runbook_text)

            # Optional Schedule Table Preview
            schedule_df = st.session_state.get("combined_home_schedule_df")
            if isinstance(schedule_df, pd.DataFrame) and not schedule_df.empty:
                st.markdown("### 📆 Schedule Summary")
                schedule_md = export_schedule_to_markdown(schedule_df)
                st.markdown(schedule_md)
            else:
                st.info("ℹ️ No schedule available to display.")

def render_prompt_preview(missing: list, section: str = "home"):
    confirmed = st.session_state.get(f"{section}_user_confirmation", False)

    with st.expander("🧠 AI Prompt Preview (Optional)", expanded=True):
        if missing:
            st.warning(f"⚠️ Cannot generate prompt. Missing: {', '.join(missing)}")
            return

        if not confirmed:
            st.info("☕️ Please check the box to confirm AI prompt generation.")
            return

        prompt = st.session_state.get("generated_prompt", "")
        prompt_blocks = st.session_state.get("prompt_blocks", [])
        schedule_md = st.session_state.get("home_schedule_markdown", "_No schedule available._")

        if not prompt:
            st.warning("⚠️ Prompt not generated yet.")
            return

        # Build combined preview by inserting schedule into the final prompt block
        if prompt_blocks:
            full_preview = "\n\n".join(prompt_blocks)
            full_preview = full_preview.replace("<<INSERT_SCHEDULE_TABLE>>", schedule_md)
        else:
            full_preview = prompt.replace("<<INSERT_SCHEDULE_TABLE>>", schedule_md)

        # Display the formatted full prompt preview
        st.markdown(full_preview)
        if section == "mail_trash_handling":
            st.markdown("### 📋 Schedule Preview")
            st.markdown(st.session_state.get("home_schedule_markdown", "_No schedule available._"))
        st.success("✅ Prompt ready! This is what will be sent to the LLM.")




def maybe_render_download(section: str = "home", filename: Optional[str] = None):
    """
    Renders download button and preview for generated runbook.
    
    Parameters:
    - section: str — used to generate default filename
    - filename: Optional[str] — custom file name for the DOCX download
    """
    buffer = st.session_state.get(f"{section}_runbook_buffer")
    runbook_text = st.session_state.get(f"{section}_runbook_text")

    if not filename:
        filename = f"{section}_emergency_runbook.docx"

    if runbook_text:
        preview_runbook_output(runbook_text)

    if buffer:
        st.download_button(
            label="📥 Download DOCX",
            data=buffer,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        st.success("✅ Runbook ready for download!")
    
