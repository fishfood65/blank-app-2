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
from .common_helpers import get_schedule_placeholder_mapping

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
        return None, prompts or []

    if prompts is None:
        prompts = []

    # Merge related data if needed
    if section == "mail_trash_handling":
        merged_inputs = (
            st.session_state.get("input_data", {}).get("mail", []) +
            st.session_state.get("input_data", {}).get("trash_handling", [])
        )
        st.session_state["input_data"]["mail_trash_handling"] = merged_inputs
        st.write("📬 [DEBUG] Saved merged input_data['mail_trash_handling']:", merged_inputs)

    # Section-specific prompt collection
    if section == "home":
        prompts.append(utilities_emergency_runbook_prompt())
    elif section == "emergency_kit":
        prompts.append(emergency_kit_utilities_runbook_prompt())
    elif section == "mail_trash_handling":
        prompts.extend([
            emergency_kit_utilities_runbook_prompt(),
            mail_trash_runbook_prompt(debug_key="trash_info_debug_preview"),
        ])
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
        prompts.append(f"# ⚠️ No prompt available for section: {section}")

    # ✅ Flatten all sub-lists into a single prompt block list
    flat_prompts = []
    for p in prompts:
        if isinstance(p, list):
            flat_prompts.extend(p)
        else:
            flat_prompts.append(p)

    combined_prompt = "\n\n".join(flat_prompts)

    # ✅ Store both combined and fragment forms
    st.session_state["generated_prompt"] = combined_prompt
    st.session_state["prompt_blocks"] = flat_prompts

    return combined_prompt, flat_prompts

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


def maybe_generate_runbook(section: str = "home", doc_heading: Optional[str] = None):
    """
    Generate a runbook DOCX from the prompt blocks stored in session state and render download.

    Parameters:
    - section: str — used to customize default document heading and filename
    - doc_heading: Optional[str] — override the heading shown in the document
    """
    prompt_blocks: List[str] = st.session_state.get("prompt_blocks", [])
    combined_prompt: str = st.session_state.get("generated_prompt", "")
    
    schedule_placeholder = "<<INSERT_SCHEDULE_TABLE>>"

    # Fallback: If no blocks but prompt string exists
    if not prompt_blocks and combined_prompt:
        prompt_blocks = [combined_prompt]
        if st.session_state.get("enable_debug_mode"):
            st.info("⚠️ No prompt blocks found — using combined prompt as fallback.")

    # Validate: Remove any blank/empty blocks
    prompt_blocks = [block for block in prompt_blocks if block.strip()]
    if not prompt_blocks:
        st.warning("⚠️ No usable prompt blocks available. Cannot generate runbook.")
        return

    if doc_heading is None:
        doc_heading = f"{section.replace('_', ' ').title()} Emergency Runbook"

    button_key = f"generate_runbook_button_{section}"
    generate_triggered = st.button("📄 Click Me to Generate Runbook", key=button_key)

    if generate_triggered:
        # ✅ Make a safe copy of the prompt blocks
        final_prompt_blocks = prompt_blocks.copy()

        # ✅ Add schedule placeholder only if needed
        if section == "mail_trash_handling":
            if all(schedule_placeholder not in block for block in final_prompt_blocks):
                final_prompt_blocks.append(f"## 📆 Mail & Trash Pickup Schedule\n\n{schedule_placeholder}")

        if st.session_state.get("enable_debug_mode"):
            st.markdown("### 🧾 Prompt Blocks Being Sent to LLM")
            for i, block in enumerate(final_prompt_blocks):
                st.code(f"[Block {i + 1}]\n{block}", language="markdown")

        try:
            # 🔁 Call LLM for each block
            buffer, llm_output = generate_docx_from_split_prompts(
                prompts=final_prompt_blocks,
                api_key=os.getenv("MISTRAL_TOKEN"),
                doc_heading=doc_heading,
                debug=st.session_state.get("enable_debug_mode", False)
            )

            # Save outputs to section-scoped session keys
            st.session_state[f"{section}_runbook_buffer"] = buffer
            st.session_state[f"{section}_runbook_text"] = llm_output
            st.session_state[f"{section}_runbook_ready"] = True

            if st.session_state.get("enable_debug_mode"):
                st.markdown("### 📥 LLM Response (Combined)")
                st.code(llm_output, language="markdown")

        except Exception as e:
            st.error(f"❌ Failed to generate runbook: {e}")
            st.session_state[f"{section}_runbook_ready"] = False

    # ✅ Always check if runbook is ready for download
    if st.session_state.get(f"{section}_runbook_ready"):
        st.markdown("___")
        st.write("⏲️ Runbook Ready")
        maybe_render_download(section=section)

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
    
