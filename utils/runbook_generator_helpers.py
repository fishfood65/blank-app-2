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
from typing import List, Tuple

def generate_docx_from_split_prompts(
    prompts: List[str],
    api_key: str,
    *,
    model: str = "mistral-small-latest",
    doc_heading: str = "Runbook",
    temperature: float = 0.5,
    max_tokens: int = 2048,
) -> Tuple[io.BytesIO, str]:
    """
    Splits prompts into individual LLM calls to manage token limits,
    stitches together the results, and returns a formatted DOCX and combined response.
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
                combined_output.append(f"### Section {i+1}\n\n{response_text.strip()}")
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

import streamlit as st

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
