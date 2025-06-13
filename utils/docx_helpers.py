from io import BytesIO
import io
from typing import Optional
import streamlit as st
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

def format_provider_markdown(providers: dict) -> str:
    """
    Returns markdown-formatted utility provider info using tables and icons.
    Adds support for utility-specific extra fields like 'Outage Support' for Internet.
    """
    icon_map = {
        "electricity": "⚡",
        "natural_gas": "🔥",
        "water": "💧",
        "internet": "🌐"
    }

    # Optional: extra fields by provider type
    extra_fields = {
        "internet": [("Outage Support", "emergency_steps", False, False)],
    }

    output = []

    for key, info in providers.items():
        name = info.get("name", "").strip()
        if not name:
            continue

        icon = icon_map.get(key, "📦")
        label = key.replace("_", " ").title()
        section_title = f"### {icon} {label} – {name}"

        table_lines = ["| Field | Value |", "|-------|--------|"]

        def row(label, field, is_link=False, is_email=False):
            value = info.get(field, "").strip()
            if not value:
                return None
            if is_link:
                return f"| **{label}** | [{value}]({value}) |"
            elif is_email and "@" in value:
                return f"| **{label}** | [{value}](mailto:{value}) |"
            else:
                return f"| **{label}** | {value} |"

        # Standard fields
        standard_fields = [
            ("Description", "description", False, False),
            ("Phone", "contact_phone", False, False),
            ("Website", "contact_website", True, False),
            ("Email", "contact_email", False, True),
            ("Address", "contact_address", False, False),
            ("Emergency Steps", "emergency_steps", False, False),
        ]

        fields_to_render = (
            extra_fields.get(key, []) + standard_fields
            if key in extra_fields
            else standard_fields
        )

        for label, field, is_link, is_email in fields_to_render:
            line = row(label, field, is_link=is_link, is_email=is_email)
            if line:
                table_lines.append(line)

        section_block = section_title + "\n" + "\n".join(table_lines)
        output.append(section_block)

    return "\n\n---\n\n".join(output)


# Utility function to add a single provider section to an existing doc
def add_provider_section_to_docx(doc: Document, providers: dict):
    """
    Appends a styled section for each utility provider to the given Document object.
    Skips fields that are missing or empty.
    """
    icons = {
        "electricity": "⚡",
        "natural_gas": "🔥",
        "water": "💧",
        "internet": "🌐"
    }

    for utility, info in providers.items():
        name = info.get("name", "").strip()
        if not name:
            continue

        icon = icons.get(utility, "🔌")
        doc.add_heading(f"{icon} {utility.replace('_', ' ').title()} – {name}", level=2)

        def add_field(label, key, bold=True):
            value = info.get(key, "").strip()
            if value:
                para = doc.add_paragraph()
                run = para.add_run(f"{label}: ")
                if bold:
                    run.bold = True
                para.add_run(value)
                para.paragraph_format.space_after = Pt(6)

        add_field("Description", "description")
        add_field("Phone", "contact_phone")
        add_field("Email", "contact_email")
        add_field("Website", "contact_website")
        add_field("Address", "contact_address")
        add_field("Emergency Steps", "emergency_steps")

# Export wrapper
def export_provider_docx(providers: dict, output_path: str= None):
    """
    Creates a full Document, adds all utility providers, and saves to disk.
    """
    doc = Document()
    doc.add_heading("📇 Utility Provider Information", level=1)
    doc.add_paragraph("This section contains contact and emergency information for each utility provider.")
    doc.add_paragraph()

    add_provider_section_to_docx(doc, providers)

    if output_path:
        doc.save(output_path)
        return None
    else:
        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return buffer.read()

def render_runbook_section_output(
    markdown_str: str,
    docx_bytes_io: Optional[BytesIO] = None,
    title: str = "Runbook Section",
    filename_prefix: str = "section",
    expand_preview: bool = False,
):
    """
    General-purpose viewer and exporter for runbook sections.
    """
    section_slug = filename_prefix.lower().replace(" ", "_")

    with st.expander(f"🧾 {title} Preview", expanded=expand_preview):
        if markdown_str:
            st.markdown(markdown_str, unsafe_allow_html=True)
        else:
            st.info(f"No markdown content available for {title}.")

    with st.expander(f"📁 Download {title}"):
        if docx_bytes_io:
            st.download_button(
                label=f"📄 Download {title} as DOCX",
                data=docx_bytes_io,
                file_name=f"{section_slug}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        elif markdown_str:
            st.download_button(
                label=f"⬇️ Download {title} as Markdown",
                data=markdown_str,
                file_name=f"{section_slug}.md",
                mime="text/markdown"
            )
        else:
            st.info(f"⚠️ No downloadable content for {title}.")
