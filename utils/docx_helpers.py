from io import BytesIO
import io
import re
from typing import Optional
import streamlit as st
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from datetime import datetime


def format_provider_markdown(providers: dict, version: str = "v1.0") -> str:
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
      #  "internet": [("Outage Support", "emergency_steps", False, False)],
    }

        # 🔹 Date and version heading
    today = datetime.today().strftime("%B %d, %Y")
    heading = (
        "# 🛠️ Utility Providers Overview\n\n"
        f"📅 **Generated:** {today} | **Version:** {version}\n\n"
        "This guide includes a summary and detailed emergency info for each utility.\n"
    )

    # 🔹 Build summary table
    summary_lines = ["\n### 📋 Utility Provider Summary", "", "| Utility | Provider | Phone | Website |", "|---------|----------|-------|---------|"]
    for key, info in providers.items():
        icon = icon_map.get(key, "📦")
        name = info.get("name", "⚠️ Not provided").strip()
        phone = info.get("contact_phone", "⚠️").strip()
        website = info.get("contact_website", "").strip()

        # Format website as markdown link if valid
        website_display = f"[{website}]({website})" if website.startswith("http") else website or "⚠️"
        summary_lines.append(f"| {icon} {key.replace('_', ' ').title()} | {name} | {phone or '⚠️'} | {website_display} |")

    output = []

    for key, info in providers.items():
        name = info.get("name", "").strip()
        if not name:
            continue

        icon = icon_map.get(key, "📦")
        label = key.replace("_", " ").title()
        section_title = f"### {icon} {label} – {name}"

        table_lines = ["| Field | Value |", "|-------|--------|"]

        def row(label, field, is_link=False, multiline=False): # Removed is_email=False
            value = info.get(field, "").strip()
            if not value:
                return None
            if is_link:
                return f"| **{label}** | [{value}]({value}) |"
            #elif is_email and "@" in value:
            #   return f"| **{label}** | [{value}](mailto:{value}) |"
            elif multiline:
                value_md = value.replace("\n", "<br>")
                return f"| **{label}** | {value_md} |"
            else:
                return f"| **{label}** | {value} |"

        # Standard fields
        standard_fields = [
            ("Description", "description", False, False),
            ("Phone", "contact_phone", False, False),
            ("Website", "contact_website", True, False),
            #("Email", "contact_email", False, True),
            ("Address", "contact_address", False, False),
        ]

        fields_to_render = extra_fields.get(key, []) + standard_fields

        for field_entry in fields_to_render:
            if len(field_entry) == 3:
                label, field, is_link = field_entry  
                multiline = False
            elif len(field_entry) == 4:
                label, field, is_link, multiline = field_entry
            else:
                raise ValueError(f"❌ Invalid field_entry in fields_to_render.\nExpected 3 or 4 elements, got {len(field_entry)}: {field_entry}")
            line = row(label, field, is_link=is_link, multiline=multiline) # Removed is_email=is_email
            if line:
                table_lines.append(line)

        section_block = section_title + "\n" + "\n".join(table_lines)

        # ✅ Append Emergency Steps (outside the table)
        emergency = info.get("emergency_steps", "").strip()
        if emergency:
            emergency_md = "\n".join([
                line.strip() if line.strip().startswith(("-", "•", "*")) else f"- {line.strip()}"
                for line in emergency.splitlines() if line.strip()
            ])
            section_block += "\n\n### 🚨 Emergency Instructions\n" + emergency_md

        output.append(section_block)

    final_output = "\n\n---\n\n".join(output)
    return "## 🛠️ Utility Providers Overview\n\n" + final_output

def add_hyperlink(paragraph, url, text=None):
    """
    Add a hyperlink to a paragraph, styled as blue and underlined.
    """
    if text is None:
        text = url

    # Create relationship ID
    part = paragraph.part
    r_id = part.relate_to(url, RT.HYPERLINK, is_external=True)

    # Create hyperlink tag
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)

    # Create run element
    new_run = OxmlElement("w:r")

    # Create run properties with blue color and underline
    r_pr = OxmlElement("w:rPr")

    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0000FF")  # Blue

    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")  # Underline

    r_pr.append(color)
    r_pr.append(underline)
    new_run.append(r_pr)

    # Add the text
    t = OxmlElement("w:t")
    t.text = text
    new_run.append(t)

    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)

    return paragraph

def add_provider_summary_table(doc: Document, providers: dict):
    """
    Adds a summary table to the given Word doc listing all utility providers
    with utility type, name, phone number, and website (as hyperlink).
    """
    icon_map = {
        "electricity": "⚡",
        "natural_gas": "🔥",
        "water": "💧",
        "internet": "🌐"
    }

    # Add a heading for the summary
    doc.add_heading("🗂️ Utility Provider Summary", level=2)
    doc.add_paragraph("")  # spacer

    # Create a table with headers
    table = doc.add_table(rows=1, cols=4)
    table.style = "Table Grid"
    table.autofit = True
    table.allow_autofit = True

    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = "Utility"
    hdr_cells[1].text = "Provider Name"
    hdr_cells[2].text = "Phone"
    hdr_cells[3].text = "Website"

    # Add a row per provider
    for utility, info in providers.items():
        name = info.get("name", "").strip()
        phone = info.get("contact_phone", "").strip()
        website = info.get("contact_website", "").strip()

        if not name:
            continue

        icon = icon_map.get(utility, "🔌")
        row_cells = table.add_row().cells
        row_cells[0].text = f"{icon} {utility.replace('_', ' ').title()}"
        row_cells[1].text = name
        row_cells[2].text = phone

        if website.startswith("http"):
            # Add hyperlink inside cell
            p = row_cells[3].paragraphs[0]
            add_hyperlink(p, website, text=website)
        else:
            row_cells[3].text = website


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

    # Fields to render as a bullet list *below* the normal block
    multiline_fields = {
        "emergency_steps": "🚨 Emergency Instructions",
        "outage_support": "📶 Outage Support Instructions"
    }

    for utility, info in providers.items():
        name = info.get("name", "").strip()
        if not name:
            continue

        icon = icons.get(utility, "🔌")
        doc.add_heading(f"{icon} {utility.replace('_', ' ').title()} – {name}", level=2)

        def add_field(label, key, bold=True):
            value = info.get(key, "").strip()
            if not value:
                return

            para = doc.add_paragraph()
            run = para.add_run(f"{label}: ")
            if bold:
                run.bold = True

            # Add hyperlink if Website or Email
            if key == "contact_website" and value.startswith("http"):
                add_hyperlink(para, value)
            elif key == "contact_email" and "@" in value:
                add_hyperlink(para, f"mailto:{value}", text=value)
            else:
                para.add_run(value)

            para.paragraph_format.space_after = Pt(6)

        add_field("Description", "description")
        add_field("Phone", "contact_phone")
       #add_field("Email", "contact_email")
        add_field("Website", "contact_website")
        add_field("Address", "contact_address")

        # ✅ Multiline fields rendered as bulleted lists
        for field_key, heading in multiline_fields.items():
            value = info.get(field_key, "").strip()
            if value:
                lines = [line.strip() for line in value.splitlines() if line.strip()]
                if lines:
                    doc.add_paragraph("")  # spacer
                    doc.add_heading(heading, level=3)
                    for line in lines:
                        # Strip leading symbols/numbers
                        clean_line = re.sub(r"^[-•*]|\d+[.)]\s*", "", line).strip()
                        if clean_line:
                            doc.add_paragraph(clean_line, style="List Bullet")

# Export wrapper
def export_provider_docx(providers: dict, output_path: str = None, version: str = "v1.0"):
    """
    Creates a full Document, adds all utility providers, and saves to disk.
    """
    doc = Document()
    doc.add_heading("⚡🔥💧🌐 Utility Providers Emergency Guide", level=1)

    # Date and version stamp
    today = datetime.today().strftime("%B %d, %Y")
    doc.add_paragraph(f"📅 Generated: {today} | Version: {version}")
    doc.add_paragraph("The guide contains contact and emergency information for each utility provider.")
    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    add_provider_summary_table(doc, providers)
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
