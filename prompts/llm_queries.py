from config.sections import check_home_progress
from utils.common_helpers import (
    extract_all_trash_tasks_grouped, 
    extract_grouped_mail_task, 
    generate_flat_home_schedule_markdown,
)
from utils.data_helpers import (
    capture_input, 
    flatten_answers_to_dict, 
    get_answer, 
    extract_and_log_providers, 
    log_provider_result, 
    autolog_location_inputs, 
    preview_input_data, 
    check_missing_utility_inputs, 
    export_input_data_as_csv, 
    render_lock_toggle
)
from prompts.templates import utility_provider_lookup_prompt
import streamlit as st
import re
from mistralai import Mistral, UserMessage, SystemMessage
from dotenv import load_dotenv
import os
import pandas as pd
from datetime import datetime, timedelta
from docx import Document
from docx.text.run import Run
import re
import time
from PIL import Image
import io
import uuid
import json

#### Security and Services Prompt ####

def home_caretaker_runbook_prompt() -> str:
    """Builds the LLM prompt for the caretaker runbook using dynamic field fetching."""
    
    def val(label, section):
        return get_answer(label, section) or "⚠️ Not provided"
        
    blocks = []

    # ── Home Security ──
    sec = "Home Security"
    blocks.append(generate_section_block("🔐 Home Security & Technology", [
        ("Security Company Name", val("Security Company Name", sec)),
        ("Security Company Number", val("Security Company Phone Number", sec)),
        ("Arming/Disarming Instructions", val("Instructions to arm/disarm system", sec)),
        ("If Alert is Triggered", val("Steps if a security alert is triggered", sec)),
        ("Indoor Camera Notes", val("Indoor cameras/monitoring details and activation", sec)),
        ("Emergency Access Instructions", val("Emergency access instructions & storage location", sec)),
        ("Wi-Fi Info Location", val("Where is Wi-Fi network name/password stored?", sec)),
        ("Guest Wi-Fi Access", val("Guest network details & password sharing method", sec)),
        ("Landline/VOIP Notes", val("Home phone setup & call-handling instructions", sec)),
    ]))

    # ── Quality-Oriented Services ──
    for service in ["Cleaning", "Gardening/Landscape", "Pool Maintenance"]:
        sec = f"Quality-Oriented Household Services.{service}"
        blocks.append(generate_section_block(f"{service} Service Instructions", [
            ("Company Name", val("Company Name", sec)),
            ("Phone Number", val("Company Phone Number", sec)),
            ("Schedule", f"{val('Frequency', sec)} on {val('Day of the Week', sec)}"),
            ("Access Method", val("Access Method", sec)),
            ("Post-Service Procedures", val("Post-Service Procedures", sec)),
            ("Crew Identity Verification", val("Crew Identity Verification", sec)),
        ]))

    # ── Property Management ──
    sec = "Rent or Own.Property Management"
    blocks.append(generate_section_block("🏢 Property Management (Renters or HOA)", [
        ("Company Name", val("Company Name", sec)),
        ("Phone Number", val("Company Phone Number", sec)),
        ("Email", val("Company Email", sec)),
        ("When to Contact", val("When to Contact", sec)),
    ]))

    # ── Homeowner Contacts ──
    for role in ["Handyman", "Electrician", "Exterminator", "Plumber"]:
        sec = f"Rent or Own.{role}"
        blocks.append(generate_section_block(f"🛠️ {role} Contact", [
            ("Name", val("Name", sec)),
            ("Phone", val("Phone Number", sec)),
            ("When to Contact", val("When to Contact", sec)),
        ]))

    # ── HOA ──
    sec = "Rent or Own.HOA"
    blocks.append(generate_section_block("🏘️ HOA / Property Management", [
        ("Company Name", val("Company Name", sec)),
        ("Phone Number", val("Phone Number", sec)),
        ("Email", val("Email", sec)),
        ("When to Contact", val("When to Contact", sec)),
    ]))

    all_blocks = "\n".join([block for block in blocks if block.strip()])
    
    return f"""
You are a helpful assistant tasked with generating a professional, detailed, and easy-to-follow Home Caretaker & Guest Runbook. The goal is to ensure a smooth experience for caretakers or guests while the home occupants are away.

Please omit any sections or bullet points where the value is "⚠️ Not provided". Do not invent or assume any missing information.

Please do not include a title. Start directly with the structured guide below.

### 📕 Security and Services Guide

{all_blocks.strip()}
""".strip()

#### Emergency Kit Documents ####

def emergency_kit_document_prompt():
    """
    Build the LLM prompt string from session_state, but first prune
    document_details so that each document only includes the fields
    for the storage locations it’s actually assigned to.
    """
    intro = (
        "Welcome to your Emergency Document Kit.\n\n"
        "- Reduces stress, and delivers peace of mind in a crisis.\n\n"
        "How to read this document:\n"
        "1. Start here for the value of the kit:\n\n"
        "   • Provides clear, step-by-step guidance to locate and retrieve vital documents quickly in a crisis.\n"
        "2. Scroll down to each storage location heading (e.g., “## Safe”)—"
        "these are sorted by where most documents live first.\n"
        "3. Under each location, you’ll find:\n\n"
        "   • Location details (address or placement info)\n\n"
        "   • Platform (if applicable, e.g., cloud service or password manager)\n\n"
        "   • Access steps (what’s required in an emergency)\n\n"
        "   • Contents notes (if there are multiple containers)\n\n"
        "4. The final list shows only the documents stored there, with categories.\n"
        "Keep this kit handy and review periodically to ensure accuracy."
    )

    global_physical  = st.session_state.get("global_physical_storage", [])
    global_digital   = st.session_state.get("global_digital_storage", [])
    raw_details      = st.session_state.get("document_details", {})

    # Build a filtered version
    filtered_details = {}
    for doc, details in raw_details.items():
        assigned = details.get("assigned_storage", [])
        fd = {"assigned_storage": assigned}

        # for every other key in details, only keep it if:
        # 1) it’s non-empty, and
        # 2) its key starts with one of the assigned-location prefixes
        for key, val in details.items():
            if key == "assigned_storage" or not val:
                continue
            for loc in assigned:
                prefix = loc.lower().replace(" ", "_").replace("/", "_") + "_"
                if key.startswith(prefix):
                    fd[key] = val
                    break

        filtered_details[doc] = fd

    phys_json    = json.dumps(global_physical, indent=2)
    digi_json    = json.dumps(global_digital, indent=2)
    details_json = json.dumps(filtered_details,  indent=2)

    return f"""
You are an expert at creating clear, action-ready Emergency Document Kits.

Below is the **introductory section** you must include exactly as written at the top of your output (no backticks):

{intro}

You will then be provided with three Python variables in JSON form:

global_physical_storage = {phys_json}  
global_digital_storage  = {digi_json}  
document_details        = {details_json}  

> **Important:** Only use the `document_details` mapping—do **not** pull from any other list.

**Your task**  
1. Output the introductory section verbatim as the first lines.  
2. Group all documents by storage location, showing physical first, then digital.  
3. Within each location, list only the documents actually stored there and include **only** these subsections **when they have data**:
   - **Location details:** the user-provided address or placement info  
   - **Platform:** the service or tool used (if present)  
   - **Access steps:** the emergency steps or authorizations required  
   - **Contents:** if multiple, what each container holds  
4. Sort locations by document count (most → least).  
5. Skip any location with zero documents.  
6. Format the rest in plain Markdown (no code fences), with one top-level heading per location:

   ## <Storage Name> (n documents)

   **Location details:**  
   <location or branch_address>  

   **Platform:**  
   <platform name>  

   **Access steps:**  
   <access_steps>  

   **Contents:**  
   <contents note>  

   **Documents stored:**  
   - **<Document A>** (Category: <Category>)  
   - **<Document B>** (Category: <Category>)

*Omit any subsection line if that field is empty or not applicable.*

Begin your response now.
""".strip()

def bonus_level_runbook_prompt():
    bonus_info = st.session_state.get("bonus_info", {})

    # Filter out empty entries
    filtered_info = {
        key: val
        for key, val in bonus_info.items()
        if val and str(val).strip()
    }
    bonus_json = json.dumps(filtered_info, indent=2)

    prompt = f"""
Append the following **Additional Information** to the home caretaker emergency runbook.  
Use clear headings and actionable bullet points.  
**Only include a section if its data appears in the JSON. Omit any heading with no data.**

Additional Information (JSON):
{bonus_json}

Your task:
1. **Home Maintenance**  
   - Only if `maintenance_tasks` or `appliance_instructions` exist, under a **Home Maintenance** heading list those items.
2. **Home Rules & Preferences**  
   - Only if `house_rules` or `cultural_practices` exist, under a **Home Rules & Preferences** heading summarize them.
3. **Housekeeping & Cleaning**  
   - Only if `housekeeping_instructions` or `cleaning_preferences` exist, under a **Housekeeping & Cleaning** heading provide routines and supply locations.
4. **Entertainment & Technology**  
   - Only if `entertainment_info` or `device_instructions` exist, under an **Entertainment & Technology** heading describe usage and charging steps.

Produce **only** the formatted runbook addition, starting with:

## Additional Information

…followed by the pertinent sections (no blank headings, no extra commentary).
""".strip()

    return prompt
