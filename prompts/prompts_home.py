from utils.common_helpers import (
    check_home_progress,
    extract_all_trash_tasks_grouped, 
    extract_grouped_mail_task, 
    generate_flat_home_schedule_markdown,
)
from utils.input_tracker import (
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

# Generate the AI prompt
api_key = os.getenv("MISTRAL_TOKEN")
client = Mistral(api_key=api_key)

if not api_key:
    api_key = st.text_input("Enter your Mistral API key:", type="password")

if api_key:
    st.success("API key successfully loaded.")
else:
   st.error("API key is not set.")
   
#### Prompts Here #####

def query_utility_providers(test_mode=False):
    """
    Queries Mistral AI for public utility providers based on city and ZIP code 
    stored in st.session_state. Stores and returns results in session state.

    Returns a dict of utility providers based on city and ZIP.
    In normal mode, reads from session_state; in test mode, accepts inputs directly.
    """
    # ✅ Retrieve inputs safely first
    city = get_answer("City", "Home Basics")
    zip_code = get_answer("ZIP Code", "Home Basics")

    # ✅ Test-mode shortcut (for snapshot testing or dev)
    if test_mode:
        return {
            "electricity": "Austin Energy",
            "natural_gas": "Atmos Energy",
            "water": "Austin Water"
        }
    
    prompt = f"""
You are a reliable assistant helping users prepare emergency documentation. 
Given the city: {city} and ZIP code: {zip_code}, list the **primary public utility provider companies** for the following:

1. Electricity
2. Natural Gas
3. Water

For each, provide only the company name. Format your response like this:

Electricity Provider: <company name>
Natural Gas Provider: <company name>
Water Provider: <company name>
""".strip()

    try:
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[UserMessage(content=prompt)],
            max_tokens=1500,
            temperature=0.5,
        )
        content = response.choices[0].message.content
    except Exception as e:
        st.error(f"Error querying Mistral API: {str(e)}")
        content = ""

    # Extract and log into input_data and interaction_log
    results = extract_and_log_providers(content)

    # Also store in session_state for correction access
    st.session_state["utility_providers"] = results
    
    return results

def fetch_utility_providers():
    results = query_utility_providers()
    st.session_state["utility_providers"] = results
    return results

def utilities_emergency_runbook_prompt():
    """
    Builds the prompt for utilities and emergency services.
    Pulls user-input data from structured input_data and LLM results from session.
    """

    # From user-input section ("Home Basics")
    city = get_answer("City", "Home Basics") or ""
    zip_code = get_answer("ZIP Code", "Home Basics") or ""
    internet_provider_name = get_answer("Internet Provider", "Home Basics") or ""

    # From LLM or corrected values stored in session
    results = st.session_state.get("utility_providers", {})
    electricity_provider_name = results.get("electricity", "")
    natural_gas_provider_name = results.get("natural_gas", "")
    water_provider_name = results.get("water", "")

    # Build your prompt using these values

    return f"""
You are an expert assistant generating a city-specific Emergency Utility Overview. First, search the internet for up-to-date local utility providers and their emergency contact information. Then, compose a comprehensive, easy-to-follow guide customized for residents of City: {city}, Zip Code: {zip_code}.

Using the following provider information:
Internet Provider: {internet_provider_name}
Electricity Provider: {electricity_provider_name}
Natural Gas Provider: {natural_gas_provider_name}
Water Provider: {water_provider_name}

For each provider, retrieve:
- Company Description
- Customer Service Phone Number
- Customer Service Address (if available)
- Official Website
- Emergency Contact Numbers (specific to outages, leaks, service disruptions)
- Steps to report issues

---


# 📕 Emergency Run Book

## ⚡ 1. Electricity – {electricity_provider_name}
- Provider Description
- Customer Service
- Website
- Emergency Contact

### Power Outage Response Guide:
- Steps to follow
- How to report
- Safety precautions

---
## 🔥 2. Natural Gas – {natural_gas_provider_name}
- Provider Description
- Customer Service
- Website
- Emergency Contact

### Gas Leak Response Guide:**
- Signs and precautions
- How to evacuate
- How to report

---
## 💧 3. Water – {water_provider_name}
- Provider Description
- Customer Service
- Website
- Emergency Contact

### Water Outage or Leak Guide:**
- Detection steps
- Shutoff procedure

---
## 🌐 4. Internet – {internet_provider_name}
- Provider Description
- Customer Service
- Website
- Emergency Contact

### Internet Outage Response Guide:
- Troubleshooting
- Reporting
- Staying informed
---

Ensure the run book is clearly formatted using Markdown, with bold headers and bullet points. Use ⚠️ to highlight missing kit items.
""".strip()

#### Emergency Kit + Utilities Prompt ####

def emergency_kit_utilities_runbook_prompt():
    """
    Generate a markdown-formatted emergency runbook prompt using user answers and utility data.
    """
    city = get_answer("City", "Home Basics") or ""
    zip_code = get_answer("ZIP Code", "Home Basics") or ""
    internet_provider_name = get_answer("Internet Provider", "Home Basics") or ""
    emergency_kit_status = get_answer("Do you have an Emergency Kit?", "Emergency Kit") or "No"
    emergency_kit_location = get_answer("Where is (or where will) the Emergency Kit be located?", "Emergency Kit") or ""
    additional_items = get_answer("Add any additional emergency kit items not in the list above (comma-separated):", "Emergency Kit") or ""

    selected_items = st.session_state.get("homeowner_kit_stock", [])
    not_selected_items = st.session_state.get("not_selected_items", [])

    results = st.session_state.get("utility_providers", {})
    electricity_provider_name = results.get("electricity", "")
    natural_gas_provider_name = results.get("natural_gas", "")
    water_provider_name = results.get("water", "")

    flashlights_info = st.session_state.get("flashlights_info", "")
    radio_info = st.session_state.get("radio_info", "")
    food_water_info = st.session_state.get("food_water_info", "")
    important_docs_info = st.session_state.get("important_docs_info", "")
    whistle_info = st.session_state.get("whistle_info", "")
    medications_info = st.session_state.get("medications_info", "")
    mask_info = st.session_state.get("mask_info", "")
    maps_contacts_info = st.session_state.get("maps_contacts_info", "")

    selected_md = "".join(f"- {item}\n" for item in selected_items)
    missing_md = "".join(f"- {item}\n" for item in not_selected_items)
    additional_list = [itm.strip() for itm in additional_items.split(",") if itm.strip()]
    additional_md = "".join(f"- {itm}\n" for itm in additional_list)
    
    kit_summary_line = (
    f"Kit is available at {emergency_kit_location}"
    if emergency_kit_status == "Yes"
    else f"Kit is a work in progress and will be located at {emergency_kit_location}"
    )

    def render_recommended(*items):
        return "".join(f"- {i}\n" for i in items if i and i.strip())

    return f"""
You are an expert assistant generating a city-specific Emergency Preparedness Run Book. First, search the internet for up-to-date local utility providers and their emergency contact information. Then, compose a comprehensive, easy-to-follow guide customized for residents of City: {city}, Zip Code: {zip_code}.

Start by identifying the following utility/service providers for the specified location:
- Internet Provider Name
- Electricity Provider Name
- Natural Gas Provider Name
- Water Provider Name

For each provider, retrieve:
- Company Description
- Customer Service Phone Number
- Customer Service Address (if available)
- Official Website
- Emergency Contact Numbers (specific to outages, leaks, service disruptions)
- Steps to report issues
---

# 🧰 Emergency Kit Summary

## Emergency Location:
{kit_summary_line}

## Kit Inventory:  
{selected_md or "_(none selected)_"}  
## ⚠️ Missing Kit Items (consider adding): 
{missing_md or "_(none missing)_"}  

## Additional User-Added Items: 
{additional_md or "_(none added)_"}  

---

# 📕 Emergency Run Book

## ⚡ 1. Electricity – {electricity_provider_name}
- Provider Description
- Customer Service
- Website
- Emergency Contact

### Power Outage Response Guide:
- Steps to follow
- How to report
- Safety precautions
##### Recommended Kit Items:
{render_recommended(flashlights_info, radio_info, food_water_info, important_docs_info)}

---

## 🔥 2. Natural Gas – {natural_gas_provider_name}
- Provider Description
- Customer Service
- Website
- Emergency Contact

### Gas Leak Response Guide:
- Signs and precautions
- How to evacuate
- How to report
##### Recommended Kit Items:
{render_recommended(whistle_info, important_docs_info, flashlights_info)}

---

## 💧 3. Water – {water_provider_name}
- Provider Description
- Customer Service
- Website
- Emergency Contact

### Water Outage or Leak Guide:
- Detection steps
- Shutoff procedure
#### Recommended Kit Items:
{render_recommended(food_water_info, medications_info, mask_info, important_docs_info)}

---

## 🌐 4. Internet – {internet_provider_name}
- Provider Description
- Customer Service
- Website
- Emergency Contact

### Internet Outage Response Guide:
- Troubleshooting
- Reporting
- Staying informed
#### Recommended Kit Items:
{render_recommended(radio_info, maps_contacts_info, important_docs_info)}

Ensure the run book is clearly formatted using Markdown, with bold headers and bullet points. Use ⚠️ to highlight missing kit items.
""".strip()

#### Mail + Trash Prompt ####
def mail_trash_runbook_prompt(debug_key="trash_info_debug"):
    input_data = st.session_state.get("input_data", {})
    st.write("🧪 [DEBUG] input_data keys:", list(input_data.keys()))

    # Check for merged view first (if maybe_generate_prompt merged subsections)
    merged_entries = st.session_state.get("input_data", {}).get("mail_trash_handling", [])
    st.write("📬 [DEBUG] mail_trash_handling entries seen by prompt:", merged_entries)

    if merged_entries:
        st.write("🧩 [DEBUG] Merged Entries:", merged_entries)
        mail_entries = [
            e for e in merged_entries
            if "mail" in str(e.get("section", "")).lower()
        ]
        trash_entries = [
            e for e in merged_entries
            if "trash" in str(e.get("section", "")).lower()
        ]

        st.write("📬 [DEBUG] Filtered Mail Entries:", mail_entries)
        st.write("🗑️ [DEBUG] Filtered Trash Entries:", trash_entries)

    else:
        st.warning("⚠️ No merged entries found; falling back to section-based retrieval.")
        input_data = st.session_state.get("input_data", {})
        mail_entries = input_data.get("mail") or input_data.get("Mail & Packages", [])
        trash_entries = input_data.get("Trash Handling", []) or input_data.get("trash_handling", [])
        st.write("📦 [DEBUG] Fallback Mail Entries:", mail_entries)
        st.write("🗑️ [DEBUG] Fallback Trash Entries:", trash_entries)


    # Debugging view
    #if st.sidebar.checkbox("🔍 Show Mail/Trash Debug"):
        #st.sidebar.markdown("**Mail Entries Found:**")
       # st.sidebar.code("\n".join(e["question"] for e in mail_entries), language="text")
       # st.sidebar.markdown("**Trash Entries Found:**")
       # st.sidebar.code("\n".join(e["question"] for e in trash_entries), language="text")

    # Flatten into dicts for easy access
    mail_info = {entry["question"]: entry["answer"] for entry in mail_entries}
    trash_info = {entry["question"]: entry["answer"] for entry in trash_entries}

    def safe_line(label, value):
        if value and str(value).strip().lower() != "no":
            return f"- **{label}**: {value}"
        return None

    def safe_yes_no(label, flag, detail_label, detail_value):
        if flag:
            return f"- **{label}**: Yes\n  - **{detail_label}**: {detail_value or 'N/A'}"
        return ""
    #### Debug check to see what is saved into session state
    if st.sidebar.checkbox("🔍 Show Trash Info Debug", key=debug_key):
        st.sidebar.write("🧾 `trash_info` Keys:")
        st.sidebar.code("\n".join(sorted(trash_info.keys())), language="text")

    # --- MAIL SECTION ---
    mail_lines = [
        safe_line("Mailbox Location", mail_info.get("📍 Mailbox Location")),
        safe_line("Mailbox Key Info", mail_info.get("🔑 Mailbox Key (Optional)")),
        safe_line("Pick-Up Schedule", mail_info.get("📆 Mail Pick-Up Schedule")),
        safe_line("Mail Sorting Instructions", mail_info.get("📥 What to Do with the Mail")),
        safe_line("Delivery Packages", mail_info.get("📦 Packages")),
    ]
    mail_block = f"## 📬 Mail Handling Instructions\n\n" + "\n".join(filter(None, mail_lines)) if mail_lines else ""

    # --- INDOOR TRASH ---
    indoor_lines = [
        safe_line("Kitchen Trash", trash_info.get("Kitchen Trash Bin Location, Emptying Schedule and Replacement Trash Bags")),
        safe_line("Bathroom Trash", trash_info.get("Bathroom Trash Bin Emptying Schedule and Replacement Trash Bags")),
        safe_line("Other Rooms Trash", trash_info.get("Other Room Trash Bin Emptying Schedule and Replacement Trash Bags")),
    ]
    indoor_block = f"### Indoor Trash\n" + "\n".join(filter(None, indoor_lines)) if indoor_lines else ""

    # --- OUTDOOR BINS ---
    outdoor_lines = [
        safe_line("Please take the bins", trash_info.get("Instructions for Placing and Returning Outdoor Bins")),
        safe_line("Bins Description", trash_info.get("What the Outdoor Trash Bins Look Like")),
        safe_line("Location", trash_info.get("Specific Location or Instructions for Outdoor Bins")),
    ]

    outdoor_image_tags = []
    if "trash_images" in st.session_state:
        for label in ["Outdoor Bin Image", "Recycling Bin Image"]:
            if st.session_state.trash_images.get(label):
                filename = st.session_state.trash_images[label]
                outdoor_image_tags.append(f'<img src="{filename}" alt="{label}" width="300"/>')

    outdoor_block = f"### Outdoor Bins\n" + "\n".join(filter(None, outdoor_lines + outdoor_image_tags)) if (outdoor_lines or outdoor_image_tags) else ""

    # --- COLLECTION SCHEDULE ---
    collection_lines = [
        safe_line("Garbage Pickup", f"{trash_info.get('Garbage Pickup Day', '')}, {trash_info.get('Garbage Pickup Time', '')}".strip(", ")),
        safe_line("Recycling Pickup", f"{trash_info.get('Recycling Pickup Day', '')}, {trash_info.get('Recycling Pickup Time', '')}".strip(", ")),
    ]
    collection_block = f"### Collection Schedule\n" + "\n".join(filter(None, collection_lines)) if collection_lines else ""

    # --- COMPOSTING ---
    composting_used = trash_info.get("Compost Used", "").strip().lower() == "yes"
    composting_text = safe_yes_no(
        "Composting Used",
        composting_used,
        "Compost Instructions",
        trash_info.get("Compost Instructions")
    )
    composting_block = f"### Composting\n{composting_text}" if composting_text else ""

    # --- COMMON DISPOSAL ---
    common_disposal_used = trash_info.get("Common Disposal Used", "").strip().lower() == "yes"
    common_disposal_text = safe_yes_no(
        "Common Disposal Area Used",
        common_disposal_used,
        "Instructions",
        trash_info.get("Common Disposal Area Instructions")
    )
    common_disposal_block = f"### Common Disposal Area\n{common_disposal_text}" if common_disposal_text else ""

    # --- WASTE MANAGEMENT ---
    wm_lines = list(filter(None, [
    safe_line("Company Name", trash_info.get("Waste Management Company Name")),
    safe_line("Phone", trash_info.get("Contact Phone Number")),
    safe_line("When to Contact", trash_info.get("When to Contact")),
    ]))
    wm_block = f"### Waste Management Contact\n" + "\n".join(wm_lines) if wm_lines else ""


    # --- TRASH SECTION COMBINED ---
    trash_blocks = "\n\n".join(filter(None, [
        indoor_block,
        outdoor_block,
        collection_block,
        composting_block,
        common_disposal_block,
        wm_block
    ]))
    trash_main = f"## 🗑️ Trash & Recycling Instructions\n\n{trash_blocks}" if trash_blocks else ""

    # --- FINAL SCHEDULE PLACEHOLDER ---
    schedule_block = "\n\n## 📆 Mail & Trash Pickup Schedule\n\n<<INSERT_SCHEDULE_TABLE>>"

    # --- FINAL PROMPT OUTPUT ---
    return f"""
You are an expert assistant generating a Mail and Waste Management Run Book. Compose a comprehensive, easy-to-follow guide for house sitters and people watching the house when occupants are out of town. For any values set to No please omit those lines.

# 📕 Mail Handling and Waste Management Instructions

{mail_block}

---

{trash_main}

---

{schedule_block}
""".strip()


#### Security and Services Prompt ####

def home_caretaker_runbook_prompt():
    csi = st.session_state.get("convenience_seeker_info", {})
    roi = st.session_state.get("rent_own_info", {})
    hsi = st.session_state.get("home_security_info", {})

    return f"""
You are a helpful assistant tasked with generating a professional, detailed, and easy-to-follow Home Caretaker & Guest Runbook. The goal is to ensure a smooth experience for caretakers or guests while the home occupants are away. 

Please use the following information provided by the homeowner to write a clear and structured guide:
Please omit any headings that return "Not provided" for all the values below it.
Please omit any sub-headings that return "Not provided" for all the values below it.
Please omit any lines that return "Not provided" or "N/A".
Please omit any sub-headings that return "Not provided" or "N/A" for all the values below it.
Please don't add a title to the runbook.

### 📕 Security and Services Guide

#### 🔐 Home Security & Technology
- Security Company Name: {hsi.get("home_security_comp_name", "Not provided")}
- Security Company Number: {hsi.get("home_security_comp_num", "Not provided")}
- Arming/Disarming Instructions: {hsi.get("arm_disarm_instructions", "Not provided")}
- If Alert is Triggered: {hsi.get("security_alert_steps", "Not provided")}
- Indoor Camera Notes: {hsi.get("indoor_cameras", "Not provided")}
- Emergency Access Instructions: {hsi.get("access_emergency", "Not provided")}
- Wi-Fi Info Location: {hsi.get("wifi_network_name", "Not provided")}
- Guest Wi-Fi Access: {hsi.get("wifi_guests", "Not provided")}
- Landline/VOIP Notes: {hsi.get("landline_voip", "Not provided")}

---

#### 🧹 Cleaning Service Instructions
- Company Name: {csi.get("cleaning_name", "Not provided")}
- Phone Number: {csi.get("cleaning_number", "Not provided")}
- Schedule: {csi.get("cleaning_schedule", "Not provided")}
- Access Method: {csi.get("cleaning_access", "Not provided")}
- Post-Cleaning Procedures: {csi.get("cleaning_finish_steps", "Not provided")}
- Crew Identity Verification: {csi.get("cleaning_identity_confirmation", "Not provided")}

---

#### 🌿 Gardening & Landscape Service Instructions
- Company Name: {csi.get("gardening_name", "Not provided")}
- Phone Number: {csi.get("gardening_number", "Not provided")}
- Schedule: {csi.get("gardening_schedule", "Not provided")}
- Access Method: {csi.get("gardening_access", "Not provided")}
- Post-Service Procedures: {csi.get("gardening_finish_steps", "Not provided")}
- Crew Identity Verification: {csi.get("gardening_identity_confirmation", "Not provided")}

---

#### 🏊 Pool Maintenance Instructions
- Company Name: {csi.get("pool_name", "Not provided")}
- Phone Number: {csi.get("pool_number", "Not provided")}
- Schedule: {csi.get("pool_schedule", "Not provided")}
- Access Method: {csi.get("pool_access", "Not provided")}
- Post-Service Procedures: {csi.get("pool_finish_steps", "Not provided")}
- Crew Identity Verification: {csi.get("pool_identity_confirmation", "Not provided")}

---

#### 🏢 Property Management (Renters or HOA)
- Company Name: {roi.get("property_management_name", "Not provided")}
- Phone Number: {roi.get("property_management_number", "Not provided")}
- Email: {roi.get("property_management_email", "Not provided")}
- When to Contact: {roi.get("property_management_description", "Not provided")}

---

#### 🛠️ Service Contacts (For Homeowners)
**Handyman**
- Name: {roi.get("handyman_name", "N/A")}
- Phone: {roi.get("handyman_number", "N/A")}
- When to Contact: {roi.get("handyman_description", "N/A")}

**Electrician**
- Name: {roi.get("electrician_name", "N/A")}
- Phone: {roi.get("electrician_number", "N/A")}
- When to Contact: {roi.get("electrician_description", "N/A")}

**Exterminator**
- Name: {roi.get("exterminator_name", "N/A")}
- Phone: {roi.get("exterminator_number", "N/A")}
- When to Contact: {roi.get("exterminator_description", "N/A")}

**Plumber**
- Name: {roi.get("plumber_name", "N/A")}
- Phone: {roi.get("plumber_number", "N/A")}
- When to Contact: {roi.get("plumber_description", "N/A")}

---

Please format the runbook clearly with headers and bullet points. Use “⚠️ Not provided” as a flag for incomplete or missing info that should be reviewed.
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
