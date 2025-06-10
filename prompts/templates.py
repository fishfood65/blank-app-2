import streamlit as st

def wrap_prompt_block(
    content: str,
    *,
    title: str = "",
    instructions: str = "",
    section: str = None,
    debug: bool = False,
    for_llm: bool = False
) -> str:
    """
    Wraps a content block with optional title, guidance, and section metadata.
    Useful for both LLM prompts and export-ready documentation.
    """
    if debug:
        st.markdown("### 🧱 DEBUG WRAP: Raw Content")
        st.code(content, language="markdown")

    # 🛡️ Normalize input types
    if isinstance(content, list):
        content = "\n\n".join(str(item) for item in content)
    elif isinstance(content, dict):
        content = "\n".join(f"- **{k}**: {v}" for k, v in content.items())
    elif not isinstance(content, str):
        content = str(content)

    block = []

    # Optional section marker (can also be used for debugging or internal tracking)
    if section:
        block.append(f"<!-- Section: {section} -->")

    # Markdown heading
    if title:
        block.append(f"# {title}")

    # Optional instructions (only if non-empty)
    if instructions and instructions.strip():
        if for_llm:
            block.append(f"_Instructions: {instructions}_")
        else:
            block.append(f"**Instructions:** {instructions}")
        block.append("")

    # ✅ Append main content
    block.append(content.strip())

    # Join and return
    final = "\n\n".join(block).strip()

    if debug:
        st.markdown(f"### 🧱 DEBUG WRAP: Final Wrapped Block ('{title}')")
        st.code(final, language="markdown")

    return final

def emergency_kit_utilities_prompt_template(
    city: str,
    zip_code: str,
    internet: str,
    electricity: str,
    gas: str,
    water: str,
    kit_summary_line: str,
    selected_md: str,
    missing_md: str,
    additional_md: str,
    flashlights_info: str,
    radio_info: str,
    food_water_info: str,
    important_docs_info: str,
    whistle_info: str,
    medications_info: str,
    mask_info: str,
    maps_contacts_info: str,
    fire_extinguisher_info: str,
    *,
    section: str = "emergency_kit",
    debug: bool = False
) -> str:

    def render_recommended(*items):
        return "".join(f"- {i.strip()}\n" for i in items if i and i.strip())

    body = f"""
City: {city}
Zip: {zip_code}
Internet Provider: {internet}
Electricity Provider: {electricity}
Natural Gas Provider: {gas}
Water Provider: {water}

Please retrieve:
- Company Description
- Contact Info
- Emergency Steps

---

# 🧰 Emergency Kit Summary

## Emergency Kit:
- {kit_summary_line or "_(no summary provided)_"}

## Kit Inventory:
{selected_md or "_(none selected)_"}

## ⚠️ Missing Kit Items (consider adding):
{missing_md or "_(none missing)_"}

## Additional User-Added Items:
{additional_md or "_(none added)_"}

---

# 🏡 Emergency Utilities Overview

## ⚡ Electricity – {electricity}
### Power Outage Response Guide:
- Company Description
- Contact Info
- Emergency Steps
### Recommended Kit Items:
{render_recommended(flashlights_info, radio_info, food_water_info, important_docs_info)}

---

## 🔥 Natural Gas – {gas}
### Gas Leak Response Guide:
- Company Description
- Contact Info
- Emergency Steps
### Recommended Kit Items:
{render_recommended(whistle_info, important_docs_info, flashlights_info, fire_extinguisher_info, maps_contacts_info)}

---

## 💧 Water – {water}
### Water Outage or Leak Guide:
- Company Description
- Contact Info
- Emergency Steps
### Recommended Kit Items:
{render_recommended(food_water_info, medications_info, mask_info, important_docs_info)}

---

## 🌐 Internet – {internet}
### Internet Outage Response Guide:
- Company Description
- Contact Info
- Emergency Steps
### Recommended Kit Items:
{render_recommended(radio_info, maps_contacts_info, important_docs_info)}

---

Ensure the run book is clearly formatted using Markdown, with bold headers and bullet points. Use ⚠️ to highlight missing kit items.
""".strip()


def utilities_emergency_prompt_template(
    city: str,
    zip_code: str,
    internet: str,
    electricity: str,
    gas: str,
    water: str,
    *,
    section: str = "home",
    debug: bool = False
) -> str:
    """
    Generates a structured LLM-safe markdown prompt block for utility and emergency provider instructions.
    """

    body = f"""
City: {city}
Zip: {zip_code}
Internet Provider: {internet}
Electricity Provider: {electricity}
Natural Gas Provider: {gas}
Water Provider: {water}

Please retrieve:
- Company Description
- Contact Info
- Emergency Steps
---

## ⚡ Electricity – {electricity}
### Power Outage Response Guide:
- Company Description
- Contact Info
- Emergency Steps

---

## 🔥 Natural Gas – {gas}
### Gas Leak Response Guide:
- Company Description
- Contact Info
- Emergency Steps

---

## 💧 Water – {water}
### Water Outage or Leak Guide:
- Company Description
- Contact Info
- Emergency Steps

---

## 🌐 Internet – {internet}
### Internet Outage Response Guide:
- Company Description
- Contact Info
- Emergency Steps

---

Ensure the runbook is clearly formatted using Markdown, with bold headers and bullet points.
""".strip()

def utility_provider_lookup_prompt(city: str, zip_code: str) -> str:
    return f"""
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

def utility_prompt(city: str, zip_code: str, internet: str, electricity: str, gas: str, water: str) -> str:
    return f"""
City: {city}
Zip: {zip_code}
Internet Provider: {internet}
Electricity Provider: {electricity}
Natural Gas Provider: {gas}
Water Provider: {water}

Please retrieve:
- Company Description
- Contact Info
- Emergency Steps
""".strip()

TEMPLATE_MAP = {
    "mail": {
        "title": "📬 Mail Handling Instructions",
        "instructions": "Use the provided instructions for where, when, and how to collect and store mail and packages. DO NOT invent details or insert a schedule table. Leave all placeholders untouched.",
        "question_order": [
            "📍 Mailbox Location",
            "🔑 Mailbox Key (Optional)",
            "📆 Mail Pick-Up Schedule",
            "📥 What to Do with the Mail",
            "📦 Packages"
        ]
    },
    "trash_handling": {
        "title": "🗑️ Trash and Recycling Instructions",
        "instructions": "Use the provided instructions and images to explain indoor and outdoor trash routines clearly. Leave all schedule placeholders untouched.",
        "question_order": [
            "🧴 Kitchen Garbage Bin",
            "♻️ Indoor Recycling Bin(s)",
            "🧃 Indoor Compost or Green Waste",
            "🧼 Bathroom Trash Bin",
            "🪑 Other Room Trash Bins",
            "🏞️ How often and when is Outdoor Garbage and Recycling Collected?",
            "📍 Where are the trash, recycling, and compost bins stored outside?",
            "🏷️ How are the outdoor bins marked?",
            "📋 Stuff to know before putting recycling or compost in the bins?",
            "🏠 Is a Single-family home?",
            "🛻 When and where should garbage, recycling, and compost bins be placed for pickup?",
            "🗑️ When and where should garbage, recycling, and compost bins be brought back in after pickup?",
            "📇 Waste Management Company Name",
        ]
    },
    "home_security": {
        "title": "🔐 Home Security & Technology",
        "instructions": "Summarize home security access, alarm procedures, emergency entry, and home communication details.",
        "question_order": [
            "Security Company Name",
            "Security Company Phone Number",
            "Instructions to arm/disarm system",
            "Steps if a security alert is triggered",
            "Indoor cameras/monitoring details and activation",
            "Emergency access instructions & storage location",
            "Where is Wi-Fi network name/password stored?",
            "Guest network details & password sharing method",
            "Home phone setup & call-handling instructions"
        ]
    },
    "cleaning": {
        "title": "🧹 Cleaning Service Instructions",
        "instructions": "Summarize how and when cleaning services arrive and how access and post-cleaning routines are handled.",
        "question_order": [
            "Company Name",
            "Phone Number",
            "Frequency",
            "Day of the Week",
            "Access Method",
            "Post-Service Procedures",
            "Crew Identity Verification"
        ]
    },
    "gardening": {
        "title": "🌿 Gardening Service Instructions",
        "instructions": "Describe how often gardening service occurs and any preparation or follow-up needed.",
        "question_order": [
            "Company Name",
            "Phone Number",
            "Frequency",
            "Day of the Week",
            "Access Method",
            "Post-Service Procedures",
            "Crew Identity Verification"
        ]
    },
    "pool": {
        "title": "🏊 Pool Maintenance Instructions",
        "instructions": "Summarize when and how pool maintenance takes place, and note any access or safety details.",
        "question_order": [
            "Company Name",
            "Phone Number",
            "Frequency",
            "Day of the Week",
            "Access Method",
            "Post-Service Procedures",
            "Crew Identity Verification"
        ]
    }
}

def home_caretaker_prompt_template(data: dict) -> str:
    """
    Generates a markdown-formatted Home Caretaker prompt.
    Skips any section or subsection where all values are "⚠️ Not provided".
    """
    def section_block(title: str, fields: dict) -> str:
        visible_items = {k: v for k, v in fields.items() if v != "⚠️ Not provided"}
        if not visible_items:
            return ""
        lines = [f"#### {title}"]
        for k, v in visible_items.items():
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)

    blocks = []

    for section_title, section_fields in data.items():
        if isinstance(section_fields, dict):
            block = section_block(section_title, section_fields)
            if block:
                blocks.append(block)

    return "\n\n---\n\n".join(blocks).strip()

def mail_runbook_prompt(section: str = "mail_trash", debug: bool = False) -> str:
    """
    Builds a structured mail handling prompt block without LLM.

    Args:
        section (str): Section name used to retrieve input_data.
        debug (bool): Whether to show debug output in Streamlit.

    Returns:
        str: A markdown-formatted prompt block.
    """
    input_data = st.session_state.get("input_data", {})
    mail_entries = input_data.get(section, []) or input_data.get("Mail & Packages", [])

    mail_info = {entry["question"]: entry["answer"] for entry in mail_entries}

    def safe_line(label, value):
        if value and str(value).strip().lower() not in ["no", "⚠️ not provided", ""]:
            return f"- **{label}**: {value}"
        return None

    mail_block = "\n".join(filter(None, [
        safe_line("The mailbox for collecting mail and small packages", mail_info.get("📍 Mailbox Location")),
        safe_line("Mailbox key location", mail_info.get("🔑 Mailbox Key (Optional)")),
        safe_line("Pick up Mail", mail_info.get("📆 Mail Pick-Up Schedule")),
        safe_line("After picking up the mail, please", mail_info.get("📥 What to Do with the Mail")),
        safe_line("Where should non-mail packages (e.g., Amazon, UPS) be picked up?", mail_info.get("🚚 Pick up oversized packages at")),
        safe_line("Where should all packages be stored after delivery?", mail_info.get("📦 Place packages after pickup")),
    ]))

    markdown = f"""

{mail_block}

### 📆 Mail Pickup Schedule

<<INSERT_MAIL_HANDLING_SCHEDULE_TABLE>>
""".strip()

    return wrap_prompt_block(
        content=markdown,
        title="📬 Mail Handling Instructions",
        instructions="",
        debug=True,
        section=section  # Pass explicitly to wrap_prompt_block
    )

def trash_runbook_prompt(section: str = "trash_handling", debug: bool = False) -> str:
    """
    Builds a structured trash handling prompt block without LLM.
    Returns an empty string if no content is provided.
    """
    import streamlit as st

    input_data = st.session_state.get("input_data", {})
    trash_entries = input_data.get(section, [])

    trash_info = {entry["question"]: entry["answer"] for entry in trash_entries}

    if debug:
        st.markdown("### 🧱 trash_info contents")
        st.json(trash_info)

    def safe_line(label, value):
        if value and str(value).strip().lower() not in ["no", "⚠️ not provided", ""]:
            return f"- **{label}**: {value.strip()}"
        return None

    # Indoor Trash Instructions
    indoor_block = "\n".join(filter(None, [
        safe_line("Kitchen Trash", trash_info.get("🧴 Kitchen Garbage Bin")),
        safe_line("Indoor Recycling", trash_info.get("♻️ Indoor Recycling Bin(s)")),
        safe_line("Compost / Green Waste", trash_info.get("🧃 Indoor Compost or Green Waste")),
        safe_line("Bathroom Trash", trash_info.get("🧼 Bathroom Trash Bin")),
        safe_line("Other Room Trash", trash_info.get("🪑 Other Room Trash Bins")),
    ]))

    # Outdoor Trash Instructions with Single-Family Merge
    outdoor_lines = list(filter(None, [
        safe_line("Location of outside trash, recycling, and compost bins", trash_info.get("📍 Bin Storage Location")),
        safe_line("Outdoor bins are marked as follows", trash_info.get("🏷️ How are bins marked?")),
        safe_line("Important steps to follow before putting recycling or compost in the bins", trash_info.get("📋 What to know before recycling or composting")),
    ]))

    # 🚩 Conditional additions for single-family homes
    flag_raw = trash_info.get("🏠 Is a Single-family home?", "")
    single_family_disposal = str(flag_raw).strip().lower() in ["yes", "true"]
    if single_family_disposal:
        outdoor_lines.extend([
            "",  # Blank line for spacing
            "**This is a single-family home.**"
        ])
        outdoor_lines.extend(filter(None, [
            safe_line("When and where to place bins for pickup", trash_info.get("🛻 When and where should garbage, recycling, and compost bins be placed for pickup?")),
            safe_line("When and where to bring bins back in", trash_info.get("🗑️ When and where should bins be brought back in?")),
        ]))

    outdoor_block = "\n".join(outdoor_lines)

    # Waste Management Contact
    wm_block = "\n".join(filter(None, [
        safe_line("Company", trash_info.get("🏢 Waste Management Company Name")),
        safe_line("Phone", trash_info.get("📞 Contact Phone Number")),
        safe_line("When to Contact", trash_info.get("📝 When to Contact")),
    ]))

    if debug:
        st.markdown("### 🧱 DEBUG Trash Prompt Blocks")
        st.write("Indoor block present?", bool(indoor_block))
        st.write("Outdoor block present?", bool(outdoor_block))
        st.write("WM block present?", bool(wm_block))

    # 🧹 Build prompt content only if there is meaningful info
    if not any([indoor_block, outdoor_block, wm_block]):
        return ""  # ✂️ Return nothing if all content is empty

    sections = []

    if indoor_block:
        sections.extend(["### Indoor Trash", indoor_block])
    if outdoor_block:
        sections.extend(["### Outdoor Bins", outdoor_block])
    if wm_block:
        sections.extend(["### Waste Management Contact", wm_block])

    sections.extend(["### 📆 Trash and Recycling Pickup Schedule", "<<INSERT_TRASH_SCHEDULE_TABLE>>"])

    markdown = "\n\n".join(sections)

    return wrap_prompt_block(
        content=markdown,
        title="🗑️ Trash, Recycling and Compost Instructions",
        instructions="",  # Omit instructions completely
        debug=debug,
        section=section
    )

def mail_trash_combined_schedule_prompt(section: str = "mail_trash", debug: bool = False) -> str:
    markdown = (
        "<<INSERT_MAIL_TRASH_SCHEDULE_TABLE>>"
    )

    return wrap_prompt_block(
        content=markdown,
        title="Combined Mail and Trash Task Schedule", # Omit if you're already setting a heading inside content
        instructions=None,
        debug=debug,
        section=section  # Pass explicitly to wrap_prompt_block
    )
