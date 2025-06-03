# utils/prompt_block_utils.py

from prompts.templates import (
    wrap_prompt_block, 
    utility_prompt, 
    utilities_emergency_prompt_template, 
    emergency_kit_utilities_prompt_template,
    mail_prompt_template,
    trash_prompt_template,
    home_caretaker_prompt_template,
    TEMPLATE_MAP
)    
from .input_tracker import get_answer  
import streamlit as st 
from typing import List, Optional

SECTION_ALIASES = {
    "Mail & Packages": "mail",
    "Quality-Oriented Household Services.Cleaning": "cleaning",
    "Quality-Oriented Household Services.Gardening": "gardening",
    "Quality-Oriented Household Services.Pool": "pool",
    "Home Security": "home_security",
    "Trash Handling": "trash_handling"
}

def build_prompt_block(
    title: str,
    content: str,
    *,
    instructions: Optional[str] = None,
    add_heading: bool = True,
    placeholder_only: bool = False
) -> str:
    """
    Constructs a structured LLM prompt block with optional debug logging.

    Parameters:
    - title: str — Logical section title (e.g. "Mail Handling")
    - content: str — Raw content or placeholder (e.g. "<<INSERT_SCHEDULE_TABLE>>")
    - instructions: Optional[str] — Optional instruction line for the LLM
    - add_heading: bool — If True, includes a markdown heading in the block
    - placeholder_only: bool — If True, LLM is told not to interpret placeholder

    Returns:
    - str: A markdown-compatible prompt block
    """

    debug = st.session_state.get("enable_debug_mode", False)

    if placeholder_only:
        instructions = (
            "⚠️ Do not interpret this placeholder. This will be replaced with a table after LLM generation."
        )
    else:
        instructions = instructions or "Write clearly and practically. Avoid hallucinating information."

    block = ""
    if add_heading:
        block += f"## {title.strip()}\n\n"

    block += f"{instructions.strip()}\n\n{content.strip()}"

    # ✅ Debug preview
    if debug:
        st.markdown(f"### 🧱 Prompt Block: `{title}`")
        st.code(block, language="markdown")

    return block.strip()

def generate_prompt_section(section: str, *, for_llm: bool = False, debug: bool = False) -> str:
    input_data = st.session_state.get("input_data", {})

    # 🔁 Resolve alias to canonical section key
    canonical_section = SECTION_ALIASES.get(section, section)
    section_data = input_data.get(section, []) or input_data.get(canonical_section, [])
    template = template_map.get(canonical_section, {})

    if not section_data:
        return f"### ⚠️ No data provided for {canonical_section.title()} section."

    question_order = template.get("question_order", [])
    title = template.get("title", canonical_section.title())
    instructions = template.get("instructions", "")

    # Build answer lookup
    answers = {item["question"]: item["answer"] for item in section_data}

    def safe_line(label):
        value = answers.get(label, "").strip()
        if value and value.lower() not in ["no", "⚠️ not provided", ""]:
            return f"- **{label}**: {value}"
        return None

    lines = [safe_line(q) for q in question_order]
    bullet_block = "\n".join(filter(None, lines))

    # Check for matching schedule DataFrame
    schedule_key = f"{canonical_section}_schedule_df"
    schedule_block = ""
    if schedule_key in st.session_state and not st.session_state[schedule_key].empty:
        schedule_block = f"\n\n### 📆 {title} Schedule\n\n<<{canonical_section.upper()}_SCHEDULE_PLACEHOLDER>>"

    full_content = f"{bullet_block}{schedule_block}".strip()

    return wrap_prompt_block(
        content=full_content,
        title=title,
        instructions=instructions,
        for_llm=for_llm,
        debug=debug
    )

def is_content_meaningful(content: str) -> bool:
    """
    Returns True if the block has meaningful content (not empty or all ⚠️ lines).
    """
    if not content.strip():
        return False
    lines = [line.strip() for line in content.strip().splitlines()]
    # Filter out lines that are just placeholders or warnings
    non_placeholder_lines = [
        line for line in lines
        if "⚠️ Not provided" not in line and line not in {"---", "", "N/A"}
    ]
    return bool(non_placeholder_lines)


def generate_all_prompt_blocks(section: str) -> List[str]:
    """
    Collects and wraps prompt content blocks based on the current section.
    Returns a list of structured prompt strings, excluding empty or placeholder-only blocks.
    """
    blocks = []

    def maybe_add_block(title: str, content: str):
        if is_content_meaningful(content):
            blocks.append(build_prompt_block(title, content))

    # LLM blocks — always included
    if section in ["home"]:
        maybe_add_block("Utilities and Emergency Services", utilities_emergency_runbook_prompt())

    if section in ["emergency_kit"]:
        maybe_add_block("Utilities Emergency Services with Kit", emergency_kit_utilities_runbook_prompt())

    # Deterministic blocks from structured data
    if section in ["mail_trash_handling"]:
        maybe_add_block("Mail Instructions", generate_prompt_section("Mail & Packages", for_llm=False))
        maybe_add_block("Trash Instructions", generate_prompt_section("Trash Handling", for_llm=False))

    if section == "home_security":
        maybe_add_block("Home Caretaker & Guest Instructions", home_caretaker_runbook_prompt())

    if section == "emergency_kit_critical_documents":
        maybe_add_block("Important Documents Checklist", emergency_kit_document_prompt())

    if section == "bonus_level":
        maybe_add_block("Additional Home Instructions", bonus_level_runbook_prompt())

    if not blocks:
        blocks.append(build_prompt_block(f"⚠️ No prompt defined for section: {section}", ""))

    return blocks

#OR def generate_all_prompt_blocks(section: str) -> List[str]: ###Combine Sections at Prompt-Building Time
# Use a synthetic or virtual section name like "home_security_overview" or "mail_trash_handling" that aggregates blocks from multiple existing sub-sections (e.g. "mail", "trash_handling").
#   if section == "mail_trash_handling":
#        return (
#            emergency_kit_utilities_runbook_prompt() +
#            mail_trash_runbook_prompt(debug_key="trash_info_debug_preview")
 #       )
 #   elif section == "home_security":
 #       return (
 #           emergency_kit_utilities_runbook_prompt() +
#            mail_trash_runbook_prompt() +
 #           home_caretaker_runbook_prompt()
#        )
#    # fallback
 #   elif section == "home":
 ##       return utilities_emergency_runbook_prompt()
#    else:
#        return [f"# ⚠️ No prompt available for section: {section}"]


def utilities_emergency_runbook_prompt(debug: bool = False) -> str:
    city = get_answer("City", "Home Basics") or ""
    zip_code = get_answer("ZIP Code", "Home Basics") or ""
    internet = get_answer("Internet Provider", "Home Basics") or ""
    providers = st.session_state.get("utility_providers", {})
    electricity = providers.get("electricity", "")
    gas = providers.get("natural_gas", "")
    water = providers.get("water", "")

    raw = utilities_emergency_prompt_template(city, zip_code, internet, electricity, gas, water)

    return wrap_prompt_block(
        raw,
        title="🏡 Emergency Utilities Overview",
        instructions="Include the heading above in your output. Format using markdown. Return structured emergency info for each utility. Format clearly.",
        debug=debug
    )

def emergency_kit_utilities_runbook_prompt(debug: bool = False) -> str:
    city = get_answer("City", "Home Basics") or ""
    zip_code = get_answer("ZIP Code", "Home Basics") or ""
    internet = get_answer("Internet Provider", "Home Basics") or ""
    emergency_kit_status = get_answer("Do you have an Emergency Kit?", "Emergency Kit") or "No"
    emergency_kit_location = get_answer("Where is (or where will) the Emergency Kit be located?", "Emergency Kit") or ""
    additional_items = get_answer("Add any additional emergency kit items not in the list above (comma-separated):", "Emergency Kit") or ""
    
    selected_items = st.session_state.get("homeowner_kit_stock", [])
    not_selected_items = st.session_state.get("not_selected_items", [])
    providers = st.session_state.get("utility_providers", {})
    
    electricity = providers.get("electricity", "")
    gas = providers.get("natural_gas", "")
    water = providers.get("water", "")
    
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

    raw = emergency_kit_utilities_prompt_template(
        city, zip_code, internet, electricity, gas, water,
        kit_summary_line, selected_md, missing_md, additional_md,
        flashlights_info, radio_info, food_water_info, important_docs_info,
        whistle_info, medications_info, mask_info, maps_contacts_info
    ) 
    return wrap_prompt_block(
        raw,
        title="🧰 Emergency Utilities and Preparedness",
        instructions="List out the Emergency Kit Summary details before summarize utility and emergency kit setup using bullet points.",
        debug=debug
    )

def mail_runbook_prompt(debug: bool = False) -> str:
    """
    Builds mail handling prompt from structured mail inputs.
    """
    section = "mail"
    input_data = st.session_state.get("input_data", {})
    mail_entries = input_data.get(section, []) or input_data.get("Mail & Packages", [])

    mail_info = {entry["question"]: entry["answer"] for entry in mail_entries}

    def safe_line(label, value):
        if value and str(value).strip().lower() != "no":
            return f"- **{label}**: {value}"
        return None

    mail_block = "\n".join(filter(None, [
        safe_line("Where to collect mail and small packages", mail_info.get("📍 Mailbox Location")),
        safe_line("Mailbox key location", mail_info.get("🔑 Mailbox Key (Optional)")),
        safe_line("Mail pick-up schedule", mail_info.get("📆 Mail Pick-Up Schedule")),
        safe_line("After mail pickup, what to do with it", mail_info.get("📥 What to Do with the Mail")),
        safe_line("Where to pick up and store all packages", mail_info.get("📦 Packages")),
    ]))

    raw = mail_prompt_template(mail_block)

    return wrap_prompt_block(
        content=raw,
        title="📬 Mail Handling Instructions",
        instructions="Use the provided instructions for where, when, and how to collect and store mail and packages. DO NOT invent details, add advice, or insert a schedule table. Leave all placeholders untouched",
        debug=debug
    )

def trash_runbook_prompt(debug: bool = False) -> list[str]:
    """
    Builds trash handling prompts as structured blocks with image embedding.
    The actual schedule table is inserted later via <<INSERT_TRASH_SCHEDULE_TABLE>>.
    """
    section = "trash_handling"
    input_data = st.session_state.get("input_data", {})
    trash_entries = input_data.get(section, [])

    trash_info = {entry["question"]: entry["answer"] for entry in trash_entries}

    def safe_line(label, value):
        return f"- **{label}**: {value}" if value and str(value).strip().lower() != "no" else None

    def safe_yes_no(label, flag, detail_label, detail_value):
        if flag:
            return f"- **{label}**: Yes\n  - **{detail_label}**: {detail_value or 'N/A'}"
        return ""

    # Indoor Trash Instructions
    indoor_block = "\n".join(filter(None, [
        safe_line("Kitchen Trash", trash_info.get("🧴 Kitchen Garbage Bin")),
        safe_line("Indoor Recycling", trash_info.get("♻️ Indoor Recycling Bin(s)")),
        safe_line("Compost / Green Waste", trash_info.get("🧃 Indoor Compost or Green Waste")),
        safe_line("Bathroom Trash", trash_info.get("🧼 Bathroom Trash Bin")),
        safe_line("Other Room Trash", trash_info.get("🪑 Other Room Trash Bins")),
    ]))

    # Outdoor Trash Instructions
    outdoor_block = "\n".join(filter(None, [
        safe_line("Outdoor Collection Schedule", trash_info.get("How often and when is Outdoor Garbage and Recycling Collected?")),
        safe_line("Bin Appearance", trash_info.get("What the Outdoor Trash Bins Look Like")),
        safe_line("Placement Instructions", trash_info.get("Specific Location or Instructions for Emptying Outdoor Bins")),
    ]))

    outdoor_image_tags = []
    if "trash_images" in st.session_state:
        for label in ["Outdoor Bin Image", "Recycling Bin Image"]:
            img_bytes = st.session_state.trash_images.get(label)
            if img_bytes:
                b64_img = base64.b64encode(img_bytes).decode("utf-8")
                mime = "image/png"
                outdoor_image_tags.append(f'<img src="data:{mime};base64,{b64_img}" alt="{label}" width="300"/>')

    outdoor_block += "\n" + "\n".join(outdoor_image_tags)

    # Common Disposal Area
    common_used = trash_info.get("Is there a common disposal area?", False)
    common_block = ""
    if common_used:
        common_instr = trash_info.get("Instructions for Common Disposal Area", "")
        common_block = safe_yes_no("Common Disposal Area", True, "Instructions", common_instr)

    # Waste Management Contact
    wm_block = "\n".join(filter(None, [
        safe_line("Company", trash_info.get("Waste Management Company Name")),
        safe_line("Phone", trash_info.get("Contact Phone Number")),
        safe_line("When to Contact", trash_info.get("When to Contact")),
    ]))

    raw = trash_prompt_template(
        indoor_block=indoor_block,
        outdoor_block=outdoor_block,
        common_disposal_block=common_block,
        wm_block=wm_block,
    )

    return wrap_prompt_block(
        content=raw,
        title="🗑️ Trash and Recycling Instructions",
        instructions="Use the provided information and images for clarity.",
        debug=debug
    )

def home_caretaker_runbook_prompt() -> str:

    categories = [
        ("🔐 Home Security & Technology", [
            ("Security Company Name", "Home Security"),
            ("Security Company Phone Number", "Home Security"),
            ("Instructions to arm/disarm system", "Home Security"),
            ("Steps if a security alert is triggered", "Home Security"),
            ("Indoor cameras/monitoring details and activation", "Home Security"),
            ("Emergency access instructions & storage location", "Home Security"),
            ("Where is Wi-Fi network name/password stored?", "Home Security"),
            ("Guest network details & password sharing method", "Home Security"),
            ("Home phone setup & call-handling instructions", "Home Security"),
        ]),
        ("🧹 Cleaning Service Instructions", [
            ("Company Name", "Quality-Oriented Household Services.Cleaning"),
            ("Phone Number", "Quality-Oriented Household Services.Cleaning"),
            ("Frequency", "Quality-Oriented Household Services.Cleaning"),
            ("Day of the Week", "Quality-Oriented Household Services.Cleaning"),
            ("Access Method", "Quality-Oriented Household Services.Cleaning"),
            ("Post-Service Procedures", "Quality-Oriented Household Services.Cleaning"),
            ("Crew Identity Verification", "Quality-Oriented Household Services.Cleaning"),
        ]),
        # Add Gardening, Pool, Rent/Own as needed...
    ]

    prompt_data = {}
    for section_label, fields in categories:
        group = {}
        for label, section in fields:
            group[label] = get_answer(label, section) or "⚠️ Not provided"
        prompt_data[section_label] = group

    return home_caretaker_prompt_template(prompt_data)
