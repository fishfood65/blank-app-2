# utils/prompt_block_utils.py

from prompts.templates import (
    wrap_prompt_block, 
    utility_prompt, 
    utilities_emergency_prompt_template, 
    emergency_kit_utilities_prompt_template
)    
from .input_tracker import get_answer  
import streamlit as st 
from typing import List, Optional

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

def generate_all_prompt_blocks(section: str) -> List[str]:
    """
    Collects and wraps prompt content blocks based on the current section.
    Returns a list of structured prompt strings.
    """
    blocks = []

    if section == "home":
        blocks.append(build_prompt_block("Utilities and Emergency Services", utilities_emergency_runbook_prompt()))
    
    elif section == "mail_trash_handling":
        blocks.append(build_prompt_block("Utilities Emergency Services with Kit", emergency_kit_utilities_runbook_prompt()))
        blocks.append(build_prompt_block("Mail and Trash Instructions", mail_trash_runbook_prompt()))
    
    elif section == "home_security":
        blocks.append(build_prompt_block("Utilities Emergency Services with Kit", emergency_kit_utilities_runbook_prompt()))
        blocks.append(build_prompt_block("Mail and Trash Instructions", mail_trash_runbook_prompt()))
        blocks.append(build_prompt_block("Caretaker Instructions", home_caretaker_runbook_prompt()))
    
    elif section == "emergency_kit":
        blocks.append(build_prompt_block("Utilities Emergency Services with Kit", emergency_kit_utilities_runbook_prompt()))
    
    elif section == "emergency_kit_critical_documents":
        blocks.append(build_prompt_block("Important Documents Checklist", emergency_kit_document_prompt()))
    
    elif section == "bonus_level":
        blocks.append(build_prompt_block("Additional Home Instructions", bonus_level_runbook_prompt()))

    else:
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

### test writing code
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