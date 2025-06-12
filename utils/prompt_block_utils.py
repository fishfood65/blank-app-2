import streamlit as st

from prompts.templates import (
    wrap_prompt_block, 
    #utility_prompt, 
    #utilities_emergency_prompt_template, 
    emergency_kit_utilities_prompt_template,
    home_caretaker_prompt_template,
    TEMPLATE_MAP
)    
from .data_helpers import get_answer  
import streamlit as st 
from typing import List, Optional, Callable
from prompts.templates import mail_runbook_prompt, trash_runbook_prompt, mail_trash_combined_schedule_prompt

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
    section: Optional[str] = None,
    instructions: Optional[str] = "Write clearly and practically. Avoid hallucinating information.",
    add_heading: bool = True,
    placeholder_only: bool = False
) -> str:
    """
    Constructs a structured LLM prompt block with optional debug logging and section tagging.

    Parameters:
    - title: str — Logical section title (e.g. "Mail Handling")
    - content: str — Raw content or placeholder (e.g. "<<INSERT_SCHEDULE_TABLE>>")
    - section: Optional[str] — Used for logging/debug grouping
    - instructions: Optional[str] — LLM guidance string
    - add_heading: bool — Include a markdown heading in the block
    - placeholder_only: bool — Flag placeholder-only logic

    Returns:
    - str: Final markdown-style LLM prompt block
    """

    debug = st.session_state.get("enable_debug_mode", False)

    if placeholder_only:
        instructions = (
            "⚠️ Do not interpret this placeholder. This will be replaced with a table after LLM generation."
        )

    block = ""
    if add_heading:
        block += f"## {title.strip()}\n\n"

    block += f"{instructions.strip()}\n\n{content.strip()}"

    # ✅ Debug Preview (optional per section)
    if debug:
        section_note = f" in section `{section}`" if section else ""
        st.markdown(f"### 🧱 Prompt Block: `{title}`{section_note}")
        st.code(block, language="markdown")

    return block.strip()

def build_simple_prompt_block(title: str, content: str, section: str = "home") -> str:
    """Returns a basic markdown block with a level-3 heading."""
    return f"### {title}\n\n{content.strip()}"

def generate_all_prompt_blocks(section: str) -> List[str]:
    """
    Collects and wraps prompt content blocks based on the current section.
    Returns a list of structured prompt strings, excluding empty or placeholder-only blocks.
    """
    blocks = []  # Expecting prompt functions to return pre-wrapped blocks using wrap_prompt_block()
    debug = st.session_state.get("enable_debug_mode", False)

    prompt_registry = {
        "utilities": [
            ("Utilities and Emergency Services", lambda: utilities_emergency_runbook_prompt(section=section, debug=debug) or
                fallback_utilities_emergency_prompt(section=section))
        ],
        "emergency_kit": [
            ("Utilities Emergency Services with Kit", lambda: emergency_kit_utilities_runbook_prompt(section=section, debug=debug) or
                fallback_emergency_kit_utilities_runbook_prompt(section=section))
        ],
        "mail_trash": [
            ("Mail Instructions", lambda: mail_runbook_prompt(section=section, debug=debug)),
            ("Trash Instructions", lambda: trash_runbook_prompt(section=section, debug=debug)),
            ("Mail and Trash Combined Schedule", lambda: mail_trash_combined_schedule_prompt(section=section, debug=debug)),
        ],
        "home_security": [
            ("Home Caretaker & Guest Instructions", lambda: home_caretaker_runbook_prompt(section=section)),
        ],
    }

    if section in prompt_registry:
        for title, fn in prompt_registry[section]:
            try:
                block = fn()
                if is_content_meaningful(block):
                    blocks.append(block)
            except Exception as e:
                if debug:
                    st.error(f"❌ Error generating block '{title}': {e}")
    else:
        blocks.append(wrap_prompt_block(
            title=f"⚠️ No prompt defined for section: {section}",
            content="This section currently has no associated prompt logic.",
            section=section
        ))

    # ✅ Final safety check: remove any None or empty blocks
    blocks = [b for b in blocks if isinstance(b, str) and b.strip() and is_content_meaningful(b)]

    return blocks

def is_content_meaningful(content: str) -> bool:
    """
    Returns True if the block has meaningful content (not empty or just headers/placeholders).
    """
    if not content or not content.strip():
        return False

    lines = [line.strip() for line in content.strip().splitlines()]

    # Filter out trivial lines
    non_trivial_lines = [
        line for line in lines
        if line
        and not line.startswith("#")         # Header
        and line not in {"---", "N/A"}       # Dividers / placeholders
        and "⚠️ Not provided" not in line     # Placeholder value
        and not line.startswith("<<INSERT_") # Template tag
    ]

    return bool(non_trivial_lines)

def utilities_emergency_runbook_prompt(section: str = "home", debug: bool = False) -> str:
    city = get_answer(key="City", section=section) or "⚠️ Not provided"
    zip_code = get_answer(key="ZIP Code", section=section) or "⚠️ Not provided"
    internet = get_answer(key="Internet Provider", section=section) or "⚠️ Not provided"

    providers = st.session_state.get("utility_providers", {})
    electricity = providers.get("electricity", "⚠️ Not provided")
    gas = providers.get("natural_gas", "⚠️ Not provided")
    water = providers.get("water", "⚠️ Not provided")

    if debug:
        st.markdown("### 🐞 Debug: Emergency Utilities Prompt")
        st.write("City:", city)
        st.write("ZIP Code:", zip_code)
        st.write("Internet:", internet)
        st.write("Electricity:", electricity)
        st.write("Gas:", gas)
        st.write("Water:", water)

    raw = utilities_emergency_prompt_template(city, zip_code, internet, electricity, gas, water)

    if not raw or not isinstance(raw, str):
        if debug:
            st.error("❌ utilities_emergency_runbook_prompt(): raw content is None or invalid.")
        return None  # 🔴 Prevents crash on `.strip()`

    wrapped = wrap_prompt_block(
        raw,
        title="🏡 Emergency Utilities Overview",
        instructions="Include the heading above in your output. Format using markdown. Return structured emergency info for each utility. Format clearly.",
        debug=debug
    )

    if debug and wrapped is None:
        st.error("❌ utilities_emergency_runbook_prompt(): Wrapped block is None.")

    # ✅ FINAL GUARD: Ensure content is meaningful
    if wrapped and is_content_meaningful(wrapped):
        return wrapped

    return None

def fallback_utilities_emergency_prompt(section: str = "utilities") -> str:
    return utilities_emergency_runbook_prompt(section=section, debug=False)

def emergency_kit_utilities_runbook_prompt(section: str = "utilities", debug: bool = False) -> str:
    city = get_answer(key="City", section="utilities") or ""
    zip_code = get_answer(key="ZIP Code", section="utilities") or ""
    internet = get_answer(key="Internet Provider", section="utilities") or ""

    emergency_kit_status = get_answer(key="Do you have an Emergency Kit?", section="emergency_kit") or "No"
    emergency_kit_location = get_answer(key="Where is (or where will) the Emergency Kit be located?", section="emergency_kit") or ""
    additional_items = get_answer(key="Add any additional emergency kit items not in the list above (comma-separated):", section="emergency_kit") or ""

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
    fire_extinguisher_info = st.session_state.get("fire_extingguisher_info", "")

    selected_md = "".join(f"- {item}\n" for item in selected_items) if selected_items else ""
    missing_md = "".join(f"- {item}\n" for item in not_selected_items) if not_selected_items else ""
    additional_list = [itm.strip() for itm in additional_items.split(",") if itm.strip()]
    additional_md = "".join(f"- {itm}\n" for itm in additional_list) if additional_list else ""

    if emergency_kit_location:
        kit_summary_line = (
            f"Kit is available at {emergency_kit_location}"
            if emergency_kit_status == "Yes"
            else f"Kit is a work in progress and will be located at {emergency_kit_location}"
        )
    else:
        kit_summary_line = ""

    if not any([
        kit_summary_line, selected_md, missing_md, additional_md,
        electricity, gas, water, internet,
        flashlights_info, radio_info, food_water_info, important_docs_info,
        whistle_info, medications_info, mask_info, maps_contacts_info, fire_extinguisher_info
    ]):
        return "⚠️ No emergency or utility data provided."

    if debug:
        st.markdown("### 🐞 Debug: Emergency Utilities + Kit Prompt")
        st.write("City:", city)
        st.write("ZIP Code:", zip_code)
        st.write("Internet:", internet)
        st.write("Electricity:", electricity)
        st.write("Gas:", gas)
        st.write("Water:", water)
        st.write("Emergency Kit Status:", emergency_kit_status)
        st.write("Emergency Kit Location", emergency_kit_location)
        st.write("Additional:", additional_md)
        
    raw = emergency_kit_utilities_prompt_template(
    city=city,
    zip_code=zip_code,
    internet=internet,
    electricity=electricity,
    gas=gas,
    water=water,
    kit_summary_line=kit_summary_line,
    selected_md=selected_md,
    missing_md=missing_md,
    additional_md=additional_md,
    flashlights_info=flashlights_info,
    radio_info=radio_info,
    food_water_info=food_water_info,
    important_docs_info=important_docs_info,
    whistle_info=whistle_info,
    medications_info=medications_info,
    mask_info=mask_info,
    maps_contacts_info=maps_contacts_info,
    fire_extinguisher_info=fire_extinguisher_info
    )

    return wrap_prompt_block(
        raw,
        title="⛑️ Emergency Utilities and Preparedness",
        instructions=("You are a structured assistant writing a house emergency runbook in Markdown.\n\n"
            "⚠️ Do not re-order or summarize sections.\n"
            "🧱 Keep each utility and emergency kit section distinct and complete."),
        debug=debug
    )


def fallback_emergency_kit_utilities_runbook_prompt(section: str = "home") -> str:
    return emergency_kit_utilities_runbook_prompt(section=section, debug=False)

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
