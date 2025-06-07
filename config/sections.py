
SECTION_METADATA: dict[str, dict] = {
    "home": {
        "label": "🏠 Home Setup",
        "icon": "🏠",
        "level": 1,
        "visible": True,
        "enabled": True,
        "requires_llm": True,
        "progress_key": "home",
    },
    "emergency_kit": {
        "label": "🧰 Emergency Kit",
        "icon": "🧰",
        "level": 2,
        "visible": True,
        "enabled": True,
        "requires_llm": True,
        "progress_key": "emergency_kit",
    },
    "mail_trash": {
        "label": "📬 Mail & Trash",
        "icon": "📬",
        "level": 3,
        "visible": True,
        "enabled": True,
        "requires_llm": True,
        "progress_key": "mail_trash",
    },
    "home_security": {
        "label": "🔐 Home Security",
        "icon": "🔐",
        "level": 4,
        "visible": True,
        "enabled": True,
        "requires_llm": False,
        "progress_key": "home_security",
    },
    "emergency_kit_critical_documents": {
        "label": "📑 Critical Documents",
        "icon": "📑",
        "level": 5,
        "visible": True,
        "enabled": True,
        "requires_llm": False,
        "progress_key": "emergency_kit_critical_documents",
    },
    "bonus_level": {
        "label": "🎁 Bonus Level",
        "icon": "🎁",
        "level": None,  # Optional/extra
        "visible": True,
        "enabled": True,
        "requires_llm": False,
        "progress_key": "bonus_level",
    },
    "runbook_date_range": {
        "label": "📅 Runbook Date Selection",
        "icon": "📅",
        "level": None,  # Optional/extra
        "visible": False, # Helps hide it from sidebar rendering logic
        "enabled": True,
        "requires_llm": False,
        "progress_key": "bonus_level",
    },
}

# Set of sections that are meant to be processed through an LLM
LLM_SECTIONS = {
    key for key, meta in SECTION_METADATA.items()
    if meta.get("requires_llm", False)
}

def check_home_progress(progress_dict):
    """
    Checks overall progress across all levels using SECTION_METADATA.
    Returns a completion percentage and sorted list of completed level keys.
    """
    total_levels = len(progress_dict)
    completed = [k for k, v in progress_dict.items() if v]

    if total_levels == 0:
        return 0, []

    # Sort completed levels by numeric level if available
    def level_sort_key(k):
        return SECTION_METADATA.get(k, {}).get("level", 99)  # fallback if level missing

    completed_sorted = sorted(completed, key=level_sort_key)
    percent_complete = int((len(completed) / total_levels) * 100)

    return percent_complete, completed_sorted

def get_all_sections(include_hidden: bool = False):
    """
    It gives you a single point of reference to retrieve the list of all known sections 
    (from SECTION_METADATA) — optionally filtering out hidden/internal ones.

    Usage:
    from config.sections import get_all_sections

    all_sections = get_all_sections()  # Just visible sections
    all_sections_debug = get_all_sections(include_hidden=True)  # All sections
    """

    if include_hidden:
        return list(SECTION_METADATA.keys())
    return [key for key, meta in SECTION_METADATA.items() if meta.get("visible", True)]