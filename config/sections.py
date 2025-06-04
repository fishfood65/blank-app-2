# config/sections.py

SECTION_METADATA = {
    "home": {
        "label": "🏠 Home Setup",
        "level": 1,
        "enabled": True,
    },
    "mail_trash_handling": {
        "label": "📬 Mail & Trash",
        "level": 3,
        "subsections": {
            "mail": "📬 Mail",
            "trash_handling": "🗑️ Trash"
            },
        "enabled": True,
    },
    "home_security": {
        "label": "🔐 Home Security",
        "level": 4,
        "enabled": True,
    },
    "emergency_kit": {
        "label": "🧰 Emergency Kit",
        "level": 2,
        "enabled": True,
    },
    "emergency_kit_critical_documents": {
        "label": "📑 Critical Documents",
        "level": 5,
        "enabled": True,
    },
    "bonus_level": {
        "label": "🎁 Bonus Level",
        "level": None,  # Not part of core progression
        "enabled": True,
    },
}

LLM_SECTIONS = {"home", "emergency_kit"}