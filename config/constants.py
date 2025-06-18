UTILITY_KEYS = ["electricity", "natural_gas", "water", "internet"]

SCHEDULE_HEADING_MAP = {
    "<<INSERT_MAIL_HANDLING_SCHEDULE_TABLE>>": "📆 Mail Pickup Schedule",
    "<<INSERT_TRASH_SCHEDULE_TABLE>>": "📆 Trash & Recycling Emptying Schedule",
    "<<INSERT_OUTDOOR_TRASH_SCHEDULE_TABLE>>": "📆 Outdoor Trash & Recycling Schedule",
    "<<INSERT_COMBINED_HOME_SCHEDULE_TABLE>>": "📆 Trash & Mail Combined Schedule",
    "<<INSERT_HOME_SERVICES_SCHEDULE_TABLE>>": "📆 Quality-Oriented Household Services Schedule",
    "<<INSERT_FULL_SCHEDULE_TABLE>>": "📆 Complete Schedule Summary",
}
# Priority order for sorting task types
PRIORITY_ORDER = [
    "Mail Handling",
    "Indoor Trash",
    "Outdoor Trash",
    "Recycling",
    "Compost",
]

# Emoji map for task type labeling
TASK_TYPE_EMOJI = {
    "Mail Handling": "📬",
    "Indoor Trash": "🗑️",
    "Outdoor Trash": "🚛",
    "Recycling": "♻️",
    "Compost": "🌱",
}

SAVED_SECTIONS = [
    {
        "label": "Utility Providers",
        "md_key": "utility_markdown",
        "docx_key": "utility_docx",
        "file_prefix": "utility_providers",
        "icon": "⚡"
    },
    {
        "label": "Emergency Kit",
        "md_key": "emergency_markdown",
        "docx_key": "emergency_docx",
        "file_prefix": "emergency_kit",
        "icon": "🧰"
    },
    {
        "label": "Home Setup",
        "md_key": "home_markdown",
        "docx_key": "home_docx",
        "file_prefix": "home_setup",
        "icon": "🏡"
    },
]
