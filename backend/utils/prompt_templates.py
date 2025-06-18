import re

def generate_single_provider_prompt(
    utility: str,
    city: str,
    zip_code: str,
    internet: str = ""
) -> str:
    """
    Generates a markdown-formatted prompt for one specific utility type.
    """
    utility_labels = {
        "electricity": "⚡ Electricity",
        "natural_gas": "🔥 Natural Gas",
        "water": "💧 Water",
        "internet": "🌐 Internet"
    }

    utility_tips = {
        "electricity": "Focus on **power outages**, avoiding **downed lines**, and **safe generator use**.",
        "natural_gas": "Focus on **leak detection**, avoiding **ignition sources**, and **safe evacuation**.",
        "water": "Focus on **shutting off water**, identifying **leaks**, and ensuring **safe drinking water**.",
        "internet": "Focus on **router status**, **contacting support**, and using **mobile data** if needed."
    }

    label = utility_labels.get(utility, utility.title())
    tip = utility_tips.get(utility, "")
    internet_note = f"Internet Provider (user provided): {internet}" if utility == "internet" and internet else ""

    heading = f"""You are a reliable assistant generating emergency utility documentation for a household in:
- City: {city}
- ZIP Code: {zip_code}"""
    if internet_note:
        heading += f"\n- {internet_note}"

    step_request = f"""
Please identify the **main {label} provider** for this location and return the following:

- **Company Name**
- **Brief Description**
- **Contact Info**:
    - Phone
    - Website
    - Email
    - Address

- **Emergency Steps**: Provide **exactly five** clear, practical safety steps that a **homeowner or house sitter** can take during a {label.lower()} emergency:
    - Do **not** include technical, professional, or hazardous instructions (e.g., repairing lines, shutting off mains).
    - Use **simple markdown bullet points**.
    - {tip}

If the provider cannot be found, return `"⚠️ Not Available"` for each missing field.
""".strip()

    non_emergency_note = """
Also include an optional **Non-Emergency Tips** section:
- Provide 1–3 brief tips on billing, service status, or support.
- Mark this section clearly.
""".strip()

    markdown_format = f"""
Use this markdown format:

## {label} – <Company Name>
**Description:** <short description>  
**Contact Info:**  
- **Phone:** <phone number>  
- **Website:** <URL>  
- **Email:** <email address>  
- **Address:** <full address>  

**Emergency Steps:**  
- Step 1  
- Step 2  
- Step 3  
- Step 4  
- Step 5

**Non-Emergency Tips:**  
- Tip 1  
- Tip 2  
""".strip()

    return f"{heading}\n\n{step_request}\n\n{non_emergency_note}\n\n{markdown_format}"


def wrap_with_claude_style_formatting(user_prompt: str) -> str:
    """
    Wraps the raw user prompt with tone and formatting instructions
    tailored for GPT-4o to behave more like Claude.
    """
    return f"""
You are a helpful assistant writing clear, practical, and human-friendly emergency documentation.

Use the following tone and formatting guidelines:
- Speak in **natural, concise language**.
- Use **markdown** for structure: `##` for sections, `**bold**` for labels, `-` for bullets.
- Be direct but **not robotic**. Write like you’re preparing a guide for a trusted house sitter.
- When citing external information (like web search results), **use markdown links**:  
  Example: [pge.com](https://www.pge.com/safety)  
- NEVER make up provider names or steps — if unknown, say so explicitly.

Now continue with the user's instruction below.

---

{user_prompt.strip()}
""".strip()

def clean_md_artifacts(text: str) -> str:
    """
    Removes markdown formatting characters from the text.
    """
    return re.sub(r"[*`_~]", "", text).strip()

def parse_utility_block(block: str) -> dict:
    """
    Extracts structured fields from a markdown-formatted utility provider block,
    with markdown artifacts removed for all values.
    """
    def extract_and_clean(pattern: str, multiline: bool = False) -> str:
        flags = re.DOTALL if multiline else 0
        match = re.search(pattern, block, flags)
        return clean_md_artifacts(match.group(1).strip()) if match else ""

    return {
        "name": extract_and_clean(r"## [^\u2013\-]+[\u2013\-]\s*(.*)"),
        "description": extract_and_clean(r"\*\*Description:\*\* (.*)"),
        "contact_phone": extract_and_clean(r"\*\*Phone:\*\* (.*)"),
        "contact_website": extract_and_clean(r"\*\*Website:\*\* (.*)"),
        "contact_address": extract_and_clean(r"\*\*Address:\*\* (.*)"),
        "emergency_steps": extract_and_clean(
            r"\*\*Emergency Steps:\*\*\s*((?:.|\n)*?)(?=\n## |\Z)", multiline=True
        ) or "⚠️ Emergency steps not provided.",
        "non_emergency_tips": extract_and_clean(
            r"\*\*Non-Emergency Tips:\*\*\s*((?:.|\n)*?)(?=\n## |\Z)", multiline=True
        ) or ""
    }

def normalize_provider_fields(parsed: dict) -> dict:
    """
    Applies fallback replacement, deduplication, and formatting cleanup.
    """
    def normalize_val(val: str) -> str:
        if not val or val.strip().lower() in ["n/a", "not found", "⚠️ not available"]:
            return "⚠️ Not Available"
        return val.strip()

    for key in parsed:
        if isinstance(parsed[key], str):
            parsed[key] = normalize_val(parsed[key])

    return parsed

def is_usable_provider_response(data: dict, placeholder: str = "⚠️ Not Available") -> bool:
    """
    Returns True if at least 2 core fields are non-empty and not common placeholders.
    """
    core_fields = ["name", "description", "contact_phone", "contact_address"]
    bad_values = {placeholder.lower(), "not available", "n/a", "none", ""}

    valid_fields = [
        v for k, v in data.items()
        if k in core_fields and isinstance(v, str) and v.strip().lower() not in bad_values
    ]

    return len(valid_fields) >= 2