# prompts/templates.py

def wrap_prompt_block(content: str, *, title: str = "", instructions: str = "", debug: bool = False) -> str:
    """Wraps a content block with optional title and LLM-specific guidance."""
    block = []
    if title:
        block.append(f"# {title}")
    if instructions:
        block.append(f"_Instructions: {instructions}_")
    block.append(content)
    final = "\n\n".join(block).strip()
    if debug:
        import streamlit as st
        st.markdown("### 🧱 Debug: Wrapped Prompt Block")
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
    maps_contacts_info: str
) -> str:
    def render_recommended(*items):
        return "".join(f"- {i}\n" for i in items if i and i.strip())

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
---

# 🧰 Emergency Kit Summary

## Emergency Kit:
{kit_summary_line}

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
{render_recommended(whistle_info, important_docs_info, flashlights_info)}

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

Ensure the run book is clearly formatted using Markdown, with bold headers and bullet points. Use ⚠️ to highlight missing kit items.
""".strip()

def utilities_emergency_prompt_template(city: str, zip_code: str, internet: str, electricity: str, gas: str, water: str) -> str:
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

Ensure the run book is clearly formatted using Markdown, with bold headers and bullet points. 
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
