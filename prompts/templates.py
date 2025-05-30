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

# prompts/templates.py

def utilities_emergency_prompt_template() -> str:
    return """
You are an expert assistant generating a city-specific Emergency Utility Overview. First, search the internet for up-to-date local utility providers and their emergency contact information. Then, compose a comprehensive, easy-to-follow guide customized for residents of City: {city}, Zip Code: {zip_code}.

Using the following provider information:
Internet Provider: {internet_provider}
Electricity Provider: {electricity_provider}
Natural Gas Provider: {natural_gas_provider}
Water Provider: {water_provider}

For each provider, retrieve:
- Company Description
- Customer Service Phone Number
- Customer Service Address (if available)
- Official Website
- Emergency Contact Numbers (specific to outages, leaks, service disruptions)
- Steps to report issues

---

# 📕 Emergency Run Book

## ⚡ 1. Electricity – {electricity_provider}
### Power Outage Response Guide:
- Steps to follow
- How to report
- Safety precautions

---

## 🔥 2. Natural Gas – {natural_gas_provider}
### Gas Leak Response Guide:
- Signs and precautions
- How to evacuate
- How to report

---

## 💧 3. Water – {water_provider}
### Water Outage or Leak Guide:
- Detection steps
- Shutoff procedure

---

## 🌐 4. Internet – {internet_provider}
### Internet Outage Response Guide:
- Troubleshooting
- Reporting
- Staying informed

---

Ensure the run book is clearly formatted using Markdown, with bold headers and bullet points. Use ⚠️ to highlight missing kit items.
""".strip()
