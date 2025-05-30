import os
from datetime import datetime
# Import your prompt function from the prompts_home module
from prompts.llm_queries import (
    query_utility_providers, 
    utilities_emergency_runbook_prompt, 
    emergency_kit_utilities_runbook_prompt,
    mail_trash_runbook_prompt
   )  

# Map each prompt-generating function to a snapshot filename
SNAPSHOT_FUNCTIONS = {
    "query_utility_prompt.md": query_utility_providers,
    "utilities_prompt.md": utilities_emergency_runbook_prompt,
    "emergency_kit_prompt.md": emergency_kit_utilities_runbook_prompt,
    "mail_trash_prompt.md": mail_trash_runbook_prompt
}

# Create a versioned filename using current timestamp
timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

for name, func in SNAPSHOT_FUNCTIONS.items():
    try:
        # Versioned file
        versioned_filename = f"{name}_{timestamp}.md"
        versioned_path = os.path.join(SNAPSHOT_DIR, versioned_filename)

        # Optional: latest copy for comparison convenience
        latest_path = os.path.join(SNAPSHOT_DIR, f"{name}_latest.md")

        result = func()

        # Write both files
        with open(versioned_path, "w", encoding="utf-8") as vf:
            vf.write(result)

        with open(latest_path, "w", encoding="utf-8") as lf:
            lf.write(result)

        print(f"✅ Saved {versioned_filename} and {name}_latest.md")

    except Exception as e:
        print(f"❌ Failed to write snapshot for {name}: {e}")