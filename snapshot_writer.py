import os
# Import your prompt function from the prompts_home module
from prompts.prompts_home import (
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

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

for filename, func in SNAPSHOT_FUNCTIONS.items():
    try:
        snapshot_path = os.path.join(SNAPSHOT_DIR, filename)
        result = func()
        with open(snapshot_path, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"✅ Snapshot written: {filename}")
    except Exception as e:
        print(f"❌ Failed to generate snapshot for {filename}: {e}")