import os
from home_app_05_23_modified import generate_mail_prompt

SNAPSHOT_DIR = os.path.join("test", "snapshots") # This creates a subdirectory snapshots inside the test/ folder.
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

### This writes the actual prompt output into that file (mail_prompt.md).
with open(os.path.join(SNAPSHOT_DIR, "mail_prompt.md"), "w", encoding="utf-8") as f:
    f.write(generate_mail_prompt().strip())

print("✅ mail_prompt.md written.")
