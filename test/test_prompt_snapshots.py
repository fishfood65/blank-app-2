### 1. Import dependencies and target functions
import json
import streamlit as st
import sys
import unittest  # Imports unittest for test structure.
import os # Imports os to build platform-independent paths.

# Import your prompt function from the prompts_home module
from prompts.prompts_home import (
    query_utility_providers, 
    utilities_emergency_runbook_prompt, 
    emergency_kit_utilities_runbook_prompt,
    mail_trash_runbook_prompt
   )   # <-- Adjust this if needed

# 2. Run Your Tests Normally or With Update Mode
# Normal test run (no overwrite):
# python -m unittest test/test_prompt_snapshots.py
# Snapshot update mode:
# python -m unittest test/test_prompt_snapshots.py --update-snapshots
# If the snapshot file differs, this will overwrite it with the latest output.

# ✅ Best Practice: Safe relative path to root-level "snapshots" directory
SNAPSHOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "snapshots"))
SNAPSHOT_FILE = os.path.join(SNAPSHOT_DIR, "query_utility_providers_snapshot.json")

# Detect update flag (must be set before unittest runs)
UPDATE_SNAPSHOTS = "--update-snapshots" in sys.argv
if UPDATE_SNAPSHOTS:
    sys.argv.remove("--update-snapshots")  # Prevent unittest from choking on unknown flag


class TestPromptSnapshots(unittest.TestCase):

    def setUp(self):
        # Ensure snapshot directory exists
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)

        # ✅ Mock required session state
        st.session_state.clear()
        st.session_state["input_data"] = {
            "Home Basics": [
                {"question": "City", "answer": "Austin"},
                {"question": "ZIP Code", "answer": "73301"},
            ]
        }

    def test_query_utility_providers_prompt_matches_snapshot(self):
        # 1. Generate the current prompt output
        actual_output = query_utility_providers(test_mode=True)

        # 2. ✅ Convert to pretty string
        actual_str = json.dumps(actual_output, indent=2) # Even though your function now returns a dict. You can't compare this dict directly to a plain-text .md snapshot without converting it first — unittest.assertEqual() expects strings for snapshot comparison.

        # 3. ✅ First time: write the snapshot if it doesn't exist
        if not os.path.exists(SNAPSHOT_FILE):
            with open(SNAPSHOT_FILE, "w") as f:
                f.write(actual_str)
            self.skipTest("Snapshot created. Re-run to compare.")

        # 4. Read the expected snapshot content and compare
        with open(SNAPSHOT_FILE, "r") as f:
            expected_prompt = f.read()

        # 5. Compare the current vs. expected snapshot
        self.assertEqual(actual_str.strip(), expected_prompt.strip(), "Prompt does not match snapshot.")

#| Problem               | Cause                                | Fix                                                          |
#| --------------------- | ------------------------------------ | ------------------------------------------------------------ |
#| `ImportError`         | Incorrect import path                | Use relative module path or add `__init__.py` in directories |
#| `ModuleNotFoundError` | Missing `__init__.py` in `home_app/` | Add empty `__init__.py` to mark it as a package              |
#| `actual != expected`  | Snapshot outdated                    | Regenerate with `snapshot_writer.py`                         |

if __name__ == "__main__":
    unittest.main()