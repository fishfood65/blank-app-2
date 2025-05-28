### 1. Import dependencies and target functions
import unittest  # Imports unittest for test structure.
import os # Imports os to build platform-independent paths.

# Import your prompt function from the prompts_home module
from prompts.prompts_home import (
    query_utility_providers, 
    utilities_emergency_runbook_prompt, 
    emergency_kit_utilities_runbook_prompt,
    mail_trash_runbook_prompt
   )   # <-- Adjust this if needed

# Path to the snapshot file
SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "..", "snapshots")
SNAPSHOT_FILE = os.path.join(SNAPSHOT_DIR, "home_prompt.md")

class TestPromptSnapshots(unittest.TestCase):

    def test_query_utility_providers_prompt_matches_snapshot(self):
        # 1. Generate the current prompt output
        actual_prompt = query_utility_providers()

        # 2. Check if the snapshot file exists
        if not os.path.exists(SNAPSHOT_FILE):
            # 3. If it doesn't, create the snapshot
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            with open(SNAPSHOT_FILE, "w", encoding="utf-8") as f:
                f.write(actual_prompt)
            self.fail("Snapshot file created. Please re-run the test to validate content.")

        # 4. Read the expected snapshot content
        with open(SNAPSHOT_FILE, "r", encoding="utf-8") as f:
            expected_prompt = f.read()

        # 5. Compare the current vs. expected snapshot
        self.assertEqual(actual_prompt.strip(), expected_prompt.strip(), "Prompt does not match snapshot.")

#| Problem               | Cause                                | Fix                                                          |
#| --------------------- | ------------------------------------ | ------------------------------------------------------------ |
#| `ImportError`         | Incorrect import path                | Use relative module path or add `__init__.py` in directories |
#| `ModuleNotFoundError` | Missing `__init__.py` in `home_app/` | Add empty `__init__.py` to mark it as a package              |
#| `actual != expected`  | Snapshot outdated                    | Regenerate with `snapshot_writer.py`                         |

if __name__ == "__main__":
    unittest.main()