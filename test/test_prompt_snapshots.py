### 1. Import dependencies and target functions
import unittest  # Imports unittest for test structure.
import os # Imports os to build platform-independent paths.

# Replace this with the actual function you're testing
from home_app_05_23_modified import generate_mail_prompt # Imports the actual prompt generator (generate_mail_prompt) you're testing.

### 2. Define the path to your snapshot directory
SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots") # This computes the full path to test/snapshots/, regardless of where the test is run from.

### 3. Create a test class using unittest.TestCase
class TestPromptSnapshots(unittest.TestCase): # Defines a test suite. All test methods inside must start with test_.

### 4. Read snapshot content from file
    def read_snapshot(self, filename): # A helper method that loads a snapshot file (e.g., mail_prompt.md) and returns its text.
        path = os.path.join(SNAPSHOT_DIR, filename)
        with open(path, encoding="utf-8") as f:
            return f.read().strip() .strip() # ensures we ignore leading/trailing whitespace differences.

### 5. Compare function output to snapshot
    def test_mail_prompt_matches_snapshot(self):
        result = generate_mail_prompt().strip() # Calls your prompt generator.
        expected = self.read_snapshot("mail_prompt.md") # Loads the expected output from the corresponding snapshot.
        self.assertEqual(result, expected) # Asserts that they are exactly equal.

    def test_trash_prompt_matches_snapshot(self):
        # Replace with your trash prompt generator function
        from home_app_05_23_modified import generate_trash_prompt
        result = generate_trash_prompt().strip()
        expected = self.read_snapshot("trash_prompt.md")
        self.assertEqual(result, expected)
