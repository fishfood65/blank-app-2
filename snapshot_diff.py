import difflib
import argparse
from pathlib import Path

def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.readlines()

def show_diff(file1, file2):
    lines1 = read_file(file1)
    lines2 = read_file(file2)

    diff = difflib.unified_diff(
        lines1, lines2,
        fromfile=file1,
        tofile=file2,
        lineterm=''
    )

    print("\n".join(diff) or "✅ No differences found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two snapshot .md files")
    parser.add_argument("file1", type=Path, help="First snapshot file")
    parser.add_argument("file2", type=Path, help="Second snapshot file")

    args = parser.parse_args()

    if not args.file1.exists() or not args.file2.exists():
        print("❌ One or both files do not exist.")
    else:
        show_diff(str(args.file1), str(args.file2))

### How to use:
# python snapshot_diff.py snapshots/home_prompt_2025-05-27T09-30-00.md snapshots/home_prompt_latest.md

### How it Works
# How It Works
# Uses difflib.unified_diff() to generate a diff view (like git diff)

# Clearly marks:

#- lines removed

# + lines added

# Context lines with no prefix

# Helps verify prompt logic changes or LLM formatting changes over time