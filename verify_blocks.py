import json

def is_content_meaningful(content: str) -> bool:
    """
    Returns True if the block has meaningful content (not empty or just headers/placeholders).
    """
    if not content or not content.strip():
        return False

    lines = [line.strip() for line in content.strip().splitlines()]
    non_placeholder_lines = [
        line for line in lines
        if line and not line.startswith("#") and line not in {"---", "⚠️ Not provided", "N/A"}
        and "<<INSERT_" not in line
    ]
    return bool(non_placeholder_lines)

# Load blocks from file
with open("debug_llm_blocks.json", "r", encoding="utf-8") as f:
    blocks = json.load(f)

print(f"🔍 Loaded {len(blocks)} blocks from debug_llm_blocks.json")

# Run the test
for i, block in enumerate(blocks):
    result = is_content_meaningful(block)
    status = "✅ Meaningful" if result else "❌ Trivial or empty"
    print(f"Block {i+1}: {status}")

# Continue from verify_blocks.py

trivial_blocks = [b for b in blocks if not is_content_meaningful(b)]

if trivial_blocks:
    with open("trivial_blocks.json", "w", encoding="utf-8") as f:
        json.dump(trivial_blocks, f, indent=2)
    print(f"🗂️ Saved {len(trivial_blocks)} trivial blocks to trivial_blocks.json")
else:
    print("🎉 No trivial blocks found.")

