import json

# Load your chunks file
with open("data/chunks.json", "r") as f:
    chunks = json.load(f)

# Use a set to collect unique section names
unique_sections = sorted(set(chunk.get("section", "UNKNOWN") for chunk in chunks))

# Print them
print("Unique Sections:")
for section in unique_sections:
    print("-", section)
