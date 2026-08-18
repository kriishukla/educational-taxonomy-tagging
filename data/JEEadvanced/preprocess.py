import json
from pathlib import Path

BASE_DIR = Path(__file__).parent

input_file = BASE_DIR / "advanced.json"
output_file = BASE_DIR / "advanced_cleaned.json"

keys_to_remove = {"index", "description", "type", "gold"}

subject_map = {
    "phy": "physics",
    "chem": "chemistry"
}

allowed_subjects = {"physics", "chemistry"}

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

cleaned_data = []

for item in data:
    cleaned_item = {k: v for k, v in item.items() if k not in keys_to_remove}

    if "subject" in cleaned_item:
        # normalize subject
        cleaned_item["subject"] = subject_map.get(
            cleaned_item["subject"],
            cleaned_item["subject"]
        )

        # skip non-physics / non-chemistry
        if cleaned_item["subject"] not in allowed_subjects:
            continue

    cleaned_data.append(cleaned_item)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, indent=4, ensure_ascii=False)

print(f"Saved → {output_file}")
