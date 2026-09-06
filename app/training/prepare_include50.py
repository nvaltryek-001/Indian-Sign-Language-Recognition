from datasets import load_dataset
import csv
import os

print("Loading cached INCLUDE dataset...")

ds = load_dataset("ai4bharat/INCLUDE")

os.makedirs("data/include50", exist_ok=True)

for split in ds:
    rows = [x for x in ds[split] if x["include_50"]]

    output = f"data/include50/{split}.csv"

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "parent_label",
                "label",
                "video_path",
                "include_50"
            ]
        )

        writer.writeheader()

        for row in rows:
            writer.writerow({
                "parent_label": row["parent_label"],
                "label": row["label"],
                "video_path": row["video_path"],
                "include_50": row["include_50"]
            })

    print(f"{split}: {len(rows)} INCLUDE-50 videos")
    print(f"Saved: {output}")

print("METADATA SAVED SUCCESSFULLY")
