import os
import requests
import pandas as pd

ZENODO = "https://zenodo.org/api/records/4010759"
OUT = "dataset/archives"

os.makedirs(OUT, exist_ok=True)

print("Reading INCLUDE-50 metadata...")
df = pd.concat([
    pd.read_csv("data/include50/train.csv"),
    pd.read_csv("data/include50/val.csv"),
    pd.read_csv("data/include50/test.csv")
], ignore_index=True)

required_categories = sorted(df["parent_label"].unique())

print("\nRequired categories:")
for x in required_categories:
    print(" ", x)

print("\nGetting Zenodo archive list...")

r = requests.get(ZENODO, timeout=60)
r.raise_for_status()

files = r.json()["files"]

archives = [
    f for f in files
    if f["key"].lower().endswith(".zip")
]

print("\nAvailable archives:")
for f in archives:
    print(f["key"], round(f["size"] / 1024 / 1024, 1), "MB")

print("\nIMPORTANT:")
print("The metadata gives video paths but not the archive-part mapping.")
print("We therefore need to inspect archive contents before downloading everything.")
