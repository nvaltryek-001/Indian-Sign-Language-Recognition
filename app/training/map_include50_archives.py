import requests
import pandas as pd
from remotezip import RemoteZip

ZENODO = "https://zenodo.org/api/records/4010759"

print("Loading required INCLUDE-50 videos...")

df = pd.read_csv("data/include50/all_required_videos.csv")

required = set(df["video_path"].tolist())

print("Required videos:", len(required))

print("\nGetting Zenodo files...")

response = requests.get(ZENODO, timeout=60)
response.raise_for_status()

files = response.json()["files"]

archives = [
    f for f in files
    if f["key"].lower().endswith(".zip")
]

print("ZIP archives:", len(archives))

results = {}

for number, archive in enumerate(archives, 1):

    name = archive["key"]
    url = archive["links"]["self"]

    print()
    print("=" * 70)
    print(f"[{number}/{len(archives)}] Checking {name}")
    print("=" * 70)

    try:

        with RemoteZip(url) as z:

            names = z.namelist()

            matches = []

            for video in required:
                if video in names:
                    matches.append(video)

            print("Videos found:", len(matches))

            for video in matches:
                results[video] = name
                print("  FOUND:", video)

    except Exception as e:

        print("ERROR:", e)

print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print("Videos mapped:", len(results))
print("Videos missing:", len(required - set(results)))

out = df.copy()

out["archive"] = out["video_path"].map(results)

out.to_csv(
    "data/include50/video_archive_mapping.csv",
    index=False
)

print()
print("Saved:")
print("data/include50/video_archive_mapping.csv")

if required - set(results):
    print()
    print("MISSING VIDEOS:")
    for x in sorted(required - set(results)):
        print(x)
