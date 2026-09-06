import os
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

MAPPING = Path("data/include50/video_archive_mapping.csv")
DOWNLOAD_DIR = Path("data/include50/archives")
VIDEO_DIR = Path("data/include50/videos")

ZENODO_API = "https://zenodo.org/api/records/4010759"

CHUNK_SIZE = 8 * 1024 * 1024
MAX_RETRIES = 10
TIMEOUT = 120

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "User-Agent": "INCLUDE50-ISL-Research/1.0"
})


# ============================================================
# LOAD MAPPING
# ============================================================

print("=" * 70)
print("          INCLUDE-50 VIDEO DOWNLOADER")
print("=" * 70)

print("\nLoading mapping...")

df = pd.read_csv(MAPPING)

expected_videos = set(
    df["video_path"].dropna()
)

archives = sorted(
    df["archive"].dropna().unique()
)

print("Unique videos :", len(expected_videos))
print("ZIP archives  :", len(archives))


# ============================================================
# GET ZENODO FILES
# ============================================================

print("\nGetting Zenodo archive list...")

r = session.get(
    ZENODO_API,
    timeout=TIMEOUT
)

r.raise_for_status()

record = r.json()

zenodo_files = {}

for f in record["files"]:

    name = f["key"]

    if name.lower().endswith(".zip"):

        zenodo_files[name] = f["links"]["self"]


print(
    "Zenodo ZIP archives available:",
    len(zenodo_files)
)


# ============================================================
# DOWNLOAD WITH RESUME
# ============================================================

def download_file(url, destination):

    destination = Path(destination)

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            existing = (
                destination.stat().st_size
                if destination.exists()
                else 0
            )

            headers = {}

            if existing > 0:

                headers["Range"] = (
                    f"bytes={existing}-"
                )

                print(
                    f"Resuming from "
                    f"{existing / 1024 / 1024:.1f} MB"
                )

            else:

                print("Starting download...")

            with session.get(
                url,
                headers=headers,
                stream=True,
                timeout=TIMEOUT
            ) as response:

                response.raise_for_status()

                # ------------------------------------------------
                # Determine download mode
                # ------------------------------------------------

                if (
                    existing > 0
                    and response.status_code == 206
                ):

                    mode = "ab"

                    remaining = int(
                        response.headers.get(
                            "Content-Length",
                            0
                        )
                    )

                    total = existing + remaining

                    print(
                        "Resume supported."
                    )

                else:

                    mode = "wb"

                    existing = 0

                    total = int(
                        response.headers.get(
                            "Content-Length",
                            0
                        )
                    )

                    if total:

                        print(
                            f"Total size: "
                            f"{total / 1024 / 1024:.1f} MB"
                        )

                downloaded = existing

                with open(
                    destination,
                    mode
                ) as output:

                    last_print = 0

                    for chunk in response.iter_content(
                        chunk_size=CHUNK_SIZE
                    ):

                        if not chunk:
                            continue

                        output.write(chunk)

                        downloaded += len(chunk)

                        now = time.time()

                        if (
                            now - last_print >= 1
                            or (
                                total > 0
                                and downloaded >= total
                            )
                        ):

                            if total > 0:

                                percent = (
                                    downloaded
                                    / total
                                    * 100
                                )

                                print(
                                    f"\rProgress: "
                                    f"{percent:6.2f}% "
                                    f"({downloaded / 1024 / 1024:.1f}"
                                    f"/"
                                    f"{total / 1024 / 1024:.1f} MB)",
                                    end="",
                                    flush=True
                                )

                            else:

                                print(
                                    f"\rDownloaded: "
                                    f"{downloaded / 1024 / 1024:.1f} MB",
                                    end="",
                                    flush=True
                                )

                            last_print = now

                print()

                final_size = (
                    destination.stat().st_size
                )

                if total > 0 and final_size != total:

                    raise RuntimeError(
                        f"Incomplete file: "
                        f"{final_size} / {total}"
                    )

                print("Download complete.")

                return True

        except Exception as error:

            print()
            print(
                f"Download failed "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )

            print("Error:", error)

            if attempt < MAX_RETRIES:

                wait = min(
                    attempt * 10,
                    60
                )

                print(
                    f"Retrying in {wait} seconds..."
                )

                time.sleep(wait)

            else:

                print(
                    "Giving up on this archive."
                )

                return False

    return False


# ============================================================
# CHECK ZIP
# ============================================================

def valid_zip(path):

    if not path.exists():
        return False

    try:

        with zipfile.ZipFile(path, "r") as z:

            bad = z.testzip()

            return bad is None

    except Exception:

        return False


# ============================================================
# PROCESS ARCHIVES
# ============================================================

for number, archive in enumerate(
    archives,
    1
):

    print()
    print("=" * 70)
    print(
        f"[{number}/{len(archives)}] {archive}"
    )
    print("=" * 70)

    if archive not in zenodo_files:

        print(
            "ERROR: Archive not found:",
            archive
        )

        continue

    zip_path = (
        DOWNLOAD_DIR / archive
    )

    url = zenodo_files[archive]

    # --------------------------------------------------------
    # Check existing ZIP
    # --------------------------------------------------------

    if zip_path.exists():

        print(
            "Existing ZIP:",
            round(
                zip_path.stat().st_size
                / 1024 / 1024,
                1
            ),
            "MB"
        )

        if valid_zip(zip_path):

            print(
                "ZIP is valid."
            )

        else:

            print(
                "ZIP is incomplete/corrupt."
            )

            print(
                "Removing invalid ZIP."
            )

            zip_path.unlink()

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    if not zip_path.exists():

        print("URL:")
        print(url)

        success = download_file(
            url,
            zip_path
        )

        if not success:

            print(
                "Archive failed:",
                archive
            )

            continue

    # --------------------------------------------------------
    # Required videos
    # --------------------------------------------------------

    required = set(
        df.loc[
            df["archive"] == archive,
            "video_path"
        ].dropna()
    )

    print(
        "Required videos:",
        len(required)
    )

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    extracted = 0
    already = 0
    missing = 0

    try:

        with zipfile.ZipFile(
            zip_path,
            "r"
        ) as z:

            names = set(
                z.namelist()
            )

            for video in sorted(required):

                output = (
                    VIDEO_DIR / video
                )

                output.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                if output.exists():

                    already += 1
                    continue

                if video not in names:

                    print(
                        "Missing in ZIP:",
                        video
                    )

                    missing += 1
                    continue

                with z.open(video) as source:

                    with open(
                        output,
                        "wb"
                    ) as target:

                        while True:

                            data = source.read(
                                4 * 1024 * 1024
                            )

                            if not data:
                                break

                            target.write(data)

                extracted += 1

    except zipfile.BadZipFile:

        print(
            "ZIP corruption detected:",
            archive
        )

        continue

    print(
        "Extracted:",
        extracted
    )

    print(
        "Already existed:",
        already
    )

    print(
        "Missing:",
        missing
    )


# ============================================================
# FINAL VERIFICATION
# ============================================================

print()
print("=" * 70)
print("              FINAL VERIFICATION")
print("=" * 70)

actual = set()

for path in VIDEO_DIR.rglob("*"):

    if path.is_file():

        relative = str(
            path.relative_to(VIDEO_DIR)
        ).replace("\\", "/")

        actual.add(relative)


found = expected_videos & actual

missing = expected_videos - actual

print()
print(
    "Expected unique videos:",
    len(expected_videos)
)

print(
    "Found videos:",
    len(found)
)

print(
    "Missing videos:",
    len(missing)
)

if missing:

    print()
    print("MISSING:")

    for video in sorted(missing):

        print(video)

else:

    print()
    print("=" * 70)
    print("ALL REQUIRED VIDEOS FOUND")
    print("=" * 70)