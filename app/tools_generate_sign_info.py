import json
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

# tools_generate_sign_info.py is inside:
# N:\ISL-Recognition-Research\app\
#
# parents[1] gives:
# N:\ISL-Recognition-Research\

BASE_DIR = Path(__file__).resolve().parents[1]

CLASSES_FILE = BASE_DIR / "models" / "include50_classes_v2.json"
OUTPUT_FILE = BASE_DIR / "app" / "data" / "sign_info.json"


# ============================================================
# EMOJI FOR 50 SIGNS
# ============================================================

EMOJIS = {
    "dog": "🐕",
    "loud": "🔊",
    "car": "🚗",
    "election": "🗳️",
    "train ticket": "🎫",
    "house": "🏠",
    "death": "⚰️",
    "quiet": "🤫",
    "court": "⚖️",
    "store or shop": "🏪",
    "window": "🪟",
    "happy": "😊",
    "pen": "🖊️",
    "bank": "🏦",
    "hat": "🎩",
    "bird": "🐦",
    "i": "🙋",
    "paint": "🎨",
    "t-shirt": "👕",
    "shoes": "👟",
    "it": "👉",
    "you (plural)": "👥",
    "red": "🔴",
    "hello": "👋",
    "cow": "🐄",
    "good morning": "🌅",
    "fan": "🌀",
    "black": "⚫",
    "cell phone": "📱",
    "thank you": "🙏",
    "white": "⚪",
    "father": "👨",
    "summer": "☀️",
    "fall": "🍂",
    "brother": "👦",
    "monday": "📅",
    "boy": "👦",
    "girl": "👧",
    "year": "📆",
    "long": "↔️",
    "short": "↕️",
    "big large": "⬆️",
    "teacher": "👨‍🏫",
    "small little": "🤏",
    "time": "⏰",
    "hot": "🔥",
    "priest": "🙏",
    "new": "✨",
    "good": "👍",
    "dry": "🏜️"
}


# ============================================================
# LOAD 50 CLASSES
# ============================================================

def load_classes():

    if not CLASSES_FILE.exists():

        raise FileNotFoundError(
            f"\nClasses file not found:\n{CLASSES_FILE}\n"
        )

    with open(
        CLASSES_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    # --------------------------------------------------------
    # List format
    # --------------------------------------------------------

    if isinstance(data, list):

        return data

    # --------------------------------------------------------
    # Dictionary format
    # --------------------------------------------------------

    if isinstance(data, dict):

        # Example:
        # {
        #   "classes": [...]
        # }

        for key in [
            "classes",
            "class_names",
            "labels"
        ]:

            if key in data:

                if isinstance(data[key], list):

                    return data[key]

        # ----------------------------------------------------
        # Your actual V2 format:
        #
        # {
        #   "0": "1. Dog",
        #   "1": "1. loud",
        #   ...
        # }
        # ----------------------------------------------------

        try:

            sorted_items = sorted(
                data.items(),
                key=lambda x: int(x[0])
            )

            return [
                value
                for key, value in sorted_items
            ]

        except Exception:

            return list(data.values())

    raise ValueError(
        "Unsupported class JSON format."
    )


# ============================================================
# CATEGORY
# ============================================================

def get_category(name):

    name = name.lower()

    # Animals
    if any(
        x in name
        for x in [
            "dog",
            "bird",
            "cow"
        ]
    ):

        return "Animals"

    # People / Pronouns
    if any(
        x in name
        for x in [
            "father",
            "brother",
            "boy",
            "girl",
            "teacher",
            "priest",
            " i",
            "you",
            "it"
        ]
    ):

        return "People / Pronouns"

    # Greetings
    if any(
        x in name
        for x in [
            "hello",
            "thank",
            "good morning"
        ]
    ):

        return "Greetings"

    # Objects / Places
    if any(
        x in name
        for x in [
            "car",
            "train",
            "house",
            "court",
            "store",
            "shop",
            "window",
            "pen",
            "bank",
            "hat",
            "paint",
            "t-shirt",
            "shoes",
            "fan",
            "cell phone"
        ]
    ):

        return "Objects / Places"

    # General
    if any(
        x in name
        for x in [
            "red",
            "black",
            "white",
            "summer",
            "fall",
            "monday",
            "year",
            "time"
        ]
    ):

        return "General"

    # Default
    return "Actions / Descriptions"


# ============================================================
# CREATE INFORMATION FOR ONE SIGN
# ============================================================

def make_info(sign):

    # Convert:
    #
    # "4. Bird"
    #
    # into:
    #
    # "Bird"

    clean = sign.split(
        ". ",
        1
    )[-1]

    key = clean.lower()

    emoji = EMOJIS.get(
        key,
        "🤟"
    )

    return {

        "title": clean,

        "emoji": emoji,

        "category": get_category(
            clean
        ),

        "instruction": (
            f"Practice the ISL sign for "
            f"'{clean}' using the reference "
            "video. Then perform the same "
            "movement naturally in front "
            "of the webcam."
        ),

        # Will be filled later when we connect
        # the reference videos/images.
        "reference": "",

        # Initial status.
        "status": "pending"

    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("GENERATING INCLUDE-50 SIGN INFORMATION")
    print("=" * 70)

    print()
    print("Project directory:")
    print(BASE_DIR)

    print()
    print("Classes file:")
    print(CLASSES_FILE)

    print()
    print("Output file:")
    print(OUTPUT_FILE)

    # --------------------------------------------------------
    # Load classes
    # --------------------------------------------------------

    classes = load_classes()

    print()
    print(
        "Classes found:",
        len(classes)
    )

    # --------------------------------------------------------
    # Verify 50 classes
    # --------------------------------------------------------

    if len(classes) != 50:

        print()
        print(
            "WARNING: Expected 50 classes, "
            f"but found {len(classes)}."
        )

    else:

        print(
            "SUCCESS: Exactly 50 classes found."
        )

    # --------------------------------------------------------
    # Generate information
    # --------------------------------------------------------

    result = {}

    for sign in classes:

        result[sign] = make_info(
            sign
        )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("CREATED SUCCESSFULLY")
    print("-" * 70)

    print()
    print(
        "Total sign information:",
        len(result)
    )

    print()
    print("50 SIGN LIST")
    print("-" * 70)

    for i, sign in enumerate(
        classes,
        1
    ):

        info = result[sign]

        print(
            f"{i:02d}. "
            f"{info['emoji']} "
            f"{info['title']}"
        )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()