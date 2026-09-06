import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


# ============================================================
# CONFIG
# ============================================================

VIDEO_DIR = Path("data/include50/videos")
OUTPUT_DIR = Path("data/include50/sequences")

MAPPING_FILE = Path(
    "data/include50/video_archive_mapping.csv"
)

ERROR_LOG = Path(
    "data/include50/landmark_extraction_errors.csv"
)

SEQUENCE_LENGTH = 45
FEATURES = 258


# ============================================================
# MEDIAPIPE
# ============================================================

mp_holistic = mp.solutions.holistic


# ============================================================
# EXTRACT 258 FEATURES
# ============================================================

def extract_landmarks(results):

    # --------------------------------------------------------
    # Pose: 33 × 4 = 132
    # x, y, z, visibility
    # --------------------------------------------------------

    pose = []

    if results.pose_landmarks:

        for lm in results.pose_landmarks.landmark:

            pose.extend([
                lm.x,
                lm.y,
                lm.z,
                lm.visibility
            ])

    else:

        pose = [0.0] * 132


    # --------------------------------------------------------
    # Left hand: 21 × 3 = 63
    # --------------------------------------------------------

    left_hand = []

    if results.left_hand_landmarks:

        for lm in results.left_hand_landmarks.landmark:

            left_hand.extend([
                lm.x,
                lm.y,
                lm.z
            ])

    else:

        left_hand = [0.0] * 63


    # --------------------------------------------------------
    # Right hand: 21 × 3 = 63
    # --------------------------------------------------------

    right_hand = []

    if results.right_hand_landmarks:

        for lm in results.right_hand_landmarks.landmark:

            right_hand.extend([
                lm.x,
                lm.y,
                lm.z
            ])

    else:

        right_hand = [0.0] * 63


    features = (
        pose +
        left_hand +
        right_hand
    )

    features = np.asarray(
        features,
        dtype=np.float32
    )


    if features.shape != (FEATURES,):

        raise ValueError(
            f"Expected {FEATURES} features, "
            f"got {features.shape}"
        )


    return features


# ============================================================
# GET VIDEO FRAMES
# ============================================================

def sample_frames(video_path):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: {video_path}"
        )


    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    if total_frames <= 0:

        cap.release()

        raise RuntimeError(
            f"No frames found: {video_path}"
        )


    indices = np.linspace(
        0,
        total_frames - 1,
        SEQUENCE_LENGTH
    ).astype(int)


    frames = []


    for index in indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(index)
        )

        ret, frame = cap.read()


        if ret:

            frames.append(frame)


    cap.release()


    return frames


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(
    video_path,
    output_path,
    holistic
):

    frames = sample_frames(
        video_path
    )


    sequence = []


    for frame in frames:

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        results = holistic.process(
            rgb
        )


        features = extract_landmarks(
            results
        )


        sequence.append(
            features
        )


    # --------------------------------------------------------
    # Padding if necessary
    # --------------------------------------------------------

    while len(sequence) < SEQUENCE_LENGTH:

        sequence.append(
            np.zeros(
                FEATURES,
                dtype=np.float32
            )
        )


    sequence = np.asarray(
        sequence[:SEQUENCE_LENGTH],
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Final shape validation
    # --------------------------------------------------------

    if sequence.shape != (
        SEQUENCE_LENGTH,
        FEATURES
    ):

        raise ValueError(
            f"Wrong sequence shape: "
            f"{sequence.shape}"
        )


    if np.isnan(sequence).any():

        raise ValueError(
            "NaN values detected"
        )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    np.save(
        output_path,
        sequence
    )


    return sequence.shape


# ============================================================
# FIND ALL VIDEOS
# ============================================================

def find_videos():

    videos = []

    videos.extend(
        VIDEO_DIR.rglob("*.MOV")
    )

    videos.extend(
        VIDEO_DIR.rglob("*.MP4")
    )

    videos.extend(
        VIDEO_DIR.rglob("*.mov")
    )

    videos.extend(
        VIDEO_DIR.rglob("*.mp4")
    )


    # Remove duplicates

    videos = sorted(
        set(videos)
    )


    return videos


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("INCLUDE-50 FULL LANDMARK EXTRACTION")
    print("=" * 70)

    print()


    # --------------------------------------------------------
    # Check directories
    # --------------------------------------------------------

    if not VIDEO_DIR.exists():

        raise RuntimeError(
            f"Video directory not found: "
            f"{VIDEO_DIR}"
        )


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Find videos
    # --------------------------------------------------------

    videos = find_videos()


    print(
        f"Videos found: {len(videos)}"
    )


    if len(videos) != 943:

        print()
        print(
            "WARNING: Expected 943 videos."
        )

        print(
            f"Found {len(videos)} videos."
        )

        print()


    # --------------------------------------------------------
    # Error records
    # --------------------------------------------------------

    errors = []


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    processed = 0
    skipped = 0
    failed = 0


    # --------------------------------------------------------
    # MediaPipe
    # --------------------------------------------------------

    print()
    print("Initializing MediaPipe Holistic...")


    with mp_holistic.Holistic(

        static_image_mode=False,

        model_complexity=1,

        smooth_landmarks=True,

        enable_segmentation=False,

        refine_face_landmarks=False

    ) as holistic:


        # ----------------------------------------------------
        # Process every video
        # ----------------------------------------------------

        for index, video_path in enumerate(
            videos,
            start=1
        ):


            relative = video_path.relative_to(
                VIDEO_DIR
            )


            output_path = (
                OUTPUT_DIR /
                relative.with_suffix(".npy")
            )


            print()
            print("-" * 70)

            print(
                f"[{index}/{len(videos)}] "
                f"{relative}"
            )


            # ------------------------------------------------
            # Skip already processed
            # ------------------------------------------------

            if output_path.exists():

                try:

                    existing = np.load(
                        output_path
                    )


                    if existing.shape == (
                        SEQUENCE_LENGTH,
                        FEATURES
                    ):

                        print(
                            "Status: SKIPPED "
                            "(already processed)"
                        )

                        skipped += 1

                        continue


                except Exception:

                    print(
                        "Existing file invalid. "
                        "Reprocessing..."
                    )


            # ------------------------------------------------
            # Process
            # ------------------------------------------------

            try:

                shape = process_video(
                    video_path,
                    output_path,
                    holistic
                )


                processed += 1


                print(
                    f"Status: SUCCESS"
                )

                print(
                    f"Shape : {shape}"
                )


            except Exception as e:

                failed += 1


                print(
                    f"Status: FAILED"
                )

                print(
                    f"Error : {e}"
                )


                errors.append({

                    "video": str(
                        relative
                    ).replace(
                        "\\",
                        "/"
                    ),

                    "error": str(e),

                    "time": datetime.now().isoformat()

                })


            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            completed = (
                processed +
                skipped
            )


            percent = (
                completed /
                len(videos) *
                100
            )


            print(
                f"Progress: "
                f"{completed}/{len(videos)} "
                f"({percent:.1f}%)"
            )


    # ========================================================
    # SAVE ERROR LOG
    # ========================================================

    if errors:

        error_df = pd.DataFrame(
            errors
        )


        error_df.to_csv(
            ERROR_LOG,
            index=False
        )


        print()
        print(
            f"Error log saved: "
            f"{ERROR_LOG}"
        )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    total_sequences = len(
        list(
            OUTPUT_DIR.rglob("*.npy")
        )
    )


    print()
    print("=" * 70)
    print("LANDMARK EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"Videos found     : {len(videos)}"
    )

    print(
        f"Newly processed   : {processed}"
    )

    print(
        f"Skipped existing  : {skipped}"
    )

    print(
        f"Failed            : {failed}"
    )

    print(
        f"Sequence files    : {total_sequences}"
    )

    print()


    if failed == 0:

        print(
            "STATUS: SUCCESS"
        )

    else:

        print(
            "STATUS: COMPLETED "
            "WITH ERRORS"
        )

        print(
            f"Check: {ERROR_LOG}"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()