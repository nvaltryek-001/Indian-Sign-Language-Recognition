import cv2
import mediapipe as mp
import numpy as np

FRAMES = 90

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

left_count = 0
right_count = 0
both_count = 0
pose_count = 0
total = 0

print("=" * 75)
print("90-FRAME HAND STABILITY TEST")
print("=" * 75)
print("Keep BOTH HANDS visible while performing the sign.")
print("Do NOT start moving until BOTH HANDS are detected.")
print("Press Q to stop.")
print()

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
) as holistic:

    # Warm-up
    stable = 0

    while stable < 20:

        ok, frame = cap.read()
        if not ok:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        r = holistic.process(rgb)
        rgb.flags.writeable = True

        left = r.left_hand_landmarks is not None
        right = r.right_hand_landmarks is not None

        if left and right:
            stable += 1
        else:
            stable = 0

        text = f"WAIT: BOTH HANDS {stable}/20"

        cv2.putText(
            frame, text, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
            (0,255,0) if left and right else (0,0,255),
            2
        )

        if left:
            mp_draw.draw_landmarks(
                frame,
                r.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        if right:
            mp_draw.draw_landmarks(
                frame,
                r.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        cv2.imshow("Hand Stability Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit

    print("Both hands stable.")
    print("RECORDING NOW...")
    print()

    while total < FRAMES:

        ok, frame = cap.read()
        if not ok:
            continue

        total += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        r = holistic.process(rgb)
        rgb.flags.writeable = True

        pose = r.pose_landmarks is not None
        left = r.left_hand_landmarks is not None
        right = r.right_hand_landmarks is not None

        if pose:
            pose_count += 1

        if left:
            left_count += 1

        if right:
            right_count += 1

        if left and right:
            both_count += 1

        text = (
            f"{total}/90 | "
            f"LEFT:{'OK' if left else 'LOST'} | "
            f"RIGHT:{'OK' if right else 'LOST'}"
        )

        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0) if left and right else (0,0,255),
            2
        )

        if not left:
            cv2.putText(
                frame,
                "LEFT HAND LOST",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0,0,255),
                3
            )

        if left:
            mp_draw.draw_landmarks(
                frame,
                r.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        if right:
            mp_draw.draw_landmarks(
                frame,
                r.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        cv2.imshow("Hand Stability Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

print()
print("=" * 75)
print("RESULT")
print("=" * 75)
print("Frames :", total)
print("Pose   :", pose_count)
print("Left   :", left_count)
print("Right  :", right_count)
print("Both   :", both_count)
print()
print("Left % :", round(left_count / max(total,1) * 100, 1))
print("Right %:", round(right_count / max(total,1) * 100, 1))
print("Both % :", round(both_count / max(total,1) * 100, 1))
