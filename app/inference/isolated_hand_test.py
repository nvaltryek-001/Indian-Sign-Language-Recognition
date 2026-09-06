import cv2
import mediapipe as mp
import numpy as np
import time

mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Don't force 1280x720 yet.
# Use whatever resolution the camera actually provides.
print("=" * 75)
print("ISOLATED MEDIAPIPE HAND TEST")
print("=" * 75)
print("Press Q to quit.")
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

    count = 0
    left_count = 0
    right_count = 0
    both_count = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            print("FRAME READ FAILED")
            continue

        count += 1

        h, w = frame.shape[:2]
        brightness = float(frame.mean())

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = holistic.process(rgb)
        rgb.flags.writeable = True

        pose = result.pose_landmarks is not None
        left = result.left_hand_landmarks is not None
        right = result.right_hand_landmarks is not None

        if left:
            left_count += 1
        if right:
            right_count += 1
        if left and right:
            both_count += 1

        if count % 10 == 0:
            print(
                f"Frame {count:03d} | "
                f"brightness={brightness:7.2f} | "
                f"size={w}x{h} | "
                f"pose={pose} left={left} right={right}"
            )

        cv2.putText(
            frame,
            f"Pose: {pose}  Left: {left}  Right: {right}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Brightness: {brightness:.1f}  {w}x{h}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        if left:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                result.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        if right:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                result.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        cv2.imshow("ISOLATED HAND TEST", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

print()
print("=" * 75)
print("FINAL")
print("=" * 75)
print("Frames:", count)
print("Left :", left_count)
print("Right:", right_count)
print("Both :", both_count)
