import cv2
import mediapipe as mp
import numpy as np

print("=" * 60)
print("MEDIAPIPE WEBCAM VISIBILITY TEST")
print("=" * 60)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera 0 cannot be opened")

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
) as holistic:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Could not read frame")
            continue

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = holistic.process(rgb)

        pose = results.pose_landmarks is not None
        left = results.left_hand_landmarks is not None
        right = results.right_hand_landmarks is not None

        cv2.putText(
            frame,
            f"POSE: {pose}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"LEFT HAND: {left}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"RIGHT HAND: {right}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS
            )

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        cv2.imshow("MediaPipe Test - Press Q", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

print()
print("TEST FINISHED")
