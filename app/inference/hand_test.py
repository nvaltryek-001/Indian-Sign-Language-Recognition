import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("Starting MediaPipe hand test...")
print("Show your hands clearly.")
print("Press Q to quit.")

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
) as holistic:

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            print("Camera read failed")
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = holistic.process(rgb)

        left = results.left_hand_landmarks
        right = results.right_hand_landmarks
        pose = results.pose_landmarks

        left_count = len(left.landmark) if left else 0
        right_count = len(right.landmark) if right else 0
        pose_count = len(pose.landmark) if pose else 0

        print(
            f"\rPose={pose_count:3d} | "
            f"Left Hand={left_count:2d} | "
            f"Right Hand={right_count:2d}",
            end=""
        )

        if pose:
            mp_drawing.draw_landmarks(
                frame,
                pose,
                mp_holistic.POSE_CONNECTIONS
            )

        if left:
            mp_drawing.draw_landmarks(
                frame,
                left,
                mp_holistic.HAND_CONNECTIONS
            )

        if right:
            mp_drawing.draw_landmarks(
                frame,
                right,
                mp_holistic.HAND_CONNECTIONS
            )

        cv2.imshow(
            "MediaPipe Hand Test",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()