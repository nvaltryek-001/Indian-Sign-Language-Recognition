import cv2
import mediapipe as mp
import time

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("=" * 75)
print("MEDIAPIPE HAND DETECTION DIAGNOSTIC")
print("=" * 75)
print("Camera: 0")
print("Resolution requested: 1280x720")
print("Show BOTH hands clearly.")
print("Press Q to quit.")
print()

prev = time.time()

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while True:
        ok, frame = cap.read()

        if not ok:
            print("Camera read failed")
            continue

        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        rgb.flags.writeable = True

        pose = results.pose_landmarks is not None
        left = results.left_hand_landmarks is not None
        right = results.right_hand_landmarks is not None

        # Draw pose
        if pose:
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS
            )

        # Draw left hand
        if left:
            mp_draw.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

        # Draw right hand
        if right:
            mp_draw.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now

        status = [
            f"Resolution: {w}x{h}",
            f"FPS: {fps:.1f}",
            f"POSE      : {'DETECTED' if pose else 'NO'}",
            f"LEFT HAND : {'DETECTED' if left else 'NO'}",
            f"RIGHT HAND: {'DETECTED' if right else 'NO'}",
        ]

        y = 35

        for i, text in enumerate(status):
            cv2.putText(
                frame,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if "DETECTED" in text else (0, 0, 255),
                2
            )
            y += 35

        cv2.rectangle(
            frame,
            (0, 0),
            (w - 1, h - 1),
            (0, 255, 0) if left and right else (0, 0, 255),
            3
        )

        cv2.imshow("MediaPipe Hand Diagnostic", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
