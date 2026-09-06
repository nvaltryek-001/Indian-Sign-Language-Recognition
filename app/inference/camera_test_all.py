import cv2

print("=" * 60)
print("CAMERA TEST")
print("=" * 60)

for index in range(5):

    cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        print(f"Camera {index}: NOT AVAILABLE")
        continue

    ret, frame = cap.read()

    if not ret or frame is None:
        print(f"Camera {index}: NO FRAME")
        cap.release()
        continue

    print(
        f"Camera {index}: "
        f"shape={frame.shape}, "
        f"mean={frame.mean():.2f}, "
        f"min={frame.min()}, "
        f"max={frame.max()}"
    )

    cv2.imshow(f"Camera {index}", frame)

    print(f"Press any key to close Camera {index}...")
    cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

print("DONE")
