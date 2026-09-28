import cv2

print("SNAPLAB AI — CAMERA TEST")
print("=" * 40)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("ERROR: Could not open camera.")

print("Camera: OK")
print("Press Q to quit.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("ERROR: Failed to read camera frame.")
        break

    cv2.putText(
        frame,
        "SnapLab AI - Camera Test",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("SnapLab AI - Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("CAMERA TEST COMPLETE")
