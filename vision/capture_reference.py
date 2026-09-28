import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera could not be opened")

print("CAMERA TEST")
print("Point camera at the same circuit image.")
print("Press S to save a frame.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    cv2.imshow("SnapLab - Capture Reference", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        cv2.imwrite(r"vision\reference_frame.jpg", frame)
        print("REFERENCE FRAME: SAVED")
        print("FILE: vision\\reference_frame.jpg")
        break

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
