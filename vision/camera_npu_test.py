import cv2
import numpy as np
from snapdragon_npu_adapter import SnapdragonNPUAdapter

adapter = SnapdragonNPUAdapter()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera could not be opened")

ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Camera frame capture failed")

img = cv2.resize(frame, (640, 640))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
tensor = img.astype(np.float32) / 255.0
tensor = np.transpose(tensor, (2, 0, 1))
tensor = np.expand_dims(tensor, axis=0)

output = adapter.infer(tensor)

print("CAMERA FRAME: PASSED")
print("FRAME:", frame.shape)
print("TENSOR:", tensor.shape)
print("OUTPUT:", output.shape)
print("OUTPUT FINITE:", np.isfinite(output).all())
