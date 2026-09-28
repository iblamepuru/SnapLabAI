from ultralytics import YOLO 
import cv2 
import sys 
from pathlib import Path 
 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)) 
 
from engineering_state import EngineeringStateEngine 
 
 
MODEL = r"runs\detect\runs\snaplab\phaseB_61class\weights\best.pt" 
 
model = YOLO(MODEL) 
 
engine = EngineeringStateEngine( 
    lost_after=10, 
    remove_after=60, 
    history_size=7 
) 
 
cap = cv2.VideoCapture(0) 
 
if not cap.isOpened(): 
    raise RuntimeError("Could not open camera") 
 
print("SnapLab-AI - Live Stabilized Engineering State") 
print("YOLO11n + ByteTrack + Temporal Stabilization") 
print("ACTIVE -> LOST -> REMOVED") 
print("Press Q to quit.") 
 
frame_count = 0 
 
while True: 
 
    ret, frame = cap.read() 
 
    if not ret: 
        print("Camera frame read failed.") 
        break 
 
    # Advance engineering state 
    engine.update_frame() 
 
    # YOLO + ByteTrack 
    results = model.track( 
        source=frame, 
        persist=True, 
        tracker="bytetrack.yaml", 
        imgsz=640, 
        conf=0.75, 
        verbose=False 
    ) 
 
    result = results[0] 
 
    # Process current detections 
    if result.boxes is not None and result.boxes.id is not None: 
 
        boxes = result.boxes.xyxy.cpu().numpy() 
        track_ids = result.boxes.id.cpu().numpy() 
        classes = result.boxes.cls.cpu().numpy() 
        confidences = result.boxes.conf.cpu().numpy() 
 
        for bbox, track_id, cls, confidence in zip( 
            boxes, 
            track_ids, 
            classes, 
            confidences 
        ): 
 
            track_id = int(track_id) 
            cls = int(cls) 
 
            raw_class = model.names[cls] 
 
            engine.update( 
                track_id=track_id, 
                class_name=raw_class, 
                confidence=float(confidence), 
                bbox=bbox 
            ) 
 
    # Update lifecycle 
    engine.update_lifecycle() 
 
    # Draw YOLO detections 
    annotated = result.plot() 
 
    # Draw stabilized engineering labels 
    state = engine.get_state() 
 
    for component in state.values(): 
 
        if component["status"] not in ["ACTIVE", "LOST"]: 
            continue 
 
        x1, y1, x2, y2 = component["bbox"] 
 
        label = ( 
            f'ID {component["track_id"]} | ' 
            f'{component["class_name"]} | ' 
            f'{component["confidence"]:.2f}' 
        ) 
 
        cv2.putText( 
            annotated, 
            label, 
            (x1, max(20, y1 - 10)), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.55, 
            (255, 255, 255), 
            2 
        ) 
 
    active_count = sum( 
        1 
        for component in state.values() 
        if component["status"] == "ACTIVE" 
    ) 
 
    lost_count = sum( 
        1 
        for component in state.values() 
        if component["status"] == "LOST" 
    ) 
 
    cv2.putText( 
        annotated, 
        f"ENGINEERING STATE | ACTIVE: {active_count} | LOST: {lost_count}", 
        (15, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.65, 
        (0, 255, 0), 
        2 
    ) 
 
    cv2.imshow( 
        "SnapLab-AI - Stabilized Engineering State", 
        annotated 
    ) 
 
    frame_count += 1 
 
    if frame_count % 30 == 0: 
        engine.print_state() 
 
    if cv2.waitKey(1) & 0xFF == ord("q"): 
        break 
 
 
cap.release() 
cv2.destroyAllWindows() 
 
print("\n========== FINAL ENGINEERING STATE ==========") 
engine.print_state() 
 
print("\nLive temporal stabilization test completed.") 
