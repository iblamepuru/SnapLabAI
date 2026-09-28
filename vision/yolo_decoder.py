import numpy as np
import cv2

CLASS_NAMES = [
    "1-5-Volt-Battery","3-3-Volt-Battery","7-Segment-Display",
    "9-Volt-Battery","Arduino-Mega","Arduino-Nano","Arduino-Uno",
    "BJT-Transistor","Bluetooth-Module","Breadboard","Bridge-Rectifier",
    "Buck-Converter","Buzzer","Capacitor-10mf","Capacitor-470mf",
    "DC-Motor","Diode","ESP32","ESP32-CAM","FT-232-USB-Serial-Module",
    "Film-Capacitor","Fuse","Fuse-Base","GSM-Module","Gas-Sensor",
    "Heat-Sink","High-Voltage-Ceramic-Capacitor","Humidity-Sensor",
    "IC-Base-14-Pin","IC-Base-28-Pin","IC-Chip","IGBT","IR-Sensor",
    "Inductor","Keypad","LCD-Display","LDR-Sensor","LED-Light",
    "Low-Voltage-Ceramic-Capacitor","MLC-Capacitor","MOSFET",
    "Motion-Sensor","Motor-Driver","NTC-Thermistor","OLED-Display",
    "Pin-Header","Push-Switch","RFID-Scanner","Raindrops-Module",
    "Relay-Module","Resistor","Rocker-Switch","Servo-Motor",
    "Soil-Moisture-Sensor","Sonar-Sensor","TCRT5000","Tact-Switch",
    "Taper-Potentiometer","Trimmer-Potentiometer","Water-Sensor",
    "Zener-Diode","OP-Amp","Cable","Generic-Capacitor","Variable-Resistor"
]

def decode_yolo_output(output, conf_threshold=0.25, iou_threshold=0.45):
    output = np.asarray(output)

    if output.ndim == 3:
        output = output[0]

    if output.shape == (69, 8400):
        output = output.T

    if output.shape[1] != 69:
        raise ValueError(f"Expected 69 channels, got {output.shape}")

    boxes_xywh = output[:, :4]
    class_scores = output[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(len(class_scores)), class_ids]

    keep = confidences >= conf_threshold

    boxes_xywh = boxes_xywh[keep]
    confidences = confidences[keep]
    class_ids = class_ids[keep]

    if len(boxes_xywh) == 0:
        return []

    boxes = []
    for x, y, w, h in boxes_xywh:
        x1 = float(x - w / 2)
        y1 = float(y - h / 2)
        boxes.append([
            x1, y1, float(w), float(h)
        ])

    indices = cv2.dnn.NMSBoxes(
        boxes,
        confidences.tolist(),
        conf_threshold,
        iou_threshold
    )

    if len(indices) == 0:
        return []

    indices = np.array(indices).reshape(-1)

    detections = []

    for i in indices:
        class_id = int(class_ids[i])

        detections.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "confidence": float(confidences[i]),
            "box": boxes[i]
        })

    return detections
