from pathlib import Path
import yaml

root = Path("datasets/snaplab_master")

classes = [
    "1-5-Volt-Battery",
    "3-3-Volt-Battery",
    "7-Segment-Display",
    "9-Volt-Battery",
    "Arduino-Mega",
    "Arduino-Nano",
    "Arduino-Uno",
    "BJT-Transistor",
    "Bluetooth-Module",
    "Breadboard",
    "Bridge-Rectifier",
    "Buck-Converter",
    "Buzzer",
    "Capacitor-10mf",
    "Capacitor-470mf",
    "DC-Motor",
    "Diode",
    "ESP32",
    "ESP32-CAM",
    "FT-232-USB-Serial-Module",
    "Film-Capacitor",
    "Fuse",
    "Fuse-Base",
    "GSM-Module",
    "Gas-Sensor",
    "Heat-Sink",
    "High-Voltage-Ceramic-Capacitor",
    "Humidity-Sensor",
    "IC-Base-14-Pin",
    "IC-Base-28-Pin",
    "IC-Chip",
    "IGBT",
    "IR-Sensor",
    "Inductor",
    "Keypad",
    "LCD-Display",
    "LDR-Sensor",
    "LED-Light",
    "Low-Voltage-Ceramic-Capacitor",
    "MLC-Capacitor",
    "MOSFET",
    "Motion-Sensor",
    "Motor-Driver",
    "NTC-Thermistor",
    "OLED-Display",
    "Pin-Header",
    "Push-Switch",
    "RFID-Scanner",
    "Raindrops-Module",
    "Relay-Module",
    "Resistor",
    "Rocker-Switch",
    "Servo-Motor",
    "Soil-Moisture-Sensor",
    "Sonar-Sensor",
    "TCRT5000",
    "Tact-Switch",
    "Taper-Potentiometer",
    "Trimmer-Potentiometer",
    "Water-Sensor",
    "Zener-Diode",
    "OP-Amp",
    "Cable",
    "Generic-Capacitor",
    "Variable-Resistor",
]

root.mkdir(parents=True, exist_ok=True)

data = {
    "path": str(root.resolve()).replace("\\", "/"),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": len(classes),
    "names": classes,
}

(root / "data.yaml").write_text(
    yaml.safe_dump(data, sort_keys=False),
    encoding="utf-8"
)

print("Master ontology created.")
print("Classes:", len(classes))
print("Config :", root / "data.yaml")

for i, name in enumerate(classes):
    print(f"{i:2d}: {name}")
