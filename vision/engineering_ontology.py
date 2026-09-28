COMPONENT_ONTOLOGY = {
    "1-5-Volt-Battery": {
        "role": "Power source",
        "category": "Power",
        "interfaces": ["power", "ground"],
        "functions": ["low-voltage supply"]
    },
    "3-3-Volt-Battery": {
        "role": "Power source",
        "category": "Power",
        "interfaces": ["power", "ground"],
        "functions": ["3.3V supply"]
    },
    "7-Segment-Display": {
        "role": "Numeric display",
        "category": "Display",
        "interfaces": ["digital", "power", "ground"],
        "functions": ["numeric output"]
    },
    "9-Volt-Battery": {
        "role": "Power source",
        "category": "Power",
        "interfaces": ["power", "ground"],
        "functions": ["portable supply"]
    },
    "Arduino-Mega": {
        "role": "Microcontroller",
        "category": "Controller",
        "interfaces": ["digital", "analog", "serial", "power", "ground"],
        "functions": ["control", "sensing", "communication"]
    },
    "Arduino-Nano": {
        "role": "Microcontroller",
        "category": "Controller",
        "interfaces": ["digital", "analog", "serial", "power", "ground"],
        "functions": ["control", "sensing", "communication"]
    },
    "Arduino-Uno": {
        "role": "Microcontroller",
        "category": "Controller",
        "interfaces": ["digital", "analog", "serial", "power", "ground"],
        "functions": ["control", "sensing", "communication"]
    },
    "BJT-Transistor": {
        "role": "Transistor",
        "category": "Switching",
        "interfaces": ["base", "collector", "emitter"],
        "functions": ["switching", "amplification"]
    },
    "Bluetooth-Module": {
        "role": "Wireless communication module",
        "category": "Communication",
        "interfaces": ["serial", "power", "ground"],
        "functions": ["wireless communication"]
    },
    "Breadboard": {
        "role": "Prototyping platform",
        "category": "Prototyping",
        "interfaces": ["electrical nodes"],
        "functions": ["temporary circuit interconnection"]
    },
    "Bridge-Rectifier": {
        "role": "AC to DC rectifier",
        "category": "Power",
        "interfaces": ["AC input", "DC output"],
        "functions": ["rectification"]
    },
    "Buck-Converter": {
        "role": "Voltage regulator",
        "category": "Power",
        "interfaces": ["input power", "regulated output", "ground"],
        "functions": ["voltage conversion"]
    },
    "Buzzer": {
        "role": "Audible indicator",
        "category": "Actuator",
        "interfaces": ["digital", "power", "ground"],
        "functions": ["audio indication"]
    },
    "Capacitor-10mf": {
        "role": "Capacitor",
        "category": "Passive",
        "interfaces": ["electrical"],
        "functions": ["filtering", "energy storage", "timing"]
    },
    "Capacitor-470mf": {
        "role": "Capacitor",
        "category": "Passive",
        "interfaces": ["electrical"],
        "functions": ["filtering", "energy storage", "smoothing"]
    },
    "DC-Motor": {
        "role": "Rotary actuator",
        "category": "Actuator",
        "interfaces": ["power"],
        "functions": ["mechanical motion"]
    },
    "Diode": {
        "role": "Semiconductor diode",
        "category": "Semiconductor",
        "interfaces": ["anode", "cathode"],
        "functions": ["rectification", "protection", "current steering"]
    },
    "ESP32": {
        "role": "Microcontroller",
        "category": "Controller",
        "interfaces": ["digital", "analog", "serial", "wireless", "power", "ground"],
        "functions": ["control", "sensing", "wireless communication"]
    },
    "ESP32-CAM": {
        "role": "Camera-enabled microcontroller",
        "category": "Controller",
        "interfaces": ["digital", "serial", "wireless", "camera", "power", "ground"],
        "functions": ["control", "vision", "wireless communication"]
    },
    "FT-232-USB-Serial-Module": {
        "role": "USB to UART interface",
        "category": "Communication",
        "interfaces": ["USB", "UART", "power", "ground"],
        "functions": ["serial communication"]
    },
    "Film-Capacitor": {
        "role": "Film capacitor",
        "category": "Passive",
        "interfaces": ["electrical"],
        "functions": ["filtering", "coupling", "timing"]
    },
    "Fuse": {
        "role": "Overcurrent protection",
        "category": "Protection",
        "interfaces": ["series electrical"],
        "functions": ["fault protection"]
    },
    "Fuse-Base": {
        "role": "Fuse holder",
        "category": "Protection",
        "interfaces": ["electrical"],
        "functions": ["fuse mounting"]
    },
    "GSM-Module": {
        "role": "Cellular communication module",
        "category": "Communication",
        "interfaces": ["serial", "power", "ground"],
        "functions": ["cellular communication"]
    },
    "Gas-Sensor": {
        "role": "Gas sensing device",
        "category": "Sensor",
        "interfaces": ["analog", "digital", "power", "ground"],
        "functions": ["gas detection"]
    },
    "Heat-Sink": {
        "role": "Thermal management",
        "category": "Thermal",
        "interfaces": ["thermal"],
        "functions": ["heat dissipation"]
    },
    "High-Voltage-Ceramic-Capacitor": {
        "role": "High-voltage capacitor",
        "category": "Passive",
        "interfaces": ["electrical"],
        "functions": ["filtering", "energy storage"]
    },
    "Humidity-Sensor": {
        "role": "Humidity sensing device",
        "category": "Sensor",
        "interfaces": ["analog", "digital", "power", "ground"],
        "functions": ["humidity measurement"]
    },
    "IC-Base-14-Pin": {
        "role": "14-pin IC socket",
        "category": "Interconnect",
        "interfaces": ["IC pins"],
        "functions": ["IC mounting"]
    },
    "IC-Base-28-Pin": {
        "role": "28-pin IC socket",
        "category": "Interconnect",
        "interfaces": ["IC pins"],
        "functions": ["IC mounting"]
    },
    "IC-Chip": {
        "role": "Integrated circuit",
        "category": "Integrated Circuit",
        "interfaces": ["power", "ground", "signal"],
        "functions": ["signal processing", "control"]
    },
    "IGBT": {
        "role": "Power transistor",
        "category": "Switching",
        "interfaces": ["gate", "collector", "emitter"],
        "functions": ["power switching"]
    },
    "IR-Sensor": {
        "role": "Infrared sensor",
        "category": "Sensor",
        "interfaces": ["digital", "analog", "power", "ground"],
        "functions": ["object detection"]
    },
    "Inductor": {
        "role": "Inductor",
        "category": "Passive",
        "interfaces": ["electrical"],
        "functions": ["energy storage", "filtering", "inductive conversion"]
    },
    "Keypad": {
        "role": "User input device",
        "category": "Input",
        "interfaces": ["digital", "ground"],
        "functions": ["user input"]
    },
    "LCD-Display": {
        "role": "Display",
        "category": "Display",
        "interfaces": ["digital", "power", "ground"],
        "functions": ["text output"]
    },
    "LDR-Sensor": {
        "role": "Light-dependent sensor",
        "category": "Sensor",
        "interfaces": ["analog", "electrical"],
        "functions": ["light sensing"]
    },
    "LED-Light": {
        "role": "Light-emitting diode",
        "category": "Indicator",
        "interfaces": ["anode", "cathode"],
        "functions": ["visual indication"]
    },
    "Low-Voltage-Ceramic-Capacitor": {
        "role": "Ceramic capacitor",
        "category": "Passive",
        "interfaces": ["electrical"],
        "functions": ["filtering", "decoupling"]
    },
    "MLC-Capacitor": {
        "role": "Multilayer ceramic capacitor",
        "category": "Passive",
        "interfaces": ["electrical"],
        "functions": ["filtering", "decoupling"]
    },
    "MOSFET": {
        "role": "Field-effect transistor",
        "category": "Switching",
        "interfaces": ["gate", "drain", "source"],
        "functions": ["switching", "power control"]
    },
    "Motion-Sensor": {
        "role": "Motion sensing device",
        "category": "Sensor",
        "interfaces": ["digital", "power", "ground"],
        "functions": ["motion detection"]
    },
    "Motor-Driver": {
        "role": "Motor control driver",
        "category": "Driver",
        "interfaces": ["control", "motor power", "ground"],
        "functions": ["motor control", "power switching"]
    },
    "NTC-Thermistor": {
        "role": "Temperature-dependent resistor",
        "category": "Sensor",
        "interfaces": ["electrical"],
        "functions": ["temperature sensing"]
    },
    "OLED-Display": {
        "role": "Display",
        "category": "Display",
        "interfaces": ["digital", "power", "ground"],
        "functions": ["text output", "graphics output"]
    },
    "Pin-Header": {
        "role": "Electrical connector",
        "category": "Interconnect",
        "interfaces": ["electrical pins"],
        "functions": ["electrical interconnection"]
    },
    "Push-Switch": {
        "role": "Momentary switch",
        "category": "Input",
        "interfaces": ["electrical"],
        "functions": ["user input", "switching"]
    },
    "RFID-Scanner": {
        "role": "RFID reader",
        "category": "Communication",
        "interfaces": ["digital", "serial", "power", "ground"],
        "functions": ["identification"]
    },
    "Raindrops-Module": {
        "role": "Rain detection module",
        "category": "Sensor",
        "interfaces": ["analog", "digital", "power", "ground"],
        "functions": ["water detection"]
    },
    "Relay-Module": {
        "role": "Electromechanical switching module",
        "category": "Driver",
        "interfaces": ["control", "load", "power", "ground"],
        "functions": ["load switching", "isolation"]
    },
    "Resistor": {
        "role": "Resistive passive component",
        "category": "Passive",
        "interfaces": ["electrical"],
        "functions": ["current limiting", "biasing", "voltage division"]
    },
    "Rocker-Switch": {
        "role": "Mechanical switch",
        "category": "Input",
        "interfaces": ["electrical"],
        "functions": ["user input", "power switching"]
    },
    "Servo-Motor": {
        "role": "Position-controlled actuator",
        "category": "Actuator",
        "interfaces": ["control", "power", "ground"],
        "functions": ["position control"]
    },
    "Soil-Moisture-Sensor": {
        "role": "Soil moisture sensing device",
        "category": "Sensor",
        "interfaces": ["analog", "digital", "power", "ground"],
        "functions": ["moisture measurement"]
    },
    "Sonar-Sensor": {
        "role": "Ultrasonic distance sensor",
        "category": "Sensor",
        "interfaces": ["trigger", "echo", "power", "ground"],
        "functions": ["distance measurement"]
    },
    "TCRT5000": {
        "role": "Reflective optical sensor",
        "category": "Sensor",
        "interfaces": ["analog", "digital", "power", "ground"],
        "functions": ["object detection", "line detection"]
    },
    "Tact-Switch": {
        "role": "Tactile switch",
        "category": "Input",
        "interfaces": ["electrical"],
        "functions": ["user input"]
    },
    "Taper-Potentiometer": {
        "role": "Variable resistor",
        "category": "Passive",
        "interfaces": ["electrical"],
        "functions": ["adjustable voltage division", "analog control"]
    },
    "Trimmer-Potentiometer": {
        "role": "Adjustable resistor",
        "category": "Passive",
        "interfaces": ["electrical"],
        "functions": ["calibration", "adjustable voltage division"]
    },
    "Water-Sensor": {
        "role": "Water detection device",
        "category": "Sensor",
        "interfaces": ["analog", "digital", "power", "ground"],
        "functions": ["water detection"]
    },
    "Zener-Diode": {
        "role": "Voltage reference and protection diode",
        "category": "Protection",
        "interfaces": ["anode", "cathode"],
        "functions": ["voltage regulation", "clamping", "protection"]
    },
    "OP-Amp": {
        "role": "Operational amplifier",
        "category": "Analog",
        "interfaces": ["input", "output", "power", "ground"],
        "functions": ["amplification", "comparison", "signal conditioning"]
    },
    "Cable": {
        "role": "Electrical conductor",
        "category": "Interconnect",
        "interfaces": ["electrical"],
        "functions": ["signal transmission", "power transmission"]
    },
    "Generic-Capacitor": {
        "role": "Capacitor",
        "category": "Passive",
        "interfaces": ["electrical"],
        "functions": ["filtering", "energy storage", "coupling"]
    },
    "Variable-Resistor": {
        "role": "Variable resistor",
        "category": "Passive",
        "interfaces": ["electrical"],
        "functions": ["adjustable resistance", "voltage division"]
    }
}


def get_component_ontology(class_name):
    return COMPONENT_ONTOLOGY.get(class_name)


def get_all_component_classes():
    return list(COMPONENT_ONTOLOGY.keys())
