# SnapLab AI - Component Knowledge Base
# Engineering metadata for the 65-class detection ontology.

COMPONENT_KNOWLEDGE = {

    "1-5-Volt-Battery": {
        "category": "Power Source",
        "function": "Provides low-voltage DC power",
        "pins": "Positive and negative terminals",
        "typical_role": "Circuit power supply"
    },

    "3-3-Volt-Battery": {
        "category": "Power Source",
        "function": "Provides approximately 3.3 V DC power",
        "pins": "Positive and negative terminals",
        "typical_role": "Low-voltage circuit power"
    },

    "7-Segment-Display": {
        "category": "Display",
        "function": "Displays numeric characters using LED segments",
        "pins": "Multiple segment and common pins",
        "typical_role": "Numeric visual output"
    },

    "9-Volt-Battery": {
        "category": "Power Source",
        "function": "Provides approximately 9 V DC power",
        "pins": "Positive and negative terminals",
        "typical_role": "Portable circuit power"
    },

    "Arduino-Mega": {
        "category": "Microcontroller Board",
        "function": "Microcontroller development and control",
        "pins": "Digital I/O, analog inputs, power and communication pins",
        "typical_role": "Main controller"
    },

    "Arduino-Nano": {
        "category": "Microcontroller Board",
        "function": "Compact microcontroller development and control",
        "pins": "Digital I/O, analog inputs, power and communication pins",
        "typical_role": "Embedded controller"
    },

    "Arduino-Uno": {
        "category": "Microcontroller Board",
        "function": "Microcontroller development and control",
        "pins": "Digital I/O, analog inputs, power and communication pins",
        "typical_role": "Main controller"
    },

    "BJT-Transistor": {
        "category": "Transistor",
        "function": "Electronic switching or amplification",
        "pins": "Base, Collector, Emitter",
        "typical_role": "Switch or amplifier"
    },

    "Bluetooth-Module": {
        "category": "Communication Module",
        "function": "Provides short-range wireless communication",
        "pins": "Power, ground and serial/interface pins",
        "typical_role": "Wireless communication"
    },

    "Breadboard": {
        "category": "Prototyping",
        "function": "Provides temporary electrical connections without soldering",
        "pins": "Connected terminal groups",
        "typical_role": "Circuit prototyping"
    },

    "Bridge-Rectifier": {
        "category": "Power Electronics",
        "function": "Converts AC into pulsating DC",
        "pins": "AC inputs and positive/negative DC outputs",
        "typical_role": "AC-to-DC rectification"
    },

    "Buck-Converter": {
        "category": "Power Electronics",
        "function": "Steps DC voltage down",
        "pins": "Input and output power terminals",
        "typical_role": "Voltage regulation"
    },

    "Buzzer": {
        "category": "Actuator",
        "function": "Produces an audible alert",
        "pins": "Power and ground/control",
        "typical_role": "Audio indication"
    },

    "Capacitor-10mf": {
        "category": "Passive Component",
        "function": "Stores electrical charge and filters voltage",
        "pins": "Two terminals",
        "typical_role": "Filtering and energy storage"
    },

    "Capacitor-470mf": {
        "category": "Passive Component",
        "function": "Stores electrical charge and filters voltage",
        "pins": "Two terminals",
        "typical_role": "Filtering and energy storage"
    },

    "DC-Motor": {
        "category": "Actuator",
        "function": "Converts electrical energy into rotational motion",
        "pins": "Two motor terminals",
        "typical_role": "Mechanical actuation"
    },

    "Diode": {
        "category": "Semiconductor",
        "function": "Allows current primarily in one direction",
        "pins": "Anode and Cathode",
        "typical_role": "Rectification and protection"
    },

    "ESP32": {
        "category": "Microcontroller / SoC",
        "function": "Embedded processing with wireless connectivity",
        "pins": "GPIO, power, ground and communication interfaces",
        "typical_role": "IoT controller"
    },

    "ESP32-CAM": {
        "category": "Embedded Vision Module",
        "function": "Provides embedded processing and camera capability",
        "pins": "GPIO, power, ground and communication interfaces",
        "typical_role": "IoT vision system"
    },

    "FT-232-USB-Serial-Module": {
        "category": "Communication Module",
        "function": "Converts USB communication to serial UART",
        "pins": "USB interface and UART pins",
        "typical_role": "USB-to-serial communication"
    },

    "Film-Capacitor": {
        "category": "Passive Component",
        "function": "Stores charge and filters electrical signals",
        "pins": "Two terminals",
        "typical_role": "Filtering and signal coupling"
    },

    "Fuse": {
        "category": "Protection",
        "function": "Protects circuits from excessive current",
        "pins": "Two terminals",
        "typical_role": "Overcurrent protection"
    },

    "Fuse-Base": {
        "category": "Protection",
        "function": "Provides a mounting interface for a fuse",
        "pins": "Input and output terminals",
        "typical_role": "Fuse mounting and protection"
    },

    "GSM-Module": {
        "category": "Communication Module",
        "function": "Provides cellular communication",
        "pins": "Power, ground and communication/control pins",
        "typical_role": "Cellular IoT communication"
    },

    "Gas-Sensor": {
        "category": "Sensor",
        "function": "Detects or measures gases",
        "pins": "Power, ground and signal pins",
        "typical_role": "Gas monitoring"
    },

    "Heat-Sink": {
        "category": "Thermal Management",
        "function": "Dissipates heat from electronic components",
        "pins": "No electrical pins",
        "typical_role": "Thermal management"
    },

    "Humidity-Sensor": {
        "category": "Sensor",
        "function": "Measures humidity",
        "pins": "Power, ground and signal/interface pins",
        "typical_role": "Environmental sensing"
    },

    "IC-Chip": {
        "category": "Integrated Circuit",
        "function": "Performs application-specific electronic processing",
        "pins": "Multiple electrical pins",
        "typical_role": "Signal processing or control"
    },

    "IGBT": {
        "category": "Power Semiconductor",
        "function": "High-power electronic switching",
        "pins": "Gate, Collector, Emitter",
        "typical_role": "Power switching"
    },

    "IR-Sensor": {
        "category": "Sensor",
        "function": "Detects infrared radiation or objects",
        "pins": "Power, ground and signal pins",
        "typical_role": "Object or proximity detection"
    },

    "Inductor": {
        "category": "Passive Component",
        "function": "Stores energy in a magnetic field",
        "pins": "Two terminals",
        "typical_role": "Filtering and energy storage"
    },

    "Keypad": {
        "category": "Input Device",
        "function": "Provides multiple user input keys",
        "pins": "Row and column connections",
        "typical_role": "User input"
    },

    "LCD-Display": {
        "category": "Display",
        "function": "Displays text and graphical information",
        "pins": "Power and communication/control pins",
        "typical_role": "Visual output"
    },

    "LDR-Sensor": {
        "category": "Sensor",
        "function": "Changes resistance according to light intensity",
        "pins": "Two terminals",
        "typical_role": "Light sensing"
    },

    "LED-Light": {
        "category": "Indicator",
        "function": "Produces light when electrically driven",
        "pins": "Anode and Cathode",
        "typical_role": "Visual indication"
    },

    "MOSFET": {
        "category": "Transistor",
        "function": "Electronic switching or amplification",
        "pins": "Gate, Drain, Source",
        "typical_role": "Switch or power control"
    },

    "Motion-Sensor": {
        "category": "Sensor",
        "function": "Detects motion",
        "pins": "Power, ground and signal pins",
        "typical_role": "Motion detection"
    },

    "Motor-Driver": {
        "category": "Driver",
        "function": "Controls motors using electronic switching",
        "pins": "Power, ground, control and motor outputs",
        "typical_role": "Motor control"
    },

    "NTC-Thermistor": {
        "category": "Sensor / Passive Component",
        "function": "Resistance decreases as temperature increases",
        "pins": "Two terminals",
        "typical_role": "Temperature sensing"
    },

    "OLED-Display": {
        "category": "Display",
        "function": "Displays text and graphics using OLED pixels",
        "pins": "Power and communication pins",
        "typical_role": "Visual output"
    },

    "Pin-Header": {
        "category": "Interconnect",
        "function": "Provides electrical connection between boards and modules",
        "pins": "Multiple connector pins",
        "typical_role": "Electrical interconnection"
    },

    "Push-Switch": {
        "category": "Input Device",
        "function": "Provides momentary user input",
        "pins": "Typically two or more switch terminals",
        "typical_role": "User input"
    },

    "RFID-Scanner": {
        "category": "Communication / Sensor",
        "function": "Reads RFID tags",
        "pins": "Power and communication/interface pins",
        "typical_role": "Identification and tracking"
    },

    "Raindrops-Module": {
        "category": "Sensor",
        "function": "Detects water or raindrops",
        "pins": "Power, ground and signal pins",
        "typical_role": "Rain detection"
    },

    "Relay-Module": {
        "category": "Switching Module",
        "function": "Controls higher-power electrical loads using a low-power control signal",
        "pins": "Power, control and relay contacts",
        "typical_role": "Electrical isolation and load switching"
    },

    "Resistor": {
        "category": "Passive Component",
        "function": "Limits current and creates voltage relationships",
        "pins": "Two terminals",
        "typical_role": "Current limiting and voltage division"
    },

    "Rocker-Switch": {
        "category": "Input / Switching",
        "function": "Provides manual electrical switching",
        "pins": "Switch terminals",
        "typical_role": "Power or signal switching"
    },

    "Servo-Motor": {
        "category": "Actuator",
        "function": "Provides controlled angular motion",
        "pins": "Power, ground and control",
        "typical_role": "Position control"
    },

    "Soil-Moisture-Sensor": {
        "category": "Sensor",
        "function": "Measures soil moisture level",
        "pins": "Power, ground and signal pins",
        "typical_role": "Agricultural sensing"
    },

    "Sonar-Sensor": {
        "category": "Sensor",
        "function": "Measures distance using ultrasonic waves",
        "pins": "Power, ground, trigger and echo/interface pins",
        "typical_role": "Distance measurement"
    },

    "TCRT5000": {
        "category": "Optical Sensor",
        "function": "Detects reflected infrared light",
        "pins": "Power, ground and output pins",
        "typical_role": "Line or object detection"
    },

    "Tact-Switch": {
        "category": "Input Device",
        "function": "Provides momentary push-button input",
        "pins": "Switch terminals",
        "typical_role": "User input"
    },

    "Taper-Potentiometer": {
        "category": "Variable Resistor",
        "function": "Provides adjustable resistance or voltage division",
        "pins": "Two end terminals and one wiper",
        "typical_role": "Adjustable control"
    },

    "Trimmer-Potentiometer": {
        "category": "Variable Resistor",
        "function": "Provides adjustable resistance for calibration",
        "pins": "Two end terminals and one wiper",
        "typical_role": "Circuit calibration"
    },

    "Water-Sensor": {
        "category": "Sensor",
        "function": "Detects presence or level of water",
        "pins": "Power, ground and signal pins",
        "typical_role": "Water detection"
    },

    "Zener-Diode": {
        "category": "Semiconductor",
        "function": "Maintains a relatively constant voltage in reverse breakdown",
        "pins": "Anode and Cathode",
        "typical_role": "Voltage regulation and protection"
    },

    "OP-Amp": {
        "category": "Analog Integrated Circuit",
        "function": "Amplifies and processes analog signals",
        "pins": "Power, input and output pins",
        "typical_role": "Signal amplification and conditioning"
    },

    "Cable": {
        "category": "Interconnect",
        "function": "Carries electrical signals or power",
        "pins": "Conductive ends",
        "typical_role": "Electrical connection"
    },

    "Generic-Capacitor": {
        "category": "Passive Component",
        "function": "Stores electrical charge and filters signals",
        "pins": "Two terminals",
        "typical_role": "Filtering and energy storage"
    },

    "Variable-Resistor": {
        "category": "Variable Resistor",
        "function": "Provides adjustable electrical resistance",
        "pins": "Typically two or three terminals",
        "typical_role": "Adjustable voltage/current control"
    },

    # Classes with intentionally generic metadata
    "IC-Base-14-Pin": {
        "category": "IC Socket",
        "function": "Provides a removable connection for a 14-pin integrated circuit",
        "pins": "14 contacts",
        "typical_role": "IC mounting"
    },

    "IC-Base-28-Pin": {
        "category": "IC Socket",
        "function": "Provides a removable connection for a 28-pin integrated circuit",
        "pins": "28 contacts",
        "typical_role": "IC mounting"
    },

    "High-Voltage-Ceramic-Capacitor": {
        "category": "Passive Component",
        "function": "Stores charge and supports high-voltage filtering applications",
        "pins": "Two terminals",
        "typical_role": "High-voltage filtering"
    },

    "Low-Voltage-Ceramic-Capacitor": {
        "category": "Passive Component",
        "function": "Stores charge and filters electrical signals",
        "pins": "Two terminals",
        "typical_role": "Filtering"
    },

    "MLC-Capacitor": {
        "category": "Passive Component",
        "function": "Provides multilayer ceramic capacitance",
        "pins": "Two terminals",
        "typical_role": "Filtering and decoupling"
    },

}


def get_component_info(component_name):
    """Return engineering metadata for a detected component."""
    return COMPONENT_KNOWLEDGE.get(
        component_name,
        {
            "category": "Electronic Component",
            "function": "Engineering function not yet defined",
            "pins": "Unknown",
            "typical_role": "Requires further inspection"
        }
    )
