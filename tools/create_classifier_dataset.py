import os
import cv2
import shutil
from pathlib import Path

SOURCE_ROOT = Path(
    r"datasets\snaplab_master"
)

OUTPUT_ROOT = Path(
    r"datasets\component_classifier"
)

PADDING = 8

MIN_CROP_SIZE = 16

JPEG_QUALITY = 95


# ============================================================
# MASTER ONTOLOGY
# ============================================================

CLASS_NAMES = [
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


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

def create_output_directories():

    for split in [
        "train",
        "val",
        "test"
    ]:

        for class_name in CLASS_NAMES:

            directory = (
                OUTPUT_ROOT
                / split
                / class_name
            )

            directory.mkdir(
                parents=True,
                exist_ok=True
            )


# ============================================================
# FIND IMAGE
# ============================================================

def find_image(
    image_directory,
    stem
):
    """
    Find image corresponding to a YOLO label file.
    """

    for extension in IMAGE_EXTENSIONS:

        image_path = (
            image_directory
            / f"{stem}{extension}"
        )

        if image_path.exists():

            return image_path

    return None


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_split(
    split
):

    image_directory = (
        SOURCE_ROOT
        / "images"
        / split
    )

    label_directory = (
        SOURCE_ROOT
        / "labels"
        / split
    )

    if not image_directory.exists():

        print(
            f"Missing image directory: "
            f"{image_directory}"
        )

        return 0, 0, 0

    if not label_directory.exists():

        print(
            f"Missing label directory: "
            f"{label_directory}"
        )

        return 0, 0, 0

    labels = list(
        label_directory.glob("*.txt")
    )

    image_count = 0
    crop_count = 0
    skipped_count = 0

    for label_path in labels:

        image_path = find_image(
            image_directory,
            label_path.stem
        )

        if image_path is None:

            skipped_count += 1

            continue

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            skipped_count += 1

            continue

        height, width = (
            image.shape[:2]
        )

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as file:

            lines = file.readlines()

        image_crop_index = 0

        for line_index, line in enumerate(
            lines
        ):

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            try:

                class_id = int(
                    parts[0]
                )

                x_center = float(
                    parts[1]
                )

                y_center = float(
                    parts[2]
                )

                box_width = float(
                    parts[3]
                )

                box_height = float(
                    parts[4]
                )

            except ValueError:

                continue

            if not (
                0 <= class_id < len(
                    CLASS_NAMES
                )
            ):

                continue

            # ------------------------------------------------
            # YOLO normalized coordinates
            # → pixel coordinates
            # ------------------------------------------------

            center_x = (
                x_center * width
            )

            center_y = (
                y_center * height
            )

            pixel_width = (
                box_width * width
            )

            pixel_height = (
                box_height * height
            )

            x1 = int(
                center_x -
                pixel_width / 2
            )

            y1 = int(
                center_y -
                pixel_height / 2
            )

            x2 = int(
                center_x +
                pixel_width / 2
            )

            y2 = int(
                center_y +
                pixel_height / 2
            )

            # ------------------------------------------------
            # Padding
            # ------------------------------------------------

            x1 -= PADDING
            y1 -= PADDING
            x2 += PADDING
            y2 += PADDING

            # ------------------------------------------------
            # Clamp to image
            # ------------------------------------------------

            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                width,
                x2
            )

            y2 = min(
                height,
                y2
            )

            crop_width = (
                x2 - x1
            )

            crop_height = (
                y2 - y1
            )

            if (
                crop_width < MIN_CROP_SIZE
                or
                crop_height < MIN_CROP_SIZE
            ):

                skipped_count += 1

                continue

            # ------------------------------------------------
            # Crop component
            # ------------------------------------------------

            crop = image[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:

                skipped_count += 1

                continue

            # ------------------------------------------------
            # Classification output
            # ------------------------------------------------

            class_name = CLASS_NAMES[
                class_id
            ]

            output_directory = (
                OUTPUT_ROOT
                / split
                / class_name
            )

            output_directory.mkdir(
                parents=True,
                exist_ok=True
            )

            # ------------------------------------------------
            # Unique crop filename
            # ------------------------------------------------

            filename = (
                f"{label_path.stem}"
                f"_box{line_index}"
                f"_crop{image_crop_index}.jpg"
            )

            output_path = (
                output_directory
                / filename
            )

            # ------------------------------------------------
            # Save crop
            # ------------------------------------------------

            cv2.imwrite(
                str(output_path),
                crop,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    JPEG_QUALITY
                ]
            )

            crop_count += 1
            image_crop_index += 1

        image_count += 1

    return (
        image_count,
        crop_count,
        skipped_count
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 65)
    print(
        "SNAPLAB AI — CLASSIFIER DATASET CREATION"
    )
    print("=" * 65)

    print(
        f"Source: {SOURCE_ROOT}"
    )

    print(
        f"Output: {OUTPUT_ROOT}"
    )

    print(
        f"Classes: {len(CLASS_NAMES)}"
    )

    print(
        f"Padding: {PADDING}px"
    )

    print()

    # --------------------------------------------------------
    # Check source
    # --------------------------------------------------------

    if not SOURCE_ROOT.exists():

        raise FileNotFoundError(
            f"Source dataset not found: "
            f"{SOURCE_ROOT}"
        )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    create_output_directories()

    total_images = 0
    total_crops = 0
    total_skipped = 0

    # --------------------------------------------------------
    # Process splits
    # --------------------------------------------------------

    for split in [
        "train",
        "val",
        "test"
    ]:

        print()
        print(
            f"Processing {split}..."
        )

        images, crops, skipped = (
            process_split(split)
        )

        total_images += images
        total_crops += crops
        total_skipped += skipped

        print(
            f"  Images processed : {images}"
        )

        print(
            f"  Component crops  : {crops}"
        )

        print(
            f"  Skipped          : {skipped}"
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 65)

    print(
        "CLASSIFIER DATASET CREATION COMPLETE"
    )

    print("=" * 65)

    print(
        f"Total images processed : "
        f"{total_images}"
    )

    print(
        f"Total component crops  : "
        f"{total_crops}"
    )

    print(
        f"Total skipped          : "
        f"{total_skipped}"
    )

    print(
        f"Output directory       : "
        f"{OUTPUT_ROOT}"
    )

    print("=" * 65)


if __name__ == "__main__":

    main()