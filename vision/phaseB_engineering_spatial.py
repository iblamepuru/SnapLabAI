from ultralytics import YOLO
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engineering_state import EngineeringStateEngine
from spatial_engine import SpatialRelationshipEngine
from connection_engine import ConnectionInferenceEngine
from wire_detector import WireDetector
from wire_association import WireComponentAssociation
from connection_graph import ConnectionGraph


MODEL = r"runs\detect\runs\snaplab\phaseB_61class\weights\best.pt"


# --------------------------------------------------
# MODELS / ENGINES
# --------------------------------------------------

model = YOLO(MODEL)

state_engine = EngineeringStateEngine(
    lost_after=10,
    remove_after=60,
    history_size=7
)

spatial_engine = SpatialRelationshipEngine(
    near_distance=150
)

connection_engine = ConnectionInferenceEngine(
    near_distance=150,
    connection_threshold=0.70
)

wire_detector = WireDetector()

wire_association = WireComponentAssociation(
    endpoint_distance=100
)

connection_graph = ConnectionGraph()


# --------------------------------------------------
# CAMERA
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")


print("SnapLab-AI - Live Engineering + Spatial State")
print("YOLO11n + ByteTrack + Temporal Stabilization")
print("Engineering State + Spatial Relationships")
print("Confidence threshold: 0.75")
print("ACTIVE -> LOST -> REMOVED")
print("Press Q to quit.")


frame_count = 0


# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera frame read failed.")
        break


    # Advance state
    state_engine.update_frame()


    # --------------------------------------------------
    # YOLO + BYTE TRACK
    # --------------------------------------------------

    results = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        imgsz=640,
        conf=0.75,
        verbose=False
    )

    result = results[0]


    # --------------------------------------------------
    # UPDATE ENGINEERING STATE
    # --------------------------------------------------

    if (
        result.boxes is not None
        and result.boxes.id is not None
    ):

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

            state_engine.update(
                track_id=track_id,
                class_name=raw_class,
                confidence=float(confidence),
                bbox=bbox
            )


    # --------------------------------------------------
    # LIFECYCLE
    # --------------------------------------------------

    state_engine.update_lifecycle()


    # --------------------------------------------------
    # GET CURRENT ENGINEERING STATE
    # --------------------------------------------------

    state = state_engine.get_state()


    # --------------------------------------------------
    # LIVE WIRE DETECTION
    # --------------------------------------------------

    wires = wire_detector.detect(frame)


    # Only currently active components participate
    # in spatial analysis.
    active_components = {
        track_id: component
        for track_id, component in state.items()
        if component["status"] == "ACTIVE"
    }


    # --------------------------------------------------
    # SPATIAL RELATIONSHIPS
    # --------------------------------------------------

    relationships = spatial_engine.analyze(
        active_components
    )


    # --------------------------------------------------
    # DRAW YOLO DETECTIONS
    # --------------------------------------------------

    annotated = result.plot()


    # --------------------------------------------------
    # DRAW STABILIZED ENGINEERING LABELS
    # --------------------------------------------------

    for component in state.values():

        if component["status"] not in [
            "ACTIVE",
            "LOST"
        ]:
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


    # --------------------------------------------------
    # COUNTERS
    # --------------------------------------------------

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


    # --------------------------------------------------
    # DISPLAY STATUS
    # --------------------------------------------------

    cv2.putText(
        annotated,
        f"ACTIVE: {active_count} | LOST: {lost_count}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )


    cv2.imshow(
        "SnapLab-AI - Engineering + Spatial State",
        annotated
    )


    # --------------------------------------------------
    # TERMINAL OUTPUT
    # --------------------------------------------------

    frame_count += 1

    if frame_count % 30 == 0:

        state_engine.print_state()

        print(
            "\n========== SPATIAL RELATIONSHIPS =========="
        )

        if relationships:

            for relation in relationships:

                print(
                    f'ID {relation["from_id"]} '
                    f'({relation["from_class"]}) -> '
                    f'ID {relation["to_id"]} '
                    f'({relation["to_class"]}) | '
                    f'Distance: {relation["distance"]} px | '
                    f'{", ".join(relation["relationships"])}'
                )

        else:

            print("No spatial relationships available.")


        # --------------------------------------------------
        # CONNECTION INFERENCE
        # --------------------------------------------------

        connection_results = connection_engine.analyze(
            relationships
        )

        print(
            "\n========== CONNECTION EVIDENCE =========="
        )

        if connection_results:

            for connection in connection_results:

                print(
                    f'{connection["from_class"]} -> '
                    f'{connection["to_class"]} | '
                    f'Spatial: {connection["spatial_score"]} | '
                    f'Wire: {connection["wire_score"]} | '
                    f'Terminal: {connection["terminal_score"]} | '
                    f'Score: {connection["connection_score"]} | '
                    f'{connection["status"]}'
                )

        else:

            print("No connection evidence available.")


        # --------------------------------------------------
        # CONNECTION GRAPH
        # --------------------------------------------------

        connection_graph = ConnectionGraph()


        # Add ACTIVE components as graph nodes

        for component in active_components.values():

            connection_graph.add_component(
                track_id=component["track_id"],
                class_name=component["class_name"],
                confidence=component["confidence"]
            )


        # Associate detected wires with ACTIVE components

        associations = wire_association.analyze(
            wires,
            active_components
        )


        wire_counter = 0

        for association in associations:

            component_a = association["component_a"]
            component_b = association["component_b"]


            # A valid graph edge requires
            # both wire endpoints to be associated
            # with components.

            if (
                component_a is None
                or component_b is None
            ):
                continue


            # Avoid self-connections.

            if (
                component_a["track_id"]
                ==
                component_b["track_id"]
            ):
                continue


            wire_counter += 1

            wire_id = f"wire_{wire_counter:03d}"


            # Find matching connection evidence.
            # The graph edge remains POSSIBLE because
            # visual association does not prove
            # electrical continuity.

            score = 0.0

            for connection in connection_results:

                same_pair = (
                    connection["from_id"]
                    == component_a["track_id"]
                    and
                    connection["to_id"]
                    == component_b["track_id"]
                )

                reverse_pair = (
                    connection["from_id"]
                    == component_b["track_id"]
                    and
                    connection["to_id"]
                    == component_a["track_id"]
                )

                if same_pair or reverse_pair:

                    score = connection["connection_score"]
                    break


            connection_graph.add_connection(
                from_id=component_a["track_id"],
                to_id=component_b["track_id"],
                wire_id=wire_id,
                score=score,
                status="POSSIBLE_CONNECTION"
            )


        print(
            "\n========== LIVE CONNECTION GRAPH =========="
        )

        connection_graph.print_graph()


    # --------------------------------------------------
    # QUIT
    # --------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# CLEANUP
# --------------------------------------------------

cap.release()
cv2.destroyAllWindows()


print("\n========== FINAL ENGINEERING STATE ==========")
state_engine.print_state()


print(
    "\nLive engineering + spatial integration test completed."
)
