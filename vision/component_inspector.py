import os
import sys
import cv2
import gradio as gr
from ultralytics import YOLO

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vision.vlm.vlm_prompt import build_engineering_prompt
from vision.vlm.vlm_engine import VLMEngine
from vision.component_knowledge import get_component_info
from vision.component_fusion_engine import fuse_yolo_results
from vision.circuit_intelligence import CircuitIntelligenceEngine
from engineering_state import EngineeringStateEngine
from vision.openai_engine import OpenAIEngineeringEngine
from vision.wire_association import WireComponentAssociation
from vision.connection_engine import ConnectionInferenceEngine
from vision.connection_graph import ConnectionGraph


MODEL_PATH = os.path.join(
    ROOT_DIR,
    "runs",
    "detect",
    "runs",
    "snaplab",
    "phaseC_master_65class-2",
    "weights",
    "best.pt"
)

model = YOLO(MODEL_PATH)
vlm_engine = VLMEngine()
openai_engine = OpenAIEngineeringEngine()
wire_association_engine = WireComponentAssociation()
connection_engine = ConnectionInferenceEngine()


def make_crop(image, box, padding=15, scale=2.0):

    x1, y1, x2, y2 = map(int, box)

    h, w = image.shape[:2]

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    if scale != 1.0:
        crop = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    return crop


def draw_fused_results(image, fused_results):

    output = image.copy()

    for fused in fused_results:

        box = fused.get("box")

        if box is None:
            continue

        x1, y1, x2, y2 = map(int, box)

        final_class = fused.get(
            "final_class",
            "Unknown"
        )

        final_confidence = fused.get(
            "final_confidence",
            0.0
        )

        source = fused.get(
            "source",
            "YOLO"
        )

        if source == "MobileNet":
            thickness = 3
        elif source == "YOLO + MobileNet":
            thickness = 3
        else:
            thickness = 2

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            thickness
        )

        label = f"{final_class} {final_confidence:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        text_thickness = 2

        text_size = cv2.getTextSize(
            label,
            font,
            font_scale,
            text_thickness
        )[0]

        label_y = max(
            y1,
            text_size[1] + 8
        )

        cv2.rectangle(
            output,
            (
                x1,
                label_y - text_size[1] - 8
            ),
            (
                x1 + text_size[0] + 8,
                label_y
            ),
            (255, 255, 255),
            -1
        )

        cv2.putText(
            output,
            label,
            (
                x1 + 4,
                label_y - 5
            ),
            font,
            font_scale,
            (0, 0, 0),
            text_thickness
        )

    return output


def build_connection_analysis(
    image,
    state
):

    wire_results = wire_association_engine.analyze(
        image,
        state
    )

    connection_graph = ConnectionGraph()

    for track_id, component in state.items():

        connection_graph.add_component(
            track_id=track_id,
            class_name=component.get(
                "class_name",
                "Unknown"
            ),
            confidence=component.get(
                "confidence",
                0.0
            )
        )

    connection_relationships = []
    wire_scores = {}
    terminal_scores = {}
    wire_metadata = []

    for wire_index, wire_result in enumerate(
        wire_results,
        start=1
    ):

        component_a = wire_result.get(
            "component_a"
        )

        component_b = wire_result.get(
            "component_b"
        )

        if not wire_result.get(
            "valid_connection",
            False
        ):
            wire_metadata.append(
                {
                    "wire_id": wire_index,
                    "valid_connection": False,
                    "endpoint_a": wire_result.get(
                        "endpoint_a"
                    ),
                    "endpoint_b": wire_result.get(
                        "endpoint_b"
                    ),
                    "distance_a": wire_result.get(
                        "distance_a"
                    ),
                    "distance_b": wire_result.get(
                        "distance_b"
                    )
                }
            )
            continue

        from_id = component_a.get(
            "track_id"
        )

        to_id = component_b.get(
            "track_id"
        )

        distance_a = float(
            wire_result.get(
                "distance_a",
                999999
            )
        )

        distance_b = float(
            wire_result.get(
                "distance_b",
                999999
            )
        )

        connection_distance = max(
            distance_a,
            distance_b
        )

        relationship = {
            "from_id": from_id,
            "from_class": component_a.get(
                "class_name",
                "Unknown"
            ),
            "to_id": to_id,
            "to_class": component_b.get(
                "class_name",
                "Unknown"
            ),
            "distance": connection_distance
        }

        connection_relationships.append(
            relationship
        )

        pair = (
            from_id,
            to_id
        )

        reverse_pair = (
            to_id,
            from_id
        )

        wire_scores[pair] = 1.0
        wire_scores[reverse_pair] = 1.0

        terminal_scores[pair] = 0.0
        terminal_scores[reverse_pair] = 0.0

        wire_metadata.append(
            {
                "wire_id": wire_index,
                "valid_connection": True,
                "endpoint_a": wire_result.get(
                    "endpoint_a"
                ),
                "endpoint_b": wire_result.get(
                    "endpoint_b"
                ),
                "distance_a": distance_a,
                "distance_b": distance_b,
                "from_id": from_id,
                "to_id": to_id
            }
        )

    connection_results = connection_engine.analyze(
        connection_relationships,
        wire_scores=wire_scores,
        terminal_scores=terminal_scores
    )

    connection_result_by_pair = {}

    for result in connection_results:

        pair = (
            result["from_id"],
            result["to_id"]
        )

        reverse_pair = (
            result["to_id"],
            result["from_id"]
        )

        connection_result_by_pair[pair] = result
        connection_result_by_pair[reverse_pair] = result

    for wire in wire_metadata:

        if not wire.get(
            "valid_connection",
            False
        ):
            continue

        pair = (
            wire["from_id"],
            wire["to_id"]
        )

        result = connection_result_by_pair.get(
            pair
        )

        if result is None:
            continue

        connection_graph.add_connection(
            from_id=result["from_id"],
            to_id=result["to_id"],
            wire_id=wire["wire_id"],
            score=result["connection_score"],
            status=result["status"]
        )

    return (
        wire_results,
        connection_results,
        connection_graph.to_dict(),
        wire_metadata
    )


def format_connection_evidence(
    wire_results,
    connection_results,
    wire_metadata
):

    evidence = {
        "status": "INSUFFICIENT_TOPOLOGY_EVIDENCE",
        "wire_count": len(wire_results),
        "valid_wire_connections": 0,
        "connections": [],
        "message": (
            "Component identity and spatial proximity do not prove "
            "an electrical connection. Physical connectivity requires "
            "visible wire, terminal, breadboard, or topology evidence."
        )
    }

    connection_lookup = {}

    for result in connection_results:

        pair = (
            result["from_id"],
            result["to_id"]
        )

        reverse_pair = (
            result["to_id"],
            result["from_id"]
        )

        connection_lookup[pair] = result
        connection_lookup[reverse_pair] = result

    for wire in wire_metadata:

        if not wire.get(
            "valid_connection",
            False
        ):
            continue

        pair = (
            wire["from_id"],
            wire["to_id"]
        )

        result = connection_lookup.get(
            pair
        )

        if result is None:
            continue

        connection_entry = {
            "wire_id": wire["wire_id"],
            "from_id": result["from_id"],
            "from_class": result["from_class"],
            "to_id": result["to_id"],
            "to_class": result["to_class"],
            "endpoint_a": wire["endpoint_a"],
            "endpoint_b": wire["endpoint_b"],
            "distance_a": wire["distance_a"],
            "distance_b": wire["distance_b"],
            "spatial_score": result["spatial_score"],
            "wire_score": result["wire_score"],
            "terminal_score": result["terminal_score"],
            "connection_score": result["connection_score"],
            "status": result["status"]
        }

        evidence["connections"].append(
            connection_entry
        )

    evidence["valid_wire_connections"] = len(
        evidence["connections"]
    )

    if evidence["connections"]:

        evidence["status"] = (
            "CONNECTION_EVIDENCE_AVAILABLE"
        )

    elif evidence["wire_count"] > 0:

        evidence["status"] = (
            "WIRE_DETECTED_BUT_NO_VALID_CONNECTION"
        )

    return evidence


def append_connection_report(
    report,
    connection_evidence,
    connection_graph
):

    report.append(
        "### Connection Evidence\n\n"
    )

    if connection_evidence["connections"]:

        for connection in connection_evidence["connections"]:

            report.append(
                f"#### Wire {connection['wire_id']}\n\n"
                f"- **{connection['from_class']} -> "
                f"{connection['to_class']}**\n"
                f"- Wire association: "
                f"{connection['from_class']} endpoint "
                f"{connection['distance_a']:.1f} px\n"
                f"- Wire association: "
                f"{connection['to_class']} endpoint "
                f"{connection['distance_b']:.1f} px\n"
                f"- Wire evidence: "
                f"{connection['wire_score']:.2f}\n"
                f"- Terminal evidence: "
                f"{connection['terminal_score']:.2f}\n"
                f"- Spatial evidence: "
                f"{connection['spatial_score']:.2f}\n"
                f"- Connection score: "
                f"{connection['connection_score']:.2f}\n"
                f"- Status: "
                f"**{connection['status']}**\n\n"
            )

        report.append(
            "#### Connection Graph\n\n"
        )

        for edge in connection_graph.get(
            "edges",
            []
        ):

            from_node = connection_graph.get(
                "nodes",
                []
            )

            from_class = next(
                (
                    node["class_name"]
                    for node in from_node
                    if node["track_id"]
                    == edge["from_id"]
                ),
                "Unknown"
            )

            to_class = next(
                (
                    node["class_name"]
                    for node in from_node
                    if node["track_id"]
                    == edge["to_id"]
                ),
                "Unknown"
            )

            report.append(
                f"- **{from_class} -> {to_class}** "
                f"| Wire {edge['wire_id']} "
                f"| Score {edge['score']:.2f} "
                f"| {edge['status']}\n"
            )

        report.append("\n")

    elif connection_evidence["wire_count"] > 0:

        report.append(
            "Wire candidates were detected, but no valid "
            "wire-to-component association was established.\n\n"
        )

    else:

        report.append(
            "No wire candidates were detected.\n\n"
        )

    report.append(
        "Connection interpretation is evidence-based. "
        "A possible visual connection does not establish exact "
        "breadboard row, component pin, net topology, or electrical "
        "continuity.\n\n"
    )


def analyze_images(
    images,
    mode,
    confidence,
    refine_mobilenet
):

    if not images:
        return [], [], "Please upload at least one image."

    annotated_gallery = []
    crop_gallery = []
    report = []

    # Aggregated data for the combined reasoning section
    all_fused_results_per_image = []
    all_selected_spatial_per_image = []
    all_selected_engineering_per_image = []
    all_connection_evidence_per_image = []
    all_openai_results = []
    all_component_counts = []
    aggregated_inventory = []
    class_frequency = {}
    per_image_evidence = []

    report.append(
        "# SnapLab AI — Combined Engineering Report\n\n"
        f"**Images analysed:** {len(images)}  "
        f"| **Mode:** {mode}\n\n"
    )

    for image_index, file_path in enumerate(
        images,
        start=1
    ):

        image_path = str(file_path)
        image = cv2.imread(image_path)

        if image is None:
            continue

        result = model.predict(
            image,
            imgsz=640,
            conf=confidence,
            device="cpu",
            verbose=False
        )[0]

        fused_results = fuse_yolo_results(
            image,
            result,
            refine_all=refine_mobilenet
        )

        state_engine = EngineeringStateEngine()

        for track_id, fused in enumerate(
            fused_results,
            start=1
        ):

            fused["track_id"] = track_id

            state_engine.update_fused(
                track_id,
                fused
            )

        state = state_engine.get_state()

        intelligence_engine = CircuitIntelligenceEngine()

        intelligence = intelligence_engine.analyze(
            state
        )

        wire_results = []
        connection_results = []
        connection_graph = {
            "nodes": [],
            "edges": []
        }

        wire_metadata = []

        connection_evidence = {
            "status": "NOT_RUN",
            "wire_count": 0,
            "valid_wire_connections": 0,
            "connections": [],
            "message": (
                "Circuit Analysis mode was not selected."
            )
        }

        if mode == "Circuit Analysis":

            (
                wire_results,
                connection_results,
                connection_graph,
                wire_metadata
            ) = build_connection_analysis(
                image,
                state
            )

            connection_evidence = format_connection_evidence(
                wire_results,
                connection_results,
                wire_metadata
            )

        annotated = draw_fused_results(
            image,
            fused_results
        )

        annotated = cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB
        )

        annotated_gallery.append(
            (
                annotated,
                f"Image {image_index}"
            )
        )

        for fused in fused_results:

            crop = make_crop(
                image,
                fused["box"],
                padding=15,
                scale=2.0
            )

            if crop is None:
                continue

            crop = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2RGB
            )

            final_class = fused.get(
                "final_class",
                "Unknown"
            )

            final_confidence = fused.get(
                "final_confidence",
                0.0
            )

            source = fused.get(
                "source",
                "YOLO"
            )

            crop_gallery.append(
                (
                    crop,
                    f"{final_class} | "
                    f"{final_confidence:.2f} | "
                    f"{source}"
                )
            )

        # ── per-image header ────────────────────────────────────────────────
        report.append(
            f"---\n\n"
            f"## 🖼️ Image {image_index} — {mode}\n\n"
            f"**Components detected:** "
            f"{len(fused_results)}\n\n"
        )
        all_component_counts.append(len(fused_results))

        report.append(
            "### Component Understanding\n\n"
        )

        for fused in fused_results:

            final_class = fused.get(
                "final_class",
                "Unknown"
            )

            final_confidence = fused.get(
                "final_confidence",
                0.0
            )

            detector_class = fused.get(
                "detector_class",
                "Unknown"
            )

            detector_confidence = fused.get(
                "detector_confidence",
                0.0
            )

            classifier_class = fused.get(
                "classifier_class"
            )

            classifier_confidence = fused.get(
                "classifier_confidence"
            )

            source = fused.get(
                "source",
                "YOLO"
            )

            status = fused.get(
                "status",
                "DETECTED"
            )

            track_id = fused.get(
                "track_id",
                "-"
            )

            info = get_component_info(
                final_class
            )

            report.append(
                f"- **{final_class}** - "
                f"{final_confidence:.2f}\n"
                f"  - Track ID: {track_id}\n"
                f"  - Source: {source}\n"
                f"  - Status: {status}\n"
                f"  - YOLO: "
                f"{detector_class} - "
                f"{detector_confidence:.2f}\n"
            )

            if refine_mobilenet and classifier_class is not None:

                report.append(
                    f"  - MobileNet: "
                    f"{classifier_class} - "
                    f"{classifier_confidence:.2f}\n"
                )

                if classifier_class == detector_class:

                    report.append(
                        "  - Evidence: "
                        "YOLO and MobileNet agree\n"
                    )

                elif source == "MobileNet" and status == "REFINED":

                    report.append(
                        "  - Evidence: "
                        "MobileNet refined the YOLO class\n"
                    )

                else:

                    report.append(
                        "  - Evidence: "
                        "YOLO retained; MobileNet confidence was "
                        "insufficient for refinement\n"
                    )

            report.append(
                f"  - Category: "
                f"{info.get('category', 'Unknown')}\n"
                f"  - Function: "
                f"{info.get('function', 'Unknown')}\n"
                f"  - Pins: "
                f"{info.get('pins', 'Unknown')}\n"
                f"  - Typical role: "
                f"{info.get('typical_role', 'Unknown')}\n\n"
            )

        if mode == "Circuit Analysis":

            spatial = intelligence.get(
                "spatial_relationships",
                []
            )

            spatial_ranked = []

            for relation in spatial:

                distance = float(
                    relation.get(
                        "distance_pixels",
                        999999
                    )
                )

                normalized = float(
                    relation.get(
                        "normalized_distance",
                        999999
                    )
                )

                spatial_ranked.append(
                    (
                        normalized,
                        distance,
                        relation
                    )
                )

            spatial_ranked.sort(
                key=lambda item: item[0]
            )

            selected_spatial = []
            seen_spatial = set()

            for normalized, distance, relation in spatial_ranked:

                component_a = relation.get(
                    "component_a",
                    "Unknown"
                )

                component_b = relation.get(
                    "component_b",
                    "Unknown"
                )

                pair_key = tuple(
                    sorted(
                        [
                            component_a,
                            component_b
                        ]
                    )
                )

                if pair_key in seen_spatial:
                    continue

                seen_spatial.add(pair_key)

                selected_spatial.append(
                    (
                        normalized,
                        distance,
                        relation
                    )
                )

                if len(selected_spatial) >= 12:
                    break

            report.append(
                "### Spatial Layout\n\n"
            )

            if selected_spatial:

                for normalized, distance, relation in selected_spatial:

                    component_a = relation.get(
                        "component_a",
                        "Unknown"
                    )

                    component_b = relation.get(
                        "component_b",
                        "Unknown"
                    )

                    position = relation.get(
                        "relative_position",
                        "unknown"
                    )

                    report.append(
                        f"- **{component_a} -> "
                        f"{component_b}**: "
                        f"{position}, "
                        f"{distance:.1f} px apart\n"
                    )

                report.append("\n")

            else:

                report.append(
                    "No reliable spatial relationships identified.\n\n"
                )

            engineering = intelligence.get(
                "engineering_relationships",
                []
            )

            spatial_distance_map = {}

            for normalized, distance, relation in spatial_ranked:

                component_a = relation.get(
                    "component_a",
                    "Unknown"
                )

                component_b = relation.get(
                    "component_b",
                    "Unknown"
                )

                pair_key = tuple(
                    sorted(
                        [
                            component_a,
                            component_b
                        ]
                    )
                )

                if pair_key not in spatial_distance_map:

                    spatial_distance_map[pair_key] = (
                        normalized,
                        distance
                    )

            ranked_engineering = []

            for relation in engineering:

                component_a = relation.get(
                    "component_a",
                    "Unknown"
                )

                component_b = relation.get(
                    "component_b",
                    "Unknown"
                )

                pair_key = tuple(
                    sorted(
                        [
                            component_a,
                            component_b
                        ]
                    )
                )

                normalized, distance = spatial_distance_map.get(
                    pair_key,
                    (
                        999999,
                        999999
                    )
                )

                ranked_engineering.append(
                    (
                        normalized,
                        distance,
                        relation
                    )
                )

            ranked_engineering.sort(
                key=lambda item: item[0]
            )

            selected_engineering = []
            seen_engineering = set()

            for normalized, distance, relation in ranked_engineering:

                component_a = relation.get(
                    "component_a",
                    "Unknown"
                )

                component_b = relation.get(
                    "component_b",
                    "Unknown"
                )

                pair_key = tuple(
                    sorted(
                        [
                            component_a,
                            component_b
                        ]
                    )
                )

                if pair_key in seen_engineering:
                    continue

                seen_engineering.add(pair_key)

                if normalized <= 1.5:

                    selected_engineering.append(
                        (
                            normalized,
                            distance,
                            relation
                        )
                    )

                if len(selected_engineering) >= 10:
                    break

            report.append(
                "### Engineering Relationships\n\n"
            )

            if selected_engineering:

                for normalized, distance, relation in selected_engineering:

                    component_a = relation.get(
                        "component_a",
                        "Unknown"
                    )

                    component_b = relation.get(
                        "component_b",
                        "Unknown"
                    )

                    relationship = relation.get(
                        "relationship",
                        "Engineering possibility"
                    )

                    reason = relation.get(
                        "reason",
                        ""
                    )

                    report.append(
                        f"- **{component_a} <-> "
                        f"{component_b}**\n"
                        f"  - Relationship: "
                        f"{relationship}\n"
                        f"  - Reason: "
                        f"{reason}\n"
                        f"  - Visual proximity: "
                        f"{distance:.1f} px\n"
                        f"  - Evidence level: "
                        f"Engineering possibility only\n\n"
                    )

            else:

                report.append(
                    "No high-relevance engineering relationships identified.\n\n"
                )

            append_connection_report(
                report,
                connection_evidence,
                connection_graph
            )

            # ── Qualcomm VLM Deployment Configuration ───────────────────────
            vlm_status = vlm_engine.status()

            report.append(
                "### Qualcomm VLM Deployment Configuration\n\n"
            )

            if vlm_status["target_reached"]:

                report.append(
                    f"- Model: "
                    f"{vlm_status['model']}\n"
                    f"- Backend: "
                    f"{vlm_status['backend']}\n"
                    f"- Status: TARGET DEVICE REACHED\n"
                    f"- Host: "
                    f"{vlm_status['processor']}\n"
                    f"- Target: "
                    f"{vlm_status['target_device']}\n"
                    f"- Execution: "
                    f"{vlm_status['execution']}\n\n"
                )

            else:

                report.append(
                    f"- Model: "
                    f"{vlm_status['model']}\n"
                    f"- Backend: "
                    f"{vlm_status['backend']}\n"
                    f"- Precision: "
                    f"{vlm_status.get('precision', 'INT4')}\n"
                    f"- Target device: "
                    f"{vlm_status.get('target_device', 'Snapdragon X Elite CRD')}\n"
                    f"- Execution: "
                    f"{vlm_status.get('execution', 'NPU / HTP')}\n"
                    f"- Target readiness: "
                    "Awaiting supported Snapdragon ARM64 target\n"
                    f"- Model bundle: "
                    f"{vlm_status.get('bundle_status', 'Ready')}\n\n"
                )

                report.append(
                    "VLM inference was not run. The deployment configuration "
                    "is shown above; actual inference requires the supported "
                    "Snapdragon target and a ready model bundle.\n\n"
                )

            # ── Collect per-image evidence for combined reasoning ────────────
            all_fused_results_per_image.append(fused_results)
            all_selected_spatial_per_image.append(selected_spatial)
            all_selected_engineering_per_image.append(selected_engineering)
            all_connection_evidence_per_image.append(connection_evidence)

            img_evidence = openai_engine.build_evidence(
                fused_results=fused_results,
                spatial_relationships=[
                    r for _, __, r in selected_spatial
                ],
                engineering_relationships=[
                    r for _, __, r in selected_engineering
                ],
                connection_evidence=connection_evidence
            )
            per_image_evidence.append(
                {f"image_{image_index}": img_evidence}
            )

            for fused in fused_results:
                cls = fused.get("final_class", "Unknown")
                aggregated_inventory.append(
                    {
                        "image": image_index,
                        "class": cls,
                        "confidence": round(
                            float(
                                fused.get("final_confidence", 0.0)
                            ),
                            3
                        ),
                        "source": fused.get("source", "YOLO")
                    }
                )
                class_frequency[cls] = class_frequency.get(cls, 0) + 1

            report.append(
                "### Circuit Interpretation\n\n"
                "This analysis combines component detection, "
                "refinement, engineering state, spatial analysis, "
                "engineering relationships, wire association, "
                "connection evidence, connection graph construction, "
                "and OpenAI engineering reasoning. Definitive "
                "electrical connectivity requires stronger visual "
                "or measured connection evidence.\n\n"
            )

        else:

            report.append(
                "### Circuit Analysis\n\n"
                "Circuit analysis mode was not selected. "
                "The report contains component-level information "
                "rather than circuit reasoning.\n\n"
            )

    # ── Combined Summary Table ───────────────────────────────────────────────
    if len(images) > 1:

        summary_rows = ""
        for idx, count in enumerate(all_component_counts, start=1):
            summary_rows += f"| Image {idx} | {count} |\n"

        report.insert(
            1,
            "## 📊 Combined Summary\n\n"
            f"| Image | Components Detected |\n"
            f"|-------|-------------------|\n"
            + summary_rows
            + f"| **Total** | **{sum(all_component_counts)}** |\n\n"
        )

    # ── Cross-Image OpenAI Unified Reasoning ─────────────────────────────────
    if mode == "Circuit Analysis" and len(images) > 1 and per_image_evidence:

        report.append(
            "---\n\n"
            "## 🔗 Combined Cross-Image Engineering Reasoning\n\n"
            "The following unified analysis synthesises all per-image "
            "evidence — components, spatial layout, engineering ontology "
            "relationships, connection evidence, and wire associations — "
            "across every uploaded image.\n\n"
        )

        multi_result = openai_engine.analyze_multi_image(
            aggregated_inventory=aggregated_inventory,
            class_frequency=class_frequency,
            per_image_evidence=per_image_evidence
        )

        if multi_result.get("available"):

            report.append(
                f"**Model:** {multi_result.get('model', 'Unknown')}\n\n"
            )

            report.append(
                multi_result.get(
                    "analysis",
                    "No multi-image analysis returned."
                )
            )

            report.append("\n\n")

        else:

            report.append(
                f"- Model: {multi_result.get('model', 'Unknown')}\n"
                f"- Status: {multi_result.get('error', 'Unavailable')}\n\n"
            )

        report.append(
            "> **Note:** Components detected in different images are "
            "treated as physically isolated unless wire or topology "
            "evidence in the same image explicitly links them. "
            "Cross-image electrical connections are never assumed.\n\n"
        )

    return (
        annotated_gallery,
        crop_gallery,
        "".join(report)
    )


with gr.Blocks(
    title="SnapLab AI — Engineering Component Inspector"
) as demo:

    gr.Markdown(
        """
# 🔬 SnapLab AI — Engineering Component Inspector

Upload one or more circuit or component images.

The system performs:

- YOLO electronics detection
- MobileNet component refinement
- Engineering state tracking
- Spatial reasoning
- Circuit relationship analysis
- Wire detection and association
- Connection evidence and graph construction
- Qualcomm VLM integration status
- OpenAI engineering reasoning

The final visualization shows the fused engineering result with spatial, circuit, connection, and Qualcomm VLM analysis status.
"""
    )

    image_input = gr.File(
        label="📷 Upload Circuit / Component Images",
        file_count="multiple",
        file_types=["image"],
        type="filepath"
    )

    with gr.Row():

        mode_input = gr.Radio(
            choices=[
                "Component Identification",
                "Circuit Analysis"
            ],
            value="Circuit Analysis",
            label="Analysis Mode"
        )

        confidence_input = gr.Slider(
            minimum=0.05,
            maximum=0.90,
            value=0.25,
            step=0.05,
            label="YOLO Confidence"
        )

        refine_input = gr.Checkbox(
            value=True,
            label="Enable MobileNet Refinement"
        )

    run_button = gr.Button(
        "🚀 Analyze Images",
        variant="primary"
    )

    annotated_output = gr.Gallery(
        label="Annotated Images",
        columns=2,
        height="auto"
    )

    crop_output = gr.Gallery(
        label="Component Crops",
        columns=4,
        height="auto"
    )

    report_output = gr.Markdown(
        label="Engineering Intelligence Report"
    )

    run_button.click(
        fn=analyze_images,
        inputs=[
            image_input,
            mode_input,
            confidence_input,
            refine_input
        ],
        outputs=[
            annotated_output,
            crop_output,
            report_output
        ]
    )


if __name__ == "__main__":
    demo.launch()