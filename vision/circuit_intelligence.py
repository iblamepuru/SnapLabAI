import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vision.spatial_engine import analyze_component_positions
from vision.relationship_engine import analyze_relationships


class CircuitIntelligenceEngine:

    def analyze(self, state):
        components = []

        for track_id, item in state.items():
            components.append({
                "track_id": track_id,
                "name": item["class_name"],
                "class_name": item["class_name"],
                "confidence": item["confidence"],
                "source": item["source"],
                "status": item["status"],
                "box": item["bbox"],
                "bbox": item["bbox"],
                "center": item["center"]
            })

        spatial_input = [
            {
                "name": component["name"],
                "box": component["box"],
                "confidence": component["confidence"]
            }
            for component in components
        ]

        spatial = analyze_component_positions(
            spatial_input
        )

        names = [
            component["class_name"]
            for component in components
        ]

        engineering = analyze_relationships(
            names
        )

        return {
            "components": components,
            "spatial_relationships": spatial,
            "engineering_relationships": engineering,
            "component_count": len(components),
            "spatial_pair_count": len(spatial),
            "engineering_pair_count": len(engineering)
        }

    def summarize(self, analysis):
        lines = []

        lines.append(
            f"Components detected: "
            f"{analysis['component_count']}"
        )

        lines.append("")
        lines.append("COMPONENTS")

        for component in analysis["components"]:
            lines.append(
                f"{component['class_name']} "
                f"({component['confidence']:.2f}) "
                f"[{component['source']}]"
            )

        lines.append("")
        lines.append("SPATIAL RELATIONSHIPS")

        if analysis["spatial_relationships"]:
            for relation in analysis["spatial_relationships"]:
                lines.append(
                    f"{relation['component_a']} -> "
                    f"{relation['component_b']} | "
                    f"{relation['relative_position']} | "
                    f"distance="
                    f"{relation['distance_pixels']:.2f}px"
                )
        else:
            lines.append(
                "No spatial relationships identified."
            )

        lines.append("")
        lines.append("ENGINEERING RELATIONSHIPS")

        if analysis["engineering_relationships"]:
            for relation in analysis[
                "engineering_relationships"
            ]:
                lines.append(
                    f"{relation['component_a']} <-> "
                    f"{relation['component_b']} | "
                    f"{relation['relationship']} | "
                    f"{relation['confidence']}"
                )
        else:
            lines.append(
                "No engineering relationships identified."
            )

        return "\n".join(lines)
