from __future__ import annotations

import json


class ConnectionGraph:

    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_component(
        self,
        track_id,
        class_name,
        confidence
    ):
        self.nodes[track_id] = {
            "track_id": track_id,
            "class_name": class_name,
            "confidence": float(confidence)
        }

    def add_connection(
        self,
        from_id,
        to_id,
        wire_id=None,
        score=0.0,
        status="UNKNOWN",
        min_threshold=0.70,
        evidence_provenance="HEURISTIC"
    ):
        if from_id == to_id:
            return False

        if float(score) < min_threshold or status in (
            "NO_CONNECTION_EVIDENCE",
            "NO_PHYSICAL_EVIDENCE_SPATIAL_ONLY",
            "WEAK_CONNECTION_EVIDENCE"
        ):
            return False

        for edge in self.edges:

            same_direction = (
                edge["from_id"] == from_id
                and edge["to_id"] == to_id
            )

            reverse_direction = (
                edge["from_id"] == to_id
                and edge["to_id"] == from_id
            )

            if same_direction or reverse_direction:
                return False

        self.edges.append(
            {
                "from_id": from_id,
                "to_id": to_id,
                "wire_id": wire_id,
                "score": float(score),
                "status": status,
                "evidence_provenance": evidence_provenance
            }
        )

        return True

    def to_dict(self):
        return {
            "nodes": list(
                self.nodes.values()
            ),
            "edges": self.edges
        }

    def to_json(self):
        return json.dumps(
            self.to_dict(),
            indent=2
        )

    def print_graph(self):

        print("NODES")

        for node in self.nodes.values():

            print(
                f'{node["track_id"]}: '
                f'{node["class_name"]}'
            )

        print("\nEDGES")

        for edge in self.edges:

            print(
                f'{edge["from_id"]} -> '
                f'{edge["to_id"]} | '
                f'wire={edge["wire_id"]} | '
                f'score={edge["score"]:.3f} | '
                f'{edge["status"]}'
            )