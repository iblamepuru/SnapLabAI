from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConnectionEvidence:
    from_id: int
    from_class: str
    to_id: int
    to_class: str
    distance: float
    spatial_score: float
    wire_score: float
    terminal_score: float
    connection_score: float
    status: str
    evidence_type: str = "GEOMETRIC_PROXIMITY_ONLY"


class ConnectionInferenceEngine:

    def __init__(
        self,
        near_distance=150,
        connection_threshold=0.70
    ):
        self.near_distance = near_distance
        self.connection_threshold = connection_threshold

    def spatial_score(
        self,
        distance
    ):
        if distance <= self.near_distance:
            return 1.0

        if distance >= self.near_distance * 3:
            return 0.0

        return max(
            0.0,
            1.0 - (
                (
                    distance
                    - self.near_distance
                )
                / (
                    self.near_distance * 2
                )
            )
        )

    def infer_connection(
        self,
        relationship,
        wire_score=0.0,
        terminal_score=0.0
    ):
        distance = float(
            relationship.get(
                "distance",
                0.0
            )
        )

        spatial = self.spatial_score(
            distance
        )

        wire_val = float(wire_score)
        term_val = float(terminal_score)

        connection_score = (
            0.30 * spatial
            + 0.40 * wire_val
            + 0.30 * term_val
        )

        if wire_val > 0.0 and term_val > 0.0:
            evidence_type = "MULTIMODAL_WIRE_AND_TERMINAL"
        elif wire_val > 0.0:
            evidence_type = "HEURISTIC_WIRE_CANDIDATE"
        elif term_val > 0.0:
            evidence_type = "TERMINAL_ALIGNMENT"
        else:
            evidence_type = "GEOMETRIC_PROXIMITY_ONLY"

        if wire_val <= 0.0 and term_val <= 0.0:
            status = "NO_PHYSICAL_EVIDENCE_SPATIAL_ONLY"
        elif connection_score >= self.connection_threshold:
            status = "POSSIBLE_CONNECTION"
        elif connection_score >= 0.40:
            status = "WEAK_CONNECTION_EVIDENCE"
        else:
            status = "NO_CONNECTION_EVIDENCE"

        return ConnectionEvidence(
            from_id=relationship["from_id"],
            from_class=relationship["from_class"],
            to_id=relationship["to_id"],
            to_class=relationship["to_class"],
            distance=distance,
            spatial_score=spatial,
            wire_score=wire_val,
            terminal_score=term_val,
            connection_score=connection_score,
            status=status,
            evidence_type=evidence_type
        )

    def analyze(
        self,
        relationships,
        wire_scores=None,
        terminal_scores=None
    ):
        wire_scores = wire_scores or {}
        terminal_scores = terminal_scores or {}

        results = []

        for relationship in relationships:

            pair = (
                relationship["from_id"],
                relationship["to_id"]
            )

            reverse_pair = (
                relationship["to_id"],
                relationship["from_id"]
            )

            wire_score = wire_scores.get(
                pair,
                wire_scores.get(
                    reverse_pair,
                    relationship.get(
                        "wire_score",
                        0.0
                    )
                )
            )

            terminal_score = terminal_scores.get(
                pair,
                terminal_scores.get(
                    reverse_pair,
                    relationship.get(
                        "terminal_score",
                        0.0
                    )
                )
            )

            results.append(
                self.infer_connection(
                    relationship,
                    wire_score,
                    terminal_score
                )
            )

        return [
            {
                "from_id": item.from_id,
                "from_class": item.from_class,
                "to_id": item.to_id,
                "to_class": item.to_class,
                "distance": item.distance,
                "spatial_score": item.spatial_score,
                "wire_score": item.wire_score,
                "terminal_score": item.terminal_score,
                "connection_score": item.connection_score,
                "status": item.status,
                "evidence_type": item.evidence_type
            }
            for item in results
        ]