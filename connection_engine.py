from dataclasses import dataclass, asdict
from typing import List, Dict


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


class ConnectionInferenceEngine:

    def __init__(
        self,
        near_distance=150,
        connection_threshold=0.70
    ):
        self.near_distance = near_distance
        self.connection_threshold = connection_threshold


    def spatial_score(self, distance):

        if distance <= self.near_distance:
            return 1.0

        if distance >= self.near_distance * 3:
            return 0.0

        return max(
            0.0,
            1.0 -
            (
                (distance - self.near_distance)
                / (self.near_distance * 2)
            )
        )


    def infer_connection(
        self,
        relation,
        wire_score=0.0,
        terminal_score=0.0
    ):

        distance = relation["distance"]

        spatial = self.spatial_score(distance)

        # Weighted evidence.
        #
        # Spatial proximity is only one signal.
        # Wire and terminal evidence will become
        # stronger when those CV modules are added.

        connection_score = (
            0.30 * spatial +
            0.40 * wire_score +
            0.30 * terminal_score
        )


        if connection_score >= self.connection_threshold:
            status = "POSSIBLE_CONNECTION"

        elif connection_score >= 0.40:
            status = "WEAK_CONNECTION_EVIDENCE"

        else:
            status = "NO_CONNECTION_EVIDENCE"


        return ConnectionEvidence(
            from_id=relation["from_id"],
            from_class=relation["from_class"],
            to_id=relation["to_id"],
            to_class=relation["to_class"],
            distance=distance,
            spatial_score=round(spatial, 3),
            wire_score=round(wire_score, 3),
            terminal_score=round(terminal_score, 3),
            connection_score=round(connection_score, 3),
            status=status
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

        for index, relation in enumerate(relationships):

            wire_score = wire_scores.get(index, 0.0)
            terminal_score = terminal_scores.get(index, 0.0)

            evidence = self.infer_connection(
                relation,
                wire_score=wire_score,
                terminal_score=terminal_score
            )

            results.append(
                asdict(evidence)
            )

        return results
