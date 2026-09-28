import json


class ConnectionGraph:

    def __init__(self):
        self.nodes = {}
        self.edges = []


    # --------------------------------------------------
    # Add / update component
    # --------------------------------------------------

    def add_component(
        self,
        track_id,
        class_name,
        confidence=0.0
    ):

        self.nodes[track_id] = {
            "track_id": int(track_id),
            "class_name": class_name,
            "confidence": round(
                float(confidence),
                3
            )
        }


    # --------------------------------------------------
    # Add connection edge
    # --------------------------------------------------

    def add_connection(
        self,
        from_id,
        to_id,
        wire_id=None,
        score=0.0,
        status="UNKNOWN"
    ):

        from_id = int(from_id)
        to_id = int(to_id)

        # Prevent duplicate connections.
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


        self.edges.append({
            "from_id": from_id,
            "to_id": to_id,
            "wire_id": wire_id,
            "score": round(
                float(score),
                3
            ),
            "status": status
        })

        return True


    # --------------------------------------------------
    # Export graph
    # --------------------------------------------------

    def to_dict(self):

        return {
            "nodes": list(
                self.nodes.values()
            ),
            "edges": self.edges
        }


    # --------------------------------------------------
    # JSON export
    # --------------------------------------------------

    def to_json(self):

        return json.dumps(
            self.to_dict(),
            indent=2
        )


    # --------------------------------------------------
    # Print graph
    # --------------------------------------------------

    def print_graph(self):

        print(
            "\n========== CONNECTION GRAPH =========="
        )

        print(
            f"Nodes: {len(self.nodes)}"
        )

        for node in self.nodes.values():

            print(
                f'  ID {node["track_id"]} | '
                f'{node["class_name"]} | '
                f'Confidence: {node["confidence"]}'
            )


        print(
            f"\nEdges: {len(self.edges)}"
        )

        for edge in self.edges:

            print(
                f'  ID {edge["from_id"]} -> '
                f'ID {edge["to_id"]} | '
                f'Wire: {edge["wire_id"]} | '
                f'Score: {edge["score"]} | '
                f'{edge["status"]}'
            )


# ------------------------------------------------------
# UNIT TEST
# ------------------------------------------------------

if __name__ == "__main__":

    graph = ConnectionGraph()


    graph.add_component(
        track_id=1,
        class_name="Arduino-Uno",
        confidence=0.91
    )


    graph.add_component(
        track_id=2,
        class_name="Breadboard",
        confidence=0.94
    )


    graph.add_component(
        track_id=3,
        class_name="LED-Light",
        confidence=0.88
    )


    graph.add_connection(
        from_id=1,
        to_id=2,
        wire_id="wire_001",
        score=0.82,
        status="POSSIBLE_CONNECTION"
    )


    graph.add_connection(
        from_id=2,
        to_id=3,
        wire_id="wire_002",
        score=0.76,
        status="POSSIBLE_CONNECTION"
    )


    # Duplicate test
    duplicate = graph.add_connection(
        from_id=2,
        to_id=1,
        wire_id="wire_duplicate",
        score=0.90,
        status="POSSIBLE_CONNECTION"
    )


    graph.print_graph()


    print(
        "\nDuplicate edge rejected:",
        duplicate is False
    )


    print(
        "\nJSON representation:"
    )

    print(
        graph.to_json()
    )


    print(
        "\nConnection graph foundation test PASSED."
    )
