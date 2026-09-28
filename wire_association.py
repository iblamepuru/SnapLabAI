from math import sqrt


class WireComponentAssociation:

    def __init__(self, endpoint_distance=100):
        self.endpoint_distance = endpoint_distance


    # --------------------------------------------------
    # Calculate Euclidean distance
    # --------------------------------------------------

    def distance(self, point_a, point_b):

        ax, ay = point_a
        bx, by = point_b

        return sqrt(
            (bx - ax) ** 2 +
            (by - ay) ** 2
        )


    # --------------------------------------------------
    # Estimate the two endpoints of a wire bbox
    # --------------------------------------------------

    def wire_endpoints(self, bbox):

        x1, y1, x2, y2 = bbox

        width = x2 - x1
        height = y2 - y1

        # Horizontal wire
        if width >= height:

            p1 = (
                int(x1),
                int((y1 + y2) / 2)
            )

            p2 = (
                int(x2),
                int((y1 + y2) / 2)
            )

        # Vertical wire
        else:

            p1 = (
                int((x1 + x2) / 2),
                int(y1)
            )

            p2 = (
                int((x1 + x2) / 2),
                int(y2)
            )

        return p1, p2


    # --------------------------------------------------
    # Distance from point to component bbox
    # --------------------------------------------------

    def point_to_bbox_distance(self, point, bbox):

        px, py = point

        x1, y1, x2, y2 = bbox

        dx = max(
            x1 - px,
            0,
            px - x2
        )

        dy = max(
            y1 - py,
            0,
            py - y2
        )

        return sqrt(
            dx ** 2 +
            dy ** 2
        )


    # --------------------------------------------------
    # Find nearest component to an endpoint
    # --------------------------------------------------

    def nearest_component(
        self,
        endpoint,
        components
    ):

        best_component = None
        best_distance = float("inf")

        for component in components.values():

            distance = self.point_to_bbox_distance(
                endpoint,
                component["bbox"]
            )

            if distance < best_distance:

                best_distance = distance
                best_component = component

        if (
            best_component is not None
            and best_distance <= self.endpoint_distance
        ):

            return {
                "track_id": best_component["track_id"],
                "class_name": best_component["class_name"],
                "distance": round(
                    best_distance,
                    2
                )
            }

        return None


    # --------------------------------------------------
    # Associate one wire with components
    # --------------------------------------------------

    def associate_wire(
        self,
        wire,
        components
    ):

        endpoints = self.wire_endpoints(
            wire["bbox"]
        )

        endpoint_a, endpoint_b = endpoints

        component_a = self.nearest_component(
            endpoint_a,
            components
        )

        component_b = self.nearest_component(
            endpoint_b,
            components
        )

        return {
            "wire": wire,
            "endpoint_a": endpoint_a,
            "endpoint_b": endpoint_b,
            "component_a": component_a,
            "component_b": component_b
        }


    # --------------------------------------------------
    # Associate all detected wires
    # --------------------------------------------------

    def analyze(
        self,
        wires,
        components
    ):

        results = []

        for wire in wires:

            result = self.associate_wire(
                wire,
                components
            )

            results.append(result)

        return results


# ------------------------------------------------------
# UNIT TEST
# ------------------------------------------------------

if __name__ == "__main__":

    engine = WireComponentAssociation(
        endpoint_distance=100
    )


    components = {

        1: {
            "track_id": 1,
            "class_name": "Arduino-Uno",
            "bbox": (100, 100, 200, 200)
        },

        2: {
            "track_id": 2,
            "class_name": "Breadboard",
            "bbox": (400, 100, 500, 200)
        }
    }


    test_wire = {
        "bbox": (190, 145, 410, 155),
        "length": 220,
        "aspect_ratio": 22.0
    }


    results = engine.analyze(
        [test_wire],
        components
    )


    print(
        "\n========== WIRE-COMPONENT ASSOCIATION =========="
    )


    for result in results:

        print(
            f'Wire endpoints: '
            f'{result["endpoint_a"]} -> '
            f'{result["endpoint_b"]}'
        )


        if result["component_a"]:

            print(
                f'Endpoint A -> '
                f'ID {result["component_a"]["track_id"]} '
                f'({result["component_a"]["class_name"]}) | '
                f'Distance: '
                f'{result["component_a"]["distance"]} px'
            )

        else:

            print(
                "Endpoint A -> No nearby component"
            )


        if result["component_b"]:

            print(
                f'Endpoint B -> '
                f'ID {result["component_b"]["track_id"]} '
                f'({result["component_b"]["class_name"]}) | '
                f'Distance: '
                f'{result["component_b"]["distance"]} px'
            )

        else:

            print(
                "Endpoint B -> No nearby component"
            )


    print(
        "\nWire-component association test PASSED."
    )
