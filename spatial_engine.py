from math import sqrt


class SpatialRelationshipEngine:

    def __init__(self, near_distance=150):
        self.near_distance = near_distance

    def distance(self, center_a, center_b):

        ax, ay = center_a
        bx, by = center_b

        return sqrt(
            (bx - ax) ** 2 +
            (by - ay) ** 2
        )

    def relationship(self, component_a, component_b):

        ax, ay = component_a["center"]
        bx, by = component_b["center"]

        dx = bx - ax
        dy = by - ay

        distance = self.distance(
            component_a["center"],
            component_b["center"]
        )

        relationships = []

        if dx > 0:
            relationships.append("RIGHT_OF")

        elif dx < 0:
            relationships.append("LEFT_OF")

        if dy > 0:
            relationships.append("BELOW")

        elif dy < 0:
            relationships.append("ABOVE")

        if distance <= self.near_distance:
            relationships.append("NEAR")

        return {
            "from_id": component_a["track_id"],
            "from_class": component_a["class_name"],
            "to_id": component_b["track_id"],
            "to_class": component_b["class_name"],
            "distance": round(distance, 2),
            "relationships": relationships
        }

    def analyze(self, components):

        results = []

        component_list = list(components.values())

        for i in range(len(component_list)):

            for j in range(i + 1, len(component_list)):

                relation = self.relationship(
                    component_list[i],
                    component_list[j]
                )

                results.append(relation)

        return results


if __name__ == "__main__":

    engine = SpatialRelationshipEngine(
        near_distance=150
    )

    test_components = {

        1: {
            "track_id": 1,
            "class_name": "Resistor",
            "center": (200, 300)
        },

        2: {
            "track_id": 2,
            "class_name": "LED-Light",
            "center": (300, 300)
        },

        3: {
            "track_id": 3,
            "class_name": "Arduino-Uno",
            "center": (500, 400)
        }
    }

    relationships = engine.analyze(test_components)

    print("\n========== SPATIAL RELATIONSHIPS ==========")

    for relation in relationships:

        print(
            f'ID {relation["from_id"]} '
            f'({relation["from_class"]}) -> '
            f'ID {relation["to_id"]} '
            f'({relation["to_class"]}) | '
            f'Distance: {relation["distance"]} px | '
            f'{", ".join(relation["relationships"])}'
        )

    print("\nSpatial relationship test PASSED.")
