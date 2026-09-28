import math


def box_center(box):
    """
    Calculate the center point of a bounding box.

    Expected format:
        [x1, y1, x2, y2]

    Returns:
        (center_x, center_y)
    """

    x1, y1, x2, y2 = box

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    return center_x, center_y


def box_size(box):
    """
    Calculate width and height of a bounding box.

    Returns:
        width, height
    """

    x1, y1, x2, y2 = box

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    return width, height


def box_area(box):
    """
    Calculate bounding-box area.
    """

    width, height = box_size(box)

    return width * height


def center_distance(box_a, box_b):
    """
    Calculate Euclidean distance between two box centers.
    """

    center_a = box_center(box_a)
    center_b = box_center(box_b)

    dx = center_a[0] - center_b[0]
    dy = center_a[1] - center_b[1]

    return math.sqrt(
        dx * dx + dy * dy
    )


def normalized_distance(box_a, box_b):
    """
    Calculate center distance normalized by the average
    size of the two components.

    This makes spatial reasoning less dependent on image
    resolution.

    Returns:
        normalized distance
    """

    distance = center_distance(
        box_a,
        box_b
    )

    width_a, height_a = box_size(box_a)
    width_b, height_b = box_size(box_b)

    size_a = math.sqrt(
        width_a * width_a +
        height_a * height_a
    )

    size_b = math.sqrt(
        width_b * width_b +
        height_b * height_b
    )

    average_size = (
        size_a + size_b
    ) / 2.0

    if average_size <= 0:
        return float("inf")

    return distance / average_size


def relative_position(box_a, box_b):
    """
    Determine the position of component B relative to
    component A.

    Returns one of:

        left
        right
        above
        below
        upper-left
        upper-right
        lower-left
        lower-right
        overlapping
    """

    center_a = box_center(box_a)
    center_b = box_center(box_b)

    dx = center_b[0] - center_a[0]
    dy = center_b[1] - center_a[1]

    # Small tolerance for nearly overlapping centers
    tolerance = 10.0

    if abs(dx) <= tolerance and abs(dy) <= tolerance:
        return "overlapping"

    if abs(dx) > abs(dy):

        if dx > 0:
            return "right"

        return "left"

    else:

        if dy > 0:
            return "below"

        return "above"


def spatial_relation(box_a, box_b):
    """
    Return a complete spatial relationship between two
    components.
    """

    return {
        "center_a": box_center(box_a),
        "center_b": box_center(box_b),
        "distance_pixels": center_distance(
            box_a,
            box_b
        ),
        "normalized_distance": normalized_distance(
            box_a,
            box_b
        ),
        "relative_position": relative_position(
            box_a,
            box_b
        )
    }


def analyze_component_positions(
    components
):
    """
    Analyze spatial relationships between detected components.

    Expected input:

    [
        {
            "name": "Arduino-Uno",
            "box": [x1, y1, x2, y2],
            "confidence": 0.95
        },
        ...
    ]

    Returns a list containing pairwise spatial relationships.
    """

    relationships = []

    for i in range(len(components)):

        component_a = components[i]

        for j in range(
            i + 1,
            len(components)
        ):

            component_b = components[j]

            box_a = component_a["box"]
            box_b = component_b["box"]

            relation = spatial_relation(
                box_a,
                box_b
            )

            relationships.append(
                {
                    "component_a": component_a[
                        "name"
                    ],

                    "component_b": component_b[
                        "name"
                    ],

                    "confidence_a": component_a.get(
                        "confidence",
                        None
                    ),

                    "confidence_b": component_b.get(
                        "confidence",
                        None
                    ),

                    **relation
                }
            )

    return relationships