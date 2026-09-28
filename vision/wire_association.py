from __future__ import annotations

import math

from vision.wire_detector_v3 import detect_wire_candidates


class WireComponentAssociation:

    def __init__(
        self,
        endpoint_distance=100
    ):
        self.endpoint_distance = endpoint_distance

    def distance(
        self,
        p1,
        p2
    ):
        return math.hypot(
            p1[0] - p2[0],
            p1[1] - p2[1]
        )

    def wire_endpoints(
        self,
        candidate
    ):
        x = candidate["x"]
        y = candidate["y"]
        w = candidate["width"]
        h = candidate["height"]

        if w >= h:
            y_center = y + h / 2

            return (
                (x, y_center),
                (x + w, y_center)
            )

        x_center = x + w / 2

        return (
            (x_center, y),
            (x_center, y + h)
        )

    def point_to_bbox_distance(
        self,
        point,
        bbox
    ):
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

        return math.hypot(
            dx,
            dy
        )

    def _normalize_components(self, components):
        if isinstance(components, dict):
            return list(components.values())
        if isinstance(components, list):
            return components
        return []

    def nearest_component(
        self,
        endpoint,
        components,
        exclude_track_id=None
    ):
        best_component = None
        best_distance = float("inf")

        comp_list = self._normalize_components(components)

        for component in comp_list:

            track_id = component.get(
                "track_id"
            )

            if (
                exclude_track_id is not None
                and track_id == exclude_track_id
            ):
                continue

            bbox = component.get("bbox") or component.get("box")
            if bbox is None:
                continue

            distance = self.point_to_bbox_distance(
                endpoint,
                bbox
            )

            if distance < best_distance:

                best_distance = distance
                best_component = component

        if (
            best_component is not None
            and best_distance <= self.endpoint_distance
        ):
            return (
                best_component,
                best_distance
            )

        return (
            None,
            best_distance
        )

    def associate_wire(
        self,
        candidate,
        components
    ):
        endpoint_a, endpoint_b = self.wire_endpoints(
            candidate
        )

        component_a, distance_a = self.nearest_component(
            endpoint_a,
            components
        )

        exclude_id = None

        if component_a is not None:
            exclude_id = component_a.get(
                "track_id"
            )

        component_b, distance_b = self.nearest_component(
            endpoint_b,
            components,
            exclude_track_id=exclude_id
        )

        valid_connection = (
            component_a is not None
            and component_b is not None
            and component_a.get("track_id")
            != component_b.get("track_id")
        )

        return {
            "wire": candidate,
            "endpoint_a": endpoint_a,
            "endpoint_b": endpoint_b,
            "endpoint_type": "HEURISTIC_BOUNDING_BOX_EXTENT",
            "component_a": component_a,
            "component_b": component_b,
            "distance_a": distance_a,
            "distance_b": distance_b,
            "valid_connection": valid_connection,
            "evidence_provenance": "HEURISTIC_GEOMETRIC_ASSOCIATION",
            "electrical_contact_verified": False
        }

    def analyze(
        self,
        image,
        components
    ):
        comp_list = self._normalize_components(components)
        component_boxes = [
            c.get("bbox") or c.get("box")
            for c in comp_list
            if (c.get("bbox") or c.get("box")) is not None
        ]

        _, _, candidates = detect_wire_candidates(
            image,
            component_boxes=component_boxes,
            min_area=80,
            min_aspect_ratio=3.0
        )

        results = []

        for candidate in candidates:

            result = self.associate_wire(
                candidate,
                components
            )

            results.append(
                result
            )

        return results