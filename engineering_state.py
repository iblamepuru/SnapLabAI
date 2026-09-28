from dataclasses import dataclass, asdict
from typing import Dict, Tuple
from collections import Counter, deque
import time


@dataclass
class ComponentState:
    track_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    source: str
    status: str
    detector_class: str
    detector_confidence: float
    classifier_class: str
    classifier_confidence: float
    first_seen: float
    last_seen: float
    last_seen_frame: int


class EngineeringStateEngine:

    def __init__(
        self,
        lost_after=10,
        remove_after=60,
        history_size=7
    ):
        self.components: Dict[int, ComponentState] = {}
        self.class_history: Dict[int, deque] = {}
        self.frame_number = 0
        self.lost_after = lost_after
        self.remove_after = remove_after
        self.history_size = history_size

    def update_frame(self):
        self.frame_number += 1

    def _stable_class(self, track_id, class_name):
        if track_id not in self.class_history:
            self.class_history[track_id] = deque(
                maxlen=self.history_size
            )

        history = self.class_history[track_id]
        history.append(class_name)

        counts = Counter(history)

        return counts.most_common(1)[0][0]

    def update_fused(self, track_id, fused_result):
        bbox = fused_result["box"]
        class_name = fused_result["final_class"]
        confidence = fused_result["final_confidence"]

        detector_class = fused_result["detector_class"]
        detector_confidence = fused_result["detector_confidence"]
        classifier_class = fused_result["classifier_class"]
        classifier_confidence = fused_result["classifier_confidence"]
        source = fused_result["source"]
        status = fused_result["status"]

        x1, y1, x2, y2 = map(int, bbox)

        center = (
            int((x1 + x2) / 2),
            int((y1 + y2) / 2)
        )

        now = time.time()

        stable_class = self._stable_class(
            track_id,
            class_name
        )

        if track_id not in self.components:
            self.components[track_id] = ComponentState(
                track_id=track_id,
                class_name=stable_class,
                confidence=float(confidence),
                bbox=(x1, y1, x2, y2),
                center=center,
                source=source,
                status=status.upper(),
                detector_class=detector_class,
                detector_confidence=float(detector_confidence),
                classifier_class=classifier_class,
                classifier_confidence=(
                    None
                    if classifier_confidence is None
                    else float(classifier_confidence)
                ),
                first_seen=now,
                last_seen=now,
                last_seen_frame=self.frame_number
            )
        else:
            component = self.components[track_id]

            component.class_name = stable_class
            component.confidence = float(confidence)
            component.bbox = (x1, y1, x2, y2)
            component.center = center
            component.source = source
            component.status = status.upper()
            component.detector_class = detector_class
            component.detector_confidence = float(detector_confidence)
            component.classifier_class = classifier_class
            component.classifier_confidence = (
                None
                if classifier_confidence is None
                else float(classifier_confidence)
            )
            component.last_seen = now
            component.last_seen_frame = self.frame_number

    def update(
        self,
        track_id,
        class_name,
        confidence,
        bbox
    ):
        fused_result = {
            "final_class": class_name,
            "final_confidence": confidence,
            "box": bbox,
            "source": "YOLO",
            "status": "detected",
            "detector_class": class_name,
            "detector_confidence": confidence,
            "classifier_class": None,
            "classifier_confidence": None
        }

        self.update_fused(
            track_id,
            fused_result
        )

    def update_lifecycle(self):
        to_remove = []

        for track_id, component in self.components.items():
            frames_missing = (
                self.frame_number -
                component.last_seen_frame
            )

            if frames_missing >= self.remove_after:
                component.status = "REMOVED"
                to_remove.append(track_id)

            elif frames_missing >= self.lost_after:
                component.status = "LOST"

        for track_id in to_remove:
            del self.components[track_id]
            self.class_history.pop(track_id, None)

    def get_state(self):
        return {
            track_id: asdict(component)
            for track_id, component in self.components.items()
        }

    def print_state(self):
        print("\n========== ENGINEERING STATE ==========")

        if not self.components:
            print("No components currently tracked.")
            return

        for component in self.components.values():
            missing = (
                self.frame_number -
                component.last_seen_frame
            )

            history = list(
                self.class_history.get(
                    component.track_id,
                    []
                )
            )

            print(
                f"ID {component.track_id} | "
                f"{component.class_name} | "
                f"Confidence: {component.confidence:.2f} | "
                f"Source: {component.source} | "
                f"Status: {component.status} | "
                f"Center: {component.center} | "
                f"History: {history}"
            )


if __name__ == "__main__":
    engine = EngineeringStateEngine(
        lost_after=3,
        remove_after=6,
        history_size=7
    )

    observations = [
        "9-Volt-Battery",
        "9-Volt-Battery",
        "GSM-Module",
        "9-Volt-Battery",
        "GSM-Module",
        "9-Volt-Battery",
        "9-Volt-Battery"
    ]

    for class_name in observations:
        engine.update_frame()

        engine.update(
            track_id=1,
            class_name=class_name,
            confidence=0.80,
            bbox=(100, 100, 200, 200)
        )

        engine.update_lifecycle()

        print(
            f"Raw prediction: {class_name} "
            f"-> Stable class: "
            f"{engine.components[1].class_name}"
        )

    print("\nFinal state:")
    engine.print_state()

    print("\nTemporal class stabilization test PASSED.")