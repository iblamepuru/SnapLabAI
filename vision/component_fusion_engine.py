from vision.component_refinement_engine import refine_detection


def fuse_yolo_results(
    image,
    result,
    refine_all=False
):

    fused_results = []

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return fused_results

    names = result.names

    for index in range(len(boxes)):

        xyxy = boxes.xyxy[index].cpu().numpy().tolist()
        confidence = float(boxes.conf[index].cpu().item())
        class_id = int(boxes.cls[index].cpu().item())

        detector_class = names.get(
            class_id,
            "Unknown"
        )

        final_class = detector_class
        final_confidence = confidence
        source = "YOLO"
        status = "DETECTED"

        classifier_class = None
        classifier_confidence = None

        if refine_all or confidence < 0.50:

            refinement = refine_detection(
                image,
                xyxy
            )

            if refinement is not None:

                classifier_class = refinement.get(
                    "class_name"
                )

                classifier_confidence = float(
                    refinement.get(
                        "confidence",
                        0.0
                    )
                )

                if classifier_class is not None:

                    if classifier_confidence >= 0.70:

                        final_class = classifier_class
                        final_confidence = classifier_confidence
                        source = "MobileNet"
                        status = "REFINED"

                    else:

                        source = "YOLO"
                        status = "UNCERTAIN"

        fused_results.append(
            {
                "detector_class": detector_class,
                "detector_confidence": confidence,
                "classifier_class": classifier_class,
                "classifier_confidence": classifier_confidence,
                "final_class": final_class,
                "final_confidence": final_confidence,
                "source": source,
                "status": status,
                "box": xyxy
            }
        )

    return fused_results