import gradio as gr
from ultralytics import YOLO
import numpy as np

MODEL_PATH = r"runs\detect\runs\snaplab\phaseC_master_65class-2\weights\best.pt"

model = YOLO(MODEL_PATH)

def detect_images(images, confidence):
    if not images:
        return [], "No images uploaded."

    results = []
    summary = []

    for i, image in enumerate(images):
        result = model.predict(
            image,
            imgsz=640,
            conf=confidence,
            verbose=False
        )[0]

        annotated = result.plot()

        results.append(annotated)

        names = []

        for cls, conf in zip(result.boxes.cls, result.boxes.conf):
            class_id = int(cls)
            score = float(conf)
            name = model.names[class_id]
            names.append(f"{name} — {score:.2f}")

        summary.append(
            f"### Image {i + 1}\n" +
            ("\n".join(f"- {x}" for x in names)
             if names else "- No components detected")
        )

    return results, "\n\n".join(summary)

with gr.Blocks(title="SnapLab AI Component Inspector") as demo:

    gr.Markdown(
        "# 🔬 SnapLab AI — Component Inspector\n"
        "Upload multiple circuit/component images and inspect detected components."
    )

    with gr.Row():
        images = gr.File(
            label="Upload component images",
            file_count="multiple",
            file_types=["image"]
        )

        confidence = gr.Slider(
            minimum=0.05,
            maximum=0.90,
            value=0.20,
            step=0.05,
            label="Confidence threshold"
        )

    run = gr.Button("🔍 Analyze Images", variant="primary")

    gallery = gr.Gallery(
        label="Detection Results",
        columns=2,
        height="auto"
    )

    report = gr.Markdown(
        "Upload images and click **Analyze Images**."
    )

    run.click(
        detect_images,
        inputs=[images, confidence],
        outputs=[gallery, report]
    )

demo.launch()
