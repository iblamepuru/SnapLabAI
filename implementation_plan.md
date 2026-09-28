# Implementation Plan — Multi-Image Component Aggregation

## Objective

Enhance SnapLab AI to fully support multiple uploaded images in a single session. Every uploaded image must be processed through the complete YOLO + MobileNet refinement pipeline, followed by an aggregated component inventory, per-image and combined engineering reports, strict cross-image connection isolation, and multi-image OpenAI reasoning.

## Audit of Existing Implementation

1. **Gradio Input**:
   - `image_input = gr.File(label=..., file_count="multiple", file_types=["image"], type="filepath")` already allows multiple image selection.
2. **Current Processing Loop in `vision\component_inspector_v2.py`**:
   - Loops over `for image_index, file_path in enumerate(images, start=1):`.
   - Generates independent per-image Markdown strings and appends them to a single `report` list.
   - Invokes `openai_engine.analyze()` **inside** the per-image loop for each image independently.
   - **Gaps**:
     - No aggregated component inventory across images.
     - No unified component frequency distribution.
     - Crop gallery labels (`{final_class} | {final_confidence:.2f} | {source}`) do not specify which image the crop belongs to.
     - If an image fails to load (`image is None`), it silently continues without logging an error for that image.
     - No combined multi-image engineering reasoning stage synthesizing findings across all images while respecting image boundaries.

## Proposed Changes

### 1. `vision\component_inspector_v2.py`

#### A. Per-Image Processing & Error Resilience
- Wrap individual image loading and inference in a per-image `try...except` block so that a corrupt/unreadable file produces an explicit error entry in the report:
  `## Image {image_index} — Error: Failed to process file`
  without terminating or losing other valid uploaded images.
- Tag every detection in `fused_results` with:
  - `image_id`: `f"Image {image_index}"`
  - `image_index`: integer
  - `component_id`: `f"img{image_index}_c{track_id}"`
  - `track_id`: integer within this image
  - `box`: `[x1, y1, x2, y2]` in original image coordinates
  - `final_class`, `final_confidence`, `source`, `status`
- Update crop gallery captions with image provenance:
  `f"[{image_id}] {final_class} | {final_confidence:.2f} | {source}"`.

#### B. Component Inventory Aggregation
- Collect all detections across all images into `unified_inventory`.
- Compute aggregate metrics:
  - `total_images_processed`: count of successfully processed images
  - `total_detections`: total count of component detections across all images
  - `class_frequency`: `Counter([c['class'] for c in unified_inventory])`
  - `per_image_distribution`: `{img_id: len(components) for ...}`
- Format a Markdown table for the unified inventory:
  `| Image | Component ID | Class | Confidence | Source | Status | Bounding Box |`
- Format a component frequency summary table:
  `| Component Class | Total Detections | Appearing In Images |`

#### C. Cross-Image Connection Isolation Guardrail
- Keep spatial relationships, wire associations, and connection inferences strictly partitioned by `image_id`.
- Explicitly enforce that components from different images are **never** paired in spatial relations, wire traces, or connection graphs.
- Include a prominent warning in both the user report and the OpenAI evidence:
  `Components across different images must NOT be inferred to be physically connected. Each image's wiring topology is strictly isolated.`

#### D. Combined Engineering Analysis & Multi-Image OpenAI Reasoning
- Call `openai_engine.analyze()` with a combined multi-image evidence payload:
  - `aggregated_inventory`: all detected components with their `image_id`
  - `class_frequency`: count summary across all uploads
  - `per_image_evidence`: dictionary mapping each `image_id` to its localized spatial, ontology, wire, and connection evidence
  - `vlm_evidence_by_image`: VLM findings or hardware status per image
- Update the report structure to present:
  1. `## 📊 Combined Multi-Image Engineering Report` (Overview, Unified Inventory Table, Frequency Distribution, Combined Circuit Synthesis)
  2. `## 📷 Individual Image Analyses` (Image 1, Image 2, etc., detailing per-image components, spatial layout, and connection topology)

### 2. `vision\openai_engine.py`

- Update `build_evidence()` and `analyze()` to seamlessly accept either single-image evidence or multi-image aggregated evidence payloads (`is_multi_image` flag or auto-detection).
- In the multi-image prompt, instruct the reasoning engine:
  - Distinguish total visual detections across photographs from unique physical components.
  - NEVER infer electrical connections between components appearing in different images.
  - Reason over multi-board or sub-system relationships when multiple images depict distinct parts of a project.

## Verification Plan

### Automated / Scripted Tests
1. **Multi-Image Processing Test**:
   - Run `analyze_images()` on two distinct images simultaneously (`vision\reference_frame.jpg` and `vision\wire_v3_test.jpg`).
   - Verify:
     - Both images are analyzed.
     - `annotated_gallery` contains 2 entries (`Image 1: ...`, `Image 2: ...`).
     - `crop_gallery` contains crops labeled with `[Image 1]` and `[Image 2]`.
     - The report contains both `Image 1 Analysis` and `Image 2 Analysis`.
     - The report contains the Combined Multi-Image Report with the Unified Inventory table and Class Frequency table.
     - Cross-image connections are NOT generated.
2. **Single-Image Backward Compatibility**:
   - Run `analyze_images()` on a single image.
   - Verify single-image mode functions seamlessly with the inventory and report.
3. **Error Resilience Test**:
   - Pass a valid image and a non-existent file path `["vision/reference_frame.jpg", "non_existent.jpg"]`.
   - Verify the valid image succeeds, the invalid image produces an inline error, and no unhandled exception is raised.
