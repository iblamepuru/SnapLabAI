# SnapLab AI — User Guide

**Live website:** [https://iblamepuru--snaplab-ai-web.modal.run/](https://iblamepuru--snaplab-ai-web.modal.run/)  
**Source code:** [https://github.com/iblamepuru/SnapLabAI](https://github.com/iblamepuru/SnapLabAI)

SnapLab AI is an engineering inspection interface for analyzing images of electronic components and circuit setups. The dashboard combines visual component detection with component crops, spatial relationships, connection analysis, engineering insights, and evidence-oriented reports.

> **Important:** The application provides model predictions and inferred relationships. Treat suggested connections, component specifications, and design-rule findings as engineering assistance—not as a substitute for checking the physical circuit, datasheets, and measurements. A visual proximity or candidate wire is not necessarily a confirmed electrical connection.

## 1. Open the website

1. Open the [SnapLab AI live website](https://iblamepuru--snaplab-ai-web.modal.run/) in a modern browser.
2. Wait for the dashboard to finish loading.
3. The header shows the SnapLab AI name and the runtime/device status reported by the application.
4. The main workspace contains an image upload area, analysis controls, an input-image preview, detected-component crops, and result panels.

![SnapLab AI dashboard](docs/userguide_images/03-dashboard-empty-state.png)

## 2. Upload a circuit or component image

1. In **Upload Circuit Image**, click the upload control or drag an image into the drop area.
2. Use a clear, well-lit photo. Keep components in focus and avoid excessive glare, blur, or objects covering the circuit.
3. Prefer an image where the relevant boards, wires, connectors, and components are visible.
4. After upload, the file appears in the upload area. The **Input Image** panel is used to inspect the image and any annotated detection output.

![Image upload and input preview](docs/userguide_images/02-dashboard-initial-view.png)

### Tips for better input images

- Photograph the circuit from directly above when possible.
- Keep the entire circuit in frame.
- Use enough resolution for small components and labels.
- Avoid strong shadows and reflections.
- For a complex setup, consider taking separate close-up photos of different sections.

## 3. Choose the analysis mode

The **Analysis Mode** control provides the modes shown in the interface:

- **Component Identification** — focuses on identifying components in the image.
- **Circuit Analysis** — uses detected components and available relationship analysis to provide circuit-oriented results.

Choose the mode that best matches what you want to inspect.

## 4. Set detection options

Before running analysis, review the available controls:

- **YOLO Confidence:** adjusts the detector confidence threshold. A lower threshold may show more candidate detections, including uncertain ones; a higher threshold may reduce false positives but can miss harder-to-detect components.
- **MobileNet Refinement:** enables the component-classification refinement stage where supported.
- **Analysis Options:** use the checkboxes to enable or disable available stages, such as component detection, refinement, connection analysis, engineering relationships, Qualcomm VLM, and OpenAI engineering reasoning.

The available stages can depend on the deployed runtime and configured services. If an optional model or external reasoning service is unavailable, that stage may not return a result.

## 5. Run the analysis

1. Upload the image.
2. Select the analysis mode.
3. Adjust confidence and analysis options if needed.
4. Click **Run Analysis**.
5. Wait while the application processes the image. The interface may show a processing indicator.
6. Review the annotated image, component crops, counts, and analysis panels.

![Analysis in progress](docs/userguide_images/04-analysis-processing.png)

## 6. Read the detection results

After processing, the dashboard can show:

- **Annotated Bounding Boxes:** the original image with detected components outlined and labeled.
- **Detected Component Crops:** individual crops that make it easier to inspect small objects.
- **Component count:** the number of detections returned for the image.
- **Possible Connections:** candidate relationships inferred by the application.
- **Spatial Relationships:** relative positions such as above, below, left, or right.
- **Engineering Relationships:** relationships inferred from component roles or engineering knowledge.

Clicking or selecting a component can synchronize its details across the relevant inspection panels.

![Detected components and component crops](docs/userguide_images/05-detection-results.png)

## 7. Inspect component details

Select a component in the interactive inspector or the relevant component selector. Depending on the result, the detail card can display:

- Component ID and predicted class.
- Confidence and classification source.
- Bounding-box coordinates.
- Engineering category and function.
- Related components.
- Spatial relationships.
- Circuit evidence and verification notes.

Use these details to investigate a particular part rather than relying only on the overall summary.

## 8. Use the Analysis Report and Evidence & Explainability panels

The **Analysis Report** summarizes the analysis. The **Evidence & Explainability** panel separates different kinds of evidence, which may include:

1. **Direct visual observations** — what is visibly present in the image.
2. **Model predictions** — detections and classifications produced by the models.
3. **Geometric or spatial inferences** — relationships based on image positions.
4. **Engineering knowledge** — possible roles or relationships based on component knowledge.
5. **Unresolved questions** — items that require a closer inspection or physical verification.

A confidence score describes the model's prediction confidence; it does not guarantee that a component is correctly identified or wired.

![Evidence and explainability](docs/userguide_images/06-evidence-explainability.png)

## 9. View component specifications and pin guidance

Use the **Component Specs & Datasets** option from the dashboard tools menu when available. Select a component class to inspect the information shown by the application, such as:

- Functional description and typical role.
- Operating-voltage or other technical fields, when available.
- Package or form-factor information, when available.
- Pin or terminal mapping.
- Wiring guidance and safety notes.

These values may be generic class-level guidance rather than verified specifications for the exact physical part in your photograph. Confirm the exact part number and manufacturer datasheet before wiring or powering a circuit.

![Component specifications and pin guidance](docs/userguide_images/07-component-specifications.png)

## 10. Review Circuit Verification & DRC

Open **Circuit Verification & DRC** from the tools menu. The panel may show checks, warnings, and critical errors related to the visual or inferred circuit structure.

Use the result as a review checklist:

- Read the finding and its rationale.
- Check which components or inferred relationships it refers to.
- Review the suggested remediation.
- Verify the issue with the actual circuit, a multimeter, and relevant datasheets.

A warning based on an image or inferred topology may be incomplete. Do not make a physical change solely because of an unverified visual inference.

![Circuit verification and DRC](docs/userguide_images/09-circuit-verification-drc.png)

## 11. Explore the Spatial Layout

Open **Spatial Layout** to view the detected components as a 2D arrangement. The panel may include:

- Component bounding boxes or positions.
- Image dimensions and spatial statistics.
- Pairwise distances and directional relationships.

These measurements describe positions in the image. They do not, by themselves, prove electrical continuity or physical connectivity.

![Spatial layout](docs/userguide_images/10-spatial-layout.png)

## 12. Review Engineering Insights

Open **Engineering Insights** to view the application’s engineering-oriented summaries and recommendations. Use them to identify items worth checking, such as power distribution, signal compatibility, component roles, or layout considerations.

Treat these as suggestions based on the available detections and rules. Confirm electrical limits and circuit behavior independently before implementing a recommendation.

![Engineering insights](docs/userguide_images/11-engineering-insights.png)

## 13. Other dashboard tools

The dashboard’s overflow menu provides access to additional views. The options visible in the deployed interface may include:

- **Component Specs & Datasets**
- **Circuit Verification & DRC**
- **Spatial Layout**
- **Engineering Insights**

The exact menu items can vary with the deployed application version.

![Dashboard tools menu](docs/userguide_images/12-dashboard-tools-menu.png)

## 14. Understanding the performance figures

The project includes a benchmark summary comparing selected model measurements on Qualcomm Hexagon NPU hardware with an x86 CPU baseline. The displayed figures include YOLO11n inference latency, peak NPU memory, ResNet50 INT8 graph latency, Hexagon HTP utilization, and sustained throughput.

These are benchmark measurements for the listed models and test setup—not a guarantee of end-to-end latency for every image or for the hosted website. The hosted Modal website runs in a cloud environment; it should not be described as executing inference on the visitor’s local Snapdragon NPU unless that behavior is independently verified.

![NPU benchmark summary](docs/userguide_images/01-performance-benchmark.png)

## 15. Troubleshooting

### The page does not load
- Refresh the page and try again.
- Check your internet connection.
- If the service is temporarily unavailable, try again later.

### Analysis takes a long time
- Try a smaller, clearer image.
- Disable optional analysis stages that you do not need.
- Wait for the current request to finish before submitting another one.

### No components are detected
- Use a sharper, brighter image.
- Ensure the components are large enough to see.
- Try adjusting the YOLO confidence threshold.
- Try another view or a close-up image.

### A component is misclassified
- Inspect the crop and confidence.
- Compare the prediction with the actual part and its markings.
- Do not rely on a class-level pinout until the exact part is identified.

### A connection or DRC warning looks incorrect
- Treat it as a candidate finding.
- Inspect the physical wire path and connection points.
- Verify continuity and electrical behavior using appropriate tools.

## 16. Safety and interpretation

- Disconnect power before changing wiring.
- Check supply voltage, polarity, current limits, and component ratings.
- Confirm pinouts against the exact manufacturer datasheet.
- Use appropriate ESD precautions when handling sensitive electronics.
- Treat inferred connections as unconfirmed until physically verified.
- Do not use the application as the sole safety check for a powered circuit.

## Quick workflow checklist

1. Open the [live website](https://iblamepuru--snaplab-ai-web.modal.run/).
2. Upload a clear circuit image.
3. Choose **Component Identification** or **Circuit Analysis**.
4. Set the confidence threshold and optional analysis stages.
5. Click **Run Analysis**.
6. Inspect the annotated image and component crops.
7. Review the report, connection graph, spatial relationships, and engineering insights.
8. Verify any proposed wiring or electrical conclusion against the real circuit and datasheets.

---

**Project repository:** [github.com/iblamepuru/SnapLabAI](https://github.com/iblamepuru/SnapLabAI)  
**Live application:** [https://iblamepuru--snaplab-ai-web.modal.run/](https://iblamepuru--snaplab-ai-web.modal.run/)
