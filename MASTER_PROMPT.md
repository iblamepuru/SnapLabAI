# 🎯 SNAPLAB AI: THE MASTER PROJECT PROMPT & PRESENTATION SCRIPT

> **How to Use This Document**: 
> This is your **Master Prompt & Presentation Guide**. You can feed this entire document into any LLM (ChatGPT, Gemini, Claude) as a system prompt to explain every facet of SnapLab AI, or use it directly as your live script when pitching to judges, investors, or viewers at hackathons.

---

## 🌟 The 60-Second Elevator Pitch

> *"Good morning/afternoon, judges! Today, we are presenting **SnapLab AI: The Real-Time On-Device Circuit & Component Engineering Copilot**, built specifically for **Qualcomm Snapdragon AI PCs**.*
> 
> *When hardware engineers, robotics builders, or students sit down with breadboards and circuit prototypes, they face a massive manual bottleneck: identifying dozens of tiny ICs, squinting at pinouts, untangling rats' nests of jumper wires, and double-checking for catastrophic wiring errors like 5V-to-3.3V rail mismatches or missing flyback protection.*
> 
> *SnapLab AI completely solves this. You take a photo of any circuit board or breadboard, and our multi-stage AI pipeline instantly detects 65 hardware classes with 94.2% mAP, refines ambiguous chips with MobileNetV3, traces sub-pixel jumper wires, constructs an interactive connection topology graph, and runs full electrical Design Rule Checks (DRC).*
> 
> *Best of all, thanks to the **Qualcomm Snapdragon X Elite Hexagon NPU**, this entire neural pipeline runs locally on-device in just **6.8 milliseconds**—over **14.9 times faster** than CPU execution, consuming just **42 MB of RAM** with zero cloud latency and complete data privacy."*

---

## 🏛️ Comprehensive Architecture & Multi-Stage Pipeline

When explaining the technical depth to judges, explain that SnapLab AI is **not a naive single-model wrapper**, but a **cascaded 6-stage engineering pipeline**:

```
[Raw Circuit Image] 
       │
       ▼
1. YOLO11n Custom Detector (65 Hardware Classes @ 94.2% mAP)
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
2. MobileNetV3-Small Refinement           3. Sub-Pixel Wire Tracing
   (Disambiguates ICs & Passives @ 94.8%)   (Adaptive HSV, Skeletons & Endpoints)
       │                                         │
       ├─────────────────────────────────────────┘
       ▼
4. Spatial Geometric Engine (2D Euclidean Coordinates, Centroids, Power Rail Alignment)
       │
       ▼
5. Dynamic Tiered Connection Graph (SVG Graph: Observed | Heuristic | Ontology)
       │
       ▼
6. AI Copilot Reasoning & Electrical DRC (Groq LLM / Qualcomm VLM / Offline Rule Engine)
       │
       ▼
[Interactive Gradio Engineering Cockpit with Synchronized Pinout Datasheets]
```

---

## 🧩 Deep Dive: Every Tool & Module Explained

### Module 1: 65-Class Hardware Component Detector
* **Code**: `vision/component_inspector_v2.py`, `models/best.pt`
* **What it does**: Ingests single or multi-angle photos and identifies 65 electronics classes—including Arduino Uno/Nano/Mega, ESP32, BJT transistors, displays, relays, IC chips, capacitors, resistors, and breadboards.
* **Key Metric**: **94.2% mAP@50**, **89.9% Precision**, **92.5% Recall**.
* **Visual in Repo**: `assets/component_detection_demo.jpg`

### Module 2: Cascaded Micro-Component Classifier
* **Code**: `vision/component_refinement_engine.py`, `tools/train_component_classifier.py`
* **What it does**: Small components (like SOT-23 voltage regulators vs. TO-92 transistors or SMD resistors) can look identical from afar. This engine extracts bounding-box crops and routes them through a specialized MobileNetV3-Small classifier.
* **Key Metric**: **94.87% accuracy** on held-out test splits.
* **Visual in Repo**: `assets/component_crop_sample.png`

### Module 3: Sub-Pixel Wire Tracing & Terminal Proximity Engine
* **Code**: `vision/wire_detector_v3.py`, `vision/wire_association.py`
* **What it does**: Leverages multi-scale color segmentation in L*a*b* space, morphological skeletonization, and endpoint extraction to trace jumper cables from breadboard row to microcontroller pin headers.
* **Key Metric**: Associates wire endpoints with component pin terminals within Euclidean tolerances.
* **Visual in Repo**: `assets/wire_tracing_pipeline.png`

### Module 4: Spatial Geometric & Relative Coordinate Engine
* **Code**: `vision/spatial_engine.py`, `engineering_state.py`
* **What it does**: Understands relative physical placement. It knows if a sensor is *adjacent to* the microcontroller, *inserted into* the breadboard power rail, or *collinear* with a header strip.

### Module 5: Dynamic Tiered Connection Graph
* **Code**: `vision/connection_graph.py`, `vision/connection_engine.py`
* **What it does**: Translates visual detections into an interactive SVG schematic graph. To prevent false claims of continuity without a physical multimeter, it stratifies edges into 3 clear confidence tiers:
  * 🟢 **Tier 1 (Observed)**: Physically traced wire contacting a pin terminal.
  * 🟡 **Tier 2 (Heuristic)**: Shared breadboard contact row or proximate terminals.
  * 🔵 **Tier 3 (Ontology)**: Datasheet pin compatibility (e.g., standard I2C or UART pairing).

### Module 6: Qualcomm Snapdragon X Elite NPU Acceleration
* **Code**: `benchmarks/v3_npu_profile.json`, `benchmark_results.json`
* **What it does**: Uses Qualcomm AI Hub to compile and quantize neural models for the Hexagon NPU (45 TOPS).
* **Key Metrics**:
  * **6.838 ms** YOLO NPU inference (**14.93× faster** than x86 CPU).
  * **42.35 MB** peak NPU memory.
  * **0.623 ms** INT8 core graph latency on HTP with **98.99% hardware utilization** and **1,194.7 inferences/sec**.
* **Visual in Repo**: `assets/snapdragon_x_elite.jpg`

### Module 7: Multi-Image Inventory Aggregator
* **Code**: `vision/component_inspector_v2.py`
* **What it does**: Accepts multiple angle photos of a complex project. Maintains strict **cross-image wiring isolation** (so components in Photo A are never mistakenly inferred to connect to Photo B) while compiling an aggregated Bill of Materials.

### Module 8: AI Engineering Copilot & Electrical DRC Engine
* **Code**: `vision/openai_engine.py`, `vision/circuit_intelligence.py`
* **What it does**: Acts as a senior electrical engineer reviewing the board.
  * Identifies the circuit archetype (e.g., *Automated Plant Watering System*).
  * Audits voltage compatibility (3.3V vs 5V logic level warnings).
  * Flags missing protective circuitry (e.g., flyback snubber diode on inductive motor loads).
  * Fully equipped with **offline rule-based fallback** if internet connectivity drops.

---

## 🎬 Live Demo Click-Flow (Step-by-Step for Video / Judges)

Follow this exact flow during your demonstration:

1. **Step 1 — Launch Dashboard**:
   * Run `python app.py` and open `http://127.0.0.1:7860`.
   * Point out the Qualcomm Snapdragon branding and responsive dark-mode UI.
2. **Step 2 — Ingest Circuit Photo**:
   * Drag & drop `vision/reference_frame.jpg` (or an electronics test photo) into the upload box.
3. **Step 3 — Run Analysis**:
   * Click **🚀 Run Analysis**.
   * Highlight the sub-second execution speed!
4. **Step 4 — Inspect Detections Canvas**:
   * Point to the annotated image: bounding boxes with confidence scores (e.g. `ESP32-CAM: 0.94`, `Rain-Sensor: 0.91`, `DC-Motor: 0.89`).
   * Show the crop gallery displaying individual high-resolution thumbnails.
5. **Step 5 — Synchronized Component Card**:
   * In the dropdown, select `#1 ESP32-CAM`.
   * Notice how the pinout datasheet, power rail specs, and dedicated crop instantly synchronize!
6. **Step 6 — Dynamic Connection Graph**:
   * Switch to the **🔀 Connection Graph** tab.
   * Toggle the checkboxes (*Observed*, *Heuristic*, *Ontology*) to filter between direct wire traces and theoretical pin mappings.
7. **Step 7 — AI Engineering Audit & DRC Report**:
   * Show the AI Copilot section: notice how it accurately identifies the circuit archetype, validates voltage rails, and warns about missing components.
8. **Step 8 — Multi-Image Comparison (Optional Power Move)**:
   * Upload two photos simultaneously to show the unified BOM table and strict cross-image isolation guardrail.

---

## 📊 Summary of Verified Benchmark Figures for Pitch

| Capability / Benchmark | Verified Figure | Hardware / Artifact |
| :--- | :--- | :--- |
| **Component Detection mAP@50** | **94.2%** | Custom 65-class YOLO11n |
| **Component Detection Precision / Recall** | **89.9% / 92.5%** | Phase C Master Evaluation |
| **MobileNetV3 Refinement Accuracy** | **94.87%** | Electronics Crop Split |
| **YOLO11n NPU Latency** | **6.838 ms** | Snapdragon X Elite CRD (Hexagon NPU) |
| **YOLO11n Speedup vs. CPU** | **14.93× Faster** | Qualcomm AI Hub Profiler |
| **Peak NPU Memory** | **42.35 MB** | Sub-50MB edge footprint |
| **INT8 Graph Latency** | **0.623 ms** | Hexagon Tensor Processor (HTP) |
| **HTP Hardware Utilization** | **98.99%** | Maximum silicon efficiency |
| **Throughput** | **1,194.7 inf/s** | Sustained real-time stream |

---

## 🏆 Key Closing Statement for Judges

> *"SnapLab AI shows what the future of AI PCs looks like: moving past generic chatbots into **physical-world edge intelligence**. By taking full advantage of the Qualcomm Snapdragon X Elite Hexagon NPU, we empower every engineer, maker, and student with an on-device electrical copilot right at their workbench. Thank you!"*
