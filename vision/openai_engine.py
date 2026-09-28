from __future__ import annotations

import os

from openai import OpenAI


class OpenAIEngineeringEngine:

    def __init__(
        self,
        model=None
    ):

        self.model = model or os.getenv(
            "GROQ_ENGINEERING_MODEL",
            os.getenv("OPENAI_ENGINEERING_MODEL", "llama-3.3-70b-versatile")
        )

        # Groq exposes an OpenAI-compatible API.
        # Keep the API key in the environment, never in source code.
        self.api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

        if not self.api_key:
            env_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                ".env"
            )
            if os.path.exists(env_path):
                try:
                    with open(env_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not self.api_key:
                                if line.startswith("GROQ_API_KEY="):
                                    self.api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                                elif line.startswith("OPENAI_API_KEY="):
                                    self.api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if not model:
                                if line.startswith("GROQ_ENGINEERING_MODEL="):
                                    self.model = line.split("=", 1)[1].strip().strip('"').strip("'")
                                elif line.startswith("OPENAI_ENGINEERING_MODEL="):
                                    self.model = line.split("=", 1)[1].strip().strip('"').strip("'")
                except Exception:
                    pass

        self.client = None
        self.init_error = None

        if self.api_key:
            try:
                # If using Groq API key, default to Groq OpenAI endpoint
                if self.api_key.startswith("gsk_") or "api.groq.com" in self.base_url:
                    self.client = OpenAI(
                        api_key=self.api_key,
                        base_url="https://api.groq.com/openai/v1"
                    )
                else:
                    self.client = OpenAI(
                        api_key=self.api_key
                    )
            except Exception as exc:
                self.client = None
                self.init_error = str(exc)

    def status(self):

        return {
            "configured": self.client is not None,
            "model": self.model,
            "has_key": bool(self.api_key),
            "init_error": self.init_error
        }

    def build_evidence(
        self,
        fused_results,
        spatial_relationships=None,
        engineering_relationships=None,
        connection_evidence=None,
        vlm_result=None
    ):

        components = []

        for item in fused_results or []:

            components.append(
                {
                    "track_id": item.get(
                        "track_id"
                    ),
                    "class": item.get(
                        "final_class",
                        "Unknown"
                    ),
                    "confidence": round(
                        float(
                            item.get(
                                "final_confidence",
                                0.0
                            )
                        ),
                        3
                    ),
                    "source": item.get(
                        "source",
                        "YOLO"
                    ),
                    "status": item.get(
                        "status",
                        "UNKNOWN"
                    ),
                    "detector_class": item.get(
                        "detector_class",
                        "Unknown"
                    ),
                    "detector_confidence": round(
                        float(
                            item.get(
                                "detector_confidence",
                                0.0
                            )
                        ),
                        3
                    ),
                    "classifier_class": item.get(
                        "classifier_class"
                    ),
                    "classifier_confidence": item.get(
                        "classifier_confidence"
                    )
                }
            )

        evidence_dict = {
            "components": components,
            "spatial_relationships": (
                spatial_relationships or []
            ),
            "engineering_relationships": (
                engineering_relationships or []
            ),
            "connection_evidence": (
                connection_evidence or {}
            )
        }

        if vlm_result is not None:
            evidence_dict["vlm_observation"] = vlm_result

        return evidence_dict

    def analyze(
        self,
        fused_results,
        spatial_relationships=None,
        engineering_relationships=None,
        connection_evidence=None,
        vlm_result=None
    ):

        if self.client is None:

            reason = (
                f"OpenAI client initialization failed: {self.init_error}"
                if self.init_error
                else (
                    "API key not configured. "
                    "Set GROQ_API_KEY or OPENAI_API_KEY to enable "
                    "engineering reasoning."
                )
            )

            return {
                "available": False,
                "model": self.model,
                "error": reason,
                "analysis": reason
            }

        evidence = self.build_evidence(
            fused_results,
            spatial_relationships,
            engineering_relationships,
            connection_evidence,
            vlm_result=vlm_result
        )

        prompt = f"""
You are the engineering reasoning layer of SnapLab AI.

SnapLab AI uses a 65-class YOLO detector, MobileNet refinement,
engineering state tracking, spatial reasoning, engineering ontology
relationships, and optional Qualcomm on-device VLM visual interpretation.

Your job is to reason over the structured evidence provided below.

Do not invent components.

Do not treat component proximity as proof of electrical connectivity.

Do not treat an ontology relationship as proof that the relationship
exists in the photographed circuit.

If Qualcomm VLM observations are provided, evaluate them as descriptive
visual context alongside the detector evidence; do not treat them as verified electrical connections.

If Qualcomm VLM is unavailable, proceed strictly using the detector,
spatial, and ontology evidence.

Clearly separate:
1. Visual observations
2. Engineering possibilities
3. Conclusions supported by evidence
4. Conclusions that require additional visual evidence

Identify the most plausible circuit purpose only when the evidence
supports it.

If the topology is insufficient, explicitly say so.

Return a concise engineering analysis with these sections:

Engineering Assessment
Observed Components
Likely Functional Relationships
Connection Confidence
Possible Circuit Purpose
Potential Issues
What Should Be Verified Next

Structured evidence:

{evidence}
"""

    def _get_candidate_models(self):
        candidates = [self.model] if self.model else []
        if self.api_key and self.api_key.startswith("gsk_"):
            groq_defaults = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b", "llama-3.3-70b-versatile"]
            for m in groq_defaults:
                if m not in candidates:
                    candidates.append(m)
        else:
            openai_defaults = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "openai/gpt-oss-120b"]
            for m in openai_defaults:
                if m not in candidates:
                    candidates.append(m)
        return candidates

    def _call_chat_with_fallback(self, messages, temperature=0.3, max_tokens=1500):
        if not self.client:
            raise RuntimeError("No active API client configured.")

        candidate_models = self._get_candidate_models()
        last_error = None

        for m in candidate_models:
            try:
                if hasattr(self.client, "chat") and hasattr(self.client.chat, "completions"):
                    response = self.client.chat.completions.create(
                        model=m,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    self.model = m
                    return response.choices[0].message.content, m
                elif hasattr(self.client, "responses"):
                    response = self.client.responses.create(
                        model=m,
                        input=messages,
                        temperature=temperature
                    )
                    self.model = m
                    return getattr(response, "output_text", str(response)), m
            except Exception as exc:
                last_error = exc
                continue

        raise RuntimeError(f"All candidate models failed. Last error: {last_error}")

    def analyze(
        self,
        fused_results,
        spatial_relationships=None,
        engineering_relationships=None,
        connection_evidence=None,
        vlm_result=None
    ):

        if self.client is None:

            reason = (
                f"OpenAI client initialization failed: {self.init_error}"
                if self.init_error
                else (
                    "API key not configured. "
                    "Set GROQ_API_KEY or OPENAI_API_KEY to enable "
                    "engineering reasoning."
                )
            )

            return {
                "available": False,
                "model": self.model,
                "error": reason,
                "analysis": reason
            }

        evidence = self.build_evidence(
            fused_results,
            spatial_relationships,
            engineering_relationships,
            connection_evidence,
            vlm_result=vlm_result
        )

        prompt = f"""
You are the engineering reasoning layer of SnapLab AI.

SnapLab AI uses a 65-class YOLO detector, MobileNet refinement,
engineering state tracking, spatial reasoning, engineering ontology
relationships, and optional Qualcomm on-device VLM visual interpretation.

Your job is to reason over the structured evidence provided below.

Do not invent components.

Do not treat component proximity as proof of electrical connectivity.

Do not treat an ontology relationship as proof that the relationship
exists in the photographed circuit.

If Qualcomm VLM observations are provided, evaluate them as descriptive
visual context alongside the detector evidence; do not treat them as verified electrical connections.

If Qualcomm VLM is unavailable, proceed strictly using the detector,
spatial, and ontology evidence.

Clearly separate:
1. Visual observations
2. Engineering possibilities
3. Conclusions supported by evidence
4. Conclusions that require additional visual evidence

Identify the most plausible circuit purpose only when the evidence
supports it.

If the topology is insufficient, explicitly say so.

Return a concise engineering analysis with these sections:

Engineering Assessment
Observed Components
Likely Functional Relationships
Connection Confidence
Possible Circuit Purpose
Potential Issues
What Should Be Verified Next

Structured evidence:

{evidence}
"""

        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a precise electronics "
                        "engineering reasoning assistant. "
                        "Never confuse inference with "
                        "observed evidence."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            analysis_text, used_model = self._call_chat_with_fallback(messages, temperature=0.3, max_tokens=1500)

            return {
                "available": True,
                "model": used_model,
                "analysis": analysis_text
            }

        except Exception as exc:

            error_msg = (
                "Engineering analysis failed: "
                f"{exc}"
            )

            return {
                "available": False,
                "model": self.model,
                "error": str(exc),
                "analysis": error_msg
            }

    def analyze_multi_image(
        self,
        aggregated_inventory,
        class_frequency,
        per_image_evidence,
        vlm_evidence_by_image=None
    ):

        if self.client is None:

            reason = (
                f"OpenAI client initialization failed: {self.init_error}"
                if self.init_error
                else (
                    "API key not configured. "
                    "Set GROQ_API_KEY or OPENAI_API_KEY to enable "
                    "engineering reasoning."
                )
            )

            return {
                "available": False,
                "model": self.model,
                "error": reason,
                "analysis": reason
            }

        evidence = {
            "total_images": len(per_image_evidence),
            "total_detections": len(aggregated_inventory),
            "class_frequency": class_frequency,
            "unified_inventory": aggregated_inventory,
            "per_image_evidence": per_image_evidence
        }

        if vlm_evidence_by_image:
            evidence["vlm_evidence_by_image"] = vlm_evidence_by_image

        prompt = f"""
You are the engineering reasoning layer of SnapLab AI, analyzing evidence from multiple uploaded images.

SnapLab AI uses a 65-class YOLO detector, MobileNet refinement, engineering state tracking,
spatial reasoning, engineering ontology relationships, and optional Qualcomm on-device VLM visual interpretation.

You have received aggregated evidence from {len(per_image_evidence)} uploaded images.

CRITICAL RULES:
- Components detected in DIFFERENT images must NOT be inferred to be electrically connected.
- Each image has its own isolated physical context.
- Do not assume different images show the same circuit, the same board, or the same components.
- Do not invent components.
- Do not treat spatial proximity as proof of electrical connectivity.
- Clearly distinguish total visual detections from unique physical components.
- If the same physical component appears in multiple images, acknowledge it may be so but do not assert it without evidence.
- If cross-image matching is uncertain, explicitly report that uncertainty.

Clearly separate for your analysis:
1. Per-image visual observations
2. Engineering possibilities within each image
3. Cross-image inventory observations (component types seen across uploads)
4. Conclusions supported by evidence
5. What cannot be determined from the available images

Return a concise multi-image engineering analysis with these sections:

Multi-Image Engineering Assessment
Observed Components Per Image
Combined Component Inventory Summary
Cross-Image Engineering Notes
Connection Evidence by Image
Possible Purpose of Each Circuit/Assembly
Limitations and Uncertainties
What Should Be Verified Next

Structured multi-image evidence:

{evidence}
"""

        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a precise electronics "
                        "engineering reasoning assistant analyzing "
                        "multiple uploaded circuit images. "
                        "Never confuse inference with observed evidence. "
                        "Never fabricate cross-image electrical connections."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            analysis_text, used_model = self._call_chat_with_fallback(messages, temperature=0.3, max_tokens=1800)

            return {
                "available": True,
                "model": used_model,
                "analysis": analysis_text
            }

        except Exception as exc:

            error_msg = (
                "Multi-image engineering analysis failed: "
                f"{exc}"
            )

            return {
                "available": False,
                "model": self.model,
                "error": str(exc),
                "analysis": error_msg
            }

    def generate_card_reasoning(
        self,
        all_components,
        per_image_evidence=None,
        vlm_evidence_by_image=None
    ):
        """
        Generate concise project archetype and 4 actionable engineering review bullet points
        specifically tailored to the detected combination of components.
        Works via active LLM API (Groq / OpenAI) with automatic fallback to rich
        rule-based ontology reasoning.
        """
        if not all_components:
            return {
                "model": self.model or "SnapLab Engineering Reasoning",
                "status": "Awaiting circuit input",
                "project_title": "Circuit Analysis Engine Ready",
                "bullets": [
                    "Upload one or more circuit images to trigger on-device component identification.",
                    "YOLO detector analyzes microcontroller, sensor, actuator, and passive footprints.",
                    "MobileNet refinement classifies fine-grained IC packages and pin orientations.",
                    "Engineering reasoning validates voltage domains, pinouts, and safety."
                ],
                "is_live_api": False
            }

        # Format detected component list
        comp_list = []
        for c in all_components:
            cls_name = c.get("final_class", "Unknown")
            conf = c.get("final_confidence", 0.90)
            comp_list.append(f"{cls_name} ({conf*100:.0f}%)")

        summary_str = ", ".join(comp_list[:12])

        # Try live API call
        if self.client is not None:
            try:
                prompt = f"""You are the electronics engineering reasoning copilot of SnapLab AI.
Analyze these detected components on the circuit board:
{summary_str}

Infer the most plausible circuit project purpose and generate 4 concise, highly technical engineering inspection bullet points.
Requirements for bullets:
1. Power rail & operating voltage compatibility check
2. Signal/GPIO bus interfacing check (I2C/SPI/PWM/ADC)
3. Protection & electrical safety (inductive kickback, current limit, fuses, heat)
4. Decoupling capacitors & noise suppression

Respond ONLY with a valid JSON object in this exact schema:
{{
  "project_title": "A short, descriptive 3-6 word project name",
  "bullets": [
    "Specific technical check 1",
    "Specific technical check 2",
    "Specific technical check 3",
    "Specific technical check 4"
  ]
}}"""

                messages = [
                    {
                        "role": "system",
                        "content": "You are a precise electronics engineering verification assistant. Output valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]

                content, used_model = self._call_chat_with_fallback(messages, temperature=0.3, max_tokens=600)
                
                import json
                import re
                cleaned = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.MULTILINE)
                cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
                
                data = json.loads(cleaned)
                project_title = data.get("project_title", "Custom Embedded Microcontroller Assembly")
                bullets = data.get("bullets", [])

                if len(bullets) >= 3:
                    return {
                        "model": used_model,
                        "status": "Analysis completed",
                        "project_title": project_title,
                        "bullets": bullets[:4],
                        "is_live_api": True
                    }
            except Exception:
                pass

        # If API call fails or client is None, run dynamic rule-based generator
        return self._generate_rule_based_reasoning(all_components)

    def _generate_rule_based_reasoning(self, all_components):
        """
        Intelligent rule-based engineering reasoning engine that inspects the detected components
        and generates an accurate, dynamic project archetype and 4 actionable engineering inspection bullets.
        Works 100% offline or as a rock-solid fallback.
        """
        if not all_components:
            return {
                "model": "SnapLab Rule-Based Reasoning Engine",
                "status": "Analysis completed",
                "project_title": "General Electronics Prototype",
                "bullets": [
                    "Ensure main supply voltage matches component tolerances (3.3V / 5.0V).",
                    "Verify common ground (GND) across all sensor and microcontroller boards.",
                    "Inspect signal traces and pin assignments against respective datasheets.",
                    "Include bulk bypass and decoupling capacitors near sensitive logic."
                ],
                "is_live_api": False
            }

        class_list = [c.get("final_class", "") for c in all_components]
        classes_lower = [c.lower() for c in class_list]

        has_esp32_cam = any("esp32-cam" in c or "esp32" in c for c in classes_lower)
        has_arduino = any("arduino" in c for c in classes_lower)
        has_rpi = any("raspberry" in c or "pi" in c or "pico" in c for c in classes_lower)
        has_camera = any("cam" in c or "camera" in c for c in classes_lower)
        has_ultrasonic = any("ultrasonic" in c or "hc-sr04" in c for c in classes_lower)
        has_raindrop = any("rain" in c or "water" in c or "moisture" in c for c in classes_lower)
        has_dht = any("dht" in c or "temp" in c or "humidity" in c for c in classes_lower)
        has_pir = any("pir" in c or "motion" in c for c in classes_lower)
        has_motor = any("motor" in c and "servo" not in c and "driver" not in c and "shield" not in c for c in classes_lower)
        has_servo = any("servo" in c for c in classes_lower)
        has_driver = any("driver" in c or "l298" in c for c in classes_lower)
        has_relay = any("relay" in c for c in classes_lower)
        has_display = any("oled" in c or "lcd" in c or "display" in c for c in classes_lower)
        has_potentiometer = any("potentiometer" in c or "pot" in c for c in classes_lower)
        has_buzzer = any("buzzer" in c for c in classes_lower)
        has_battery = any("battery" in c or "cell" in c for c in classes_lower)
        has_capacitor = any("capacitor" in c or "cap" in c for c in classes_lower)

        # 1. Project Archetype
        if has_camera and (has_raindrop or has_dht or has_pir):
            project_title = "Vision IoT Environmental Sensing Station"
        elif has_camera:
            project_title = "On-Device Visual Inspection & Telemetry Node"
        elif has_ultrasonic and (has_servo or has_motor):
            project_title = "Obstacle-Avoidance Autonomous Robotics Platform"
        elif has_raindrop and (has_motor or has_relay or has_buzzer):
            project_title = "Automated Rain-Triggered Actuation & Drainage System"
        elif has_raindrop:
            project_title = "Precipitation & Liquid Level Detection Monitor"
        elif has_dht and has_display:
            project_title = "Environmental Climate Telemetry Instrumentation"
        elif has_motor and has_potentiometer:
            project_title = "Closed-Loop Variable Speed Motor Drive Controller"
        elif has_motor or has_servo:
            project_title = "Electro-Mechanical Actuator & Motion Subsystem"
        elif has_relay:
            project_title = "Mains / High-Power Load Switching Controller"
        elif has_arduino or has_esp32_cam or has_rpi:
            mcu_name = "ESP32-CAM" if has_esp32_cam else ("Arduino" if has_arduino else "Embedded")
            project_title = f"{mcu_name} Multi-Sensor Embedded Microcontroller Core"
        else:
            project_title = "Embedded Circuit Module & Sensor Prototype"

        # 2. Four tailored engineering bullets
        bullets = []

        if has_esp32_cam and (has_motor or has_servo):
            bullets.append("ESP32-CAM requires a dedicated 5V/2A power rail; isolate motor current spikes with separate power or Schottky barrier to prevent brownout resets.")
        elif has_motor or has_servo:
            bullets.append("Verify DC motor / servo peak stall current capacity; never power inductive loads directly from microcontroller 3.3V or 5V logic pins.")
        elif has_esp32_cam:
            bullets.append("ESP32-CAM operating voltage is strictly 3.3V logic (5V VCC input permitted only on onboard regulator pin). Verify supply ripple < 100mV.")
        elif has_battery:
            bullets.append("Ensure battery terminal voltage is regulated; verify reverse polarity protection diode and low-dropout (LDO) regulator thermal headroom.")
        else:
            bullets.append("Confirm logic voltage domain compatibility across all connected modules (3.3V CMOS vs 5.0V TTL).")

        if has_ultrasonic:
            bullets.append("HC-SR04 Echo pin outputs 5V TTL; implement a voltage divider (1kΩ/2kΩ) if connected to 3.3V microcontroller inputs to prevent GPIO damage.")
        elif has_raindrop:
            bullets.append("Raindrop sensor LM393 comparator threshold must be calibrated via onboard trimmer; verify digital D0 vs analog A0 routing.")
        elif has_potentiometer:
            bullets.append("Verify potentiometer wiper is wired to an ADC-capable analog pin (e.g., ESP32 GPIO32-39 or Arduino A0-A5) with low source impedance.")
        elif has_display:
            bullets.append("Inspect display communication bus (I2C SDA/SCL lines require 4.7kΩ pull-up resistors; verify I2C device address 0x3C or 0x3D).")
        else:
            bullets.append("Verify all GPIO pins are assigned to non-bootloader strapping pins to prevent boot failures on power-up.")

        if has_motor or has_relay:
            bullets.append("Ensure a flyback / freewheeling diode (e.g. 1N4007 or 1N4148) is installed antiparallel across the coil to clamp inductive voltage spikes.")
        elif has_servo:
            bullets.append("Verify PWM control frequency (typically 50 Hz, 1ms - 2ms pulse width) and confirm common ground between MCU and servo power.")
        else:
            bullets.append("Check current-limiting series resistors on any LEDs or optocouplers to prevent junction overcurrent.")

        if has_capacitor:
            bullets.append("Electrolytic and ceramic decoupling capacitors must be positioned as close as possible to the IC power pins to suppress transient voltage dips.")
        else:
            bullets.append("Add 0.1µF ceramic bypass capacitors near active IC VCC pins and 100µF bulk capacitor near power entry to stabilize transient response.")

        return {
            "model": "SnapLab Engineering Rule Engine",
            "status": "Analysis completed",
            "project_title": project_title,
            "bullets": bullets,
            "is_live_api": False
        }
