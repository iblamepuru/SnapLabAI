import json
import os
import platform
import subprocess
from pathlib import Path


class VLMEngine:

    def __init__(self, manifest_path=None):
        root_dir = Path(__file__).resolve().parents[2]

        if manifest_path is None:
            manifest_path = root_dir / "vision" / "vlm" / "vlm_deployment.json"

        self.manifest_path = Path(manifest_path)
        self.manifest = self._load_manifest()

        self.model_name = self.manifest["model"]["name"]
        self.backend = self.manifest["runtime"]["backend"]
        self.model_dir = root_dir / self.manifest["model"]["bundle"]

        self.target_device = self.manifest["target"]["device"]
        self.target_architecture = self.manifest["target"]["architecture"]

        self.model = None
        self.available = False

    def _load_manifest(self):
        with open(self.manifest_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _processor_name(self):
        values = []

        for key in (
            "PROCESSOR_IDENTIFIER",
            "PROCESSOR_ARCHITECTURE",
            "PROCESSOR_LEVEL",
            "PROCESSOR_REVISION"
        ):
            value = os.environ.get(key)

            if value:
                values.append(value)

        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )

            value, _ = winreg.QueryValueEx(
                key,
                "ProcessorNameString"
            )

            if value:
                values.append(str(value))

            winreg.CloseKey(key)

        except Exception:
            pass

        if platform.system() == "Windows":

            try:
                result = subprocess.run(
                    [
                        "powershell",
                        "-NoProfile",
                        "-Command",
                        "(Get-CimInstance Win32_Processor).Name"
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                if result.stdout.strip():
                    values.append(result.stdout.strip())

            except Exception:
                pass

        return " | ".join(
            dict.fromkeys(
                value.strip()
                for value in values
                if value and value.strip()
            )
        )

    def _is_snapdragon_x_series(self, processor_name):
        text = processor_name.lower()

        indicators = [
            "qualcomm",
            "snapdragon",
            "oryon",
            "x elite",
            "x plus",
            "x1e",
            "x1p"
        ]

        return any(
            indicator in text
            for indicator in indicators
        )

    def status(self):

        host_os = platform.system()
        host_architecture = platform.machine()
        processor = self._processor_name()

        arm64 = host_architecture.upper() in {
            "ARM64",
            "AARCH64"
        }

        snapdragon_x = self._is_snapdragon_x_series(
            processor
        )

        target_reached = (
            host_os == "Windows"
            and arm64
            and snapdragon_x
        )

        bundle_files = [
            "config.json",
            "genie_config.json",
            "img-enc-htp.json",
            "text-encoder.json",
            "text-generator.json",
            "tokenizer.json",
            "vision_encoder.bin"
        ]

        bundle_ready = self.model_dir.exists() and all(
            (self.model_dir / filename).exists()
            for filename in bundle_files
        )

        if target_reached:

            if bundle_ready:

                status = "TARGET DEVICE REACHED"

                reason = (
                    "Qualcomm Snapdragon X-series ARM64 hardware "
                    "matches the target execution platform."
                )

            else:

                status = "TARGET DEVICE REACHED"

                reason = (
                    "Qualcomm Snapdragon X-series ARM64 hardware "
                    "matches the target platform, but the VLM bundle "
                    "is incomplete."
                )

        else:

            status = "TARGET BACKEND UNAVAILABLE"

            if host_os != "Windows":

                reason = (
                    "GenieX QAIRT VLM execution requires the "
                    "supported Windows Snapdragon environment."
                )

            elif not arm64:

                reason = (
                    "Snapdragon X-series ARM64 hardware was not detected."
                )

            elif not snapdragon_x:

                reason = (
                    "ARM64 Windows detected, but Qualcomm Snapdragon "
                    "X-series hardware was not detected."
                )

            else:

                reason = (
                    "The current host does not match the required "
                    "Snapdragon X-series target."
                )

        return {
            "model": self.model_name,
            "backend": self.backend,
            "model_dir": str(self.model_dir),
            "available": target_reached and bundle_ready,
            "target_reached": target_reached,
            "host_os": host_os,
            "host_architecture": host_architecture,
            "processor": processor,
            "target_device": self.target_device,
            "target_architecture": self.target_architecture,
            "execution": self.manifest["runtime"]["execution"],
            "status": status,
            "reason": reason,
            "bundle_ready": bundle_ready
        }

    def load(self):

        current_status = self.status()

        if not current_status["target_reached"]:
            self.available = False
            return False

        if not current_status["bundle_ready"]:
            self.available = False
            return False

        try:
            from geniex import AutoModelForVision2Seq

            self.model = AutoModelForVision2Seq.from_pretrained(
                str(self.model_dir),
                device_map="qairt"
            )

            self.available = True

            return True

        except Exception:
            self.model = None
            self.available = False

            return False

    def analyze(
        self,
        image_path,
        prompt,
        max_new_tokens=256
    ):

        if not self.available:

            if not self.load():
                return None

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str(image_path)
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        formatted_prompt = self.model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        output = self.model.generate(
            formatted_prompt,
            images=[str(image_path)],
            max_new_tokens=max_new_tokens
        )

        return output

    def close(self):

        self.model = None
        self.available = False
