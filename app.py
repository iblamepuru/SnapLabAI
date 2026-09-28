import os
import sys

# Ensure project root is in the Python search path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

print("=" * 60, flush=True)
print("⚡ SnapLab AI: Real-Time Circuit Engineering Copilot", flush=True)
print("=" * 60, flush=True)
print("[SnapLab AI] Loading neural detectors, spatial engines & Gradio UI...", flush=True)
print("[SnapLab AI] Please wait ~30-40 seconds for model initialization...", flush=True)

# Import the flagship Gradio application
from vision.component_inspector_v2 import demo

print("[SnapLab AI] Components loaded successfully! Starting web server...", flush=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    # Default share=False to avoid hanging on frpc tunnel downloads on Windows
    share = os.environ.get("SHARE", "false").lower() in ("true", "1", "yes")

    print(f"\n[SnapLab AI] 🚀 Launching on http://127.0.0.1:{port} (share={share})\n", flush=True)
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=share
    )

