import os
import sys

# Ensure project root is in the Python search path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

print("=" * 60, flush=True)
print("SnapLab AI: Real-Time Circuit Engineering Copilot", flush=True)
print("=" * 60, flush=True)
print(
    "[SnapLab AI] Loading neural detectors, spatial engines & Gradio UI...",
    flush=True,
)
print(
    "[SnapLab AI] Please wait ~30-40 seconds for model initialization...",
    flush=True,
)

# Import the flagship Gradio application
from vision.component_inspector_v2 import demo, CUSTOM_CSS

print(
    "[SnapLab AI] Components loaded successfully! Starting web server...",
    flush=True,
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    # On Render/cloud, bind to 0.0.0.0; for local development, default to 127.0.0.1
    # so clicking the printed URL opens properly in Windows browsers without ERR_ADDRESS_INVALID.
    default_host = "0.0.0.0" if os.environ.get("RENDER") else "127.0.0.1"
    server_name = os.environ.get("SERVER_NAME", default_host)

    # Disable Gradio's public share tunnel by default
    share = os.environ.get("SHARE", "false").lower() in (
        "true",
        "1",
        "yes",
    )

    print(
        f"\n[SnapLab AI] Launching on {server_name}:{port} (share={share})",
        flush=True,
    )
    print(
        f"[SnapLab AI] Open in browser: http://localhost:{port} or http://127.0.0.1:{port}\n",
        flush=True,
    )

    demo.launch(
        server_name=server_name,
        server_port=port,
        share=share,
        inbrowser=False,
        css=CUSTOM_CSS,
    )