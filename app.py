import os
import sys

# Ensure project root is in the Python search path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import the flagship Gradio application
from vision.component_inspector_v2 import demo

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    share = os.environ.get("SHARE", "true").lower() in ("true", "1", "yes")
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        theme=None
    )
