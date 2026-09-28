from pathlib import Path
import onnxruntime as ort
import numpy as np

MODEL_DIR = Path(r"models\snaplab_v3_65class_npu\job_jp2r0zz6g_optimized_onnx")
MODEL_PATH = MODEL_DIR / "model.onnx"

class SnapdragonNPUAdapter:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def infer(self, tensor):
        tensor = np.asarray(tensor, dtype=np.float32)

        if tensor.shape != (1, 3, 640, 640):
            raise ValueError(
                f"Expected input shape (1, 3, 640, 640), got {tensor.shape}"
            )

        return self.session.run(
            [self.output_name],
            {self.input_name: tensor}
        )[0]

    def info(self):
        return {
            "model": str(MODEL_PATH),
            "input": self.session.get_inputs()[0].shape,
            "output": self.session.get_outputs()[0].shape,
            "providers": self.session.get_providers()
        }
