from pathlib import Path

import numpy as np
import onnx
import qai_hub as hub


# ============================================================
# SNAPLAB AI — CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

# Your downloaded Qualcomm AI Hub target model
SOURCE_DIR = (
    PROJECT_DIR
    / "models"
    / "resnet50_xelite"
    / "SnapLab-ResNet50-XElite.onnx"
)

SOURCE_ONNX = SOURCE_DIR / "model.onnx"
SOURCE_DATA = SOURCE_DIR / "model.data"

# New single-file ONNX model containing embedded weights
MERGED_ONNX = (
    PROJECT_DIR
    / "models"
    / "resnet50_xelite"
    / "SnapLab-ResNet50-XElite-Full.onnx"
)

DEVICE_NAME = "Snapdragon X Elite CRD"


# ============================================================
# STEP 1 — CHECK SOURCE FILES
# ============================================================

print("=" * 65)
print("SNAPLAB AI")
print("Snapdragon X Elite — ResNet50 Inference Validation")
print("=" * 65)

print("\n[1/7] Checking Qualcomm target model files...")

print("\nTarget directory:")
print(SOURCE_DIR)

print("\nONNX file:")
print(SOURCE_ONNX)

print("\nDATA file:")
print(SOURCE_DATA)


if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"\nTarget model directory not found:\n{SOURCE_DIR}"
    )

if not SOURCE_ONNX.exists():
    raise FileNotFoundError(
        f"\nONNX file not found:\n{SOURCE_ONNX}"
    )

if not SOURCE_DATA.exists():
    raise FileNotFoundError(
        f"\nDATA file not found:\n{SOURCE_DATA}"
    )


onnx_size = SOURCE_ONNX.stat().st_size
data_size = SOURCE_DATA.stat().st_size

print("\n✓ model.onnx found")
print(
    f"  Size: {onnx_size / (1024 * 1024):.2f} MB"
)

print("✓ model.data found")
print(
    f"  Size: {data_size / (1024 * 1024):.2f} MB"
)


# ============================================================
# STEP 2 — LOAD ONNX + EXTERNAL WEIGHTS
# ============================================================

print("\n" + "=" * 65)
print("[2/7] Loading ONNX model with external weights...")
print("=" * 65)

print("\nLoading:")

print(SOURCE_ONNX)

model = onnx.load(
    str(SOURCE_ONNX),
    load_external_data=True
)

print("\n✓ ONNX model loaded")
print("✓ External model.data loaded")


# ============================================================
# STEP 3 — EMBED EXTERNAL WEIGHTS
# ============================================================

print("\n" + "=" * 65)
print("[3/7] Embedding external weights...")
print("=" * 65)

print("\nConverting external data into the ONNX model...")

onnx.external_data_helper.convert_model_from_external_data(
    model
)

print("✓ External weights converted")


print("\nSaving single-file ONNX model:")

print(MERGED_ONNX)

onnx.save(
    model,
    str(MERGED_ONNX),
    save_as_external_data=False
)

print("\n✓ Single-file ONNX model created")


merged_size = MERGED_ONNX.stat().st_size

print(
    f"✓ Merged model size: "
    f"{merged_size / (1024 * 1024):.2f} MB"
)


# ============================================================
# STEP 4 — VERIFY MERGED MODEL
# ============================================================

print("\n" + "=" * 65)
print("[4/7] Verifying merged ONNX model...")
print("=" * 65)

verified_model = onnx.load(
    str(MERGED_ONNX),
    load_external_data=False
)

print("\n✓ Merged ONNX model loaded successfully")


print("\nMODEL INPUTS")

for inp in verified_model.graph.input:

    print(
        f"  Name: {inp.name}"
    )

    print(
        f"  Type: {inp.type.tensor_type}"
    )


print("\nMODEL OUTPUTS")

for out in verified_model.graph.output:

    print(
        f"  Name: {out.name}"
    )

    print(
        f"  Type: {out.type.tensor_type}"
    )


# ============================================================
# STEP 5 — CREATE TEST INPUT
# ============================================================

print("\n" + "=" * 65)
print("[5/7] Creating test input...")
print("=" * 65)

# ResNet50 input specification from Qualcomm AI Hub
input_data = np.random.rand(
    1,
    3,
    224,
    224
).astype(np.float32)

print("\nInput information:")
print("  Shape :", input_data.shape)
print("  Dtype :", input_data.dtype)
print(
    "  Min   :",
    input_data.min()
)
print(
    "  Max   :",
    input_data.max()
)


# ============================================================
# STEP 6 — SUBMIT AI HUB INFERENCE
# ============================================================

print("\n" + "=" * 65)
print("[6/7] Submitting Snapdragon X Elite inference job...")
print("=" * 65)

print("\nTarget device:")
print(DEVICE_NAME)

print("\nUploading merged ONNX model to Qualcomm AI Hub...")

inference_job = hub.submit_inference_job(
    model=str(MERGED_ONNX),
    device=hub.Device(DEVICE_NAME),
    inputs={
        "x": [input_data]
    },
    name="SnapLab-X-Elite-ResNet50-Inference"
)

print("\n✓ Inference job submitted successfully")

print("\nJob ID:")
print(inference_job.job_id)

try:
    print("\nJob URL:")
    print(inference_job.url)
except Exception:
    pass


# ============================================================
# STEP 7 — WAIT + GET RESULT
# ============================================================

print("\n" + "=" * 65)
print("[7/7] Waiting for inference result...")
print("=" * 65)

inference_job.wait()

print("\n✓ Inference job finished")


print("\nDownloading output data...")

output = inference_job.download_output_data()


if output is None:

    print("\n❌ No output data was returned.")

    print(
        "\nPlease open the AI Hub job page and check "
        "the detailed error."
    )

    raise SystemExit(1)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 65)
print("INFERENCE SUCCESS")
print("=" * 65)

print(
    "\nThe Snapdragon X Elite target model "
    "successfully produced output."
)


for name, tensors in output.items():

    print("\n--------------------------------------------")

    print(
        f"Output name: {name}"
    )

    if isinstance(tensors, list):

        for index, tensor in enumerate(tensors):

            print(
                f"\nTensor {index}:"
            )

            print(
                "  Shape :",
                tensor.shape
            )

            print(
                "  Dtype :",
                tensor.dtype
            )

            if tensor.size > 0:

                print(
                    "  Min   :",
                    float(tensor.min())
                )

                print(
                    "  Max   :",
                    float(tensor.max())
                )

                print(
                    "  Mean  :",
                    float(tensor.mean())
                )

    else:

        print(
            "  Shape :",
            tensors.shape
        )

        print(
            "  Dtype :",
            tensors.dtype
        )


print("\n" + "=" * 65)
print("SNAPLAB AI — INFERENCE VALIDATION COMPLETE")
print("=" * 65)

print("\nBaseline:")
print("  Device          : Snapdragon X Elite CRD")
print("  Model           : ResNet50")
print("  Input           : 1 × 3 × 224 × 224")
print("  Profile latency : 1.8 ms")
print("  Peak memory     : 49 MB")

print("\n✓ Qualcomm AI Hub target inference test completed.")