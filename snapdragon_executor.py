import json
import glob
import numpy as np
import qai_hub as hub


INFERENCE_JOB_ID = "jp8xv9w8g"
PROFILE_JOB_ID = "j5w7nvw4g"


def find_qhas_summary():
    files = glob.glob(
        r"C:\Users\purus\AppData\Local\Temp\snaplab_profile_*\*_qhas_summary.json"
    )

    if not files:
        raise FileNotFoundError(
            "QHAS summary not found. Download the profile artifact first."
        )

    return files[-1]


def extract_qhas_metrics():
    qhas_file = find_qhas_summary()

    with open(qhas_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data["data"]["htp_overall_summary"]["data"][0]

    return {
        "graph_execute_us": summary["graph_execute_us"],
        "time_us": summary["time_us"],
        "htp_utilization_percent": summary["percent_utilization"],
        "inferences_per_second": summary["inf_per_s"],
        "peak_vtcm_bytes": summary["peak_vtcm_alloc"],
        "dram_read_bytes": summary["total_dram_read"],
        "dram_write_bytes": summary["total_dram_write"],
        "qnn_nodes": summary["qnn_nodes"],
        "htp_nodes": summary["htp_nodes"],
        "qhas_file": qhas_file,
    }


def run_snapdragon_inference():

    print("=" * 70)
    print("SNAPLAB-AI — SNAPDRAGON EXECUTION REPORT")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. Qualcomm AI Hub inference
    # ------------------------------------------------------------

    inference_job = hub.get_job(INFERENCE_JOB_ID)
    inference_status = inference_job.get_status()

    print()
    print("INFERENCE")
    print("-" * 70)
    print(f"Job ID       : {INFERENCE_JOB_ID}")
    print(f"Status       : {inference_status.code}")

    if inference_status.code != "SUCCESS":
        raise RuntimeError(
            f"Snapdragon inference failed: {inference_status}"
        )

    output_data = inference_job.download_output_data()

    if "output_0" not in output_data:
        raise KeyError(
            f"Expected output_0, found: {list(output_data.keys())}"
        )

    output = np.asarray(output_data["output_0"][0])

    logits = output.reshape(-1)

    top1 = int(np.argmax(logits))
    top5 = np.argsort(logits)[-5:][::-1].tolist()

    print(f"Device       : Snapdragon X Elite")
    print(f"Runtime      : Qualcomm AI Hub / NPU")
    print(f"Output       : output_0")
    print(f"Shape        : {output.shape}")
    print(f"Dtype        : {output.dtype}")
    print(f"Top-1 class  : {top1}")
    print(f"Top-5 classes: {top5}")

    # ------------------------------------------------------------
    # 2. Qualcomm profile
    # ------------------------------------------------------------

    profile_job = hub.get_job(PROFILE_JOB_ID)
    profile_status = profile_job.get_status()

    print()
    print("NPU PERFORMANCE")
    print("-" * 70)
    print(f"Profile Job  : {PROFILE_JOB_ID}")
    print(f"Status       : {profile_status.code}")

    if profile_status.code != "SUCCESS":
        raise RuntimeError(
            f"Snapdragon profile failed: {profile_status}"
        )

    metrics = extract_qhas_metrics()

    graph_latency_ms = metrics["graph_execute_us"] / 1000.0
    htp_time_ms = metrics["time_us"] / 1000.0
    peak_vtcm_mib = metrics["peak_vtcm_bytes"] / (1024 ** 2)

    print(f"Graph latency: {graph_latency_ms:.3f} ms")
    print(f"HTP time     : {htp_time_ms:.3f} ms")
    print(
        f"HTP utilization: "
        f"{metrics['htp_utilization_percent']:.2f}%"
    )
    print(
        f"Throughput   : "
        f"{metrics['inferences_per_second']:.2f} inf/s"
    )
    print(f"Peak VTCM    : {peak_vtcm_mib:.2f} MiB")
    print(f"DRAM read    : {metrics['dram_read_bytes']:,} bytes")
    print(f"DRAM write   : {metrics['dram_write_bytes']:,} bytes")
    print(f"QNN nodes    : {metrics['qnn_nodes']}")
    print(f"HTP nodes    : {metrics['htp_nodes']}")

    # ------------------------------------------------------------
    # 3. Structured result
    # ------------------------------------------------------------

    result = {
        "prediction": top1,
        "top5": top5,

        "device": "Snapdragon X Elite",
        "runtime": "Qualcomm AI Hub / NPU",

        "inference_job_id": INFERENCE_JOB_ID,
        "profile_job_id": PROFILE_JOB_ID,

        "latency_ms": graph_latency_ms,
        "htp_time_ms": htp_time_ms,

        "htp_utilization_percent":
            metrics["htp_utilization_percent"],

        "throughput_inferences_per_second":
            metrics["inferences_per_second"],

        "peak_vtcm_bytes":
            metrics["peak_vtcm_bytes"],

        "dram_read_bytes":
            metrics["dram_read_bytes"],

        "dram_write_bytes":
            metrics["dram_write_bytes"],

        "qnn_nodes":
            metrics["qnn_nodes"],

        "htp_nodes":
            metrics["htp_nodes"],
    }

    print()
    print("STRUCTURED RESULT")
    print("-" * 70)
    print(json.dumps(result, indent=2))

    print()
    print("EXECUTION SUCCESS")
    print("=" * 70)

    return result


if __name__ == "__main__":
    run_snapdragon_inference()
