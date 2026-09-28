import json
from pathlib import Path


REPORT_FILE = Path("engineering_report.json")


def load_report():
    with REPORT_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def answer_query(query):
    report = load_report()

    q = query.lower().strip()

    comparison = report["performance_comparison"]
    accuracy = report["accuracy_consistency"]
    recommendation = report["engineering_recommendation"]
    benchmarks = report["benchmark_data"]

    if "latency" in q or "speed" in q or "faster" in q or "speedup" in q:
        return (
            f"Snapdragon INT8 graph latency is "
            f"{comparison['int8_graph_latency_ms']:.3f} ms. "
            f"This is {comparison['compiled_to_int8_latency_reduction_percent']:.2f}% "
            f"lower than compiled Snapdragon FP32 latency, and the CPU FP32 "
            f"baseline is {benchmarks['cpu_fp32']['latency_ms_mean']:.3f} ms."
        )

    if "memory" in q:
        return (
            f"INT8 reduces estimated peak memory by "
            f"{comparison['fp32_to_int8_memory_reduction_percent']:.2f}% "
            f"relative to the Snapdragon FP32 compiled configuration. "
            f"The INT8 estimated peak memory is "
            f"{benchmarks['snapdragon_int8']['estimated_peak_memory_bytes'] / (1024**2):.2f} MiB."
        )

    if "npu" in q or "utilization" in q or "throughput" in q:
        return (
            f"The Snapdragon NPU/HTP shows "
            f"{comparison['int8_htp_utilization_percent']:.2f}% utilization "
            f"with INT8. Measured graph latency is "
            f"{comparison['int8_graph_latency_ms']:.3f} ms and throughput is "
            f"{comparison['int8_throughput_inf_per_s']:.2f} inferences/s."
        )

    if "accuracy" in q or "accurate" in q or "consistency" in q:
        return (
            f"The 10-sample synthetic validation produced "
            f"{accuracy['top1_agreement_percent']:.2f}% Top-1 prediction agreement "
            f"and mean cosine similarity of "
            f"{accuracy['mean_cosine_similarity']:.6f}. "
            f"This is a prediction-consistency test, not ImageNet accuracy."
        )

    if ("recommend" in q or "deploy" in q or "decision" in q or "why" in q and "int8" in q or "use int8" in q):
        return (
            f"Engineering decision: {recommendation['decision']} "
            f"with {recommendation['confidence']} confidence. "
            f"{recommendation['action']}"
        )

    if "summary" in q or "overall" in q:
        return (
            f"For {report['model']} on Snapdragon X Elite, INT8 achieves "
            f"{comparison['int8_graph_latency_ms']:.3f} ms graph latency, "
            f"{comparison['cpu_to_int8_speedup']:.2f}x speedup over the CPU FP32 "
            f"baseline, {comparison['fp32_to_int8_memory_reduction_percent']:.2f}% "
            f"estimated memory reduction, and "
            f"{comparison['int8_htp_utilization_percent']:.2f}% HTP utilization. "
            f"The current engineering recommendation is "
            f"{recommendation['decision']}."
        )

    return (
        "I can answer questions about latency, memory, NPU utilization, "
        "throughput, accuracy consistency, recommendation, or provide an "
        "overall summary."
    )


if __name__ == "__main__":
    print("=" * 70)
    print("SNAPLAB-AI — ENGINEERING COPILOT")
    print("=" * 70)

    print()
    print("Available queries:")
    print("  summary")
    print("  latency")
    print("  memory")
    print("  npu")
    print("  accuracy")
    print("  recommendation")

    print()
    print("Example:")
    print("  python copilot_reasoning.py \"Why should I use INT8?\"")

    print()
    print("=" * 70)



