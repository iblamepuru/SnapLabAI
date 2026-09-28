import json
from pathlib import Path


BENCHMARK_FILE = Path(__file__).resolve().parent / "benchmark_results.json"


def load_benchmarks():
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_comparison(data):
    b = data["benchmarks"]
    accuracy = data["accuracy_consistency"]

    cpu = b["cpu_fp32"]
    source = b["snapdragon_fp32_source"]
    compiled = b["snapdragon_fp32_compiled"]
    int8 = b["snapdragon_int8"]

    cpu_to_int8_speedup = (
        cpu["latency_ms_mean"] / int8["graph_latency_ms"]
    )

    compiled_to_int8_reduction = (
        1 - int8["graph_latency_ms"] / compiled["latency_ms"]
    ) * 100

    source_to_compiled_reduction = (
        1 - compiled["latency_ms"] / source["latency_ms"]
    ) * 100

    memory_reduction = (
        1
        - int8["estimated_peak_memory_bytes"]
        / source["peak_memory_bytes"]
    ) * 100

    return {
        "cpu_to_int8_speedup": cpu_to_int8_speedup,
        "compiled_to_int8_latency_reduction_percent":
            compiled_to_int8_reduction,
        "source_to_compiled_latency_reduction_percent":
            source_to_compiled_reduction,
        "fp32_to_int8_memory_reduction_percent":
            memory_reduction,
        "int8_graph_latency_ms":
            int8["graph_latency_ms"],
        "int8_htp_time_ms":
            int8["htp_time_ms"],
        "int8_htp_utilization_percent":
            int8["htp_utilization_percent"],
        "int8_throughput_inf_per_s":
            int8["throughput_inferences_per_second"],
        "top1_agreement_percent":
            accuracy["top1_agreement_percent"],
        "mean_cosine_similarity":
            accuracy["mean_cosine_similarity"],
    }


def print_comparison(results):
    print("=" * 70)
    print("SNAPLAB-AI — PERFORMANCE COMPARISON")
    print("=" * 70)

    print()
    print("LATENCY")
    print("-" * 70)
    print(
        f"CPU FP32 mean              : "
        f"{data['benchmarks']['cpu_fp32']['latency_ms_mean']:.3f} ms"
    )
    print(
        f"Snapdragon FP32 source     : "
        f"{data['benchmarks']['snapdragon_fp32_source']['latency_ms']:.3f} ms"
    )
    print(
        f"Snapdragon FP32 compiled   : "
        f"{data['benchmarks']['snapdragon_fp32_compiled']['latency_ms']:.3f} ms"
    )
    print(
        f"Snapdragon INT8 graph      : "
        f"{results['int8_graph_latency_ms']:.3f} ms"
    )

    print()
    print("OPTIMIZATION")
    print("-" * 70)
    print(
        f"CPU → INT8 speedup         : "
        f"{results['cpu_to_int8_speedup']:.2f}x"
    )
    print(
        f"Compiled FP32 → INT8       : "
        f"{results['compiled_to_int8_latency_reduction_percent']:.2f}% lower latency"
    )
    print(
        f"Source → compiled FP32     : "
        f"{results['source_to_compiled_latency_reduction_percent']:.2f}% lower latency"
    )
    print(
        f"FP32 → INT8 memory         : "
        f"{results['fp32_to_int8_memory_reduction_percent']:.2f}% reduction"
    )

    print()
    print("NPU")
    print("-" * 70)
    print(
        f"HTP utilization            : "
        f"{results['int8_htp_utilization_percent']:.2f}%"
    )
    print(
        f"Throughput                 : "
        f"{results['int8_throughput_inf_per_s']:.2f} inf/s"
    )

    print()
    print("PREDICTION CONSISTENCY")
    print("-" * 70)
    print(
        f"Top-1 agreement            : "
        f"{results['top1_agreement_percent']:.2f}%"
    )
    print(
        f"Mean cosine similarity     : "
        f"{results['mean_cosine_similarity']:.6f}"
    )

    print()
    print("=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    data = load_benchmarks()
    results = calculate_comparison(data)
    print_comparison(results)
