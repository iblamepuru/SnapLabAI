import json
from pathlib import Path

from performance_comparison import load_benchmarks, calculate_comparison
from recommendation_engine import generate_recommendation


OUTPUT_FILE = Path("engineering_report.json")


def build_report():
    data = load_benchmarks()
    comparison = calculate_comparison(data)
    recommendation = generate_recommendation(data, comparison)

    report = {
        "project": data["project"],
        "model": data["model"],
        "input_shape": data["input_shape"],

        "benchmark_data": data["benchmarks"],

        "accuracy_consistency": data["accuracy_consistency"],

        "performance_comparison": comparison,

        "engineering_recommendation": recommendation
    }

    return report


def save_report(report):
    OUTPUT_FILE.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )


if __name__ == "__main__":
    report = build_report()
    save_report(report)

    print("=" * 70)
    print("SNAPLAB-AI — UNIFIED ENGINEERING REPORT")
    print("=" * 70)

    print()
    print(f"Project       : {report['project']}")
    print(f"Model         : {report['model']}")
    print(f"Input shape   : {report['input_shape']}")

    print()
    print("PERFORMANCE")
    print("-" * 70)

    comparison = report["performance_comparison"]

    print(
        f"CPU → INT8 speedup          : "
        f"{comparison['cpu_to_int8_speedup']:.2f}x"
    )

    print(
        f"FP32 → INT8 latency         : "
        f"{comparison['compiled_to_int8_latency_reduction_percent']:.2f}% lower"
    )

    print(
        f"FP32 → INT8 memory          : "
        f"{comparison['fp32_to_int8_memory_reduction_percent']:.2f}% lower"
    )

    print()
    print("NPU")
    print("-" * 70)

    print(
        f"HTP utilization             : "
        f"{comparison['int8_htp_utilization_percent']:.2f}%"
    )

    print(
        f"INT8 throughput             : "
        f"{comparison['int8_throughput_inf_per_s']:.2f} inf/s"
    )

    print()
    print("CONSISTENCY")
    print("-" * 70)

    accuracy = report["accuracy_consistency"]

    print(
        f"Top-1 agreement             : "
        f"{accuracy['top1_agreement_percent']:.2f}%"
    )

    print(
        f"Mean cosine similarity      : "
        f"{accuracy['mean_cosine_similarity']:.6f}"
    )

    print()
    print("ENGINEERING DECISION")
    print("-" * 70)

    recommendation = report["engineering_recommendation"]

    print(f"Decision                    : {recommendation['decision']}")
    print(f"Confidence                  : {recommendation['confidence']}")

    print()
    print("Recommended action:")
    print(recommendation["action"])

    print()
    print(f"Report saved                : {OUTPUT_FILE}")
    print()
    print("=" * 70)
    print("REPORT GENERATION COMPLETE")
    print("=" * 70)

