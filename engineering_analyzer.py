def analyze_performance(result):
    latency = result["latency_ms"]
    utilization = result["htp_utilization_percent"]
    throughput = result["throughput_inferences_per_second"]
    peak_vtcm = result["peak_vtcm_bytes"]

    assessment = []

    # Transparent engineering heuristics.
    # These are project thresholds, not Qualcomm specifications.

    if latency < 1.0:
        latency_status = "EXCELLENT"
    elif latency < 5.0:
        latency_status = "GOOD"
    else:
        latency_status = "HIGH"

    if utilization >= 90.0:
        utilization_status = "VERY HIGH"
    elif utilization >= 70.0:
        utilization_status = "HIGH"
    elif utilization >= 40.0:
        utilization_status = "MODERATE"
    else:
        utilization_status = "LOW"

    if throughput >= 1000:
        throughput_status = "EXCELLENT"
    elif throughput >= 200:
        throughput_status = "GOOD"
    else:
        throughput_status = "LOW"

    assessment.append(
        f"Latency: {latency_status} ({latency:.3f} ms)"
    )

    assessment.append(
        f"HTP utilization: {utilization_status} "
        f"({utilization:.2f}%)"
    )

    assessment.append(
        f"Throughput: {throughput_status} "
        f"({throughput:.2f} inf/s)"
    )

    assessment.append(
        f"Peak VTCM: {peak_vtcm / (1024 ** 2):.2f} MiB"
    )

    if latency < 1.0 and utilization >= 90.0:
        overall = "NPU-EFFICIENT"
        recommendation = (
            "INT8 deployment is performing efficiently on the "
            "Snapdragon NPU. Retain the current optimization "
            "unless accuracy or memory constraints require "
            "further tuning."
        )
    elif latency < 5.0:
        overall = "PERFORMANCE-ACCEPTABLE"
        recommendation = (
            "Deployment performance is acceptable. Further "
            "optimization may target latency or memory if "
            "required by the application."
        )
    else:
        overall = "OPTIMIZATION-RECOMMENDED"
        recommendation = (
            "Consider additional model optimization, "
            "quantization, operator optimization, or model "
            "architecture changes."
        )

    return {
        "overall": overall,
        "assessment": assessment,
        "recommendation": recommendation,
    }


if __name__ == "__main__":
    from snapdragon_executor import run_snapdragon_inference

    result = run_snapdragon_inference()
    analysis = analyze_performance(result)

    print()
    print("=" * 70)
    print("ENGINEERING PERFORMANCE ANALYZER")
    print("=" * 70)

    for item in analysis["assessment"]:
        print(item)

    print()
    print(f"Overall       : {analysis['overall']}")
    print(f"Recommendation: {analysis['recommendation']}")
    print("=" * 70)
