from performance_comparison import load_benchmarks, calculate_comparison


def generate_recommendation(data, comparison):
    accuracy = data["accuracy_consistency"]

    latency_reduction = comparison[
        "compiled_to_int8_latency_reduction_percent"
    ]

    memory_reduction = comparison[
        "fp32_to_int8_memory_reduction_percent"
    ]

    top1_agreement = accuracy["top1_agreement_percent"]
    cosine_similarity = accuracy["mean_cosine_similarity"]

    reasons = []
    warnings = []

    # Performance evidence
    if latency_reduction > 0:
        reasons.append(
            f"INT8 reduces compiled-FP32 latency by "
            f"{latency_reduction:.2f}%."
        )

    if memory_reduction > 0:
        reasons.append(
            f"INT8 reduces estimated peak memory by "
            f"{memory_reduction:.2f}%."
        )

    if comparison["int8_htp_utilization_percent"] >= 90:
        reasons.append(
            f"HTP utilization is very high at "
            f"{comparison['int8_htp_utilization_percent']:.2f}%."
        )

    # Accuracy / consistency evidence
    if top1_agreement >= 99 and cosine_similarity >= 0.98:
        reasons.append(
            f"Prediction consistency is strong: "
            f"{top1_agreement:.2f}% Top-1 agreement and "
            f"{cosine_similarity:.6f} mean cosine similarity."
        )
    else:
        warnings.append(
            "Prediction consistency does not meet the current "
            "project acceptance heuristic."
        )

    # Decision
    if (
        latency_reduction > 0
        and memory_reduction > 0
        and top1_agreement >= 99
        and cosine_similarity >= 0.98
    ):
        decision = "RECOMMEND INT8"
        confidence = "HIGH"

        action = (
            "Deploy the INT8 model on the Snapdragon NPU. "
            "Continue monitoring accuracy with labeled real-world "
            "validation data before production deployment."
        )

    elif latency_reduction > 0 and top1_agreement >= 95:
        decision = "CONSIDER INT8"
        confidence = "MEDIUM"

        action = (
            "INT8 provides performance benefits, but additional "
            "accuracy validation is recommended before deployment."
        )

    else:
        decision = "RETAIN FP32"
        confidence = "LOW"

        action = (
            "Do not switch to INT8 yet. Investigate accuracy, "
            "latency, or memory trade-offs before changing the "
            "deployment configuration."
        )

    return {
        "decision": decision,
        "confidence": confidence,
        "reasons": reasons,
        "warnings": warnings,
        "action": action,
    }


if __name__ == "__main__":
    data = load_benchmarks()
    comparison = calculate_comparison(data)

    recommendation = generate_recommendation(
        data,
        comparison
    )

    print("=" * 70)
    print("SNAPLAB-AI — ENGINEERING RECOMMENDATION ENGINE")
    print("=" * 70)

    print()
    print(f"Decision   : {recommendation['decision']}")
    print(f"Confidence : {recommendation['confidence']}")

    print()
    print("EVIDENCE")
    print("-" * 70)

    for reason in recommendation["reasons"]:
        print(f"✓ {reason}")

    if recommendation["warnings"]:
        print()
        print("WARNINGS")
        print("-" * 70)

        for warning in recommendation["warnings"]:
            print(f"! {warning}")

    print()
    print("RECOMMENDED ACTION")
    print("-" * 70)
    print(recommendation["action"])

    print()
    print("=" * 70)
    print("RECOMMENDATION COMPLETE")
    print("=" * 70)
