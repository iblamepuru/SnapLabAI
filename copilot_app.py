import gradio as gr
import json
from pathlib import Path

from copilot_reasoning import answer_query


REPORT_FILE = Path("engineering_report.json")


def load_report():
    with REPORT_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


report = load_report()

comparison = report["performance_comparison"]
accuracy = report["accuracy_consistency"]
recommendation = report["engineering_recommendation"]
benchmarks = report["benchmark_data"]

model_name = report["model"]
input_shape = report["input_shape"]

int8_latency = comparison["int8_graph_latency_ms"]
throughput = comparison["int8_throughput_inf_per_s"]
htp_utilization = comparison["int8_htp_utilization_percent"]
speedup = comparison["cpu_to_int8_speedup"]
memory_reduction = comparison["fp32_to_int8_memory_reduction_percent"]
top1_agreement = accuracy["top1_agreement_percent"]

decision = recommendation["decision"]
confidence = recommendation["confidence"]


def copilot_response(message):
    if not message or not message.strip():
        return "Please enter an engineering question."

    return answer_query(message)


CSS = """
body {
    background: #0b0f14;
}

.gradio-container {
    max-width: 1250px !important;
    margin: auto !important;
}

.header {
    padding: 24px;
    border-radius: 16px;
    margin-bottom: 18px;
    background: linear-gradient(135deg, #111827, #172033);
    border: 1px solid #263449;
}

.header h1 {
    margin: 0;
    font-size: 32px;
}

.header p {
    margin-top: 8px;
    opacity: 0.8;
}

.kpi {
    padding: 20px;
    border-radius: 14px;
    background: #151b25;
    border: 1px solid #283344;
    min-height: 105px;
}

.kpi-title {
    font-size: 13px;
    opacity: 0.65;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.kpi-value {
    font-size: 28px;
    font-weight: 700;
    margin-top: 8px;
}

.kpi-sub {
    font-size: 12px;
    opacity: 0.6;
    margin-top: 4px;
}

.recommendation {
    padding: 24px;
    border-radius: 16px;
    background: #151b25;
    border: 1px solid #283344;
    margin-top: 8px;
}

.recommendation h2 {
    margin-top: 0;
}

.decision {
    font-size: 30px;
    font-weight: 800;
}

.confidence {
    font-size: 15px;
    opacity: 0.75;
}

.note {
    font-size: 12px;
    opacity: 0.6;
    margin-top: 12px;
}
"""


def kpi(title, value, subtitle):
    return f"""
    <div class="kpi">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{subtitle}</div>
    </div>
    """


with gr.Blocks(
    title="SnapLab-AI — Engineering Copilot",
    css=CSS
) as app:

    gr.HTML(
        f"""
        <div class="header">
            <h1>🧠 SnapLab-AI — Engineering Copilot</h1>
            <p>
                Real-Time On-Device AI Performance Assistant for Snapdragon AI PCs
            </p>
        </div>
        """
    )

    gr.Markdown("## Deployment Status")

    with gr.Row():
        gr.HTML(kpi("MODEL", model_name, f"Input: {input_shape}"))
        gr.HTML(kpi("TARGET", "Snapdragon X Elite", "Qualcomm AI Hub / NPU"))
        gr.HTML(kpi("DEPLOYMENT", "INT8", "Recommended configuration"))

    gr.Markdown("## Performance")

    with gr.Row():
        gr.HTML(kpi("NPU LATENCY", f"{int8_latency:.3f} ms", "Graph execution"))
        gr.HTML(kpi("THROUGHPUT", f"{throughput:.2f}", "inferences / second"))
        gr.HTML(kpi("HTP UTILIZATION", f"{htp_utilization:.2f}%", "NPU / HTP"))
        
    with gr.Row():
        gr.HTML(kpi("CPU → INT8", f"{speedup:.2f}×", "Speedup vs CPU FP32"))
        gr.HTML(kpi("MEMORY", f"↓ {memory_reduction:.2f}%", "Estimated reduction"))
        gr.HTML(kpi("TOP-1 AGREEMENT", f"{top1_agreement:.2f}%", "10 synthetic samples"))

    gr.HTML(
        f"""
        <div class="recommendation">
            <h2>⚙️ Engineering Recommendation</h2>
            <div class="decision">{decision}</div>
            <div class="confidence">Confidence: {confidence}</div>
            <p>
                INT8 provides substantially lower latency and memory usage
                while maintaining strong prediction consistency in the
                current validation test.
            </p>
            <div class="note">
                ⚠ Top-1 agreement is based on synthetic consistency testing,
                not ImageNet classification accuracy. Labeled real-world
                validation is required before production deployment.
            </div>
        </div>
        """
    )

    gr.Markdown("## 💬 Ask the Engineering Copilot")

    question = gr.Textbox(
        label="Engineering Question",
        placeholder="e.g. Should I deploy the INT8 model?",
        lines=2
    )

    ask_button = gr.Button(
        "🔍 Analyze",
        variant="primary"
    )

    response = gr.Textbox(
        label="Copilot Analysis",
        lines=7,
        interactive=False
    )

    ask_button.click(
        fn=copilot_response,
        inputs=question,
        outputs=response
    )

    question.submit(
        fn=copilot_response,
        inputs=question,
        outputs=response
    )

    gr.Markdown(
        """
        ---
        **SnapLab-AI** · Qualcomm AI Hub · Snapdragon X Elite ·
        ResNet50 · INT8 NPU Optimization
        """
    )


if __name__ == "__main__":
    app.launch()
