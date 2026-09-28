def build_engineering_prompt(components, question=None):
    lines = [
        "You are an engineering copilot analyzing an electronic circuit image.",
        "Identify visible components and reason about their likely roles and relationships.",
        "Do not claim an electrical connection unless the image provides sufficient visual evidence.",
        "Distinguish visual observations from engineering inferences.",
        ""
    ]

    if components:
        lines.append("Detected components:")
        for item in components:
            name = item.get("final_class", item.get("class_name", "Unknown"))
            confidence = item.get("final_confidence", item.get("confidence", 0.0))
            lines.append(f"- {name}: confidence={confidence:.3f}")
        lines.append("")

    if question:
        lines.append(f"Engineering question: {question}")
    else:
        lines.append("Analyze the circuit and explain its likely purpose, component relationships, and any visible issues.")

    return "\n".join(lines)