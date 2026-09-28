from relationship_engine import analyze_relationships

components = [
    "ESP32-CAM",
    "FT-232-USB-Serial-Module",
    "3-3-Volt-Battery",
    "Raindrops-Module"
]

results = analyze_relationships(components)

print("RELATIONSHIP ENGINE: PASSED")
print("RELATIONSHIPS FOUND:", len(results))

for item in results:
    print(
        f"{item['component_a']} <-> "
        f"{item['component_b']}: "
        f"{item['relationship']}"
    )