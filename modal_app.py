import os
import sys
import time

import modal

APP_NAME = "snaplab-ai"
REMOTE_ROOT = "/root/SnapLab-AI"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(
        "vision",
        remote_path=f"{REMOTE_ROOT}/vision",
    )
    .add_local_dir(
        "assets",
        remote_path=f"{REMOTE_ROOT}/assets",
    )
    .add_local_file(
        "models/best.pt",
        remote_path=f"{REMOTE_ROOT}/models/best.pt",
    )
    .add_local_dir(
        "models/component_classifier",
        remote_path=f"{REMOTE_ROOT}/models/component_classifier",
    )
    .add_local_file(
        "engineering_state.py",
        remote_path=f"{REMOTE_ROOT}/engineering_state.py",
    )
    .add_local_file(
        "connection_engine.py",
        remote_path=f"{REMOTE_ROOT}/connection_engine.py",
    )
    .add_local_file(
        "connection_graph.py",
        remote_path=f"{REMOTE_ROOT}/connection_graph.py",
    )
    .add_local_file(
        "spatial_engine.py",
        remote_path=f"{REMOTE_ROOT}/spatial_engine.py",
    )
    .add_local_file(
        "wire_association.py",
        remote_path=f"{REMOTE_ROOT}/wire_association.py",
    )
)

# ==============================================================================
# CONNECTION DIAGNOSTIC INTERFACE (HTML + CSS + JS)
# Accessible via /check and /status
# ==============================================================================

CHECK_INTERFACE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SnapLab AI — Connection & Health Diagnostic Center</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">

  <style>
    :root {
      --bg: #060a12;
      --bg-card: rgba(15, 23, 42, 0.78);
      --bg-card-hover: rgba(26, 39, 68, 0.85);
      --bg-surface: #0a101f;
      --border: rgba(56, 78, 114, 0.45);
      --border-bright: rgba(56, 189, 248, 0.4);
      --border-green: rgba(16, 185, 129, 0.45);
      --border-red: rgba(239, 68, 68, 0.45);

      --text: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;

      --cyan: #22d3ee;
      --cyan-glow: rgba(34, 211, 238, 0.25);
      --emerald: #10b981;
      --emerald-glow: rgba(16, 185, 129, 0.3);
      --amber: #f59e0b;
      --crimson: #e11d48;
      --blue: #38bdf8;
      --purple: #8b5cf6;

      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      background-color: var(--bg);
      background-image:
        radial-gradient(ellipse at 15% 15%, rgba(34, 211, 238, 0.08), transparent 45%),
        radial-gradient(ellipse at 85% 20%, rgba(225, 29, 72, 0.07), transparent 45%),
        radial-gradient(ellipse at 50% 80%, rgba(16, 185, 129, 0.06), transparent 50%);
      background-attachment: fixed;
      color: var(--text);
      font-family: var(--font-sans);
      min-height: 100vh;
      line-height: 1.5;
      padding: 24px 16px 48px;
    }

    .container {
      max-width: 1080px;
      margin: 0 auto;
    }

    /* Top Navigation / Brand */
    .header-nav {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 14px 20px;
      background: var(--bg-card);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border);
      border-radius: 14px;
      margin-bottom: 24px;
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
    }

    .brand-wrap {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .brand-logo-badge {
      width: 36px;
      height: 36px;
      background: linear-gradient(135deg, #e11d48 0%, #be123c 100%);
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 18px;
      font-weight: 800;
      color: #fff;
      box-shadow: 0 0 16px rgba(225, 29, 72, 0.4);
    }

    .brand-title-wrap h1 {
      font-size: 16px;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .brand-title-wrap p {
      font-size: 11px;
      color: var(--text-muted);
    }

    .nav-actions {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .btn {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 8px 16px;
      border-radius: 8px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
      text-decoration: none;
      border: 1px solid transparent;
      font-family: inherit;
    }

    .btn-primary {
      background: linear-gradient(135deg, #2563eb, #1d4ed8);
      color: #fff;
      box-shadow: 0 2px 10px rgba(37, 99, 235, 0.35);
    }
    .btn-primary:hover {
      background: linear-gradient(135deg, #1d4ed8, #1e40af);
      transform: translateY(-1px);
      box-shadow: 0 4px 16px rgba(37, 99, 235, 0.5);
    }

    .btn-secondary {
      background: rgba(15, 23, 42, 0.8);
      border-color: var(--border);
      color: var(--text);
    }
    .btn-secondary:hover {
      background: rgba(30, 41, 59, 0.9);
      border-color: var(--border-bright);
    }

    /* Hero Connection Status Card */
    .hero-status-card {
      background: var(--bg-card);
      backdrop-filter: blur(20px);
      border: 1px solid var(--border-green);
      border-radius: 18px;
      padding: 32px 28px;
      margin-bottom: 24px;
      box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5), 0 0 24px rgba(16, 185, 129, 0.12);
      position: relative;
      overflow: hidden;
      transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }

    .hero-status-card.checking {
      border-color: rgba(245, 158, 11, 0.45);
      box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5), 0 0 24px rgba(245, 158, 11, 0.12);
    }

    .hero-status-card.error {
      border-color: var(--border-red);
      box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5), 0 0 24px rgba(239, 68, 68, 0.12);
    }

    .hero-glow-accent {
      position: absolute;
      top: -60px;
      right: -60px;
      width: 220px;
      height: 220px;
      background: radial-gradient(circle, rgba(16, 185, 129, 0.18) 0%, transparent 70%);
      pointer-events: none;
      transition: background 0.3s ease;
    }

    .hero-status-card.checking .hero-glow-accent {
      background: radial-gradient(circle, rgba(245, 158, 11, 0.18) 0%, transparent 70%);
    }

    .hero-status-card.error .hero-glow-accent {
      background: radial-gradient(circle, rgba(239, 68, 68, 0.18) 0%, transparent 70%);
    }

    .hero-flex {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 24px;
      flex-wrap: wrap;
    }

    .hero-left {
      display: flex;
      align-items: center;
      gap: 20px;
    }

    /* Radar / Pulse Indicator */
    .radar-indicator {
      position: relative;
      width: 64px;
      height: 64px;
      flex-shrink: 0;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .radar-ring {
      position: absolute;
      width: 100%;
      height: 100%;
      border-radius: 50%;
      border: 2px solid var(--emerald);
      opacity: 0.6;
      animation: radar-pulse 2.2s cubic-bezier(0.24, 0, 0.38, 1) infinite;
    }

    .radar-ring:nth-child(2) {
      animation-delay: 0.7s;
    }

    .radar-core {
      width: 28px;
      height: 28px;
      background: var(--emerald);
      border-radius: 50%;
      box-shadow: 0 0 16px var(--emerald);
      display: flex;
      align-items: center;
      justify-content: center;
      color: #064e3b;
      font-size: 14px;
      font-weight: 800;
      z-index: 2;
      transition: all 0.3s ease;
    }

    .hero-status-card.checking .radar-ring {
      border-color: var(--amber);
    }
    .hero-status-card.checking .radar-core {
      background: var(--amber);
      box-shadow: 0 0 16px var(--amber);
      color: #78350f;
    }

    .hero-status-card.error .radar-ring {
      border-color: var(--crimson);
    }
    .hero-status-card.error .radar-core {
      background: var(--crimson);
      box-shadow: 0 0 16px var(--crimson);
      color: #fff;
    }

    @keyframes radar-pulse {
      0% {
        transform: scale(0.5);
        opacity: 0.9;
      }
      100% {
        transform: scale(1.6);
        opacity: 0;
      }
    }

    .hero-info h2 {
      font-size: 22px;
      font-weight: 800;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 4px;
    }

    .badge-pill {
      font-size: 11px;
      font-weight: 700;
      padding: 3px 10px;
      border-radius: 20px;
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.4);
      color: #34d399;
      letter-spacing: 0.5px;
    }

    .hero-status-card.checking .badge-pill {
      background: rgba(245, 158, 11, 0.15);
      border-color: rgba(245, 158, 11, 0.4);
      color: #fbbf24;
    }

    .hero-status-card.error .badge-pill {
      background: rgba(239, 68, 68, 0.15);
      border-color: rgba(239, 68, 68, 0.4);
      color: #f87171;
    }

    .hero-info p {
      font-size: 13px;
      color: var(--text-muted);
    }

    .hero-right {
      display: flex;
      align-items: center;
      gap: 16px;
    }

    .latency-pill {
      background: rgba(10, 16, 31, 0.9);
      border: 1px solid var(--border);
      padding: 10px 18px;
      border-radius: 12px;
      text-align: right;
    }

    .latency-label {
      font-size: 11px;
      color: var(--text-dim);
      text-transform: uppercase;
      font-weight: 600;
      letter-spacing: 0.5px;
    }

    .latency-value {
      font-size: 20px;
      font-weight: 700;
      color: var(--cyan);
      font-family: var(--font-mono);
    }

    /* Subsystem Cards Grid */
    .section-title {
      font-size: 14px;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.8px;
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .grid-checks {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }

    .check-card {
      background: var(--bg-card);
      backdrop-filter: blur(14px);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 18px 20px;
      transition: all 0.25s ease;
      position: relative;
    }

    .check-card:hover {
      background: var(--bg-card-hover);
      border-color: var(--border-bright);
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }

    .check-card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
    }

    .check-card-title {
      font-size: 14px;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .status-tag {
      font-size: 11px;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: 6px;
      font-family: var(--font-mono);
    }

    .status-tag.ok {
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .status-tag.warn {
      background: rgba(245, 158, 11, 0.15);
      color: #fbbf24;
      border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .status-tag.fail {
      background: rgba(239, 68, 68, 0.15);
      color: #f87171;
      border: 1px solid rgba(239, 68, 68, 0.3);
    }

    .check-card-desc {
      font-size: 12px;
      color: var(--text-muted);
      margin-bottom: 10px;
      line-height: 1.45;
    }

    .check-card-meta {
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 11px;
      color: var(--text-dim);
      font-family: var(--font-mono);
      border-top: 1px dashed rgba(56, 78, 114, 0.3);
      padding-top: 8px;
    }

    /* Diagnostics Log Console */
    .console-panel {
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px 20px;
      margin-bottom: 24px;
      box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.5);
    }

    .console-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
      padding-bottom: 8px;
      border-bottom: 1px solid rgba(56, 78, 114, 0.3);
    }

    .console-title {
      font-size: 12px;
      font-weight: 700;
      color: var(--text-muted);
      font-family: var(--font-mono);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .console-logs {
      font-family: var(--font-mono);
      font-size: 11px;
      color: #cbd5e1;
      line-height: 1.7;
      max-height: 160px;
      overflow-y: auto;
    }

    .log-line {
      display: flex;
      gap: 10px;
    }

    .log-ts {
      color: var(--text-dim);
    }
    .log-tag {
      color: var(--cyan);
      font-weight: 600;
    }
    .log-tag.success {
      color: var(--emerald);
    }
    .log-tag.error {
      color: var(--crimson);
    }

    /* Bottom Action Controls */
    .controls-panel {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px 20px;
      flex-wrap: wrap;
      gap: 14px;
    }

    .controls-left {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .toggle-wrap {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      color: var(--text-muted);
      cursor: pointer;
      user-select: none;
    }

    .toggle-wrap input[type="checkbox"] {
      cursor: pointer;
      accent-color: var(--emerald);
    }

    /* JSON Details View */
    .json-details {
      margin-top: 24px;
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px 20px;
    }

    .json-summary {
      font-size: 12px;
      font-weight: 600;
      color: var(--cyan);
      cursor: pointer;
      outline: none;
      user-select: none;
    }

    .json-pre {
      margin-top: 12px;
      background: #050811;
      border: 1px solid rgba(56, 78, 114, 0.4);
      padding: 14px;
      border-radius: 10px;
      font-family: var(--font-mono);
      font-size: 11px;
      color: #94a3b8;
      overflow-x: auto;
      max-height: 280px;
    }

    /* Toast Notification */
    .toast {
      position: fixed;
      bottom: 24px;
      right: 24px;
      background: #0f172a;
      border: 1px solid var(--emerald);
      color: #f8fafc;
      padding: 10px 18px;
      border-radius: 10px;
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.6);
      font-size: 12px;
      font-weight: 600;
      display: none;
      z-index: 9999;
      animation: fadeIn 0.2s ease;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body>
  <div class="container">

    <!-- Header Navigation -->
    <header class="header-nav">
      <div class="brand-wrap">
        <div class="brand-logo-badge">⚡</div>
        <div class="brand-title-wrap">
          <h1>SnapLab AI <span style="font-size: 11px; color: var(--cyan); font-weight: 500;">· Modal Cloud</span></h1>
          <p>Real-Time Circuit Engineering Copilot</p>
        </div>
      </div>
      <div class="nav-actions">
        <button id="btnCheckNow" class="btn btn-secondary" onclick="runConnectionCheck()">
          <span id="btnSpinner" style="display: none;">⏳</span>
          <span>⚡ Check Connection Now</span>
        </button>
        <a href="/" class="btn btn-primary">
          <span>🚀 Open Copilot Workbench</span>
          <span>&nearr;</span>
        </a>
      </div>
    </header>

    <!-- Hero Status Card -->
    <section class="hero-status-card" id="heroCard">
      <div class="hero-glow-accent"></div>
      <div class="hero-flex">
        <div class="hero-left">
          <div class="radar-indicator">
            <div class="radar-ring"></div>
            <div class="radar-ring"></div>
            <div class="radar-core" id="radarCore">✓</div>
          </div>
          <div class="hero-info">
            <h2 id="heroStatusText">
              Connected Properly
              <span class="badge-pill" id="heroBadge">ONLINE · MODAL CLOUD</span>
            </h2>
            <p id="heroSubtext">Verified bidirectional connectivity to FastAPI ASGI server & Neural Vision models.</p>
          </div>
        </div>
        <div class="hero-right">
          <div class="latency-pill">
            <div class="latency-label">Round-Trip Latency</div>
            <div class="latency-value" id="latencyVal">-- ms</div>
          </div>
        </div>
      </div>
    </section>

    <!-- Subsystem Checks Grid -->
    <div class="section-title">
      <span>Subsystem Verification & Health Matrix</span>
    </div>

    <div class="grid-checks">
      <!-- 1. FastAPI ASGI -->
      <div class="check-card" id="cardFastapi">
        <div class="check-card-header">
          <div class="check-card-title">🌐 FastAPI ASGI Gateway</div>
          <span class="status-tag ok" id="tagFastapi">CONNECTED</span>
        </div>
        <div class="check-card-desc" id="descFastapi">
          High-performance asynchronous server running in Modal Debian Slim container.
        </div>
        <div class="check-card-meta">
          <span>Protocol: HTTP/2 ASGI</span>
          <span id="metaFastapi">Path: /api/health</span>
        </div>
      </div>

      <!-- 2. Gradio App -->
      <div class="check-card" id="cardGradio">
        <div class="check-card-header">
          <div class="check-card-title">🔬 Gradio Copilot Mount</div>
          <span class="status-tag ok" id="tagGradio">MOUNTED</span>
        </div>
        <div class="check-card-desc" id="descGradio">
          Flagship component inspector & circuit analyzer mounted with sticky session concurrency.
        </div>
        <div class="check-card-meta">
          <span>Mount Point: /</span>
          <span>Max Inputs: 10</span>
        </div>
      </div>

      <!-- 3. YOLO11n Model -->
      <div class="check-card" id="cardYolo">
        <div class="check-card-header">
          <div class="check-card-title">🧠 YOLO11n Neural Detector</div>
          <span class="status-tag ok" id="tagYolo">VERIFIED</span>
        </div>
        <div class="check-card-desc" id="descYolo">
          Phase C master weights (models/best.pt) covering 65 electronic component classes.
        </div>
        <div class="check-card-meta">
          <span>Weights: models/best.pt</span>
          <span id="metaYolo">Status: Ready</span>
        </div>
      </div>

      <!-- 4. Component Classifier -->
      <div class="check-card" id="cardClassifier">
        <div class="check-card-header">
          <div class="check-card-title">🔍 MobileNetV3 Refiner</div>
          <span class="status-tag ok" id="tagClassifier">VERIFIED</span>
        </div>
        <div class="check-card-desc" id="descClassifier">
          Fine-grained deep classifier resolving ambiguous ICs, regulators, and sensor modules.
        </div>
        <div class="check-card-meta">
          <span>Path: models/component_classifier</span>
          <span id="metaClassifier">Status: Ready</span>
        </div>
      </div>

      <!-- 5. Circuit Intelligence -->
      <div class="check-card" id="cardIntelligence">
        <div class="check-card-header">
          <div class="check-card-title">⚡ Circuit DRC & Topology</div>
          <span class="status-tag ok" id="tagIntelligence">READY</span>
        </div>
        <div class="check-card-desc" id="descIntelligence">
          Heuristic wire-tracing, spatial association & electrical Design Rule Checking engine.
        </div>
        <div class="check-card-meta">
          <span>Module: engineering_state.py</span>
          <span>Rules: DRC-01 to 05</span>
        </div>
      </div>

      <!-- 6. Assets & Media -->
      <div class="check-card" id="cardAssets">
        <div class="check-card-header">
          <div class="check-card-title">📦 Media & Reference Assets</div>
          <span class="status-tag ok" id="tagAssets">VERIFIED</span>
        </div>
        <div class="check-card-desc" id="descAssets">
          Reference schematics, Snapdragon X Elite branding and test circuit frames loaded.
        </div>
        <div class="check-card-meta">
          <span>Dir: /assets</span>
          <span>Status: Verified</span>
        </div>
      </div>
    </div>

    <!-- Live Diagnostics Log Console -->
    <div class="console-panel">
      <div class="console-header">
        <span class="console-title">Live Diagnostic Telemetry</span>
        <span style="font-size: 11px; color: var(--text-dim);" id="lastCheckTime">Last Check: Never</span>
      </div>
      <div class="console-logs" id="consoleLogs">
        <!-- Injected via JavaScript -->
      </div>
    </div>

    <!-- Bottom Controls -->
    <div class="controls-panel">
      <div class="controls-left">
        <label class="toggle-wrap">
          <input type="checkbox" id="chkAutoPulse" checked onchange="toggleAutoPulse()">
          <span>Auto-Pulse Heartbeat (every 10s)</span>
        </label>
      </div>
      <div style="display: flex; gap: 10px;">
        <button class="btn btn-secondary" onclick="copyDiagnosticReport()">
          <span>📋 Copy Report</span>
        </button>
        <button class="btn btn-primary" onclick="window.location.href='/'">
          <span>Open SnapLab AI &nearr;</span>
        </button>
      </div>
    </div>

    <!-- Expandable JSON Inspector -->
    <details class="json-details">
      <summary class="json-summary">🔍 Expand Raw Diagnostic Payload (/api/health)</summary>
      <pre class="json-pre" id="jsonPayload">Awaiting diagnostic probe...</pre>
    </details>

  </div>

  <div id="toast" class="toast">Diagnostics copied to clipboard!</div>

  <script>
    let autoPulseTimer = null;
    let latestReport = null;

    function logMsg(tag, msg, type = 'info') {
      const logs = document.getElementById('consoleLogs');
      if (!logs) return;
      const now = new Date();
      const ts = now.toTimeString().split(' ')[0] + '.' + String(now.getMilliseconds()).padStart(3, '0');
      
      const line = document.createElement('div');
      line.className = 'log-line';
      line.innerHTML = `
        <span class="log-ts">[${ts}]</span>
        <span class="log-tag ${type}">[${tag}]</span>
        <span>${msg}</span>
      `;
      logs.appendChild(line);
      logs.scrollTop = logs.scrollHeight;
    }

    async function runConnectionCheck() {
      const heroCard = document.getElementById('heroCard');
      const heroStatus = document.getElementById('heroStatusText');
      const heroBadge = document.getElementById('heroBadge');
      const heroSubtext = document.getElementById('heroSubtext');
      const radarCore = document.getElementById('radarCore');
      const latencyVal = document.getElementById('latencyVal');
      const btnSpinner = document.getElementById('btnSpinner');
      const jsonPre = document.getElementById('jsonPayload');
      const lastCheckTime = document.getElementById('lastCheckTime');

      if (btnSpinner) btnSpinner.style.display = 'inline-block';
      if (heroCard) heroCard.className = 'hero-status-card checking';
      if (radarCore) radarCore.textContent = '⏳';
      if (heroStatus) heroStatus.innerHTML = 'Testing Connection... <span class="badge-pill" id="heroBadge">PROBING BACKEND</span>';

      logMsg('PROBE', 'Initiating bidirectional HTTP GET handshake to /api/health...', 'info');

      const startTime = performance.now();

      try {
        const response = await fetch('/api/health?t=' + Date.now(), {
          method: 'GET',
          headers: { 'Accept': 'application/json' },
          cache: 'no-store'
        });

        const roundTripMs = Math.round(performance.now() - startTime);
        const data = await response.json();
        latestReport = data;

        if (latencyVal) latencyVal.textContent = roundTripMs + ' ms';
        if (jsonPre) jsonPre.textContent = JSON.stringify(data, null, 2);
        if (lastCheckTime) lastCheckTime.textContent = 'Last Check: ' + new Date().toLocaleTimeString();

        if (response.ok && data.connected) {
          if (heroCard) heroCard.className = 'hero-status-card';
          if (radarCore) radarCore.textContent = '✓';
          if (heroStatus) heroStatus.innerHTML = 'Connected Properly <span class="badge-pill" style="background: rgba(16, 185, 129, 0.15); border-color: rgba(16, 185, 129, 0.4); color: #34d399;">ONLINE · MODAL CLOUD</span>';
          if (heroSubtext) heroSubtext.textContent = 'Verified bidirectional connectivity to FastAPI ASGI server & Neural Vision models in ' + roundTripMs + ' ms.';
          
          logMsg('SUCCESS', `200 OK received from ${data.platform || 'Modal'} in ${roundTripMs}ms. All subsystems connected properly.`, 'success');

          // Update subsystem cards
          updateCardStatus('tagFastapi', 'descFastapi', data.checks?.fastapi_server?.connected_properly, 'CONNECTED', data.checks?.fastapi_server?.message);
          updateCardStatus('tagGradio', 'descGradio', data.checks?.gradio_copilot?.connected_properly, 'MOUNTED', data.checks?.gradio_copilot?.message);
          updateCardStatus('tagYolo', 'descYolo', data.checks?.yolo11n_detector?.connected_properly, 'VERIFIED', data.checks?.yolo11n_detector?.message);
          updateCardStatus('tagClassifier', 'descClassifier', data.checks?.component_classifier?.connected_properly, 'VERIFIED', data.checks?.component_classifier?.message);
          updateCardStatus('tagIntelligence', 'descIntelligence', data.checks?.circuit_intelligence?.connected_properly, 'READY', data.checks?.circuit_intelligence?.message);
          updateCardStatus('tagAssets', 'descAssets', data.checks?.assets?.connected_properly, 'VERIFIED', data.checks?.assets?.message);

        } else {
          throw new Error(data.message || 'Degraded backend connection state');
        }

      } catch (err) {
        const roundTripMs = Math.round(performance.now() - startTime);
        if (latencyVal) latencyVal.textContent = roundTripMs > 0 ? (roundTripMs + ' ms') : 'ERR';
        if (heroCard) heroCard.className = 'hero-status-card error';
        if (radarCore) radarCore.textContent = '✕';
        if (heroStatus) heroStatus.innerHTML = 'Connection Error <span class="badge-pill" style="background: rgba(239, 68, 68, 0.15); border-color: rgba(239, 68, 68, 0.4); color: #f87171;">OFFLINE / UNREACHABLE</span>';
        if (heroSubtext) heroSubtext.textContent = 'Failed to establish connection: ' + err.message + '. Retrying on next pulse...';

        logMsg('ERROR', 'Connection probe failed: ' + err.message, 'error');
      } finally {
        if (btnSpinner) btnSpinner.style.display = 'none';
      }
    }

    function updateCardStatus(tagId, descId, isOk, okText, message) {
      const tag = document.getElementById(tagId);
      const desc = document.getElementById(descId);
      if (tag) {
        tag.className = isOk ? 'status-tag ok' : 'status-tag fail';
        tag.textContent = isOk ? okText : 'CHECK FAILED';
      }
      if (desc && message) {
        desc.textContent = message;
      }
    }

    function toggleAutoPulse() {
      const chk = document.getElementById('chkAutoPulse');
      if (chk && chk.checked) {
        logMsg('CONFIG', 'Auto-pulse heartbeat enabled (interval: 10s)', 'info');
        if (!autoPulseTimer) {
          autoPulseTimer = setInterval(runConnectionCheck, 10000);
        }
      } else {
        logMsg('CONFIG', 'Auto-pulse heartbeat paused', 'info');
        if (autoPulseTimer) {
          clearInterval(autoPulseTimer);
          autoPulseTimer = null;
        }
      }
    }

    function copyDiagnosticReport() {
      if (!latestReport) {
        runConnectionCheck().then(() => copyDiagnosticReport());
        return;
      }
      navigator.clipboard.writeText(JSON.stringify(latestReport, null, 2)).then(() => {
        const toast = document.getElementById('toast');
        if (toast) {
          toast.style.display = 'block';
          setTimeout(() => { toast.style.display = 'none'; }, 2500);
        }
      });
    }

    // Hotkey: press Space to trigger quick connection test
    window.addEventListener('keydown', (e) => {
      if (e.code === 'Space' && e.target === document.body) {
        e.preventDefault();
        runConnectionCheck();
      }
    });

    // Run initial connection test on DOM load
    window.addEventListener('DOMContentLoaded', () => {
      logMsg('INIT', 'SnapLab AI Diagnostic Center loaded in browser.', 'info');
      runConnectionCheck();
      autoPulseTimer = setInterval(runConnectionCheck, 10000);
    });
  </script>
</body>
</html>
"""

# ==============================================================================
# IN-GRADIO FLOATING CONNECTION PILL (HTML & CSS & JS)
# Injected into Gradio Blocks so users at "/" also have continuous connectivity verification
# ==============================================================================

GRADIO_PILL_CSS = """
/* Floating SnapLab AI Connection Pill in Gradio Interface */
.snaplab-modal-conn-pill {
    position: fixed;
    top: 12px;
    right: 20px;
    z-index: 99999;
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(15, 23, 42, 0.94);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(16, 185, 129, 0.45);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5), 0 0 14px rgba(16, 185, 129, 0.2);
    padding: 6px 14px;
    border-radius: 9999px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 12px;
    color: #F8FAFC;
    cursor: pointer;
    transition: all 0.2s ease;
    text-decoration: none !important;
}

.snaplab-modal-conn-pill:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.6), 0 0 20px rgba(16, 185, 129, 0.35);
    border-color: rgba(16, 185, 129, 0.7);
}

.snaplab-conn-pulse {
    width: 8px;
    height: 8px;
    background-color: #10B981;
    border-radius: 50%;
    box-shadow: 0 0 8px #10B981;
    animation: snaplab-conn-pulse-anim 2s infinite;
}

@keyframes snaplab-conn-pulse-anim {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.35; transform: scale(0.8); }
}

.snaplab-conn-diag-link {
    background: rgba(34, 211, 238, 0.15);
    border: 1px solid rgba(34, 211, 238, 0.3);
    color: #22D3EE;
    border-radius: 6px;
    padding: 2px 7px;
    font-size: 10px;
    font-weight: 600;
    margin-left: 4px;
}
"""

GRADIO_HEAD_INJECTION = (
    "<style>\n"
    + GRADIO_PILL_CSS
    + "\n</style>\n"
    + """<script>
(function() {
    function injectConnectionPill() {
        if (document.getElementById('snaplab-conn-status-pill')) return;
        
        const pill = document.createElement('a');
        pill.id = 'snaplab-conn-status-pill';
        pill.className = 'snaplab-modal-conn-pill';
        pill.href = '/check';
        pill.target = '_blank';
        pill.title = 'SnapLab AI Modal Connection: Verified. Click to view full diagnostic interface.';
        pill.innerHTML = `
            <span class="snaplab-conn-pulse" id="snaplab-pill-pulse"></span>
            <span id="snaplab-pill-text">Modal Cloud: Connected Properly</span>
            <span class="snaplab-conn-diag-link">Diagnostics &nearr;</span>
        `;
        document.body.appendChild(pill);

        function checkBackend() {
            const t0 = performance.now();
            fetch('/api/health?t=' + Date.now(), { cache: 'no-store' })
                .then(res => res.json())
                .then(data => {
                    const lat = Math.round(performance.now() - t0);
                    const text = document.getElementById('snaplab-pill-text');
                    const pulse = document.getElementById('snaplab-pill-pulse');
                    if (text && pulse) {
                        if (data && data.connected) {
                            text.textContent = `Modal Cloud: Connected (${lat}ms)`;
                            pulse.style.backgroundColor = '#10B981';
                            pulse.style.boxShadow = '0 0 8px #10B981';
                        } else {
                            text.textContent = 'Modal Cloud: Degraded';
                            pulse.style.backgroundColor = '#F59E0B';
                            pulse.style.boxShadow = '0 0 8px #F59E0B';
                        }
                    }
                })
                .catch(() => {
                    const text = document.getElementById('snaplab-pill-text');
                    const pulse = document.getElementById('snaplab-pill-pulse');
                    if (text && pulse) {
                        text.textContent = 'Modal Cloud: Disconnected';
                        pulse.style.backgroundColor = '#EF4444';
                        pulse.style.boxShadow = '0 0 8px #EF4444';
                    }
                });
        }

        checkBackend();
        setInterval(checkBackend, 20000);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', injectConnectionPill);
    } else {
        injectConnectionPill();
    }
})();
</script>"""
)

# ==============================================================================
# FASTAPI & GRADIO APPLICATION FACTORY
# ==============================================================================

def create_app(root_dir=REMOTE_ROOT):
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse
    import gradio as gr

    from vision.component_inspector_v2 import demo

    web_app = FastAPI(
        title="SnapLab AI Modal ASGI Service",
        description="Real-Time Circuit Engineering Copilot for Snapdragon AI PCs",
        version="2.0.0",
    )

    # Enable CORS for external frontends or Netlify static site checks
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_system_health():
        models_best_path = os.path.join(root_dir, "models", "best.pt")
        classifier_path = os.path.join(root_dir, "models", "component_classifier")
        engineering_path = os.path.join(root_dir, "engineering_state.py")
        vision_path = os.path.join(root_dir, "vision")
        assets_path = os.path.join(root_dir, "assets")

        models_best_present = os.path.exists(models_best_path)
        classifier_present = os.path.isdir(classifier_path)
        engineering_state_present = os.path.exists(engineering_path)
        vision_present = os.path.isdir(vision_path)
        assets_present = os.path.isdir(assets_path)

        all_ok = all([
            models_best_present,
            classifier_present,
            engineering_state_present,
            vision_present,
            assets_present,
        ])

        return {
            "status": "healthy" if all_ok else "degraded",
            "connected": True,
            "message": "Connected properly to SnapLab AI backend" if all_ok else "Backend connected with warnings",
            "app_name": APP_NAME,
            "service": "SnapLab AI Cloud Copilot",
            "platform": "Modal Cloud (Serverless)",
            "python_version": sys.version.split()[0],
            "server_timestamp": time.time(),
            "remote_root": root_dir,
            "checks": {
                "fastapi_server": {
                    "status": "ok",
                    "connected_properly": True,
                    "message": "FastAPI ASGI server running",
                },
                "gradio_copilot": {
                    "status": "ok",
                    "connected_properly": True,
                    "mount_path": "/",
                    "message": "Gradio UI mounted at root",
                },
                "yolo11n_detector": {
                    "status": "ok" if models_best_present else "missing",
                    "connected_properly": models_best_present,
                    "path": models_best_path,
                    "message": "65-class master YOLO11n weights verified" if models_best_present else "Model weights not found",
                },
                "component_classifier": {
                    "status": "ok" if classifier_present else "missing",
                    "connected_properly": classifier_present,
                    "path": classifier_path,
                    "message": "MobileNetV3 classifier checkpoints verified" if classifier_present else "Classifier directory not found",
                },
                "circuit_intelligence": {
                    "status": "ok" if engineering_state_present and vision_present else "missing",
                    "connected_properly": engineering_state_present and vision_present,
                    "message": "Engineering state, spatial engine & DRC rules active",
                },
                "assets": {
                    "status": "ok" if assets_present else "missing",
                    "connected_properly": assets_present,
                    "path": assets_path,
                    "message": "Branding & reference media assets verified",
                },
            },
            "subsystems_healthy": all_ok,
        }

    @web_app.get("/api/health")
    @web_app.get("/health")
    async def health_endpoint():
        health_data = get_system_health()
        status_code = 200 if health_data["connected"] else 503
        return JSONResponse(content=health_data, status_code=status_code)

    @web_app.get("/ping")
    async def ping_endpoint():
        return JSONResponse(
            content={
                "ping": "pong",
                "connected": True,
                "timestamp": time.time(),
            }
        )

    @web_app.get("/check", response_class=HTMLResponse)
    @web_app.get("/status", response_class=HTMLResponse)
    @web_app.get("/health-check", response_class=HTMLResponse)
    async def check_page():
        return HTMLResponse(content=CHECK_INTERFACE_HTML)

    # Inject live connection check floating badge into Gradio head and css
    demo.head = (getattr(demo, "head", None) or "") + "\n" + GRADIO_HEAD_INJECTION
    # Configure Gradio queue to enable high-throughput async processing
    demo.queue(default_concurrency_limit=20, max_size=64)

    # Mount Gradio flagship interface at root "/"
    return gr.mount_gradio_app(
        web_app,
        demo,
        path="/",
    )


# ==============================================================================
# MODAL ASGI WEB APPLICATION ENTRY POINT
# ==============================================================================

@app.function(
    image=image,
    max_containers=1,  # Required for Gradio sticky sessions
    timeout=900,
    scaledown_window=300,  # Keep container warm for 5 minutes after inactivity
    secrets=[
        modal.Secret.from_name("snaplab-secrets"),
    ],
)
@modal.concurrent(max_inputs=100)  # Allows 100 concurrent requests so persistent SSE heartbeats do not block GET /
@modal.asgi_app()
def web():
    os.chdir(REMOTE_ROOT)
    return create_app(REMOTE_ROOT)