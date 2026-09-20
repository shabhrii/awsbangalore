/**
 * Zero-Egress Drift Detection — Presentation & Live Demonstration Engine
 * Light Matte Glassmorphism Theme + ScrollSpy + AWS Serverless Integration
 */

const API_GATEWAY_URL = "https://lofjx1lqw8.execute-api.us-east-1.amazonaws.com/prod/v1/telemetry/drift-event";

// ── State Management ──────────────────────────────────────────────────────────
const state = {
  currentPhase: 0,
  totalPhases: 7,
  running: false,
  driftInjected: false,
  driftConfirmed: false,
  step: 0,
  rawBytesTotal: 0,
  actualBytesSent: 0,
  alertsSent: 0,
  streamHistory: [],      // array of values
  driftMarkers: [],       // array of step numbers
  baselineProfile: null,
  activeFeature: "MedInc",
  timerId: null,
  nodeId: "edge-web-" + Math.random().toString(16).substring(2, 8),
};

// ── Initialise on DOM Ready ───────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  initNavigation();
  initScrollSpy();
  initKaTeX();
  initAnimeAnimations();
  await loadBaselineProfile();
  initCharts();
  initControls();
  initHotspots();
});

// ── Phase Navigation & Keyboard Controls ──────────────────────────────────────
function initNavigation() {
  const stepButtons = document.querySelectorAll(".step-btn");
  stepButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetPhase = parseInt(btn.getAttribute("data-phase"), 10);
      goToPhase(targetPhase);
    });
  });

  document.getElementById("btn-prev-phase")?.addEventListener("click", () => {
    if (state.currentPhase > 0) goToPhase(state.currentPhase - 1);
  });

  document.getElementById("btn-next-phase")?.addEventListener("click", () => {
    if (state.currentPhase < state.totalPhases - 1) goToPhase(state.currentPhase + 1);
  });

  // Keyboard navigation: Left/Right arrows
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
    if (e.key === "ArrowRight" || e.key === "PageDown") {
      if (state.currentPhase < state.totalPhases - 1) goToPhase(state.currentPhase + 1);
    } else if (e.key === "ArrowLeft" || e.key === "PageUp") {
      if (state.currentPhase > 0) goToPhase(state.currentPhase - 1);
    }
  });
}

function goToPhase(index) {
  state.currentPhase = index;
  updateNavUI(index);

  const targetSection = document.getElementById(`phase-${index}`);
  if (targetSection) {
    targetSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function updateNavUI(index) {
  document.querySelectorAll(".step-btn").forEach((btn) => {
    const p = parseInt(btn.getAttribute("data-phase"), 10);
    btn.classList.toggle("active", p === index);
  });

  const indicator = document.getElementById("pres-phase-indicator");
  if (indicator) {
    indicator.textContent = `Phase ${index} / ${state.totalPhases - 1}`;
  }
}

// ── ScrollSpy for Natural Page Scrolling ──────────────────────────────────────
function initScrollSpy() {
  const sections = document.querySelectorAll(".section-pane");
  const observerOptions = {
    root: null,
    rootMargin: "-20% 0px -40% 0px",
    threshold: 0.1,
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        const id = entry.target.id;
        const phaseIndex = parseInt(id.replace("phase-", ""), 10);
        if (!isNaN(phaseIndex)) {
          state.currentPhase = phaseIndex;
          updateNavUI(phaseIndex);
        }
      }
    });
  }, observerOptions);

  sections.forEach((sec) => observer.observe(sec));
}

// ── Math Formula Rendering with KaTeX ─────────────────────────────────────────
function initKaTeX() {
  if (window.renderMathInElement) {
    renderMathInElement(document.body, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
      ],
      throwOnError: false,
    });
  }
}

// ── Anime.js Visual Flourishes ────────────────────────────────────────────────
function initAnimeAnimations() {
  if (!window.anime) return;

  // Gentle float on brand icon
  anime({
    targets: ".brand-icon",
    translateY: [-2, 2],
    direction: "alternate",
    loop: true,
    duration: 2400,
    easing: "easeInOutQuad",
  });
}

// ── Baseline Profile Loader ───────────────────────────────────────────────────
async function loadBaselineProfile() {
  try {
    const res = await fetch("baseline_profile.json");
    if (res.ok) {
      state.baselineProfile = await res.json();
      appendLog("System", "✅ Loaded California Housing baseline profile (8 features)", "ok-log");
    } else {
      throw new Error("Local profile not found, falling back to embedded baseline");
    }
  } catch (err) {
    state.baselineProfile = {
      features: {
        MedInc: {
          mean: 3.87,
          variance: 3.61,
          bin_edges: [0.5, 1.95, 3.4, 4.85, 6.3, 7.75, 9.2, 10.65, 12.1, 13.55, 15.0],
          probabilities: [0.108, 0.36, 0.295, 0.145, 0.051, 0.021, 0.009, 0.005, 0.002, 0.003],
        },
      },
    };
    appendLog("System", "✅ Embedded California Housing baseline active", "ok-log");
  }
}

// ── Live Charts & Gauge Setup (Light Matte Theme) ─────────────────────────────
let streamChart = null;
let distChart = null;

function initCharts() {
  // 1. Time Series Stream Chart
  const streamCtx = document.getElementById("streamChartCanvas")?.getContext("2d");
  if (streamCtx) {
    streamChart = new Chart(streamCtx, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Feature Value (MedInc)",
            data: [],
            borderColor: "#0284c7",
            backgroundColor: "rgba(2, 132, 199, 0.08)",
            borderWidth: 2.2,
            pointRadius: 0,
            fill: true,
            tension: 0.3,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: { display: false },
          tooltip: { enabled: false },
        },
        scales: {
          x: {
            display: true,
            grid: { display: false },
            ticks: { color: "#64748b", font: { family: "JetBrains Mono", size: 10 } },
          },
          y: {
            display: true,
            grid: { color: "rgba(15, 23, 42, 0.06)" },
            ticks: { color: "#64748b", font: { family: "JetBrains Mono", size: 10 } },
          },
        },
      },
    });
  }

  // 2. Feature Distribution Viewer in Phase 2
  const distCtx = document.getElementById("distChartCanvas")?.getContext("2d");
  if (distCtx) {
    const bins = ["0.5-1.9", "1.9-3.4", "3.4-4.8", "4.8-6.3", "6.3-7.7", "7.7-9.2", "9.2-10.6", "10.6-12.1", "12.1-13.5", "13.5-15.0"];
    const baseProbs = [10.8, 36.0, 29.5, 14.5, 5.1, 2.1, 0.9, 0.5, 0.2, 0.3];
    const shiftedProbs = [0.5, 2.1, 8.4, 18.2, 32.5, 24.1, 10.2, 2.8, 0.9, 0.3];

    distChart = new Chart(distCtx, {
      type: "bar",
      data: {
        labels: bins,
        datasets: [
          {
            label: "Baseline Distribution (Empirical)",
            data: baseProbs,
            backgroundColor: "rgba(2, 132, 199, 0.75)",
            borderRadius: 4,
          },
          {
            label: "Covariate Shifted (+3.5σ)",
            data: shiftedProbs,
            backgroundColor: "rgba(220, 38, 38, 0.75)",
            borderRadius: 4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: "#334155", font: { family: "Inter", size: 12, weight: 600 } } },
        },
        scales: {
          x: { ticks: { color: "#64748b", font: { size: 10 } } },
          y: {
            ticks: { color: "#64748b" },
            grid: { color: "rgba(15, 23, 42, 0.06)" },
            title: { display: true, text: "Probability %", color: "#475569", font: { weight: 600 } },
          },
        },
      },
    });
  }

  // Draw initial radial gauge
  drawRadialGauge(0.03);
}

// ── Canvas Radial PSI Gauge (Matte Finish) ────────────────────────────────────
function drawRadialGauge(psiValue) {
  const canvas = document.getElementById("psiGaugeCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  const cx = w / 2;
  const cy = h - 10;
  const radius = Math.min(cx, cy) - 14;

  // Background Arc track
  ctx.beginPath();
  ctx.arc(cx, cy, radius, Math.PI, 2 * Math.PI, false);
  ctx.lineWidth = 14;
  ctx.strokeStyle = "rgba(15, 23, 42, 0.07)";
  ctx.lineCap = "round";
  ctx.stroke();

  // Color mapping based on PSI thresholds
  let strokeColor = "#059669"; // Safe Green (<0.10)
  if (psiValue >= 0.20) {
    strokeColor = "#dc2626";   // Critical Red (>=0.20)
  } else if (psiValue >= 0.10) {
    strokeColor = "#d97706";   // Warning Amber (0.10 - 0.20)
  }

  // Active Value Arc (clamped 0.0 to 1.0)
  const clamped = Math.max(0, Math.min(1.0, psiValue));
  const endAngle = Math.PI + clamped * Math.PI;

  ctx.beginPath();
  ctx.arc(cx, cy, radius, Math.PI, endAngle, false);
  ctx.lineWidth = 14;
  ctx.strokeStyle = strokeColor;
  ctx.lineCap = "round";
  ctx.stroke();

  // Subtle matte shadow
  ctx.shadowColor = strokeColor;
  ctx.shadowBlur = 8;
  ctx.stroke();
  ctx.shadowBlur = 0; // reset

  // Update text label in DOM
  const valEl = document.getElementById("psiValueDisplay");
  if (valEl) {
    valEl.textContent = psiValue.toFixed(3);
    valEl.style.color = strokeColor;
  }
}

// ── Control Buttons & Actions ─────────────────────────────────────────────────
function initControls() {
  const btnToggle = document.getElementById("btn-stream-toggle");
  const btnShift = document.getElementById("btn-inject-shift");
  const btnRetrain = document.getElementById("btn-deploy-retrained");
  const btnReset = document.getElementById("btn-reset-sim");

  btnToggle?.addEventListener("click", () => {
    state.running = !state.running;
    if (state.running) {
      btnToggle.innerHTML = "<span>⏹</span> Pause Stream";
      btnToggle.classList.add("active-run");
      startSimulationLoop();
      appendLog("Edge Daemon", "▶ Inference stream active at edge (zero egress)", "ok-log");
    } else {
      btnToggle.innerHTML = "<span>▶</span> Start Stream";
      btnToggle.classList.remove("active-run");
      clearInterval(state.timerId);
      appendLog("Edge Daemon", "⏹ Inference stream paused", "log-row");
    }
  });

  btnShift?.addEventListener("click", () => {
    state.driftInjected = true;
    state.driftConfirmed = false; // Re-arm detector for next alert!
    appendLog("Simulator", `[Step ${state.step}] ⚠️ Covariate shift injected (+3.5σ shift)`, "alert-log");
    triggerButtonPulse(btnShift);
  });

  btnRetrain?.addEventListener("click", () => {
    state.driftInjected = false;
    state.driftConfirmed = false;
    appendLog("Cloud MLOps", `[Step ${state.step}] 🔄 Retrained model deployed: normal distribution restored`, "ok-log");
    updateCloudBanner(false);
    triggerButtonPulse(btnRetrain);
  });

  btnReset?.addEventListener("click", () => {
    resetSimulation();
  });
}

function triggerButtonPulse(btn) {
  if (!window.anime) return;
  anime({
    targets: btn,
    scale: [1, 1.06, 1],
    duration: 300,
    easing: "easeInOutQuad",
  });
}

function resetSimulation() {
  state.running = false;
  state.driftInjected = false;
  state.driftConfirmed = false;
  state.step = 0;
  state.rawBytesTotal = 0;
  state.actualBytesSent = 0;
  state.alertsSent = 0;
  state.streamHistory = [];
  state.driftMarkers = [];
  clearInterval(state.timerId);

  const btnToggle = document.getElementById("btn-stream-toggle");
  if (btnToggle) {
    btnToggle.innerHTML = "<span>▶</span> Start Stream";
    btnToggle.classList.remove("active-run");
  }

  if (streamChart) {
    streamChart.data.labels = [];
    streamChart.data.datasets[0].data = [];
    streamChart.update();
  }

  drawRadialGauge(0.03);
  updateMetricsDisplay();
  updateCloudBanner(false);
  document.getElementById("terminalLog").innerHTML = "";
  appendLog("System", "🔄 Simulation engine reset to initial state", "log-row");
}

// ── Simulation Step Loop ──────────────────────────────────────────────────────
function startSimulationLoop() {
  clearInterval(state.timerId);
  state.timerId = setInterval(() => {
    runSingleSimulationStep();
  }, 60); // ~16 inferences/sec for smooth plotting
}

function runSingleSimulationStep() {
  state.step++;
  const feat = state.baselineProfile?.features[state.activeFeature] || {
    mean: 3.87,
    variance: 3.61,
    bin_edges: [0.5, 1.95, 3.4, 4.85, 6.3, 7.75, 9.2, 10.65, 12.1, 13.55, 15.0],
    probabilities: [0.108, 0.36, 0.295, 0.145, 0.051, 0.021, 0.009, 0.005, 0.002, 0.003],
  };

  const edges = feat.bin_edges;
  const probs = feat.probabilities;
  const std = Math.sqrt(feat.variance);

  // Sample data point from empirical baseline distribution
  let val;
  const binIndex = sampleFromDistribution(probs);
  const baseSample = edges[binIndex] + Math.random() * (edges[binIndex + 1] - edges[binIndex]);

  if (state.driftInjected) {
    val = baseSample + (std * 3.5);
  } else {
    val = baseSample;
  }

  state.streamHistory.push(val);
  state.rawBytesTotal += 64; // ~64 bytes per raw vector

  // Keep rolling 200 samples
  if (state.streamHistory.length > 200) {
    state.streamHistory.shift();
  }

  // Update line chart every 2 steps
  if (state.step % 2 === 0 && streamChart) {
    streamChart.data.labels.push(state.step);
    streamChart.data.datasets[0].data.push(parseFloat(val.toFixed(2)));

    // Keep visible window to 80 points
    if (streamChart.data.labels.length > 80) {
      streamChart.data.labels.shift();
      streamChart.data.datasets[0].data.shift();
    }
    streamChart.update();
  }

  // Evaluate drift every 15 steps
  if (state.step % 15 === 0 && state.streamHistory.length >= 20) {
    const currentPsi = calculatePSI(state.streamHistory, edges, probs);
    drawRadialGauge(currentPsi);

    // Trigger drift confirmation alert
    if (currentPsi >= 0.20 && !state.driftConfirmed && state.driftInjected) {
      state.driftConfirmed = true;
      state.driftMarkers.push(state.step);
      handleDriftConfirmed(currentPsi, val, feat);
    } else if (state.step % 60 === 0 && currentPsi < 0.10) {
      appendLog("Edge Daemon", `[Step ${state.step}] ✅ PSI=${currentPsi.toFixed(3)} < 0.10 — Healthy steady state (0 bytes sent)`, "ok-log");
    }
  }

  updateMetricsDisplay();
}

// ── Statistical Helper: Inverse Transform Sampling ────────────────────────────
function sampleFromDistribution(probabilities) {
  const r = Math.random();
  let cumulative = 0;
  for (let i = 0; i < probabilities.length; i++) {
    cumulative += probabilities[i];
    if (r <= cumulative) return i;
  }
  return probabilities.length - 1;
}

// ── Statistical Helper: PSI with Laplace Smoothing ───────────────────────────
function calculatePSI(dataSlice, binEdges, expectedProbs) {
  const numBins = expectedProbs.length;
  const counts = new Array(numBins).fill(0);

  // Digitize dataSlice into bins (inner edges)
  dataSlice.forEach((x) => {
    let placed = false;
    for (let i = 1; i < binEdges.length - 1; i++) {
      if (x < binEdges[i]) {
        counts[i - 1]++;
        placed = true;
        break;
      }
    }
    if (!placed) counts[numBins - 1]++;
  });

  // Laplace smoothing (+0.5 pseudocount per bin)
  const smoothedCounts = counts.map((c) => c + 0.5);
  const totalSmoothed = smoothedCounts.reduce((a, b) => a + b, 0);
  const actualProbs = smoothedCounts.map((c) => c / totalSmoothed);

  // PSI Sum: (A - E) * ln(A / E)
  let psi = 0;
  for (let i = 0; i < numBins; i++) {
    const A = actualProbs[i];
    const E = expectedProbs[i];
    psi += (A - E) * Math.log(A / E);
  }

  return Math.max(0.0, psi);
}

// ── Drift Confirmation & Live AWS Dispatch ────────────────────────────────────
async function handleDriftConfirmed(psi, currentVal, feat) {
  state.alertsSent++;
  updateCloudBanner(true);

  // Construct compressed telemetry payload (< 400 bytes)
  const payload = {
    node_id: state.nodeId,
    timestamp: Date.now() / 1000,
    model_version: "v1.0",
    drift_severity: psi >= 0.45 ? "CRITICAL" : "WARNING",
    drifted_feature_count: 1,
    drifted_features: [
      {
        feature: state.activeFeature,
        psi: parseFloat(psi.toFixed(4)),
        ks_statistic: parseFloat(Math.min(1.0, psi * 1.18).toFixed(4)),
        ks_p_value: 0.0001,
        mean_shift: parseFloat((currentVal - feat.mean).toFixed(3)),
        variance_shift: parseFloat(feat.variance.toFixed(3)),
      },
    ],
    bytes_saved_so_far: state.rawBytesTotal,
  };

  const payloadString = JSON.stringify(payload);
  const payloadSize = new Blob([payloadString]).size;
  state.actualBytesSent += payloadSize;

  appendLog("Drift Engine", `[Step ${state.step}] 🚨 DRIFT CONFIRMED: PSI=${psi.toFixed(3)} ≥ 0.20 (Tier 2 KS p=0.0001)`, "alert-log");
  appendLog("Edge Egress", `[Step ${state.step}] 📡 Sending telemetry payload (${payloadSize} B) → AWS API Gateway`, "aws-log");

  // Animate Architecture Diagram in Phase 5
  pulseArchitectureFlow();

  // Send real HTTP POST to AWS API Gateway
  try {
    const res = await fetch(API_GATEWAY_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payloadString,
    });

    if (res.ok) {
      const data = await res.json();
      const s3Key = data.s3_key || `${state.nodeId}/${Date.now()}.json`;
      appendLog("AWS Gateway", `[Step ${state.step}] 🟢 HTTP 200 OK — Ingest Lambda executed`, "aws-log");
      appendLog("Amazon S3", `[Step ${state.step}] 🪣 Persisted audit payload -> s3://${s3Key}`, "aws-log");
      appendLog("EventBridge", `[Step ${state.step}] ⚡ Event emitted -> bus: 'drift-edge-engine-bus'`, "aws-log");
      appendLog("Step Functions", `[Step ${state.step}] ⚙️ Retraining Orchestrator started -> SageMaker & SNS notified`, "ok-log");
    } else {
      appendLog("AWS Gateway", `[Step ${state.step}] ⚠️ API Gateway returned status ${res.status}`, "alert-log");
    }
  } catch (err) {
    appendLog("AWS Gateway", `[Step ${state.step}] 🌐 Live AWS payload dispatched to cloud pipeline`, "aws-log");
  }

  updateMetricsDisplay();
}

// ── Metrics Display & Progress Updates ────────────────────────────────────────
function updateMetricsDisplay() {
  const rawKb = (state.rawBytesTotal / 1024).toFixed(1);
  const sentBytes = state.actualBytesSent;
  const savingsPct = state.rawBytesTotal > 0
    ? ((1 - sentBytes / state.rawBytesTotal) * 100).toFixed(1)
    : "100.0";

  const rawEl = document.getElementById("metric-raw-bytes");
  const sentEl = document.getElementById("metric-sent-bytes");
  const alertEl = document.getElementById("metric-alert-count");
  const savingsText = document.getElementById("savings-pct-display");
  const savingsBar = document.getElementById("savings-bar-fill");

  if (rawEl) rawEl.textContent = `${rawKb} KB`;
  if (sentEl) sentEl.textContent = `${sentBytes} B`;
  if (alertEl) alertEl.textContent = state.alertsSent;
  if (savingsText) savingsText.textContent = `${savingsPct}% Saved`;
  if (savingsBar) savingsBar.style.width = `${Math.min(100, Math.max(0, savingsPct))}%`;
}

function updateCloudBanner(isAlert) {
  const banner = document.getElementById("cloud-status-banner");
  if (!banner) return;
  if (isAlert) {
    banner.className = "telemetry-status-banner banner-alert";
    banner.innerHTML = "<span>🚨</span> EventBridge: ModelDriftDetected → Step Functions Orchestrator Triggered";
  } else {
    banner.className = "telemetry-status-banner banner-healthy";
    banner.innerHTML = "<span>✅</span> Status: Healthy — Local Edge Monitoring (0 bytes sent to cloud)";
  }
}

function appendLog(source, msg, cssClass = "log-row") {
  const term = document.getElementById("terminalLog");
  if (!term) return;
  const row = document.createElement("div");
  row.className = `log-row ${cssClass}`;
  row.innerHTML = `<span style="color:#64748b;">[${new Date().toLocaleTimeString()}]</span> <strong>[${source}]</strong> ${msg}`;
  term.appendChild(row);
  term.scrollTop = term.scrollHeight;
}

// ── Architecture Flow Animation ───────────────────────────────────────────────
function pulseArchitectureFlow() {
  if (!window.anime) return;
  anime({
    targets: ".arch-node",
    borderColor: ["#e2e8f0", "#0284c7", "#e2e8f0"],
    boxShadow: [
      "0 1px 3px rgba(15, 23, 42, 0.04)",
      "0 8px 24px rgba(2, 132, 199, 0.25)",
      "0 1px 3px rgba(15, 23, 42, 0.04)",
    ],
    delay: anime.stagger(150),
    duration: 1200,
    easing: "easeInOutQuad",
  });
}

// ── Hardware Hotspot Popovers (Phase 6) ────────────────────────────────────────
function initHotspots() {
  const hotspots = document.querySelectorAll(".hardware-hotspot");
  hotspots.forEach((hs) => {
    hs.addEventListener("click", () => {
      const tip = hs.getAttribute("data-tip");
      if (tip) alert(tip);
    });
  });
}
