"""
edge_daemon/daemon.py
---------------------
Zero-Egress Edge Statistical Drift Daemon.

Implements a two-tiered drift detection pipeline:
  Tier 1 — Population Stability Index (PSI)  ≥ 0.2  → escalate
  Tier 2 — Kolmogorov-Smirnov two-sample test p ≤ 0.05 → confirm & alert

Only a compressed JSON summary (<2 KB) is ever sent to the cloud; zero raw
rows or PII leave the edge device.
"""

import json
import os
import sys
import time
import uuid
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import requests
from scipy.stats import ks_2samp

# Ensure UTF-8 output on Windows (needed for emoji/unicode in print statements)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Configuration (override via environment variables for production)
# ---------------------------------------------------------------------------
PSI_THRESHOLD: float = float(os.getenv("PSI_THRESHOLD", "0.2"))
KS_P_VALUE_THRESHOLD: float = float(os.getenv("KS_P_THRESHOLD", "0.05"))
WINDOW_SIZE: int = int(os.getenv("WINDOW_SIZE", "200"))       # reduced for demo
EVAL_EVERY: int = int(os.getenv("EVAL_EVERY", "50"))          # evaluate every N inferences
API_GATEWAY_URL: str = os.getenv(
    "API_GATEWAY_URL",
    "https://lofjx1lqw8.execute-api.us-east-1.amazonaws.com/prod/v1/telemetry/drift-event",
)
NODE_ID: str = os.getenv("EDGE_NODE_ID", f"edge-{uuid.uuid4().hex[:8]}")


# ---------------------------------------------------------------------------
# Core Engine
# ---------------------------------------------------------------------------
class DriftDaemon:
    """
    Maintains a rolling window of inference feature vectors and periodically
    evaluates statistical drift against the stored baseline profile.
    """

    def __init__(
        self,
        baseline_path: str = "baseline_profile.json",
        window_size: int = WINDOW_SIZE,
        eval_every: int = EVAL_EVERY,
        api_url: str = API_GATEWAY_URL,
        node_id: str = NODE_ID,
        verbose: bool = True,
    ):
        if not os.path.exists(baseline_path):
            raise FileNotFoundError(
                f"Baseline profile not found: {baseline_path}. "
                "Run baseline_generator.py first."
            )
        with open(baseline_path, "r") as f:
            self.baseline: dict = json.load(f)

        self.feature_names: List[str] = list(self.baseline["features"].keys())
        self.window_size = window_size
        self.eval_every = eval_every
        self.api_url = api_url
        self.node_id = node_id
        self.verbose = verbose

        # One deque per feature — O(1) append + O(1) popleft
        self.buffer: Dict[str, deque] = {
            feat: deque(maxlen=window_size) for feat in self.feature_names
        }
        self.count: int = 0
        self.alerts_sent: int = 0
        self.bytes_saved: int = 0  # simulated raw bytes NOT sent

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ingest(self, features: Dict[str, float]) -> Optional[dict]:
        """
        Accept one inference-time feature vector.  Call this in your model
        serving loop immediately after prediction.

        Returns the alert payload dict if drift was detected, else None.
        """
        for name in self.feature_names:
            val = features.get(name)
            if val is not None:
                self.buffer[name].append(float(val))

        # Simulate bytes saved vs. streaming every raw record (~64 bytes/record)
        self.bytes_saved += 64

        self.count += 1
        if self.count % self.eval_every == 0 and self._buffer_full():
            return self._evaluate_drift()
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _buffer_full(self) -> bool:
        return len(self.buffer[self.feature_names[0]]) == self.window_size

    def _calculate_psi(
        self,
        expected_probs: np.ndarray,
        actual_data: np.ndarray,
        bin_edges: np.ndarray,
    ) -> float:
        """
        Population Stability Index.
        PSI < 0.1  → no shift
        PSI 0.1-0.2 → slight shift
        PSI ≥ 0.2  → significant shift (trigger Tier 2)
        """
        actual_counts, _ = np.histogram(actual_data, bins=bin_edges)
        actual_probs = actual_counts.astype(float)
        actual_probs = np.where(actual_probs == 0, 1e-4, actual_probs)
        actual_probs /= actual_probs.sum()

        psi = np.sum((actual_probs - expected_probs) * np.log(actual_probs / expected_probs))
        return float(psi)

    def _evaluate_drift(self) -> Optional[dict]:
        drifted: List[dict] = []

        for feat in self.feature_names:
            base_stats = self.baseline["features"][feat]
            actual_data = np.array(list(self.buffer[feat]))

            # ── Tier 1: PSI ────────────────────────────────────────────
            psi = self._calculate_psi(
                expected_probs=np.array(base_stats["probabilities"]),
                actual_data=actual_data,
                bin_edges=np.array(base_stats["bin_edges"]),
            )

            if psi < PSI_THRESHOLD:
                continue  # no drift — skip Tier 2

            # ── Tier 2: KS-Test (two-sample) ───────────────────────────
            baseline_sample = np.array(base_stats["baseline_sample"])
            ks_stat, p_value = ks_2samp(baseline_sample, actual_data)

            if p_value > KS_P_VALUE_THRESHOLD:
                continue  # PSI triggered but KS-Test did not confirm

            mean_shift = float(np.mean(actual_data)) - base_stats["mean"]
            var_shift = float(np.var(actual_data)) - base_stats["variance"]

            drifted.append(
                {
                    "feature": feat,
                    "psi": round(psi, 4),
                    "ks_statistic": round(float(ks_stat), 4),
                    "ks_p_value": round(float(p_value), 6),
                    "mean_shift": round(mean_shift, 4),
                    "variance_shift": round(var_shift, 4),
                }
            )

            if self.verbose:
                print(
                    f"  [DRIFT] {feat}: PSI={psi:.3f} KS_stat={ks_stat:.3f} p={p_value:.5f}"
                )

        if drifted:
            if self.verbose:
                print(f"\n🚨 Drift confirmed in {len(drifted)} feature(s).")
            return self._send_alert(drifted)

        if self.verbose:
            print(f"  [OK] Inference #{self.count}: no drift detected.")
        return None

    def _send_alert(self, drifted_features: List[dict]) -> dict:
        severity = (
            "CRITICAL"
            if any(f["psi"] > 0.5 for f in drifted_features)
            else "WARNING"
        )

        payload = {
            "node_id": self.node_id,
            "timestamp": time.time(),
            "model_version": os.getenv("MODEL_VERSION", "v1.0"),
            "drift_severity": severity,
            "drifted_feature_count": len(drifted_features),
            "drifted_features": drifted_features,
            "bytes_saved_so_far": self.bytes_saved,
        }

        payload_json = json.dumps(payload)
        payload_bytes = len(payload_json.encode())

        if self.verbose:
            print(
                f"\n📡 Sending compressed drift payload "
                f"({payload_bytes} bytes) → {self.api_url}"
            )

        try:
            resp = requests.post(
                self.api_url,
                data=payload_json,
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            if self.verbose:
                print(f"   Response: {resp.status_code}")
        except Exception as exc:
            if self.verbose:
                print(f"   ⚠️  Could not reach API: {exc}")

        self.alerts_sent += 1
        return payload


# ---------------------------------------------------------------------------
# Standalone Demo Simulation
# ---------------------------------------------------------------------------
def run_simulation(
    baseline_path: str = "baseline_profile.json",
    n_clean: int = 300,
    n_drifted: int = 300,
    drift_magnitude: float = 3.0,
):
    """
    Simulates an edge device running inferences.
    Phase 1 (n_clean records)  : normal distribution → no alerts expected.
    Phase 2 (n_drifted records): shifted distribution → drift should be caught.
    """
    from baseline_generator import generate_baseline

    if not os.path.exists(baseline_path):
        print("No baseline found — generating one now...")
        generate_baseline(output_path=baseline_path)

    daemon = DriftDaemon(baseline_path=baseline_path, verbose=True)
    feature_names = daemon.feature_names

    # Build a simple normal-distribution mock using baseline means/stds
    base_means = {
        f: daemon.baseline["features"][f]["mean"] for f in feature_names
    }
    base_stds = {
        f: float(np.sqrt(daemon.baseline["features"][f]["variance"]))
        for f in feature_names
    }

    print(f"\n{'='*60}")
    print(f"Edge Node: {daemon.node_id}")
    print(f"Window size: {daemon.window_size} | Eval every: {daemon.eval_every}")
    print(f"{'='*60}")

    # ── Phase 1: Clean Data ──────────────────────────────────────────
    print(f"\n▶ Phase 1: {n_clean} clean inferences (no drift expected)\n")
    for i in range(n_clean):
        features = {
            f: float(np.random.normal(base_means[f], base_stds[f]))
            for f in feature_names
        }
        daemon.ingest(features)
        time.sleep(0.005)  # simulate 200 inferences/sec

    # ── Phase 2: Drifted Data ────────────────────────────────────────
    print(f"\n▶ Phase 2: {n_drifted} drifted inferences (drift_magnitude={drift_magnitude})\n")
    for i in range(n_drifted):
        features = {
            f: float(np.random.normal(base_means[f] + drift_magnitude, base_stds[f] * 2))
            for f in feature_names
        }
        daemon.ingest(features)
        time.sleep(0.005)

    print(f"\n{'='*60}")
    print(f"Simulation Complete.")
    print(f"  Total inferences  : {daemon.count}")
    print(f"  Alerts sent       : {daemon.alerts_sent}")
    print(f"  Estimated bytes saved (raw not sent): {daemon.bytes_saved:,}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_simulation()
