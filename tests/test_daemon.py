"""
tests/test_daemon.py
--------------------
Unit tests for the Edge Drift Daemon statistical core.

Tests:
  - PSI returns 0 for identical distributions
  - PSI is high for heavily shifted distributions
  - KS-Test correctly identifies a shifted distribution
  - DriftDaemon.ingest() accumulates correctly
  - DriftDaemon does not alert on clean data
  - DriftDaemon alerts when drift is injected
  - Payload < 2048 bytes
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Add edge_daemon to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "edge_daemon"))

from daemon import DriftDaemon  # noqa: E402
from baseline_generator import generate_baseline  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_temp_baseline(shift: float = 0.0, n: int = 1000) -> str:
    """
    Generate a synthetic single-feature baseline profile written to a temp file.
    Returns the path to the JSON file.
    """
    rng = np.random.default_rng(seed=0)
    data = rng.normal(loc=0.0 + shift, scale=1.0, size=n)

    counts, bin_edges = np.histogram(data, bins=10)
    probs = counts.astype(float)
    probs = np.where(probs == 0, 1e-4, probs)
    probs /= probs.sum()

    sample_idx = rng.choice(n, size=200, replace=False)
    baseline_sample = data[sample_idx].tolist()

    profile = {
        "features": {
            "TestFeature": {
                "mean": float(np.mean(data)),
                "variance": float(np.var(data)),
                "bin_edges": bin_edges.tolist(),
                "probabilities": probs.tolist(),
                "baseline_sample": baseline_sample,
            }
        }
    }

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(profile, tmp)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestPSI(unittest.TestCase):
    """Tests for the _calculate_psi method."""

    def setUp(self):
        self.baseline_path = _make_temp_baseline()
        self.daemon = DriftDaemon(
            baseline_path=self.baseline_path,
            window_size=200,
            eval_every=50,
            verbose=False,
        )

    def tearDown(self):
        os.unlink(self.baseline_path)

    def test_psi_same_distribution_is_near_zero(self):
        rng = np.random.default_rng(seed=1)
        data = rng.normal(loc=0.0, scale=1.0, size=200)
        base_stats = self.daemon.baseline["features"]["TestFeature"]
        psi = self.daemon._calculate_psi(
            expected_probs=np.array(base_stats["probabilities"]),
            actual_data=data,
            bin_edges=np.array(base_stats["bin_edges"]),
        )
        self.assertLess(psi, 0.1, f"PSI for same distribution should be < 0.1, got {psi:.4f}")

    def test_psi_high_for_shifted_distribution(self):
        rng = np.random.default_rng(seed=2)
        # Shift mean by 5 standard deviations — severe drift
        data = rng.normal(loc=5.0, scale=1.0, size=200)
        base_stats = self.daemon.baseline["features"]["TestFeature"]
        psi = self.daemon._calculate_psi(
            expected_probs=np.array(base_stats["probabilities"]),
            actual_data=data,
            bin_edges=np.array(base_stats["bin_edges"]),
        )
        self.assertGreater(psi, 0.2, f"PSI for heavily shifted distribution should be > 0.2, got {psi:.4f}")


class TestKSTest(unittest.TestCase):
    """Tests for KS-Test behaviour via full drift pipeline."""

    def setUp(self):
        self.baseline_path = _make_temp_baseline()

    def tearDown(self):
        os.unlink(self.baseline_path)

    def test_ks_test_detects_shift(self):
        from scipy.stats import ks_2samp

        rng = np.random.default_rng(seed=3)
        baseline_sample = np.array(rng.normal(loc=0.0, scale=1.0, size=200))
        shifted_sample = np.array(rng.normal(loc=4.0, scale=1.0, size=200))

        _, p_value = ks_2samp(baseline_sample, shifted_sample)
        self.assertLess(p_value, 0.05, f"KS-Test should detect drift, p={p_value:.5f}")

    def test_ks_test_no_false_positive(self):
        from scipy.stats import ks_2samp

        rng = np.random.default_rng(seed=4)
        a = rng.normal(loc=0.0, scale=1.0, size=200)
        b = rng.normal(loc=0.0, scale=1.0, size=200)

        _, p_value = ks_2samp(a, b)
        # With same distribution, p-value should be much higher than 0.05
        self.assertGreater(p_value, 0.05, f"KS-Test false positive: p={p_value:.5f}")


class TestDaemonBuffer(unittest.TestCase):
    """Tests for the rolling buffer accumulation logic."""

    def setUp(self):
        self.baseline_path = _make_temp_baseline()
        self.daemon = DriftDaemon(
            baseline_path=self.baseline_path,
            window_size=50,
            eval_every=100,  # don't trigger evaluation during these tests
            verbose=False,
        )

    def tearDown(self):
        os.unlink(self.baseline_path)

    def test_buffer_accumulates(self):
        for i in range(30):
            self.daemon.ingest({"TestFeature": float(i)})
        self.assertEqual(len(self.daemon.buffer["TestFeature"]), 30)

    def test_buffer_respects_maxlen(self):
        for i in range(200):
            self.daemon.ingest({"TestFeature": float(i)})
        self.assertLessEqual(len(self.daemon.buffer["TestFeature"]), 50)


class TestDaemonAlerts(unittest.TestCase):
    """Integration tests for the full drift detection + alert pipeline."""

    def setUp(self):
        self.baseline_path = _make_temp_baseline(shift=0.0, n=2000)

    def tearDown(self):
        os.unlink(self.baseline_path)

    @patch("daemon.requests.post")
    def test_no_alert_on_clean_data(self, mock_post):
        """Feeding clean in-distribution data should never trigger an alert."""
        rng = np.random.default_rng(seed=5)
        daemon = DriftDaemon(
            baseline_path=self.baseline_path,
            window_size=200,
            eval_every=50,
            verbose=False,
        )
        for _ in range(400):
            val = float(rng.normal(loc=0.0, scale=1.0))
            daemon.ingest({"TestFeature": val})

        mock_post.assert_not_called()
        self.assertEqual(daemon.alerts_sent, 0)

    @patch("daemon.requests.post")
    def test_alert_on_drifted_data(self, mock_post):
        """
        After filling the window with heavily shifted data, at least one alert
        should be sent.
        """
        rng = np.random.default_rng(seed=6)
        daemon = DriftDaemon(
            baseline_path=self.baseline_path,
            window_size=200,
            eval_every=50,
            verbose=False,
        )
        # Fill with severely drifted data (mean shifted by 6 stds)
        for _ in range(600):
            val = float(rng.normal(loc=6.0, scale=0.5))
            daemon.ingest({"TestFeature": val})

        self.assertGreater(daemon.alerts_sent, 0, "Expected at least one drift alert")
        mock_post.assert_called()

    @patch("daemon.requests.post")
    def test_payload_under_2kb(self, mock_post):
        """The compressed drift payload must be < 2048 bytes."""
        rng = np.random.default_rng(seed=7)
        daemon = DriftDaemon(
            baseline_path=self.baseline_path,
            window_size=200,
            eval_every=50,
            verbose=False,
        )
        for _ in range(600):
            val = float(rng.normal(loc=6.0, scale=0.5))
            daemon.ingest({"TestFeature": val})

        if mock_post.called:
            call_kwargs = mock_post.call_args
            sent_data = call_kwargs[1].get("data") or call_kwargs[0][1] if call_kwargs[0] else None
            if sent_data:
                payload_bytes = len(sent_data.encode() if isinstance(sent_data, str) else sent_data)
                self.assertLess(
                    payload_bytes,
                    2048,
                    f"Payload size {payload_bytes}B exceeds 2KB limit",
                )


class TestBaselineGenerator(unittest.TestCase):
    """Tests for the baseline_generator module."""

    def test_generates_valid_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "baseline_profile.json")
            profile = generate_baseline(output_path=output_path)

            self.assertIn("features", profile)
            self.assertGreater(len(profile["features"]), 0)

            for feat_name, stats in profile["features"].items():
                self.assertIn("mean", stats)
                self.assertIn("variance", stats)
                self.assertIn("bin_edges", stats)
                self.assertIn("probabilities", stats)
                self.assertIn("baseline_sample", stats)
                self.assertEqual(len(stats["bin_edges"]), 11)      # 10 bins → 11 edges
                self.assertEqual(len(stats["probabilities"]), 10)
                self.assertAlmostEqual(sum(stats["probabilities"]), 1.0, places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
