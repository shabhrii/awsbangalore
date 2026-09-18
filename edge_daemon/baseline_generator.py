import json
import numpy as np
from sklearn.datasets import fetch_california_housing


def generate_baseline(output_path: str = "baseline_profile.json") -> dict:
    """
    Loads California Housing dataset and computes a statistical baseline
    profile (mean, variance, histogram bins/probabilities) per feature.
    The profile is serialised to `output_path` and also returned as a dict.
    """
    print("Fetching California Housing dataset...")
    data = fetch_california_housing()
    X = data.data
    feature_names = list(data.feature_names)

    baseline_profile: dict = {"features": {}}

    print("Computing per-feature statistical baseline profiles...")
    for idx, name in enumerate(feature_names):
        feature_col = X[:, idx].astype(float)

        mean = float(np.mean(feature_col))
        variance = float(np.var(feature_col))

        # Build 10-bin histogram for PSI; store edges + normalised probabilities
        counts, bin_edges = np.histogram(feature_col, bins=10)

        # Avoid zero-probability bins (causes log(0) in PSI)
        probs = counts.astype(float)
        probs = np.where(probs == 0, 1e-4, probs)
        probs /= probs.sum()

        # Also store a small representative sample (100 pts) for KS baseline
        rng = np.random.default_rng(seed=42)
        sample_idx = rng.choice(len(feature_col), size=min(200, len(feature_col)), replace=False)
        baseline_sample = feature_col[sample_idx].tolist()

        baseline_profile["features"][name] = {
            "mean": mean,
            "variance": variance,
            "bin_edges": bin_edges.tolist(),
            "probabilities": probs.tolist(),
            "baseline_sample": baseline_sample,   # used by Tier-2 KS-Test
        }

    with open(output_path, "w") as f:
        json.dump(baseline_profile, f, indent=2)

    print(f"Saved baseline profile -> {output_path}")
    return baseline_profile


if __name__ == "__main__":
    generate_baseline()
