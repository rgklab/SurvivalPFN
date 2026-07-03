from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from survivalpfn import SurvivalEstimator, harrell_concordance_index


def make_tiny_censored_dataset(
    n_train: int = 64,
    n_test: int = 8,
    n_features: int = 5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_train + n_test, n_features)).astype(np.float32)
    risk = 0.8 * X[:, 0] - 0.4 * X[:, 1] + 0.2 * X[:, 2]
    event_time = np.exp(1.0 - risk + rng.normal(scale=0.35, size=X.shape[0]))
    censor_time = rng.exponential(scale=3.0, size=X.shape[0]) + 0.05
    observed_time = np.minimum(event_time, censor_time).astype(np.float32)
    event_indicator = (event_time <= censor_time).astype(np.float32)

    if event_indicator[:n_train].sum() < 2:
        censor_time[:4] = event_time[:4] + 1.0
        observed_time[:4] = event_time[:4].astype(np.float32)
        event_indicator[:4] = 1.0
    if event_indicator[:n_train].sum() > n_train - 2:
        censor_time[:4] = event_time[:4] * 0.5
        observed_time[:4] = censor_time[:4].astype(np.float32)
        event_indicator[:4] = 0.0

    return (
        X[:n_train],
        observed_time[:n_train],
        event_indicator[:n_train],
        X[n_train:],
        observed_time[n_train:],
        event_indicator[n_train:],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SurvivalPFN release-checkpoint quickstart."
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-path", default="shi-ang/SurvivalPFN")
    parser.add_argument("--cache-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    X_train, T_train, delta_train, X_test, T_test, delta_test = (
        make_tiny_censored_dataset()
    )

    estimator = SurvivalEstimator(
        device=args.device,
        model_path=args.model_path,
        cache_dir=args.cache_dir,
    )
    estimator.fit(X=X_train, T=T_train, delta=delta_train)

    distribution = estimator.predict_event_distribution(X_test)
    predicted_times = distribution.median().detach().cpu().numpy()
    time_grid = np.linspace(0.0, float(np.max(T_train)), 16, dtype=np.float32)
    survival_curves = distribution.survival_function(
        torch.from_numpy(time_grid).to(device=args.device, dtype=torch.float32)
    )
    survival_curves = survival_curves.detach().cpu().numpy()

    # Score the predictions with a real survival metric. The synthetic data has a
    # known risk structure, so a working model should clear random (c-index 0.5).
    c_index = harrell_concordance_index(
        preds=predicted_times, T=T_test, delta=delta_test
    )

    assert survival_curves.shape == (X_test.shape[0], time_grid.shape[0])
    assert ((survival_curves >= -1e-6) & (survival_curves <= 1.0 + 1e-6)).all()
    assert c_index > 0.5

    print(f"event_fraction={delta_train.mean():.3f}")
    print(f"concordance_index={c_index:.3f}")


if __name__ == "__main__":
    main()
