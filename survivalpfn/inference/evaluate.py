from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch

from survivalpfn import (
    DEFAULT_MODEL_REPO_ID,
    SurvivalEstimator,
    d_cal,
    harrell_concordance_index,
    integrated_brier_score,
    logrank,
    mae,
    nll_surv,
)
from survivalpfn.inference.data import ALL_DATA, SurvivalBenchmarkDataset
from survivalpfn.inference.paper import (
    BENCHMARK_RESULTS_COLUMNS,
    PAPER_TEST_DATASETS,
    PAPER_VALIDATION_DATASETS,
)

DATASET_ALIASES = {
    "paper-validation": PAPER_VALIDATION_DATASETS,
    "paper-test": PAPER_TEST_DATASETS,
    "paper-all": [*PAPER_VALIDATION_DATASETS, *PAPER_TEST_DATASETS],
}

METRIC_BACKENDS = ("paper", "lightweight")


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"yes", "true", "t", "y", "1"}:
        return True
    if normalized in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def _parse_datasets(value: str) -> list[str]:
    if value in DATASET_ALIASES:
        return DATASET_ALIASES[value]
    datasets = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(datasets) - set(ALL_DATA))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown dataset(s): {unknown}")
    return datasets


def _resolve_preprocess(dataset_name: str, preprocess: bool | None) -> bool:
    if preprocess is not None:
        return preprocess
    return dataset_name == "MSKCC"


def _seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _time_grid_from_training(T_train: np.ndarray, T_test: np.ndarray) -> np.ndarray:
    grid = np.unique(np.concatenate([np.array([0.0]), T_train.astype(float)]))
    if grid.shape[0] < 2:
        grid = np.unique(np.concatenate([grid, T_test.astype(float)]))
    if grid.shape[0] < 2:
        upper = float(grid[0] + 1.0) if grid.shape[0] == 1 else 1.0
        grid = np.array([0.0, upper], dtype=float)
    return grid


def _load_paper_metric_dependencies():
    from functools import cached_property

    from scipy.stats import chisquare
    from SurvivalEVAL import SurvivalEvaluator

    return cached_property, chisquare, SurvivalEvaluator


def _evaluate_once(
    *,
    dataset_name: str,
    model_path: str,
    model_filename: str | None,
    model_revision: str | None,
    device: torch.device,
    seed: int,
    train_ratio: float,
    test_ratio: float,
    fixed_split: bool,
    calibrate_temperature: bool,
    calibrate_n_folds: int,
    calibrate_t_min: float,
    calibrate_t_max: float,
    calibrate_t_size: int,
    calibrate_t_batch_size: int,
    preprocess: bool | None,
    metrics_backend: str,
    verbose: bool,
) -> dict[str, float]:
    _seed_everything(seed, device)
    paper_metric_dependencies = (
        _load_paper_metric_dependencies() if metrics_backend == "paper" else None
    )
    dataset = SurvivalBenchmarkDataset(
        data_name=dataset_name,
        train_ratio=train_ratio,
        val_ratio=0.0,
        test_ratio=test_ratio,
        preprocess=_resolve_preprocess(dataset_name, preprocess),
        seed=seed,
        fixed_split=fixed_split,
    )
    split = dataset()

    start = time.perf_counter()
    estimator = SurvivalEstimator(
        model_path=model_path,
        model_filename=model_filename,
        model_revision=model_revision,
        device=str(device),
        verbose=verbose,
        calibrate=calibrate_temperature,
        n_folds=calibrate_n_folds,
        calibrate_T_min=calibrate_t_min,
        calibrate_T_max=calibrate_t_max,
        calibrate_T_size=calibrate_t_size,
        calibrate_T_batch_size=calibrate_t_batch_size,
    )
    estimator.fit(X=split["X_train"], delta=split["delta_train"], T=split["T_train"])
    dist = estimator.predict_event_distribution(split["X_test"])
    pred_times = dist.median().cpu().numpy()

    time_grid = _time_grid_from_training(split["T_train"], split["T_test"])
    time_grid_tensor = torch.from_numpy(time_grid).to(
        device=estimator.device, dtype=torch.float32
    )
    survival_curves = dist.survival_function(time_grid_tensor).cpu().numpy()
    obs_time = torch.from_numpy(split["T_test"]).to(
        device=estimator.device, dtype=torch.float32
    )
    pred_surv = dist.survival_at(obs_time).cpu().numpy()
    pred_dens = dist.density_at(obs_time).cpu().numpy()
    runtime_seconds = time.perf_counter() - start

    if metrics_backend == "paper":
        cached_property, chisquare, SurvivalEvaluator = paper_metric_dependencies

        class PFNSurvivalEvaluator(SurvivalEvaluator):
            @cached_property
            def predicted_event_times(self, *args, **kwargs):
                return pred_times

        evaluator = PFNSurvivalEvaluator(
            survival_curves,
            time_grid,
            split["T_test"],
            split["delta_test"],
            split["T_train"],
            split["delta_train"],
            predict_time_method="Median",
            interpolation="Linear",
        )
        dcal_pvalue, dcal_hist = evaluator.d_calibration()
        dcal_stat, _ = chisquare(dcal_hist)
        logrank_pvalue, logrank_stat = evaluator.log_rank()
        metrics = {
            "CI": evaluator.concordance(ties="Risk")[0],
            "IBS": evaluator.integrated_brier_score(num_points=10),
            "MAE": evaluator.mae(method="Pseudo_obs", verbose=False, weighted=True),
            "LR(pval)": logrank_pvalue,
            "LR(stats)": logrank_stat,
            "Dcal(pval)": dcal_pvalue,
            "Dcal(stats)": dcal_stat,
            "NLL": nll_surv(
                pred_surv=pred_surv, pred_dens=pred_dens, delta=split["delta_test"]
            ),
            "runtime_seconds": runtime_seconds,
        }
    elif metrics_backend == "lightweight":
        dcal_stat, dcal_pvalue, _ = d_cal(
            survival_curves=survival_curves,
            time_grid=time_grid,
            T_obs=split["T_test"],
            delta_obs=split["delta_test"],
        )
        logrank_pvalue, logrank_stat = logrank(
            pred_times=pred_times, T=split["T_test"], delta=split["delta_test"]
        )
        metrics = {
            "CI": harrell_concordance_index(
                preds=pred_times, T=split["T_test"], delta=split["delta_test"]
            ),
            "IBS": integrated_brier_score(
                survival_curves=survival_curves,
                time_grid=time_grid,
                T_obs=split["T_test"],
                delta_obs=split["delta_test"],
                train_T=split["T_train"],
                train_delta=split["delta_train"],
            ),
            "MAE": mae(
                pred_times=pred_times,
                T=split["T_test"],
                delta=split["delta_test"],
                train_T=split["T_train"],
                train_delta=split["delta_train"],
            ),
            "LR(pval)": logrank_pvalue,
            "LR(stats)": logrank_stat,
            "Dcal(pval)": dcal_pvalue,
            "Dcal(stats)": dcal_stat,
            "NLL": nll_surv(
                pred_surv=pred_surv, pred_dens=pred_dens, delta=split["delta_test"]
            ),
            "runtime_seconds": runtime_seconds,
        }
    else:
        raise ValueError(f"Unknown metrics backend: {metrics_backend}")
    return metrics


def _summarize(
    dataset_name: str, runs: list[dict[str, float]]
) -> list[dict[str, object]]:
    rows = []
    runtimes = np.array([run["runtime_seconds"] for run in runs], dtype=float)
    for metric in sorted(key for key in runs[0] if key != "runtime_seconds"):
        values = np.array([run[metric] for run in runs], dtype=float)
        rows.append(
            {
                "model": "SurvivalPFN",
                "dataset": dataset_name,
                "metric": metric,
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values, ddof=1)) if values.shape[0] > 1 else 0.0,
                "runtime_seconds": float(np.nanmean(runtimes)),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BENCHMARK_RESULTS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate SurvivalPFN on survival benchmark datasets."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Dataset name, comma-separated names, or one of: paper-validation, paper-test, paper-all.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_REPO_ID,
        help="Local checkpoint or Hugging Face repo id.",
    )
    parser.add_argument(
        "--model-filename",
        default=None,
        help="Checkpoint filename when --model-path is a Hugging Face repo id.",
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Hugging Face branch, tag, or commit for --model-path.",
    )
    parser.add_argument(
        "--n-exp", type=int, default=10, help="Number of random splits to evaluate."
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--test-ratio", type=float, default=0.3)
    parser.add_argument("--fixed-split", type=_str_to_bool, default=False)
    parser.add_argument(
        "--out", type=Path, default=None, help="Optional CSV output path."
    )
    parser.add_argument("--verbose", type=_str_to_bool, default=False)
    parser.add_argument("--calibrate-temperature", type=_str_to_bool, default=False)
    parser.add_argument("--calibrate-n-folds", type=int, default=3)
    parser.add_argument("--calibrate-t-min", type=float, default=0.001)
    parser.add_argument("--calibrate-t-max", type=float, default=10.0)
    parser.add_argument("--calibrate-t-size", type=int, default=500)
    parser.add_argument("--calibrate-t-batch-size", type=int, default=50)
    parser.add_argument(
        "--preprocess",
        type=_str_to_bool,
        default=None,
        help="Impute/scale benchmark covariates before evaluation. Defaults to the paper protocol: false except MSKCC.",
    )
    parser.add_argument(
        "--metrics-backend",
        choices=METRIC_BACKENDS,
        default="paper",
        help="Metric implementation to use. 'paper' mirrors the SurvivalEVAL calls used for the paper benchmark; 'lightweight' uses local metric wrappers.",
    )
    return parser


def main(argv: list[str] | None = None) -> list[dict[str, object]]:
    args = build_parser().parse_args(argv)
    datasets = _parse_datasets(args.data)
    device = torch.device(args.device)

    all_rows: list[dict[str, object]] = []
    for dataset_name in datasets:
        runs = [
            _evaluate_once(
                dataset_name=dataset_name,
                model_path=args.model_path,
                model_filename=args.model_filename,
                model_revision=args.model_revision,
                device=device,
                seed=args.seed + i,
                train_ratio=args.train_ratio,
                test_ratio=args.test_ratio,
                fixed_split=args.fixed_split,
                calibrate_temperature=args.calibrate_temperature,
                calibrate_n_folds=args.calibrate_n_folds,
                calibrate_t_min=args.calibrate_t_min,
                calibrate_t_max=args.calibrate_t_max,
                calibrate_t_size=args.calibrate_t_size,
                calibrate_t_batch_size=args.calibrate_t_batch_size,
                preprocess=args.preprocess,
                metrics_backend=args.metrics_backend,
                verbose=args.verbose,
            )
            for i in range(args.n_exp)
        ]
        rows = _summarize(dataset_name, runs)
        all_rows.extend(rows)
        for row in rows:
            print(
                f"{row['dataset']} {row['metric']}: "
                f"{row['mean']:.6g} +/- {row['std']:.6g} "
                f"(runtime {row['runtime_seconds']:.3f}s)"
            )

    if args.out is not None:
        _write_csv(args.out, all_rows)
    return all_rows


if __name__ == "__main__":
    main()
