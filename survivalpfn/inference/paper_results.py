from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from survivalpfn.inference.paper import (
    ABLATION_RESULTS_COLUMNS,
    BENCHMARK_RESULTS_COLUMNS,
    SPLIT_RATIO_RESULTS_COLUMNS,
)

LOWER_IS_BETTER = {
    "IBS",
    "MAE",
    "NLL",
    "Dcal(stats)",
    "LR(stats)",
    "runtime_seconds",
}


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"Results file is missing required column(s): {missing}")


def _placeholder(outdir: Path, stem: str, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, "No result rows found", ha="center", va="center")
    ax.set_title(title)
    ax.set_axis_off()
    out_path = outdir / f"{stem}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _plot_benchmark(df: pd.DataFrame, outdir: Path) -> list[Path]:
    _require_columns(df, BENCHMARK_RESULTS_COLUMNS)
    if df.empty:
        return [_placeholder(outdir, "benchmark_ranks", "Benchmark Results")]

    plot_df = df.copy()
    plot_df["mean"] = pd.to_numeric(plot_df["mean"], errors="coerce")
    plot_df = plot_df.dropna(subset=["model", "dataset", "metric", "mean"])

    rank_rows = []
    for (dataset, metric), group in plot_df.groupby(["dataset", "metric"]):
        ascending = metric in LOWER_IS_BETTER
        ranks = group["mean"].rank(method="average", ascending=ascending)
        for idx, rank in ranks.items():
            rank_rows.append(
                {
                    "model": group.loc[idx, "model"],
                    "metric": metric,
                    "rank": float(rank),
                }
            )
    rank_df = pd.DataFrame(rank_rows)
    if rank_df.empty:
        return [_placeholder(outdir, "benchmark_ranks", "Benchmark Results")]

    summary = (
        rank_df.groupby("model", as_index=False)["rank"]
        .median()
        .sort_values("rank", ascending=True)
    )
    fig_height = max(3.0, 0.32 * len(summary))
    fig, ax = plt.subplots(figsize=(7, fig_height))
    ax.barh(summary["model"], summary["rank"], color="#4C78A8")
    ax.invert_yaxis()
    ax.set_xlabel("Median Rank")
    ax.set_title("Paper Benchmark Median Ranks")
    ax.grid(axis="x", alpha=0.25)
    out_path = outdir / "benchmark_ranks.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return [out_path]


def _plot_split_ratio(df: pd.DataFrame, outdir: Path) -> list[Path]:
    _require_columns(df, SPLIT_RATIO_RESULTS_COLUMNS)
    if df.empty:
        return [_placeholder(outdir, "split_ratio_results", "Split-Ratio Results")]

    paths = []
    plot_df = df.copy()
    plot_df["training_ratio"] = pd.to_numeric(
        plot_df["training_ratio"], errors="coerce"
    )
    plot_df["mean"] = pd.to_numeric(plot_df["mean"], errors="coerce")
    plot_df["se"] = pd.to_numeric(plot_df["se"], errors="coerce").fillna(0.0)
    plot_df = plot_df.dropna(
        subset=["model", "dataset", "training_ratio", "metric", "mean"]
    )

    for metric, metric_df in plot_df.groupby("metric"):
        fig, ax = plt.subplots(figsize=(7, 4))
        for (dataset, model), group in metric_df.groupby(["dataset", "model"]):
            group = group.sort_values("training_ratio")
            label = f"{model} / {dataset}"
            ax.errorbar(
                group["training_ratio"],
                group["mean"],
                yerr=group["se"],
                marker="o",
                linewidth=1.5,
                label=label,
            )
        ax.set_xlabel("Training Ratio")
        ax.set_ylabel(metric)
        ax.set_title(f"Split-Ratio Results: {metric}")
        ax.grid(alpha=0.25)
        if metric_df[["dataset", "model"]].drop_duplicates().shape[0] <= 12:
            ax.legend(fontsize=7)
        out_path = outdir / f"split_ratio_{_safe_name(metric)}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        paths.append(out_path)
    return paths or [_placeholder(outdir, "split_ratio_results", "Split-Ratio Results")]


def _plot_ablation(df: pd.DataFrame, outdir: Path) -> list[Path]:
    _require_columns(df, ABLATION_RESULTS_COLUMNS)
    if df.empty:
        return [_placeholder(outdir, "ablation_ranks", "Ablation Results")]

    plot_df = df.copy()
    plot_df["rank"] = pd.to_numeric(plot_df["rank"], errors="coerce")
    plot_df = plot_df.dropna(subset=["version", "metric", "rank"])
    if plot_df.empty:
        return [_placeholder(outdir, "ablation_ranks", "Ablation Results")]

    versions = sorted(plot_df["version"].unique())
    x_lookup = {version: i for i, version in enumerate(versions)}
    fig, ax = plt.subplots(figsize=(max(7, 0.45 * len(versions)), 4))
    for metric, group in plot_df.groupby("metric"):
        xs = [x_lookup[version] for version in group["version"]]
        ax.scatter(xs, group["rank"], label=metric, alpha=0.8)
    ax.set_xticks(np.arange(len(versions)), versions, rotation=45, ha="right")
    ax.set_ylabel("Rank")
    ax.set_title("SurvivalPFN Ablation Ranks")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7)
    out_path = outdir / "ablation_ranks.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return [out_path]


def _safe_name(value: str) -> str:
    return (
        "".join(
            char if char.isalnum() or char in {"-", "_"} else "_" for char in value
        ).strip("_")
        or "metric"
    )


def plot_results(results: Path, outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(results)
    columns = set(df.columns)
    if set(BENCHMARK_RESULTS_COLUMNS).issubset(columns):
        return _plot_benchmark(df, outdir)
    if set(SPLIT_RATIO_RESULTS_COLUMNS).issubset(columns):
        return _plot_split_ratio(df, outdir)
    if set(ABLATION_RESULTS_COLUMNS).issubset(columns):
        return _plot_ablation(df, outdir)
    raise ValueError(f"Unrecognized result schema in {results}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot frozen SurvivalPFN paper result tables."
    )
    parser.add_argument(
        "--results", type=Path, required=True, help="Frozen CSV result table."
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("output/plots"),
        help="Output directory for plots.",
    )
    return parser


def main(argv: list[str] | None = None) -> list[Path]:
    args = build_parser().parse_args(argv)
    paths = plot_results(args.results, args.outdir)
    for path in paths:
        print(path)
    return paths


if __name__ == "__main__":
    main()
