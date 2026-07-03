"""
A callback designed for evaluating the model performance on a test set.
"""

from typing import Callable, Dict, List

import numpy as np
import torch
import wandb
from tqdm import tqdm

from survivalpfn.evaluation import (
    harrell_concordance_index,
    integrated_brier_score,
    nll_surv,
)
from survivalpfn.inference.data import SurvivalBenchmarkDataset
from survivalpfn.inference.paper import PAPER_VALIDATION_DATASETS
from survivalpfn.models import InContextModel
from survivalpfn.survival_estimator import SurvivalEstimator

from .base import Callback


class EvalSurvival(Callback):
    """
    Evaluates the trained survival model on a validation set of survival datasets. Specifically, we use an aggregated
    score across multiple datasets to evaluate the model performance (we do weighted average based on square root
    of the number of samples in each dataset):

    - Average Harrell's Concordance Index
    - Average Integrated Brier Score
    - Average Negative Log-Likelihood
    """

    def __init__(
        self,
        eval_datasets_names: List[str] | None,
        survival_estimator_partial: Callable[[InContextModel], SurvivalEstimator],
        callback_name: str | None,
        test_ratio: float = 0.3,
        seed: int = 183742,
        frequency: int = 1,
        estimator_fitting_kwargs: Dict | None = None,
    ):
        super().__init__(callback_name=callback_name)
        self.frequency = frequency
        eval_datasets_names = eval_datasets_names or PAPER_VALIDATION_DATASETS
        self.eval_datasets = {
            name: SurvivalBenchmarkDataset(
                data_name=name,
                train_ratio=1 - test_ratio,
                val_ratio=0.0,
                test_ratio=test_ratio,
                preprocess=False,
                seed=seed,
                fixed_split=True,
            )
            for name in eval_datasets_names
        }
        self.survival_estimator_partial = survival_estimator_partial
        self.pbar = tqdm(
            range(len(self.eval_datasets)), desc="Evaluating Survival Analysis"
        )
        self.estimator_fitting_kwargs = estimator_fitting_kwargs or {}

    @torch.no_grad()
    def __call__(self, context):
        """
        Evaluate the model on survival analysis benchmarks.

        Args:
            context: CallbackContext with training state
        """
        if not context.is_main_process():
            return

        if (context.epoch + 1) % self.frequency == 0:
            context.model.eval()
            metrics = {
                "Avg_Harrell_Concordance": [],
                "Avg_Integrated_Brier": [],
                "Avg_NLL": [],
            }
            weights = []
            for eval_dataset in self.eval_datasets.values():
                survival_estimator: SurvivalEstimator = self.survival_estimator_partial(
                    icl_model=context.model
                )
                survival_data = eval_dataset()
                X_train, T_train, delta_train, X_test = (
                    survival_data["X_train"],
                    survival_data["T_train"],
                    survival_data["delta_train"],
                    survival_data["X_test"],
                )
                weights.append(np.sqrt(X_train.shape[0]))
                T_test, delta_test = (
                    survival_data["T_test"],
                    survival_data["delta_test"],
                )
                survival_estimator.fit(
                    X=X_train,
                    delta=delta_train,
                    T=T_train,
                    **self.estimator_fitting_kwargs,
                )

                dist = survival_estimator.predict_event_distribution(X=X_test)
                T_pred = dist.median().cpu().numpy()
                harrell_c_index = harrell_concordance_index(
                    preds=T_pred, T=T_test, delta=delta_test
                )

                # The paper protocol defines evaluation support from training data only.
                t_grid = np.unique(T_train)
                if t_grid.shape[0] < 2:
                    t_grid = np.unique(np.concatenate([T_train, T_test]))
                t_grid_tensor = torch.from_numpy(t_grid).to(
                    device=survival_estimator.device, dtype=torch.float32
                )
                survival_curves_pred = (
                    dist.survival_function(time=t_grid_tensor).cpu().numpy()
                )
                integrated_brier = integrated_brier_score(
                    survival_curves=survival_curves_pred,
                    time_grid=t_grid,
                    T_obs=T_test,
                    delta_obs=delta_test,
                    train_T=T_train,
                    train_delta=delta_train,
                )

                obs_time_tensor = torch.from_numpy(T_test).to(
                    device=survival_estimator.device, dtype=torch.float32
                )
                pred_surv = (
                    dist.survival_at(obs_time_tensor).cpu().numpy()
                )  # shape: (N,)
                pred_dens = (
                    dist.density_at(obs_time_tensor).cpu().numpy()
                )  # shape: (N,)
                nll = nll_surv(
                    pred_surv=pred_surv, pred_dens=pred_dens, delta=delta_test
                )

                self.pbar.update(1)
                metrics["Avg_Harrell_Concordance"].append(harrell_c_index)
                metrics["Avg_Integrated_Brier"].append(integrated_brier)
                metrics["Avg_NLL"].append(nll)
            self.pbar.reset()

            metrics = {k: np.average(v, weights=weights) for k, v in metrics.items()}
            report = {k: float(v) for k, v in metrics.items()}
            context.callback_metrics["avg_harrell_concordance"] = report[
                "Avg_Harrell_Concordance"
            ]
            context.callback_metrics["avg_integrated_brier"] = report[
                "Avg_Integrated_Brier"
            ]
            self.pbar.set_postfix(report)

            if context.should_log():
                wandb.log(
                    dict(
                        [
                            (
                                f"callback{context.callback_idx}:{self.callback_name}/{k}",
                                v,
                            )
                            for k, v in report.items()
                        ]
                    ),
                )

            context.model.train()
