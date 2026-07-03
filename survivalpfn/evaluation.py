from typing import Optional, Tuple

import numpy as np
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index


def logrank(
    pred_times: np.ndarray,
    T: np.ndarray,
    delta: np.ndarray,
    weightings: Optional[str] = None,
    p: Optional[float] = None,
    q: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Perform the log-rank test to compare survival distributions.

    [1] Mantel N. Evaluation of survival data and two new rank order statistics.
        Cancer Chemotherapy Reports. 1966 Mar;50(3):163-70.

    Args:
        pred_times (np.ndarray): Predicted survival times.
        T (np.ndarray): True survival times.
        delta (np.ndarray): Event indicators (1 if event occurred, 0 if censored
        weightings (Optional[str]): Weighting scheme for the log-rank test.
            Options are None (default), 'wilcoxon', 'tarone-ware', 'fleming-harrington'.
        p (Optional[float]): Parameter p for the Fleming-Harrington
        q (Optional[float]): Parameter q for the Fleming-Harrington
    """
    results = logrank_test(
        durations_A=T,
        durations_B=pred_times,
        event_observed_A=delta,
        event_observed_B=np.ones_like(delta, dtype=bool),
        weightings=weightings,
        p=p,
        q=q,
    )
    return results.p_value, results.test_statistic


def d_cal(
    survival_curves: np.ndarray,
    time_grid: np.ndarray,
    T_obs: np.ndarray,
    delta_obs: np.ndarray,
    n_bins: int = 10,
    interpolation: str = "Linear",
) -> Tuple[float, np.ndarray]:
    """
    Calculate the D-Calibration score for survival predictions.

    [1] Haider S, Lee KM, Shah NH. Distribution calibration of survival
    models. InInternational Conference on Machine Learning 2020 Jul 13 (pp.
    4232-4242). PMLR.

    Args:
        survival_curves (np.ndarray): Predicted survival curves with shape (N, T_grid).
        time_grid (np.ndarray): Monotonic time support of the survival curves with shape (T_grid,).
        T_obs (np.ndarray): Observed survival/censoring times per individual with shape (N,).
        delta_obs (np.ndarray): Event indicators (1 if event, 0 if censored) with shape (N,).
        n_bins (int): Number of bins to use for D-Calibration (default: 10).
        interpolation (str): Interpolation method for predicting probabilities
            from survival curves. Options are 'Linear' (default) and 'Pchip'.
    Returns:
        Tuple[float, np.ndarray]: D-Calibration score and binning of the score.
    """
    n_samples = survival_curves.shape[0]
    predict_probs = np.empty(n_samples, dtype=survival_curves.dtype)

    for i in range(n_samples):
        predict_probs[i] = _predict_prob_from_curve(
            survival_curves[i, :], time_grid, T_obs[i]
        )

    from SurvivalEVAL.Evaluations.DistributionCalibration import d_calibration

    return d_calibration(predict_probs, delta_obs, num_bins=n_bins)


def mae(
    pred_times: np.ndarray,
    T: np.ndarray,
    delta: np.ndarray,
    train_T: Optional[np.ndarray] = None,
    train_delta: Optional[np.ndarray] = None,
    log_scale: bool = False,
    verbose: bool = False,
) -> float:
    """
    Calculate Mean Absolute Error (MAE) for survival predictions.
    If training event times and indicators are provided, use the
    Pseudo-observation method; otherwise, use the MAE-hinge method.

    [1] Qi SA, Kumar N, Farrokh M, Sun W, Kuan LH, Ranganath R, Henao R, Greiner R. An
     effective meaningful way to evaluate survival models. arXiv preprint
     arXiv:2306.01196. 2023 Jun 1.

    Args:
        pred_times (np.ndarray): Predicted survival times.
        T (np.ndarray): True survival times.
        delta (np.ndarray): Event indicators (1 if event occurred, 0 if censored
        train_T (Optional[np.ndarray]): event times for training set for Pseudo-observation method.
        train_delta (Optional[np.ndarray]): event indicators for training set for Pseudo-observation method.
        log_scale (bool): Whether to compute MAE in log scale.
    """
    if train_T is None or train_delta is None:
        if verbose:
            print("Train event times and indicators are not provided. Using MAE-hinge.")
        from SurvivalEVAL.Evaluations.MeanError import mean_error

        return mean_error(
            pred_times,
            T,
            delta,
            train_event_times=None,
            train_event_indicators=None,
            error_type="absolute",
            method="Hinge",
            weighted=False,
            log_scale=log_scale,
            verbose=False,
            truncated_time=None,
        )
    else:
        from SurvivalEVAL.Evaluations.MeanError import mean_error

        return mean_error(
            pred_times,
            T,
            delta,
            train_event_times=train_T,
            train_event_indicators=train_delta,
            error_type="absolute",
            method="Pseudo_obs",
            weighted=True,
            log_scale=log_scale,
            verbose=False,
            truncated_time=None,
        )


def harrell_concordance_index(
    preds: np.ndarray,
    T: np.ndarray,
    delta: np.ndarray,
) -> float:
    """
    Calculate Harrell's concordance index for survival predictions.

    [1] Harrell FE, Lee KL, Mark DB. Multivariable prognostic models: issues in
        developing models, evaluating assumptions and adequacy, and measuring
        and reducing errors. Statistics in Medicine 1996;15(4):361-87.

    Args:
        preds (np.ndarray): Predicted survival times (E[Event | X]).
        T (np.ndarray): Survival times.
        delta (np.ndarray): Event indicators (1 if event occurred, 0 if censored).

    Returns:
        float: Concordance index.
    """
    return concordance_index(T, preds, delta)



def integrated_brier_score(
    survival_curves: np.ndarray,
    time_grid: np.ndarray,
    T_obs: np.ndarray,
    delta_obs: np.ndarray,
    train_T: Optional[np.ndarray] = None,
    train_delta: Optional[np.ndarray] = None,
) -> float:
    """
    Calculate the integrated Brier score for survival predictions.

    [1] Graf E, Schmoor C, Sauerbrei W, Schumacher M. Assessment and comparison of
    prognostic classification schemes for survival data. Statistics in medicine.
    1999 Sep 15;18(17‐18):2529-45.

    Args:
        survival_curves (np.ndarray): Predicted survival curves for each individual (N, T_grid).
        time_grid (np.ndarray): Monotonic time support used for survival curves (T_grid,).
        T_obs (np.ndarray): Observed times per individual (N,).
        delta_obs (np.ndarray): Event indicators per individual (N,).
        train_T (Optional[np.ndarray]): Training-set observed times used to estimate censoring weights.
        train_delta (Optional[np.ndarray]): Training-set event indicators used to estimate censoring weights.

    Returns:
        float: Integrated Brier score.
    """
    from SurvivalEVAL import SurvivalEvaluator

    evaluator = SurvivalEvaluator(
        pred_survs=survival_curves,
        time_coordinates=time_grid,
        event_times=T_obs,
        event_indicators=delta_obs,
        train_event_times=train_T,
        train_event_indicators=train_delta,
        predict_time_method="Median",
        interpolation="Linear",
    )
    return evaluator.integrated_brier_score(
        target_times=np.asarray(time_grid), IPCW_weighted=True
    )


def nll_surv(
    pred_surv: np.ndarray,  # (N,)
    pred_dens: np.ndarray,  # (N,)
    delta: np.ndarray,  # (N,)
    eps: float = 1e-12,
) -> float:
    """
    Calculate Negative Log-Likelihood (NLL) for survival predictions.

    Args:
        pred_surv (np.ndarray): Predicted survival probabilities at observed times (N,).
        pred_dens (np.ndarray): Predicted density values at observed times (N,).
        delta (np.ndarray): Event indicators (1 if event, 0 if censored) (N,).
        eps (float): Small value to avoid log(0).

    Returns:
        float: Negative Log-Likelihood (NLL) value.
    """
    if pred_surv.shape != pred_dens.shape or pred_surv.shape != delta.shape:
        raise ValueError(
            f"pred_surv, pred_dens, and delta must have the same shape, got "
            f"{pred_surv.shape}, {pred_dens.shape}, {delta.shape}."
        )
    if pred_surv.ndim != 1:
        raise ValueError(
            f"pred_surv, pred_dens, and delta must be 1-dimensional, got ndim={pred_surv.ndim}."
        )

    nll = -(delta * np.log(pred_dens + eps) + (1 - delta) * np.log(pred_surv + eps))
    return nll.mean()


def _predict_prob_from_curve(
    survival_curve: np.ndarray, time_grid: np.ndarray, time: float
) -> float:
    survival_curve = np.asarray(survival_curve, dtype=float)
    time_grid = np.asarray(time_grid, dtype=float)
    if time <= time_grid[0]:
        return float(survival_curve[0])
    if time >= time_grid[-1]:
        return float(survival_curve[-1])
    return float(np.interp(time, time_grid, survival_curve))
