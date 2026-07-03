import math
from typing import Literal, NamedTuple

import torch
import torch.nn.functional as F

from .constants import EPSILON_STABILITY


def pad_x(X: torch.Tensor, num_features=100):
    if num_features is None:
        return X
    n_features = X.shape[-1]
    zero_feature_padding = torch.zeros(
        (*X.shape[:-1], num_features - n_features), device=X.device, dtype=X.dtype
    )
    return torch.cat([X, zero_feature_padding], dim=-1)


def surv_nll(
    logits: torch.Tensor,
    t_target: torch.Tensor,
    delta_target: torch.Tensor,
    bin_edges: torch.Tensor,
):
    """
    Calculate the negative log-likelihood for survival data given predicted logits, target times, and event indicators.
    The method uses discretized likelihood.

    Args:
      logits: Tensor of shape (..., nbins) containing the predicted distribution
      t_target: Tensor of shape (...) containing target times
      delta_target: Tensor of shape (...) containing event indicators (1 if event occurred, 0 if censored)
      bin_edges: Tensor of shape (nbins,) containing the left edges of each bin, the right edge of a bin is the left edge of the next bin,
                and the last value is the left edge of the last bin for residual mass beyond vmax (the right edge of this bin is infinite)
    Note:
      The event likelihood p(T = t) is approximated by the probability mass in the bin
      containing t_target. The censoring likelihood is approximated by the inclusive
      right-tail mass from the target bin's left edge.
      The NLL loss is then computed as:
      NLL = - delta_target * log(p(T = t_target)) - (1 - delta_target) * log(p(T >= t_bin_left))
    """
    assert t_target.shape == logits.shape[:-1] == delta_target.shape, (
        "t_target and delta_target must match logits shape except for the last dimension."
    )

    nbins = bin_edges.shape[0]

    # Convert time targets to bin indices based on bin_edges
    bin_indices = (
        torch.searchsorted(bin_edges, t_target, right=True) - 1
    )  # shape: (...,)
    bin_indices = torch.clamp(bin_indices, min=0, max=nbins - 1)

    # Log-probabilities over bins
    log_prob_mass = F.log_softmax(logits, dim=-1)  # (..., nbins)

    # Event likelihood: p(T = t) approximated by mass in the target bin
    log_mass_at_t = torch.gather(
        log_prob_mass, dim=-1, index=bin_indices.unsqueeze(-1)
    ).squeeze(-1)

    # Censoring likelihood uses an inclusive right tail:
    # p(T >= t_bin_left) ~= sum_{j >= bin_idx} p_j
    # Compute in log-space for numerical stability.
    log_tail_mass = torch.logcumsumexp(
        torch.flip(log_prob_mass, dims=[-1]), dim=-1
    ).flip(dims=[-1])
    log_surv_at_t = torch.gather(
        log_tail_mass, dim=-1, index=bin_indices.unsqueeze(-1)
    ).squeeze(-1)

    # NLL: event term for uncensored, survival term for censored
    delta = delta_target.to(dtype=logits.dtype)
    nll = -(delta * log_mass_at_t + (1.0 - delta) * log_surv_at_t)
    return nll


def forward_KL_one_hot_loss(
    logits: torch.Tensor,
    y_target: torch.Tensor,
    bin_centers: torch.Tensor,
    vmin: float,
    vmax: float,
):
    """
    Calculate the cross-entropy loss (forward KL) between the predicted distribution and the target tensor.

    Args:
        logits: Tensor of shape (..., nbins) containing the predicted distribution
        y_target: Tensor of shape (...) containing target values
        bin_centers: Tensor of shape (nbins - 1,) containing the centers of the bins, excluding the extra bin for residual mass beyond vmax
        vmin: Minimum value of the distribution
        vmax: Maximum value of the distribution
    Note:
        This function constructs a hard target distribution (one per element in y_target)
        by assigning all mass to the bin that contains the target value.
    """

    bin_width = bin_centers[1] - bin_centers[0]
    nbins = bin_centers.shape[0] + 1

    # no smoothing, just use the bin that contains the target value
    bin_indices = torch.clamp(
        ((y_target - vmin) / bin_width).long(), min=0, max=nbins - 1
    )
    target_distribution = F.one_hot(bin_indices, num_classes=nbins).float()
    log_probs = F.log_softmax(logits, dim=-1)  # shape: (..., nbins)
    ce_loss = -torch.sum(target_distribution * log_probs, dim=-1)
    return ce_loss


def forward_KL_hl_gaussian_loss(
    logits: torch.Tensor,
    y_target: torch.Tensor,
    bin_edges: torch.Tensor,
    sigma: float,
    vmin: float,
    vmax: float,
) -> torch.Tensor:
    """
    Calculate the cross-entropy loss (forward KL) between the predicted distribution and the target tensor.

    Args:
        logits: Tensor of shape (..., nbins) containing the predicted distribution
        y_target: Tensor of shape (...) containing target values
        bin_edges: Tensor of shape (nbins,) containing the edges of the all the bins, the last value is the left
                   edge of the extra bin for the residual mass beyond vmax
        sigma: Standard deviation for the Gaussian smoothing
        vmin: Minimum value of the distribution
        vmax: Maximum value of the distribution

    Note:
        This function constructs a soft target distribution (one per element in y_target)
        by integrating a Gaussian centered at y_i over each bin. The resulting per-sample distribution is
        renormalized to sum to 1.
    """

    assert y_target.shape == logits.shape[:-1], (
        "y_target must have the same shape as logits except for the last dimension."
    )

    bin_width = bin_edges[1] - bin_edges[0]
    nbins = bin_edges.shape[0]

    # Expand y_target so we can broadcast:
    # y_target: (...) => (..., 1) so we can compare with each bin edge
    y_target_expanded = y_target.unsqueeze(-1)  # => (..., 1)

    # cdf_lower = CDF at the left edge of each bin, cdf_upper = CDF at the right edge of each bin
    # for the last bin, the upper edge is effectively +inf, so CDF is 1
    # Phi(x) = 0.5 * [1 + erf((x - mean) / (sqrt(2) * std))], shape: (..., nbins)
    cdf_lower = 0.5 * (
        1.0
        + torch.erf(
            (bin_edges - y_target_expanded)
            / (math.sqrt(2.0) * (sigma + EPSILON_STABILITY))
        )
    )
    cdf_upper = torch.cat(
        [cdf_lower[..., 1:], torch.ones_like(cdf_lower[..., :1])], dim=-1
    )  # shift left and pad right with 1s

    # Integrate over full Gaussian for each bin: P_k = CDF(U_k) - CDF(L_k)
    p = torch.clamp(cdf_upper - cdf_lower, min=0.0)  # shape: (..., nbins)

    # Handle degenerate cases where truncated integration yields zero mass in-range
    # Use functional approach to avoid data-dependent branching for torch.compile
    p_sum = p.sum(dim=-1, keepdim=True)
    zero_mass_mask = p_sum <= EPSILON_STABILITY

    # Always compute fallback, use where to select
    bin_indices = torch.clamp(
        ((y_target - vmin) / bin_width).long(), min=0, max=nbins - 1
    )
    one_hot = F.one_hot(bin_indices, num_classes=nbins).float()
    p = torch.where(zero_mass_mask, one_hot, p)

    # Renormalize to sum to 1
    p = p / p.sum(dim=-1, keepdim=True)

    # Cross-entropy with soft targets: CE = - sum_k [ p_k * log(q_k) ]
    log_probs = F.log_softmax(logits, dim=-1)  # shape: (..., nbins)
    ce_loss = -torch.sum(p * log_probs, dim=-1)
    return ce_loss


def _gather_time_function(
    time: torch.Tensor, bin_centers: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """
    Gather values at the nearest bin center for each time point.

    Args:
        time: Tensor of shape (T,) containing time points to evaluate
        bin_centers: Tensor of shape (..., B) containing the centers of the bins
        values: Tensor of shape (..., B) containing values at each bin center

    Returns:
        Tensor of shape (..., T) containing gathered values at nearest bin centers
    """
    assert time.dim() == 1, "time must be a 1D tensor"

    T = time.shape[0]
    prefix_shape = bin_centers.shape[:-1]
    time_expanded = (
        time.reshape((1,) * len(prefix_shape) + (T,))
        .expand(prefix_shape + (T,))
        .contiguous()
    )
    idx_right = torch.searchsorted(
        bin_centers, time_expanded, right=True
    )  # (..., T) in [0, B]
    B = bin_centers.shape[-1]
    idx_left = (idx_right - 1).clamp(min=0, max=B - 1)
    idx_right_clamped = idx_right.clamp(min=0, max=B - 1)

    # gather centers and probs (..., T)
    left_centers = torch.gather(bin_centers, -1, idx_left)
    right_centers = torch.gather(bin_centers, -1, idx_right_clamped)
    left_values = torch.gather(values, -1, idx_left)
    right_values = torch.gather(values, -1, idx_right_clamped)

    dist_left = (time_expanded - left_centers).abs()
    dist_right = (right_centers - time_expanded).abs()
    use_left = dist_left <= dist_right
    return torch.where(use_left, left_values, right_values)


def _gather_per_sample(
    time: torch.Tensor, bin_centers: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """
    Gather one value per sample at the nearest bin center.

    Args:
        time: Tensor of shape (...) containing one timestamp per prefix position.
        bin_centers: Tensor of shape (..., B) containing the centers of the bins
        values: Tensor of shape (..., B) containing per-bin values.

    Returns:
        Tensor of shape (...) with gathered values.
    """
    if values.shape != bin_centers.shape:
        raise ValueError("values and bin_centers must have identical shapes.")
    if tuple(time.shape) != tuple(values.shape[:-1]):
        raise ValueError(
            f"time must have shape {values.shape[:-1]}, got {tuple(time.shape)}"
        )

    B = values.shape[-1]
    idx_right = torch.searchsorted(
        bin_centers, time.unsqueeze(-1).contiguous(), right=True
    ).squeeze(-1)
    idx_left = (idx_right - 1).clamp(min=0, max=B - 1)
    idx_right_clamped = idx_right.clamp(min=0, max=B - 1)

    left_centers = torch.gather(bin_centers, -1, idx_left.unsqueeze(-1)).squeeze(-1)
    right_centers = torch.gather(
        bin_centers, -1, idx_right_clamped.unsqueeze(-1)
    ).squeeze(-1)

    dist_left = (time - left_centers).abs()
    dist_right = (right_centers - time).abs()
    use_left = dist_left <= dist_right
    nearest_idx = torch.where(use_left, idx_left, idx_right_clamped)

    return torch.gather(values, -1, nearest_idx.unsqueeze(-1)).squeeze(-1)


def _linear_interp_on_grid(
    x: torch.Tensor, y: torch.Tensor, xnew: torch.Tensor
) -> torch.Tensor:
    """
    Linear interpolation for values defined on a monotone time grid.

    Args:
        x: Tensor of shape (..., N), monotonically increasing along the last dimension
        y: Tensor of shape (..., N), values at x
        xnew: Tensor of shape (T,), query coordinates

    Returns:
        Tensor of shape (..., T)
    """
    if x.shape != y.shape:
        raise ValueError("x and y must have identical shapes.")
    if xnew.dim() != 1:
        raise ValueError("xnew must be a 1D tensor.")

    prefix_shape = x.shape[:-1]
    n_grid = x.shape[-1]
    n_query = xnew.shape[0]
    xnew_expanded = xnew.reshape((1,) * len(prefix_shape) + (n_query,)).expand(
        prefix_shape + (n_query,)
    )

    n_series = math.prod(prefix_shape) if len(prefix_shape) > 0 else 1
    x_2d = x.reshape(n_series, n_grid)
    y_2d = y.reshape(n_series, n_grid)
    xnew_2d = xnew_expanded.reshape(n_series, n_query)
    xnew_2d = torch.minimum(torch.maximum(xnew_2d, x_2d[:, :1]), x_2d[:, -1:])
    ynew_2d = _interp1d_per_batch(x_2d, y_2d, xnew_2d)
    return ynew_2d.reshape(prefix_shape + (n_query,))


def _linear_interp_per_sample(
    x: torch.Tensor, y: torch.Tensor, xnew: torch.Tensor
) -> torch.Tensor:
    """
    Linear interpolation for one query point per sample.

    Args:
        x: Tensor of shape (..., N), monotonically increasing along the last dimension
        y: Tensor of shape (..., N), values at x
        xnew: Tensor of shape (...,), one query value per prefix position

    Returns:
        Tensor of shape (...,)
    """
    if x.shape != y.shape:
        raise ValueError("x and y must have identical shapes.")
    if tuple(xnew.shape) != tuple(x.shape[:-1]):
        raise ValueError(
            f"xnew must have shape {x.shape[:-1]}, got {tuple(xnew.shape)}"
        )

    prefix_shape = x.shape[:-1]
    n_grid = x.shape[-1]
    n_series = math.prod(prefix_shape) if len(prefix_shape) > 0 else 1

    x_2d = x.reshape(n_series, n_grid)
    y_2d = y.reshape(n_series, n_grid)
    xnew_2d = xnew.reshape(n_series, 1)
    xnew_2d = torch.minimum(torch.maximum(xnew_2d, x_2d[:, :1]), x_2d[:, -1:])

    ynew_2d = _interp1d_per_batch(x_2d, y_2d, xnew_2d)

    return ynew_2d.reshape(prefix_shape)


class HistogramDistribution:
    def __init__(
        self, bin_edges: torch.Tensor, bin_centers: torch.Tensor, logits: torch.Tensor
    ):
        # NOTE: Here the last dim size are not matched, because we use the last logit for the residual mass
        # beyond vmax (which has no bin-center)

        self.bin_edges = bin_edges  # shape: (..., num_bins)
        self.bin_centers = bin_centers  # shape: (..., num_bins - 1)
        self.logits = logits  # shape: (..., num_bins)
        self.probs = torch.softmax(logits, dim=-1)  # shape: (..., num_bins)
        self.bin_widths = (
            self.bin_edges[..., 1:] - self.bin_edges[..., :-1]
        )  # shape: (..., num_bins - 1)
        self.density_bins = (
            self.probs[..., :-1] / self.bin_widths
        )  # shape: (..., num_bins - 1)

        survival_at_edges = 1.0 - torch.cumsum(
            self.probs, dim=-1
        )  # shape: (..., num_bins)
        self.survival_at_edges = torch.cat(
            [torch.ones_like(survival_at_edges[..., :1]), survival_at_edges[..., :-1]],
            dim=-1,
        )  # S(t) at left edge of each bin, shape: (..., num_bins)

        self.tmin = torch.min(self.bin_edges)
        self.tmax = torch.max(self.bin_edges)

    def density_function(self, time: torch.Tensor) -> torch.Tensor:
        # time shape: (T,), return shape: (..., T)
        result = _gather_time_function(time, self.bin_centers, self.density_bins)
        # The final logit is residual mass for [last_edge, inf), which has no
        # finite-width density estimate. Use a tiny density for NLL stability.
        time_view = time.reshape((1,) * (self.bin_edges.dim() - 1) + (time.shape[0],))
        outside_support = (time_view < 0.0) | (time_view >= self.bin_edges[..., -1:])
        result = torch.where(
            outside_support, torch.ones_like(result) * EPSILON_STABILITY, result
        )
        return result

    def survival_function(self, time: torch.Tensor) -> torch.Tensor:
        # time shape: (T,), return shape: (..., T)

        if self.tmin > 0:
            # add a point at (0, 1) to bin_edges and survival_at_edges
            bin_edges = torch.cat(
                [torch.zeros_like(self.bin_edges[..., :1]), self.bin_edges], dim=-1
            )
            survival_at_edges = torch.cat(
                [
                    torch.ones_like(self.survival_at_edges[..., :1]),
                    self.survival_at_edges,
                ],
                dim=-1,
            )
        else:
            bin_edges = self.bin_edges
            survival_at_edges = self.survival_at_edges

        survival = _linear_interp_on_grid(bin_edges, survival_at_edges, time)
        time_view = time.reshape((1,) * (self.bin_edges.dim() - 1) + (time.shape[0],))
        below_zero = time_view < 0.0
        in_residual_bin = time_view >= self.bin_edges[..., -1:]
        # Beyond the last finite edge, survival is the residual tail mass.
        survival = torch.where(
            in_residual_bin, self.survival_at_edges[..., -1:], survival
        )
        survival = torch.where(below_zero, torch.ones_like(survival), survival)

        return torch.clamp(survival, min=0.0, max=1.0)

    def density_at(self, time: torch.Tensor) -> torch.Tensor:
        # time shape: (...,), return shape: (...,)
        result = _gather_per_sample(time, self.bin_centers, self.density_bins)
        # The residual bin has infinite width, so event density there is
        # approximated by a tiny positive value instead of raising.
        outside_support = (time < 0.0) | (time >= self.bin_edges[..., -1])
        return torch.where(
            outside_support, torch.ones_like(result) * EPSILON_STABILITY, result
        )

    def survival_at(self, time: torch.Tensor) -> torch.Tensor:
        # time shape: (...,), return shape: (...,)
        if self.tmin > 0:
            # add a point at (0, 1) to bin_edges and survival_at_edges
            bin_edges = torch.cat(
                [torch.zeros_like(self.bin_edges[..., :1]), self.bin_edges], dim=-1
            )
            survival_at_edges = torch.cat(
                [
                    torch.ones_like(self.survival_at_edges[..., :1]),
                    self.survival_at_edges,
                ],
                dim=-1,
            )
        else:
            bin_edges = self.bin_edges
            survival_at_edges = self.survival_at_edges
        survival = _linear_interp_per_sample(bin_edges, survival_at_edges, time)
        below_zero = time < 0.0
        in_residual_bin = time >= self.bin_edges[..., -1]
        # Query times beyond the final finite edge live in the model's residual
        # bin, whose survival probability is the remaining tail mass.
        survival = torch.where(
            in_residual_bin, self.survival_at_edges[..., -1], survival
        )
        survival = torch.where(below_zero, torch.ones_like(survival), survival)
        return torch.clamp(survival, min=0.0, max=1.0)

    def rmst(self) -> torch.Tensor:
        # restricted mean survival time up to the last edge, using trapezoidal rule
        # return shape: (...,)
        cdf = torch.cumsum(self.probs, dim=-1)  # (..., num_bins)
        S = 1.0 - cdf  # (..., num_bins)
        # prepend S(0)=1, and remove the last S which corresponds to the residual mass beyond vmax
        S = torch.cat(
            [torch.ones_like(S[..., :1]), S[..., :-1]], dim=-1
        )  # (..., num_bins)
        rmst = torch.trapz(S, self.bin_edges, dim=-1)  # (...,)
        return rmst

    def median(self, extrapolation: bool = False) -> torch.Tensor:
        cdf = torch.cumsum(self.probs, dim=-1)  # (..., num_bins)
        num_bins = cdf.shape[-1]

        median_probs = cdf.new_full(cdf.shape[:-1] + (1,), 0.5)
        idx = torch.searchsorted(cdf, median_probs, side="left")  # (..., 1)
        in_residual_bin = idx.squeeze(-1) >= (num_bins - 1)

        # Interpolate in the finite bin where the CDF first crosses 0.5.
        finite_idx = idx.clamp(min=0, max=num_bins - 2)
        cdf_left = torch.cat(
            [torch.zeros_like(cdf[..., :1]), cdf[..., :-1]], dim=-1
        )  # (..., num_bins)
        cdf_lo = torch.gather(cdf_left, -1, finite_idx).squeeze(-1)
        cdf_hi = torch.gather(cdf, -1, finite_idx).squeeze(-1)

        left_edge = torch.gather(self.bin_edges, -1, finite_idx).squeeze(-1)
        right_edge = torch.gather(
            self.bin_edges,
            -1,
            (finite_idx + 1).clamp(max=self.bin_edges.shape[-1] - 1),
        ).squeeze(-1)

        cdf_span = torch.clamp(cdf_hi - cdf_lo, min=EPSILON_STABILITY)
        alpha = torch.clamp(
            (median_probs.squeeze(-1) - cdf_lo) / cdf_span, min=0.0, max=1.0
        )
        median_finite = left_edge + alpha * (right_edge - left_edge)

        inf_values = torch.full_like(median_finite, float("inf"))
        if extrapolation and torch.any(in_residual_bin):
            # Extrapolate with the line through (0, 0) and (last_bin_edge, CDF(last_bin_edge)).
            cdf_last_edge = cdf[..., -2]
            last_edge = self.bin_edges[..., -1]
            median_residual = torch.where(
                cdf_last_edge > EPSILON_STABILITY,
                0.5 * last_edge / cdf_last_edge,
                inf_values,
            )
        else:
            median_residual = inf_values

        return torch.where(in_residual_bin, median_residual, median_finite)

    def mode(self) -> torch.Tensor:
        # return shape: (...,)

        max_indices = torch.argmax(self.probs, dim=-1, keepdim=True)  # shape: (..., 1)
        # we do not know the bin center for the last bin (residual mass beyond vmax), putting inf here
        bin_centers_with_res = torch.cat(
            [
                self.bin_centers,
                torch.full_like(self.bin_centers[..., :1], float("inf")),
            ],
            dim=-1,
        )  # shape: (..., num_bins)

        return torch.gather(bin_centers_with_res, -1, max_indices).squeeze(
            -1
        )  # shape: (...,)

    def hazard_function(self, time: torch.Tensor) -> torch.Tensor:
        return self.density_function(time) / (
            self.survival_function(time) + EPSILON_STABILITY
        )

    def cumulative_hazard(self, time: torch.Tensor) -> torch.Tensor:
        return -torch.log(self.survival_function(time) + EPSILON_STABILITY)


def _interp1d_per_batch(
    x: torch.Tensor, y: torch.Tensor, xnew: torch.Tensor
) -> torch.Tensor:
    """
    Linear interpolation over one monotone grid per batch row.

    Args:
        x: Tensor of shape (B, K), monotonically increasing along K.
        y: Tensor of shape (B, K), values at each x-grid point.
        xnew: Tensor of shape (B, M), query points.

    Returns:
        Tensor of shape (B, M).
    """
    if x.shape != y.shape:
        raise ValueError("x and y must have identical shapes.")
    if x.dim() != 2 or xnew.dim() != 2 or x.shape[0] != xnew.shape[0]:
        raise ValueError("x, y, and xnew must have shapes (B, K), (B, K), and (B, M).")

    idx_right = torch.searchsorted(x.contiguous(), xnew.contiguous(), right=False)
    idx_right = idx_right.clamp(min=1, max=x.shape[-1] - 1)
    idx_left = idx_right - 1

    x_left = torch.gather(x, -1, idx_left)
    x_right = torch.gather(x, -1, idx_right)
    y_left = torch.gather(y, -1, idx_left)
    y_right = torch.gather(y, -1, idx_right)

    denom = torch.clamp(x_right - x_left, min=torch.finfo(x.dtype).eps)
    alpha = torch.clamp((xnew - x_left) / denom, min=0.0, max=1.0)
    return y_left + alpha * (y_right - y_left)


class QuantileTransformState(NamedTuple):
    time_grid: torch.Tensor
    quantile_grid: torch.Tensor
    lengths: torch.Tensor


def quantile_grids(time_context: torch.Tensor) -> QuantileTransformState:
    """
    Build per-context interpolation grids between observed time and ECDF quantile space.

    The grid is shared by event and censoring targets: it depends only on the
    observed context times. Duplicate context times are collapsed to one knot
    and assigned their right-continuous ECDF value, cumulative_count / n.
    """
    if time_context.dim() != 2:
        raise ValueError(
            f"time_context must have shape (batch_size, context_len), got {tuple(time_context.shape)}"
        )
    if time_context.shape[-1] < 1:
        raise ValueError("time_context must contain at least one context time.")

    batch_size, context_len = time_context.shape
    dtype = time_context.dtype
    device = time_context.device

    min_positive = time_context.new_tensor(
        max(EPSILON_STABILITY, torch.finfo(dtype).tiny)
    )
    sorted_context = torch.sort(
        torch.clamp(time_context, min=min_positive), dim=-1
    ).values

    run_end = torch.ones((batch_size, context_len), device=device, dtype=torch.bool)
    run_end[:, :-1] = sorted_context[:, :-1] != sorted_context[:, 1:]
    unique_slots = torch.cumsum(run_end.to(dtype=torch.long), dim=-1)
    scatter_idx = torch.where(run_end, unique_slots, torch.zeros_like(unique_slots))

    max_grid_len = context_len + 1
    time_grid = torch.zeros((batch_size, max_grid_len), device=device, dtype=dtype)
    quantile_grid = torch.zeros((batch_size, max_grid_len), device=device, dtype=dtype)

    ranks = torch.arange(1, context_len + 1, device=device, dtype=dtype).view(
        1, context_len
    )
    right_continuous_ecdf = ranks / float(context_len)
    time_grid.scatter_(
        dim=1,
        index=scatter_idx,
        src=torch.where(run_end, sorted_context, torch.zeros_like(sorted_context)),
    )
    quantile_grid.scatter_(
        dim=1,
        index=scatter_idx,
        src=torch.where(
            run_end,
            right_continuous_ecdf.expand_as(sorted_context),
            torch.zeros_like(sorted_context),
        ),
    )

    unique_counts = run_end.sum(dim=-1)
    lengths = unique_counts + 1
    positions = torch.arange(max_grid_len, device=device).view(1, max_grid_len)
    pad_mask = positions >= lengths.unsqueeze(-1)
    max_time = sorted_context[:, -1:]
    time_grid = torch.where(pad_mask, max_time.expand_as(time_grid), time_grid)
    quantile_grid = torch.where(pad_mask, torch.ones_like(quantile_grid), quantile_grid)

    return QuantileTransformState(
        time_grid=time_grid, quantile_grid=quantile_grid, lengths=lengths
    )


def quantile_transform(
    samples: torch.Tensor,
    state: QuantileTransformState,
    type: Literal["time2quantile", "quantile2time"],
) -> torch.Tensor:
    """
    Transform times to/from a precomputed context-fitted ECDF quantile scale.

    time2quantile maps non-negative times into [0, 1]. Times beyond the largest
    context time map to 1. quantile2time maps [0, 1] back to finite times, with
    q=1 returning the largest context time; the model's final histogram logit
    represents residual raw-time mass beyond that point.
    """
    assert type in ["time2quantile", "quantile2time"], (
        "type must be 'time2quantile' or 'quantile2time'"
    )

    time_grid = state.time_grid
    quantile_grid = state.quantile_grid
    if time_grid.shape != quantile_grid.shape:
        raise ValueError("time_grid and quantile_grid must have identical shapes.")
    if samples.shape[0] != time_grid.shape[0]:
        raise ValueError(
            "samples and precomputed grids must share the batch dimension, "
            f"got {samples.shape[0]} and {time_grid.shape[0]}"
        )

    batch_size = samples.shape[0]
    samples_flat = samples.reshape(batch_size, -1)
    max_time = time_grid[..., -1:]

    if type == "time2quantile":
        samples_flat = torch.clamp(samples_flat, min=0.0)
        transformed = _interp1d_per_batch(time_grid, quantile_grid, samples_flat)
        transformed = torch.where(
            samples_flat >= max_time, torch.ones_like(transformed), transformed
        )
        transformed = torch.where(
            samples_flat <= 0.0, torch.zeros_like(transformed), transformed
        )
    else:
        samples_flat = torch.clamp(samples_flat, min=0.0, max=1.0)
        transformed = _interp1d_per_batch(quantile_grid, time_grid, samples_flat)
        transformed = torch.where(
            samples_flat >= 1.0, max_time.expand_as(transformed), transformed
        )
        transformed = torch.where(
            samples_flat <= 0.0, torch.zeros_like(transformed), transformed
        )

    return transformed.reshape(samples.shape)


def lognormal_transform(
    samples: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    type: Literal["lognormal2normal", "normal2lognormal"],
):
    """
    Transform between standard normal and log-normal distributions using PyTorch tensors.

    normal2lognormal: Transforms from standard normal N(0,1) to log-normal with mean, and std
    lognormal2normal: Transforms from log-normal with mean, and std back to standard normal N(0,1)

    Parameters:
    -----------
    samples : torch.Tensor of shape (..., n_samples)
        Input values to transform.
    mean : float or torch.Tensor of shape (...)
        Target mean of the log-normal distribution (must be positive)
    std : float or torch.Tensor of shape (...)
        Target standard deviation of the log-normal distribution (must be positive)
    type : str
        Type of transformation: "lognormal2normal" or "normal2lognormal"

    Returns:
    --------
    torch.Tensor: Transformed values with same shape as input samples

    Raises:
    -------
    ValueError: If mean <= 0 or std <= 0, or if input values are non-positive for lognormal2normal transformation

    Notes:
    -------
    Suppose Z ~ N(0,1) and X = exp(μ + σ Z) ~ LogNormal('mean', 'std').
    Then, according to (https://en.wikipedia.org/wiki/Log-normal_distribution), we have:
    μ = ln ('mean' ** 2 / sqrt('std' ** 2 + 'mean' ** 2)), and σ = sqrt(ln(1 + ('std' / 'mean') ** 2)).

    Therefore, to transform X to Z (lognormal2normal):
    Z = (ln(X) - μ) / σ
    and to transform Z to X (normal2lognormal):
    X = exp(μ + σ Z)
    """

    # Input validation - use clamp instead of raising for torch.compile compatibility
    # Raising errors causes graph breaks, so we clamp to valid range instead
    mean = torch.clamp(mean, min=EPSILON_STABILITY)
    std = torch.clamp(std, min=EPSILON_STABILITY)

    assert mean.shape == std.shape == samples.shape[:-1], (
        f"mean and std must have the same shape as samples except for the last dimension, got {mean.shape}, {std.shape}, {samples.shape}"
    )

    assert type in ["lognormal2normal", "normal2lognormal"], (
        "type must be 'lognormal2normal' or 'normal2lognormal'"
    )

    mean = mean.unsqueeze(-1)  # Add dimension for n_samples:
    std = std.unsqueeze(-1)  # Add dimension for n_samples

    # Get dtype-dependent safe constants
    finfo = torch.finfo(samples.dtype)
    # Smallest positive normal number (strictly > 0); using it avoids log(0) and zero-division
    tiny = finfo.tiny
    # Lower bound sigma to prevent division by zero; choose sqrt(tiny) so that sigma^2 >= tiny
    min_sigma = torch.tensor(tiny, device=samples.device, dtype=samples.dtype).sqrt()

    # Clamp mean/std to be at least tiny to avoid log(0) due to potential subnormal underflow
    mean_safe = torch.clamp(mean, min=tiny)
    std_safe = torch.clamp(std, min=tiny)

    # Compute log parameters (mu, sigma) of the underlying normal in a numerically stable way:
    # sigma^2 = ln(1 + (std/mean)^2) = logaddexp(0, 2*ln(std) - 2*ln(mean))
    # Using logs avoids overflow in (std/mean)^2 for extreme ratios and underflow for tiny ratios.
    log_std = torch.log(
        mean_safe.new_tensor(1.0)
    )  # placeholder to ensure correct dtype/device
    log_std = torch.log(std_safe)  # ln(std)
    log_mean = torch.log(mean_safe)  # ln(mean)
    two_log_r = 2.0 * (log_std - log_mean)  # 2*ln(std/mean)

    # logaddexp(0, a) computes ln(exp(0) + exp(a)) = ln(1 + exp(a)) stably for all 'a'
    zero = torch.zeros((), device=samples.device, dtype=samples.dtype)
    sigma2 = torch.logaddexp(zero, two_log_r)  # sigma^2 >= 0 by construction

    # Guard against tiny negative due to numerical noise (shouldn't happen but defensive)
    sigma2 = torch.clamp(sigma2, min=0.0)

    # Compute sigma; clamp to min_sigma to avoid division by zero in the inverse transform
    sigma = torch.sqrt(sigma2)
    sigma = torch.clamp(sigma, min=min_sigma)  # ensure strictly positive sigma

    # Compute mu using the stable identity: mu = ln(mean) - 0.5 * sigma^2
    mu = log_mean - 0.5 * sigma2

    if type == "normal2lognormal":
        # Transform Z -> X = exp(mu + sigma * Z)
        # Clamp exponent to avoid exp overflow/underflow to inf/0 while keeping X strictly positive.
        # Using finfo.max and finfo.tiny ensures exp(exponent_clamped) is within [tiny, max].
        log_max = torch.log(
            torch.tensor(finfo.max, device=samples.device, dtype=samples.dtype)
        )
        log_min = torch.log(
            torch.tensor(finfo.tiny, device=samples.device, dtype=samples.dtype)
        )

        # Compute exponent; no risk of NaN/Inf here as inputs were finite and sigma is clamped
        exponent = mu + sigma * samples

        # Clamp exponent to [log_min, log_max] so exp() cannot overflow to inf or underflow to 0
        exponent_clamped = torch.clamp(exponent, min=log_min, max=log_max)

        # Safe exponential; result is in (0, +inf) but bounded by [tiny, finfo.max] -> finite, non-zero
        x = torch.exp(exponent_clamped)
        return x

    else:  # type == "lognormal2normal"
        # For inverse transform, inputs must be strictly positive (log requires X > 0)
        # Clamp samples instead of raising for torch.compile compatibility
        samples = torch.clamp(samples, min=tiny)

        # Compute Z = (ln(X) - mu) / sigma with safeguards:
        # - ln(X) is safe since X > 0 (clamped)
        # - sigma is clamped to be strictly positive to avoid division by zero
        log_x = torch.log(samples)
        z = (log_x - mu) / sigma  # sigma > 0 by clamping above
        return z
