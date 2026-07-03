import math
from typing import Tuple

import torch

from survivalpfn.models.constants import EPSILON_STABILITY


class PriorGenerationError(Exception):
    pass


class MetaBetaSampler:
    """Samples noise from a truncated normal distribution upon calling"""

    def __init__(self, scale: float, min: float, max: float):
        self.scale = scale
        self.b_sampler = UniformSampler(min, max)
        self.k_sampler = UniformSampler(min, max)

    def __call__(self, shape: torch.Size | None = None):
        b = self.b_sampler(shape)
        k = self.k_sampler(shape)
        return self.scale * BetaSampler(b, k)(shape)


class DeepTruncNormLogScaledSampler:
    """Samples noise from a truncated normal distribution upon calling"""

    def __init__(self, mean_min, mean_max, is_int, min_val):
        self.mean_min = mean_min
        self.mean_max = mean_max
        self.is_int = is_int
        self.min_val = min_val

    def __call__(self, shape: torch.Size | None = None):
        log_min = math.log(self.mean_min)
        log_max = math.log(self.mean_max)
        size = shape or (1,)
        sigma = torch.exp((log_max - log_min) * torch.rand(size) + log_min)
        mu = torch.exp((log_max - log_min) * torch.rand(size) + log_min)

        loi = torch.distributions.Normal(mu, sigma)
        round = (lambda x: torch.round(x).int()) if self.is_int else (lambda x: x)
        samples = round(
            loi.icdf((torch.rand(size) - 1) * (1 - loi.cdf(torch.zeros(size))) + 1)
            + self.min_val
        )
        samples = torch.where(
            samples > self.min_val, samples, self.min_val * torch.ones_like(samples)
        )
        samples = samples if shape is not None else samples.item()
        return samples


class LaplaceSampler:
    """Samples noise from a laplace distribution upon calling"""

    def __init__(self, loc: float, scale: float):
        self.distr = torch.distributions.Laplace(loc, scale)

    def __call__(self, shape: torch.Size | None = None) -> torch.Tensor:
        if shape is None:
            return self.distr.sample().item()
        else:
            return self.distr.sample(shape)


class UniformSampler:
    """Samples noise from a laplace distribution upon calling"""

    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def __call__(self, shape: torch.Size | None = None) -> torch.Tensor:
        if shape is None:
            ret = torch.rand((1,)).item()
        else:
            ret = torch.rand(shape)
        return ret * (self.high - self.low) + self.low


class BetaSampler:
    """Samples noise from a beta distribution upon calling"""

    def __init__(self, b: float, k: float):
        self.distr = torch.distributions.Beta(b, k)

    def __call__(self, shape: torch.Size | None = None) -> torch.Tensor | float:
        if shape is None:
            return self.distr.sample().item()
        else:
            return self.distr.sample(shape)


class GaussianSampler:
    """Samples noise from a laplace distribution upon calling"""

    def __init__(self, loc: float, scale: float):
        self.distr = torch.distributions.Normal(loc, scale)

    def __call__(self, shape: torch.Size | None = None) -> torch.Tensor | float:
        if shape is None:
            return self.distr.sample().item()
        else:
            return self.distr.sample(shape)


class BernoulliSampler:
    """Samples from a boolean Bernoulli distribution upon calling"""

    def __init__(self, prob: float):
        self.distr = torch.distributions.Bernoulli(prob)

    def __call__(self, shape: torch.Size | None = None) -> torch.Tensor | bool:
        if shape is None:
            return self.distr.sample().bool().item()
        else:
            return self.distr.sample(shape).bool()


class UniformIntegerSampler:
    """Samples noise from a laplace distribution upon calling"""

    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high

    def __call__(self, shape: torch.Size | None = None) -> torch.Tensor | int:
        if shape is None:
            ret = torch.randint(self.low, self.high + 1, (1,)).item()
        else:
            ret = torch.randint(self.low, self.high + 1, shape)
        return ret


### Categorical ###
def num2cat(
    numerical_tensor: torch.Tensor, max_categories: int, ordered_p: float
) -> torch.Tensor:
    """
    Convert continuous numeric data into categorical data with optional randomization.

    This function transforms a continuous numeric tensor into a categorical tensor
    by creating class boundaries. It can also randomize the ordering of categories
    based on the provided probability parameter.

    Args:
        numerical_tensor: Input tensor with continuous numeric values, shape: (seq_len,)
        max_categories: Maximum number of categories to create
        ordered_p: Probability to keep categories ordered (vs randomized) Higher values mean categories are more likely
                  to remain ordered

    Returns:
        Categorical tensor as float values
    """

    if numerical_tensor.ndim != 1:
        raise PriorGenerationError("Input tensor must be 1-dimensional")

    categorical = torch.empty_like(numerical_tensor)
    num_classes = torch.randint(
        2, max(3, max_categories + 1), (1,), device=numerical_tensor.device
    ).item()

    if torch.rand(1) > 0.5:
        numerical_tensor = -numerical_tensor

    sorted_num, categorical = torch.unique(numerical_tensor, return_inverse=True)
    if len(sorted_num) >= num_classes + 1:
        # Randomly select boundary values
        boundary_indices = torch.randperm(
            len(sorted_num) - 2, device=numerical_tensor.device
        )[: num_classes - 1]
        cls_bdry: torch.Tensor = sorted_num[1:-1][boundary_indices]

        # Convert to categorical by counting how many boundaries each value exceeds
        reshaped_num = numerical_tensor.reshape((-1, 1))  # shape: (seq_len, 1)
        reshaped_bdry = cls_bdry.reshape(1, -1)  # shape: (1, num_classes - 1)
        categorical = (reshaped_num > reshaped_bdry).sum(dim=1)  # shape: (seq_len,)

    if torch.rand(1) > ordered_p:
        classes = torch.arange(0, num_classes, device=categorical.device)
        random_classes = torch.randperm(num_classes, device=categorical.device).type(
            categorical.type()
        )
        categorical = ((categorical.unsqueeze(-1) == classes) * random_classes).sum(-1)

    return categorical.float()


def _compute_scaling_factor(
    ratios: torch.Tensor, m: int, n: int, eps: float = 1e-8, greater: bool = True
) -> float:
    """
    Compute the scaling factor k so that exactly m out of n elements satisfy:
        - if greater=True: k > ratios_i for m elements
        - if greater=False: k < ratios_i for m elements

    Args:
        ratios: Tensor of ratios
        m: Number of elements to satisfy the condition
        n: Total number of elements
        eps: Small epsilon to avoid zero scaling
        greater: Whether the condition is k > ratios_i (True) or k < ratios_i (False)

    Returns:
        Scaling factor k
    """
    r_sorted, _ = torch.sort(ratios)

    if m <= 0:
        # want 0 satisfying the condition
        return (
            max(r_sorted[0].item() * (0.5 if greater else 1.0 + 1e-6), eps)
            if greater
            else r_sorted[-1].item() * (1.0 + 1e-6)
        )
    elif m >= n:
        # want all satisfying the condition
        return (
            r_sorted[-1].item() * (1.0 + 1e-6)
            if greater
            else max(r_sorted[0].item() * 0.5, eps)
        )
    else:
        a = r_sorted[m - 1].item()
        b = r_sorted[m].item()
        # Geometric mean keeps ratio scale stable
        if greater:
            return (a * b) ** 0.5 if a < b else b * (1.0 + 1e-6)
        else:
            k = (a * b) ** 0.5 if a < b else b * (1.0 - 1e-6)
            return max(k, eps)


def fix_censoring_rate_pos(
    censoring: torch.Tensor,
    events: torch.Tensor,
    alpha: float,
    scale_which: str = "events",
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    This function tweaks the event and censoring times to achieve a desired censoring rate alpha.
    The input event and censoring times must be strictly positive.

    Args:
        censoring: Censoring times tensor
        events: Event times tensor
        alpha: Desired censoring rate in [0, 1]
        scale_which: Which times to scale ("events" or "censoring")
        eps: Small epsilon to avoid division by zero

    Returns:
        Tuple of (adjusted_censoring, adjusted_events)
    """
    e = events.reshape(-1)
    c = censoring.reshape(-1)
    n = e.numel()
    assert n == c.numel() and n > 0, (
        "events and censoring must have the same number of elements"
    )
    assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]"
    assert torch.all(e > 0), "All event times must be strictly positive"
    assert torch.all(c > 0), "All censoring times must be strictly positive"

    m = int(torch.floor(torch.tensor(alpha * n, dtype=torch.float64)).item())

    if scale_which == "events":
        ratios = c / e
        k = _compute_scaling_factor(ratios, m, n, eps, greater=True)
        e_new = e * k
        c_new = c
    elif scale_which == "censoring":
        ratios = e / c
        k = _compute_scaling_factor(ratios, m, n, eps, greater=False)
        e_new = e
        c_new = c * k
    else:
        raise ValueError("scale_which must be 'events' or 'censoring'.")

    return c_new.reshape_as(censoring), e_new.reshape_as(events)


def fix_censoring_rate(
    censoring: torch.Tensor, events: torch.Tensor, desired_censoring_rate: float
) -> tuple[torch.Tensor, torch.Tensor]:
    # Compute deltas = (censoring - event). For a shift s applied to events:
    # censoring_rate(s) = 1 - (# of deltas > s) / n
    deltas = (censoring - events).reshape(-1)
    deltas_sorted, _ = torch.sort(deltas)
    n = deltas_sorted.numel()

    # Target number of censored samples
    target_censored = int(math.floor(desired_censoring_rate * n))

    # Choose shift so that exactly target_censored events are censored
    if target_censored <= 0:
        shift = deltas_sorted[0].item() - EPSILON_STABILITY
    elif target_censored >= n:
        shift = deltas_sorted[-1].item() + EPSILON_STABILITY
    else:
        a = deltas_sorted[target_censored - 1].item()
        b = deltas_sorted[target_censored].item()
        shift = 0.5 * (a + b) if a < b else b - EPSILON_STABILITY

    # Apply the shift to events
    events_shifted = events + shift
    censoring_shifted = censoring

    # Ensure all times are nonnegative by shifting both upward if necessary
    min_time = min(events_shifted.min().item(), censoring_shifted.min().item())
    if min_time < 0:
        offset = -min_time
        events_shifted = events_shifted + offset
        censoring_shifted = censoring_shifted + offset

    return censoring_shifted, events_shifted


def delete_unique_features(
    X: torch.Tensor, d: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Removes features that have only one unique value across all samples.

    Single-value features provide no useful information for learning since they
    have zero variance. This method identifies and removes such constant features
    to improve model training efficiency and stability. The removed features are
    replaced with zero padding to maintain tensor dimensions.

    Parameters
    ----------
    X : Tensor
        Input features tensor of shape (B, T, H) where:
        - B is batch size
        - T is sequence length
        - H is feature dimensionality

    d : Tensor
        Number of features per dataset of shape (B,), indicating how many
        features are actually used in each dataset (rest is padding)

    Returns
    -------
    tuple
        (X_new, d_new) where:
        - X_new is the filtered tensor with non-informative features removed
        - d_new is the updated feature count per dataset
    """

    def filter_unique_features(
        xi: torch.Tensor, di: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Filters features with only one unique value from a single dataset."""
        num_features = xi.shape[-1]
        # Only consider actual features (up to di, ignoring padding)
        xi = xi[:, :di]
        # Identify features with more than one unique value (informative features)
        unique_mask = [len(torch.unique(xi[:, j])) > 1 for j in range(di)]
        di_new = sum(unique_mask)
        # Create new tensor with only informative features, padding the rest
        xi_new = torch.nn.functional.pad(
            xi[:, unique_mask], pad=(0, num_features - di_new), mode="constant", value=0
        )
        return xi_new, torch.tensor(di_new, device=xi.device)

    # Process each dataset in the batch independently
    filtered_results = [filter_unique_features(xi, di) for xi, di in zip(X, d)]
    X_new, d_new = [torch.stack(res) for res in zip(*filtered_results)]

    return X_new, d_new


def monotone_bernstein_map(c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Evaluate a smooth, monotone map f:[0,1]->[0,1] from unconstrained parameters c
    using a Bernstein/Bezier representation with de Casteljau's algorithm.

    Args:
        c: Tensor of shape (..., K) with unconstrained parameters (e.g., Normal).
          K >= 1 is the degree of the polynomial (n = K).
        x: Tensor of shape broadcastable to (...,) with inputs in [0, 1].

    Returns:
        y: Tensor of shape broadcasted to x and c batch shape (...,) with values in [0,1].

    Notes:
        - Monotonicity is enforced by constructing nonnegative "increments" via softmax.
        - Control points b_0..b_n (n=K) satisfy 0 = b_0 <= ... <= b_n = 1.
        - The function is strictly increasing almost everywhere due to strictly positive increments.
    """
    if c.size(-1) < 1:
        raise ValueError("c must have last dimension K >= 1")

    # Positive increments that sum to 1
    deltas = torch.nn.functional.softmax(c, dim=-1)  # (..., K)
    b_cum = torch.cumsum(deltas, dim=-1)  # (..., K)
    b0 = torch.zeros_like(b_cum[..., :1])  # (..., 1)
    b = torch.cat([b0, b_cum], dim=-1)  # (..., K+1) control points in [0,1]

    # Vectorized Bernstein basis evaluation (equivalent to de Casteljau), no Python loop
    t = x.unsqueeze(-1)  # (..., 1)
    n = b.size(-1) - 1  # degree
    one_minus_t = 1.0 - t

    # Indices 0..n for basis terms
    i = torch.arange(n + 1, device=c.device, dtype=c.dtype)  # (n+1,)
    n_f = torch.tensor(float(n), device=c.device, dtype=c.dtype)

    # Binomial coefficients comb(n, i) computed via lgamma for numerical stability
    lnC = (
        torch.lgamma(n_f + 1.0) - torch.lgamma(i + 1.0) - torch.lgamma((n_f - i) + 1.0)
    )  # (n+1,)
    C = torch.exp(lnC)  # (n+1,)

    # Bernstein basis: comb(n,i) * t^i * (1-t)^(n-i), all broadcasted over batch dims
    t_pows = t**i  # (..., n+1)
    om_pows = one_minus_t ** (n_f - i)  # (..., n+1)
    bern = C * t_pows * om_pows  # (..., n+1)

    # Bezier evaluation: y = sum_i bern_i * b_i
    y = torch.sum(bern * b, dim=-1)

    # Clamp only to guard against tiny numerical excursions outside [0,1]
    return y.clamp(0.0, 1.0)
