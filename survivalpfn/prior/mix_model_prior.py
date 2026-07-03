from typing import Tuple

import torch

from survivalpfn.models.constants import DIST_CHOICES, EPSILON_STABILITY, K_CHOICES

from .base_survival_prior import BaseSurvivalPrior
from .table_generators import TableGenerator
from .utils import fix_censoring_rate_pos


class MixtureModelSurvivalPrior(BaseSurvivalPrior):
    """
    A data generating process to sample covariates + event/censoring times for survival analysis.

    This prior generates first sample a covariate table using a provided table generator.

    The event and censoring times are then generated using a mixture model approach.
    1. First, we generate parameters (shape, scale, mixture weights) of a mixture model conditioned on the covariates,
        using a provided table generator.
    2. Next, we sample event and censoring times from the generated mixture model parameters.

    The censoring rate is then adjusted to achieve a desired target rate using a shifting heuristic.
    """

    def __init__(
        self,
        covariates_generator: TableGenerator,
        event_censoring_generator: TableGenerator,
        max_n_covariates: int,
        layer_norm_covariates_prob: float,
        n_samples: int,
        device: str,
        k: int = None,
        dist: str = None,
        desired_censoring_rate: float = None,
        censoring_type: str = None,
        scale_which: str = "events",
        outlier_removal_quantile: float = 0.90,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialize the mixture model survival prior.

        Args:
            covariates_generator: Generator for sampling covariate tables
            event_censoring_generator: Generator for sampling event and censoring times
            k: the number of mixture components
            dist: the type of distribution to sample from (e.g., "Weibull", "LogNormal")
            max_n_covariates: Maximum number of covariates to generate
            layer_norm_covariates_prob: Probability of applying layer normalization to covariates
            n_samples: Number of samples to generate per dataset
            device: Device to generate data on (e.g., 'cuda', 'cpu')
            desired_censoring_rate: Target censoring rate (if None, randomly sampled)
            scale_which: Which times to scale to adjust censoring rate ("events" or "censoring")
            outlier_removal_quantile: Quantile threshold for outlier removal in event and censoring times
            verbose: Whether to print verbose logs
            *args, **kwargs: Additional arguments passed to BaseSurvivalPrior
        """
        super().__init__(
            covariates_generator=covariates_generator,
            max_n_covariates=max_n_covariates,
            layer_norm_covariates_prob=layer_norm_covariates_prob,
            n_samples=n_samples,
            device=device,
            desired_censoring_rate=desired_censoring_rate,
            censoring_type=censoring_type,
            *args,
            **kwargs,
        )
        self.event_censoring_generator = event_censoring_generator
        self.scale_which = scale_which
        self.outlier_removal_quantile = outlier_removal_quantile
        self.verbose = verbose

        if k is not None:
            assert k > 0, "k must be a positive integer"
            self.k_fn = lambda: k
        else:
            self.k_fn = lambda: K_CHOICES[
                torch.randint(len(K_CHOICES), (1,), device=self.device).item()
            ]

        if dist is not None:
            assert dist in ["Weibull", "LogNormal"], (
                "dist must be either 'Weibull' or 'LogNormal'"
            )
            self.dist_fn = lambda: dist
        else:
            self.dist_fn = lambda: DIST_CHOICES[
                torch.randint(len(DIST_CHOICES), (1,), device=self.device).item()
            ]

    def _generate_event_and_censoring_times(
        self, covariates: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Generate event and censoring times using a mixture model.

        Args:
            covariates: Covariate tensor of shape (n_samples, n_covariates)

        Returns:
            Tuple of (events, censoring, censoring_type) where events and censoring
            are tensors of shape (n_samples, 1) and censoring_type is a string.
        """
        k = self.k_fn()
        dist = self.dist_fn()
        censoring_type = self.censoring_type_fn()

        if censoring_type in ["Uniform", "Tab", "Admin"]:
            # generate event times
            # for each mixture component, we have three parameters: weight, shape, scale
            n_hidden_params = k * 3  # 3 params per mixture component
            hid_params = self.event_censoring_generator.sample_conditional_table(
                input=covariates, n_columns=n_hidden_params
            )

            # reshape to (n_samples, k * 3)
            hid_params = hid_params.view(hid_params.shape[0], k * 3)

            param1 = hid_params[:, 0::3]  # shape parameter > 0
            param2 = hid_params[:, 1::3]  # scale parameter > 0
            weights = torch.nn.functional.softmax(hid_params[:, 2::3], dim=-1)

            if dist == "Weibull":
                events = sample_weibull(weights, param1, param2)
            elif dist == "LogNormal":
                events = sample_lognormal(weights, param1, param2)
            else:
                raise ValueError(f"Unsupported distribution type: {dist}")

            # generate censoring times
            if censoring_type == "Uniform":
                # uniform censoring between [min_event_time, max_event_time]
                min_event_time = torch.min(events)
                max_event_time = torch.max(events)
                censoring = (max_event_time - min_event_time) * torch.rand_like(
                    events
                ) + min_event_time
            elif censoring_type in ["Tab", "Admin"]:
                hid_params = self.covariates_generator.sample_table(
                    n_samples=self.n_samples, n_columns=n_hidden_params
                )

                # reshape to (n_samples, k * 3)
                hid_params = hid_params.view(hid_params.shape[0], k * 3)
                param1_censoring = hid_params[:, 0::3]  # shape parameter > 0
                param2_censoring = hid_params[:, 1::3]  # scale parameter > 0
                weights_censoring = torch.nn.functional.softmax(
                    hid_params[:, 2::3], dim=-1
                )

                if dist == "Weibull":
                    times = sample_weibull(
                        weights_censoring, param1_censoring, param2_censoring
                    )
                elif dist == "LogNormal":
                    times = sample_lognormal(
                        weights_censoring, param1_censoring, param2_censoring
                    )

                if censoring_type == "Tab":
                    censoring = times
                elif censoring_type == "Admin":
                    entry_dates = times
                    # shift it to be all positive
                    entry_dates = (
                        entry_dates - torch.min(entry_dates) + EPSILON_STABILITY
                    )
                    # residual sampled from a distribution, e.g, uniform between [0, max_enter_time]
                    res = torch.rand(1, device=events.device) * torch.max(entry_dates)
                    # fixed administrative censoring time
                    admin_date = torch.max(entry_dates) + res
                    censoring = admin_date - entry_dates
            else:
                raise ValueError(f"Unsupported censoring type: {censoring_type}")
        elif censoring_type == "Cond Ind":
            # for each mixture component, we have three parameters: weight, shape, scale
            n_hidden_params = (
                k * 3 * 2
            )  # 3 params per mixture component, 2 for event and censoring
            ind_struc = (
                [k * 3] * 2
            )  # the hidden params for event are dependent, same for censoring, but event and censoring are independent
            hid_params = self.event_censoring_generator.sample_conditional_table(
                input=covariates,
                n_columns=n_hidden_params,
                conditional_independence=True,
                indep_structure=ind_struc,
            )

            # reshape to (n_samples, 2, k * 3)
            hid_params = hid_params.view(hid_params.shape[0], k * 3, 2).permute(0, 2, 1)

            param1 = hid_params[:, :, 0::3]  # shape parameter > 0
            param2 = hid_params[:, :, 1::3]  # scale parameter > 0
            weights = torch.nn.functional.softmax(hid_params[:, :, 2::3], dim=-1)

            if dist == "Weibull":
                events = sample_weibull(
                    weights[:, 0, :], param1[:, 0, :], param2[:, 0, :]
                )
                censoring = sample_weibull(
                    weights[:, 1, :], param1[:, 1, :], param2[:, 1, :]
                )
            elif dist == "LogNormal":
                events = sample_lognormal(
                    weights[:, 0, :], param1[:, 0, :], param2[:, 0, :]
                )
                censoring = sample_lognormal(
                    weights[:, 1, :], param1[:, 1, :], param2[:, 1, :]
                )
            else:
                raise ValueError(f"Unsupported distribution type: {dist}")
        assert torch.all(events > 0), "All event times must be strictly positive"
        assert torch.all(censoring > 0), "All censoring times must be strictly positive"
        return events, censoring, censoring_type

    def _adjust_censoring_rate(
        self, events: torch.Tensor, censoring: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Adjust censoring times to achieve the desired censoring rate.

        Args:
            events: Event times tensor
            censoring: Censoring times tensor

        Returns:
            Tuple of (adjusted_censoring, adjusted_events)
        """
        # Sample censoring rate if not specified
        desired_censoring_rate = self.desired_censoring_rate_fn()

        C, E = fix_censoring_rate_pos(
            censoring=censoring,
            events=events,
            alpha=desired_censoring_rate,
            scale_which=self.scale_which,
            eps=EPSILON_STABILITY,
        )
        return C, E, desired_censoring_rate

    @torch.no_grad()
    def get_sample(self) -> dict[str, torch.Tensor]:
        """
        Generate a single survival analysis dataset.

        Returns:
            Dictionary containing:
            - X: Covariates (n_samples, n_covariates)
            - T: Observed survival times (n_samples,)
            - delta: Event indicators (n_samples,)
            - E: True event times (n_samples,)
            - C: True censoring times (n_samples,)
        """
        # Step 1: Generate covariates
        covariates = self._generate_covariates()

        # Step 2: Generate event and censoring times (subclass-specific)
        # This is different from the base class to avoid inf values
        events, censoring, censoring_type = self._generate_event_and_censoring_times(
            covariates
        )

        # Step 3: Special treatment for right-tail outliers for both event and censoring times
        event_threshold = torch.quantile(events, self.outlier_removal_quantile)
        censoring_threshold = torch.quantile(censoring, self.outlier_removal_quantile)
        if self.verbose:
            # Count how many would have been removed
            removed_event = (events > event_threshold).sum().item()
            removed_censoring = (censoring > censoring_threshold).sum().item()
            print(
                f"[Info] Truncated {removed_event} event times and "
                f"{removed_censoring} censoring times above "
                f"{self.outlier_removal_quantile * 100}th percentile."
            )
        # Instead of removing samples, clamp/truncate them
        events = torch.minimum(events, event_threshold)
        censoring = torch.minimum(censoring, censoring_threshold)

        # Step 4: Adjust censoring rate
        C, E, censoring_rate = self._adjust_censoring_rate(events, censoring)

        # Step 5: Compute observed times and indicators
        T, delta = self._compute_observed_times(E, C)

        # Step 6: Validate output
        self._validate_output(T, delta, covariates)

        return dict(
            X=covariates,
            T=T.squeeze(),  # Observed survival time
            delta=delta.squeeze(),  # Event indicator (1=event, 0=censored)
            E=E.squeeze(),  # True event time
            C=C.squeeze(),  # True censoring time
            censoring_rate=censoring_rate,
            censoring_type=censoring_type,
        )


def sample_weibull(
    weights: torch.Tensor, shape: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """
    Sample from a mixture of Weibull distributions.

    Args:
        weights: Mixture weights tensor of shape (n_samples, k)
        shape: Shape parameters (k) tensor of shape (n_samples, k)
        scale: Scale parameters (lambda) tensor of shape (n_samples, k)
    Returns:
        Sampled times tensor of shape (n_samples, 1)
    """
    # Transform parameters to ensure they are strictly positive.
    # Softplus ensures smooth gradients during training, and the +0.1 offset
    # prevents parameters from being too close to zero, which could cause
    # numerical instability in the inverse transform sampling (power operations).
    shape = torch.nn.functional.softplus(shape) + 0.1
    scale = torch.nn.functional.softplus(scale) + 0.1

    n_samples = weights.shape[0]
    # Sample mixture component indices
    mixture_indices = torch.multinomial(weights, num_samples=1).squeeze(-1)

    # Gather the corresponding shape and scale parameters
    selected_shape = shape[torch.arange(n_samples), mixture_indices]
    selected_scale = scale[torch.arange(n_samples), mixture_indices]

    # Sample from the selected Weibull distributions using inverse transform sampling
    eps_temp = 1e-6
    u = eps_temp + (1 - 2 * eps_temp) * torch.rand(n_samples, device=weights.device)
    sampled_times = selected_scale * (-torch.log(1 - u)) ** (1 / selected_shape)

    # Add small epsilon to avoid zero times
    sampled_times = sampled_times + EPSILON_STABILITY

    return sampled_times.unsqueeze(-1)


def sample_lognormal(
    weights: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    """
    Sample from a mixture of LogNormal distributions.

    Args:
        weights: Mixture weights tensor of shape (n_samples, k)
        mu: Mu parameters tensor of shape (n_samples, k)
        sigma: Sigma parameters tensor of shape (n_samples, k)
    Returns:
        Sampled times tensor of shape (n_samples, 1)
    """
    mu = torch.nn.functional.softplus(mu)
    sigma = torch.nn.functional.softplus(sigma) + EPSILON_STABILITY
    n_samples = weights.shape[0]
    # Sample mixture component indices
    mixture_indices = torch.multinomial(weights, num_samples=1).squeeze(-1)

    # Gather the corresponding mu and sigma parameters
    selected_mu = mu[torch.arange(n_samples), mixture_indices]
    selected_sigma = sigma[torch.arange(n_samples), mixture_indices]

    # Sample from the selected LogNormal distributions
    eps_samples = torch.randn(n_samples, device=weights.device)
    log_times = selected_mu + selected_sigma * eps_samples

    # Clamp to prevent exp() overflow; underflow is harmless (produces near-zero + epsilon).
    MAX_LOG_TIME = 11  # exp(11) ≈ 59874
    log_times = torch.clamp(log_times, max=MAX_LOG_TIME)

    sampled_times = torch.exp(log_times) + EPSILON_STABILITY

    return sampled_times.unsqueeze(-1)
