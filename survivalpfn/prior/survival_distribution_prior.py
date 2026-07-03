from typing import Callable, Tuple

import torch

from .base_survival_prior import BaseSurvivalPrior
from .table_generators import TableGenerator
from .utils import monotone_bernstein_map


class SurvivalDistributionPrior(BaseSurvivalPrior):
    """
    A survival prior that generates event/censoring times via distributions.

    This prior samples coefficients for Bernstein polynomials that define monotone
    cumulative distribution functions (CDFs). Event and censoring times are then
    sampled from these CDFs using inverse transform sampling.

    The approach allows for flexible, smooth survival distributions while maintaining
    monotonicity and ensuring times are in [0, t_max].
    """

    def __init__(
        self,
        covariates_generator: TableGenerator,
        event_censoring_coefficient_generator: TableGenerator,
        t_max_dist: Callable[[], float],
        n_knot_dist: Callable[[], int],
        max_n_covariates: int,
        layer_norm_covariates_prob: float,
        n_samples: int,
        device: str,
        desired_censoring_rate: float = None,
        censoring_type: str = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the survival distribution prior.

        Args:
            covariates_generator: Generator for sampling covariate tables
            event_censoring_coefficient_generator: Generator for sampling Bernstein coefficients
            t_max_dist: Callable that returns the maximum time value
            n_knot_dist: Callable that returns the number of knots (Bernstein degree)
            max_n_covariates: Maximum number of covariates to generate
            layer_norm_covariates_prob: Probability of applying layer normalization to covariates
            n_samples: Number of samples to generate per dataset
            device: Device to generate data on (e.g., 'cuda', 'cpu')
            desired_censoring_rate: Target censoring rate (if None, randomly sampled)
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
        self.t_max_dist = t_max_dist
        self.n_knot_dist = n_knot_dist
        self.event_censoring_coefficient_generator = (
            event_censoring_coefficient_generator
        )

    def _generate_event_and_censoring_times(
        self, covariates: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Generate event and censoring times using Bernstein polynomial CDFs.

        This method samples coefficients for Bernstein polynomials conditioned on covariates,
        then uses inverse transform sampling to generate times. With 50% probability, it
        samples event and censoring coefficients independently or jointly.

        Args:
            covariates: Covariate tensor of shape (n_samples, n_covariates)

        Returns:
            Tuple of (events, censoring, censoring_type) where events and censoring
            are tensors of shape (n_samples,) and censoring_type is a string.
        """
        censoring_type = self.censoring_type_fn()
        t_max = self.t_max_dist()

        if censoring_type in ["Uniform", "Tab", "Admin"]:
            # generate event times
            events_coeffs = (
                self.event_censoring_coefficient_generator.sample_conditional_table(
                    input=covariates,
                    n_columns=self.n_knot_dist(),
                )
            )
            # Use inverse CDF sampling to generate event times
            u_events = torch.rand(self.n_samples, device=self.device)
            events = monotone_bernstein_map(x=u_events, c=events_coeffs) * t_max

            # generate censoring times
            if censoring_type == "Uniform":
                censoring = torch.rand(self.n_samples, device=self.device) * t_max
            elif censoring_type == "Tab":
                censoring_coeffs = self.covariates_generator.sample_table(
                    n_samples=self.n_samples,
                    n_columns=self.n_knot_dist(),
                )
                u_censoring = torch.rand(self.n_samples, device=self.device)
                censoring = (
                    monotone_bernstein_map(x=u_censoring, c=censoring_coeffs) * t_max
                )
            elif censoring_type == "Admin":
                entry_dates_coeffs = self.covariates_generator.sample_table(
                    n_samples=self.n_samples,
                    n_columns=self.n_knot_dist(),
                )
                u_entry_dates = torch.rand(self.n_samples, device=self.device)
                entry_dates = (
                    monotone_bernstein_map(x=u_entry_dates, c=entry_dates_coeffs)
                    * t_max
                )
                # residual sampled from a uniform distribution
                res = torch.rand(1, device=entry_dates.device) * torch.max(entry_dates)
                # fixed administrative censoring time
                admin_date = torch.max(events) + res
                censoring = admin_date - entry_dates
            else:
                raise ValueError(f"Unsupported censoring type: {censoring_type}")
        elif censoring_type == "Cond Ind":
            # Randomly choose between independent or joint sampling of coefficients
            if torch.rand(1) < 0.5:
                # Sample event and censoring coefficients independently
                events_coeffs = (
                    self.event_censoring_coefficient_generator.sample_conditional_table(
                        input=covariates,
                        n_columns=self.n_knot_dist(),
                    )
                )
                censoring_coeffs = (
                    self.event_censoring_coefficient_generator.sample_conditional_table(
                        input=covariates,
                        n_columns=self.n_knot_dist(),
                    )
                )
            else:
                # Sample event and censoring coefficients jointly
                n_event_coeffs, n_censoring_coeffs = (
                    self.n_knot_dist(),
                    self.n_knot_dist(),
                )
                events_censoring = (
                    self.event_censoring_coefficient_generator.sample_conditional_table(
                        input=covariates,
                        n_columns=n_event_coeffs + n_censoring_coeffs,
                        conditional_independence=True,
                        indep_structure=[n_event_coeffs, n_censoring_coeffs],
                    )
                )
                events_coeffs = events_censoring[:, :n_event_coeffs]
                censoring_coeffs = events_censoring[:, n_event_coeffs:]

            # Use inverse CDF sampling to generate times
            # Sample uniform random variables
            u_events = torch.rand(self.n_samples, device=self.device)
            u_censoring = torch.rand(self.n_samples, device=self.device)

            # Map through monotone Bernstein polynomials and scale to [0, t_max]
            events = monotone_bernstein_map(x=u_events, c=events_coeffs) * t_max
            censoring = (
                monotone_bernstein_map(x=u_censoring, c=censoring_coeffs) * t_max
            )
        else:
            raise ValueError(f"Unsupported censoring type: {censoring_type}")

        return events, censoring, censoring_type
