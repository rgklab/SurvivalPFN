from typing import Tuple

import torch

from .base_survival_prior import BaseSurvivalPrior
from .table_generators import TableGenerator


class NaiveSurvivalPrior(BaseSurvivalPrior):
    """
    A data generating process to sample covariates + event/censoring times for survival analysis.

    This prior generates event and censoring times directly from a table generator. It samples
    both event and censoring times jointly to ensure conditional independence given covariates.

    Event and censoring times are generated from a provided table generator, which samples
    both times jointly from the same distribution to ensure conditional independence between
    event and censoring times given covariates.

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
        desired_censoring_rate: float = None,
        censoring_type: str = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the naive survival prior.

        Args:
            covariates_generator: Generator for sampling covariate tables
            event_censoring_generator: Generator for sampling event and censoring times
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
        self.event_censoring_generator = event_censoring_generator

    def _generate_event_and_censoring_times(
        self, covariates: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Generate event and censoring times from the same generator.

        This method samples event and censoring times jointly from a single call to the
        generator, which ensures conditional independence between event and censoring
        times given the covariates.

        Args:
            covariates: Covariate tensor of shape (n_samples, n_covariates)

        Returns:
            Tuple of (events, censoring, censoring_type) where events and censoring
            are tensors of shape (n_samples, 1) and censoring_type is a string.
        """
        censoring_type = self.censoring_type_fn()

        if censoring_type in ["Uniform", "Tab", "Admin"]:
            # generate event times
            events = self.event_censoring_generator.sample_conditional_table(
                input=covariates, n_columns=1
            )
            # generate censoring times
            if censoring_type == "Uniform":
                # Uniform censoring: sample from uniform distribution
                min_event_time = torch.min(events)
                max_event_time = torch.max(events)
                censoring = (
                    torch.rand_like(events) * (max_event_time - min_event_time)
                    + min_event_time
                )
            elif censoring_type == "Tab":
                # Tabular censoring: sample from a 1-D table generator
                censoring = self.covariates_generator.sample_table(
                    n_samples=self.n_samples, n_columns=1
                )
            elif censoring_type == "Admin":
                # Administrative censoring: sample entry dates from a 1-D table generator
                entry_dates = self.covariates_generator.sample_table(
                    n_samples=self.n_samples, n_columns=1
                )
                # residual sampled from a uniform distribution
                res = torch.rand(1, device=entry_dates.device) * (
                    torch.max(entry_dates) - torch.min(entry_dates)
                )
                # fixed administrative censoring time
                fixed_admin_date = torch.max(events) + res
                censoring = fixed_admin_date - entry_dates
            else:
                raise ValueError(f"Unsupported censoring type: {censoring_type}")
        elif censoring_type == "Cond Ind":
            # Sample event and censoring times with conditional independence given covariates
            events_censoring = self.event_censoring_generator.sample_conditional_table(
                input=covariates, n_columns=2
            )
            events = events_censoring[:, 0:1]
            censoring = events_censoring[:, 1:2]
        else:
            raise ValueError(f"Unsupported censoring type: {censoring_type}")

        return events, censoring, censoring_type
