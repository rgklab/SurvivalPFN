"""
Base class for survival analysis priors.

This module provides a base class that encapsulates common logic for generating
survival analysis data, including covariate generation, censoring rate adjustment,
and validation.
"""

from abc import abstractmethod
from typing import Dict, Tuple

import torch

from survivalpfn.models.constants import (
    CENSORING_CHOICES,
    EPSILON_STABILITY,
    MAX_CENSORING_RATE,
)

from .meta_dataset import MetaDataset
from .table_generators import TableGenerator
from .utils import PriorGenerationError, fix_censoring_rate


class BaseSurvivalPrior(MetaDataset):
    """
    Base class for survival analysis data generation.

    This class provides common functionality for generating survival analysis datasets,
    including:
    - Covariate generation and normalization
    - Censoring rate adjustment
    - Validation of generated data
    - Computation of observed survival times

    Subclasses must implement the `_generate_event_and_censoring_times` method to
    specify how event and censoring times are generated.
    """

    def __init__(
        self,
        covariates_generator: TableGenerator,
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
        Initialize the base survival prior.

        Args:
            covariates_generator: Generator for sampling covariate tables
            max_n_covariates: Maximum number of covariates to generate
            layer_norm_covariates_prob: Probability of applying layer normalization to covariates
            n_samples: Number of samples to generate per dataset
            device: Device to generate data on (e.g., 'cuda', 'cpu')
            desired_censoring_rate: Target censoring rate (if None, randomly sampled)
            *args, **kwargs: Additional arguments passed to MetaDataset
        """
        self.covariates_generator = covariates_generator
        self.max_n_covariates = max_n_covariates
        self.layer_norm_covariates_prob = layer_norm_covariates_prob
        self.n_samples = n_samples
        self.device = device
        if desired_censoring_rate is not None:
            assert 0.0 < desired_censoring_rate < 1, (
                f"Desired censoring rate must be between 0 and 1, got {desired_censoring_rate}"
            )
            self.desired_censoring_rate_fn = lambda: desired_censoring_rate
        else:
            min_censoring_rate = 1.0 - MAX_CENSORING_RATE
            self.desired_censoring_rate_fn = lambda: (
                torch.rand(1, device=self.device).item()
                * (MAX_CENSORING_RATE - min_censoring_rate)
                + min_censoring_rate
            )

        if censoring_type is not None:
            if censoring_type not in CENSORING_CHOICES:
                raise ValueError(
                    f"Censoring type '{censoring_type}' is not recognized. Choose from {CENSORING_CHOICES}."
                )
            self.censoring_type_fn = lambda: censoring_type
        else:
            self.censoring_type_fn = lambda: CENSORING_CHOICES[
                torch.randint(len(CENSORING_CHOICES), (1,)).item()
            ]

        super().__init__(*args, **kwargs)

    def _generate_covariates(self) -> torch.Tensor:
        """
        Generate and normalize covariates.

        Returns:
            Normalized covariate tensor of shape (n_samples, n_covariates)
        """
        n_covariates = torch.randint(
            1, self.max_n_covariates + 1, (1,), device=self.device
        ).item()

        covariates = self.covariates_generator.sample_table(
            n_samples=self.n_samples, n_columns=n_covariates
        )

        # Standardize covariates (zero mean, unit variance)
        covariates = (covariates - covariates.mean(dim=0)) / (
            covariates.std(dim=0) + EPSILON_STABILITY
        )

        # Optionally apply layer normalization
        if torch.rand(1) < self.layer_norm_covariates_prob:
            covariates = torch.nn.functional.layer_norm(
                covariates, covariates.shape[1:]
            )

        return covariates

    @abstractmethod
    def _generate_event_and_censoring_times(
        self, covariates: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Generate event and censoring times given covariates.

        This method must be implemented by subclasses to define how event and
        censoring times are generated.

        Args:
            covariates: Covariate tensor of shape (n_samples, n_covariates)

        Returns:
            Tuple of (events, censoring, censoring_type), where events and censoring are tensors of shape
              (n_samples, 1) or (n_samples,), and censoring_type is a string indicating the type of censoring used.
        """
        pass

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
        desired_censoring_rate = self.desired_censoring_rate_fn()

        C, E = fix_censoring_rate(
            censoring=censoring,
            events=events,
            desired_censoring_rate=desired_censoring_rate,
        )

        return C, E, desired_censoring_rate

    def _compute_observed_times(
        self, E: torch.Tensor, C: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute observed survival times and event indicators.

        Args:
            E: True event times
            C: True censoring times

        Returns:
            Tuple of (T, delta) where:
            - T: Observed time = min(event time, censoring time)
            - delta: Event indicator (1 if event occurred, 0 if censored)
        """
        T = torch.min(E, C)  # Observed time
        delta = (E < C).float()  # Event indicator: 1 if event occurred, 0 if censored
        return T, delta

    def _validate_output(
        self, T: torch.Tensor, delta: torch.Tensor, covariates: torch.Tensor
    ) -> None:
        """
        Validate that generated data contains no NaN values.

        Args:
            T: Observed survival times
            delta: Event indicators
            covariates: Covariates

        Raises:
            PriorGenerationError: If any NaN values are detected
        """
        if (
            torch.isnan(T).any()
            or torch.isnan(covariates).any()
            or torch.isnan(delta).any()
        ):
            raise PriorGenerationError("NaN in Survival Data Generating Process")

    @torch.no_grad()
    def get_sample(self) -> Dict[str, torch.Tensor]:
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
        events, censoring, censoring_type = self._generate_event_and_censoring_times(
            covariates
        )

        # Step 3: Adjust censoring rate
        C, E, censoring_rate = self._adjust_censoring_rate(events, censoring)

        # Step 4: Compute observed times and indicators
        T, delta = self._compute_observed_times(E, C)

        # Step 5: Validate output
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
