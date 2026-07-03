from abc import abstractmethod
from typing import Dict

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import IterableDataset

from survivalpfn.prior.utils import PriorGenerationError
from survivalpfn.utils import pad_x


class MetaDataset(IterableDataset):
    """
    Data used for training: for each call, it will return
    a set of samples from a random data-generating process. It will output
    covariates, event/censoring times, censoring indicators, and any other data
    needed for training the survival analysis model.
    """

    def __init__(self, name: str, post_padding_n_cols: int):
        self.name = name
        # padding the number of columns so that tables can be batched together for training
        self.post_padding_n_cols = post_padding_n_cols

        # the set of all of the warnings that have been raised
        self.prior_generation_warnings = set()

    @abstractmethod
    def get_sample(self) -> dict:
        raise NotImplementedError(
            "Implement generating covariates and potential outcomes separately"
        )

    def __iter__(self):
        while (
            True
        ):  # The loop here is to retry if we get NaNs or other issues in the prior
            try:
                sample = self.get_sample()
            except PriorGenerationError as e:
                # Even though the warning is raised once for each worker, we might have multiple workers too.
                msg = str(e)
                if msg not in self.prior_generation_warnings:
                    self.prior_generation_warnings.add(msg)
                    print(f"[Prior Warning] {msg}")
                continue

            sample["X"] = pad_x(sample["X"], num_features=self.post_padding_n_cols)
            yield sample


class KitchenSinkPrior(MetaDataset):
    """
    A survival prior that combines multiple data-generating processes.

    This prior randomly selects from a set of predefined survival data-generating
    processes (DGPs) (e.g., NaiveSurvivalPrior, SurvivalDistributionPrior, MixtureModelSurvivalPrior)
    for each dataset generation. It allows for diverse survival
    datasets by mixing different underlying mechanisms.
    """

    def __init__(
        self,
        priors: list[MetaDataset] | Dict[str, MetaDataset],
        categorical_probabilities: list[float],
        seed: int,
        device: str = "cpu",  # device and n_samples below are unused but kept for hydra instantiation
        n_samples=2048,
    ):
        self.seed = seed
        self.device = device
        self.n_samples = n_samples
        if isinstance(priors, DictConfig):
            self.dgp_priors = OmegaConf.to_container(priors)

        self.all_dgp_priors = (
            priors if isinstance(priors, list) else list(priors.values())
        )
        self.probs = torch.tensor(categorical_probabilities) / sum(
            categorical_probabilities
        )
        self.rng = torch.Generator().manual_seed(seed)

        name = "KitchenSinkPrior: " + ", ".join(
            [dgp.name for dgp in self.all_dgp_priors]
        )
        super().__init__(
            name=name,
            post_padding_n_cols=max(
                dgp.post_padding_n_cols for dgp in self.all_dgp_priors
            ),
        )

    def get_sample(
        self,
    ) -> dict:
        """
        Generate a sample from a randomly selected DGP.

        Returns:
            A dictionary containing covariates, event times, censoring times, and indicators.
        """
        choose_idx = torch.multinomial(
            self.probs, num_samples=1, replacement=True, generator=self.rng
        ).item()
        chosen_dgp: MetaDataset = self.all_dgp_priors[choose_idx]
        return chosen_dgp.get_sample()
