from abc import ABC
from typing import Dict, List

import torch
from omegaconf import DictConfig, OmegaConf


class TableGenerator(ABC):
    """
    Abstract base class for table generation models.

    This class defines the interface for generating synthetic or real-world tabular data, from either
    joint distributions like P(X1, X2, ...) or conditional distributions like P(Y1, Y2, ... | X1, X2, ...).
    Subclasses must implement the core sampling methods, i.e., `_sample_table` to generate tables unconditionally
    and/or `_sample_conditional_table` to generate tables conditionally based on input data.

    Attributes:
        name (str): Human-readable name identifier for the generator.
        device (str): Device string (e.g., 'cpu', 'cuda:0') where tensors will be placed.
    """

    def __init__(self, name: str, device: str, *args, **kwargs):
        self.name = name
        self.device = device

    def __str__(self):
        return self.name

    def _sample_table(
        self,
        n_samples: int,
        n_columns: int,
        independence: bool,
        indep_structure: List[int] | None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Internal method to sample a table unconditionally.

        This method must be implemented by subclasses to define the core table generation logic.

        Args:
            n_samples (int): Number of rows to generate.
            n_columns (int): Number of columns in the generated table.
            independence (bool): If True, assumes independence among output columns.
            indep_structure (List[int]):
                List defining the independence structure among columns. For example, [2, 3] means the first two columns
                are independent of the next three columns.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: Generated table tensor of shape (n_samples, n_columns).

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError(
            "Subclasses should implement this method to sample a table."
        )

    def _sample_conditional_table(
        self,
        input: torch.Tensor,
        n_columns: int,
        conditional_independence: bool,
        indep_structure: List[int] | None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Internal method to sample a table conditionally based on input data.

        This method must be implemented by subclasses to define conditional table generation logic.

        Args:
            input (torch.Tensor): Conditioning input tensor with shape (n_samples, n_input_features).
            n_columns (int): Number of columns in the generated table.
            conditional_independence (bool): If True, assumes conditional independence among output columns.
            indep_structure (List[int]):
                List defining the independence structure among columns. For example, [2, 3] means the first two columns
                are independent of the next three columns.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: Generated table tensor conditioned on input of shape (n_samples, n_columns).

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError(
            "Subclasses should implement this method to sample a conditional table."
        )

    def sample_table(
        self,
        n_samples: int,
        n_columns: int,
        independence: bool = False,
        indep_structure: List[int] | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return self._sample_table(
            n_samples=n_samples,
            n_columns=n_columns,
            independence=independence,
            indep_structure=indep_structure,
            *args,
            **kwargs,
        ).to(self.device)

    def sample_conditional_table(
        self,
        input: torch.Tensor,
        n_columns: int,
        conditional_independence: bool = False,
        indep_structure: List[int] | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return self._sample_conditional_table(
            input=input,
            n_columns=n_columns,
            conditional_independence=conditional_independence,
            indep_structure=indep_structure,
            *args,
            **kwargs,
        ).to(self.device)


class KitchenSinkTableGenerator(TableGenerator):
    """
    A composite table generator that randomly selects from multiple underlying generators.

    This class implements an ensemble approach where each sampling operation randomly
    chooses one of the provided table generators based on specified probabilities.
    The name "KitchenSink" reflects its nature of combining multiple different generators.

    Attributes:
        seed (int): Random seed for reproducible generator selection.
        table_generators (List[TableGenerator] | Dict[str, TableGenerator]):
            List or dictionary of table generator instances.
        categorical_probabilities (List[float]):  Probability weights for generator selection.
        device (str): Device string (e.g., 'cpu', 'cuda:0') where tensors will be placed.
    """

    def __init__(
        self,
        table_generators: List[TableGenerator] | Dict[str, TableGenerator],
        categorical_probabilities: List[float],
        device: str,
        seed: int,
    ):

        self.seed = seed
        if isinstance(table_generators, DictConfig):
            table_generators = OmegaConf.to_container(table_generators)

        self.all_table_generators = (
            table_generators
            if isinstance(table_generators, list)
            else list(table_generators.values())
        )
        self.probs = torch.tensor(categorical_probabilities) / sum(
            categorical_probabilities
        )
        self.rng = torch.Generator().manual_seed(seed)

        name = "KitchenSink: " + ", ".join(
            [str(ds) for ds in self.all_table_generators]
        )
        super().__init__(name=name, device=device)

    def _sample_table(
        self,
        n_samples: int,
        n_columns: int,
        independence: bool,
        indep_structure: List[int] | None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # sample from one of the table generators using a categorical distribution
        choose_index = torch.multinomial(self.probs, 1, generator=self.rng).item()
        chosen_table_generator: TableGenerator = self.all_table_generators[choose_index]
        return chosen_table_generator.sample_table(
            n_samples=n_samples,
            n_columns=n_columns,
            independence=independence,
            indep_structure=indep_structure,
            *args,
            **kwargs,
        )

    def _sample_conditional_table(
        self,
        input: torch.Tensor,
        n_columns: int,
        conditional_independence: bool,
        indep_structure: List[int] | None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # sample from one of the table generators using a categorical distribution
        choose_index = torch.multinomial(self.probs, 1, generator=self.rng).item()
        chosen_table_generator: TableGenerator = self.all_table_generators[choose_index]
        return chosen_table_generator.sample_conditional_table(
            input=input,
            n_columns=n_columns,
            conditional_independence=conditional_independence,
            indep_structure=indep_structure,
            *args,
            **kwargs,
        )
