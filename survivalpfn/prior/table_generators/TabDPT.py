import math
from typing import Callable, List, Tuple

import torch
from torch import nn
from torch.distributions import Bernoulli as TorchBernoulli

from survivalpfn.prior.utils import PriorGenerationError, num2cat

from .base import TableGenerator


class StepActivation(nn.Module):
    def forward(self, input):
        return torch.sign(input)


class SineActivation(nn.Module):
    def forward(self, input):
        return torch.sin(input)


class CosineActivation(nn.Module):
    def forward(self, input):
        return torch.cos(input)


DEFAULT_ACT_FUNCS = [
    CosineActivation,
    SineActivation,
    StepActivation,
    nn.ELU,
    nn.GELU,
    nn.Hardsigmoid,
    nn.Identity,
    nn.LeakyReLU,
    nn.LogSigmoid,
    nn.Sigmoid,
    nn.SiLU,
    nn.Softplus,
    nn.Tanh,
]


class GaussianNoise(nn.Module):
    def __init__(self, std):
        super().__init__()
        self.std = std

    def forward(self, x):
        return x + torch.randn_like(x) * self.std


class MaskedLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: torch.Tensor,
        bias: bool = False,
        device: str = "cpu",
    ):
        """
        Linear layer with an applied mask over the weights.

        Parameters:
            in_features: Number of input features.
            out_features: Number of output features.
            mask: Mask to apply over weights. Zero values in the mask will zero out the corresponding weights.
            bias: If set to False, the layer will not learn an additive bias.
        """
        super().__init__(
            in_features=in_features, out_features=out_features, bias=bias, device=device
        )
        if mask.size() != (in_features, out_features):
            raise PriorGenerationError(
                f"Mask size {mask.size()} does not match weight size {(in_features, out_features)}"
            )
        self.register_buffer("mask", mask.to(torch.bool))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MaskedLinear layer."""
        return torch.nn.functional.linear(input, self.weight * self.mask.T, self.bias)


class MaskedBlockLinear(MaskedLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: torch.Tensor,
        bias: bool = False,
        init_std: float = 1.0,
        n_block_max: int | None = None,
        device: str = "cpu",
    ):
        """
        Block linear layer with an applied mask over the weights.

        Parameters:
            in_features: Number of input features.
            out_features: Number of output features.
            mask: Mask to apply over weights.
            bias: If set to False, the layer will not learn an additive bias.
            init_std: Standard deviation for the initialization.
            n_block_max: Maximum number of blocks to use.
        """
        self.init_std = init_std
        self.n_block_max = n_block_max
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            mask=mask,
            bias=bias,
            device=device,
        )
        self.reset_parameters()

    def reset_parameters(self):
        # Blockwise dropout
        torch.nn.init.zeros_(self.weight)
        if self.n_block_max is None:
            n_block_max = math.ceil(
                math.sqrt(min(self.weight.shape[0], self.weight.shape[1]))
            )
        else:
            n_block_max = self.n_block_max

        n_blocks = torch.randint(1, n_block_max + 1, ()).item()
        n_blocks = min(n_blocks, self.weight.shape[0], self.weight.shape[1])
        w, h = self.weight.shape[0] // n_blocks, self.weight.shape[1] // n_blocks
        keep_prob = (
            (n_blocks * w * h) / self.weight.numel()
        )  # Keeps the same standard deviation for the sume of the weights
        for block in range(0, n_blocks):
            torch.nn.init.normal_(
                self.weight[
                    w * block : w * (block + 1),
                    h * block : h * (block + 1),
                ],
                std=self.init_std / keep_prob**0.5,
            )


class PFN_MLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        covariates_dim: int,
        outcome_dim: int,
        num_layers: int,
        mlp_hidden_dim: int,
        dense_prob: float,
        init_std: float,
        noise_std: float,
        n_block_max: int | None,
        device: str,
        pre_sample: bool,  # "Ignore these -- even I don't have an understanding of what these are." - Vahid, 2025
        in_clique: bool,  # "Ignore these -- even I don't have an understanding of what these are." - Vahid, 2025
        outcome_after_covariates: bool,
        name: str = "PFN MLP",
    ):
        super().__init__()

        if num_layers <= 1:
            raise PriorGenerationError(
                f"num_layers must be greater than 1, got {num_layers}"
            )
        if covariates_dim < 0 or outcome_dim < 0 or covariates_dim + outcome_dim <= 0:
            raise PriorGenerationError(
                f"covariates_dim and outcome_dim must be non-negative, got {covariates_dim}, {outcome_dim}"
            )
        if input_dim <= 0:
            raise PriorGenerationError(
                f"input_dim must be greater than 0, got {input_dim}"
            )

        mlp_hidden_dim = max(mlp_hidden_dim, outcome_dim + 2 * covariates_dim)

        self.input_dim = input_dim
        self.device = device
        self.covariates_dim = covariates_dim
        self.outcome_dim = outcome_dim
        self.name = name
        self.pre_sample = pre_sample
        self.in_clique = in_clique
        self.outcome_after_covariates = outcome_after_covariates

        def rand_layer(out_dim):
            mask = TorchBernoulli(
                torch.ones((mlp_hidden_dim, out_dim), device=device) * dense_prob
            ).sample()
            modules = [
                DEFAULT_ACT_FUNCS[torch.randint(0, len(DEFAULT_ACT_FUNCS), (1,))[0]](),
                MaskedBlockLinear(
                    mlp_hidden_dim,
                    out_dim,
                    mask=mask,
                    bias=False,
                    device=device,
                    init_std=init_std,
                    n_block_max=n_block_max,
                ),
                GaussianNoise(noise_std),
            ]
            return nn.Sequential(*modules)

        mask = TorchBernoulli(
            torch.ones((input_dim, mlp_hidden_dim), device=device) * dense_prob
        ).sample()
        self.layers = [
            MaskedBlockLinear(
                input_dim,
                mlp_hidden_dim,
                mask=mask,
                bias=False,
                device=device,
                init_std=init_std,
                n_block_max=n_block_max,
            )
        ]
        self.layers += [rand_layer(mlp_hidden_dim) for _ in range(num_layers - 1)]
        self.layers = nn.Sequential(*self.layers)

    def forward(
        self, inp: torch.Tensor | None, n_samples: int | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if inp is not None and self.input_dim != inp.shape[-1]:
            raise PriorGenerationError(
                f"Input shape {inp.shape} does not match {self.input_dim}"
            )
        if inp is None and n_samples is None:
            raise PriorGenerationError("Either x or n_samples must be provided")

        if inp is None:
            if self.pre_sample:
                x_mean = torch.normal(0, 1, size=(self.input_dim,), device=self.device)
                x_std = (
                    torch.normal(0, 1, size=(self.input_dim,), device=self.device)
                    * x_mean
                ).abs()
                inp = torch.normal(
                    mean=x_mean.unsqueeze(0).expand(n_samples, -1),
                    std=x_std.unsqueeze(0).expand(n_samples, -1),
                )
            else:
                inp = torch.normal(
                    0.0, 1.0, size=(n_samples, self.input_dim), device=self.device
                )

        inp = inp.to(self.device)
        outputs = [inp]
        for layer in self.layers:
            outputs.append(layer(outputs[-1]))
        outputs = outputs[2:]
        outputs_flat: torch.Tensor = torch.cat(outputs, -1)

        if not self.outcome_after_covariates:
            if self.in_clique:
                num_choices = (
                    outputs_flat.shape[-1] - self.covariates_dim - self.outcome_dim
                )
                random_perm_base = (
                    torch.randint(0, num_choices, (1,), device=self.device)
                    if num_choices > 0
                    else 0
                )
                random_perm = random_perm_base + torch.randperm(
                    self.covariates_dim + self.outcome_dim, device=self.device
                )
            else:
                num_choices = outputs_flat.shape[-1] - 1
                random_perm = (
                    torch.randperm(num_choices, device=self.device)
                    if num_choices > 0
                    else 0
                )

            random_idx_outcome = random_perm[: self.outcome_dim]
            random_idx_covs = random_perm[
                self.outcome_dim : self.outcome_dim + self.covariates_dim
            ]

        else:
            cov_last_idx = int(
                (self.covariates_dim / (self.covariates_dim + self.outcome_dim))
                * outputs_flat.shape[-1]
            )
            all_cov_idx = torch.arange(0, cov_last_idx, device=self.device)
            if self.in_clique:
                num_choices = len(all_cov_idx) - self.covariates_dim
                random_perm_base = (
                    torch.randint(0, num_choices, (1,), device=self.device)
                    if num_choices > 0
                    else 0
                )
                random_idx_covs = random_perm_base + torch.randperm(
                    self.covariates_dim, device=self.device
                )
            else:
                random_idx_covs = torch.randperm(len(all_cov_idx), device=self.device)[
                    : self.covariates_dim
                ]

            all_outcome_idx = torch.arange(
                cov_last_idx, outputs_flat.shape[-1], device=self.device
            )
            random_idx_outcome = all_outcome_idx[
                torch.randperm(len(all_outcome_idx), device=self.device)[
                    : self.outcome_dim
                ]
            ]

        outcome = outputs_flat[:, random_idx_outcome]
        covariates = outputs_flat[:, random_idx_covs]

        if torch.isnan(outcome).any() or torch.isnan(covariates).any():
            raise PriorGenerationError(f"NaN in {self.name}")

        # random covariates rotation
        if covariates.shape[-1] > 0:
            covariates_shift = torch.randint(
                0, covariates.shape[-1], (1,), device=self.device
            ).item()
            covariates = torch.cat(
                (
                    covariates[..., covariates_shift:],
                    covariates[..., :covariates_shift],
                ),
                dim=-1,
            )

        # random outcome rotation
        if outcome.shape[-1] > 0:
            outcome_shift = torch.randint(
                0, outcome.shape[-1], (1,), device=self.device
            ).item()
            outcome = torch.cat(
                (outcome[..., outcome_shift:], outcome[..., :outcome_shift]), dim=-1
            )
        return covariates, outcome


class TabDPTTableGenerator(TableGenerator):
    def __init__(
        self,
        device: str,
        n_layer_dist: Callable[[], int],
        n_hidden_dist: Callable[[], int],
        dense_prob_dist: Callable[[], float],
        init_std_dist: Callable[[], float],
        noise_std_dist: Callable[[], float],
        n_block_max_dist: Callable[[], int] | None,
        in_clique_prob: float,
        outcome_after_covariates_prob: float,
        categorical_columns_prob: float,
        categorical_columns_ordered_prob: float,
        pre_sample_prob: float = 0.0,
    ):
        self.n_layer_dist = n_layer_dist
        self.n_hidden_dist = n_hidden_dist
        self.dense_prob_dist = dense_prob_dist
        self.init_std_dist = init_std_dist
        self.noise_std_dist = noise_std_dist
        self.n_block_max_dist = n_block_max_dist
        self.in_clique_prob = in_clique_prob
        self.outcome_after_covariates_prob = outcome_after_covariates_prob
        self.pre_sample_prob = pre_sample_prob
        self.categorical_columns_prob = categorical_columns_prob
        self.categorical_columns_ordered_prob = categorical_columns_ordered_prob

        super().__init__(device=device, name="TabDPT MLP Table Generator")

    def _add_categorical_covariates(self, table: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.categorical_columns_prob:
            for col in range(table.shape[-1]):
                if torch.rand(1) < 0.5:
                    max_categories = max(
                        round(torch.distributions.Gamma(1, 0.1).sample().item()), 2
                    )
                    table[..., col] = num2cat(
                        table[..., col],
                        max_categories,
                        ordered_p=self.categorical_columns_ordered_prob,
                    )

        return table

    def get_get_model(self, input_dim: int | None = None) -> Callable[[int], PFN_MLP]:
        input_dim = self.n_hidden_dist() if input_dim is None else input_dim
        num_layers = self.n_layer_dist()
        mlp_hidden_dim = self.n_hidden_dist()
        dense_prob = self.dense_prob_dist()
        init_std = self.init_std_dist()
        noise_std = self.noise_std_dist()
        n_block_max = (
            self.n_block_max_dist() if self.n_block_max_dist is not None else None
        )
        device = self.device
        pre_sample = torch.rand(1) < self.pre_sample_prob
        in_clique = torch.rand(1) < self.in_clique_prob
        outcome_after_covariates = torch.rand(1) < self.outcome_after_covariates_prob

        return lambda n_col: PFN_MLP(
            input_dim=input_dim,
            covariates_dim=n_col,
            outcome_dim=1,
            num_layers=num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            dense_prob=dense_prob,
            init_std=init_std,
            noise_std=noise_std,
            n_block_max=n_block_max,
            device=device,
            pre_sample=pre_sample,
            in_clique=in_clique,
            outcome_after_covariates=outcome_after_covariates,
        )

    def _sample_table(
        self,
        n_samples: int,
        n_columns: int,
        independence: bool,
        indep_structure: List[int] | None,
    ) -> torch.Tensor:
        get_model = self.get_get_model()
        if not independence:
            model = get_model(n_columns)
            table, _ = model.forward(inp=None, n_samples=n_samples)
        else:
            if indep_structure is None:
                indep_structure = [1 for _ in range(n_columns)]

            assert sum(indep_structure) == n_columns, (
                "Independence structure does not match number of columns"
            )

            table = torch.zeros((n_samples, n_columns), device=self.device)
            for i, n_col in enumerate(indep_structure):
                model = get_model(n_col)
                table_i, _ = model.forward(inp=None, n_samples=n_samples)
                table[:, sum(indep_structure[:i]) : sum(indep_structure[: i + 1])] = (
                    table_i
                )

        return self._add_categorical_covariates(table)

    def _sample_conditional_table(
        self,
        input: torch.Tensor,
        n_columns: int,
        conditional_independence: bool,
        indep_structure: List[int] | None,
    ) -> torch.Tensor:

        get_model = self.get_get_model(input_dim=input.shape[-1])

        if conditional_independence:
            if indep_structure is None:
                indep_structure = [1 for _ in range(n_columns)]

            assert sum(indep_structure) == n_columns, (
                "Independence structure does not match number of columns"
            )

            table = torch.zeros((input.shape[0], n_columns), device=self.device)
            for i, n_col in enumerate(indep_structure):
                model = get_model(n_col)
                table_i, _ = model.forward(inp=input)
                table[:, sum(indep_structure[:i]) : sum(indep_structure[: i + 1])] = (
                    table_i
                )
        else:
            model = get_model(n_columns)
            table, _ = model.forward(inp=input)

        return table  # no categorical covariates in conditional generation
