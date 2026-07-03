import pytest
import torch

from survivalpfn.prior.table_generators import (
    KitchenSinkTableGenerator,
    TabDPTTableGenerator,
)
from survivalpfn.prior.table_generators.TabDPT import PFN_MLP

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def tabdpt_generator():
    return TabDPTTableGenerator(
        device=DEVICE,
        n_layer_dist=lambda: 3,
        n_hidden_dist=lambda: 16,
        dense_prob_dist=lambda: 0.5,
        init_std_dist=lambda: 1.0,
        noise_std_dist=lambda: 0.1,
        n_block_max_dist=lambda: 2,
        in_clique_prob=0.5,
        outcome_after_covariates_prob=0.5,
        categorical_columns_prob=0.0,
        categorical_columns_ordered_prob=0.5,
        pre_sample_prob=0.5,
    )


class TestTabDPTTableGenerator:
    def test_sample_table_shape(self, tabdpt_generator):
        torch.manual_seed(42)
        table = tabdpt_generator.sample_table(n_samples=100, n_columns=5)
        assert table.shape == (100, 5)

    def test_sample_table_no_nan(self, tabdpt_generator):
        torch.manual_seed(42)
        table = tabdpt_generator.sample_table(n_samples=100, n_columns=5)
        assert not torch.isnan(table).any()

    def test_sample_conditional_table_shape(self, tabdpt_generator):
        torch.manual_seed(42)
        input_data = torch.randn(100, 3, device=DEVICE)
        table = tabdpt_generator.sample_conditional_table(input=input_data, n_columns=2)
        assert table.shape == (100, 2)

    def test_sample_conditional_table_no_nan(self, tabdpt_generator):
        torch.manual_seed(42)
        input_data = torch.randn(100, 3, device=DEVICE)
        table = tabdpt_generator.sample_conditional_table(input=input_data, n_columns=2)
        assert not torch.isnan(table).any()

    def test_independence_structure(self, tabdpt_generator):
        torch.manual_seed(42)
        table = tabdpt_generator.sample_table(
            n_samples=100,
            n_columns=6,
            independence=True,
            indep_structure=[2, 4],
        )
        assert table.shape == (100, 6)
        assert not torch.isnan(table).any()

    def test_conditional_independence_structure(self, tabdpt_generator):
        torch.manual_seed(42)
        input_data = torch.randn(100, 3, device=DEVICE)
        table = tabdpt_generator.sample_conditional_table(
            input=input_data,
            n_columns=4,
            conditional_independence=True,
            indep_structure=[2, 2],
        )
        assert table.shape == (100, 4)
        assert not torch.isnan(table).any()

    def test_categorical_columns(self):
        torch.manual_seed(42)
        generator = TabDPTTableGenerator(
            device=DEVICE,
            n_layer_dist=lambda: 3,
            n_hidden_dist=lambda: 16,
            dense_prob_dist=lambda: 0.5,
            init_std_dist=lambda: 1.0,
            noise_std_dist=lambda: 0.1,
            n_block_max_dist=lambda: 2,
            in_clique_prob=0.5,
            outcome_after_covariates_prob=0.5,
            categorical_columns_prob=1.0,
            categorical_columns_ordered_prob=0.5,
            pre_sample_prob=0.0,
        )
        table = generator.sample_table(n_samples=100, n_columns=5)
        assert table.shape == (100, 5)

    def test_deterministic_with_seed(self, tabdpt_generator):
        torch.manual_seed(42)
        table1 = tabdpt_generator.sample_table(n_samples=50, n_columns=3)
        torch.manual_seed(42)
        table2 = tabdpt_generator.sample_table(n_samples=50, n_columns=3)
        assert torch.allclose(table1, table2, atol=1e-6)


class TestPFNMLP:
    def test_forward_with_input(self):
        torch.manual_seed(42)
        model = PFN_MLP(
            input_dim=10,
            covariates_dim=5,
            outcome_dim=1,
            num_layers=3,
            mlp_hidden_dim=16,
            dense_prob=0.5,
            init_std=1.0,
            noise_std=0.1,
            n_block_max=2,
            device=DEVICE,
            pre_sample=False,
            in_clique=False,
            outcome_after_covariates=False,
        )
        covariates, outcome = model.forward(torch.randn(100, 10, device=DEVICE))
        assert covariates.shape == (100, 5)
        assert outcome.shape == (100, 1)
        assert not torch.isnan(covariates).any()
        assert not torch.isnan(outcome).any()

    def test_forward_without_input(self):
        torch.manual_seed(42)
        model = PFN_MLP(
            input_dim=10,
            covariates_dim=5,
            outcome_dim=1,
            num_layers=3,
            mlp_hidden_dim=16,
            dense_prob=0.5,
            init_std=1.0,
            noise_std=0.1,
            n_block_max=2,
            device=DEVICE,
            pre_sample=False,
            in_clique=False,
            outcome_after_covariates=False,
        )
        covariates, outcome = model.forward(inp=None, n_samples=100)
        assert covariates.shape == (100, 5)
        assert outcome.shape == (100, 1)
        assert not torch.isnan(covariates).any()
        assert not torch.isnan(outcome).any()


class TestKitchenSinkTableGenerator:
    def test_sample_table_shape(self, tabdpt_generator):
        torch.manual_seed(42)
        kitchen_sink = KitchenSinkTableGenerator(
            table_generators=[tabdpt_generator],
            categorical_probabilities=[1.0],
            device=DEVICE,
            seed=42,
        )
        table = kitchen_sink.sample_table(n_samples=100, n_columns=5)
        assert table.shape == (100, 5)
        assert not torch.isnan(table).any()

    def test_sample_conditional_table(self, tabdpt_generator):
        torch.manual_seed(42)
        kitchen_sink = KitchenSinkTableGenerator(
            table_generators=[tabdpt_generator],
            categorical_probabilities=[1.0],
            device=DEVICE,
            seed=42,
        )
        input_data = torch.randn(100, 3, device=DEVICE)
        table = kitchen_sink.sample_conditional_table(input=input_data, n_columns=2)
        assert table.shape == (100, 2)
        assert not torch.isnan(table).any()
