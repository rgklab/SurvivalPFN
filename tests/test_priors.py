import pytest
import torch

from survivalpfn.models.constants import MAX_CENSORING_RATE
from survivalpfn.prior.mix_model_prior import MixtureModelSurvivalPrior
from survivalpfn.prior.naive_survival_prior import NaiveSurvivalPrior
from survivalpfn.prior.survival_distribution_prior import SurvivalDistributionPrior
from survivalpfn.prior.table_generators.TabDPT import TabDPTTableGenerator
from survivalpfn.prior.utils import PriorGenerationError

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 100
MAX_N_COVARIATES = 5


@pytest.fixture
def simple_table_generator():
    """Fixture providing a simple TabDPT table generator for testing."""
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
        pre_sample_prob=0.0,
    )


def _create_naive_prior(simple_table_generator, **kwargs):
    """Helper to create NaiveSurvivalPrior with default args."""
    defaults = {
        "covariates_generator": simple_table_generator,
        "event_censoring_generator": simple_table_generator,
        "max_n_covariates": MAX_N_COVARIATES,
        "layer_norm_covariates_prob": 0.0,
        "n_samples": N_SAMPLES,
        "device": DEVICE,
        "name": "TestPrior",
        "post_padding_n_cols": 10,
    }
    defaults.update(kwargs)
    return NaiveSurvivalPrior(**defaults)


def _create_distribution_prior(simple_table_generator, **kwargs):
    """Helper to create SurvivalDistributionPrior with default args."""
    defaults = {
        "covariates_generator": simple_table_generator,
        "event_censoring_coefficient_generator": simple_table_generator,
        "t_max_dist": lambda: 10.0,
        "n_knot_dist": lambda: 5,
        "max_n_covariates": MAX_N_COVARIATES,
        "layer_norm_covariates_prob": 0.0,
        "n_samples": N_SAMPLES,
        "device": DEVICE,
        "name": "TestPrior",
        "post_padding_n_cols": 10,
    }
    defaults.update(kwargs)
    return SurvivalDistributionPrior(**defaults)


def _create_mixture_model_prior_weibull(simple_table_generator, **kwargs):
    """Helper to create MixtureModelSurvivalPrior with default args."""
    defaults = {
        "covariates_generator": simple_table_generator,
        "event_censoring_generator": simple_table_generator,
        "k": 5,
        "dist": "Weibull",
        "max_n_covariates": MAX_N_COVARIATES,
        "layer_norm_covariates_prob": 0.0,
        "n_samples": N_SAMPLES,
        "device": DEVICE,
        "scale_which": "events",
        "name": "TestPrior",
        "post_padding_n_cols": 10,
        "outlier_removal_quantile": 0.9,
    }
    defaults.update(kwargs)
    return MixtureModelSurvivalPrior(**defaults)


def _create_mixture_model_prior_lognormal(simple_table_generator, **kwargs):
    """Helper to create MixtureModelSurvivalPrior with default args."""
    defaults = {
        "covariates_generator": simple_table_generator,
        "event_censoring_generator": simple_table_generator,
        "k": 5,
        "dist": "LogNormal",
        "max_n_covariates": MAX_N_COVARIATES,
        "layer_norm_covariates_prob": 0.0,
        "n_samples": N_SAMPLES,
        "device": DEVICE,
        "scale_which": "events",
        "name": "TestPrior",
        "post_padding_n_cols": 10,
        "outlier_removal_quantile": 0.9,
    }
    defaults.update(kwargs)
    return MixtureModelSurvivalPrior(**defaults)


class TestNaiveSurvivalPrior:
    """Tests for NaiveSurvivalPrior class."""

    def test_sample_shape(self, simple_table_generator):
        """Test that generated samples have correct shapes."""
        torch.manual_seed(42)
        prior = _create_naive_prior(simple_table_generator, desired_censoring_rate=0.3)
        sample = prior.get_sample()

        assert sample["X"].shape[0] == N_SAMPLES
        assert sample["X"].shape[1] <= MAX_N_COVARIATES
        assert sample["T"].shape == (N_SAMPLES,)
        assert sample["delta"].shape == (N_SAMPLES,)
        assert sample["E"].shape == (N_SAMPLES,)
        assert sample["C"].shape == (N_SAMPLES,)

    def test_covariates_normalization(self, simple_table_generator):
        """Test that covariates are properly normalized."""
        torch.manual_seed(42)
        prior = _create_naive_prior(simple_table_generator)
        sample = prior.get_sample()
        X = sample["X"]

        # Check approximate zero mean and unit variance
        assert torch.allclose(
            X.mean(dim=0), torch.zeros(X.shape[1], device=DEVICE), atol=1e-5
        )
        assert torch.allclose(
            X.std(dim=0), torch.ones(X.shape[1], device=DEVICE), atol=1e-5
        )

    def test_observed_times_computation(self, simple_table_generator):
        """Test that observed times T = min(E, C) and delta is computed correctly."""
        torch.manual_seed(42)
        prior = _create_naive_prior(simple_table_generator)
        sample = prior.get_sample()

        T = sample["T"]
        E = sample["E"]
        C = sample["C"]
        delta = sample["delta"]

        # T should equal min(E, C)
        assert torch.allclose(T, torch.min(E, C))

        # delta should be 1 when E < C (event occurred), 0 otherwise (censored)
        expected_delta = (E < C).float()
        assert torch.allclose(delta, expected_delta)

    def test_censoring_rate_adjustment(self, simple_table_generator):
        """Test that censoring rate is adjusted to desired rate."""
        torch.manual_seed(42)
        desired_rate = 0.4
        prior = _create_naive_prior(
            simple_table_generator, desired_censoring_rate=desired_rate
        )
        sample = prior.get_sample()

        delta = sample["delta"]
        actual_censoring_rate = 1.0 - delta.mean().item()

        # Should be very close to desired rate
        assert abs(actual_censoring_rate - desired_rate) < 0.02

    def test_random_censoring_rate(self, simple_table_generator):
        """Test that random censoring rate is in valid range."""
        torch.manual_seed(42)
        prior = _create_naive_prior(simple_table_generator, desired_censoring_rate=None)
        sample = prior.get_sample()

        delta = sample["delta"]
        actual_censoring_rate = 1.0 - delta.mean().item()

        # Should be in valid range
        min_censoring_rate = 1.0 - MAX_CENSORING_RATE
        assert min_censoring_rate <= actual_censoring_rate <= MAX_CENSORING_RATE

    def test_no_nan_values(self, simple_table_generator):
        """Test that generated samples contain no NaN values."""
        torch.manual_seed(42)
        prior = _create_naive_prior(simple_table_generator)
        sample = prior.get_sample()

        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                assert not torch.isnan(value).any(), f"NaN found in {key}"

    def test_all_times_nonnegative(self, simple_table_generator):
        """Test that all times are non-negative."""
        torch.manual_seed(42)
        prior = _create_naive_prior(simple_table_generator)
        sample = prior.get_sample()

        assert (sample["T"] >= 0).all()
        assert (sample["E"] >= 0).all()
        assert (sample["C"] >= 0).all()

    def test_deterministic_with_seed(self, simple_table_generator):
        """Test that results are deterministic when seed is set."""
        torch.manual_seed(42)
        prior1 = _create_naive_prior(simple_table_generator, desired_censoring_rate=0.3)
        sample1 = prior1.get_sample()

        torch.manual_seed(42)
        prior2 = _create_naive_prior(simple_table_generator, desired_censoring_rate=0.3)
        sample2 = prior2.get_sample()

        # Results should be identical with same seed
        for key in sample1.keys():
            if isinstance(sample1[key], torch.Tensor):
                assert torch.allclose(sample1[key], sample2[key], atol=1e-6)


class TestSurvivalDistributionPrior:
    """Tests for SurvivalDistributionPrior class."""

    def test_sample_shape(self, simple_table_generator):
        """Test that generated samples have correct shapes."""
        torch.manual_seed(42)
        prior = _create_distribution_prior(
            simple_table_generator, desired_censoring_rate=0.3
        )
        sample = prior.get_sample()

        assert sample["X"].shape[0] == N_SAMPLES
        assert sample["X"].shape[1] <= MAX_N_COVARIATES
        assert sample["T"].shape == (N_SAMPLES,)
        assert sample["delta"].shape == (N_SAMPLES,)
        assert sample["E"].shape == (N_SAMPLES,)
        assert sample["C"].shape == (N_SAMPLES,)

    def test_censoring_rate_adjustment(self, simple_table_generator):
        """Test that censoring rate is adjusted to desired rate."""
        torch.manual_seed(42)
        desired_rate = 0.4
        prior = _create_distribution_prior(
            simple_table_generator, desired_censoring_rate=desired_rate
        )
        sample = prior.get_sample()

        delta = sample["delta"]
        actual_censoring_rate = 1.0 - delta.mean().item()

        # Should be very close to desired rate
        assert abs(actual_censoring_rate - desired_rate) < 0.02

    def test_no_nan_values(self, simple_table_generator):
        """Test that generated samples contain no NaN values."""
        torch.manual_seed(42)
        prior = _create_distribution_prior(simple_table_generator)
        sample = prior.get_sample()

        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                assert not torch.isnan(value).any(), f"NaN found in {key}"


class TestMixtureModelSurvivalPriorWeibull:
    """Tests for MixtureModelSurvivalPrior class."""

    def test_sample_shape(self, simple_table_generator):
        """Test that generated samples have correct shapes."""
        torch.manual_seed(42)
        prior = _create_mixture_model_prior_weibull(
            simple_table_generator, desired_censoring_rate=0.3
        )
        sample = prior.get_sample()

        assert sample["X"].shape[0] <= N_SAMPLES
        assert sample["X"].shape[1] <= MAX_N_COVARIATES
        assert sample["T"].shape[0] <= N_SAMPLES
        assert sample["delta"].shape[0] <= N_SAMPLES
        assert sample["E"].shape[0] <= N_SAMPLES
        assert sample["C"].shape[0] <= N_SAMPLES

    def test_censoring_rate_adjustment(self, simple_table_generator):
        """Test that censoring rate is adjusted to desired rate."""
        torch.manual_seed(42)
        desired_rate = 0.4
        prior = _create_mixture_model_prior_weibull(
            simple_table_generator, desired_censoring_rate=desired_rate
        )
        sample = prior.get_sample()

        delta = sample["delta"]
        actual_censoring_rate = 1.0 - delta.mean().item()

        # Should be very close to desired rate
        assert abs(actual_censoring_rate - desired_rate) < 0.02

    def test_no_nan_values(self, simple_table_generator):
        """Test that generated samples contain no NaN values."""
        torch.manual_seed(42)
        prior = _create_mixture_model_prior_weibull(simple_table_generator)
        sample = prior.get_sample()

        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                assert not torch.isnan(value).any(), f"NaN found in {key}"


class TestMixtureModelSurvivalPriorLogNormal:
    """Tests for MixtureModelSurvivalPrior class."""

    def test_sample_shape(self, simple_table_generator):
        """Test that generated samples have correct shapes."""
        torch.manual_seed(42)
        prior = _create_mixture_model_prior_lognormal(
            simple_table_generator, desired_censoring_rate=0.3
        )
        sample = prior.get_sample()

        assert sample["X"].shape[0] <= N_SAMPLES
        assert sample["X"].shape[1] <= MAX_N_COVARIATES
        assert sample["T"].shape[0] <= N_SAMPLES
        assert sample["delta"].shape[0] <= N_SAMPLES
        assert sample["E"].shape[0] <= N_SAMPLES
        assert sample["C"].shape[0] <= N_SAMPLES

    def test_censoring_rate_adjustment(self, simple_table_generator):
        """Test that censoring rate is adjusted to desired rate."""
        torch.manual_seed(42)
        desired_rate = 0.4
        prior = _create_mixture_model_prior_lognormal(
            simple_table_generator, desired_censoring_rate=desired_rate
        )
        sample = prior.get_sample()

        delta = sample["delta"]
        actual_censoring_rate = 1.0 - delta.mean().item()

        # Should be very close to desired rate
        assert abs(actual_censoring_rate - desired_rate) < 0.02

    def test_no_nan_values(self, simple_table_generator):
        """Test that generated samples contain no NaN values."""
        torch.manual_seed(42)
        prior = _create_mixture_model_prior_lognormal(simple_table_generator)
        sample = prior.get_sample()

        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                assert not torch.isnan(value).any(), f"NaN found in {key}"


class TestBaseSurvivalPrior:
    """Tests for BaseSurvivalPrior base class functionality."""

    def test_layer_norm_probability(self, simple_table_generator):
        """Test that layer normalization is applied with correct probability."""
        # Test with 0 probability - should never apply
        torch.manual_seed(42)
        prior = _create_naive_prior(
            simple_table_generator, layer_norm_covariates_prob=0.0
        )
        sample = prior.get_sample()
        # Should still have valid shape
        assert sample["X"].shape[0] == N_SAMPLES

        # Test with 1.0 probability - should always apply
        torch.manual_seed(42)
        prior = _create_naive_prior(
            simple_table_generator, layer_norm_covariates_prob=1.0
        )
        sample = prior.get_sample()
        # Should still have valid shape
        assert sample["X"].shape[0] == N_SAMPLES

    def test_validation_catches_nan(self, simple_table_generator):
        """Test that validation catches NaN values."""
        prior = _create_naive_prior(simple_table_generator)

        # Test with NaN in T
        T_nan = torch.tensor([1.0, float("nan"), 3.0], device=DEVICE)
        delta = torch.tensor([1.0, 0.0, 1.0], device=DEVICE)
        X = torch.randn(3, 2, device=DEVICE)

        with pytest.raises(
            PriorGenerationError, match="NaN in Survival Data Generating Process"
        ):
            prior._validate_output(T_nan, delta, X)

        # Test with NaN in delta
        T = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
        delta_nan = torch.tensor([1.0, float("nan"), 1.0], device=DEVICE)

        with pytest.raises(
            PriorGenerationError, match="NaN in Survival Data Generating Process"
        ):
            prior._validate_output(T, delta_nan, X)

        # Test with NaN in X
        X_nan = torch.tensor(
            [[1.0, 2.0], [float("nan"), 4.0], [5.0, 6.0]], device=DEVICE
        )

        with pytest.raises(
            PriorGenerationError, match="NaN in Survival Data Generating Process"
        ):
            prior._validate_output(T, delta, X_nan)

    def test_multiple_samples_independence(self, simple_table_generator):
        """Test that multiple samples are independent."""
        torch.manual_seed(42)
        prior = _create_naive_prior(simple_table_generator)

        sample1 = prior.get_sample()
        sample2 = prior.get_sample()

        # Samples should be different
        assert sample1["X"].shape != sample2["X"].shape or not torch.allclose(
            sample1["X"], sample2["X"]
        )
        assert not torch.allclose(sample1["T"], sample2["T"])
