import numpy as np
import torch

from survivalpfn.evaluation import nll_surv
from survivalpfn.inference.evaluate import (
    _resolve_preprocess,
    build_parser,
    main as evaluate_main,
)
from survivalpfn.models import InContextModel, TabDPTLongContextModel
from survivalpfn.models.utils import HistogramDistribution
from survivalpfn.survival_estimator import (
    DEFAULT_MODEL_REPO_ID,
    SurvivalEstimator,
    _select_hf_checkpoint_file,
)


def test_survival_estimator_defaults_to_release_huggingface_checkpoint():
    estimator = SurvivalEstimator(device="cpu")
    assert estimator.model_path == DEFAULT_MODEL_REPO_ID


def test_evaluation_cli_defaults_to_release_huggingface_checkpoint():
    args = build_parser().parse_args(["--data", "ovarian"])
    assert args.model_path == DEFAULT_MODEL_REPO_ID
    assert args.preprocess is None
    assert args.metrics_backend == "paper"
    assert _resolve_preprocess("PBC", args.preprocess) is False
    assert _resolve_preprocess("MSKCC", args.preprocess) is True


def test_evaluation_cli_preserves_raw_benchmark_data_by_default(monkeypatch, tmp_path):
    captured = {}

    class DummyDataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __call__(self):
            return {
                "X_train": np.zeros((2, 1), dtype=np.float32),
                "T_train": np.array([1.0, 2.0], dtype=np.float32),
                "delta_train": np.array([1.0, 0.0], dtype=np.float32),
                "X_test": np.zeros((2, 1), dtype=np.float32),
                "T_test": np.array([1.5, 2.5], dtype=np.float32),
                "delta_test": np.array([1.0, 0.0], dtype=np.float32),
            }

    class DummyDistribution:
        def median(self):
            return torch.tensor([1.25, 2.25], dtype=torch.float32)

        def survival_function(self, time_grid):
            return torch.full((2, time_grid.shape[0]), 0.5, dtype=torch.float32)

        def survival_at(self, obs_time):
            return torch.full_like(obs_time, 0.5, dtype=torch.float32)

        def density_at(self, obs_time):
            return torch.full_like(obs_time, 0.25, dtype=torch.float32)

    class DummyEstimator:
        def __init__(self, **kwargs):
            self.device = kwargs["device"]

        def fit(self, **kwargs):
            return self

        def predict_event_distribution(self, X):
            return DummyDistribution()

    monkeypatch.setattr(
        "survivalpfn.inference.evaluate.SurvivalBenchmarkDataset", DummyDataset
    )
    monkeypatch.setattr(
        "survivalpfn.inference.evaluate.SurvivalEstimator", DummyEstimator
    )
    monkeypatch.setattr(
        "survivalpfn.inference.evaluate.harrell_concordance_index",
        lambda **kwargs: 0.5,
    )
    monkeypatch.setattr(
        "survivalpfn.inference.evaluate.integrated_brier_score",
        lambda **kwargs: 0.1,
    )
    monkeypatch.setattr("survivalpfn.inference.evaluate.mae", lambda **kwargs: 1.0)
    monkeypatch.setattr(
        "survivalpfn.inference.evaluate.logrank", lambda **kwargs: (0.5, 1.0)
    )
    monkeypatch.setattr(
        "survivalpfn.inference.evaluate.d_cal", lambda **kwargs: (1.0, 0.5, None)
    )
    monkeypatch.setattr("survivalpfn.inference.evaluate.nll_surv", lambda **kwargs: 0.3)

    evaluate_main(
        [
            "--data",
            "PBC",
            "--n-exp",
            "1",
            "--device",
            "cpu",
            "--metrics-backend",
            "lightweight",
            "--out",
            str(tmp_path / "eval.csv"),
        ]
    )

    assert captured["data_name"] == "PBC"
    assert captured["preprocess"] is False


def test_evaluation_cli_can_enable_preprocessing():
    args = build_parser().parse_args(["--data", "PBC", "--preprocess", "true"])
    assert args.preprocess is True



def test_huggingface_checkpoint_selection_prefers_v0_1(monkeypatch):
    def fake_list_repo_files(model_path, revision=None):
        assert model_path == DEFAULT_MODEL_REPO_ID
        assert revision == "main"
        return ["latest.pt", "survivalpfn_v0.1.pt", "survivalpfn_v0.pt"]

    monkeypatch.setattr(
        "survivalpfn.survival_estimator.list_repo_files", fake_list_repo_files
    )

    assert (
        _select_hf_checkpoint_file(DEFAULT_MODEL_REPO_ID, revision="main")
        == "survivalpfn_v0.1.pt"
    )
    assert (
        _select_hf_checkpoint_file(
            DEFAULT_MODEL_REPO_ID,
            revision="main",
            model_filename="custom-release.pt",
        )
        == "custom-release.pt"
    )


def test_incontext_model_loads_sanitized_release_config():
    nbins = 16
    base_model = TabDPTLongContextModel(
        dropout=0.0,
        n_out=10,
        nhead=2,
        nhid=32,
        ninp=16,
        nlayers=1,
        num_features=6,
        nbins=nbins,
    )
    wrapped = InContextModel(
        model=base_model,
        model_config={"model": {"nbins": nbins}},
        vmin=-3.0,
        vmax=7.0,
        query_strategy="both",
        time_transform="lognormal",
    )
    sanitized_config = {
        "model": {
            "use_cls_embedding": False,
            "emsize": 16,
            "max_num_classes": 10,
            "nbins": nbins,
            "max_num_features": 6,
            "nhead": 2,
            "nhid_factor": 2,
            "nlayers": 1,
            "norm_first": True,
        }
    }

    loaded = InContextModel.load(
        model_state=wrapped.state_dict(),
        model_config=sanitized_config,
    )

    assert loaded.model_config["query_strategy"] == "both"
    assert loaded.model_config["time_transform"] == "lognormal"
    assert loaded.model_config["sigma"] == 0.5
    assert loaded.vmin == -3.0
    assert loaded.vmax == 7.0


class DummyICLModel:
    def __init__(self, nbins: int = 5):
        self.nbins = nbins
        self.model_config = {
            "model": {
                "max_num_features": 4,  # estimator uses max_num_features - 1
                "nbins": nbins,
            }
        }
        self.last_query_dim = None

    def to(self, _device):
        return self

    def eval(self):
        return self

    def predict(
        self,
        X_context: torch.Tensor,
        delta_context: torch.Tensor,
        T_context: torch.Tensor,
        X_query: torch.Tensor,
        temperature: torch.Tensor,
    ):
        del X_context, delta_context, T_context, temperature
        self.last_query_dim = int(X_query.shape[-1])

        batch_size, query_size, _ = X_query.shape
        logits = torch.zeros(
            (batch_size, 1, query_size, self.nbins),
            device=X_query.device,
            dtype=torch.float32,
        )
        bin_edges = torch.linspace(
            0.0, 5.0, self.nbins, device=X_query.device, dtype=torch.float32
        )
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = bin_edges[:-1] + bin_width / 2.0
        bin_centers = (
            bin_centers.view(1, 1, 1, self.nbins - 1)
            .expand(batch_size, 1, query_size, self.nbins - 1)
            .clone()
        )
        bin_edges = (
            bin_edges.view(1, 1, 1, self.nbins)
            .expand(batch_size, 1, query_size, self.nbins)
            .clone()
        )
        return logits, bin_centers, bin_edges


def test_histogram_distribution_broadcasting_and_1d_support():
    logits = torch.zeros((2, 5), dtype=torch.float32)
    bin_centers = torch.stack(
        [
            torch.linspace(0.1, 1.0, 4),
            torch.linspace(0.2, 2.0, 4),
        ]
    )
    bin_edges = torch.stack(
        [
            torch.linspace(0.0, 1.1, 5),
            torch.linspace(0.0, 2.2, 5),
        ]
    )
    dist = HistogramDistribution(
        bin_centers=bin_centers, logits=logits, bin_edges=bin_edges
    )

    t = torch.tensor([0.1, 0.7, 1.3], dtype=torch.float32)
    density = dist.density_function(t)
    assert density.shape == (2, 3)

    dist_1d = HistogramDistribution(
        bin_centers=torch.linspace(0.1, 1.0, 4),
        bin_edges=torch.linspace(0.0, 1.1, 5),
        logits=torch.zeros(5, dtype=torch.float32),
    )
    surv = dist_1d.survival_function(torch.tensor([0.2, 0.8], dtype=torch.float32))
    assert surv.shape == (2,)


def test_histogram_distribution_density_normalizes_with_nonuniform_bins():
    logits = torch.zeros(6, dtype=torch.float32)  # uniform probabilities
    # Deliberately non-uniform spacing
    bin_centers = torch.tensor([0.2, 0.4, 0.9, 1.8, 3.2], dtype=torch.float32)
    bin_edges = torch.tensor(
        [0.0, 0.3, 0.5, 1.0, 2.0, 4.0], dtype=torch.float32
    )  # 6 edges for 5 finite bins
    dist = HistogramDistribution(
        bin_centers=bin_centers, logits=logits, bin_edges=bin_edges
    )

    density_at_centers = dist.density_function(bin_centers)
    integral = torch.sum(density_at_centers * dist.bin_widths)
    expected_finite_mass = torch.sum(dist.probs[:-1])
    assert torch.isclose(integral, expected_finite_mass, atol=1e-5)


def test_survival_estimator_applies_query_dim_transform_consistently():
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(32, 6)).astype(np.float32)
    T_train = np.abs(rng.normal(loc=2.0, scale=0.5, size=32)).astype(np.float32) + 1e-3
    delta_train = rng.integers(low=0, high=2, size=32).astype(np.float32)

    X_test = rng.normal(size=(8, 6)).astype(np.float32)
    T_test = np.abs(rng.normal(loc=2.0, scale=0.5, size=8)).astype(np.float32) + 1e-3

    dummy_model = DummyICLModel(nbins=7)
    estimator = SurvivalEstimator(device="cpu", model_path=None, icl_model=dummy_model)
    estimator.fit(X=X_train, delta=delta_train, T=T_train)

    pred_mean = estimator.predict_event_time(X_test)
    assert pred_mean.shape == (8,)
    assert dummy_model.last_query_dim == estimator.X_train.shape[1]

    surv_obs = estimator.survival_at_observed_time(X_test, T_test)
    dens_obs = estimator.event_density_at_obs(X_test, T_test)
    assert surv_obs.shape == (8,)
    assert dens_obs.shape == (8,)
    assert np.isfinite(surv_obs).all()
    assert np.isfinite(dens_obs).all()


def test_survival_estimator_imputes_before_feature_reduction():
    rng = np.random.default_rng(123)
    X_train = rng.normal(size=(32, 6)).astype(np.float32)
    X_train[0, 0] = np.nan
    X_test = rng.normal(size=(4, 6)).astype(np.float32)
    X_test[1, 2] = np.nan
    T_train = np.abs(rng.normal(loc=2.0, scale=0.5, size=32)).astype(np.float32) + 1e-3
    delta_train = rng.integers(low=0, high=2, size=32).astype(np.float32)

    dummy_model = DummyICLModel(nbins=7)
    estimator = SurvivalEstimator(device="cpu", model_path=None, icl_model=dummy_model)
    estimator.fit(X=X_train, delta=delta_train, T=T_train)

    pred = estimator.predict_event_time(X_test)
    assert pred.shape == (4,)
    assert np.isfinite(estimator.X_train).all()
    assert np.isfinite(pred).all()


def test_nll_surv_matches_manual_computation():
    pred_surv = np.array([0.8, 0.6, 0.9], dtype=np.float64)
    pred_dens = np.array([0.2, 0.4, 0.1], dtype=np.float64)
    delta = np.array([1.0, 0.0, 1.0], dtype=np.float64)

    expected = -np.mean(delta * np.log(pred_dens) + (1 - delta) * np.log(pred_surv))
    assert np.isclose(
        nll_surv(pred_surv=pred_surv, pred_dens=pred_dens, delta=delta), expected
    )
