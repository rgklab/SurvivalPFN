import math

import pytest
import torch

from survivalpfn.models import InContextModel, TabDPTLongContextModel
from survivalpfn.models.constants import EPSILON_STABILITY
from survivalpfn.models.utils import quantile_grids, quantile_transform

NUM_FEATURES = 8
N_BINS = 5000
VMIN = -15.0
VMAX = 15.0

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
FORK_RNG_DEVICES = [0] if torch.cuda.is_available() else []


def _make_test_model(
    time_transform: str = "quantile",
    query_strategy: str = "random",
    nbins: int = N_BINS,
) -> InContextModel:
    tabdpt_backbone = TabDPTLongContextModel(
        dropout=0.1,
        nhead=4,
        nhid=12,
        ninp=8,
        nlayers=3,
        num_features=NUM_FEATURES + 1,
        n_out=10,
        nbins=nbins,
    )
    return InContextModel(
        model=tabdpt_backbone,
        model_config={"model": {"nbins": nbins}},
        vmin=VMIN,
        vmax=VMAX,
        query_strategy=query_strategy,
        time_transform=time_transform,
    )


with torch.random.fork_rng(devices=FORK_RNG_DEVICES):
    torch.manual_seed(1342)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    models = []
    models_by_transform = {}
    for time_transform in ["lognormal", "quantile"]:
        tabdpt = _make_test_model(time_transform=time_transform)
        models_by_transform[time_transform] = tabdpt
        models.append(pytest.param(tabdpt, id=time_transform))


def test_quantile_ecdf_transform_collapses_ties_and_roundtrips_inside_support():
    T_context = torch.tensor([[1.0, 1.0, 2.0, 4.0], [2.0, 3.0, 3.0, 5.0]])
    state = quantile_grids(T_context)

    assert torch.equal(state.lengths.cpu(), torch.tensor([4, 4]))
    assert torch.allclose(state.time_grid[0, :4], torch.tensor([0.0, 1.0, 2.0, 4.0]))
    assert torch.allclose(
        state.quantile_grid[0, :4], torch.tensor([0.0, 0.5, 0.75, 1.0])
    )
    assert torch.allclose(state.time_grid[1, :4], torch.tensor([0.0, 2.0, 3.0, 5.0]))
    assert torch.allclose(
        state.quantile_grid[1, :4], torch.tensor([0.0, 0.25, 0.75, 1.0])
    )

    samples = torch.tensor(
        [
            [0.0, 0.5, 1.0, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
            [0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 10.0],
        ]
    )
    quantiles = quantile_transform(samples, state=state, type="time2quantile")

    expected_quantiles = torch.tensor(
        [
            [0.0, 0.25, 0.5, 0.5, 0.625, 0.75, 0.875, 1.0, 1.0],
            [0.0, 0.125, 0.25, 0.75, 0.75, 0.875, 1.0, 1.0, 1.0],
        ]
    )
    assert torch.allclose(quantiles, expected_quantiles)

    restored = quantile_transform(quantiles, state=state, type="quantile2time")
    max_times = T_context.max(dim=1).values.unsqueeze(-1)
    assert torch.allclose(
        restored, torch.minimum(samples, max_times), rtol=1e-4, atol=1e-5
    )


def test_quantile_model_config_and_old_name_rejection():
    model = models_by_transform["quantile"]
    assert model.vmin == 0.0
    assert model.vmax == 1.0
    assert model.model_config["vmin"] == 0.0
    assert model.model_config["vmax"] == 1.0
    assert model.loss_sigma == 0.05
    assert torch.allclose(model.bin_edges, torch.linspace(0.0, 1.0, N_BINS))

    with pytest.raises(
        ValueError, match="Supported transforms are 'quantile' and 'lognormal'"
    ):
        _make_test_model(time_transform="quantile_normal")


def test_quantile_prediction_residual_tail_starts_at_max_context_time():
    model = _make_test_model(time_transform="quantile", nbins=128)
    model.eval()

    b_size = 2
    n_train = 4
    n_test = 3
    X_context = torch.randn(b_size, n_train, NUM_FEATURES)
    X_query = torch.randn(b_size, n_test, NUM_FEATURES)
    T_context = torch.tensor([[1.0, 1.0, 2.0, 4.0], [2.0, 3.0, 3.0, 5.0]])
    delta_context = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])

    logits, bin_centers, bin_edges = model.predict(
        X_context, delta_context, T_context, X_query, torch.tensor([1.0, 2.0])
    )

    assert logits.shape == (b_size, 2, n_test, 128)
    assert bin_centers.shape == (b_size, 2, n_test, 127)
    assert bin_edges.shape == (b_size, 2, n_test, 128)
    assert torch.allclose(bin_edges[..., 0], torch.zeros_like(bin_edges[..., 0]))

    expected_tail_start = T_context.max(dim=1).values + EPSILON_STABILITY
    assert torch.allclose(
        bin_edges[..., -1],
        expected_tail_start[:, None, None].expand_as(bin_edges[..., -1]),
    )
    assert torch.all(bin_centers[..., -1] < bin_edges[..., -1])


@pytest.mark.parametrize("query_strategy", ["random", "event", "both", "both_fix_len"])
def test_quantile_losses_support_query_strategies(query_strategy: str):
    with torch.random.fork_rng(devices=FORK_RNG_DEVICES):
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True

        model = _make_test_model(
            time_transform="quantile", query_strategy=query_strategy, nbins=128
        )
        model.eval()
        b_size = 4
        n_train = 6
        n_test = 5
        X_context = torch.randn(b_size, n_train, NUM_FEATURES)
        T_context = torch.rand(b_size, n_train) + 0.1
        delta_context = torch.randint(0, 2, (b_size, n_train)).float()
        X_query = torch.randn(b_size, n_test, NUM_FEATURES)
        E_query = torch.rand(b_size, n_test) + 0.1
        C_query = torch.rand(b_size, n_test) + 0.1

        losses = model.survival_losses(
            X_context, delta_context, T_context, X_query, E_query, C_query
        )

        assert losses.shape == (b_size,)
        assert torch.isfinite(losses).all()


@pytest.mark.parametrize("model", models)
def test_shapes(model: InContextModel):
    b_size = 5
    n_train = 10
    n_test = 15
    X_context = torch.randn(b_size, n_train, NUM_FEATURES)
    X_query = torch.randn(b_size, n_test, NUM_FEATURES)
    T_context = torch.randn(b_size, n_train).abs()
    delta_context = torch.randint(0, 2, (b_size, n_train)).float()

    logits, bin_centers, bin_edges = model.predict(
        X_context, delta_context, T_context, X_query, torch.tensor([1.0, 2.0])
    )

    assert logits.shape == (
        b_size,
        2,
        n_test,
        N_BINS,
    ), f"Expected shape {(b_size, 2, n_test, N_BINS)}, got {logits.shape}"
    assert bin_centers.shape == (
        b_size,
        2,
        n_test,
        N_BINS - 1,
    ), f"Expected shape {(b_size, 2, n_test, N_BINS - 1)}, got {bin_centers.shape}"
    assert bin_edges.shape == (
        b_size,
        2,
        n_test,
        N_BINS,
    ), f"Expected shape {(b_size, 2, n_test, N_BINS)}, got {bin_edges.shape}"


@pytest.mark.parametrize("model", models)
def test_losses(model: InContextModel):
    with torch.random.fork_rng(devices=FORK_RNG_DEVICES):
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True

        model.eval()
        b_size = 8
        n_train = 10
        n_test = 15
        X_context = torch.randn(b_size, n_train, NUM_FEATURES)
        T_context = torch.randn(b_size, n_train).abs()
        delta_context = torch.randint(0, 2, (b_size, n_train)).float()

        X_query = torch.randn(b_size, n_test, NUM_FEATURES)

        E_query = torch.randn(b_size, n_test).abs()
        C_query = E_query + 1.0  # ensure uncensored targets (delta_query == 1)

        losses = (
            model.survival_losses(
                X_context, delta_context, T_context, X_query, E_query, C_query
            )
            .detach()
            .cpu()
        )

        assert losses.shape == (b_size,), (
            f"Expected shape {(b_size,)}, got {losses.shape}"
        )

        assert torch.allclose(losses, torch.tensor(math.log(N_BINS)), rtol=0.1), (
            f"Expected {math.log(N_BINS)}, got {losses}"
        )


@pytest.mark.parametrize("model", models)
def test_query_permutation_invariance(model: InContextModel):
    with torch.random.fork_rng(devices=FORK_RNG_DEVICES):
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
        model.eval()
        b_size = 8
        n_train = 10
        n_test = 15
        X_context = torch.randn(b_size, n_train, NUM_FEATURES)
        T_context = torch.randn(b_size, n_train).abs()
        delta_context = torch.randint(0, 2, (b_size, n_train)).float()

        X_query = torch.randn(b_size, n_test, NUM_FEATURES)
        perm = torch.randperm(n_test)

        outcome1, _, _ = model.predict(
            X_context, delta_context, T_context, X_query, torch.tensor([1.0, 2.0])
        )
        outcome2, _, _ = model.predict(
            X_context,
            delta_context,
            T_context,
            X_query[:, perm],
            torch.tensor([1.0, 2.0]),
        )

        assert torch.allclose(outcome1[:, :, perm], outcome2), (
            f"Expected {outcome1[:, perm]}, got {outcome2} after permutation of query data"
        )

        outcome3, _, _ = model.predict(
            X_context,
            delta_context,
            T_context,
            X_query[:, :1],
            torch.tensor([1.0, 2.0]),
        )

        assert torch.allclose(outcome1[:, :, :1], outcome3, atol=1e-2), (
            f"Expected {outcome1[:, :1]}, got {outcome3} after permutation of query data"
        )
