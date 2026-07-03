import numpy as np
import pytest

from survivalpfn.inference.data import (
    ALL_DATA,
    SurvivalBenchmarkDataset,
    make_aids,
    make_bmt,
    make_metabric,
    make_pbc,
    make_support,
)


@pytest.mark.parametrize("data_name", ALL_DATA)
def test_eval_data_reproducible(data_name):
    data1 = SurvivalBenchmarkDataset(
        data_name=data_name,
        train_ratio=0.6,
        val_ratio=0.1,
        test_ratio=0.3,
        preprocess=False,
    )()
    data2 = SurvivalBenchmarkDataset(
        data_name=data_name,
        train_ratio=0.6,
        val_ratio=0.1,
        test_ratio=0.3,
        preprocess=False,
    )()

    for key in data1.keys():
        assert np.array_equal(data1[key], data2[key], equal_nan=True)


def test_support_drops_unique_row_identifier():
    df = make_support()
    assert "num_sno" not in df.columns
    assert "sno" not in df.columns


def test_pbc_keeps_continuous_labs_numeric():
    df = make_pbc()
    feature_columns = set(df.columns) - {"time", "event"}
    assert len(feature_columns) < 100
    assert "Drug" in feature_columns
    assert "Drug_Placebo" not in feature_columns
    assert "num_Platelets" in feature_columns
    assert not any(column.startswith("num_Platelets_") for column in feature_columns)


def test_bmt_uses_paper_loader_encoding():
    df = make_bmt()
    feature_columns = set(df.columns) - {"time", "event"}
    assert "Donorage35" not in feature_columns
    assert "Recipientage10" not in feature_columns
    assert "Recipientageint" not in feature_columns
    assert "num_Donorage" in feature_columns
    assert "DonorABO_A" in feature_columns
    assert "RecipientABO_O" in feature_columns


def test_aids_uses_event_indicator_not_censor_indicator():
    df = make_aids()
    assert np.isclose(df["event"].mean(), 1 - 0.24357176250584386)


def test_metabric_uses_paper_clinical_snapshot():
    df = make_metabric()
    feature_columns = set(df.columns) - {"time", "event"}
    assert df.shape == (1981, 81)
    assert "num_age_at_diagnosis" in feature_columns
    assert "num_NPI" in feature_columns
    assert "num_x0" not in feature_columns
    assert df["time"].max() > 9000
