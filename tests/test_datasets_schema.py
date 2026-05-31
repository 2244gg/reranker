"""Schema-level tests for dataset adapters.

Tests for ML-1M run against the in-repo data. Tests for the external
datasets are skipped if the raw files are not present.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.datasets.base import UserRecord
from src.datasets.factory import build_dataset


def _take_first_users(adapter, n: int = 100):
    users = adapter.load()
    return users[:n]


# ---------------------------------------------------------------------------
# ML-1M
# ---------------------------------------------------------------------------

def test_ml1m_demographic_schema():
    adapter = build_dataset("ml1m")
    schema = adapter.demographic_schema()
    # Default is 2 dims (gender + age_group); occupation is opt-in.
    assert schema == ["gender", "age_group"]
    assert adapter.domain() == "movie"


def test_ml1m_include_occupation_flag():
    adapter = build_dataset("ml1m", include_occupation=True)
    assert adapter.demographic_schema() == ["gender", "age_group", "occupation_group"]


def test_ml1m_users_loaded_with_demographics():
    adapter = build_dataset("ml1m")
    users = _take_first_users(adapter, 100)
    assert len(users) > 0
    for u in users:
        assert isinstance(u, UserRecord)
        assert "gender" in u.demographics
        assert "age_group" in u.demographics
        assert "occupation_group" not in u.demographics  # opt-in only
        assert u.demographics["gender"] in ("male", "female")
        assert u.demographics["age_group"] in ("young", "middle-aged", "elderly", "unknown")
        assert len(u.history) > 0


def test_ml1m_split_populates_test_set():
    adapter = build_dataset("ml1m")
    users = adapter.split(seed=42, test_size=0.7)
    has_test = sum(1 for u in users if u.test_set)
    assert has_test > 0
    # Re-split with same seed must be deterministic.
    users2 = adapter.split(seed=42, test_size=0.7)
    same = all(
        [u1.test_set] == [u2.test_set]
        for u1, u2 in zip(users[:50], users2[:50])
    )
    assert same


def test_ml1m_stratified_sample_balanced():
    adapter = build_dataset("ml1m")
    sample = adapter.stratified_sample(n_users=300, seed=42)
    assert 0 < len(sample) <= 300
    # All sampled users must be UserRecord with required demographic dims.
    for u in sample:
        for dim in adapter.demographic_schema():
            assert dim in u.demographics


def test_ml1m_legacy_only_two_dims():
    adapter = build_dataset("ml1m-legacy")
    assert adapter.demographic_schema() == ["gender", "age_group"]


def test_scenario_combinations_neutral_present():
    adapter = build_dataset("ml1m")
    scenarios = adapter.scenario_combinations()
    assert () in scenarios  # neutral
    # All single-dim scenarios.
    for d in adapter.demographic_schema():
        assert (d,) in scenarios
    # The full cross of all demographic dims.
    assert tuple(adapter.demographic_schema()) in scenarios


# ---------------------------------------------------------------------------
# External datasets - skipped when raw files missing
# ---------------------------------------------------------------------------

def _bx_available() -> bool:
    return Path("data/raw/bookcrossing/BX-Users.csv").exists()


def _lfm_available() -> bool:
    return Path("data/raw/lfm-2b/users.tsv").exists()


def _tenrec_available() -> bool:
    return Path("data/raw/tenrec/user_features.csv").exists()


@pytest.mark.skipif(not _bx_available(), reason="BookCrossing raw not present")
def test_bookcrossing_schema():
    adapter = build_dataset("bookcrossing")
    assert adapter.demographic_schema() == ["age_group", "country"]
    assert adapter.domain() == "book"
    users = _take_first_users(adapter, 50)
    assert all("age_group" in u.demographics for u in users)
    assert all("country" in u.demographics for u in users)


@pytest.mark.skipif(not _lfm_available(), reason="LFM-2b raw not present")
def test_lfm_schema():
    adapter = build_dataset("lfm")
    assert adapter.demographic_schema() == ["gender", "age_group", "country"]
    assert adapter.domain() == "music"


@pytest.mark.skipif(not _tenrec_available(), reason="Tenrec raw not present")
def test_tenrec_schema():
    adapter = build_dataset("tenrec")
    assert adapter.demographic_schema() == ["gender", "age_group"]
    assert adapter.domain() == "video"


def test_factory_unknown_dataset_raises():
    with pytest.raises(ValueError):
        build_dataset("nonexistent-dataset")
