"""Tests for ms2pip_features_from_prediction_peak_arrays (incl. similarity features)."""

import numpy as np
import pytest
from ms2rescore_rs import ms2pip_features_from_prediction_peak_arrays

SIMILARITY = [
    "spectrast", "spectrast_ionb", "spectrast_iony",
    "spectral_angle", "spectral_angle_ionb", "spectral_angle_iony",
]
MZ_WEIGHTED = [
    "weighted_dotprod", "weighted_dotprod_ionb", "weighted_dotprod_iony", "nist_match_factor",
]


def _log2(x):
    return np.log2(np.asarray(x, dtype=np.float32) + 0.001).astype(np.float32)


@pytest.fixture
def arrays():
    return dict(
        psm_indices=[0],
        predicted_b=[_log2([0.20, 0.05, 0.10])],
        predicted_y=[_log2([0.30, 0.15, 0.20])],
        observed_b=[_log2([0.25, 0.04, 0.12])],
        observed_y=[_log2([0.28, 0.10, 0.21])],
    )


def _reference_spectral_angle(x, y, eps=1e-7):
    xn = x / np.sqrt(max(np.sum(x**2), eps))
    yn = y / np.sqrt(max(np.sum(y**2), eps))
    return 1 - 2 * np.arccos(np.clip(np.dot(xn, yn), -1, 1)) / np.pi


def _reference_weighted_dotprod(mz, x, y):
    a, b = np.sqrt(x) * mz, np.sqrt(y) * mz
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def test_similarity_features_without_mz(arrays):
    (idx, feats), = ms2pip_features_from_prediction_peak_arrays(**arrays)
    assert idx == 0
    for name in SIMILARITY:
        assert name in feats and f"{name}_norm" in feats
    for name in MZ_WEIGHTED:
        assert name not in feats and f"{name}_norm" not in feats


def test_similarity_features_with_mz(arrays):
    mz_b, mz_y = np.array([175.119, 288.203, 401.287]), np.array([147.113, 260.197, 373.281])
    (_, feats), = ms2pip_features_from_prediction_peak_arrays(
        **arrays, theoretical_mz_b=[mz_b], theoretical_mz_y=[mz_y]
    )
    for name in SIMILARITY + MZ_WEIGHTED:
        assert name in feats and f"{name}_norm" in feats
    unlog = lambda a: 2 ** np.clip(a.astype(np.float64), np.log2(0.001), None) - 0.001  # noqa: E731
    pred = np.concatenate([unlog(arrays["predicted_b"][0]), unlog(arrays["predicted_y"][0])])
    obs = np.concatenate([unlog(arrays["observed_b"][0]), unlog(arrays["observed_y"][0])])
    mz = np.concatenate([mz_b, mz_y])
    assert feats["spectral_angle"] == pytest.approx(_reference_spectral_angle(pred, obs), abs=1e-6)
    assert feats["weighted_dotprod"] == pytest.approx(
        _reference_weighted_dotprod(mz, pred, obs), abs=1e-6
    )
    cos = np.dot(pred, obs) / (np.linalg.norm(pred) * np.linalg.norm(obs))
    assert feats["spectrast"] == pytest.approx(cos**2, abs=1e-6)
    assert 0 < feats["nist_match_factor"] <= 1


def test_mz_arrays_must_be_given_together(arrays):
    with pytest.raises(ValueError):
        ms2pip_features_from_prediction_peak_arrays(
            **arrays, theoretical_mz_b=[np.array([1.0, 2.0, 3.0])]
        )
