"""Information-layer tests (offline, synthetic)."""
import numpy as np

from ffields.fields import EventDensityField
from ffields.geometry import Grid, Pitch
from ffields.information import (
    RateDistortionSignature,
    field_kl_divergence,
    move_self_information,
    shannon_entropy_bits,
)


def test_entropy_uniform_is_log2_k():
    assert np.isclose(shannon_entropy_bits(np.ones(8)), 3.0)
    assert np.isclose(shannon_entropy_bits(np.array([1.0, 0, 0, 0])), 0.0)


def test_kl_self_is_zero_and_nonnegative():
    g = Grid(Pitch(), nx=48, ny=32)
    rng = np.random.default_rng(0)
    pa = rng.normal([60, 34], [12, 8], size=(500, 2))
    pb = rng.normal([45, 34], [12, 8], size=(500, 2))
    fa = EventDensityField(g).estimate(pa)
    fb = EventDensityField(g).estimate(pb)
    assert abs(field_kl_divergence(fa, fa)) < 1e-9
    assert field_kl_divergence(fa, fb) > 0  # divergent distributions


def test_self_information_higher_in_rare_cells():
    g = Grid(Pitch(), nx=48, ny=32)
    rng = np.random.default_rng(0)
    # density concentrated near (80, 34)
    ref = EventDensityField(g).estimate(rng.normal([80, 34], [6, 5], size=(2000, 2)))
    si_common = move_self_information(np.array([[80.0, 34.0]]), ref)
    si_rare = move_self_information(np.array([[10.0, 5.0]]), ref)
    assert si_rare[0] > si_common[0]


def test_rate_distortion_monotonicity():
    rng = np.random.default_rng(0)
    # three tight clusters -> distortion should fall quickly with k
    pts = np.vstack([
        rng.normal([20, 20], 1.5, size=(300, 2)),
        rng.normal([60, 40], 1.5, size=(300, 2)),
        rng.normal([90, 60], 1.5, size=(300, 2)),
    ])
    sig = RateDistortionSignature(ks=(2, 4, 8, 16, 32)).fit(pts)
    d = [c["distortion_rmse_m"] for c in sig.curve]
    r = [c["rate_bits"] for c in sig.curve]
    # distortion non-increasing as k grows
    assert all(d[i] >= d[i + 1] - 1e-6 for i in range(len(d) - 1))
    # rate non-decreasing as k grows
    assert all(r[i] <= r[i + 1] + 1e-6 for i in range(len(r) - 1))
    s = sig.signature((5.0, 10.0))
    assert np.isfinite(s["rate_bits_at_5m"]) and np.isfinite(s["rate_bits_at_10m"])
    # finer localisation (5 m) costs at least as many bits as coarser (10 m)
    assert s["rate_bits_at_5m"] >= s["rate_bits_at_10m"] - 1e-6


def test_signature_stable_under_small_jitter():
    rng = np.random.default_rng(1)
    pts = np.vstack([
        rng.normal([30, 30], 5, size=(800, 2)),
        rng.normal([75, 45], 5, size=(800, 2)),
    ])
    base = RateDistortionSignature().fit(pts).rate_at_distortion(8.0)
    jit = RateDistortionSignature().fit(pts + rng.normal(0, 1.0, pts.shape)).rate_at_distortion(8.0)
    assert abs(base - jit) / base < 0.05  # stable to 1 m jitter
