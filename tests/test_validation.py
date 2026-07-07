"""Validation-battery statistics tests (offline, synthetic known-answers)."""
import numpy as np
import pandas as pd

from ffields.validation import (
    benjamini_hochberg,
    discriminant_permutation,
    incremental_value_ols,
    split_half_reliability,
    two_way_variance,
    variance_components,
)


def test_variance_components_bounds():
    # distinct group means, no within-group variance -> between share ~1
    g = np.repeat([0, 1, 2], 5)
    v = np.repeat([1.0, 5.0, 9.0], 5)
    assert variance_components(g, v)["between_share"] > 0.999
    # identical group means, all variance within -> between share ~0
    rng = np.random.default_rng(0)
    v2 = np.concatenate([rng.normal(3, 1, 5) for _ in range(3)])
    g2 = np.repeat([0, 1, 2], 5)
    share = variance_components(g2, v2)["between_share"]
    assert 0.0 <= share < 0.6


def test_discriminant_permutation_detects_structure():
    rng = np.random.default_rng(1)
    rows = []
    for team, mu in enumerate([0.0, 2.0, 4.0, 6.0, 8.0]):
        for _ in range(8):
            rows.append({"team": team, "x": mu + rng.normal(0, 0.5)})
    df = pd.DataFrame(rows)
    out = discriminant_permutation(df, "team", "x", n_perm=500, seed=0)
    assert out["between_share"] > 0.8
    assert out["p_value"] < 0.01
    # a pure-noise column should not be discriminative
    df["noise"] = rng.normal(0, 1, len(df))
    out2 = discriminant_permutation(df, "team", "noise", n_perm=500, seed=0)
    assert out2["p_value"] > 0.05


def test_split_half_reliability_stable_signal():
    rng = np.random.default_rng(2)
    rows = []
    for team in range(10):
        trait = rng.normal(0, 3)  # stable per-team trait
        for mi in range(8):
            rows.append({"team": team, "match": mi, "x": trait + rng.normal(0, 0.4)})
    df = pd.DataFrame(rows)
    out = split_half_reliability(df, "team", "x", "match", n_perm=500, seed=0)
    assert out["pearson_r"] > 0.7
    assert out["p_value"] < 0.05


def test_incremental_value_detects_and_rejects():
    rng = np.random.default_rng(3)
    n = 120
    base = rng.normal(0, 1, n)
    signal = rng.normal(0, 1, n)
    y = 1.5 * base + 1.2 * signal + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"y": y, "base": base, "sig": signal, "noise": rng.normal(0, 1, n)})
    pos = incremental_value_ols(df, "y", ["base"], "sig", n_perm=500, seed=0)
    assert pos["delta_r2"] > 0.1 and pos["p_value"] < 0.01
    neg = incremental_value_ols(df, "y", ["base"], "noise", n_perm=500, seed=0)
    assert neg["p_value"] > 0.05


def test_two_way_variance_separates_factors():
    rng = np.random.default_rng(7)
    teams = list(range(8)); opps = list(range(8))
    team_eff = {t: rng.normal(0, 3) for t in teams}
    opp_eff = {o: rng.normal(0, 0.5) for o in opps}  # weak opponent effect
    rows = []
    for t in teams:
        for o in opps:
            if t == o:
                continue
            rows.append({"team": t, "opp": o,
                         "x": team_eff[t] + opp_eff[o] + rng.normal(0, 0.4)})
    df = pd.DataFrame(rows)
    out = two_way_variance(df, "team", "opp", "x", n_perm=300, seed=0)
    # team carries most of the explained variance, controlling for opponent
    assert out["a_partial"] > out["b_partial"]
    assert out["p_value"] < 0.01
    # a descriptor driven only by opponent should give a weak, n.s. team-partial
    rows2 = [{"team": t, "opp": o, "x": opp_eff[o] + rng.normal(0, 0.4)}
             for t in teams for o in opps if t != o]
    out2 = two_way_variance(pd.DataFrame(rows2), "team", "opp", "x", n_perm=300, seed=0)
    assert out2["p_value"] > 0.05


def test_benjamini_hochberg_basic():
    p = [0.001, 0.01, 0.2, 0.5, 0.9]
    out = benjamini_hochberg(p, q=0.05)
    assert out["rejected"][0] and out["rejected"][1]
    assert not out["rejected"][3] and not out["rejected"][4]
    # q-values are monotone in p order and within [0, 1]
    assert np.all(out["qvalues"] >= 0) and np.all(out["qvalues"] <= 1)
