"""Validation battery statistics.

Reusable, dependency-light (numpy/pandas only) statistical tests for the
validation battery. The unit of analysis is always a (team, match) row -- never
an individual event -- so that resampling and cross-validation respect the
match-level dependence structure. Everything is associational; permutation nulls
and Benjamini-Hochberg FDR keep the multiple-comparison bookkeeping honest.

Provided tests
--------------
* ``variance_components`` / ``discriminant_permutation``: how much of a
  descriptor's variance is *between teams* vs within-team match-to-match noise
  (an eta^2 / ICC-style discriminant ratio), with a permutation null from
  shuffling team labels.
* ``split_half_reliability``: temporal stability -- correlation of per-team
  descriptor means computed on two disjoint halves of each team's matches, with
  a permutation null.
* ``incremental_value_ols``: confounder-controlled incremental value -- the
  extra variance in a match-level outcome explained by a descriptor *beyond* a
  baseline (e.g. possession volume), via nested OLS Delta-R^2 with a permutation
  p-value.
* ``benjamini_hochberg``: FDR control across the family of tests.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# -- discriminant validity ----------------------------------------------------
def variance_components(groups: np.ndarray, values: np.ndarray) -> dict[str, float]:
    """Between-group variance share (eta^2) of ``values`` grouped by ``groups``.

    eta^2 = SS_between / SS_total in [0, 1]; higher means the descriptor is more
    a property of the group (team) than of the individual observation (match).
    """
    g = np.asarray(groups)
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    g, v = g[ok], v[ok]
    if len(v) < 3:
        return {"between_share": float("nan"), "n_groups": 0, "n_total": int(len(v))}
    grand = v.mean()
    ss_total = float(np.sum((v - grand) ** 2))
    ss_between = 0.0
    uniq = pd.unique(g)
    for gg in uniq:
        vv = v[g == gg]
        ss_between += len(vv) * (vv.mean() - grand) ** 2
    share = ss_between / ss_total if ss_total > 0 else float("nan")
    return {"between_share": float(share), "n_groups": int(len(uniq)), "n_total": int(len(v))}


def discriminant_permutation(
    df: pd.DataFrame, group_col: str, value_col: str, n_perm: int = 2000, seed: int = 20260618
) -> dict[str, float]:
    """Observed between-group share plus a permutation null (shuffle groups)."""
    g = df[group_col].to_numpy()
    v = df[value_col].to_numpy(dtype=float)
    obs = variance_components(g, v)["between_share"]
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = variance_components(rng.permutation(g), v)["between_share"]
    p = (1 + np.sum(null >= obs)) / (n_perm + 1)
    return {
        "between_share": float(obs),
        "null_mean": float(np.nanmean(null)),
        "p_value": float(p),
        "n_perm": int(n_perm),
    }


# -- temporal stability -------------------------------------------------------
def split_half_reliability(
    df: pd.DataFrame, group_col: str, value_col: str, match_col: str,
    n_perm: int = 2000, seed: int = 20260618, min_per_half: int = 2,
) -> dict[str, float]:
    """Correlate per-group descriptor means across two disjoint match-halves.

    Matches of each group are sorted by ``match_col`` and split by parity
    (even-rank -> half A, odd-rank -> half B), a deterministic, reproducible
    split. Returns Pearson r and Spearman rho across groups present in both
    halves, with a permutation p-value (shuffling the half-B group labels).
    """
    a_vals, b_vals = {}, {}
    for gg, sub in df.groupby(group_col):
        sub = sub.sort_values(match_col)
        a = sub.iloc[0::2][value_col].to_numpy(dtype=float)
        b = sub.iloc[1::2][value_col].to_numpy(dtype=float)
        if len(a) >= min_per_half and len(b) >= min_per_half:
            a_vals[gg] = np.nanmean(a)
            b_vals[gg] = np.nanmean(b)
    common = [g for g in a_vals if g in b_vals]
    if len(common) < 4:
        return {"pearson_r": float("nan"), "spearman_rho": float("nan"),
                "p_value": float("nan"), "n_groups": len(common)}
    A = np.array([a_vals[g] for g in common])
    B = np.array([b_vals[g] for g in common])
    r = float(np.corrcoef(A, B)[0, 1])
    rho = float(_spearman(A, B))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = np.corrcoef(A, rng.permutation(B))[0, 1]
    p = (1 + np.sum(np.abs(null) >= abs(r))) / (n_perm + 1)
    return {"pearson_r": r, "spearman_rho": rho, "p_value": float(p), "n_groups": len(common)}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


# -- incremental value --------------------------------------------------------
def _ols_r2(X: np.ndarray, y: np.ndarray) -> float:
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def incremental_value_ols(
    df: pd.DataFrame, outcome: str, baseline_cols: list[str], test_col: str,
    n_perm: int = 2000, seed: int = 20260618,
) -> dict[str, float]:
    """Nested-OLS incremental value of ``test_col`` over ``baseline_cols``.

    Returns baseline and full R^2, their difference (Delta-R^2), and a
    permutation p-value obtained by shuffling the test predictor. Predictors are
    standardised; a column of ones is added for the intercept.
    """
    cols = baseline_cols + [test_col]
    d = df[[outcome] + cols].replace([np.inf, -np.inf], np.nan).dropna()
    y = d[outcome].to_numpy(dtype=float)
    n = len(y)
    if n < len(cols) + 3:
        return {"r2_base": float("nan"), "r2_full": float("nan"),
                "delta_r2": float("nan"), "p_value": float("nan"), "n": int(n)}

    def _std(a):
        a = np.asarray(a, float)
        s = a.std()
        return (a - a.mean()) / s if s > 0 else a - a.mean()

    Xb = np.column_stack([np.ones(n)] + [_std(d[c]) for c in baseline_cols])
    tcol = _std(d[test_col])
    Xf = np.column_stack([Xb, tcol])
    r2b = _ols_r2(Xb, y)
    r2f = _ols_r2(Xf, y)
    delta = r2f - r2b
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        Xp = np.column_stack([Xb, rng.permutation(tcol)])
        null[i] = _ols_r2(Xp, y) - r2b
    p = (1 + np.sum(null >= delta)) / (n_perm + 1)
    return {"r2_base": float(r2b), "r2_full": float(r2f), "delta_r2": float(delta),
            "p_value": float(p), "n": int(n)}


# -- confounder-controlled two-way decomposition ------------------------------
def _dummies(series: pd.Series) -> np.ndarray:
    """One-hot design (drop-first) for a categorical column, no intercept."""
    d = pd.get_dummies(series.astype("category"), drop_first=True)
    return d.to_numpy(dtype=float) if d.shape[1] else np.empty((len(series), 0))


def two_way_variance(
    df: pd.DataFrame, factor_a: str, factor_b: str, value_col: str,
    n_perm: int = 2000, seed: int = 20260618,
) -> dict[str, float]:
    """Partition a descriptor's variance between two crossed factors.

    Typical use: factor_a = team, factor_b = opponent. Using nested OLS on
    dummy designs, returns:

    * ``a_partial`` -- variance uniquely explained by A *controlling for* B
      (R^2_full - R^2_B): the team signal net of who they played;
    * ``b_partial`` -- variance uniquely explained by B controlling for A;
    * ``r2_full`` and ``residual`` (= 1 - R^2_full);
    * a permutation p-value for ``a_partial`` (shuffling A labels).

    This is the confounder-controlled successor to the one-way discriminant
    eta^2: it asks whether a descriptor reflects the team rather than the
    opponent/context they happened to face.
    """
    d = df[[factor_a, factor_b, value_col]].replace([np.inf, -np.inf], np.nan).dropna()
    y = d[value_col].to_numpy(dtype=float)
    n = len(y)
    if n < 8:
        return {"a_partial": float("nan"), "b_partial": float("nan"),
                "r2_full": float("nan"), "residual": float("nan"),
                "p_value": float("nan"), "n": int(n)}
    A = _dummies(d[factor_a]); B = _dummies(d[factor_b])
    one = np.ones((n, 1))

    def r2(X):
        return _ols_r2(X, y)

    r2_full = r2(np.column_stack([one, A, B]))
    r2_a = r2(np.column_stack([one, A]))
    r2_b = r2(np.column_stack([one, B]))
    a_partial = r2_full - r2_b
    b_partial = r2_full - r2_a

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    a_df = d[factor_a].to_numpy()
    for i in range(n_perm):
        Ap = _dummies(pd.Series(rng.permutation(a_df)))
        null[i] = r2(np.column_stack([one, Ap, B])) - r2_b
    p = (1 + np.sum(null >= a_partial)) / (n_perm + 1)
    return {"a_partial": float(a_partial), "b_partial": float(b_partial),
            "r2_full": float(r2_full), "residual": float(1.0 - r2_full),
            "p_value": float(p), "n": int(n)}


# -- multiple comparisons -----------------------------------------------------
def benjamini_hochberg(pvals: list[float], q: float = 0.05) -> dict[str, np.ndarray]:
    """Benjamini-Hochberg FDR. Returns rejection mask and adjusted q-values."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * m / (np.arange(1, m + 1))
    # enforce monotonicity of adjusted q-values
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    qvals = np.empty(m)
    qvals[order] = np.clip(adj, 0, 1)
    rejected = qvals <= q
    return {"rejected": rejected, "qvalues": qvals}
