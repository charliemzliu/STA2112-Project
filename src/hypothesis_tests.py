"""Invariant causal prediction (ICP) tests from Peters, Buehlmann & Meinshausen.

Implements Method I (held-out-environment Gaussian F test) and Method II
(fast, approximate pooled-residual mean/variance tests), Section 3.1.

Expected InterventionData interface
----------------------------------
- data.p: number of predictors
- data.E: iterable of environment labels
- data.get_environment(e, S, add_intercept=True) -> (X_e, y_e)
- data.get_complement(e, S, add_intercept=True) -> (X_minus_e, y_minus_e)
- data.get_all(S) -> (X_pooled, y_pooled), with intercept in column 0

S consists of zero-based predictor column indices (intercept excluded).
No knowledge of intervention targets or data-generation mechanism is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable

import numpy as np
from scipy import stats
from scipy.linalg import solve


def _subset(data: Any, S: Iterable[int]) -> tuple[int, ...]:
    S = tuple(int(j) for j in S)
    if len(S) != len(set(S)) or any(j < 0 or j >= data.p for j in S):
        raise ValueError("S must contain distinct, zero-based predictor indices.")
    return S


def _environments(data: Any) -> tuple[Any, ...]:
    envs = tuple(data.E)
    if len(envs) < 2:
        raise ValueError("ICP requires at least two environments.")
    return envs


def _ols(X: np.ndarray, y: np.ndarray):
    """OLS with an explicit full-column-rank check."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.shape[0] != y.size:
        raise ValueError("The design matrix and response have different row counts.")
    n, k = X.shape
    if n <= k:
        raise ValueError(f"Need n > number of fitted parameters; got n={n}, k={k}.")
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    if rank != k:
        raise ValueError("OLS design matrix is not full column rank.")
    residuals = y - X @ beta
    mse = float(residuals @ residuals / (n - k))
    if mse <= 0:
        raise ValueError("Estimated noise variance is zero; F/t tests are undefined.")
    return beta, residuals, mse


def method1_pvalue(data: Any, S: Iterable[int]) -> float:
    """Method I: environment-versus-complement prediction F tests.

    For each environment e, fit OLS on its complement, predict in e, and
    test the F statistic in equation (17). Bonferroni-adjust across e.
    Assumes independent Gaussian errors and full rank in each complement.
    """
    S = _subset(data, S)
    envs = _environments(data)
    p_values = []
    for e in envs:
        Xe, ye = data.get_environment(e, S, add_intercept=True)
        Xt, yt = data.get_complement(e, S, add_intercept=True)
        Xe, ye = np.asarray(Xe, dtype=float), np.asarray(ye, dtype=float).reshape(-1)
        Xt, yt = np.asarray(Xt, dtype=float), np.asarray(yt, dtype=float).reshape(-1)
        if len(ye) == 0:
            raise ValueError(f"Environment {e!r} has no observations.")
        try:
            beta, _, sigma2 = _ols(Xt, yt)
        except ValueError as ex:
            raise ValueError(f"Method I cannot fit complement of environment {e!r}: {ex}") from ex

        D = ye - Xe @ beta
        A = Xt.T @ Xt
        # Woodbury: (I + Xe A^{-1} Xe.T)^{-1}
        #           = I - Xe (A + Xe.T Xe)^{-1} Xe.T.
        v = Xe.T @ D
        q = float(D @ D - v @ solve(A + Xe.T @ Xe, v, assume_a="pos"))
        # Only remove negative roundoff error, not a substantively bad value.
        if q < -1e-7 * max(1.0, float(D @ D)):
            raise ArithmeticError("Negative prediction quadratic form (numerical instability).")
        q = max(q, 0.0)
        ne = len(ye)
        df_train = len(yt) - Xt.shape[1]
        statistic = q / (ne * sigma2)
        p_values.append(float(stats.f.sf(statistic, ne, df_train)))

    return float(min(1.0, len(envs) * min(p_values)))


def method2_pvalue(data: Any, S: Iterable[int]) -> float:
    """Method II: pooled OLS, residual mean t tests and variance F tests.

    Follows Section 3.1: Bonferroni across environments separately for
    means and variances, then Bonferroni across these two test families.
    Approximate only: pooled fitted residuals are dependent and coefficient
    estimation uncertainty is ignored by the residual tests.
    """
    S = _subset(data, S)
    envs = _environments(data)
    X, y = data.get_all(S)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    beta, residuals, _ = _ols(X, y)
    labels = np.asarray(data.environments)
    if labels.shape != (y.size,):
        raise ValueError("data.environments must contain one label per observation.")

    p_means = []
    p_variances = []
    for e in envs:
        a, b = residuals[labels == e], residuals[labels != e]
        if min(a.size, b.size) < 2:
            raise ValueError(f"Environment {e!r} or its complement has fewer than 2 observations.")
        # Classic pooled-variance two-sample t test, as the null assumes equal variances.
        p_mean = float(stats.ttest_ind(a, b, equal_var=True).pvalue)
        variance_ratio = float(np.var(a, ddof=1) / np.var(b, ddof=1))
        p_var = 2 * min(
            stats.f.cdf(variance_ratio, a.size - 1, b.size - 1),
            stats.f.sf(variance_ratio, a.size - 1, b.size - 1),
        )
        if not (np.isfinite(p_mean) and np.isfinite(p_var)):
            raise ValueError("Residual tests produced a non-finite p-value.")
        p_means.append(p_mean)
        p_variances.append(p_var)

    p_equal_means = min(1.0, len(envs) * min(p_means))
    p_equal_variances = min(1.0, len(envs) * min(p_variances))
    return float(min(1.0, 2.0 * min(p_equal_means, p_equal_variances)))


def pooled_rectangular_ci(data: Any, S: Iterable[int], alpha: float = 0.05) -> dict[int, tuple[float, float]]:
    """Paper's pooled (1-alpha) rectangular, Bonferroni OLS region.

    Returns intervals for coefficients of S only. Coefficients outside S are
    fixed at 0; the intercept is fitted but is not included in the output.
    """
    S = _subset(data, S)
    if not (0 < alpha < 1):
        raise ValueError("alpha must be strictly between 0 and 1.")
    if not S:
        return {}
    X, y = data.get_all(S)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.shape[1] != len(S) + 1:
        raise ValueError("get_all(S) must return a design with an intercept in column 0.")
    beta, _, sigma2 = _ols(X, y)
    df = X.shape[0] - X.shape[1]
    # Solve rather than constructing an explicit inverse.
    inv_gram = solve(X.T @ X, np.eye(X.shape[1]), assume_a="pos")
    se = np.sqrt(sigma2 * np.diag(inv_gram)[1:])
    cutoff = stats.t.ppf(1 - alpha / (2 * len(S)), df)
    return {j: (float(b - cutoff * s), float(b + cutoff * s))
            for j, b, s in zip(S, beta[1:], se)}


@dataclass
class ICPResult:
    selected: set[int]
    accepted_sets: list[tuple[int, ...]]
    pvalues: dict[tuple[int, ...], float]
    confidence_regions: dict[tuple[int, ...], dict[int, tuple[float, float]]]
    any_model_accepted: bool


def invariant_prediction(
    data: Any,
    *,
    method: str = "I",
    alpha: float = 0.05,
    max_size: int | None = None,
    compute_ci: bool = True,
) -> ICPResult:
    """Test all specified candidate subsets and intersect non-rejected sets.

    max_size=None enumerates all 2**p subsets, as in the original paper.
    Restricting max_size can invalidate the coverage guarantee unless the
    true causal set is known to lie in the enumerated candidate family.
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be strictly between 0 and 1.")
    tests = {"I": method1_pvalue, "II": method2_pvalue}
    if method not in tests:
        raise ValueError("method must be 'I' or 'II'.")
    if max_size is None:
        max_size = data.p
    if not (0 <= max_size <= data.p):
        raise ValueError("max_size must be between 0 and data.p.")

    accepted = []
    pvalues = {}
    regions = {}
    for size in range(max_size + 1):
        for S in combinations(range(data.p), size):
            pvalue = tests[method](data, S)
            pvalues[S] = pvalue
            if pvalue >= alpha:
                accepted.append(S)
                if compute_ci:
                    regions[S] = pooled_rectangular_ci(data, S, alpha)

    selected = set.intersection(*(set(S) for S in accepted)) if accepted else set()
    return ICPResult(selected, accepted, pvalues, regions, bool(accepted))
