"""
src/regime/risk_checks.py — Pure quantitative risk calculation functions.

Standalone functions with no I/O and no external service dependency.
`detect_market_regime` is the authoritative regime classifier used by the
decision layer (src/decision/signal_aggregator.py).

Functions:
    fit_student_t              — MLE fit of Student-t to log returns
    compute_var_cvar           — VaR and CVaR at 95% and 99%
    detect_market_regime       — Regime classification from vol ratio
    compute_asymmetric_volatility — Downside/upside vol decomposition
    select_vol_window          — Out-of-sample window-length selection via Kupiec POF
"""

import math

import numpy as np
from scipy import stats


def fit_student_t(log_returns: list[float]) -> dict:
    """
    Fit a Student-t distribution to log returns using MLE.

    Returns fitted df, mu, sigma, KS test result, tail classification,
    skewness, excess_kurtosis, and annualized_vol.

    Raises ValueError if fewer than 30 observations are provided.
    """
    arr = np.array(log_returns, dtype=float)
    if len(arr) < 30:
        raise ValueError(f"Insufficient data: need at least 30 observations, got {len(arr)}.")

    df, mu, sigma = stats.t.fit(arr)
    ks_stat, ks_p = stats.kstest(arr, "t", args=(df, mu, sigma))
    skew = float(stats.skew(arr))
    kurt = float(stats.kurtosis(arr))
    tail = "fat" if df < 5 else ("moderate" if df < 10 else "near-normal")

    return {
        "df": round(float(df), 4),
        "mu": round(float(mu), 8),
        "sigma": round(float(sigma), 8),
        "ks_stat": round(float(ks_stat), 4),
        "ks_pvalue": round(float(ks_p), 4),
        "fit_quality": "good" if ks_p > 0.05 else "poor",
        "tail_heaviness": tail,
        "skewness": round(skew, 4),
        "excess_kurtosis": round(kurt, 4),
        "n": len(arr),
        "annualized_vol": round(float(sigma) * float(np.sqrt(8760)), 6),
    }


def compute_var_cvar(df: float, mu: float, sigma: float) -> dict:
    """
    Compute 90%, 95%, and 99% VaR and CVaR (Expected Shortfall) from Student-t
    parameters.

    VaR and CVaR are expressed as log returns (negative = losses).
    Uses numerical integration for accurate CVaR on fat-tailed distributions.

    Returns a dict keyed by confidence level ("0.9", "0.95", "0.99").
    """
    results = {}
    for level in [0.90, 0.95, 0.99]:
        var_val = float(stats.t.ppf(1.0 - level, df, loc=mu, scale=sigma))
        x_min = float(stats.t.ppf(1e-6, df, loc=mu, scale=sigma))
        x_grid = np.linspace(x_min, var_val, 3000)
        pdf_vals = stats.t.pdf(x_grid, df, loc=mu, scale=sigma)
        prob_below = float(stats.t.cdf(var_val, df, loc=mu, scale=sigma))
        cvar_val = (
            float(np.trapezoid(x_grid * pdf_vals, x_grid) / prob_below)
            if prob_below > 0
            else var_val
        )
        results[str(level)] = {
            "var": round(var_val, 8),
            "cvar": round(cvar_val, 8),
        }
    return results


def detect_market_regime(log_returns: list[float]) -> dict:
    """
    Detect current market regime by comparing recent volatility to the
    historical baseline.

    Returns regime label (stress / high_vol / normal / low_vol), confidence,
    vol_ratio, skewness, and excess_kurtosis.
    """
    arr = np.array(log_returns, dtype=float)
    overall_vol = float(np.std(arr))
    window = max(1, min(168, len(arr) // 4))
    recent_vol = float(np.std(arr[-window:])) if len(arr) >= window else overall_vol
    vol_ratio = recent_vol / overall_vol if overall_vol > 0 else 1.0
    skew = float(stats.skew(arr))
    kurt = float(stats.kurtosis(arr))
    mean_ret = float(np.mean(arr))

    # Kurtosis threshold calibrated for crypto: hourly ETH/BTC returns have
    # structural excess kurtosis of 10–30 due to fat tails. Threshold of 5
    # (appropriate for equities) classifies all crypto periods as "stress".
    # 15 retains stress classification only for genuinely extreme distributions.
    if vol_ratio > 1.5 or abs(skew) > 1.5 or kurt > 30:
        regime = "stress"
        confidence = min(0.92, 0.65 + abs(vol_ratio - 1.0) * 0.2 + min(abs(skew), 2) * 0.05)
    elif vol_ratio > 1.2:
        regime = "high_vol"
        confidence = min(0.85, 0.6 + (vol_ratio - 1.0) * 0.5)
    elif vol_ratio < 0.7:
        regime = "low_vol"
        confidence = min(0.82, 0.6 + (1.0 - vol_ratio) * 0.5)
    else:
        regime = "normal"
        confidence = 0.80

    return {
        "regime": regime,
        "confidence": round(confidence, 3),
        "recent_vol_window": window,
        "recent_vol": round(recent_vol, 8),
        "overall_vol": round(overall_vol, 8),
        "vol_ratio": round(vol_ratio, 3),
        "skewness": round(skew, 4),
        "excess_kurtosis": round(kurt, 4),
        "mean_return": round(mean_ret, 8),
    }


def compute_asymmetric_volatility(log_returns: list[float]) -> dict:
    """
    Compute downside semi-deviation and asymmetric risk metrics.

    Returns downside_vol, upside_vol, downside_upside_ratio,
    sortino_ratio (annualized), max_drawdown_log, and pct_negative_returns.
    """
    arr = np.array(log_returns, dtype=float)
    mu = float(np.mean(arr))
    downside = arr[arr < mu]
    upside = arr[arr >= mu]
    down_vol = float(np.sqrt(np.mean((downside - mu) ** 2))) if len(downside) > 1 else 0.0
    up_vol = float(np.sqrt(np.mean((upside - mu) ** 2))) if len(upside) > 1 else 0.0
    ratio = round(down_vol / up_vol, 4) if up_vol > 0 else 0.0
    sortino = float((mu / down_vol) * np.sqrt(8760)) if down_vol > 0 else 0.0
    cumulative = np.cumsum(arr)
    rolling_max = np.maximum.accumulate(cumulative)
    max_dd = float(np.min(cumulative - rolling_max))

    return {
        "downside_vol": round(down_vol, 8),
        "upside_vol": round(up_vol, 8),
        "downside_upside_ratio": ratio,
        "sortino_ratio_annualized": round(sortino, 4),
        "max_drawdown_log": round(max_dd, 6),
        "pct_negative_returns": round(float(np.mean(arr < 0) * 100), 2),
        "n_downside": int(len(downside)),
        "n_upside": int(len(upside)),
    }


def full_risk_profile(log_returns: list[float]) -> dict:
    """
    Run all four risk checks on a return series and return a single
    combined dict. Convenient entry point for strategy evaluation.

    Keys: "regime", "distribution", "var_cvar", "asymmetric_vol"
    """
    t_fit = fit_student_t(log_returns)
    return {
        "regime": detect_market_regime(log_returns),
        "distribution": t_fit,
        "var_cvar": compute_var_cvar(t_fit["df"], t_fit["mu"], t_fit["sigma"]),
        "asymmetric_vol": compute_asymmetric_volatility(log_returns),
    }


def select_vol_window(
    log_returns: list[float],
    candidate_windows: list[int] | None = None,
    test_frac: float = 0.2,
    confidence: float = 0.99,
) -> dict:
    """
    Select the best rolling estimation window for VaR via out-of-sample
    Kupiec Proportion of Failures (POF) test.

    For each candidate window W:
      - Walk-forward over the out-of-sample region (last test_frac of data)
      - At each step, fit Student-t to the preceding W observations
      - Compute 1-day-ahead VaR; record whether the next return breaches it
      - Score by |empirical_violation_rate - nominal_rate|  (lower = better)
      - Report Kupiec POF LR statistic and p-value for each window

    Candidate windows are in hours (default: [500, 1000, 2000, 4000, 6000, 12000]).
    Requires at least W + 30 total observations for the smallest window.

    Returns:
        best_window         — window length (hours) with lowest POF deviation
        confidence          — VaR confidence level used
        nominal_rate        — expected violation rate (1 - confidence)
        results             — list of dicts per window, sorted best → worst:
            window          — hours
            n_tests         — number of OOS VaR forecasts evaluated
            n_violations    — actual violations
            empirical_rate  — n_violations / n_tests
            deviation       — |empirical_rate - nominal_rate|
            kupiec_lr       — Kupiec POF likelihood-ratio statistic
            kupiec_pvalue   — p-value (H0: empirical_rate == nominal_rate)
            rejected        — True if H0 rejected at 5% (model misspecified)
    """
    arr = np.array(log_returns, dtype=float)
    n = len(arr)

    if candidate_windows is None:
        candidate_windows = [500, 1000, 2000, 4000, 6000, 12000]

    nominal = 1.0 - confidence
    test_start = int(n * (1.0 - test_frac))

    results = []
    for W in candidate_windows:
        if test_start < W + 30:
            results.append(
                {
                    "window": W,
                    "n_tests": 0,
                    "n_violations": 0,
                    "empirical_rate": None,
                    "deviation": None,
                    "kupiec_lr": None,
                    "kupiec_pvalue": None,
                    "rejected": None,
                    "note": "insufficient data",
                }
            )
            continue

        violations = 0
        n_tests = 0
        for t in range(test_start, n):
            window_returns = arr[t - W : t]
            try:
                fit = fit_student_t(window_returns.tolist())
            except ValueError:
                continue
            var_val = float(stats.t.ppf(nominal, fit["df"], loc=fit["mu"], scale=fit["sigma"]))
            if arr[t] < var_val:
                violations += 1
            n_tests += 1

        if n_tests == 0:
            results.append(
                {
                    "window": W,
                    "n_tests": 0,
                    "n_violations": 0,
                    "empirical_rate": None,
                    "deviation": None,
                    "kupiec_lr": None,
                    "kupiec_pvalue": None,
                    "rejected": None,
                    "note": "no valid fits",
                }
            )
            continue

        emp_rate = violations / n_tests
        deviation = abs(emp_rate - nominal)

        x, n_t, p0 = violations, n_tests, nominal
        p_hat = emp_rate
        if p_hat in (0.0, 1.0) or p0 in (0.0, 1.0):
            lr = float("nan")
            pval = float("nan")
        else:
            ll_null = x * math.log(p0) + (n_t - x) * math.log(1.0 - p0)
            ll_alt = x * math.log(p_hat) + (n_t - x) * math.log(1.0 - p_hat)
            lr = round(-2.0 * (ll_null - ll_alt), 4)
            pval = round(float(stats.chi2.sf(lr, df=1)), 4)

        results.append(
            {
                "window": W,
                "n_tests": n_tests,
                "n_violations": violations,
                "empirical_rate": round(emp_rate, 6),
                "deviation": round(deviation, 6),
                "kupiec_lr": lr if math.isnan(lr) else round(lr, 4),
                "kupiec_pvalue": pval if math.isnan(pval) else round(pval, 4),
                "rejected": bool(pval < 0.05) if not math.isnan(pval) else None,
            }
        )

    results.sort(key=lambda r: r["deviation"] if r["deviation"] is not None else float("inf"))
    best = results[0]["window"] if results and results[0]["deviation"] is not None else None

    return {
        "best_window": best,
        "confidence": confidence,
        "nominal_rate": nominal,
        "test_frac": test_frac,
        "n_total": n,
        "results": results,
    }
