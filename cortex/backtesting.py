"""Multi-frequency VaR backtesting engine.

Implements Kupiec (1995) proportion-of-failures test and Christoffersen (1998)
independence test for VaR model validation at multiple time horizons.

References:
  - Kupiec (1995) "Techniques for Verifying the Accuracy of Risk Measurement Models"
  - Christoffersen (1998) "Evaluating Interval Forecasts"
"""
from __future__ import annotations

import logging
import math

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def kupiec_test(
    n_obs: int, n_violations: int, confidence: float = 95.0
) -> dict:
    """Kupiec (1995) proportion-of-failures (POF) test.

    Tests H0: observed violation rate = expected violation rate.
    Uses likelihood ratio statistic ~ chi2(1).

    Args:
        n_obs: Total number of observations.
        n_violations: Number of VaR violations (actual loss > VaR).
        confidence: VaR confidence level (e.g. 95.0 for 95% VaR).

    Returns:
        Dict with statistic, p_value, pass (True if model not rejected at 5%).
    """
    if n_obs < 1:
        return {"statistic": 0.0, "p_value": 1.0, "pass": True, "violation_rate": 0.0, "expected_rate": 0.0}

    p_expected = 1.0 - confidence / 100.0
    p_observed = n_violations / n_obs if n_obs > 0 else 0.0

    if n_violations == 0:
        # No violations — log-likelihood ratio simplifies
        lr = -2.0 * n_obs * math.log(1.0 - p_expected) if p_expected < 1.0 else 0.0
    elif n_violations == n_obs:
        lr = -2.0 * n_obs * math.log(p_expected) if p_expected > 0 else 0.0
    else:
        # LR = -2 * [log(p^x * (1-p)^(n-x)) - log(p_hat^x * (1-p_hat)^(n-x))]
        x = n_violations
        n = n_obs
        lr_null = x * math.log(p_expected) + (n - x) * math.log(1.0 - p_expected)
        lr_alt = x * math.log(p_observed) + (n - x) * math.log(1.0 - p_observed)
        lr = -2.0 * (lr_null - lr_alt)

    lr = max(lr, 0.0)
    p_value = float(1.0 - stats.chi2.cdf(lr, df=1))

    return {
        "statistic": float(lr),
        "p_value": p_value,
        "pass": p_value > 0.05,
        "violation_rate": p_observed,
        "expected_rate": p_expected,
    }


def christoffersen_test(violations: np.ndarray) -> dict:
    """Christoffersen (1998) independence test for VaR violations.

    Tests H0: violations are independent (no clustering).
    Uses likelihood ratio statistic ~ chi2(1).

    Args:
        violations: Binary array (1 = violation, 0 = no violation).

    Returns:
        Dict with statistic, p_value, pass.
    """
    v = np.asarray(violations, dtype=int)
    n = len(v)
    if n < 4:
        return {"statistic": 0.0, "p_value": 1.0, "pass": True}

    # Transition counts
    n00 = n01 = n10 = n11 = 0
    for i in range(1, n):
        prev, curr = v[i - 1], v[i]
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1

    # Transition probabilities
    denom_0 = n00 + n01
    denom_1 = n10 + n11
    if denom_0 == 0 or denom_1 == 0:
        return {"statistic": 0.0, "p_value": 1.0, "pass": True}

    p01 = n01 / denom_0
    p11 = n11 / denom_1
    p_hat = (n01 + n11) / (n00 + n01 + n10 + n11)

    if p_hat <= 0 or p_hat >= 1 or p01 <= 0 or p01 >= 1 or p11 <= 0 or p11 >= 1:
        return {"statistic": 0.0, "p_value": 1.0, "pass": True}

    # LR independence test
    lr_null = (n01 + n11) * math.log(p_hat) + (n00 + n10) * math.log(1.0 - p_hat)
    lr_alt = (n01 * math.log(p01) + n00 * math.log(1.0 - p01)
              + n11 * math.log(p11) + n10 * math.log(1.0 - p11))
    lr = -2.0 * (lr_null - lr_alt)
    lr = max(lr, 0.0)

    p_value = float(1.0 - stats.chi2.cdf(lr, df=1))
    return {"statistic": float(lr), "p_value": p_value, "pass": p_value > 0.05}


def backtest_var(
    returns: np.ndarray,
    var_values: np.ndarray,
    confidence: float = 95.0,
) -> dict:
    """Run VaR backtest on a return series against VaR forecasts.

    Args:
        returns: Realized returns (%).
        var_values: VaR forecasts (negative numbers, same length as returns).
        confidence: VaR confidence level.

    Returns:
        Dict with n_obs, n_violations, violation_rate, kupiec, christoffersen.
    """
    returns = np.asarray(returns, dtype=float)
    var_values = np.asarray(var_values, dtype=float)

    n = min(len(returns), len(var_values))
    if n < 5:
        return {
            "n_obs": n, "n_violations": 0, "violation_rate": 0.0,
            "kupiec": kupiec_test(n, 0, confidence),
            "christoffersen": {"statistic": 0.0, "p_value": 1.0, "pass": True},
        }

    returns = returns[:n]
    var_values = var_values[:n]

    # Violation: actual return < VaR (both negative, so loss exceeds VaR)
    violations = (returns < var_values).astype(int)
    n_violations = int(np.sum(violations))

    kup = kupiec_test(n, n_violations, confidence)
    chris = christoffersen_test(violations)

    return {
        "n_obs": n,
        "n_violations": n_violations,
        "violation_rate": n_violations / n if n > 0 else 0.0,
        "kupiec": kup,
        "christoffersen": chris,
    }


def backtest_multi_horizon(
    bars_by_horizon: dict[int, list[dict]],
    var_forecast_fn,
    confidence: float = 95.0,
) -> list[dict]:
    """Run VaR backtest at multiple time horizons.

    Args:
        bars_by_horizon: Dict mapping horizon_minutes -> list of OHLCV bars.
        var_forecast_fn: Callable(returns, confidence) -> float (VaR forecast).
        confidence: VaR confidence level.

    Returns:
        List of per-horizon backtest results.
    """
    from cortex.data.tick_data import bars_to_returns

    results = []
    for horizon_min, bars in sorted(bars_by_horizon.items()):
        rets = bars_to_returns(bars)
        if len(rets) < 10:
            results.append({
                "horizon_minutes": horizon_min,
                "n_observations": len(rets),
                "n_violations": 0,
                "violation_rate": 0.0,
                "expected_rate": 1.0 - confidence / 100.0,
                "kupiec_stat": 0.0,
                "kupiec_pvalue": 1.0,
                "kupiec_pass": True,
                "christoffersen_stat": None,
                "christoffersen_pvalue": None,
            })
            continue

        # Rolling VaR: use expanding window with min 20 observations
        var_forecasts = np.full(len(rets), np.nan)
        min_window = 20
        for i in range(min_window, len(rets)):
            window = rets[:i]
            var_forecasts[i] = var_forecast_fn(window, confidence)

        valid = ~np.isnan(var_forecasts)
        valid_rets = rets[valid]
        valid_vars = var_forecasts[valid]

        bt = backtest_var(valid_rets, valid_vars, confidence)

        results.append({
            "horizon_minutes": horizon_min,
            "n_observations": bt["n_obs"],
            "n_violations": bt["n_violations"],
            "violation_rate": bt["violation_rate"],
            "expected_rate": 1.0 - confidence / 100.0,
            "kupiec_stat": bt["kupiec"]["statistic"],
            "kupiec_pvalue": bt["kupiec"]["p_value"],
            "kupiec_pass": bt["kupiec"]["pass"],
            "christoffersen_stat": bt["christoffersen"]["statistic"],
            "christoffersen_pvalue": bt["christoffersen"]["p_value"],
        })

    return results


def simple_var_forecast(returns: np.ndarray, confidence: float = 95.0) -> float:
    """Simple historical VaR forecast (percentile method)."""
    alpha = 1.0 - confidence / 100.0
    return float(np.percentile(returns, alpha * 100))

