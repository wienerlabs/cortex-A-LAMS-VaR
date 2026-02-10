"""
Stochastic Volatility with Jumps (SVJ) — Bates (1996) model.

Combines Heston stochastic volatility with Merton jump-diffusion:

    Asset:    dS/S = μdt + √V dW₁ + (J-1)dN
    Variance: dV   = κ(θ - V)dt + σ√V dW₂
    Corr(dW₁, dW₂) = ρ

    Jump component: log(J) ~ N(μⱼ, σⱼ²), N ~ Poisson(λ)

Seven parameters:
    κ  — mean reversion speed of variance
    θ  — long-run variance level
    σ  — vol-of-vol (volatility of variance)
    ρ  — correlation between asset and variance Brownians
    λ  — jump intensity (expected jumps per unit time)
    μⱼ — mean jump size (in log-return space)
    σⱼ — jump size volatility

References:
    Bates, D. (1996). "Jumps and Stochastic Volatility."
    Duffie, Pan, Singleton (2000). "Transform Analysis and Asset Pricing."
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def detect_jumps(
    returns: pd.Series | np.ndarray,
    threshold_multiplier: float = 3.0,
    window: int = 20,
) -> dict:
    """
    Detect jumps using Barndorff-Nielsen-Shephard (BNS) bipower variation test.

    Compares realized variance (RV) with bipower variation (BV). When RV >> BV,
    the excess is attributed to jumps. Individual jumps are flagged when
    |r_t| > threshold_multiplier × local_vol.

    Args:
        returns: Return series (percentage).
        threshold_multiplier: Sigma multiplier for individual jump detection.
        window: Rolling window for local volatility estimation.

    Returns:
        Dict with jump_dates, jump_returns, n_jumps, jump_fraction,
        avg_jump_size, jump_vol, bns_statistic, bns_pvalue.
    """
    if isinstance(returns, pd.Series):
        values = returns.values.astype(float)
        index = returns.index
    else:
        values = np.asarray(returns, dtype=float)
        index = np.arange(len(values))

    n = len(values)
    if n < window + 10:
        raise ValueError(f"Need at least {window + 10} observations, got {n}")

    rv = np.sum(values ** 2)
    abs_ret = np.abs(values)
    bv = (np.pi / 2.0) * np.sum(abs_ret[1:] * abs_ret[:-1])

    # BNS test statistic: (RV - BV) / RV ~ asymptotically standard normal under H0
    jump_variation = max(rv - bv, 0.0)
    tri_power = 0.0
    for i in range(2, n):
        tri_power += abs_ret[i] ** (2/3) * abs_ret[i-1] ** (2/3) * abs_ret[i-2] ** (2/3)
    mu_1 = math.sqrt(2.0 / math.pi)
    tri_power *= (mu_1 ** (-3)) * (2.0 ** (2/3)) * math.gamma(7/6) / math.gamma(0.5)

    iq_estimate = (n / 3.0) * tri_power if tri_power > 0 else rv ** 2 / n
    denom = max(math.sqrt(2.0 * iq_estimate / n), 1e-12)
    bns_stat = (rv - bv) / denom
    bns_pvalue = 1.0 - stats.norm.cdf(bns_stat)

    # Individual jump detection via rolling local vol
    rolling_std = pd.Series(values).rolling(window=window, min_periods=max(5, window // 2)).std().values
    rolling_std = np.where(np.isnan(rolling_std), np.nanstd(values), rolling_std)
    rolling_std = np.where(rolling_std < 1e-10, np.nanstd(values), rolling_std)

    jump_mask = np.abs(values) > threshold_multiplier * rolling_std
    jump_indices = np.where(jump_mask)[0]
    jump_returns_arr = values[jump_indices]

    return {
        "jump_dates": [index[i] for i in jump_indices],
        "jump_returns": jump_returns_arr.tolist(),
        "jump_indices": jump_indices.tolist(),
        "n_jumps": int(len(jump_indices)),
        "jump_fraction": round(len(jump_indices) / n, 6),
        "avg_jump_size": round(float(np.mean(np.abs(jump_returns_arr))), 6) if len(jump_indices) > 0 else 0.0,
        "jump_vol": round(float(np.std(jump_returns_arr)), 6) if len(jump_indices) > 1 else 0.0,
        "bns_statistic": round(float(bns_stat), 6),
        "bns_pvalue": round(float(bns_pvalue), 6),
        "jump_variation": round(float(jump_variation), 6),
        "realized_variance": round(float(rv), 6),
        "bipower_variation": round(float(bv), 6),
        "threshold_multiplier": threshold_multiplier,


def calibrate_svj(
    returns: pd.Series | np.ndarray,
    use_hawkes: bool = False,
    jump_threshold_multiplier: float = 3.0,
) -> dict:
    """
    Calibrate the 7-parameter Bates SVJ model.

    Uses method of moments for initial estimates, then refines via
    quasi-MLE optimization. Optionally integrates Hawkes process for
    time-varying jump intensity.

    Args:
        returns: Return series (percentage).
        use_hawkes: If True, use Hawkes process for jump clustering.
        jump_threshold_multiplier: Multiplier for jump detection.

    Returns:
        Dict with kappa, theta, sigma, rho, lambda_, mu_j, sigma_j,
        plus diagnostics and optional hawkes_params.
    """
    if isinstance(returns, pd.Series):
        values = returns.values.astype(float)
    else:
        values = np.asarray(returns, dtype=float)

    n = len(values)
    if n < 50:
        raise ValueError(f"Need at least 50 observations for SVJ calibration, got {n}")

    # Scale returns to decimal (from percentage)
    r = values / 100.0

    # Step 1: Detect jumps for initial parameter estimation
    jumps = detect_jumps(returns, threshold_multiplier=jump_threshold_multiplier)
    jump_mask = np.zeros(n, dtype=bool)
    for idx in jumps["jump_indices"]:
        jump_mask[idx] = True

    # Separate diffusion and jump returns
    diffusion_returns = r[~jump_mask]
    jump_returns_raw = r[jump_mask]

    # Step 2: Method of moments initial estimates
    total_var = float(np.var(r))
    diffusion_var = float(np.var(diffusion_returns)) if len(diffusion_returns) > 5 else total_var * 0.8

    # Jump parameters from detected jumps
    n_jumps = len(jump_returns_raw)
    lambda_init = max(n_jumps / (n / 252.0), 0.5)  # annualized jump rate
    mu_j_init = float(np.mean(jump_returns_raw)) if n_jumps > 0 else -0.02
    sigma_j_init = float(np.std(jump_returns_raw)) if n_jumps > 1 else 0.03

    # Heston parameters from diffusion component
    theta_init = diffusion_var * 252.0  # annualized variance
    kappa_init = 5.0  # moderate mean reversion
    sigma_init = 0.5  # vol-of-vol

    # Leverage effect: correlation between returns and squared returns
    r_lag = r[:-1]
    r2_lead = r[1:] ** 2
    if len(r_lag) > 10:
        rho_init = float(np.corrcoef(r_lag, r2_lead)[0, 1])
        rho_init = np.clip(rho_init, -0.95, -0.05)
    else:
        rho_init = -0.5

    # Step 3: Quasi-MLE optimization
    # params = [kappa, theta, sigma, rho, lambda, mu_j, sigma_j]
    x0 = np.array([kappa_init, theta_init, sigma_init, rho_init,
                    lambda_init, mu_j_init, sigma_j_init])

    bounds = [
        (0.1, 50.0),      # kappa
        (1e-6, 1.0),      # theta (annualized variance)
        (0.01, 5.0),      # sigma (vol-of-vol)
        (-0.99, -0.01),   # rho (leverage, typically negative)
        (0.01, 100.0),    # lambda (jump intensity)
        (-0.20, 0.05),    # mu_j (jump mean, typically negative)
        (0.001, 0.30),    # sigma_j (jump vol)
    ]

    best_result = None
    best_nll = float("inf")

    starts = [
        x0,
        np.array([3.0, theta_init * 0.8, 0.3, -0.7, lambda_init * 0.5, -0.03, 0.04]),
        np.array([8.0, theta_init * 1.2, 0.8, -0.3, lambda_init * 1.5, -0.01, 0.02]),
        np.array([5.0, theta_init, 0.5, -0.5, 5.0, -0.02, 0.03]),
    ]

    for start in starts:
        start_clipped = np.clip(start, [b[0] for b in bounds], [b[1] for b in bounds])
        try:
            result = minimize(
                _svj_neg_log_likelihood,
                start_clipped,
                args=(r,),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-10},
            )
            if result.fun < best_nll:
                best_nll = result.fun
                best_result = result
        except Exception as e:
            logger.debug("SVJ MLE attempt failed: %s", e)
            continue

    if best_result is None:
        logger.warning("All SVJ optimization attempts failed, using moment estimates")
        opt_params = x0
        opt_success = False
        opt_nit = 0
    else:
        opt_params = best_result.x
        opt_success = bool(best_result.success)
        opt_nit = int(best_result.nit)

    kappa, theta, sigma, rho, lambda_, mu_j, sigma_j = opt_params

    # Feller condition: 2κθ > σ² (ensures variance stays positive)
    feller_ratio = 2.0 * kappa * theta / (sigma ** 2) if sigma > 0 else float("inf")
    feller_satisfied = feller_ratio > 1.0

    # Step 4: Optional Hawkes integration for time-varying jump intensity
    hawkes_params = None
    if use_hawkes:
        try:
            from hawkes_process import extract_events, fit_hawkes, hawkes_intensity
            events = extract_events(returns, threshold_percentile=5.0, use_absolute=True)
            if events["n_events"] >= 5:
                h_fit = fit_hawkes(events["event_times"], events["T"])
                h_intens = hawkes_intensity(events["event_times"], h_fit)
                hawkes_params = {
                    "mu": round(h_fit["mu"], 6),
                    "alpha": round(h_fit["alpha"], 6),
                    "beta": round(h_fit["beta"], 6),
                    "branching_ratio": round(h_fit["branching_ratio"], 6),
                    "current_intensity": round(h_intens["current_intensity"], 6),
                    "baseline_intensity": round(h_intens["baseline"], 6),
                    "intensity_ratio": round(
                        h_intens["current_intensity"] / max(h_intens["baseline"], 1e-12), 4
                    ),
                }
        except Exception as e:
            logger.debug("Hawkes integration failed: %s", e)

    ll = -best_nll if best_result is not None else float("nan")
    k_params = 7
    aic = 2 * k_params - 2 * ll if np.isfinite(ll) else float("nan")
    bic = k_params * math.log(n) - 2 * ll if np.isfinite(ll) else float("nan")

    return {
        "kappa": round(float(kappa), 6),
        "theta": round(float(theta), 6),
        "sigma": round(float(sigma), 6),
        "rho": round(float(rho), 6),
        "lambda_": round(float(lambda_), 6),
        "mu_j": round(float(mu_j), 6),
        "sigma_j": round(float(sigma_j), 6),
        "feller_ratio": round(float(feller_ratio), 4),
        "feller_satisfied": feller_satisfied,
        "log_likelihood": round(float(ll), 4) if np.isfinite(ll) else None,
        "aic": round(float(aic), 4) if np.isfinite(aic) else None,
        "bic": round(float(bic), 4) if np.isfinite(bic) else None,
        "n_obs": n,
        "n_jumps_detected": jumps["n_jumps"],
        "jump_fraction": jumps["jump_fraction"],
        "bns_statistic": jumps["bns_statistic"],
        "bns_pvalue": jumps["bns_pvalue"],
        "optimization_success": opt_success,
        "optimization_nit": opt_nit,
        "use_hawkes": use_hawkes,
        "hawkes_params": hawkes_params,
    }



def _svj_neg_log_likelihood(params: np.ndarray, returns: np.ndarray) -> float:
    """
    Negative log-likelihood for the SVJ model using a mixture approximation.

    Approximates the SVJ density as a Poisson-weighted mixture of Gaussians:
    f(r) ≈ Σ_{j=0}^{J_max} P(N=j|λΔt) × N(r | μ + jμⱼ, V_t + jσⱼ²)

    where V_t follows a simplified discretized Heston process.
    """
    kappa, theta, sigma_v, rho, lambda_, mu_j, sigma_j = params

    dt = 1.0 / 252.0
    n = len(returns)
    j_max = 5  # truncate Poisson at 5 jumps per day

    # Poisson probabilities for 0..j_max jumps per day
    lam_dt = lambda_ * dt
    poisson_probs = np.array([
        np.exp(-lam_dt) * lam_dt**j / math.factorial(j) for j in range(j_max + 1)
    ])
    poisson_probs /= poisson_probs.sum()  # normalize truncation

    # Discretized variance process (Euler-Maruyama)
    V = np.empty(n)
    V[0] = theta  # start at long-run level

    total_ll = 0.0
    for t in range(n):
        v_t = max(V[t], 1e-10)
        sqrt_v = math.sqrt(v_t)

        # Jump-compensated drift
        drift = -0.5 * v_t * dt - lambda_ * (math.exp(mu_j + 0.5 * sigma_j**2) - 1.0) * dt

        # Mixture density: sum over possible jump counts
        log_density = -np.inf
        for j in range(j_max + 1):
            mean_j = drift + j * mu_j
            var_j = v_t * dt + j * sigma_j**2
            if var_j <= 0:
                continue
            std_j = math.sqrt(var_j)
            z = (returns[t] - mean_j) / std_j
            log_comp = math.log(max(poisson_probs[j], 1e-300)) - 0.5 * math.log(2 * math.pi) - math.log(std_j) - 0.5 * z**2
            if log_comp > log_density:
                # log-sum-exp trick
                log_density = log_comp + math.log(1.0 + math.exp(log_density - log_comp)) if np.isfinite(log_density) else log_comp
            else:
                log_density = log_density + math.log(1.0 + math.exp(log_comp - log_density)) if np.isfinite(log_comp) else log_density

        total_ll += log_density

        # Update variance for next step
        if t < n - 1:
            dV = kappa * (theta - v_t) * dt + sigma_v * sqrt_v * math.sqrt(dt) * (rho * returns[t] / sqrt_v + math.sqrt(1 - rho**2) * 0.0)
            V[t + 1] = max(v_t + dV, 1e-10)

    if not np.isfinite(total_ll):
        return 1e12

    return -total_ll
