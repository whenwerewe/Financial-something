#!/usr/bin/env python3
#hedger_postproc.py
#computes stats

import os
import re
import glob
import math
from typing import Optional, Dict

import numpy as np
import pandas as pd
from scipy import stats

TIMESERIES_GLOB = "output/data/*_timeseries.csv"
OUT_DIR = "output/analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# Optional manual parameter overrides
MANUAL_PARAMS: Dict[str, Dict] = {
    # "hedge_2008": {"K": 140.0, "T": 1.0, "r": 0.02, "initial_capital": 1e6},
}

#VaR confidence level
VAR_LEVEL = 0.95

#observatiosn to do rolling averages
MIN_OBS = 10

def bs_delta(S, K, T, r, sigma, option_type="call", sigma_floor=1e-4):
    """Black-Scholes delta (call by default)."""
    T = max(T, 1e-12)
    sigma = max(sigma, sigma_floor)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return stats.norm.cdf(d1)
    else:
        return stats.norm.cdf(d1) - 1.0

def historical_var_es(series, alpha=VAR_LEVEL):
    #VaR: positive number representing loss at alpha
    #ES: average loss conditional on losses >= VaR.
    #Series should be PnL; we compute losses = -pnl.

    pnl = series.dropna().to_numpy()
    if pnl.size == 0:
        return np.nan, np.nan
    losses = -pnl
    q = np.quantile(losses, alpha)
    tail = losses[losses >= q]
    if tail.size == 0:
        es = q
    else:
        es = tail.mean()
    return float(q), float(es)

def max_drawdown(portfolio):
    #index of peak+max drawdown as a fraction
    s = pd.Series(portfolio)
    if s.size == 0:
        return 0.0, None, None
    running_max = s.cummax()
    drawdown = (running_max - s) / running_max
    max_dd = drawdown.max()
    if np.isnan(max_dd):
        return 0.0, None, None
    trough_idx = drawdown.idxmax()
    peak_idx = s[:trough_idx].idxmax() if trough_idx is not None else None
    return float(max_dd), peak_idx, trough_idx

def robust_skew_kurt(series):
    s = series.dropna()
    if s.size == 0:
        return np.nan, np.nan
    return float(stats.skew(s)), float(stats.kurtosis(s, fisher=True))

def regression_leakage(x, y):
    #literally just OLS
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size < 2 or np.all(np.isclose(x, 0)):
        return np.nan, np.nan, np.nan, np.nan, np.nan
    res = stats.linregress(x, y)
    return float(res.slope), float(res.intercept), float(res.rvalue), float(res.pvalue), float(res.stderr)

def infer_params_from_filename(fname):
    #guess some params
    base = os.path.basename(fname)
    m = re.match(r"(.*)_timeseries\.csv$", base)
    prefix = m.group(1) if m else base
    # heuristics
    if "2008" in prefix:
        return prefix, {"T": 1.0, "r": 0.02}
    if "2020" in prefix:
        return prefix, {"T": 0.5, "r": 0.005}
    # fallback
    return prefix, {"T": 0.5, "r": 0.01}

def analyze_file(path):
    
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    prefix, inferred = infer_params_from_filename(path)
    params = MANUAL_PARAMS.get(prefix, {})
    # merge inferred defaults with manual overrides
    T = float(params.get("T", inferred.get("T", 0.5)))
    r = float(params.get("r", inferred.get("r", 0.01)))
    initial_capital = float(params.get("initial_capital", params.get("initial_capital", 0.0)))

    #if we don't need K default to being near the money
    possible_summary = f"output/data/{prefix}_summary.csv"
    K = None
    if os.path.exists(possible_summary):
        try:
            s = pd.read_csv(possible_summary)
            if "K_used" in s.columns:
                K = float(s["K_used"].iloc[0])
        except Exception:
            K = None

    if K is None:
        try:
            S0 = float(df["price"].iloc[0])
        except Exception:
            S0 = float(df["price"].dropna().iloc[0])
        K = float(params.get("K", S0 * 1.05))

    if "trade" in df.columns:
        pos = df["trade"].fillna(0).cumsum()
    else:
        pos = pd.Series(0, index=df.index)

    price = df["price"] if "price" in df.columns else pd.Series(np.nan, index=df.index)
    sigma = df["sigma"] if "sigma" in df.columns else pd.Series(np.nan, index=df.index)
    portfolio = df["portfolio"] if "portfolio" in df.columns else pd.Series(np.nan, index=df.index)
    option_mt = df["option_mt"] if "option_mt" in df.columns else pd.Series(np.nan, index=df.index)
    daily_pnl = df["daily_pnl"] if "daily_pnl" in df.columns else portfolio.diff().fillna(0)
    trade_cost = df["trade_cost"] if "trade_cost" in df.columns else pd.Series(0.0, index=df.index)
    trades = df["trade"] if "trade" in df.columns else pd.Series(0.0, index=df.index)

    #basic metrics
    total_trades = int((trades != 0).sum())
    turnover = float((trades.abs() * price).sum()) if ("trade" in df.columns) else np.nan
    total_costs = float(trade_cost.sum()) if "trade_cost" in df.columns else np.nan

    #PnL facts
    pnl_mean = float(daily_pnl.mean())
    pnl_std = float(daily_pnl.std())
    pnl_skew, pnl_kurt = robust_skew_kurt(daily_pnl)

    # VaR/ES for daily PnL
    var95, es95 = historical_var_es(daily_pnl, alpha=VAR_LEVEL)

    #drawdown
    md_drawdown, peak_idx, trough_idx = max_drawdown(portfolio)
    md_abs = md_drawdown

    #tracking error
    if "tracking_error" in df.columns:
        te = df["tracking_error"]
        te_std = float(te.std())
        te_q95 = float(np.quantile(np.abs(te.dropna()), 0.95)) if te.notna().any() else np.nan
    else:
        te = pd.Series(np.nan, index=df.index)
        te_std = np.nan
        te_q95 = np.nan

    #leakages
    spot_ret = price.pct_change().fillna(0)
    port_ret = portfolio.pct_change().fillna(0)
    global_corr = float(spot_ret.corr(port_ret) or 0.0)
    slope, intercept, rval, pval, stderr = regression_leakage(spot_ret, daily_pnl)

    #tail metrics
    te_abs = np.abs(te.dropna())
    te_tail_95 = float(np.quantile(te_abs, 0.95)) if te_abs.size > 0 else np.nan
    te_tail_99 = float(np.quantile(te_abs, 0.99)) if te_abs.size > 0 else np.nan

    #normalise by turnover if we have initial capital
    if initial_capital and initial_capital != 0:
        turnover_perc = turnover / initial_capital
    else:
        turnover_perc = np.nan

    #trade stats
    trade_count = int((trades != 0).sum())
    avg_trade_size = float((trades.abs() * price).sum() / trade_count) if trade_count > 0 else np.nan
    avg_trade_qty = float(trades.abs().sum() / trade_count) if trade_count > 0 else np.nan
    days = df.shape[0]
    trades_per_day = trade_count / max(days, 1)

    #rolling stats
    rolling_te_std = te.abs().rolling(window=10, min_periods=3).std()
    rolling_te_std_last = float(rolling_te_std.iloc[-1]) if rolling_te_std.notna().any() else np.nan

    #PnL drawdown
    cum_pnl = daily_pnl.cumsum()
    cum_pnl_md, _, _ = max_drawdown(cum_pnl)

    #store it all
    extended = {
        "prefix": prefix,
        "path": path,
        "K_used": float(K),
        "T": float(T),
        "r": float(r),
        "initial_capital": float(initial_capital),
        "total_trades": int(total_trades),
        "trade_count": int(trade_count),
        "turnover": float(turnover) if not np.isnan(turnover) else np.nan,
        "turnover_norm": float(turnover_perc) if not np.isnan(turnover_perc) else np.nan,
        "total_costs": float(total_costs) if not np.isnan(total_costs) else np.nan,
        "pnl_mean": pnl_mean,
        "pnl_std": pnl_std,
        "pnl_skew": pnl_skew,
        "pnl_kurtosis": pnl_kurt,
        f"VaR_{int(VAR_LEVEL*100)}": float(var95),
        f"ES_{int(VAR_LEVEL*100)}": float(es95),
        "tracking_error_std": te_std,
        "te_tail_95": te_tail_95,
        "te_tail_99": te_tail_99,
        "max_drawdown_frac": md_abs,
        "cum_pnl_max_drawdown": cum_pnl_md,
        "global_corr_port_spot": global_corr,
        "leak_slope": slope,
        "leak_r": rval,
        "leak_pval": pval,
        "avg_trade_size": float(avg_trade_size) if not np.isnan(avg_trade_size) else np.nan,
        "avg_trade_qty": float(avg_trade_qty) if not np.isnan(avg_trade_qty) else np.nan,
        "trades_per_day": float(trades_per_day),
        "rolling_te_std_last": rolling_te_std_last,
    }

    ext_df = pd.DataFrame([extended])
    ext_fname = os.path.join(os.path.dirname(path), f"{prefix}_extended_summary.csv")
    ext_df.to_csv(ext_fname, index=False)

    print(f"[postproc] processed {prefix} -> saved extended summary: {ext_fname}")

    return extended, {"ext_summary": ext_fname}

def main():
    files = glob.glob(TIMESERIES_GLOB)
    if not files:
        print(f"No timeseries files found with pattern {TIMESERIES_GLOB}. Exiting.")
        return

    all_summaries = []
    results = {}
    for f in sorted(files):
        try:
            ext, info = analyze_file(f)
            all_summaries.append(ext)
            results[ext["prefix"]] = info
        except Exception as e:
            print(f"[postproc] failed on {f}: {e}")

    if all_summaries:
        combined = pd.DataFrame(all_summaries)
        combined_fname = os.path.join(OUT_DIR, "combined_extended_summary.csv")
        combined.to_csv(combined_fname, index=False)
        print(f"[postproc] wrote combined extended summary: {combined_fname}")
    else:
        print("[postproc] no successful summaries to combine.")

    return results

if __name__ == "__main__":
    main()
