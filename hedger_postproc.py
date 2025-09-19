#!/usr/bin/env python3
#hedger_postproc.py
#finds a bunch of further statistics from existing hedging timeseries, with normalization options

import os
import re
import glob
import math
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

#config
TIMESERIES_GLOB = "output/data/*_timeseries.csv"
OUT_DIR = "output/analysis"
os.makedirs(OUT_DIR, exist_ok=True)

#optional manual parameter overrides
MANUAL_PARAMS: Dict[str, Dict] = {
    # "hedge_2008": {"K": 140.0, "T": 1.0, "r": 0.02, "initial_capital": 1e6},
}

VAR_LEVEL = 0.95
MIN_OBS = 10

def bs_delta(S, K, T, r, sigma, option_type="call", sigma_floor=1e-4):
    T = max(T, 1e-12)
    sigma = max(sigma, sigma_floor)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return stats.norm.cdf(d1)
    else:
        return stats.norm.cdf(d1) - 1.0

def historical_var_es(series: pd.Series, alpha: float = VAR_LEVEL) -> Tuple[float, float]:
    pnl = series.dropna().to_numpy()
    if pnl.size == 0:
        return float("nan"), float("nan")
    losses = -pnl
    q = np.quantile(losses, alpha)
    tail = losses[losses >= q]
    es = tail.mean() if tail.size > 0 else q
    return float(q), float(es)

def max_drawdown(portfolio: pd.Series) -> Tuple[float, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    s = pd.Series(portfolio).dropna()
    if s.size == 0:
        return 0.0, None, None
    running_max = s.cummax()
    drawdown = (running_max - s) / running_max
    max_dd = float(drawdown.max())
    if np.isnan(max_dd):
        return 0.0, None, None
    trough_idx = drawdown.idxmax()
    try:
        peak_idx = s[:trough_idx].idxmax()
    except Exception:
        peak_idx = None
    return max_dd, peak_idx, trough_idx

def robust_skew_kurt(series: pd.Series) -> Tuple[float, float]:
    s = series.dropna()
    if s.size == 0:
        return float("nan"), float("nan")
    return float(stats.skew(s)), float(stats.kurtosis(s, fisher=True))

def regression_leakage(x: pd.Series, y: pd.Series) -> Tuple[float, float, float, float, float]:
    xr = pd.Series(x).fillna(0)
    yr = pd.Series(y).fillna(0)
    xr, yr = xr.align(yr, join="inner")
    xarr = np.asarray(xr)
    yarr = np.asarray(yr)
    if xarr.size < 2 or np.all(np.isclose(xarr, 0)):
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    res = stats.linregress(xarr, yarr)
    return float(res.slope), float(res.intercept), float(res.rvalue), float(res.pvalue), float(res.stderr)

def infer_params_from_filename(fname: str) -> Tuple[str, Dict[str, float]]:
    base = os.path.basename(fname)
    m = re.match(r"(.*)_timeseries\.csv$", base)
    prefix = m.group(1) if m else os.path.splitext(base)[0]
    if "2008" in prefix:
        return prefix, {"T": 1.0, "r": 0.02}
    if "2020" in prefix:
        return prefix, {"T": 0.5, "r": 0.005}
    return prefix, {"T": 0.5, "r": 0.01}

def analyse_file(path: str) -> Tuple[Dict, Dict]:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    prefix, inferred = infer_params_from_filename(path)
    params = MANUAL_PARAMS.get(prefix, {})
    T = float(params.get("T", inferred.get("T", 0.5)))
    r = float(params.get("r", inferred.get("r", 0.01)))
    initial_capital = float(params.get("initial_capital", params.get("initial_capital", 0.0)))

    #try to read off K
    possible_summary = f"output/data/{prefix}_summary.csv"
    K = None
    if os.path.exists(possible_summary):
        try:
            s = pd.read_csv(possible_summary)
            if "K_used" in s.columns:
                K = float(s["K_used"].iloc[0])
        except Exception:
            K = None

    #existence checks
    price = df["price"] if "price" in df.columns else pd.Series(np.nan, index=df.index)
    option_mt = df["option_mt"] if "option_mt" in df.columns else pd.Series(np.nan, index=df.index)
    portfolio = df["portfolio"] if "portfolio" in df.columns else pd.Series(np.nan, index=df.index)
    daily_pnl = df["daily_pnl"] if "daily_pnl" in df.columns else portfolio.diff().fillna(0)
    trade_cost = df["trade_cost"] if "trade_cost" in df.columns else pd.Series(0.0, index=df.index)

    if "trade" in df.columns:
        trades = pd.to_numeric(df["trade"].fillna(0), errors="coerce").fillna(0)
    else:
        trades = pd.Series(0.0, index=df.index)

    #K sanity check
    try:
        S0 = float(price.dropna().iloc[0])
    except Exception:
        S0 = float("nan")
    if K is None:
        if not np.isnan(S0):
            K = float(params.get("K", S0 * 1.05))
        else:
            K = float(params.get("K", 100.0))

    try:
        pos = trades.cumsum()
    except Exception:
        pos = pd.Series(0.0, index=df.index)

    #basic metrics
    trade_count = int((trades != 0).sum())  # FIX: always define trade_count
    price_for_turnover = price.reindex(trades.index).fillna(method="ffill").fillna(method="bfill")
    turnover = float((trades.abs() * price_for_turnover).sum()) if trade_count > 0 else 0.0
    total_costs = float(trade_cost.sum()) if trade_cost.notna().any() else 0.0

    #PnL metrics
    pnl_mean = float(daily_pnl.mean())
    pnl_std = float(daily_pnl.std(ddof=0))
    pnl_skew, pnl_kurt = robust_skew_kurt(daily_pnl)
    var95, es95 = historical_var_es(daily_pnl, alpha=VAR_LEVEL)

    #drawdown
    md_drawdown_frac, peak_idx, trough_idx = max_drawdown(portfolio)

    #error
    if "tracking_error" in df.columns:
        te = df["tracking_error"].dropna()
        te_std = float(te.std(ddof=0)) if not te.empty else float("nan")
        te_q95 = float(np.quantile(np.abs(te), 0.95)) if te.size > 0 else float("nan")
    else:
        te = pd.Series(np.nan, index=df.index)
        te_std = float("nan")
        te_q95 = float("nan")

    #correlation & regression leakage
    spot_ret = price.pct_change().fillna(0)
    port_ret = portfolio.pct_change().fillna(0)
    try:
        global_corr = float(spot_ret.corr(port_ret) or 0.0)
    except Exception:
        global_corr = float("nan")
    slope, intercept, rval, pval, stderr = regression_leakage(spot_ret, daily_pnl)

    #tail metrics
    te_abs = np.abs(te.dropna())
    te_tail_95 = float(np.quantile(te_abs, 0.95)) if te_abs.size > 0 else float("nan")
    te_tail_99 = float(np.quantile(te_abs, 0.99)) if te_abs.size > 0 else float("nan")

    #try to normalise...
    option_value0 = None
    if option_mt.notna().any():
        try:
            option_value0 = float(option_mt.dropna().iloc[0])
        except Exception:
            option_value0 = None
    if option_value0 is None and not np.isnan(S0):
        option_value0 = float(S0)
    # final safety
    if option_value0 is None or not np.isfinite(option_value0) or option_value0 <= 0:
        option_value0 = float("nan")

    #...by capital
    turnover_norm_by_capital = float(turnover / initial_capital) if (initial_capital and initial_capital != 0) else float("nan")
    turnover_norm_by_option = float(turnover / option_value0) if (not np.isnan(option_value0)) else float("nan")
    costs_norm_by_capital = float(total_costs / initial_capital) if (initial_capital and initial_capital != 0) else float("nan")
    costs_norm_by_option = float(total_costs / option_value0) if (not np.isnan(option_value0)) else float("nan")
    te_std_norm_by_option = float(te_std / option_value0) if (not np.isnan(option_value0) and np.isfinite(te_std)) else float("nan")
    pnl_std_norm_by_option = float(pnl_std / option_value0) if (not np.isnan(option_value0)) else float("nan")

    #...by trade
    turnover_per_trade = float(turnover / trade_count) if trade_count > 0 else float("nan")
    costs_per_trade = float(total_costs / trade_count) if trade_count > 0 else float("nan")
    avg_trade_size = float((trades.abs() * price_for_turnover).sum() / trade_count) if trade_count > 0 else float("nan")
    avg_trade_qty = float(trades.abs().sum() / trade_count) if trade_count > 0 else float("nan")
    days = df.shape[0]
    trades_per_day = float(trade_count / max(days, 1))

    #rolling stdev
    rolling_te_std = te.abs().rolling(window=10, min_periods=3).std()
    rolling_te_std_last = float(rolling_te_std.iloc[-1]) if rolling_te_std.notna().any() else float("nan")

    #drawdown
    cum_pnl = daily_pnl.cumsum()
    cum_pnl_md, _, _ = max_drawdown(cum_pnl)

    extended = {
        "prefix": prefix,
        "path": path,
        "K_used": float(K),
        "T": float(T),
        "r": float(r),
        "initial_capital": float(initial_capital),
        "total_trades": int((trades != 0).sum()),
        "trade_count": int(trade_count),
        "turnover": float(turnover),
        "turnover_norm_by_capital": turnover_norm_by_capital,
        "turnover_norm_by_option": turnover_norm_by_option,
        "turnover_per_trade": turnover_per_trade,
        "total_costs": float(total_costs),
        "costs_norm_by_capital": costs_norm_by_capital,
        "costs_norm_by_option": costs_norm_by_option,
        "costs_per_trade": costs_per_trade,
        "pnl_mean": float(pnl_mean),
        "pnl_std": float(pnl_std),
        "pnl_std_norm_by_option": pnl_std_norm_by_option,
        "pnl_skew": float(pnl_skew),
        "pnl_kurtosis": float(pnl_kurt),
        f"VaR_{int(VAR_LEVEL*100)}": float(var95),
        f"ES_{int(VAR_LEVEL*100)}": float(es95),
        "tracking_error_std": float(te_std) if not np.isnan(te_std) else float("nan"),
        "tracking_error_std_norm_by_option": te_std_norm_by_option,
        "te_tail_95": float(te_tail_95),
        "te_tail_99": float(te_tail_99),
        "max_drawdown_frac": float(md_drawdown_frac),
        "cum_pnl_max_drawdown": float(cum_pnl_md),
        "global_corr_port_spot": float(global_corr),
        "leak_slope": float(slope),
        "leak_r": float(rval),
        "leak_pval": float(pval),
        "avg_trade_size": float(avg_trade_size) if not np.isnan(avg_trade_size) else float("nan"),
        "avg_trade_qty": float(avg_trade_qty) if not np.isnan(avg_trade_qty) else float("nan"),
        "trades_per_day": float(trades_per_day),
        "rolling_te_std_last": float(rolling_te_std_last),
        "option_value0": float(option_value0) if not np.isnan(option_value0) else float("nan"),
    }

    ext_df = pd.DataFrame([extended])
    ext_fname = os.path.join(OUT_DIR, f"{prefix}_extended_summary.csv")
    ext_df.to_csv(ext_fname, index=False)

    #long summary message
    print("------------------------------------------------------------")
    print(f"Processed: {prefix}")
    print(f" Rows: {df.shape[0]}  |  K_used: {extended['K_used']:.4g}  |  T: {extended['T']}, r: {extended['r']}")
    print(f" Total trades: {extended['total_trades']}, Turnover (notional): {extended['turnover']:.4g}")
    print(f" Turnover / initial capital: {extended['turnover_norm_by_capital']:.6g}, Turnover / option_value0: {extended['turnover_norm_by_option']:.6g}")
    print(f" Total costs: {extended['total_costs']:.6g}, Costs / capital: {extended['costs_norm_by_capital']:.6g}, Costs / option: {extended['costs_norm_by_option']:.6g}")
    print(f" Tracking error std: {extended['tracking_error_std']:.6g}, TE_norm_by_option: {extended['tracking_error_std_norm_by_option']:.6g}")
    print(f" Daily PnL mean/std: {extended['pnl_mean']:.6g} / {extended['pnl_std']:.6g}, PnL_std_norm_by_option: {extended['pnl_std_norm_by_option']:.6g}")
    print(f" Avg trade size (notional): {extended['avg_trade_size']:.6g}, Turnover/trade: {extended['turnover_per_trade']:.6g}, Costs/trade: {extended['costs_per_trade']:.6g}")
    print(f" Max drawdown (fraction): {extended['max_drawdown_frac']:.6g}")
    print(f" {int(VAR_LEVEL*100)}% VaR (loss): {extended[f'VaR_{int(VAR_LEVEL*100)}']:.6g}, ES: {extended[f'ES_{int(VAR_LEVEL*100)}']:.6g}")
    print("------------------------------------------------------------")

    return extended, {"ext_summary": ext_fname}

def main():
    files = glob.glob(TIMESERIES_GLOB)
    if not files:
        print(f"No timeseries files found with pattern {TIMESERIES_GLOB}. Exiting.")
        return {}

    all_summaries = []
    results = {}
    for f in sorted(files):
        try:
            ext, info = analyse_file(f)
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
