#!/usr/bin/env python3
# hedger_resilient.py
# this can in principle use any local csv though in practice just use the synthetics from the other file

import os
import math
import warnings
from typing import Optional, Tuple, Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

warnings.filterwarnings("ignore")

OUTPUT_DIR = "output"
DATA_DIR = "data"
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "data"), exist_ok=True)

def log(*args, **kwargs):
    print(*args, **kwargs)

def load_price_data_prefer_local(
    ticker: str = "AAPL",
    crisis_period: str = "2008",
    data_dir: str = DATA_DIR,
    explicit_path: Optional[str] = None,
    explicit_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    In order this tries
      -an explicitly provided dataframe
      -an explicit filepath
      -data/XYZ_TIME.csv (these three in the working directory)
      -data/TIME.csv
      -data/XYZ.csv
      -./synthetic_TIME.csv or ./synthetic_XYZ.csv as above
    Returned DataFrame has index = business days and columns: Underlying, Returns, Sigma
    """
    import re

    if explicit_df is not None:
        if not isinstance(explicit_df, pd.DataFrame):
            raise ValueError("explicit_df is not a df")
        # ensure needed columns
        if "Underlying" not in explicit_df.columns:
            raise ValueError("df does not contain column 'Underlying'")
        df = explicit_df.copy()
        df = df.sort_index()
        if "Returns" not in df.columns:
            df["Returns"] = np.log(df["Underlying"] / df["Underlying"].shift(1))
        if "Sigma" not in df.columns:
            df["Sigma"] = df["Returns"].rolling(10).std() * math.sqrt(252.0)
        return df.dropna(subset=["Returns"])

    #handle a csv
    def _coerce_and_clean(df):
        #try a few common names
        if "Close" in df.columns:
            df = df.rename(columns={"Close": "Underlying"})
        elif "Adj Close" in df.columns:
            df = df.rename(columns={"Adj Close": "Underlying"})
        else:
            numcols = df.select_dtypes(include="number").columns.tolist()
            if len(numcols) > 0:
                df = df.rename(columns={numcols[0]: "Underlying"})
            else:
                df = df.rename(columns={df.columns[0]: "Underlying"})

        if "Underlying" not in df.columns:
            raise ValueError("could not identify price data ('Underlying')")

        underlying = df["Underlying"]
        if isinstance(underlying, pd.DataFrame):
            underlying = underlying.iloc[:, 0]

        if underlying.dtype == object:
            def clean_val(x):
                if pd.isna(x):
                    return np.nan
                s = str(x).strip()
                s = s.replace(",", "").replace("$", "").replace("£", "").replace("€", "").replace("\xa0", "")
                if re.match(r'^\(.*\)$', s):
                    s = "-" + s.strip("()")
                if s == "":
                    return np.nan
                return s
            df["Underlying"] = pd.to_numeric(underlying.apply(clean_val), errors="coerce")
        else:
            df["Underlying"] = pd.to_numeric(underlying, errors="coerce")

        if not isinstance(df.index, pd.DatetimeIndex):
            for col in ("Date", "DATE", "date"):
                if col in df.columns:
                    df.index = pd.to_datetime(df[col], errors="coerce")
                    break
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.iloc[:, 0], errors="coerce")
                except Exception:
                    pass

        df = df.sort_index()
        df = df.dropna(subset=["Underlying"])
        if df.shape[0] < 10:
            raise ValueError("too few rows to be useful (sigma rolling average not computed), need at least ten")

        df = df.asfreq("B")
        df["Underlying"] = df["Underlying"].ffill().bfill()

        df["Returns"] = np.log(df["Underlying"] / df["Underlying"].shift(1))
        df["Sigma"] = df["Returns"].rolling(10).std() * math.sqrt(252.0)
        df = df.dropna(subset=["Returns"])
        return df[["Underlying", "Returns", "Sigma"]]

    #look for some possble files
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    candidates += [
        os.path.join(data_dir, f"{ticker}_{crisis_period}.csv"),
        os.path.join(data_dir, f"{crisis_period}.csv"),
        os.path.join(data_dir, f"{ticker}.csv"),
    ]

    cwd_synth = os.path.join(os.getcwd(), f"synthetic_{crisis_period}.csv")
    if os.path.exists(cwd_synth):
        candidates.insert(0, cwd_synth)

    for path in candidates:
        if path and os.path.exists(path):
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
            except Exception:
                df = pd.read_csv(path)
            try:
                cleaned = _coerce_and_clean(df)
                log(f"[loader] loaded and cleaned local CSV: {path} (rows: {len(cleaned)})")
                return cleaned
            except Exception as e:
                log(f"[loader] found {path} but cleaning failed: {e} -- trying next candidate")

    #finally try the synthetics (in practice we'll use this every time)
    try:
        import synthetic_crises
        if crisis_period == "2008" and hasattr(synthetic_crises, "generate_2008_synthetic"):
            log("[loader] using synthetic_crises.generate_2008_synthetic()")
            return synthetic_crises.generate_2008_synthetic()
        if crisis_period == "2020" and hasattr(synthetic_crises, "generate_2020_synthetic"):
            log("[loader] using synthetic_crises.generate_2020_synthetic()")
            return synthetic_crises.generate_2020_synthetic()
    except Exception as e:
        log(f"[loader] synthetic_crises module not available or import failed: {e}")


def fit_ewma_sigma(data_slice: pd.DataFrame, lam: float = 0.94) -> float:
    returns = data_slice["Returns"].dropna()
    if len(returns) < 5:
        return float(data_slice["Sigma"].iloc[-5:].mean())
    sq = returns ** 2
    ewma_var = sq.ewm(alpha=1 - lam).mean().iloc[-1]
    return float(np.sqrt(ewma_var * 252.0))

def try_fit_garch_sigma(data_slice: pd.DataFrame) -> Optional[float]:
    try:
        from arch import arch_model
    except Exception:
        return None
    try:
        returns = 100.0 * data_slice["Returns"].dropna()
        if len(returns) < 30:
            return None
        model = arch_model(returns, vol="Garch", p=1, q=1, dist="t")
        res = model.fit(disp="off", update_freq=0)
        var = res.forecast(horizon=1).variance.iloc[-1, 0]
        annualised_percent_sigma = math.sqrt(var * 252.0)
        return float(annualised_percent_sigma / 100.0)
    except Exception:
        return None

def black_scholes(S: float, K: float, T: float, r: float, sigma: float,
                  option_type: str = "call", sigma_floor: float = 0.01):
    #T in years, sigma annualised
    T = max(T, 1e-6)
    sigma = max(sigma, sigma_floor)
    S = max(float(S), 1e-8)
    K = max(float(K), 1e-8)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    return float(price), float(delta), float(gamma)

class AdvancedCrisisHedger:
    def __init__(self,
                 initial_capital: float = 0.0,
                 delta_thresh: float = 0.005,
                 gamma_thresh: float = 0.01,
                 r: float = 0.01,
                 transaction_cost: float = 0.001,
                 margin_limit: float = 0.2,
                 sigma_floor: float = 0.01,
                 use_arch: bool = False,
                 sell_option: bool = True,
                 auto_rescale_strike: bool = True):
        self.initial_capital = float(initial_capital)
        self.delta_thresh = float(delta_thresh)
        self.gamma_thresh = float(gamma_thresh)
        self.r = float(r)
        self.cost = float(transaction_cost)
        self.margin_limit = float(margin_limit)
        self.sigma_floor = float(sigma_floor)
        self.use_arch = bool(use_arch)
        self.sell_option = bool(sell_option)
        self.auto_rescale_strike = bool(auto_rescale_strike)

    def compute_target_delta(self, prev_delta: float, desired_delta: float, gamma: float) -> float:
        if gamma > self.gamma_thresh:
            mult = min(2.0, 1.0 + gamma / max(self.gamma_thresh, 1e-12))
            return prev_delta + mult * (desired_delta - prev_delta)
        else:
            return prev_delta + 0.8 * (desired_delta - prev_delta)

    def hedge(self, data: pd.DataFrame, K: float, T_years: float, option_type: str = "call") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if data is None or len(data) == 0:
            raise ValueError("Empty price DataFrame supplied to hedge().")

        S = data["Underlying"].to_numpy().flatten()
        n = len(S)
        dates = data.index

        cash = np.zeros(n)
        pos = np.zeros(n)
        port = np.zeros(n)
        option_mt = np.zeros(n)
        te = np.zeros(n)
        pnl = np.zeros(n)
        gexp = np.zeros(n)
        corr_returns = np.zeros(n)
        corr_pnl = np.zeros(n)
        sig = data["Sigma"].to_numpy().flatten()

        trades = np.zeros(n)
        trade_costs = np.zeros(n)
        turnover = 0.0

        #initial
        S0 = float(S[0])
        sigma0 = max(sig[0] if not (sig is None or np.isnan(sig[0])) else 0.2, self.sigma_floor)
        price0, delta0, gamma0 = black_scholes(S0, K, T_years, self.r, sigma0, option_type, sigma_floor=self.sigma_floor)
        option_mt[0] = price0

        if self.auto_rescale_strike:
            if K <= 0 or (S0 < 50 and K > 10 * S0):
                oldK = K
                K = S0 * 1.05
                price0, delta0, gamma0 = black_scholes(S0, K, T_years, self.r, sigma0, option_type, sigma_floor=self.sigma_floor)
                option_mt[0] = price0
                log(f"[defensive] rescaled K from {oldK} to {K:.3f} (S0={S0:.3f})")
        else:
            if K <= 0 or (S0 < 50 and K > 10 * S0):
                log(f"[warning] strike K={K} looks mismatched for S0={S0:.3f} but auto_rescale_strike=False")

        #intial obligatory hedge: long
        if self.sell_option:
            pos[0] = float(delta0)
            initial_cost = abs(pos[0] * S0) * self.cost
            cash[0] = self.initial_capital + price0 - pos[0] * S0 - initial_cost
            trades[0] = pos[0]
            trade_costs[0] = initial_cost
            turnover += abs(pos[0] * S0)
        else:
            pos[0] = 0.0
            cash[0] = self.initial_capital

        port[0] = cash[0] + pos[0] * S0
        te[0] = (port[0] - option_mt[0]) - self.initial_capital
        prev_delta = pos[0]
        dt = 1 / 252.0

        def safe_corr(a, b):
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < 2:
                return 0.0
            a = a[mask]; b = b[mask]
            if np.std(a) == 0 or np.std(b) == 0:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])

        win = 6

        for i in range(1, n):
            #guess vol
            if i > 30:
                v = None
                if self.use_arch:
                    v = try_fit_garch_sigma(data.iloc[:i])
                if v is None:
                    v = fit_ewma_sigma(data.iloc[:i])
                sig[i] = v

            t_rem = max(T_years - i * dt, 0.0)
            price_opt, delta, gamma = black_scholes(S[i], K, t_rem, self.r, max(sig[i], self.sigma_floor), option_type, sigma_floor=self.sigma_floor)
            option_mt[i] = price_opt

            target = self.compute_target_delta(prev_delta, delta, gamma)

            trade = 0.0
            if abs(target - pos[i - 1]) > self.delta_thresh:
                trade = target - pos[i - 1]
                cost = abs(trade * S[i]) * self.cost
                cash_after_growth = cash[i - 1] * math.exp(self.r * dt)
                required_cash = trade * S[i] + cost
                projected_cash = cash_after_growth - required_cash
                if projected_cash < -self.margin_limit * self.initial_capital:
                    max_neg = -self.margin_limit * self.initial_capital
                    allowed_trade_value = cash_after_growth - max_neg - cost
                    allowed_trade_qty = allowed_trade_value / S[i] if S[i] > 0 else 0.0
                    trade = math.copysign(max(0.0, min(abs(trade), abs(allowed_trade_qty))), trade)
                    cost = abs(trade * S[i]) * self.cost
                    required_cash = trade * S[i] + cost
                    projected_cash = cash_after_growth - required_cash
                cash[i] = projected_cash
                pos[i] = pos[i - 1] + trade
                trades[i] = trade
                trade_costs[i] = cost
                turnover += abs(trade * S[i])
            else:
                cash[i] = cash[i - 1] * math.exp(self.r * dt)
                pos[i] = pos[i - 1]

            port[i] = cash[i] + pos[i] * S[i]
            pnl[i] = port[i] - port[i - 1]
            te[i] = (port[i] - option_mt[i]) - self.initial_capital
            gexp[i] = gamma * (pos[i] ** 2)

            if i >= (win - 1):
                start = i - (win - 1)
                spot_slice = S[start:i+1]
                port_slice = port[start:i+1]
                pnl_slice = port_slice[1:] - port_slice[:-1]

                with np.errstate(divide='ignore', invalid='ignore'):
                    spot_ret = np.diff(spot_slice) / spot_slice[:-1]

                #watch for zeros here
                with np.errstate(divide='ignore', invalid='ignore'):
                    port_ret = np.diff(port_slice) / np.where(port_slice[:-1] == 0, np.nan, port_slice[:-1])

                try:
                    corr_returns[i] = safe_corr(port_ret, spot_ret)
                except Exception:
                    corr_returns[i] = 0.0
                try:
                    corr_pnl[i] = safe_corr(pnl_slice, spot_ret)
                except Exception:
                    corr_pnl[i] = 0.0
            else:
                corr_returns[i] = 0.0
                corr_pnl[i] = 0.0

            prev_delta = pos[i]

        total_trades = int(np.count_nonzero(trades))
        total_costs = float(trade_costs.sum())

        df = pd.DataFrame({
            "price": S,
            "sigma": sig,
            "portfolio": port,
            "option_mt": option_mt,
            "tracking_error": te,
            "daily_pnl": pnl,
            "gamma_exposure": gexp,
            "corr_returns": corr_returns,
            "corr_pnl": corr_pnl,
            "trade": trades,
            "trade_cost": trade_costs
        }, index=dates)

        meta = {
            "total_trades": total_trades,
            "turnover": float(turnover),
            "total_costs": total_costs,
            "K_used": float(K),
            "sell_option": self.sell_option,
            "initial_capital": float(self.initial_capital),
        }
        return df, meta

#diagnostics
def save_pdfs(df: pd.DataFrame, meta: dict, prefix: str) -> List[str]:
    filenames = []

    #price+vol
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df.index, df["price"], 'b-', label="Price")
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.plot(df.index, df["sigma"], 'r--', label="Volatility")
    ax2.set_ylabel('Volatility', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    fig.tight_layout()
    fname = os.path.join(OUTPUT_DIR, "figures", f"{prefix}_price_vol.pdf")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    filenames.append(fname)

    #portfolio+option MTM
    init_cap = float(meta.get("initial_capital", 0.0))
    price_series = df["price"]
    portfolio_series = df["portfolio"]
    option_series = df["option_mt"]

    #centre
    portfolio_centered = portfolio_series - init_cap
    option_centered = option_series - option_series.iloc[0] if len(option_series) > 0 else option_series

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index, portfolio_centered, label="Portfolio (minus initial capital)", linestyle='-', linewidth=1.5)
    ax.plot(df.index, option_centered, label="Option MTM", linestyle='--', linewidth=1.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()
    fig.tight_layout()
    fname = os.path.join(OUTPUT_DIR, "figures", f"{prefix}_portfolio_vs_option.pdf")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    filenames.append(fname)

    #tracking error
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index, df["tracking_error"], label="Tracking Error")
    te_std = float(df["tracking_error"].std())
    ax.axhline(te_std, color="red", linestyle=":", label=f"Std Dev = {te_std:.4f}")
    ax.set_xlabel('Date')
    ax.set_ylabel('Tracking Error')
    ax.legend()
    fig.tight_layout()
    fname = os.path.join(OUTPUT_DIR, "figures", f"{prefix}_tracking_error.pdf")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    filenames.append(fname)

    #gamma + correlations (returns / pnl)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df.index, df["gamma_exposure"], 'b-', label="Gamma Exposure")
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Gamma Exposure', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    # plot both correlation diagnostics on the twin axis
    ax2.plot(df.index, df["corr_pnl"], 'r--', label="Corr (daily PnL vs spot ret)")
    ax2.set_ylabel('Correlation', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    # sensible legend combining both
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    fig.tight_layout()
    fname = os.path.join(OUTPUT_DIR, "figures", f"{prefix}_gamma_corr.pdf")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    filenames.append(fname)

    log(f"Saved {len(filenames)} separate PDFs for {prefix}")
    return filenames

def save_summary_csv(df: pd.DataFrame, meta: dict, prefix: str) -> str:
    summary = {
        "total_trades": meta["total_trades"],
        "turnover": meta["turnover"],
        "total_costs": meta["total_costs"],
        "tracking_error_std": float(df["tracking_error"].std()),
        "daily_pnl_mean": float(df["daily_pnl"].mean()),
        "daily_pnl_std": float(df["daily_pnl"].std()),
        "avg_gamma_exposure": float(df["gamma_exposure"].mean()),
        # updated summary fields: mean of the two new correlation diagnostics
        "avg_corr_returns": float(df["corr_returns"].mean()),
        "avg_corr_pnl": float(df["corr_pnl"].mean()),
    }
    df_sum = pd.DataFrame([summary])
    fname = os.path.join(OUTPUT_DIR, "data", f"{prefix}_summary.csv")
    df_sum.to_csv(fname, index=False)
    log(f"Saved summary {fname}")
    return fname


def run_demo(period: str, explicit_csv: Optional[str] = None, explicit_df: Optional[pd.DataFrame] = None,
             auto_rescale_strike: bool = True):
    data = load_price_data_prefer_local(ticker="AAPL", crisis_period=period, explicit_path=explicit_csv, explicit_df=explicit_df)

    S0 = float(data["Underlying"].iloc[0])
    K = S0 * 1.05  #by default we're near the money
    if period == "2008":
        T = 1.0
        r = 0.02
    else:
        T = 0.5
        r = 0.005

    hedger = AdvancedCrisisHedger(initial_capital=S0*10, r=r, sell_option=True, use_arch=True, auto_rescale_strike=auto_rescale_strike)
    df, meta = hedger.hedge(data, K, T)
    prefix = f"hedge_{period}"
    timeseries_path = os.path.join(OUTPUT_DIR, "data", f"{prefix}_timeseries.csv")
    df.to_csv(timeseries_path)
    figs = save_pdfs(df, meta, prefix)
    summ = save_summary_csv(df, meta, prefix)
    return {"timeseries": timeseries_path, "figures": figs, "summary": summ}, df, meta

if __name__ == "__main__":
    #if you've used synthetic_crises to produce data in the same directory the loader should pick them up
    #or you can use your own file as long as it's labelled correctly
    res1, df1, meta1 = run_demo("2008")
    res2, df2, meta2 = run_demo("2020")

    print("Done demo runs:", res1, res2)
