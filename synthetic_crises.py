#synthetic_crises.py
#generate synthetic data for two famous crises
#(getting real data is a headache)

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple

def _simulate_garch_jumps(
    n_days: int,
    s0: float,
    mu_daily: float,
    sigma0_annual: float,
    alpha: float,
    beta: float,
    omega: Optional[float] = None,
    jump_prob_daily: float = 0.0,
    jump_mu: float = 0.0,
    jump_sigma: float = 0.0,
    forced_jumps: Optional[List[Tuple[int, float]]] = None,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Simulate log-price process with GARCH-like variance update and Poisson jumps.
    - n_days: number of business days
    - s0: initial spot price
    - mu_daily: drift (log-return) per day (decimal, e.g. -0.0005)
    - sigma0_annual: initial annualised sigma (decimal, e.g. 0.6)
    - alpha, beta: GARCH parameters for daily var: v_t = omega + alpha * r_{t-1}^2 + beta * v_{t-1}
    - omega: if None, set so that long-run var matches sigma0_annual^2 / 252 with stationarity
    - jump_prob_daily: probability of a jump on any day
    - jump_mu, jump_sigma: normal(mean,sd) for jump log-return size (additive)
    - forced_jumps: list of (index, jump_size) for deterministic spikes (index in [0..n_days-1])
    Returns DataFrame indexed by business days with columns Underlying, Returns, Sigma (annualised rolling 10-day)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = 1.0 / 252.0
    #inital daily var
    v0 = (sigma0_annual ** 2) * dt
    if omega is None:
        #omega has to satisfy eqn
        denom = max(1.0 - alpha - beta, 1e-8)
        omega = v0 * denom

    r = np.zeros(n_days)          #daily log-returns
    s = np.zeros(n_days)          #prices
    v = np.zeros(n_days)          #daily variance (not annualised)
    s[0] = s0
    v[0] = v0

    forced = {idx: size for idx, size in (forced_jumps or [])}

    for t in range(1, n_days):
        eps = np.random.normal()
        jump = 0.0 #draw jump
        if t in forced:
            jump = forced[t]
        else:
            if np.random.rand() < jump_prob_daily:
                jump = np.random.normal(loc=jump_mu, scale=jump_sigma)
        #draw return
        rt = mu_daily + math.sqrt(max(v[t-1], 1e-12)) * eps + jump
        r[t] = rt
        s[t] = s[t-1] * math.exp(rt)
        v[t] = omega + alpha * (r[t] ** 2) + beta * v[t-1] #updates variance
        if v[t] < 1e-12:
            v[t] = 1e-12

    return s, r, v

def _make_df_from_sim(s, r, v, start_date: str):
    n = len(s)
    dates = pd.bdate_range(start=start_date, periods=n)
    df = pd.DataFrame(index=dates)
    df['Underlying'] = s
    df['Returns'] = r
    df['Sigma'] = df['Returns'].rolling(window=10, min_periods=3).std() * math.sqrt(252.0) #ten-day rolling window
    
    if df['Sigma'].isna().any():
        ann_sigma_from_v = np.sqrt(v * 252.0) #annualise
        df['Sigma'] = df['Sigma'].fillna(pd.Series(ann_sigma_from_v, index=df.index))
        
    df['Sigma'] = df['Sigma'].clip(lower=0.005, upper=3.0)
    return df

def generate_2008_synthetic(
    n_days: int = 300,
    start_date: str = "2007-10-01",
    s0: float = 5.0,
    seed: Optional[int] = 1
) -> pd.DataFrame:
    
    #imitates 2008: high vol, many negative jumps, high persistence
    sigma0_annual = 0.60
    mu_daily = -0.0003
    alpha = 0.07
    beta = 0.89

    jump_prob_daily = 0.01
    jump_mu = -0.06
    jump_sigma = 0.08

    forced_jumps = [(25, -0.15), (80, -0.12), (230, -0.16)]  #force notable jumps
    s, r, v = _simulate_garch_jumps(
        n_days=n_days, s0=s0, mu_daily=mu_daily, sigma0_annual=sigma0_annual,
        alpha=alpha, beta=beta, omega=None,
        jump_prob_daily=jump_prob_daily, jump_mu=jump_mu, jump_sigma=jump_sigma,
        forced_jumps=forced_jumps, seed=seed
    )
    df = _make_df_from_sim(s, r, v, start_date=start_date)
    return df

def generate_2020_synthetic(
    n_days: int = 250,
    start_date: str = "2020-01-01",
    s0: float = 75.0,
    seed: Optional[int] = 2
) -> pd.DataFrame:
    
    #less immediate vol, faster recovery

    sigma0_annual = 0.35
    mu_daily = 0.0003
    alpha = 0.10
    beta = 0.6
    jump_prob_daily = 0.01
    jump_mu = 0.03
    jump_sigma = 0.06

    forced_jumps = [(50, -0.30)]  #march-ish crash
    s, r, v = _simulate_garch_jumps(
        n_days=n_days, s0=s0, mu_daily=mu_daily, sigma0_annual=sigma0_annual,
        alpha=alpha, beta=beta, omega=None,
        jump_prob_daily=jump_prob_daily, jump_mu=jump_mu, jump_sigma=jump_sigma,
        forced_jumps=forced_jumps, seed=seed
    )
    df = _make_df_from_sim(s, r, v, start_date=start_date)
    return df

#actually do it
if __name__ == "__main__":
    df2008 = generate_2008_synthetic(n_days=300, start_date="2007-10-01", s0=5.0, seed=42)
    df2020 = generate_2020_synthetic(n_days=250, start_date="2020-01-01", s0=75.0, seed=43)

    df2008.to_csv("synthetic_2008.csv")
    df2020.to_csv("synthetic_2020.csv")
    print("Saved synthetic_2008.csv and synthetic_2020.csv (columns: Underlying, Returns, Sigma)")

    #what does this actually look like?
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    df2008['Underlying'].plot(ax=axes[0,0], title='2008 Synthetic Price')
    df2008['Sigma'].plot(ax=axes[1,0], title='2008 Sigma (annualised)')
    df2020['Underlying'].plot(ax=axes[0,1], title='2020 Synthetic Price')
    df2020['Sigma'].plot(ax=axes[1,1], title='2020 Sigma (annualised)')
    plt.tight_layout()
    plt.show()
