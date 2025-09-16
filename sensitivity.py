#!/usr/bin/env python3
#sensitivity.py
#makes a sensitvity grid

import csv
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from typing import Sequence, Tuple

def compute_grid_metrics(data: pd.DataFrame,
                         K: float,
                         T_years: float,
                         delta_vals: Sequence[float],
                         gamma_vals: Sequence[float],
                         hedger_kwargs: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    nd = len(delta_vals)
    ng = len(gamma_vals)
    te_mat = np.full((nd, ng), np.nan, dtype=float)
    turnover_mat = np.full((nd, ng), np.nan, dtype=float)
    costs_mat = np.full((nd, ng), np.nan, dtype=float)
    trades_mat = np.full((nd, ng), np.nan, dtype=float)

    for i, d in enumerate(delta_vals):
        for j, g in enumerate(gamma_vals):
            try:
                hedger = AdvancedCrisisHedger(delta_thresh=float(d), gamma_thresh=float(g), **hedger_kwargs)
                df, meta = hedger.hedge(data.copy(), K, T_years)
                te_mat[i, j] = float(df["tracking_error"].std())
                turnover_mat[i, j] = float(meta.get("turnover", np.nan))
                costs_mat[i, j] = float(meta.get("total_costs", float(df.get("trade_cost", pd.Series()).sum())))
                trades_mat[i, j] = float(meta.get("total_trades", np.count_nonzero(df.get("trade", []))))
                
            except Exception as e:
                print(f"[sensitivity] failed at delta={d}, gamma={g}: {e}")
                continue
    return te_mat, turnover_mat, costs_mat, trades_mat

def pareto_frontier_from_pairs(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    mask = np.isfinite(x) & np.isfinite(y)
    xs = x[mask]
    ys = y[mask]
    if len(xs) == 0:
        return np.array([]), np.array([])

    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    pareto_x = []
    pareto_y = []
    best_y = np.inf

    for xi, yi in zip(xs, ys):
        if yi < best_y - 1e-12:  # tolerance
            pareto_x.append(xi)
            pareto_y.append(yi)
            best_y = yi
    return np.array(pareto_x), np.array(pareto_y)

def sensitivity_grid_and_save_pdf(period: str,
                                  K: float,
                                  T_years: float,
                                  delta_vals: Sequence[float] = None,
                                  gamma_vals: Sequence[float] = None,
                                  hedger_kwargs: dict = None,
                                  output_prefix: str = None):

    if delta_vals is None:
        delta_vals = np.linspace(0.02, 0.30, 10)
    if gamma_vals is None:
        gamma_vals = np.linspace(0.002, 0.04, 10)
    if hedger_kwargs is None:
        hedger_kwargs = {"initial_capital": 0.0, "r": 0.01, "transaction_cost": 0.001, "margin_limit": 0.2}
    if output_prefix is None:
        output_prefix = f"sensitivity_{period}"

    data = load_price_data_prefer_local(ticker="AAPL", crisis_period=period)
    print(f"[sensitivity] Using data for period {period} with {len(data)} rows. Running grid {len(delta_vals)}x{len(gamma_vals)}")

    te_mat, turnover_mat, costs_mat, trades_mat = compute_grid_metrics(data, K, T_years, delta_vals, gamma_vals, hedger_kwargs)

    os.makedirs(os.path.join(OUTPUT_DIR, "data"), exist_ok=True)
    te_df = pd.DataFrame(te_mat, index=delta_vals, columns=gamma_vals)
    to_df = pd.DataFrame(turnover_mat, index=delta_vals, columns=gamma_vals)
    costs_df = pd.DataFrame(costs_mat, index=delta_vals, columns=gamma_vals)
    trades_df = pd.DataFrame(trades_mat, index=delta_vals, columns=gamma_vals)

    te_csv = os.path.join(OUTPUT_DIR, "data", f"{output_prefix}_te_grid.csv")
    to_csv = os.path.join(OUTPUT_DIR, "data", f"{output_prefix}_turnover_grid.csv")
    costs_csv = os.path.join(OUTPUT_DIR, "data", f"{output_prefix}_costs_grid.csv")
    trades_csv = os.path.join(OUTPUT_DIR, "data", f"{output_prefix}_trades_grid.csv")

    te_df.to_csv(te_csv)
    to_df.to_csv(to_csv)
    costs_df.to_csv(costs_csv)
    trades_df.to_csv(trades_csv)
    print(f"[sensitivity] Saved CSVs: {te_csv}, {to_csv}, {costs_csv}, {trades_csv}")

    #plots
    os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)
    pdf_path = os.path.join(OUTPUT_DIR, "figures", f"{output_prefix}.pdf")
    with PdfPages(pdf_path) as pdf:
        #error stdev
        fig, ax = plt.subplots(figsize=(8,6))
        im = ax.imshow(te_mat, origin="lower", aspect="auto",
                       interpolation="nearest")
        ax.set_title(f"{period} - Tracking Error Std (grid)")
        ax.set_xlabel("gamma_thresh index")
        ax.set_ylabel("delta_thresh index")

        nx = len(gamma_vals); ny = len(delta_vals)
        xt = np.linspace(0, nx - 1, min(nx, 6)).astype(int)
        yt = np.linspace(0, ny - 1, min(ny, 6)).astype(int)
        ax.set_xticks(xt); ax.set_yticks(yt)
        ax.set_xticklabels([f"{gamma_vals[k]:.4f}" for k in xt], rotation=45)
        ax.set_yticklabels([f"{delta_vals[k]:.3f}" for k in yt])
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Tracking Error Std")
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        #turnover
        fig, ax = plt.subplots(figsize=(8,6))
        im = ax.imshow(turnover_mat, origin="lower", aspect="auto",
                       interpolation="nearest")
        ax.set_title(f"{period} - Turnover (notional)")
        ax.set_xlabel("gamma_thresh index")
        ax.set_ylabel("delta_thresh index")
        ax.set_xticks(xt); ax.set_yticks(yt)
        ax.set_xticklabels([f"{gamma_vals[k]:.4f}" for k in xt], rotation=45)
        ax.set_yticklabels([f"{delta_vals[k]:.3f}" for k in yt])
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Turnover")
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        #pareto
        X = turnover_mat.flatten()
        Y = te_mat.flatten()
        mask = np.isfinite(X) & np.isfinite(Y)
        X_valid = X[mask]
        Y_valid = Y[mask]

        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(X_valid, Y_valid, alpha=0.8, s=40, label="grid points")
        if len(Y_valid) > 0:
            idx_min_te = np.argmin(Y_valid)
            ax.scatter([X_valid[idx_min_te]], [Y_valid[idx_min_te]], color='red', s=100, label='min TE')
            ax.annotate("min TE", (X_valid[idx_min_te], Y_valid[idx_min_te]), textcoords="offset points", xytext=(5,5))

        #frontier plot
        px, py = pareto_frontier_from_pairs(X_valid, Y_valid)
        if px.size > 0:
            ax.plot(px, py, color='black', linestyle='-', linewidth=2, label='Pareto frontier')
        ax.set_xlabel("Turnover (notional)")
        ax.set_ylabel("Tracking Error Std")
        ax.set_title(f"{period} - Turnover vs TE (Pareto)")
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        #best points
        best_idx = np.argsort(Y_valid)[:10]
        rows = []
        for k in best_idx:
            # find grid coords
            flat_index = np.where((turnover_mat.flatten() == X_valid[k]) & (te_mat.flatten() == Y_valid[k]))[0]
            # best match: take first flat index if multiple
            if flat_index.size > 0:
                fi = flat_index[0]
                di = fi // turnover_mat.shape[1]
                gi = fi % turnover_mat.shape[1]
                rows.append((delta_vals[di], gamma_vals[gi], float(Y_valid[k]), float(X_valid[k])))

        if rows:
            df_rows = pd.DataFrame(rows, columns=["delta_thresh","gamma_thresh","te_std","turnover"])
            fig, ax = plt.subplots(figsize=(8,3 + 0.3 * len(df_rows)))
            ax.axis('off')
            tbl = ax.table(cellText=df_rows.round(6).values, colLabels=df_rows.columns, loc='center')
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1, 1.2)
            plt.title(f"{period} - Top {len(df_rows)} grid points by TE")
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

    print(f"[sensitivity] Saved PDF to {pdf_path}")
    return {
        "te_csv": te_csv,
        "turnover_csv": to_csv,
        "costs_csv": costs_csv,
        "trades_csv": trades_csv,
        "pdf": pdf_path
    }

#the strike and values will often need adjusting to match the relevant value here
#if you pick a gamma range too large or too small you get no useful information
result = sensitivity_grid_and_save_pdf("2008", K=5.25, T_years=1.0,
                                      delta_vals=np.linspace(0.02,0.4,10),
                                      gamma_vals=np.linspace(0.02,0.5,10),
                                      hedger_kwargs={"initial_capital":0.0,"r":0.02,"transaction_cost":0.001})
result = sensitivity_grid_and_save_pdf("2020", K=140, T_years=1.0,
                                      delta_vals=np.linspace(0.02,0.4,10),
                                      gamma_vals=np.linspace(0.003,0.008,10),
                                      hedger_kwargs={"initial_capital":0.0,"r":0.02,"transaction_cost":0.001})
print("Grid done:", result)
