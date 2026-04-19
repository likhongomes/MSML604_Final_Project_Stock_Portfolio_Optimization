"""
ML-Enhanced Mean-Variance Portfolio Optimization
=================================================
Pipeline:
  1. Download historical stock data (yfinance)
  2. Engineer features from return history
  3. Train OLS, Ridge, and Lasso models to predict next-period returns
  4. Feed ML predictions into a convex QP (cvxpy) to find optimal weights
  5. Evaluate: Sharpe ratio, volatility, efficient frontier, out-of-sample test
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# --- Data ---
import yfinance as yf

# --- ML ---
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# --- Optimization ---
import cvxpy as cp


# ===========================================================================
# 0. Configuration
# ===========================================================================

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "GS", "JNJ", "PFE", "XOM", "CVX"]
START_DATE = "2018-01-01"
END_DATE   = "2024-01-01"
RISK_FREE_RATE = 0.05          # annual
LOOKBACK = 60                  # trading days for feature window
N_LAGS = 5                     # lagged return features
TRAIN_RATIO = 0.75
TARGET_RETURN = 0.12           # annual target for constrained optimisation
N_FRONTIER_POINTS = 50
RANDOM_STATE = 42


# ===========================================================================
# 1. Data download & preprocessing
# ===========================================================================

def download_data(tickers, start, end):
    print("Downloading price data...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    raw = raw.dropna(axis=1, how="all").dropna()
    print(f"  {raw.shape[1]} assets, {raw.shape[0]} trading days")
    return raw

def compute_returns(prices):
    return prices.pct_change().dropna()


# ===========================================================================
# 2. Feature engineering
# ===========================================================================

def build_features(returns, n_lags=N_LAGS, lookback=LOOKBACK):
    """
    For each asset we create:
      - n_lags lagged daily returns
      - rolling mean / std over `lookback` days
      - rolling momentum (cumulative return over lookback)
    Returns X (features), y (next-day return) — stacked across all assets.
    """
    X_list, y_list, meta = [], [], []

    for ticker in returns.columns:
        s = returns[ticker]
        df = pd.DataFrame(index=s.index)

        for lag in range(1, n_lags + 1):
            df[f"lag_{lag}"] = s.shift(lag)

        df["roll_mean"] = s.shift(1).rolling(lookback).mean()
        df["roll_std"]  = s.shift(1).rolling(lookback).std()
        df["momentum"]  = s.shift(1).rolling(lookback).apply(lambda x: (1+x).prod() - 1)

        df["target"] = s  # predict current-day return

        df = df.dropna()
        X_list.append(df.drop(columns="target"))
        y_list.append(df["target"])
        meta.extend([(ticker, idx) for idx in df.index])

    X = pd.concat(X_list).sort_index()
    y = pd.concat(y_list).sort_index()
    return X, y, meta


# ===========================================================================
# 3. Train / test split (time-series safe)
# ===========================================================================

def time_split(X, y, ratio=TRAIN_RATIO):
    n = len(X)
    cut = int(n * ratio)
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


# ===========================================================================
# 4. Model training with hyperparameter tuning
# ===========================================================================

def train_models(X_train, y_train):
    """
    Train OLS, Ridge, and Lasso with time-series cross-validation.
    Returns fitted (scaler, model) tuples.
    """
    tscv = TimeSeriesSplit(n_splits=5)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)

    # --- OLS ---
    ols = LinearRegression()
    ols.fit(Xs, y_train)
    print("  OLS trained")

    # --- Ridge (tune alpha) ---
    ridge_cv = GridSearchCV(
        Ridge(),
        {"alpha": np.logspace(-3, 3, 20)},
        cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1
    )
    ridge_cv.fit(Xs, y_train)
    ridge = ridge_cv.best_estimator_
    print(f"  Ridge trained  | best alpha = {ridge.alpha:.4f}")

    # --- Lasso (tune alpha) ---
    lasso_cv = GridSearchCV(
        Lasso(max_iter=10_000),
        {"alpha": np.logspace(-5, 0, 20)},
        cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1
    )
    lasso_cv.fit(Xs, y_train)
    lasso = lasso_cv.best_estimator_
    print(f"  Lasso trained  | best alpha = {lasso.alpha:.6f}")

    return scaler, {"OLS": ols, "Ridge": ridge, "Lasso": lasso}


# ===========================================================================
# 5. Generate per-asset expected return estimates
# ===========================================================================

def predict_expected_returns(returns, scaler, models, test_start_idx):
    """
    For each asset, predict next-period expected return using the latest
    available feature vector from the test window.
    Returns a dict: model_name -> pd.Series(ticker -> expected_return)
    """
    X_full, _, _ = build_features(returns)

    # Latest feature row per asset
    tickers = returns.columns.tolist()
    latest_features = {}
    for ticker in tickers:
        mask = [m[0] == ticker for m in _meta_lookup(X_full, ticker)]
        subset = X_full[X_full.index >= returns.index[test_start_idx]]
        # grab the last available row for this ticker from test period
        # We reconstruct per-asset last row
        s = returns[ticker]
        df = pd.DataFrame(index=s.index)
        for lag in range(1, N_LAGS + 1):
            df[f"lag_{lag}"] = s.shift(lag)
        df["roll_mean"] = s.shift(1).rolling(LOOKBACK).mean()
        df["roll_std"]  = s.shift(1).rolling(LOOKBACK).std()
        df["momentum"]  = s.shift(1).rolling(LOOKBACK).apply(lambda x: (1+x).prod() - 1)
        df = df.dropna()
        if len(df) == 0:
            latest_features[ticker] = None
        else:
            latest_features[ticker] = df.iloc[-1].values

    results = {}
    for name, model in models.items():
        preds = {}
        for ticker, feat in latest_features.items():
            if feat is None:
                preds[ticker] = 0.0
            else:
                feat_s = scaler.transform(feat.reshape(1, -1))
                preds[ticker] = float(model.predict(feat_s)[0])
        # Annualise daily prediction
        results[name] = pd.Series(preds) * 252
    return results

def _meta_lookup(X_full, ticker):
    # Helper placeholder; actual filtering done inline above
    return []


# ===========================================================================
# 6. Historical mean baseline
# ===========================================================================

def historical_expected_returns(returns):
    return returns.mean() * 252  # annualised


# ===========================================================================
# 7. Covariance matrix (Ledoit-Wolf shrinkage optional)
# ===========================================================================

def covariance_matrix(returns):
    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf()
        lw.fit(returns)
        cov = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns) * 252
        print("  Covariance: Ledoit-Wolf shrinkage applied")
    except Exception:
        cov = returns.cov() * 252
        print("  Covariance: sample covariance")
    return cov


# ===========================================================================
# 8. Portfolio optimisation (convex QP via cvxpy)
# ===========================================================================

def min_variance_portfolio(mu, cov, target_return=None, allow_short=False):
    """
    Solve:
      min   w'Σw
      s.t.  sum(w) = 1
            w'µ >= target_return  (if provided)
            w >= 0                (if no short selling)
    Returns optimal weights as pd.Series.
    """
    n = len(mu)
    w = cp.Variable(n)
    Sigma = cov.values
    objective = cp.Minimize(cp.quad_form(w, Sigma))

    constraints = [cp.sum(w) == 1]
    if not allow_short:
        constraints.append(w >= 0)
    if target_return is not None:
        constraints.append(mu.values @ w >= target_return)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None

    return pd.Series(w.value, index=mu.index)


def max_sharpe_portfolio(mu, cov, rf=RISK_FREE_RATE, allow_short=False):
    """
    Maximise Sharpe ratio via the Tobin trick (convex reformulation).
    """
    n = len(mu)
    excess = mu.values - rf
    if np.all(excess <= 0):
        return min_variance_portfolio(mu, cov, allow_short=allow_short)

    y = cp.Variable(n)
    kappa = cp.Variable(nonneg=True)
    Sigma = cov.values

    objective = cp.Minimize(cp.quad_form(y, Sigma))
    constraints = [excess @ y == 1, cp.sum(y) == kappa]
    if not allow_short:
        constraints.append(y >= 0)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate") or kappa.value is None or kappa.value < 1e-8:
        return min_variance_portfolio(mu, cov, allow_short=allow_short)

    w_raw = y.value / kappa.value
    return pd.Series(w_raw, index=mu.index)


def efficient_frontier(mu, cov, n_points=N_FRONTIER_POINTS, allow_short=False):
    """
    Trace the efficient frontier by solving min-variance for a range of
    target returns between the minimum-variance return and the max return.
    """
    min_ret = mu.min() * 0.5
    max_ret = mu.max() * 0.95
    targets = np.linspace(min_ret, max_ret, n_points)

    vols, rets = [], []
    for t in targets:
        w = min_variance_portfolio(mu, cov, target_return=t, allow_short=allow_short)
        if w is not None:
            port_ret = float(mu.values @ w.values)
            port_vol = float(np.sqrt(w.values @ cov.values @ w.values))
            rets.append(port_ret)
            vols.append(port_vol)

    return np.array(vols), np.array(rets)


# ===========================================================================
# 9. Performance metrics
# ===========================================================================

def portfolio_metrics(weights, returns_oos, rf=RISK_FREE_RATE):
    """
    Compute out-of-sample metrics given daily returns in the test period.
    """
    tickers = weights.index
    daily = returns_oos[tickers].dropna()
    port_daily = daily @ weights
    ann_return = port_daily.mean() * 252
    ann_vol    = port_daily.std() * np.sqrt(252)
    sharpe     = (ann_return - rf) / ann_vol if ann_vol > 0 else np.nan
    max_dd     = (port_daily.cumsum() - port_daily.cumsum().cummax()).min()
    return {
        "Annual Return": ann_return,
        "Annual Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
    }


# ===========================================================================
# 10. Reporting & visualisation
# ===========================================================================

def print_summary(label, weights, metrics):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Annual Return    : {metrics['Annual Return']:>8.2%}")
    print(f"  Annual Volatility: {metrics['Annual Volatility']:>8.2%}")
    print(f"  Sharpe Ratio     : {metrics['Sharpe Ratio']:>8.3f}")
    print(f"  Max Drawdown     : {metrics['Max Drawdown']:>8.2%}")
    print(f"\n  Top 5 weights:")
    top = weights.abs().nlargest(5)
    for ticker in top.index:
        print(f"    {ticker:6s}  {weights[ticker]:>7.2%}")


def plot_results(results, cov, returns_oos):
    """
    Multi-panel figure:
      A – Efficient frontiers (all methods)
      B – Weight bar chart (all methods)
      C – Cumulative OOS returns
      D – Metrics comparison table
    """
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#0f0f0f")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    COLORS = {
        "Historical": "#888888",
        "OLS":        "#4A90D9",
        "Ridge":      "#1D9E75",
        "Lasso":      "#F2A623",
    }
    TEXT_COLOR = "#e0e0e0"
    GRID_COLOR = "#2a2a2a"
    BG_COLOR   = "#181818"

    def style_ax(ax, title):
        ax.set_facecolor(BG_COLOR)
        ax.set_title(title, color=TEXT_COLOR, fontsize=12, pad=10)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax.spines[:].set_color(GRID_COLOR)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.grid(color=GRID_COLOR, linewidth=0.4, linestyle="--")

    # -- A: Efficient Frontiers --
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a, "A — Efficient Frontiers")
    for name, info in results.items():
        vols, rets = efficient_frontier(info["mu"], cov)
        ax_a.plot(vols * 100, rets * 100,
                  color=COLORS[name], linewidth=1.8,
                  label=name, zorder=3)
        # Tangency
        tang_w = info["tangency"]
        if tang_w is not None:
            t_ret = float(info["mu"].values @ tang_w.values) * 100
            t_vol = float(np.sqrt(tang_w.values @ cov.values @ tang_w.values)) * 100
            ax_a.scatter(t_vol, t_ret, color=COLORS[name], s=60, zorder=5,
                         edgecolors="white", linewidths=0.8)
    ax_a.set_xlabel("Volatility (%)", color=TEXT_COLOR, fontsize=9)
    ax_a.set_ylabel("Return (%)", color=TEXT_COLOR, fontsize=9)
    ax_a.legend(fontsize=8, facecolor="#1a1a1a", labelcolor=TEXT_COLOR,
                edgecolor=GRID_COLOR, framealpha=0.8)

    # -- B: Portfolio Weights --
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b, "B — Optimal Weights (Max Sharpe)")
    tickers = list(results["Historical"]["tangency"].index)
    x = np.arange(len(tickers))
    bar_w = 0.2
    offsets = np.linspace(-(len(results)-1)/2, (len(results)-1)/2, len(results)) * bar_w
    for i, (name, info) in enumerate(results.items()):
        w = info["tangency"]
        if w is None:
            continue
        ax_b.bar(x + offsets[i], w.values * 100, bar_w * 0.9,
                 color=COLORS[name], alpha=0.85, label=name)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(tickers, rotation=45, ha="right", fontsize=8)
    ax_b.set_ylabel("Weight (%)", color=TEXT_COLOR, fontsize=9)
    ax_b.axhline(0, color=GRID_COLOR, linewidth=0.5)
    ax_b.legend(fontsize=8, facecolor="#1a1a1a", labelcolor=TEXT_COLOR,
                edgecolor=GRID_COLOR, framealpha=0.8)

    # -- C: Cumulative OOS Returns --
    ax_c = fig.add_subplot(gs[1, 0])
    style_ax(ax_c, "C — Cumulative Out-of-Sample Returns")
    for name, info in results.items():
        w = info["tangency"]
        if w is None:
            continue
        daily_ret = returns_oos[w.index].dropna() @ w
        cumret = (1 + daily_ret).cumprod() - 1
        ax_c.plot(cumret.index, cumret.values * 100,
                  color=COLORS[name], linewidth=1.6, label=name)
    ax_c.set_ylabel("Cumulative Return (%)", color=TEXT_COLOR, fontsize=9)
    ax_c.set_xlabel("Date", color=TEXT_COLOR, fontsize=9)
    ax_c.legend(fontsize=8, facecolor="#1a1a1a", labelcolor=TEXT_COLOR,
                edgecolor=GRID_COLOR, framealpha=0.8)

    # -- D: Metrics Table --
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor(BG_COLOR)
    ax_d.axis("off")
    ax_d.set_title("D — Out-of-Sample Metrics", color=TEXT_COLOR, fontsize=12, pad=10)

    col_labels = ["Method", "Return", "Vol", "Sharpe", "MaxDD"]
    rows = []
    for name, info in results.items():
        m = info["oos_metrics"]
        rows.append([
            name,
            f"{m['Annual Return']:.1%}",
            f"{m['Annual Volatility']:.1%}",
            f"{m['Sharpe Ratio']:.3f}",
            f"{m['Max Drawdown']:.1%}",
        ])

    tbl = ax_d.table(
        cellText=rows, colLabels=col_labels,
        cellLoc="center", loc="center",
        bbox=[0, 0.2, 1, 0.7]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#1e1e1e" if r % 2 == 0 else "#151515")
        cell.set_edgecolor(GRID_COLOR)
        cell.set_text_props(color=TEXT_COLOR)
        if r == 0:
            cell.set_facecolor("#252525")
            cell.set_text_props(color="#aaaaaa", fontweight="bold")

    plt.suptitle(
        "ML-Enhanced Mean-Variance Portfolio Optimization",
        color=TEXT_COLOR, fontsize=14, y=0.98, fontweight="bold"
    )
    plt.savefig("portfolio_results.png",
                dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("\nFigure saved → portfolio_results.png")
    plt.show()


# ===========================================================================
# 11. Main pipeline
# ===========================================================================

def main():
    np.random.seed(RANDOM_STATE)

    # --- Data ---
    prices  = download_data(TICKERS, START_DATE, END_DATE)
    returns = compute_returns(prices)
    tickers = returns.columns.tolist()

    split_idx = int(len(returns) * TRAIN_RATIO)
    returns_train = returns.iloc[:split_idx]
    returns_oos   = returns.iloc[split_idx:]
    print(f"\nTrain: {returns_train.index[0].date()} – {returns_train.index[-1].date()}")
    print(f"Test:  {returns_oos.index[0].date()}  – {returns_oos.index[-1].date()}")

    # --- Covariance (full history for stability) ---
    cov = covariance_matrix(returns_train)

    # --- Features & model training ---
    print("\nBuilding features...")
    X, y, _ = build_features(returns_train)
    X_tr, X_te, y_tr, y_te = time_split(X, y)

    print("\nTraining models...")
    scaler, models = train_models(X_tr, y_tr)

    # In-sample MSE
    for name, model in models.items():
        pred = model.predict(scaler.transform(X_te))
        mse = mean_squared_error(y_te, pred)
        print(f"  {name:6s} OOS MSE = {mse:.6f}")

    # --- Expected returns ---
    print("\nGenerating expected return estimates...")
    ml_mu = predict_expected_returns(returns, scaler, models, split_idx)
    hist_mu = historical_expected_returns(returns_train)

    # Align tickers across all mu vectors
    for name in ml_mu:
        ml_mu[name] = ml_mu[name].reindex(tickers).fillna(hist_mu)

    all_mu = {"Historical": hist_mu.reindex(tickers), **ml_mu}

    # --- Optimisation & evaluation ---
    print("\nOptimising portfolios...")
    results = {}
    for name, mu in all_mu.items():
        tang_w = max_sharpe_portfolio(mu, cov)
        minvar_w = min_variance_portfolio(mu, cov)
        oos_m = portfolio_metrics(tang_w, returns_oos) if tang_w is not None else {}
        results[name] = {
            "mu":         mu,
            "tangency":   tang_w,
            "min_var":    minvar_w,
            "oos_metrics": oos_m,
        }
        print_summary(f"{name} — Tangency Portfolio", tang_w, oos_m)

    # --- Visualisation ---
    print("\nGenerating plots...")
    plot_results(results, cov, returns_oos)

    return results, cov, returns_oos


if __name__ == "__main__":
    results, cov, returns_oos = main()