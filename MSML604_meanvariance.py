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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Optimization ---
import cvxpy as cp


# ===========================================================================
# 0. Configuration
# ===========================================================================

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "GS", "JNJ", "PFE", "XOM", "CVX"]
BENCHMARK_TICKERS = ["SPY", "QQQ"]  # Market benchmarks, not optimized portfolios
START_DATE = "2018-01-01"
END_DATE   = "2024-01-01"
RISK_FREE_RATE = 0.05          # annual
LOOKBACK = 60                  # trading days for feature window
N_LAGS = 5                     # lagged return features
TRAIN_RATIO = 0.75
TARGET_RETURN = 0.12           # annual target for constrained optimisation
N_FRONTIER_POINTS = 50
RANDOM_STATE = 42

# Realistic portfolio constraints/extensions
ALLOW_SHORT = False
MAX_WEIGHT = 0.30              # no asset can exceed 30% of the portfolio
L2_REGULARIZATION = 1e-4       # set to 0.0 to remove portfolio regularization


# ===========================================================================
# 1. Data download & preprocessing
# ===========================================================================

def download_data(tickers, start, end):
    print("Downloading price data...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers[0])
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

    Returns:
      X: feature matrix stacked across all assets
      y: next-period/current-period daily return target
      meta: MultiIndex-like metadata mapping each row to (ticker, date)
    """
    X_list, y_list, meta = [], [], []

    for ticker in returns.columns:
        s = returns[ticker]
        df = pd.DataFrame(index=s.index)

        for lag in range(1, n_lags + 1):
            df[f"lag_{lag}"] = s.shift(lag)

        df["roll_mean"] = s.shift(1).rolling(lookback).mean()
        df["roll_std"]  = s.shift(1).rolling(lookback).std()
        df["momentum"]  = s.shift(1).rolling(lookback).apply(lambda x: (1 + x).prod() - 1)
        df["target"] = s  # predict current-day return using past information only
        df = df.dropna()

        X_list.append(df.drop(columns="target"))
        y_list.append(df["target"])
        meta.extend([(ticker, idx) for idx in df.index])

    X = pd.concat(X_list)
    y = pd.concat(y_list)
    return X, y, meta


def latest_feature_vector_for_asset(asset_returns, n_lags=N_LAGS, lookback=LOOKBACK):
    """Create the most recent feature vector for one asset using only past returns."""
    df = pd.DataFrame(index=asset_returns.index)
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = asset_returns.shift(lag)
    df["roll_mean"] = asset_returns.shift(1).rolling(lookback).mean()
    df["roll_std"]  = asset_returns.shift(1).rolling(lookback).std()
    df["momentum"]  = asset_returns.shift(1).rolling(lookback).apply(lambda x: (1 + x).prod() - 1)
    df = df.dropna()
    return None if df.empty else df.iloc[-1].values


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
    Returns fitted scaler and model dictionary.
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


def evaluate_models(models, scaler, X_test, y_test):
    """Return a dataframe of prediction metrics for the held-out feature split."""
    rows = []
    Xs = scaler.transform(X_test)
    for name, model in models.items():
        pred = model.predict(Xs)
        rows.append({
            "Model": name,
            "MSE": mean_squared_error(y_test, pred),
            "MAE": mean_absolute_error(y_test, pred),
            "R2": r2_score(y_test, pred),
        })
    return pd.DataFrame(rows).sort_values("MSE")


# ===========================================================================
# 5. Generate per-asset expected return estimates
# ===========================================================================

def predict_expected_returns(returns, scaler, models):
    """
    For each asset, predict a next-period daily return using the latest available
    feature vector, then annualize it for portfolio optimization.

    Returns a dict: model_name -> pd.Series(ticker -> annualized expected return)
    """
    latest_features = {
        ticker: latest_feature_vector_for_asset(returns[ticker])
        for ticker in returns.columns
    }

    results = {}
    for name, model in models.items():
        preds = {}
        for ticker, feat in latest_features.items():
            if feat is None:
                preds[ticker] = np.nan
            else:
                feat_s = scaler.transform(feat.reshape(1, -1))
                preds[ticker] = float(model.predict(feat_s)[0])
        results[name] = pd.Series(preds) * 252
    return results


# ===========================================================================
# 6. Historical mean baseline
# ===========================================================================

def historical_expected_returns(returns):
    """Classical Markowitz baseline: historical mean return, annualized."""
    return returns.mean() * 252


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

def min_variance_portfolio(
    mu,
    cov,
    target_return=None,
    allow_short=ALLOW_SHORT,
    max_weight=MAX_WEIGHT,
    l2_reg=L2_REGULARIZATION,
):
    """
    Solve a convex quadratic program:
      min   w'Σw + λ||w||_2^2
      s.t.  sum(w) = 1
            w'µ >= target_return  (if provided)
            w >= 0                (if no short selling)
            w <= max_weight       (if no short selling and max_weight is provided)

    Returns optimal weights as pd.Series.
    """
    n = len(mu)
    w = cp.Variable(n)
    Sigma = cov.values
    objective = cp.Minimize(cp.quad_form(w, Sigma) + l2_reg * cp.sum_squares(w))

    constraints = [cp.sum(w) == 1]
    if not allow_short:
        constraints.append(w >= 0)
        if max_weight is not None:
            constraints.append(w <= max_weight)
    if target_return is not None:
        constraints.append(mu.values @ w >= target_return)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None

    # Clip tiny numerical negatives/overshoots that can appear from solver tolerance.
    weights = pd.Series(w.value, index=mu.index)
    if not allow_short:
        weights = weights.clip(lower=0)
        weights = weights / weights.sum()
    return weights


def max_sharpe_portfolio(
    mu,
    cov,
    rf=RISK_FREE_RATE,
    allow_short=ALLOW_SHORT,
    max_weight=MAX_WEIGHT,
    l2_reg=L2_REGULARIZATION,
):
    """
    Maximise Sharpe ratio via a convex reformulation.

    The no-short and max-weight constraints are transformed consistently:
      y >= 0 and y <= max_weight * kappa.
    """
    n = len(mu)
    excess = mu.values - rf
    if np.all(excess <= 0):
        return min_variance_portfolio(mu, cov, allow_short=allow_short, max_weight=max_weight, l2_reg=l2_reg)

    y = cp.Variable(n)
    kappa = cp.Variable(nonneg=True)
    Sigma = cov.values

    objective = cp.Minimize(cp.quad_form(y, Sigma) + l2_reg * cp.sum_squares(y))
    constraints = [excess @ y == 1, cp.sum(y) == kappa]
    if not allow_short:
        constraints.append(y >= 0)
        if max_weight is not None:
            constraints.append(y <= max_weight * kappa)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate") or kappa.value is None or kappa.value < 1e-8:
        return min_variance_portfolio(mu, cov, allow_short=allow_short, max_weight=max_weight, l2_reg=l2_reg)

    weights = pd.Series(y.value / kappa.value, index=mu.index)
    if not allow_short:
        weights = weights.clip(lower=0)
        weights = weights / weights.sum()
    return weights


def efficient_frontier(
    mu,
    cov,
    n_points=N_FRONTIER_POINTS,
    allow_short=ALLOW_SHORT,
    max_weight=MAX_WEIGHT,
    l2_reg=L2_REGULARIZATION,
):
    """
    Trace the efficient frontier by solving min-variance for a range of
    target returns between feasible low/high expected-return levels.
    """
    low_target = float(mu.quantile(0.10))
    high_target = float(mu.max() * 0.95)
    targets = np.linspace(low_target, high_target, n_points)

    vols, rets = [], []
    for t in targets:
        w = min_variance_portfolio(
            mu, cov, target_return=t, allow_short=allow_short,
            max_weight=max_weight, l2_reg=l2_reg
        )
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
    """Compute out-of-sample metrics given daily returns in the test period."""
    tickers = weights.index
    daily = returns_oos[tickers].dropna()
    port_daily = daily @ weights
    ann_return = port_daily.mean() * 252
    ann_vol    = port_daily.std() * np.sqrt(252)
    sharpe     = (ann_return - rf) / ann_vol if ann_vol > 0 else np.nan

    wealth = (1 + port_daily).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    max_dd = drawdown.min()

    return {
        "Annual Return": ann_return,
        "Annual Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
    }


def single_asset_metrics(asset_returns, rf=RISK_FREE_RATE):
    """Benchmark ETF metrics using the same formulas as optimized portfolios."""
    r = asset_returns.dropna()
    ann_return = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe = (ann_return - rf) / ann_vol if ann_vol > 0 else np.nan
    wealth = (1 + r).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    return {
        "Annual Return": ann_return,
        "Annual Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": drawdown.min(),
    }


def metrics_to_dataframe(results, benchmark_metrics=None):
    rows = []
    for name, info in results.items():
        for portfolio_type, key in [
            ("Max Sharpe", "tangency"),
            ("Minimum Variance", "min_var"),
            ("Target Return", "target_return"),
        ]:
            w = info.get(key)
            m = info.get(f"{key}_metrics")
            if w is not None and m:
                rows.append({"Method": name, "Portfolio": portfolio_type, **m})

    if benchmark_metrics:
        for name, metrics in benchmark_metrics.items():
            rows.append({"Method": name, "Portfolio": "ETF Benchmark", **metrics})

    return pd.DataFrame(rows)


# ===========================================================================
# 10. Reporting & visualisation
# ===========================================================================

def print_summary(label, weights, metrics):
    if weights is None or not metrics:
        print(f"\n{label}: no feasible solution")
        return

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


def plot_results(results, cov, returns_oos, benchmark_returns=None):
    """
    Multi-panel figure:
      A - Efficient frontiers (all optimized methods)
      B - Weight bar chart (max Sharpe portfolios)
      C - Cumulative OOS returns, including benchmarks if available
      D - Metrics comparison table
    """
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#0f0f0f")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    COLORS = {
        "Historical": "#888888",
        "OLS":        "#4A90D9",
        "Ridge":      "#1D9E75",
        "Lasso":      "#F2A623",
        "SPY":        "#CCCCCC",
        "QQQ":        "#BBBBBB",
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
    style_ax(ax_a, "A - Efficient Frontiers")
    for name, info in results.items():
        vols, rets = efficient_frontier(info["mu"], cov)
        ax_a.plot(vols * 100, rets * 100,
                  color=COLORS.get(name, None), linewidth=1.8,
                  label=name, zorder=3)
        tang_w = info["tangency"]
        if tang_w is not None:
            t_ret = float(info["mu"].values @ tang_w.values) * 100
            t_vol = float(np.sqrt(tang_w.values @ cov.values @ tang_w.values)) * 100
            ax_a.scatter(t_vol, t_ret, color=COLORS.get(name, None), s=60, zorder=5,
                         edgecolors="white", linewidths=0.8)
    ax_a.set_xlabel("Volatility (%)", color=TEXT_COLOR, fontsize=9)
    ax_a.set_ylabel("Expected Return (%)", color=TEXT_COLOR, fontsize=9)
    ax_a.legend(fontsize=8, facecolor="#1a1a1a", labelcolor=TEXT_COLOR,
                edgecolor=GRID_COLOR, framealpha=0.8)

    # -- B: Portfolio Weights --
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b, "B - Optimal Weights (Max Sharpe)")
    tickers = list(results["Historical"]["tangency"].index)
    x = np.arange(len(tickers))
    bar_w = 0.2
    offsets = np.linspace(-(len(results)-1)/2, (len(results)-1)/2, len(results)) * bar_w
    for i, (name, info) in enumerate(results.items()):
        w = info["tangency"]
        if w is None:
            continue
        ax_b.bar(x + offsets[i], w.values * 100, bar_w * 0.9,
                 color=COLORS.get(name, None), alpha=0.85, label=name)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(tickers, rotation=45, ha="right", fontsize=8)
    ax_b.set_ylabel("Weight (%)", color=TEXT_COLOR, fontsize=9)
    ax_b.axhline(0, color=GRID_COLOR, linewidth=0.5)
    ax_b.axhline(MAX_WEIGHT * 100, color=GRID_COLOR, linewidth=0.8, linestyle=":", label="Max weight")
    ax_b.legend(fontsize=8, facecolor="#1a1a1a", labelcolor=TEXT_COLOR,
                edgecolor=GRID_COLOR, framealpha=0.8)

    # -- C: Cumulative OOS Returns --
    ax_c = fig.add_subplot(gs[1, 0])
    style_ax(ax_c, "C - Cumulative Out-of-Sample Returns")
    for name, info in results.items():
        w = info["tangency"]
        if w is None:
            continue
        daily_ret = returns_oos[w.index].dropna() @ w
        cumret = (1 + daily_ret).cumprod() - 1
        ax_c.plot(cumret.index, cumret.values * 100,
                  color=COLORS.get(name, None), linewidth=1.6, label=name)

    if benchmark_returns is not None:
        for b in benchmark_returns.columns:
            r = benchmark_returns[b].dropna()
            cumret = (1 + r).cumprod() - 1
            ax_c.plot(cumret.index, cumret.values * 100,
                      linewidth=1.4, linestyle="--", label=f"{b} benchmark")

    ax_c.set_ylabel("Cumulative Return (%)", color=TEXT_COLOR, fontsize=9)
    ax_c.set_xlabel("Date", color=TEXT_COLOR, fontsize=9)
    ax_c.legend(fontsize=8, facecolor="#1a1a1a", labelcolor=TEXT_COLOR,
                edgecolor=GRID_COLOR, framealpha=0.8)

    # -- D: Metrics Table --
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor(BG_COLOR)
    ax_d.axis("off")
    ax_d.set_title("D - Out-of-Sample Metrics (Max Sharpe + Benchmarks)", color=TEXT_COLOR, fontsize=12, pad=10)

    col_labels = ["Method", "Return", "Vol", "Sharpe", "MaxDD"]
    rows = []
    for name, info in results.items():
        m = info["tangency_metrics"]
        rows.append([
            name,
            f"{m['Annual Return']:.1%}",
            f"{m['Annual Volatility']:.1%}",
            f"{m['Sharpe Ratio']:.3f}",
            f"{m['Max Drawdown']:.1%}",
        ])

    if benchmark_returns is not None:
        for b in benchmark_returns.columns:
            m = single_asset_metrics(benchmark_returns[b])
            rows.append([
                f"{b} ETF",
                f"{m['Annual Return']:.1%}",
                f"{m['Annual Volatility']:.1%}",
                f"{m['Sharpe Ratio']:.3f}",
                f"{m['Max Drawdown']:.1%}",
            ])

    tbl = ax_d.table(
        cellText=rows, colLabels=col_labels,
        cellLoc="center", loc="center",
        bbox=[0, 0.15, 1, 0.75]
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
    print("\nFigure saved -> portfolio_results.png")
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
    print(f"\nTrain: {returns_train.index[0].date()} - {returns_train.index[-1].date()}")
    print(f"Test:  {returns_oos.index[0].date()} - {returns_oos.index[-1].date()}")

    # --- Benchmarks for comparison only, not part of optimization ---
    benchmark_returns_oos = None
    benchmark_metrics = {}
    try:
        benchmark_prices = download_data(BENCHMARK_TICKERS, START_DATE, END_DATE)
        benchmark_returns = compute_returns(benchmark_prices)
        benchmark_returns_oos = benchmark_returns.loc[returns_oos.index.min():returns_oos.index.max()]
        benchmark_metrics = {b: single_asset_metrics(benchmark_returns_oos[b]) for b in benchmark_returns_oos.columns}
    except Exception as exc:
        print(f"Benchmark download skipped: {exc}")

    # --- Covariance from training period only ---
    cov = covariance_matrix(returns_train)

    # --- Features & model training ---
    print("\nBuilding features...")
    X, y, _ = build_features(returns_train)
    X_tr, X_te, y_tr, y_te = time_split(X, y)

    print("\nTraining models...")
    scaler, models = train_models(X_tr, y_tr)

    # Held-out feature-split ML metrics
    model_metrics = evaluate_models(models, scaler, X_te, y_te)
    print("\nHeld-out prediction metrics:")
    print(model_metrics.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    model_metrics.to_csv("model_prediction_metrics.csv", index=False)
    print("Model metrics saved -> model_prediction_metrics.csv")

    # --- Expected returns ---
    print("\nGenerating expected return estimates...")
    ml_mu = predict_expected_returns(returns, scaler, models)
    hist_mu = historical_expected_returns(returns_train)

    # Align tickers across all mu vectors; fall back to historical means if needed.
    for name in ml_mu:
        ml_mu[name] = ml_mu[name].reindex(tickers).fillna(hist_mu)

    all_mu = {"Historical": hist_mu.reindex(tickers), **ml_mu}

    # --- Optimisation & evaluation ---
    print("\nOptimising portfolios...")
    results = {}
    for name, mu in all_mu.items():
        tang_w = max_sharpe_portfolio(mu, cov)
        minvar_w = min_variance_portfolio(mu, cov)
        target_w = min_variance_portfolio(mu, cov, target_return=TARGET_RETURN)

        tang_m = portfolio_metrics(tang_w, returns_oos) if tang_w is not None else {}
        minvar_m = portfolio_metrics(minvar_w, returns_oos) if minvar_w is not None else {}
        target_m = portfolio_metrics(target_w, returns_oos) if target_w is not None else {}

        results[name] = {
            "mu": mu,
            "tangency": tang_w,
            "min_var": minvar_w,
            "target_return": target_w,
            "tangency_metrics": tang_m,
            "min_var_metrics": minvar_m,
            "target_return_metrics": target_m,
        }
        print_summary(f"{name} - Tangency Portfolio", tang_w, tang_m)
        print_summary(f"{name} - Minimum Variance Portfolio", minvar_w, minvar_m)
        print_summary(f"{name} - Target Return Portfolio ({TARGET_RETURN:.0%})", target_w, target_m)

    # --- Save report-ready result tables ---
    portfolio_metrics_df = metrics_to_dataframe(results, benchmark_metrics)
    portfolio_metrics_df.to_csv("portfolio_performance_metrics.csv", index=False)
    print("\nPortfolio metrics saved -> portfolio_performance_metrics.csv")

    if benchmark_metrics:
        print("\nBenchmark metrics:")
        for b, m in benchmark_metrics.items():
            print(f"  {b}: return={m['Annual Return']:.2%}, vol={m['Annual Volatility']:.2%}, "
                  f"Sharpe={m['Sharpe Ratio']:.3f}, maxDD={m['Max Drawdown']:.2%}")

    # --- Visualisation ---
    print("\nGenerating plots...")
    plot_results(results, cov, returns_oos, benchmark_returns_oos)

    return results, cov, returns_oos, model_metrics, portfolio_metrics_df


if __name__ == "__main__":
    results, cov, returns_oos, model_metrics, portfolio_metrics_df = main()
