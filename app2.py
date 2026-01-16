import numpy as np
import pandas as pd
import streamlit as st

from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.covariance import LedoitWolf
from hmmlearn.hmm import GaussianHMM

import plotly.express as px
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# ======================================================
# TITRE
# ======================================================
st.title("📊 ETF Portfolio Optimizer & Risk Engine")

# ======================================================
# CHARGEMENT DES DONNÉES ETF VIA CSV
# ======================================================
st.sidebar.header("📂 Données ETF")

uploaded_file = st.sidebar.file_uploader(
    "Upload un CSV example here : https://github.com/anthonyrottino/Streamlit",
    type=["csv"]
)

if uploaded_file is None:
    st.info(
        "Merci d'uploader un CSV contenant au minimum :\n"
        "- une colonne Date\n"
        "- une colonne par ETF (prix de clôture ou ajustés)\n\n"
        "Exemple de format :\n"
        "Date,SPY,TLT,GLD\n"
        "2020-01-02,321.5,137.0,146.2\n"
        "2020-01-03,320.0,138.1,145.7\n"
    )
    st.stop()

raw_df = pd.read_csv(uploaded_file)

# Détection automatique de la colonne Date
date_cols = [c for c in raw_df.columns if "date" in c.lower()]
date_col = date_cols[0] if len(date_cols) > 0 else raw_df.columns[0]

# -----------------------------
# Parsing dates robuste (EU dayfirst + fallback mixed)
# -----------------------------
raw_df[date_col] = raw_df[date_col].astype(str).str.strip()

dt = pd.to_datetime(raw_df[date_col], format="%d/%m/%Y", errors="coerce")

bad = dt.isna() & raw_df[date_col].ne("") & raw_df[date_col].ne("nan")
if bad.any():
    dt2 = pd.to_datetime(raw_df.loc[bad, date_col], format="mixed", dayfirst=True, errors="coerce")
    dt.loc[bad] = dt2

still_bad = dt.isna() & raw_df[date_col].ne("") & raw_df[date_col].ne("nan")
if still_bad.any():
    st.error("Dates illisibles. Exemples:\n" + "\n".join(raw_df.loc[still_bad, date_col].head(10).tolist()))
    st.stop()

raw_df[date_col] = dt
raw_df = raw_df.sort_values(date_col).set_index(date_col)
raw_df = raw_df[~raw_df.index.duplicated(keep="last")]

# Colonnes numériques = ETF
numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) < 2:
    st.error("Il faut au moins 2 colonnes d'ETF numériques dans le CSV.")
    st.stop()

st.sidebar.write("ETF détectés :", ", ".join(numeric_cols))

# Filtre de période (start / end)
min_date = raw_df.index.min()
max_date = raw_df.index.max()

st.sidebar.subheader("🗓 Période d'analyse")
start_date = st.sidebar.date_input(
    "Start date",
    value=min_date.date() if isinstance(min_date, pd.Timestamp) else min_date,
    min_value=min_date.date() if isinstance(min_date, pd.Timestamp) else min_date,
    max_value=max_date.date() if isinstance(max_date, pd.Timestamp) else max_date
)
end_date = st.sidebar.date_input(
    "End date",
    value=max_date.date() if isinstance(max_date, pd.Timestamp) else max_date,
    min_value=min_date.date() if isinstance(min_date, pd.Timestamp) else min_date,
    max_value=max_date.date() if isinstance(max_date, pd.Timestamp) else max_date
)

if start_date > end_date:
    st.error("La date de début est après la date de fin.")
    st.stop()

prices_full = raw_df[numeric_cols].dropna(how="all")
prices_full = prices_full.loc[
    (prices_full.index >= pd.to_datetime(start_date)) &
    (prices_full.index <= pd.to_datetime(end_date))
]

# -----------------------------
# Coverage + fill (évite dropna(any) trop agressif)
# -----------------------------
st.sidebar.subheader("🧼 Nettoyage")
min_coverage = st.sidebar.slider("Min coverage ETF (%)", 50, 100, 80) / 100.0
ok = prices_full.notna().mean(axis=1) >= min_coverage
prices_full = prices_full.loc[ok]
prices_full = prices_full.ffill().dropna()

if prices_full.shape[0] < 20:
    st.error("Pas assez d'observations dans la période sélectionnée (min 20 jours).")
    st.stop()

# Rendements journaliers en log
daily_ret = np.log(prices_full / prices_full.shift(1)).dropna()

# ======================================================
# PARAMÈTRES GLOBAUX
# ======================================================
st.sidebar.header("⚙️ Paramètres Globaux")

target_vol = st.sidebar.number_input(
    "Target Volatility (%)",
    min_value=1.0,
    max_value=50.0,
    value=10.0
) / 100.0

rf_rate = st.sidebar.number_input(
    "Taux sans risque annuel (%)",
    min_value=-5.0,
    max_value=10.0,
    value=4.0
) / 100.0

conf_level = st.sidebar.slider(
    "Niveau de confiance VaR/ES",
    min_value=0.90,
    max_value=0.99,
    value=0.95,
    step=0.01
)

horizon_days = st.sidebar.number_input(
    "Horizon (jours) pour VaR/ES",
    min_value=1,
    max_value=60,
    value=1,
    step=1
)

st.sidebar.markdown("---")
st.sidebar.subheader("📏 Contraintes par ETF")

min_weight = st.sidebar.number_input(
    "Poids minimum par ETF (%)",
    min_value=0.0,
    max_value=100.0,
    value=0.0,
    step=1.0
) / 100.0

max_weight = st.sidebar.number_input(
    "Poids maximum par ETF (%)",
    min_value=float(min_weight * 100.0),
    max_value=100.0,
    value=20.0,
    step=1.0
) / 100.0

# ======================================================
# STATS DE BASE (mean + shrinkage cov)
# ======================================================
tickers = daily_ret.columns.tolist()
n_assets = len(tickers)

mean_ret = daily_ret.mean() * 252.0

lw = LedoitWolf().fit(daily_ret.values)
cov_matrix = pd.DataFrame(
    lw.covariance_ * 252.0,
    index=daily_ret.columns,
    columns=daily_ret.columns
)

# Faisabilité des contraintes
if max_weight * n_assets < 1 - 1e-9:
    st.error(
        f"⚠️ Avec {n_assets} ETF et un poids max de {max_weight:.2%}, "
        f"la somme des poids max ({max_weight * n_assets:.2%}) est < 100%.\n"
        "Augmente le poids max ou sélectionne plus d'ETF."
    )
    st.stop()

if min_weight * n_assets > 1 + 1e-9:
    st.error(
        f"⚠️ Avec {n_assets} ETF et un poids min de {min_weight:.2%}, "
        f"la somme des poids min ({min_weight * n_assets:.2%}) est > 100%.\n"
        "Réduis le poids min ou sélectionne moins d'ETF."
    )
    st.stop()

# ======================================================
# 📌 FONCTIONS UTILITAIRES
# ======================================================

# -----------------------------
# Portfolio math
# -----------------------------
def portfolio_vol(w, cov):
    w = np.array(w, dtype=float)
    return float(np.sqrt(w.T @ cov @ w))

def portfolio_return(w, mu):
    w = np.array(w, dtype=float)
    return float(w.T @ mu)

def sharpe_ratio_calc(w, mu, cov, rf=0.0):
    vol = portfolio_vol(w, cov)
    ret = portfolio_return(w, mu)
    return (ret - rf) / vol if vol > 0 else -np.inf

# -----------------------------
# Performance
# -----------------------------
def annualized_volatility(r):
    r = pd.Series(r).dropna()
    return float(r.std() * np.sqrt(252.0))

def sharpe_ratio_series(r, rf=0.0):
    r = pd.Series(r).dropna()
    if r.std() == 0:
        return np.nan
    ann_ret = float(r.mean() * 252.0)
    ann_vol = float(r.std() * np.sqrt(252.0))
    return (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan

def max_drawdown(returns):
    r = pd.Series(returns).dropna()
    wealth = (1 + r).cumprod()
    roll_max = wealth.cummax()
    dd = wealth / roll_max - 1.0
    return float(dd.min())

# -----------------------------
# VaR & ES
# -----------------------------
def compute_var_es(returns, alpha=0.95, horizon_days=1):
    r = pd.Series(returns).dropna().values
    if len(r) == 0:
        return np.nan, np.nan
    q = np.quantile(r, 1 - alpha)
    var_1d = -q
    tail = r[r <= q]
    es_1d = -tail.mean() if len(tail) > 0 else np.nan
    scale = np.sqrt(horizon_days)
    return float(var_1d * scale), float(es_1d * scale)

# -----------------------------
# Gestion des contraintes
# -----------------------------
def _init_weights(n_assets, min_weight, max_weight):
    w0 = np.full(n_assets, 1.0 / n_assets)
    w0 = np.clip(w0, min_weight, max_weight)
    s = w0.sum()
    return (w0 / s) if s > 0 else np.full(n_assets, 1.0 / n_assets)

def _finalize_weights(w, n_assets, min_weight, max_weight):
    w = np.array(w, dtype=float)
    w = np.clip(w, min_weight, max_weight)
    s = w.sum()
    return (w / s) if s > 0 else np.full(n_assets, 1.0 / n_assets)

# -----------------------------
# Optimiseurs (bounded)
# -----------------------------
def min_variance(cov, n_assets, min_weight, max_weight):
    x0 = _init_weights(n_assets, min_weight, max_weight)
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = [(min_weight, max_weight) for _ in range(n_assets)]
    res = minimize(lambda w: portfolio_vol(w, cov), x0, bounds=bounds, constraints=cons, method="SLSQP")
    return _finalize_weights(res.x, n_assets, min_weight, max_weight)

def max_sharpe(mu, cov, rf, n_assets, min_weight, max_weight):
    x0 = _init_weights(n_assets, min_weight, max_weight)
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = [(min_weight, max_weight) for _ in range(n_assets)]
    res = minimize(lambda w: -sharpe_ratio_calc(w, mu, cov, rf), x0, bounds=bounds, constraints=cons, method="SLSQP")
    return _finalize_weights(res.x, n_assets, min_weight, max_weight)

def max_return(mu, n_assets, min_weight, max_weight):
    x0 = _init_weights(n_assets, min_weight, max_weight)
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = [(min_weight, max_weight) for _ in range(n_assets)]
    res = minimize(lambda w: -portfolio_return(w, mu), x0, bounds=bounds, constraints=cons, method="SLSQP")
    return _finalize_weights(res.x, n_assets, min_weight, max_weight)

def risk_parity(cov, n_assets, min_weight, max_weight):
    x0 = _init_weights(n_assets, min_weight, max_weight)
    bounds = [(min_weight, max_weight) for _ in range(n_assets)]
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

    def obj(w):
        w = np.array(w, dtype=float)
        port_var = float(w.T @ cov @ w)
        if port_var <= 0:
            return 1e6
        contrib_var = w * (cov @ w)
        risk_contrib = contrib_var / port_var
        return float(np.sum((risk_contrib - 1.0 / n_assets) ** 2))

    res = minimize(obj, x0, bounds=bounds, constraints=cons, method="SLSQP")
    return _finalize_weights(res.x, n_assets, min_weight, max_weight)

def max_diversification(cov, n_assets, min_weight, max_weight):
    sigma = np.sqrt(np.diag(cov))
    x0 = _init_weights(n_assets, min_weight, max_weight)
    bounds = [(min_weight, max_weight) for _ in range(n_assets)]
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

    def obj(w):
        w = np.array(w, dtype=float)
        port_vol = portfolio_vol(w, cov)
        numer = float(w @ sigma)
        if port_vol <= 0 or numer <= 0:
            return 1e6
        return -numer / port_vol

    res = minimize(obj, x0, bounds=bounds, constraints=cons, method="SLSQP")
    return _finalize_weights(res.x, n_assets, min_weight, max_weight)

def target_vol_portfolio(cov, target_vol, n_assets, min_weight, max_weight):
    w_base = min_variance(cov, n_assets, min_weight, max_weight)
    base_vol = portfolio_vol(w_base, cov)
    if base_vol <= 0:
        return w_base
    scaling = target_vol / base_vol
    w_scaled = w_base * scaling
    # On renormalise pour éviter un portefeuille "notional" étrange côté UI
    # (si tu veux du vrai leverage, commente la ligne suivante)
    if w_scaled.sum() != 0:
        w_scaled = w_scaled / w_scaled.sum()
    return w_scaled

# -----------------------------
# HRP (Hierarchical Risk Parity)
# -----------------------------
def _corr_to_dist(corr: pd.DataFrame) -> pd.DataFrame:
    return np.sqrt(0.5 * (1 - corr)).clip(lower=0.0)

def _get_quasi_diag(link) -> list:
    # leaves_list renvoie l’ordre des feuilles
    return leaves_list(link).tolist()

def _get_cluster_var(cov: pd.DataFrame, cluster_items: list) -> float:
    sub = cov.loc[cluster_items, cluster_items]
    w = np.ones(len(cluster_items)) / len(cluster_items)
    return float(w @ sub.values @ w)

def hrp_allocation(cov: pd.DataFrame, corr: pd.DataFrame) -> pd.Series:
    dist = _corr_to_dist(corr)
    dist_condensed = squareform(dist.values, checks=False)
    link = linkage(dist_condensed, method="single")

    sort_ix = _get_quasi_diag(link)
    ordered = corr.index[sort_ix].tolist()

    w = pd.Series(1.0, index=ordered)
    clusters = [ordered]

    while len(clusters) > 0:
        cluster = clusters.pop(0)
        if len(cluster) <= 1:
            continue

        split = len(cluster) // 2
        c1 = cluster[:split]
        c2 = cluster[split:]

        var1 = _get_cluster_var(cov, c1)
        var2 = _get_cluster_var(cov, c2)
        if (var1 + var2) <= 0:
            alpha = 0.5
        else:
            alpha = 1 - var1 / (var1 + var2)

        w[c1] *= alpha
        w[c2] *= (1 - alpha)

        clusters.append(c1)
        clusters.append(c2)

    w = w.reindex(corr.index).fillna(0.0)
    if w.sum() != 0:
        w = w / w.sum()
    return w

# -----------------------------
# Anti-switch (min holding) pour régimes
# -----------------------------
def apply_min_holding(reg_series: pd.Series, min_hold: int = 10) -> pd.Series:
    reg = reg_series.copy()
    if reg.empty or min_hold <= 0:
        return reg
    last = reg.iloc[0]
    last_change_i = 0
    for i in range(1, len(reg)):
        if reg.iloc[i] != last and (i - last_change_i) < min_hold:
            reg.iloc[i] = last
        elif reg.iloc[i] != last:
            last = reg.iloc[i]
            last_change_i = i
    return reg

# -----------------------------
# Backtest avec rebal + coûts
# -----------------------------
etf_costs_bps = {
    "VT": 5, "SPY": 2, "VGK": 4, "QQQ": 3,
    "LQD": 10, "HYG": 20, "IEF": 3, "DBC": 8, "GLD": 4
}

def compute_cost_per_etf(w_prev, w_new, tickers, etf_costs_bps, default_bps=10.0):
    if w_prev is None:
        return 0.0
    total_cost = 0.0
    for i, t in enumerate(tickers):
        delta = abs(w_new[i] - w_prev[i])
        cost_bps = etf_costs_bps.get(t, default_bps)
        total_cost += delta * (cost_bps / 10000.0)
    return float(total_cost)

def run_backtest_rebalanced(
    daily_ret: pd.DataFrame,
    strat_name: str,
    window_size: int,
    reb_freq: int,
    min_weight: float,
    max_weight: float,
    rf_rate: float,
    tickers: list,
    etf_costs_bps: dict,
    default_bps: float = 10.0
):
    ret = daily_ret.copy()
    dates = ret.index
    n_assets = ret.shape[1]

    port_ret = pd.Series(index=dates, dtype=float)
    turnover = pd.Series(0.0, index=dates, dtype=float)
    costs = pd.Series(0.0, index=dates, dtype=float)

    w_prev = None
    rebal_records = []

    for i in range(window_size, len(dates)):
        is_rebal = (i == window_size) or ((i - window_size) % reb_freq == 0)

        if is_rebal:
            hist = ret.iloc[i - window_size:i]
            mu = hist.mean() * 252.0
            cov = hist.cov() * 252.0

            if strat_name.startswith("Min Variance"):
                w = min_variance(cov.values, n_assets, min_weight, max_weight)
            elif strat_name.startswith("Max Sharpe"):
                w = max_sharpe(mu.values, cov.values, rf_rate, n_assets, min_weight, max_weight)
            elif strat_name.startswith("Max Return"):
                w = max_return(mu.values, n_assets, min_weight, max_weight)
            elif strat_name.startswith("Risk Parity"):
                w = risk_parity(cov.values, n_assets, min_weight, max_weight)
            elif strat_name.startswith("Max Diversification"):
                w = max_diversification(cov.values, n_assets, min_weight, max_weight)
            elif strat_name.startswith("HRP"):
                corr = hist.corr()
                w_ser = hrp_allocation(cov, corr)
                w = w_ser.values
            else:
                w = np.full(n_assets, 1.0 / n_assets)

            w_old = np.zeros(n_assets) if w_prev is None else w_prev
            to = 0.0 if w_prev is None else float(0.5 * np.sum(np.abs(w - w_prev)))

            cost_t = compute_cost_per_etf(w_prev, w, tickers, etf_costs_bps, default_bps=default_bps)

            date_i = dates[i]
            for j, t in enumerate(tickers):
                delta = float(w[j] - w_old[j])
                abs_delta = abs(delta)
                cost_bps = etf_costs_bps.get(t, default_bps)

                trade_cost_pct = abs_delta * (cost_bps / 10000.0) if w_prev is not None else 0.0
                turnover_contrib = 0.5 * abs_delta if w_prev is not None else 0.0

                rebal_records.append(
                    {
                        "Date": date_i,
                        "Strategy": strat_name,
                        "Ticker": t,
                        "w_prev": float(w_old[j]),
                        "w_new": float(w[j]),
                        "Delta_w": delta,
                        "AbsDelta": abs_delta,
                        "Turnover_contrib": turnover_contrib,
                        "Cost_bps": float(cost_bps),
                        "Trade_cost_%": float(trade_cost_pct),
                    }
                )

            w_prev = w.copy()
        else:
            to = 0.0
            cost_t = 0.0

        day_ret = ret.iloc[i].values
        gross = float(np.dot(w_prev, day_ret))
        net = gross - cost_t

        port_ret.iloc[i] = net
        turnover.iloc[i] = to
        costs.iloc[i] = cost_t

    port_ret = port_ret.iloc[window_size:]
    turnover = turnover.iloc[window_size:]
    costs = costs.iloc[window_size:]

    rebal_log = pd.DataFrame(rebal_records)
    if not rebal_log.empty:
        rebal_log = rebal_log.sort_values(["Date", "Ticker"])

    return port_ret, turnover, costs, rebal_log

# ======================================================
# CALCUL DES POIDS – STRATÉGIES
# ======================================================
weights_dict = {
    "Min Variance (bounded)": min_variance(cov_matrix.values, n_assets, min_weight, max_weight),
    "Max Sharpe (bounded)": max_sharpe(mean_ret.values, cov_matrix.values, rf_rate, n_assets, min_weight, max_weight),
    "Max Return (bounded)": max_return(mean_ret.values, n_assets, min_weight, max_weight),
    "Risk Parity (bounded)": risk_parity(cov_matrix.values, n_assets, min_weight, max_weight),
    "Max Diversification (bounded)": max_diversification(cov_matrix.values, n_assets, min_weight, max_weight),
    "HRP": hrp_allocation(cov_matrix, daily_ret.corr()).values,
    f"Target Volatility ({target_vol*100:.0f}%)": target_vol_portfolio(cov_matrix.values, target_vol, n_assets, min_weight, max_weight),
}

# ======================================================
# MAPPING ETF -> NOM + CLASSE
# ======================================================
etf_mapping = {
    "VT":  ("Vanguard Total World Stock",       "Equity"),
    "SPY": ("SPDR S&P 500",                     "Equity"),
    "VGK": ("Vanguard FTSE Europe",             "Equity"),
    "QQQ": ("Invesco QQQ Trust (Nasdaq-100)",   "Equity"),
    "LQD": ("iShares iBoxx $ IG Corporate",     "Credit"),
    "HYG": ("iShares iBoxx $ HY Corporate",     "Credit"),
    "IEF": ("iShares 7-10Y Treasury",           "Rates"),
    "DBC": ("Invesco DB Commodity Index",       "Commodity"),
    "GLD": ("SPDR Gold Shares",                 "Commodity"),
}

mapping_rows = []
for t in tickers:
    name, bucket = etf_mapping.get(t, (t, "Other"))
    mapping_rows.append({"Ticker": t, "Nom ETF": name, "Classe": bucket})

mapping_df = pd.DataFrame(mapping_rows)
st.markdown("### 🗺 Mapping ETF")
st.dataframe(mapping_df)

# ======================================================
# TABS
# ======================================================
tab_data, tab_opt, tab_risk, tab_stress, tab_backtest, tab_dynreg = st.tabs(
    ["📄 Data", "📊 Portefeuilles", "⚠️ Risk", "🚨 Stress Test", "🔁 Backtest & Costs", "🧭 Dyn Regime"]
)

# ===========================
# TAB DATA
# ===========================
with tab_data:
    st.subheader("📄 Data – Prix & Rendements")
    st.write("#### 1) Prix (100 derniers points)")
    st.dataframe(prices_full.tail(100))

    st.write("#### 2) Graphique des prix")
    fig_price = px.line(
        prices_full, x=prices_full.index, y=prices_full.columns,
        labels={"index": "Date", "value": "Prix"},
        title="Prix des ETF"
    )
    st.plotly_chart(fig_price, use_container_width=True)

    st.write("#### 3) Rendements journaliers")
    fig_ret = px.line(
        daily_ret, x=daily_ret.index, y=daily_ret.columns,
        labels={"index": "Date", "value": "Return"},
        title="Rendements journaliers (log)"
    )
    st.plotly_chart(fig_ret, use_container_width=True)

    st.write("#### 4) Rendements cumulés")
    cum_ret = np.exp(daily_ret.cumsum()) - 1
    fig_cum = px.line(
        cum_ret, x=cum_ret.index, y=cum_ret.columns,
        labels={"index": "Date", "value": "Cumulative Return"},
        title="Rendements cumulés par ETF"
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    st.write("#### 5) Heatmap de corrélation (rendements)")
    corr_mat = daily_ret.corr()
    fig_corr = px.imshow(
        corr_mat, x=corr_mat.columns, y=corr_mat.index,
        labels={"x": "ETF", "y": "ETF", "color": "Corrélation"},
        title="Corrélation entre ETF"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ===========================
# TAB OPT
# ===========================
with tab_opt:
    st.subheader("📊 Optimisation de portefeuilles")

    st.write("### 🧮 Poids par stratégie")
    weights_df = pd.DataFrame(weights_dict, index=tickers)
    st.dataframe(weights_df.style.format("{:.2%}"))

    strategies = list(weights_dict.keys())
    selected_strats = st.multiselect("Stratégies à afficher", options=strategies, default=strategies)

    cumulative_dict = {}
    daily_port_ret = {}
    for name, w in weights_dict.items():
        r = daily_ret.dot(np.array(w))
        daily_port_ret[name] = r
        cumulative_dict[name] = np.exp(r.cumsum()) - 1

    cumulative_df = pd.DataFrame(cumulative_dict, index=daily_ret.index)
    daily_port_df = pd.DataFrame(daily_port_ret, index=daily_ret.index)

    if selected_strats:
        st.write("### 📈 Performance cumulée des portefeuilles")
        fig_cum_pf = px.line(
            cumulative_df[selected_strats],
            x=cumulative_df.index, y=selected_strats,
            labels={"index": "Date", "value": "Cumulative Return"},
            title="Cumulative Return par stratégie"
        )
        st.plotly_chart(fig_cum_pf, use_container_width=True)

        st.write("### 🔎 Performance quotidienne des portefeuilles")
        fig_daily_pf = px.line(
            daily_port_df[selected_strats],
            x=daily_port_df.index, y=selected_strats,
            labels={"index": "Date", "value": "Daily Return"},
            title="Rendements journaliers par stratégie"
        )
        st.plotly_chart(fig_daily_pf, use_container_width=True)

    st.write("### 📌 Metrics par stratégie")
    metrics_port = pd.DataFrame(columns=weights_dict.keys())
    for name, w in weights_dict.items():
        r = daily_ret.dot(np.array(w))
        metrics_port[name] = [
            f"{annualized_volatility(r):.2%}",
            f"{sharpe_ratio_series(r, rf=rf_rate):.2f}",
            f"{max_drawdown(r):.2%}",
        ]
    metrics_port.index = ["Vol Annualisée", "Sharpe Ratio", "Max Drawdown"]
    st.dataframe(metrics_port)

# ===========================
# TAB RISK
# ===========================
with tab_risk:
    st.subheader(f"⚠️ VaR & Expected Shortfall (niveau {conf_level:.0%}, horizon {horizon_days} j)")

    risk_table = pd.DataFrame(index=["VaR (perte max)", "Expected Shortfall"], columns=weights_dict.keys())
    for name, w in weights_dict.items():
        r = daily_ret.dot(np.array(w))
        var_val, es_val = compute_var_es(r, alpha=conf_level, horizon_days=horizon_days)
        risk_table.loc["VaR (perte max)", name] = f"{var_val:.2%}"
        risk_table.loc["Expected Shortfall", name] = f"{es_val:.2%}"
    st.dataframe(risk_table)

    st.subheader("🎯 Choisir un portefeuille pour l'analyse factorielle")
    pf_choice_risk = st.selectbox("Portefeuille", list(weights_dict.keys()))
    w_sel = np.array(weights_dict[pf_choice_risk], dtype=float)
    r_pf = daily_ret.dot(w_sel)
    r_pf.name = "Portfolio"

    st.subheader(f"🧩 PCA – Factor Risk ({pf_choice_risk})")
    X = daily_ret.dropna()
    k = st.number_input("Nombre de facteurs PCA", 1, min(10, n_assets), 3)

    pca = PCA(n_components=k)
    pca.fit(X)

    factors_names = [f"Factor {i+1}" for i in range(k)]
    loadings = pd.DataFrame(pca.components_.T, index=tickers, columns=factors_names)

    betas_pf = pca.components_ @ w_sel
    eigenvalues = pca.explained_variance_[:k]
    explained_pf = (betas_pf**2 * eigenvalues)
    explained_pf = explained_pf / explained_pf.sum() if explained_pf.sum() != 0 else explained_pf

    explained_df = pd.DataFrame({"Explained Variance (%)": explained_pf}, index=factors_names)
    st.write("### 🎯 Variance expliquée PAR portefeuille")
    st.dataframe(explained_df.style.format("{:.2%}"))

    fig_heat = px.imshow(loadings, labels={"x": "Factor", "y": "Asset"}, title="Heatmap PCA Loadings")
    st.plotly_chart(fig_heat, use_container_width=True)

    st.write("### 💸 Attribution du PnL par facteur PCA")
    F = pd.DataFrame(pca.transform(X), index=X.index, columns=factors_names)

    pf_choice_attr = st.selectbox("Portefeuille pour l'attribution PCA", list(weights_dict.keys()), index=0)
    w_pf = np.array(weights_dict[pf_choice_attr], dtype=float)
    r_pf_attr = X.dot(w_pf)

    model_pca = LinearRegression(fit_intercept=True)
    model_pca.fit(F.values, r_pf_attr.values)

    betas_pca = model_pca.coef_
    intercept_pca = model_pca.intercept_

    contrib_factors_pca = pd.DataFrame(index=F.index, columns=factors_names, dtype=float)
    for j, f_name in enumerate(factors_names):
        contrib_factors_pca[f_name] = betas_pca[j] * F[f_name]

    fitted_pca = contrib_factors_pca.sum(axis=1) + intercept_pca
    resid_pca = r_pf_attr - fitted_pca

    level_vals = []
    names_all = list(factors_names) + ["Residual"]

    for f_name in factors_names:
        level_vals.append(float(contrib_factors_pca[f_name].mean() * 252.0))
    level_vals.append(float(resid_pca.mean() * 252.0))

    total_level = float(np.sum(level_vals))
    pct_vals = [lv / total_level for lv in level_vals] if total_level != 0 else [0.0 for _ in level_vals]

    attribution_pca_df = pd.DataFrame(
        {"Level (ann. return)": level_vals + [total_level], "Percentage": pct_vals + [1.0]},
        index=names_all + ["TOTAL"]
    )

    st.write(f"Attribution PCA pour le portefeuille : **{pf_choice_attr}**")
    st.dataframe(attribution_pca_df.style.format({"Level (ann. return)": "{:.2%}", "Percentage": "{:.2%}"}))

    # Style factors
    st.subheader(f"🧬 Style / Macro Factors – {pf_choice_risk}")

    style_factors = {}
    cols = daily_ret.columns

    if {"VT", "SPY", "VGK", "QQQ"}.issubset(cols):
        style_factors["MKT_Equity"] = daily_ret[["VT", "SPY", "VGK", "QQQ"]].mean(axis=1)
        style_factors["HML_proxy"] = (daily_ret["SPY"] + daily_ret["VGK"]) / 2 - daily_ret["QQQ"]
        style_factors["MOM_proxy"] = daily_ret["QQQ"] - daily_ret["SPY"]

    if {"LQD", "HYG"}.issubset(cols):
        style_factors["Credit_Spread"] = daily_ret["HYG"] - daily_ret["LQD"]

    if "IEF" in cols:
        style_factors["Rates_Duration"] = daily_ret["IEF"]

    if "DBC" in cols:
        style_factors["Commodities"] = daily_ret["DBC"]
    if "GLD" in cols:
        style_factors["Gold"] = daily_ret["GLD"]

    if not style_factors:
        st.info("Pas assez d'ETF pour construire des facteurs de style.")
        style_df = None
    else:
        style_df = pd.DataFrame(style_factors)

        df_merged = pd.concat([style_df, r_pf], axis=1, join="inner").dropna()
        style_df_aligned = df_merged[style_df.columns]
        r_pf_aligned = df_merged["Portfolio"]

        corr_style = style_df_aligned.apply(lambda x: x.corr(r_pf_aligned))
        corr_style_df = corr_style.to_frame("Corrélation")

        st.write("### Corrélations Style ↔ Portefeuille")
        st.dataframe(corr_style_df.style.format("{:.2f}"))

        fig_corr_st = px.bar(corr_style_df, x=corr_style_df.index, y="Corrélation", title="Corrélation Style ↔ Portefeuille")
        st.plotly_chart(fig_corr_st, use_container_width=True)

        X_style = style_df_aligned.values
        y_style = r_pf_aligned.values.reshape(-1, 1)
        betas_style, *_ = np.linalg.lstsq(X_style, y_style, rcond=None)
        betas_style = betas_style.flatten()

        contrib_style = {f: style_df_aligned[f] * betas_style[j] for j, f in enumerate(style_df_aligned.columns)}
        fitted_st = sum(contrib_style.values())
        resid_st = r_pf_aligned - fitted_st

        rows_st = list(style_df_aligned.columns) + ["Residual"]
        level_vals_st = [float(s.sum()) for s in contrib_style.values()] + [float(resid_st.sum())]

        level_vals_st_100 = np.array(level_vals_st) * 100.0
        total_st = float(level_vals_st_100.sum())
        level_series_st = pd.Series(level_vals_st_100, index=rows_st)

        if abs(total_st) > 1e-12:
            pct_series_st = level_series_st / total_st
        else:
            pct_series_st = pd.Series(np.nan, index=rows_st)

        level_series_st.loc["TOTAL"] = float(level_series_st.sum())
        pct_series_st.loc["TOTAL"] = 1.0

        attribution_style_df = pd.DataFrame({"Level": level_series_st, "Percentage": pct_series_st})
        st.write("### 📌 Style Attribution – Level×100 + Percentage + TOTAL")
        st.dataframe(attribution_style_df.style.format({"Level": "{:.4f}", "Percentage": "{:.2%}"}))

    # Mapping PCA ↔ Style
    st.subheader("🔗 Mapping PCA ↔ Style Factors")
    if style_factors and style_df is not None:
        style_map = style_df.loc[F.index].dropna()
        F_aligned = F.loc[style_map.index]

        map_corr = pd.DataFrame(index=factors_names, columns=style_map.columns, dtype=float)
        for p in factors_names:
            for s in style_map.columns:
                map_corr.loc[p, s] = float(F_aligned[p].corr(style_map[s]))

        st.write("### Matrice de corrélation PCA ↔ Style")
        st.dataframe(map_corr.style.format("{:.2f}"))

        best_map = {}
        for p in factors_names:
            best_style = map_corr.loc[p].abs().idxmax()
            corr_val = map_corr.loc[p, best_style]
            best_map[p] = (best_style, corr_val)

        best_map_df = pd.DataFrame.from_dict(best_map, orient="index", columns=["Style", "Corrélation"])
        st.write("### 🎯 Style dominant par facteur PCA")
        st.dataframe(best_map_df.style.format({"Corrélation": "{:.2f}"}))

        fig_map = px.bar(best_map_df, x=best_map_df.index, y="Corrélation", color="Style", text="Style", title="Mapping PCA → Style")
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("Pas assez de facteurs Style pour le mapping PCA ↔ Style.")

# ===========================
# TAB STRESS
# ===========================
with tab_stress:
    st.subheader("🚨 Stress Test – Shocks directement sur les ETF")

    etf_list = list(daily_ret.columns)

    if "etf_shocks" not in st.session_state:
        st.session_state["etf_shocks"] = {t: 0.0 for t in etf_list}
    if "stress_scenarios" not in st.session_state:
        st.session_state["stress_scenarios"] = {}

    pf_choice_stress = st.selectbox("Portefeuille à tester", list(weights_dict.keys()), key="pf_choice_stress_etf")
    w_stress = np.array(weights_dict[pf_choice_stress], dtype=float)

    st.write(f"Portefeuille sélectionné : **{pf_choice_stress}**")

    preset = st.selectbox(
        "Scénario preset",
        [
            "Custom",
            "Equity Crash (-20%)",
            "Tech Crash (QQQ -30%)",
            "Credit Stress (HY -15%, IG -7%)",
            "Rates +100bps proxy (IEF -5%)",
            "Gold Rally (+10%)",
            "Commodities Crash (DBC -15%)",
        ],
        key="preset_stress_etf"
    )

    base_shocks = {t: 0.0 for t in etf_list}
    equities = {"VT", "SPY", "VGK", "QQQ"}
    tech = {"QQQ"}
    credit_hy = {"HYG"}
    credit_ig = {"LQD"}
    rates = {"IEF"}
    commodities = {"DBC"}
    gold = {"GLD"}

    if preset == "Equity Crash (-20%)":
        for t in etf_list:
            if t in equities:
                base_shocks[t] = -20.0
    elif preset == "Tech Crash (QQQ -30%)":
        for t in etf_list:
            if t in tech:
                base_shocks[t] = -30.0
            elif t in equities:
                base_shocks[t] = -15.0
    elif preset == "Credit Stress (HY -15%, IG -7%)":
        for t in etf_list:
            if t in credit_hy:
                base_shocks[t] = -15.0
            elif t in credit_ig:
                base_shocks[t] = -7.0
    elif preset == "Rates +100bps proxy (IEF -5%)":
        for t in etf_list:
            if t in rates:
                base_shocks[t] = -5.0
    elif preset == "Gold Rally (+10%)":
        for t in etf_list:
            if t in gold:
                base_shocks[t] = 10.0
    elif preset == "Commodities Crash (DBC -15%)":
        for t in etf_list:
            if t in commodities:
                base_shocks[t] = -15.0

    col_reset, _ = st.columns([1, 3])
    with col_reset:
        if st.button("🔄 Reset tous les shocks à 0"):
            st.session_state["etf_shocks"] = {t: 0.0 for t in etf_list}
            st.experimental_rerun()

    st.write("Ajuste les shocks par ETF si besoin :")
    shocks_user = {}
    for t in etf_list:
        default_val = st.session_state["etf_shocks"].get(t, base_shocks[t])
        shocks_user[t] = st.slider(
            f"Shock {t} (%)", -50.0, 50.0, float(default_val), 1.0, key=f"shock_{t}_etf"
        )
    st.session_state["etf_shocks"] = shocks_user.copy()

    shock_vec = np.array([shocks_user[t] / 100.0 for t in etf_list], dtype=float)
    pnl_stress = float(np.dot(w_stress, shock_vec))

    st.metric(label=f"PnL de stress estimé sur {pf_choice_stress}", value=f"{pnl_stress * 100:.2f} %")

    contrib_level = w_stress * shock_vec * 100.0
    contrib_series = pd.Series(contrib_level, index=etf_list)

    total_level = float(contrib_series.sum())
    pct_series = contrib_series / total_level if abs(total_level) > 1e-12 else pd.Series(np.nan, index=etf_list)

    contrib_series.loc["TOTAL"] = float(contrib_series.sum())
    pct_series.loc["TOTAL"] = 1.0

    w_series = pd.Series(w_stress, index=etf_list)
    w_series.loc["TOTAL"] = float(w_series.sum())

    shock_series = pd.Series([shocks_user[t] for t in etf_list] + [np.nan], index=list(etf_list) + ["TOTAL"])

    stress_etf_df = pd.DataFrame(
        {"Weight": w_series, "Shock (%)": shock_series, "Level (×100)": contrib_series, "Percentage": pct_series}
    )

    st.write("### 📊 Tableau de stress par ETF")
    st.dataframe(
        stress_etf_df.style.format(
            {"Weight": "{:.2%}", "Shock (%)": "{:.1f}", "Level (×100)": "{:.4f}", "Percentage": "{:.2%}"}
        )
    )

    st.write("### 📈 Contribution au PnL de stress par ETF")
    contrib_plot = contrib_series.drop("TOTAL", errors="ignore")
    fig_stress = px.bar(contrib_plot.to_frame("Level (×100)"), x=contrib_plot.index, y="Level (×100)",
                        title=f"Contribution au PnL de stress – {pf_choice_stress}")
    st.plotly_chart(fig_stress, use_container_width=True)

    st.write("### 💾 Sauvegarde & comparaison de scénarios")
    scenario_name = st.text_input("Nom du scénario", value=f"{preset} ({pf_choice_stress})")

    if st.button("💾 Sauvegarder ce scénario courant"):
        if scenario_name.strip():
            st.session_state["stress_scenarios"][scenario_name] = {
                "pnl": pnl_stress, "shocks": shocks_user.copy(), "contrib": contrib_series.to_dict()
            }
            st.success(f"Scénario '{scenario_name}' sauvegardé.")

    scenarios = st.session_state["stress_scenarios"]
    if scenarios:
        summary_df = pd.DataFrame({name: {"PnL (%)": info["pnl"] * 100.0} for name, info in scenarios.items()}).T
        st.dataframe(summary_df.style.format({"PnL (%)": "{:.2f}"}))

        selected_scen = st.multiselect("Scénarios à comparer (contribution par ETF)",
                                       list(scenarios.keys()), default=list(scenarios.keys()))
        if selected_scen:
            contrib_all = {}
            for name in selected_scen:
                d = scenarios[name]["contrib"].copy()
                d.pop("TOTAL", None)
                contrib_all[name] = d

            contrib_df = pd.DataFrame(contrib_all)
            contrib_long = contrib_df.reset_index().melt(id_vars="index", var_name="Scenario", value_name="Level (×100)") \
                                     .rename(columns={"index": "ETF"})
            fig_multi = px.bar(contrib_long, x="ETF", y="Level (×100)", color="Scenario", barmode="group",
                               title="Contributions au PnL de stress – Multi-scénarios")
            st.plotly_chart(fig_multi, use_container_width=True)
    else:
        st.info("Aucun scénario sauvegardé pour le moment.")

# ===========================
# TAB BACKTEST
# ===========================
with tab_backtest:
    st.subheader("🔁 Backtest avec rebalancing, coûts par ETF & Buy & Hold benchmark")

    max_window = len(daily_ret)
    if max_window < 60:
        st.warning("Pas assez d'historique pour faire un backtest (min 60 jours).")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        window_size = st.number_input("Fenêtre lookback (jours)", 60, max_window // 2, min(252, max_window // 2), 5)
    with col2:
        freq_label = st.selectbox("Fréquence de rebalancing", ["Mensuel (~21j)", "Trimestriel (~63j)", "Annuel (~252j)"])
        reb_freq = 21 if "Mensuel" in freq_label else (63 if "Trimestriel" in freq_label else 252)
    with col3:
        default_bps = st.number_input("Coût par défaut (bps) pour ETF non mappés", 0.0, 100.0, 10.0, 1.0)

    st.markdown(
        "- **Coût par ETF** : |Δw_i| × coût_i_bps / 10 000  \n"
        "- **Turnover_t** (info) = ½ Σ |w_t − w_{t-1}|  \n"
        "- Retour **net** = Retour **brut** − Σ coûts par ETF"
    )

    st.write("### 🧾 Tableau des coûts par ETF (bps)")
    cost_table = pd.DataFrame({"Ticker": list(etf_costs_bps.keys()), "Coût (bps)": list(etf_costs_bps.values())})
    st.dataframe(cost_table)

    st.write("### 🎯 Stratégies à backtester")
    available_strats = [s for s in weights_dict.keys() if any(tag in s for tag in
                      ["Min Variance", "Max Sharpe", "Max Return", "Risk Parity", "Max Diversification", "HRP"])]
    if not available_strats:
        st.info("Aucune stratégie classique trouvée dans weights_dict.")
        st.stop()

    selected_strats_bt = st.multiselect("Sélectionne les stratégies", options=available_strats, default=available_strats)
    if not selected_strats_bt:
        st.info("Sélectionne au moins une stratégie pour lancer le backtest.")
        st.stop()

    bt_returns_net, bt_turnover, bt_costs, bt_logs = {}, {}, {}, {}
    for strat in selected_strats_bt:
        r_bt, to_bt, c_bt, log_bt = run_backtest_rebalanced(
            daily_ret, strat,
            window_size=window_size, reb_freq=reb_freq,
            min_weight=min_weight, max_weight=max_weight,
            rf_rate=rf_rate, tickers=tickers,
            etf_costs_bps=etf_costs_bps, default_bps=default_bps
        )
        bt_returns_net[strat] = r_bt
        bt_turnover[strat] = to_bt
        bt_costs[strat] = c_bt
        bt_logs[strat] = log_bt

    common_index = bt_returns_net[selected_strats_bt[0]].index
    for s in selected_strats_bt[1:]:
        common_index = common_index.intersection(bt_returns_net[s].index)

    bt_returns_net = {k: v.loc[common_index] for k, v in bt_returns_net.items()}
    bt_turnover = {k: v.loc[common_index] for k, v in bt_turnover.items()}
    bt_costs = {k: v.loc[common_index] for k, v in bt_costs.items()}

    # Benchmark Buy & Hold (EW)
    w_bh = np.full(daily_ret.shape[1], 1.0 / daily_ret.shape[1])
    r_bh = daily_ret.dot(w_bh).loc[common_index]

    bt_returns_net["Buy & Hold (EW)"] = r_bh
    bt_turnover["Buy & Hold (EW)"] = pd.Series(0.0, index=common_index)
    bt_costs["Buy & Hold (EW)"] = pd.Series(0.0, index=common_index)
    bt_logs["Buy & Hold (EW)"] = pd.DataFrame(columns=[
        "Date", "Strategy", "Ticker", "w_prev", "w_new", "Delta_w", "AbsDelta",
        "Turnover_contrib", "Cost_bps", "Trade_cost_%"
    ])

    bt_ret_df = pd.DataFrame(bt_returns_net, index=common_index)
    bt_cum_df = np.exp(bt_ret_df.cumsum()) - 1

    st.write("### 📈 Performance cumulée (net des coûts)")
    fig_bt_cum = px.line(bt_cum_df, x=bt_cum_df.index, y=bt_cum_df.columns,
                         labels={"index": "Date", "value": "Cumulative Return"},
                         title="Cumulative Return – Backtest rebalancé (net de coûts) + Buy & Hold")
    st.plotly_chart(fig_bt_cum, use_container_width=True)

    st.write("### 📉 Rendements journaliers (net des coûts)")
    fig_bt_daily = px.line(bt_ret_df, x=bt_ret_df.index, y=bt_ret_df.columns,
                           labels={"index": "Date", "value": "Daily Return"},
                           title="Daily Returns – Backtest rebalancé (net de coûts) + Buy & Hold")
    st.plotly_chart(fig_bt_daily, use_container_width=True)

    st.write("### 📌 Metrics : brut vs net + Turnover + Coûts par ETF")
    strategies_metrics = selected_strats_bt + ["Buy & Hold (EW)"]
    metrics_cols = [
        "Ann. Return (brut)", "Ann. Return (net)", "Ann. Vol (net)", "Sharpe (net)",
        "Max DD (net)", "Turnover moyen", "Turnover annuel", "Coût total (%)"
    ]
    metrics_bt = pd.DataFrame(index=strategies_metrics, columns=metrics_cols)

    for strat in strategies_metrics:
        r_net = bt_returns_net[strat]
        c = bt_costs[strat]
        to = bt_turnover[strat]
        r_gross = r_net + c

        ann_ret_net = float(r_net.mean() * 252.0)
        ann_ret_gross = float(r_gross.mean() * 252.0)
        ann_vol_net = float(r_net.std() * np.sqrt(252.0))
        sharpe_net = (ann_ret_net - rf_rate) / ann_vol_net if ann_vol_net > 0 else np.nan
        mdd_net = max_drawdown(r_net)

        avg_turnover = float(to.mean())
        ann_turnover = float(avg_turnover * 252.0)
        total_cost = float(c.sum())

        metrics_bt.loc[strat] = [
            f"{ann_ret_gross:.2%}",
            f"{ann_ret_net:.2%}",
            f"{ann_vol_net:.2%}",
            f"{sharpe_net:.2f}",
            f"{mdd_net:.2%}",
            f"{avg_turnover:.2%}",
            f"{ann_turnover:.2%}",
            f"{total_cost:.2%}",
        ]

    st.dataframe(metrics_bt)

    st.write("### 📋 Détail des opérations à chaque rebalancement")
    strat_log_choice = st.selectbox("Portefeuille pour afficher le log des opérations", strategies_metrics)
    log_df = bt_logs.get(strat_log_choice, pd.DataFrame())

    if log_df is None or log_df.empty:
        st.info("Aucun rebalancement enregistré pour cette stratégie (ou Buy & Hold).")
    else:
        log_display = log_df.copy()
        for col in ["w_prev", "w_new", "Delta_w", "AbsDelta", "Turnover_contrib", "Trade_cost_%"]:
            log_display[col] = log_display[col].astype(float)

        st.dataframe(
            log_display.style.format({
                "w_prev": "{:.2%}", "w_new": "{:.2%}",
                "Delta_w": "{:.2%}", "AbsDelta": "{:.2%}",
                "Turnover_contrib": "{:.2%}",
                "Cost_bps": "{:.1f}",
                "Trade_cost_%": "{:.4%}",
            })
        )

# ===========================
# TAB Dyn Regime
# ===========================
# ===========================
# TAB Dyn Regime : HMM 3 régimes
# ===========================
# ===========================
# TAB Dyn Regime (COMPACT) : HMM 3 régimes + Perf + Trade log
# ===========================
# ===========================
# TAB Dyn Regime : HMM 3 REGIMES + TABLES + TRADE LOG
# ===========================
# ===========================
# TAB Dyn Regime (FINAL)
# ===========================
# ===========================
# TAB Dyn Regime (FULL FINAL)
# ===========================
# ===========================
# TAB Dyn Regime (FULL + HMM + ONE DYN TABLE + PERF + REGIME PLOT + TRADE LOG)
# ===========================
with tab_dynreg:

    from hmmlearn.hmm import GaussianHMM
    import plotly.graph_objects as go

    # ======================================================
    # 📘 INTRO
    # ======================================================
    st.title("🧭 Dynamic Allocation by Regime (HMM)")

    st.markdown("""
    **Dynamic Allocation Framework**

    This module implements a **regime-aware dynamic allocation strategy**.
    Each date, the portfolio selects **one allocation profile** conditional on the
    detected regime (**Risk-Off / Neutral / Risk-On**).

    - Regimes are inferred via a **3-state HMM** on an equity proxy (return + rolling vol)
    - Exactly **one allocation is active per date**
    - Weights change **only when regime changes** (unless you choose daily re-optimization)

    > Only the **Hybrid (USED)** allocation is traded.
    """)

    # ======================================================
    # 1) DATA & EQUITY PROXY
    # ======================================================
    X_ret = daily_ret.dropna()
    tickers = X_ret.columns.tolist()

    equity_candidates = ["VT", "SPY", "QQQ", "VGK"]
    eq_cols = [c for c in equity_candidates if c in X_ret.columns]
    eq_proxy = X_ret[eq_cols].mean(axis=1) if eq_cols else X_ret.mean(axis=1)

    st.caption(f"Equity proxy used: {', '.join(eq_cols) if eq_cols else 'Mean of all ETFs'}")

    # ======================================================
    # 2) HMM (3 STATES) + CONFIDENCE FILTER
    # ======================================================
    st.subheader("🧠 Regime detection (3-state HMM)")

    vol_21 = eq_proxy.rolling(21).std()
    feat = pd.DataFrame({"ret": eq_proxy, "vol": vol_21}).dropna()

    feat_std = (feat - feat.mean()) / feat.std(ddof=0)
    feat_std = feat_std.replace([np.inf, -np.inf], np.nan).dropna()

    hmm = GaussianHMM(
        n_components=3,
        covariance_type="diag",
        n_iter=500,
        random_state=42
    )
    hmm.fit(feat_std.values)

    states = pd.Series(hmm.predict(feat_std.values), index=feat_std.index, name="state")
    probs = pd.DataFrame(
        hmm.predict_proba(feat_std.values),
        index=feat_std.index,
        columns=[f"p_state{i}" for i in range(3)]
    )

    # --- map states -> regimes
    stats = []
    for s in range(3):
        idx = states.index[states == s]
        stats.append({
            "State": s,
            "MeanReturn": feat.loc[idx, "ret"].mean(),
            "MeanVol": feat.loc[idx, "vol"].mean(),
            "Obs": len(idx)
        })
    stats_df = pd.DataFrame(stats).set_index("State")
    st.dataframe(stats_df.style.format({"MeanReturn": "{:.4%}", "MeanVol": "{:.4%}"}), use_container_width=True)

    risk_on_state = int(stats_df["MeanReturn"].idxmax())
    risk_off_state = int(stats_df["MeanVol"].idxmax())
    neutral_state = int([s for s in stats_df.index.tolist() if s not in [risk_on_state, risk_off_state]][0])

    state_to_regime = {
        risk_off_state: "Risk-Off",
        neutral_state: "Neutral",
        risk_on_state: "Risk-On"
    }

    p_thresh = st.slider("Min regime probability (stability filter)", 0.33, 0.90, 0.55, 0.01)

    pmax = probs.max(axis=1)
    regime_used_str = states.map(state_to_regime)
    regime_used_str[pmax < p_thresh] = np.nan
    regime_used_str = regime_used_str.ffill().dropna()

    # Numeric regimes for downstream
    regime_rev_map = {"Risk-Off": 0, "Neutral": 1, "Risk-On": 2}
    regime_used = regime_used_str.map(regime_rev_map).astype(int)
    regime_map = {0: "Risk-Off", 1: "Neutral", 2: "Risk-On"}

    st.write("Regime counts:")
    st.dataframe(regime_used_str.value_counts().rename("Obs"), use_container_width=True)

    # ======================================================
    # 3) BUILD CLASS PROFILES (Prior / Optimized / Hybrid)
    # ======================================================
    st.subheader("🧩 Allocation profiles (Prior / Optimized / Hybrid)")

    asset_classes = ["Equity", "Credit", "Rates", "Commodity"]
    class_by_ticker = mapping_df.set_index("Ticker")["Classe"].to_dict()

    # --- Prior (macro)
    profile_class_prior = {
        "Risk-Off": {"Equity": 0.05, "Credit": 0.25, "Rates": 0.50, "Commodity": 0.20},
        "Neutral":  {"Equity": 0.40, "Credit": 0.30, "Rates": 0.20, "Commodity": 0.10},
        "Risk-On":  {"Equity": 0.70, "Credit": 0.20, "Rates": 0.05, "Commodity": 0.05},
    }

    # --- Class returns
    def build_class_returns(X):
        out = {}
        for ac in asset_classes:
            cols = [t for t in X.columns if class_by_ticker.get(t) == ac]
            if cols:
                out[ac] = X[cols].mean(axis=1)
        return pd.DataFrame(out)

    class_ret = build_class_returns(X_ret)

    # --- Optimized by regime: positive Sharpe weights
    def optimized_profile(reg_label):
        idx = regime_used_str.index[regime_used_str == reg_label]
        sub = class_ret.loc[idx].dropna()
        if sub.empty:
            return profile_class_prior[reg_label]

        sharpe = {}
        for ac in asset_classes:
            if ac not in sub.columns:
                sharpe[ac] = 0.0
                continue
            r = sub[ac].dropna()
            if len(r) < 20 or r.std() == 0:
                sharpe[ac] = 0.0
            else:
                sharpe[ac] = max(0.0, (r.mean() * 252) / (r.std() * np.sqrt(252)))
        s = sum(sharpe.values())
        return {ac: (sharpe.get(ac, 0.0) / s if s > 0 else profile_class_prior[reg_label].get(ac, 0.0))
                for ac in asset_classes}

    profile_class_opt = {r: optimized_profile(r) for r in profile_class_prior.keys()}

    lambda_mix = st.slider("λ (optimization weight)", 0.0, 1.0, 0.30, 0.05)

    profile_class_hybrid = {}
    for r in profile_class_prior.keys():
        mix = {}
        for ac in asset_classes:
            wp = profile_class_prior[r].get(ac, 0.0)
            wo = profile_class_opt[r].get(ac, 0.0)
            mix[ac] = max(0.0, (1 - lambda_mix) * wp + lambda_mix * wo)
        s = sum(mix.values())
        profile_class_hybrid[r] = {ac: (mix[ac] / s if s > 0 else 1.0 / len(asset_classes)) for ac in asset_classes}

    # ======================================================
    # 4) ONE ALLOCATION TABLE (Prior / Opt / Hybrid in ONE table)
    # ======================================================
    st.subheader("📋 Regime Allocation Profiles (Single Table)")

    rows = []
    for r in ["Risk-Off", "Neutral", "Risk-On"]:
        for ac in asset_classes:
            wp = float(profile_class_prior.get(r, {}).get(ac, 0.0))
            wo = float(profile_class_opt.get(r, {}).get(ac, 0.0))
            wh = float(profile_class_hybrid.get(r, {}).get(ac, 0.0))
            rows.append({
                "Regime": r,
                "Asset Class": ac,
                "Prior": wp,
                "Optimized": wo,
                "Hybrid (USED)": wh,
                "Δ Hybrid − Prior": wh - wp,
                "Δ Hybrid − Optimized": wh - wo
            })
    alloc_table = pd.DataFrame(rows)

    st.dataframe(
        alloc_table.style.format({
            "Prior": "{:.0%}",
            "Optimized": "{:.0%}",
            "Hybrid (USED)": "{:.0%}",
            "Δ Hybrid − Prior": "{:+.0%}",
            "Δ Hybrid − Optimized": "{:+.0%}",
        }),
        use_container_width=True
    )

    # ======================================================
    # 5) BUILD DYNAMIC ETF WEIGHTS (from Hybrid class alloc)
    # ======================================================
    def build_etf_weights_from_class_alloc(class_alloc, tickers_list, class_by_ticker_dict):
        w = np.zeros(len(tickers_list), dtype=float)
        for ac, w_ac in class_alloc.items():
            idx = [i for i, t in enumerate(tickers_list) if class_by_ticker_dict.get(t, "Other") == ac]
            if not idx or w_ac <= 0:
                continue
            per = w_ac / len(idx)
            for i in idx:
                w[i] += per
        s = w.sum()
        return w / s if s > 0 else np.full(len(tickers_list), 1.0 / len(tickers_list))

    dyn_idx = regime_used_str.index.intersection(X_ret.index)
    dyn_weights = pd.DataFrame(index=dyn_idx, columns=tickers, dtype=float)

    for dt in dyn_idx:
        reg = regime_used_str.loc[dt]  # label
        w_t = build_etf_weights_from_class_alloc(profile_class_hybrid[reg], tickers, class_by_ticker)
        dyn_weights.loc[dt] = w_t

    dyn_weights = dyn_weights.apply(pd.to_numeric, errors="coerce").astype(float).dropna(how="all")

    # ======================================================
    # 6) FULL DYNAMIC TABLE (Regime + Dynamic weights by date)
    # ======================================================
    st.subheader("📋 Dynamic allocation table — Regime + Dynamic weights by date")

    reg_num = regime_used.reindex(dyn_weights.index).ffill().bfill()
    reg_label = reg_num.map(regime_map).fillna("Unknown")

    turnover = dyn_weights.diff().abs().sum(axis=1).fillna(0.0)
    trade_flag = (turnover > 1e-8).astype(int)

    dyn_table = dyn_weights.copy()
    dyn_table.insert(0, "Turnover (1-norm)", turnover.values)
    dyn_table.insert(0, "Trade?", trade_flag.values)
    dyn_table.insert(0, "Regime_state", reg_num.astype(int).values)
    dyn_table.insert(0, "Regime", reg_label.values)

    show_last_n = st.number_input(
        "Show last N dates (display only)",
        min_value=20,
        max_value=len(dyn_table),
        value=min(100, len(dyn_table)),
        step=20
    )

    st.dataframe(
        dyn_table.tail(int(show_last_n)).style.format(
            {**{c: "{:.2%}" for c in tickers},
             "Turnover (1-norm)": "{:.2%}"}
        ),
        use_container_width=True
    )

    dyn_table_csv = (
        dyn_table.reset_index()
        .rename(columns={dyn_table.index.name or "index": "Date"})
        .to_csv(index=False)
    )

    st.download_button(
        "⬇️ Download dynamic allocation table (CSV)",
        data=dyn_table_csv.encode("utf-8"),
        file_name="dynamic_allocation_by_date.csv",
        mime="text/csv"
    )

    # ======================================================
    # 7) PERFORMANCE VS BENCHMARK
    # ======================================================
    st.subheader("📈 Performance: Dynamic vs Benchmark")

    # Dynamic returns from weights
    ret_dyn_net = (X_ret.loc[dyn_weights.index] * dyn_weights).sum(axis=1).astype(float)

    # Benchmark (equal-weight or choose your own earlier)
    w_bench = np.full(len(tickers), 1.0 / len(tickers))
    r_bench = X_ret.dot(w_bench).reindex(ret_dyn_net.index).astype(float)

    bench_choice = "Equal-Weight"

    perf_df = pd.DataFrame({
        "Dynamic (net)": pd.to_numeric(ret_dyn_net, errors="coerce"),
        f"Benchmark: {bench_choice}": pd.to_numeric(r_bench, errors="coerce")
    }).dropna().astype(float)

    cum_df = np.exp(perf_df.cumsum()) - 1

    fig_perf = px.line(
        cum_df,
        x=cum_df.index,
        y=cum_df.columns,
        labels={"index": "Date", "value": "Cumulative return"},
        title="Cumulative return — Dynamic (net) vs Benchmark"
    )
    st.plotly_chart(fig_perf, use_container_width=True)

    # ======================================================
    # 8) SUMMARY STATS (UNDER GRAPH)
    # ======================================================
    st.subheader("📌 Performance Summary (below chart)")

    def _stats(r: pd.Series, rf=rf_rate):
        r = r.dropna().astype(float)
        ann_ret = r.mean() * 252
        ann_vol = r.std() * np.sqrt(252)
        sharpe = (ann_ret - rf) / ann_vol if ann_vol > 1e-12 else np.nan
        mdd = max_drawdown(r)
        return ann_ret, ann_vol, sharpe, mdd

    rows = []
    for col in perf_df.columns:
        a, v, s, d = _stats(perf_df[col], rf=rf_rate)
        rows.append([col, a, v, s, d])

    summary_df = pd.DataFrame(rows, columns=["Strategy", "Ann.Return", "Ann.Vol", "Sharpe", "Max DD"]).set_index("Strategy")

    st.dataframe(
        summary_df.style.format({
            "Ann.Return": "{:.2%}",
            "Ann.Vol": "{:.2%}",
            "Sharpe": "{:.2f}",
            "Max DD": "{:.2%}",
        }),
        use_container_width=True
    )

    # ======================================================
    # 9) REGIME PLOT (STEP-LIKE)
    # ======================================================
    st.subheader("🧭 Regime timeline")

    reg_plot = pd.DataFrame({
        "Regime state": reg_num.reindex(perf_df.index).ffill().bfill().astype(int)
    }, index=perf_df.index)

    fig_state = px.line(
        reg_plot,
        x=reg_plot.index,
        y="Regime state",
        title="HMM Regime State (0=Risk-Off, 1=Neutral, 2=Risk-On)"
    )
    fig_state.update_traces(line_shape="hv")
    fig_state.update_yaxes(
        tickmode="array",
        tickvals=[0, 1, 2],
        ticktext=["Risk-Off", "Neutral", "Risk-On"]
    )
    st.plotly_chart(fig_state, use_container_width=True)

    # ======================================================
    # 10) TRADE LOG (REGIME SWITCH TRADES)
    # ======================================================
    st.subheader("🧾 Trade Log (Regime Switches)")

    reg_lbl_aligned = reg_label.reindex(dyn_weights.index).ffill().bfill()

    switch_mask = reg_lbl_aligned.ne(reg_lbl_aligned.shift(1))
    switch_dates = dyn_weights.index[switch_mask].tolist()

    trade_rows = []
    for k in range(1, len(switch_dates)):
        dt = switch_dates[k]
        dt_prev = switch_dates[k - 1]

        w_prev = dyn_weights.loc[dt_prev].astype(float).values
        w_new = dyn_weights.loc[dt].astype(float).values
        delta = w_new - w_prev

        for i, t in enumerate(tickers):
            if abs(delta[i]) < 1e-8:
                continue
            trade_rows.append({
                "Trade Date": dt,
                "From Regime": str(reg_lbl_aligned.loc[dt_prev]),
                "To Regime": str(reg_lbl_aligned.loc[dt]),
                "Ticker": t,
                "Side": "BUY" if delta[i] > 0 else "SELL",
                "Δw": float(delta[i])
            })

    trade_log = pd.DataFrame(trade_rows)

    if trade_log.empty:
        st.info("No regime switches detected → no trades to log.")
    else:
        st.dataframe(
            trade_log.style.format({"Δw": "{:+.2%}"}),
            use_container_width=True
        )
        st.download_button(
            "⬇️ Download Trade Log (CSV)",
            data=trade_log.to_csv(index=False).encode("utf-8"),
            file_name="dynreg_trade_log.csv",
            mime="text/csv"
        )

