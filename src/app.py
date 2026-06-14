from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "src" / "data"
DATASET_CANDIDATES = (
    DATA_DIR / "master_dataset.csv",
    ROOT_DIR / "data" / "master_dataset.csv",
)
DATASET_PATH = next((path for path in DATASET_CANDIDATES if path.exists()), DATASET_CANDIDATES[0])
REQUIRED_DATA_COLUMNS = {"spx_close", "vix_decimal", "forward_realized_vol"}

REGIME_COLORS = {
    "Bull": "#16a34a",
    "Bear": "#d4a017",
    "Crisis": "#dc2626",
}

TAIL_EVENTS = {
    pd.Timestamp("2011-08-01"): "2011 Debt Crisis",
    pd.Timestamp("2018-02-05"): "Volmageddon",
    pd.Timestamp("2020-03-15"): "COVID Crash",
}

ANN_FACTORS: dict[str, int] = {"Monthly": 12, "Weekly": 52, "Daily": 252}
ENTRY_FREQUENCY_OPTIONS = tuple(ANN_FACTORS.keys())

PLOTLY_CONFIG = {"displaylogo": False, "responsive": True}


# ─── Fast DataFrame hasher ────────────────────────────────────────────────────
# Streamlit's default DataFrame hasher is slow. pd.util.hash_pandas_object is
# an O(n) vectorised hash — far cheaper for large frames.

def _hash_df(df: pd.DataFrame) -> str:
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()


# ─── Data layer ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_master_dataset(dataset_path: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_path, index_col="Date", parse_dates=True)
    missing_cols = REQUIRED_DATA_COLUMNS.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Master dataset missing column(s): {', '.join(sorted(missing_cols))}")
    return df.sort_index()


def normalize_entry_frequency(value: str) -> str:
    if value in ANN_FACTORS:
        return value
    normalized = str(value).strip().lower()
    for option in ENTRY_FREQUENCY_OPTIONS:
        if option.lower() in normalized:
            return option
    return "Monthly"


@st.cache_data(show_spinner=False)
def prepare_dataset(dataset_path: str) -> pd.DataFrame:
    """
    PERF: All expensive one-time computations live here and are cached for the
    entire session. Previously, expanding().rank() and isocalendar() were
    recomputed inside compute_strategy_frame on every slider interaction.
    """
    df = load_master_dataset(dataset_path).copy()
    df["vrp"] = df["vix_decimal"] - df["forward_realized_vol"]
    df["spx_sma200"] = df["spx_close"].rolling(window=200).mean()

    # PERF: pre-compute expanding VRP percentile rank once.
    # expanding().rank(pct=True) is O(n²). Previously called on the
    # filtered DataFrame inside compute_strategy_frame — i.e. on every
    # slider move. Now runs once and is reused via a cheap .reindex().
    df["vrp_pct_rank"] = df["vrp"].expanding().rank(pct=True)

    # PERF: pre-compute ISO week columns to avoid calling .isocalendar()
    # (which returns a full DataFrame) on every strategy recompute.
    iso = df.index.isocalendar()
    df["_iso_year"] = iso["year"].values
    df["_iso_week"] = iso["week"].values

    valid = df["spx_sma200"].notna()
    df["regime"] = pd.Series(
        np.select(
            [
                valid & (df["vix_decimal"] >= 0.30),
                valid & (df["vix_decimal"] < 0.20) & (df["spx_close"] > df["spx_sma200"]),
                valid,
            ],
            [
                np.array("Crisis", dtype=object),
                np.array("Bull", dtype=object),
                np.array("Bear", dtype=object),
            ],
            default=np.nan,
        ),
        index=df.index,
        dtype="object",
    )
    return df


@st.cache_data(show_spinner=False)
def compute_strategy_frame(
    dataset_path: str,
    start_date: str,
    end_date: str,
    selected_regime: str,
    vrp_pct_lower: int = 0,
    vrp_pct_upper: int = 100,
    entry_frequency: str = "Monthly",
    vol_scaling: bool = False,
    circuit_breaker_pct: int = 60,
    leverage_cap: float = 5.0,
) -> pd.DataFrame:
    """
    PERF: Signature changed from (df: pd.DataFrame, ...) to primitive scalars only.

    The old signature forced Streamlit to hash the entire filtered DataFrame on
    every interaction to compute the cache key — tens of milliseconds per call,
    and almost always a cache miss because the filtered frame changed with each
    slider move. With primitive args the cache key hashes in microseconds, and
    hits are reliable across rerenders that don't change the strategy params.

    The prepare_dataset() call below is free: it just returns the already-cached
    object without any computation.
    """
    df = prepare_dataset(dataset_path).loc[start_date:end_date].copy()
    if selected_regime != "All":
        df = df.loc[df["regime"] == selected_regime]

    if df.empty:
        return pd.DataFrame()

    entry_frequency = normalize_entry_frequency(entry_frequency)

    # ── 1. Entry dates ────────────────────────────────────────────────────────
    if entry_frequency == "Monthly":
        entries = df.groupby([df.index.year, df.index.month]).head(1).copy()
    elif entry_frequency == "Weekly":
        # PERF: uses pre-computed _iso_year/_iso_week — no isocalendar() call
        entries = df.groupby(["_iso_year", "_iso_week"], sort=False).head(1).copy()
    else:
        entries = df.copy()

    if entries.empty:
        return entries

    entries["ann_factor"] = ANN_FACTORS[entry_frequency]

    # ── 2. VRP Percentile Gate ────────────────────────────────────────────────
    # PERF: vrp_pct_rank pre-computed in prepare_dataset — just a reindex here
    entries["vrp_pct_rank"] = df["vrp_pct_rank"].reindex(entries.index)
    lower_bound = vrp_pct_lower / 100.0
    upper_bound = vrp_pct_upper / 100.0
    gate_pass = (
        (entries["vrp_pct_rank"] >= lower_bound)
        & (entries["vrp_pct_rank"] <= upper_bound)
    ) | entries["vrp_pct_rank"].isna()

    # ── 3. Circuit Breaker ────────────────────────────────────────────────────
    kappa = circuit_breaker_pct / 100.0
    breaker_pass = entries["vix_decimal"] <= kappa
    entries["active"] = gate_pass & breaker_pass

    # ── 4. Position sizing ────────────────────────────────────────────────────
    V0_sq = 0.20 ** 2
    if vol_scaling:
        raw_weight = V0_sq / (entries["vix_decimal"] ** 2)
        entries["weight"] = raw_weight.clip(upper=leverage_cap)
    else:
        entries["weight"] = 1.0

    # ── 5. PnL ────────────────────────────────────────────────────────────────
    raw_pnl = entries["weight"] * (
        entries["vix_decimal"] ** 2 - entries["forward_realized_vol"] ** 2
    )
    entries["pnl"] = np.where(entries["active"], raw_pnl, 0.0)
    entries["cum_return"] = entries["pnl"].cumsum()
    entries["capital_index"] = 10.0 + entries["cum_return"]
    entries["rolling_peak"] = entries["capital_index"].cummax()
    entries["drawdown"] = (entries["capital_index"] / entries["rolling_peak"]) - 1.0
    return entries


def compute_summary_metrics(
    df: pd.DataFrame, strategy_df: pd.DataFrame
) -> dict[str, float]:
    if strategy_df.empty:
        return {k: np.nan for k in
                ["avg_vrp", "win_rate", "sharpe_ratio", "max_drawdown", "entry_rate"]}

    ann_factor = (
        int(strategy_df["ann_factor"].iloc[0])
        if "ann_factor" in strategy_df.columns
        else 12
    )
    pnl = strategy_df["pnl"]
    pnl_std = pnl.std()

    sharpe_ratio = np.nan
    if pd.notna(pnl_std) and pnl_std != 0:
        ann_return = pnl.mean() * ann_factor
        ann_vol = pnl_std * np.sqrt(ann_factor)
        sharpe_ratio = ann_return / ann_vol if ann_vol != 0 else np.nan

    entry_rate = (
        float(strategy_df["active"].mean())
        if "active" in strategy_df.columns
        else np.nan
    )

    return {
        "avg_vrp": df["vrp"].mean() if not df.empty else np.nan,
        "win_rate": (df["vrp"] > 0).mean() if not df.empty else np.nan,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": strategy_df["drawdown"].min(),
        "entry_rate": entry_rate,
    }


def format_percent(value: float, decimals: int = 2) -> str:
    return "N/A" if pd.isna(value) else f"{value:.{decimals}%}"


def format_number(value: float, decimals: int = 2) -> str:
    return "N/A" if pd.isna(value) else f"{value:.{decimals}f}"


# ─── Chart helpers ────────────────────────────────────────────────────────────

def _get_regime_blocks(
    df: pd.DataFrame,
) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    regime_df = df.loc[df["regime"].notna(), ["regime"]]
    if regime_df.empty:
        return []
    blocks = []
    groups = (regime_df["regime"] != regime_df["regime"].shift()).cumsum()
    for _, block in regime_df.groupby(groups):
        blocks.append((block.index[0], block.index[-1], block["regime"].iloc[0]))
    return blocks


def _add_regime_shading(fig: go.Figure, blocks, opacity: float = 0.09) -> None:
    for start, end, regime in blocks:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=REGIME_COLORS[regime],
            opacity=opacity,
            line_width=0,
            layer="below",
        )


def _apply_layout(fig: go.Figure, title: str, yaxis_title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Date",
        yaxis_title=yaxis_title,
    )
    return fig


# ─── Chart builders ───────────────────────────────────────────────────────────
# PERF: All chart builders are now @st.cache_data.
# Overview charts accept primitive params so the cache key is trivially cheap.
# Strategy charts receive the already-cached strategy_df with _hash_df so
# Streamlit doesn't use its slow default DataFrame hasher.
# PERF: display_df.copy() + NaN-fill loop eliminated — regime masking is done
# inline with .where(), never allocating a full extra DataFrame copy.

@st.cache_data(show_spinner=False)
def build_volatility_chart(
    dataset_path: str, start_date: str, end_date: str, selected_regime: str,
) -> go.Figure:
    date_df = prepare_dataset(dataset_path).loc[start_date:end_date]
    regime_blocks = _get_regime_blocks(date_df)

    if selected_regime != "All":
        mask = date_df["regime"] == selected_regime
        vix = date_df["vix_decimal"].where(mask)
        fwd = date_df["forward_realized_vol"].where(mask)
    else:
        vix = date_df["vix_decimal"]
        fwd = date_df["forward_realized_vol"]

    fig = go.Figure()
    _add_regime_shading(fig, regime_blocks)
    fig.add_trace(go.Scatter(
        x=date_df.index, y=vix,
        mode="lines", name="VIX (Implied Vol)",
        line=dict(color="#d97706", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=date_df.index, y=fwd,
        mode="lines", name="30D Forward Realized Vol",
        line=dict(color="#2563eb", width=2.5),
    ))
    fig.update_yaxes(tickformat=".0%")
    return _apply_layout(fig, "VIX vs. 30-Day Forward Realized Volatility", "Volatility")


@st.cache_data(show_spinner=False)
def build_vrp_chart(
    dataset_path: str, start_date: str, end_date: str, selected_regime: str,
) -> go.Figure:
    date_df = prepare_dataset(dataset_path).loc[start_date:end_date]

    vrp = (
        date_df["vrp"].where(date_df["regime"] == selected_regime)
        if selected_regime != "All"
        else date_df["vrp"]
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=date_df.index, y=vrp.where(vrp >= 0),
        mode="lines", name="Positive VRP",
        line=dict(color="#15803d", width=2),
        fill="tozeroy", fillcolor="rgba(21, 128, 61, 0.20)",
    ))
    fig.add_trace(go.Scatter(
        x=date_df.index, y=vrp.where(vrp < 0),
        mode="lines", name="Negative VRP",
        line=dict(color="#b91c1c", width=2),
        fill="tozeroy", fillcolor="rgba(185, 28, 28, 0.22)",
    ))
    fig.add_hline(y=0, line_color="#111827", line_dash="dash", line_width=1.5)
    fig.update_yaxes(tickformat=".1%")
    return _apply_layout(fig, "Volatility Risk Premium Through Time", "VRP")


@st.cache_data(show_spinner=False)
def build_distribution_chart(
    dataset_path: str, start_date: str, end_date: str, selected_regime: str,
) -> go.Figure:
    date_df = prepare_dataset(dataset_path).loc[start_date:end_date]
    vrp_data = (
        date_df.loc[date_df["regime"] == selected_regime, "vrp"]
        if selected_regime != "All"
        else date_df["vrp"]
    )

    mean_vrp = vrp_data.mean()
    median_vrp = vrp_data.median()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=vrp_data, nbinsx=60,
        marker=dict(color="#7dd3fc", line=dict(color="#0f172a", width=0.6)),
        opacity=0.85, name="VRP Frequency",
    ))
    fig.add_vline(x=mean_vrp, line_color="#dc2626", line_dash="dash", line_width=2,
                  annotation_text=f"Mean {mean_vrp:.2%}", annotation_position="top right")
    fig.add_vline(x=median_vrp, line_color="#15803d", line_dash="solid", line_width=2,
                  annotation_text=f"Median {median_vrp:.2%}", annotation_position="top left")
    fig.update_layout(
        title="VRP Empirical Distribution", template="plotly_white", bargap=0.04,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="VRP", yaxis_title="Frequency",
    )
    fig.update_xaxes(tickformat=".1%")
    return fig


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: _hash_df})
def build_cumulative_pnl_chart(
    strategy_df: pd.DataFrame,
    entry_frequency: str = "Monthly",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=strategy_df.index,
        y=strategy_df["capital_index"],
        mode="lines",
        name="Capital Index",
        line=dict(color="#1d4ed8", width=2.5),
        hovertemplate=(
            "Date=%{x|%Y-%m-%d}<br>Capital Index=%{y:.3f}"
            "<br>Cumulative P&L=%{customdata:.3f}<extra></extra>"
        ),
        customdata=strategy_df["cum_return"],
    ))

    if "active" in strategy_df.columns and entry_frequency != "Daily":
        skipped = strategy_df.loc[~strategy_df["active"]]
        if not skipped.empty:
            fig.add_trace(go.Scatter(
                x=skipped.index,
                y=skipped["capital_index"],
                mode="markers",
                name="Gate / Breaker: held cash",
                marker=dict(
                    color="#9ca3af", size=7, symbol="x",
                    line=dict(width=1.5, color="#6b7280"),
                ),
                hovertemplate=(
                    "Date=%{x|%Y-%m-%d}<br>Blocked — held cash"
                    "<br>Capital=%{y:.3f}<extra></extra>"
                ),
            ))

    return _apply_layout(
        fig, "Variance Swap Harvesting Strategy: Cumulative P&L", "Capital Index"
    )


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: _hash_df})
def build_drawdown_chart(strategy_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=strategy_df.index,
        y=strategy_df["drawdown"],
        mode="lines", name="Drawdown",
        line=dict(color="#b91c1c", width=2),
        fill="tozeroy", fillcolor="rgba(185, 28, 28, 0.22)",
    ))
    for event_date, label in TAIL_EVENTS.items():
        if not (strategy_df.index.min() <= event_date <= strategy_df.index.max()):
            continue
        idx = strategy_df.index.get_indexer([event_date], method="nearest")[0]
        closest = strategy_df.index[idx]
        fig.add_annotation(
            x=closest, y=strategy_df.loc[closest, "drawdown"],
            text=label, showarrow=True, arrowhead=2, ax=0, ay=-45,
            bgcolor="rgba(255,255,255,0.85)", bordercolor="#111827",
            font=dict(size=11),
        )
    fig.update_yaxes(tickformat=".0%")
    return _apply_layout(fig, "Drawdown Profile", "Drawdown")


# ─── Page ─────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="30-Day Volatility Risk Premium Dashboard", layout="wide")
st.title("30-Day Volatility Risk Premium Dashboard")

if not DATASET_PATH.exists():
    candidates = "\n".join(f"- `{path}`" for path in DATASET_CANDIDATES)
    st.error(
        "Master dataset not found.\n\n"
        f"Checked:\n{candidates}\n\n"
        "Run `python -m src.dataloader --build-master-dataset` to regenerate it."
    )
    st.stop()

try:
    df = prepare_dataset(str(DATASET_PATH))
except Exception as exc:
    st.error(f"Unable to load master dataset from `{DATASET_PATH}`.")
    st.exception(exc)
    st.stop()

min_date = df.index.min().date()
max_date = df.index.max().date()
DATASET_PATH_STR = str(DATASET_PATH)  # single str used everywhere as the cache key

# ─── Sidebar: Global Filters ──────────────────────────────────────────────────

st.sidebar.header("Global Filters")
selected_dates = st.sidebar.slider(
    "Date Range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
)
selected_regime = st.sidebar.radio(
    "Market Regime",
    options=["All", "Bull", "Bear", "Crisis"],
    horizontal=False,
)

# ─── Sidebar: Strategy Parameters ─────────────────────────────────────────────

st.sidebar.divider()
st.sidebar.header("Strategy Parameters")

entry_frequency = st.sidebar.radio(
    "Entry Frequency",
    options=ENTRY_FREQUENCY_OPTIONS,
    index=0,
    help=(
        "Monthly: one non-overlapping contract per month (K=12). "
        "Weekly: 52 entries/yr with substantial overlap between 30-day contracts (K=52). "
        "Daily: 252 entries/yr, 22 simultaneous overlapping contracts (K=252). "
        "Daily is smoother in bull runs but crashes hit all 22 positions at once."
    ),
)
entry_frequency = normalize_entry_frequency(entry_frequency)

vrp_pct_range = st.sidebar.slider(
    "VRP Percentile Range Gate",
    min_value=0, max_value=100, value=(0, 100), step=5, format="%d%%",
    help=(
        "Select the lower and upper bounds of historical VRP percentiles allowed for "
        "entry. Lower bounds avoid selling underpriced insurance. Upper bounds avoid "
        "selling directly into explosive panics."
    ),
)
vrp_pct_lower, vrp_pct_upper = vrp_pct_range
if (vrp_pct_lower, vrp_pct_upper) != (0, 100):
    st.sidebar.caption(
        f"Entering only when VRP is between its {vrp_pct_lower}th and "
        f"{vrp_pct_upper}th historical percentiles."
    )

st.sidebar.divider()

vol_scaling = st.sidebar.checkbox(
    "Volatility-Inverse Sizing",
    value=False,
    help=(
        "Scale each contract by w = 0.04 / VIX². "
        "Normalises dollar risk across regimes so a Crisis entry isn't "
        "7× larger in variance points than a Bull entry."
    ),
)

leverage_cap = 5.0
if vol_scaling:
    leverage_cap = st.sidebar.slider(
        "Max Leverage Cap (Λ)",
        min_value=1.0, max_value=5.0, value=5.0, step=0.25, format="%.2f×",
        help=(
            "Hard upper bound on w_t. Prevents complacency-trap over-leveraging "
            "when VIX collapses to historic lows. 5.0× = no effective cap."
        ),
    )
    vix_trigger = 0.20 / (leverage_cap ** 0.5)
    if leverage_cap < 5.0:
        st.sidebar.caption(
            f"Cap of {leverage_cap:.2f}× activates when VIX < {vix_trigger:.1%}."
        )

circuit_breaker_pct = st.sidebar.slider(
    "Circuit Breaker VIX Ceiling (κ)",
    min_value=20, max_value=60, value=60, step=5, format="%d%%",
    help=(
        "Block all new entries when spot VIX exceeds κ. Strategy holds cash until "
        "VIX falls back below the threshold. 60% = effectively disabled."
    ),
)
if circuit_breaker_pct < 60:
    st.sidebar.caption(f"No new entries when VIX > {circuit_breaker_pct}%.")

# ─── Derive filter primitives ─────────────────────────────────────────────────
# PERF: Convert dates to strings once here. All cache-key arguments downstream
# are now plain scalars — no DataFrame hashing anywhere in the hot path.

start_date_str = str(selected_dates[0])
end_date_str   = str(selected_dates[1])

# These are zero-copy slices of the cached master DataFrame, not new allocations.
date_filtered_df = df.loc[start_date_str:end_date_str]
filtered_df = (
    date_filtered_df
    if selected_regime == "All"
    else date_filtered_df.loc[date_filtered_df["regime"] == selected_regime]
)

strategy_df = compute_strategy_frame(
    DATASET_PATH_STR,
    start_date_str,
    end_date_str,
    selected_regime,
    vrp_pct_lower=vrp_pct_lower,
    vrp_pct_upper=vrp_pct_upper,
    entry_frequency=entry_frequency,
    vol_scaling=vol_scaling,
    circuit_breaker_pct=circuit_breaker_pct,
    leverage_cap=leverage_cap,
)
summary_metrics = compute_summary_metrics(filtered_df, strategy_df)

# ─── Metrics row ──────────────────────────────────────────────────────────────

metric_cols = st.columns(5)
metric_cols[0].metric("Average VRP",     format_percent(summary_metrics["avg_vrp"]))
metric_cols[1].metric("Win Rate",        format_percent(summary_metrics["win_rate"]))
metric_cols[2].metric("Strategy Sharpe", format_number(summary_metrics["sharpe_ratio"]))
metric_cols[3].metric("Max Drawdown",    format_percent(summary_metrics["max_drawdown"]))
metric_cols[4].metric(
    "Entry Rate",
    format_percent(summary_metrics["entry_rate"]),
    help="Fraction of candidate entry points where all active filters allowed entry.",
)

st.write(
    "The volatility risk premium measures how much investors systematically pay up "
    "for convex downside insurance: option prices embed risk aversion, crash "
    "insurance demand, and compensation for jump risk, so implied volatility "
    "tends to trade above the volatility the market subsequently realizes."
)

if filtered_df.empty:
    st.warning(
        "The selected date range and regime filter returned no observations. "
        "Adjust the sidebar filters to view the dashboard."
    )
    st.stop()

freq_label = entry_frequency.lower()
n_active = int(strategy_df["active"].sum()) if not strategy_df.empty else 0
st.caption(
    f"{len(filtered_df):,} daily observations | "
    f"{len(strategy_df):,} {freq_label} entry points | "
    f"{n_active:,} active after filters"
)
st.caption("Regime shading: Bull = green, Bear = gold, Crisis = red.")

# ─── Dashboard tabs ───────────────────────────────────────────────────────────

overview_tab, strategy_tab = st.tabs(
    ["VRP Overview", "Strategy Performance & Tail-Risk Drawdowns"]
)

with overview_tab:
    st.plotly_chart(
        build_volatility_chart(DATASET_PATH_STR, start_date_str, end_date_str, selected_regime),
        width="stretch", config=PLOTLY_CONFIG,
    )
    st.plotly_chart(
        build_vrp_chart(DATASET_PATH_STR, start_date_str, end_date_str, selected_regime),
        width="stretch", config=PLOTLY_CONFIG,
    )
    st.plotly_chart(
        build_distribution_chart(DATASET_PATH_STR, start_date_str, end_date_str, selected_regime),
        width="stretch", config=PLOTLY_CONFIG,
    )

with strategy_tab:
    if strategy_df.empty:
        st.warning(
            "The current filter selection does not contain enough observations to form "
            "the strategy sample."
        )
    else:
        pnl_tab, drawdown_tab = st.tabs(["Cumulative P&L", "Drawdown Profile"])

        with pnl_tab:
            st.plotly_chart(
                build_cumulative_pnl_chart(strategy_df, entry_frequency),
                width="stretch", config=PLOTLY_CONFIG,
            )
            n_skipped = int((~strategy_df["active"]).sum())
            if n_skipped > 0 and entry_frequency != "Daily":
                st.caption(
                    f"Grey x markers: {n_skipped} entry point(s) blocked by VRP range gate "
                    f"or circuit breaker — held cash at 0 PnL."
                )
            elif n_skipped > 0:
                st.caption(
                    f"{n_skipped:,} of {len(strategy_df):,} daily entry points were "
                    "blocked by the active filters (markers suppressed at daily frequency)."
                )

        with drawdown_tab:
            st.plotly_chart(
                build_drawdown_chart(strategy_df),
                width="stretch", config=PLOTLY_CONFIG,
            )
