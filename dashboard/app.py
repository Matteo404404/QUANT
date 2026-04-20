"""
app.py
======
Streamlit Dashboard for Optiver Realized Volatility

Tabs:
  1. Volatility Predictions — interactive correlation graph, pred vs actual
  2. Systemic Risk          — hub ranking, contagion heatmap, time series
  3. Model Performance      — RMSPE/R² comparison, feature importance

Run:
  streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.lob_features import PROCESSED_DIR
from src.metrics import rmspe as _rmspe, r2_score as _r2

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# ---------------------------------------------------------------------------
# Config pagina
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title = "Optiver Volatility — Research Dashboard",
    page_icon  = "📈",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ---------------------------------------------------------------------------
# CSS custom
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: #1c1f26;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #00d4aa;
        margin-bottom: 10px;
    }
    .metric-title { color: #8b9bb4; font-size: 12px; text-transform: uppercase; }
    .metric-value { color: #ffffff; font-size: 28px; font-weight: 700; }
    .metric-delta { font-size: 12px; }
    .section-header {
        color: #00d4aa;
        font-size: 18px;
        font-weight: 600;
        margin: 20px 0 10px 0;
        border-bottom: 1px solid #2a2d35;
        padding-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_features():
    p = PROCESSED_DIR / "features_with_nn.parquet"
    if not p.exists():
        p = PROCESSED_DIR / "features.parquet"
    return pd.read_parquet(p)

@st.cache_data(show_spinner=False)
def load_predictions():
    lgb_path = PROCESSED_DIR / "lgb_test.parquet"
    if not lgb_path.exists():
        return None
    df = pd.read_parquet(lgb_path)
    # add gnn_pred if available
    gnn_path = PROCESSED_DIR / "gnn_test.parquet"
    if gnn_path.exists():
        gnn_df = pd.read_parquet(gnn_path)
        df = df.merge(gnn_df[["stock_id","time_id","gnn_pred"]], on=["stock_id","time_id"], how="left")
    # Ensemble weights loaded from results if available, otherwise default
    results_csv = Path(__file__).resolve().parents[1] / "results_final.csv"
    w_lgb, w_gnn = 0.623, 0.377
    if results_csv.exists():
        res_df = pd.read_csv(results_csv)
        ens_row = res_df[res_df["model"].str.contains("Ensemble", case=False, na=False)]
        if len(ens_row) > 0:
            import re
            m = re.search(r"LGB.?(\d+\.\d+).*GNN.?(\d+\.\d+)", ens_row.iloc[0]["model"])
            if m:
                w_lgb, w_gnn = float(m.group(1)), float(m.group(2))

    if "gnn_pred" in df.columns:
        df["ensemble_pred"] = w_lgb * df["lgb_pred"] + w_gnn * df["gnn_pred"]
    else:
        df["ensemble_pred"] = df["lgb_pred"]
    return df

@st.cache_data(show_spinner=False)
def load_systemic_risk():
    sr_path  = PROCESSED_DIR / "systemic_risk.parquet"
    hub_path = PROCESSED_DIR / "hub_stocks.csv"
    if not sr_path.exists():
        return None, None
    return pd.read_parquet(sr_path), pd.read_csv(hub_path)

@st.cache_data(show_spinner=False)
def load_lgb_importance():
    # loads importance from the last saved fold
    imp_path = PROCESSED_DIR / "lgb_importance.csv"
    if imp_path.exists():
        return pd.read_csv(imp_path)
    return None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rmspe(y_true, y_pred):
    return _rmspe(np.asarray(y_true), np.asarray(y_pred))

def r2(y_true, y_pred):
    return _r2(np.asarray(y_true), np.asarray(y_pred))

def build_graph_figure(
    features_df: pd.DataFrame,
    preds_df:    pd.DataFrame,
    time_id:     int,
    top_k_edges: int = 30,
) -> go.Figure:
    """
    Interactive graph for a given time_id.
    Nodes = stocks (coloured by prediction error).
    Edges = top-K strongest correlations. Circular layout.
    """
    feat = features_df[features_df["time_id"] == time_id].copy()
    pred = preds_df[preds_df["time_id"] == time_id].copy()

    if len(pred) == 0:
        return go.Figure().update_layout(
            title="No predictions available for this time_id",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        )

    merged = feat.merge(pred[["stock_id","lgb_pred","ensemble_pred"]], on="stock_id", how="left")
    merged = merged.dropna(subset=["lgb_pred"])

    stocks   = sorted(merged["stock_id"].unique())
    n_stocks = len(stocks)
    s_to_idx = {s: i for i, s in enumerate(stocks)}

    # absolute error for colour
    merged["abs_error"] = np.abs(merged["target"] - merged["ensemble_pred"])
    merged["pct_error"] = merged["abs_error"] / (merged["target"] + 1e-9) * 100

    # Layout circolare
    angles = np.linspace(0, 2 * np.pi, n_stocks, endpoint=False)
    pos_x  = np.cos(angles)
    pos_y  = np.sin(angles)
    idx_to_pos = {i: (pos_x[i], pos_y[i]) for i in range(n_stocks)}

    # Similarity proxy: inverse absolute RV difference (single-snapshot approximation,
    # not the rolling Pearson correlation the GNN uses, but fast enough for the dashboard)
    rv_series = feat.set_index("stock_id")["rv_full"].reindex(stocks).fillna(0).values
    rv_mat   = rv_series.reshape(-1, 1) - rv_series.reshape(1, -1)
    sim_mat  = 1.0 / (1.0 + np.abs(rv_mat) * 1000)
    np.fill_diagonal(sim_mat, 0)

    # top-K edges
    triu_idx = np.triu_indices(n_stocks, k=1)
    weights  = sim_mat[triu_idx]
    top_k    = min(top_k_edges, len(weights))
    top_idx  = np.argsort(weights)[-top_k:]

    edge_traces = []
    for idx in top_idx:
        i, j = triu_idx[0][idx], triu_idx[1][idx]
        w    = weights[idx]
        x0, y0 = idx_to_pos[i]
        x1, y1 = idx_to_pos[j]
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=w * 3, color=f"rgba(0,212,170,{min(w, 0.6)})"),
            hoverinfo="none",
            showlegend=False,
        ))

    # nodes
    node_x      = [idx_to_pos[s_to_idx[s]][0] for s in stocks]
    node_y      = [idx_to_pos[s_to_idx[s]][1] for s in stocks]
    node_color  = merged.set_index("stock_id")["pct_error"].reindex(stocks).fillna(0).values
    node_size   = 8 + merged.set_index("stock_id")["rv_full"].reindex(stocks).fillna(0).values * 3000
    node_size   = np.clip(node_size, 6, 30)

    node_text = []
    for s in stocks:
        row = merged[merged["stock_id"] == s]
        if len(row) == 0:
            node_text.append(f"Stock {int(s)}")
            continue
        row = row.iloc[0]
        node_text.append(
            f"Stock {int(s)}<br>"
            f"RV actual: {row['target']:.5f}<br>"
            f"RV pred:   {row['ensemble_pred']:.5f}<br>"
            f"Error:     {row['pct_error']:.1f}%<br>"
            f"RV full:   {row['rv_full']:.5f}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale="RdYlGn_r",
            colorbar=dict(title="Pred error %", thickness=12),
            showscale=True,
            line=dict(width=1, color="#ffffff"),
        ),
        text=[f"{int(s)}" for s in stocks],
        textposition="top center",
        textfont=dict(size=7, color="#cccccc"),
        hovertext=node_text,
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title=dict(
            text=f"Volatility Graph — Time ID {time_id}  ({n_stocks} stocks, top-{top_k} edges)",
            font=dict(size=14, color="#ffffff"),
        ),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=550,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def build_scatter_pred_actual(preds_df: pd.DataFrame, time_id: int) -> go.Figure:
    sub = preds_df[preds_df["time_id"] == time_id].copy()
    if len(sub) == 0:
        return go.Figure()

    sub = sub.sort_values("stock_id")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["target"], y=sub["ensemble_pred"],
        mode="markers",
        marker=dict(
            color=sub["stock_id"], colorscale="Viridis",
            size=7, opacity=0.8,
            colorbar=dict(title="Stock ID", thickness=10),
        ),
        hovertext=[f"Stock {int(r.stock_id)}<br>Actual: {r.target:.5f}<br>Pred: {r.ensemble_pred:.5f}"
                   for _, r in sub.iterrows()],
        hoverinfo="text",
        name="Predictions",
    ))
    # ideal line
    lim = max(sub["target"].max(), sub["ensemble_pred"].max()) * 1.05
    fig.add_trace(go.Scatter(
        x=[0, lim], y=[0, lim],
        mode="lines",
        line=dict(color="#00d4aa", dash="dash", width=1.5),
        name="Perfect prediction",
    ))
    fig.update_layout(
        title=dict(text=f"Predicted vs Actual RV — Time ID {time_id}", font=dict(color="#fff")),
        xaxis=dict(title="Actual RV", gridcolor="#2a2d35", color="#aaa"),
        yaxis=dict(title="Predicted RV", gridcolor="#2a2d35", color="#aaa"),
        paper_bgcolor="#0e1117", plot_bgcolor="#13161d",
        height=380, legend=dict(font=dict(color="#aaa")),
        margin=dict(l=50, r=20, t=50, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📈 Optiver Volatility")
    st.markdown("**Research Dashboard**")
    st.divider()

    tab_choice = st.radio(
        "Navigate",
        ["🔮 Volatility Predictions", "⚠️ Systemic Risk", "📊 Model Performance"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(
        "<small style='color:#555'>GNN + LightGBM ensemble<br>"
        "on Optiver LOB data<br><br>"
        "RMSPE: **0.269** | R²: **0.838**</small>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

with st.spinner("Loading data..."):
    features_df  = load_features()
    preds_df     = load_predictions()
    sr_df, hub_df = load_systemic_risk()

all_time_ids = sorted(features_df["time_id"].unique())
test_time_ids = (
    sorted(preds_df["time_id"].unique()) if preds_df is not None else all_time_ids[-100:]
)

# ---------------------------------------------------------------------------
# TAB 1 — Volatility Predictions
# ---------------------------------------------------------------------------

if tab_choice == "🔮 Volatility Predictions":

    st.markdown("## 🔮 Volatility Predictions")
    st.markdown(
        "Interactive volatility correlation graph. "
        "**Nodes** = stocks (colour = prediction error). "
        "**Edges** = top-30 strongest correlations."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_tid = st.select_slider(
            "Time ID",
            options=test_time_ids,
            value=test_time_ids[len(test_time_ids) // 2],
        )
    with col2:
        top_k = st.number_input("Top-K edges", min_value=10, max_value=100, value=30, step=5)

    if preds_df is not None:
        # quick metrics for this time_id
        sub = preds_df[preds_df["time_id"] == selected_tid]
        if len(sub) > 0:
            tid_rmspe = rmspe(sub["target"].values, sub["ensemble_pred"].values)
            tid_r2    = r2(sub["target"].values, sub["ensemble_pred"].values)
            worst_stock = sub.assign(
                err=lambda d: np.abs(d["target"] - d["ensemble_pred"]) / (d["target"] + 1e-9)
            ).nlargest(1, "err").iloc[0]

            m1, m2, m3 = st.columns(3)
            m1.metric("RMSPE (this session)", f"{tid_rmspe:.4f}")
            m2.metric("R² (this session)", f"{tid_r2:.4f}")
            m3.metric("Worst stock", f"Stock {int(worst_stock['stock_id'])}",
                      f"{worst_stock['err']*100:.1f}% error")

        col_graph, col_scatter = st.columns([3, 2])
        with col_graph:
            fig_graph = build_graph_figure(features_df, preds_df, selected_tid, top_k_edges=int(top_k))
            st.plotly_chart(fig_graph, use_container_width=True)

        with col_scatter:
            fig_scatter = build_scatter_pred_actual(preds_df, selected_tid)
            st.plotly_chart(fig_scatter, use_container_width=True)

            # top-5 errors for this time_id
            st.markdown("**Largest prediction errors**")
            worst = (
                preds_df[preds_df["time_id"] == selected_tid]
                .assign(pct_err=lambda d: np.abs(d["target"] - d["ensemble_pred"]) / (d["target"] + 1e-9) * 100)
                .nlargest(5, "pct_err")[["stock_id","target","ensemble_pred","pct_err"]]
                .rename(columns={"stock_id":"Stock","target":"Actual RV","ensemble_pred":"Pred RV","pct_err":"Error %"})
            )
            worst["Actual RV"] = worst["Actual RV"].map("{:.5f}".format)
            worst["Pred RV"]   = worst["Pred RV"].map("{:.5f}".format)
            worst["Error %"]   = worst["Error %"].map("{:.1f}%".format)
            st.dataframe(worst, hide_index=True, use_container_width=True)
    else:
        st.warning("Run `python src/models/baseline.py` first to generate predictions.")

# ---------------------------------------------------------------------------
# TAB 2 — Systemic Risk
# ---------------------------------------------------------------------------

elif tab_choice == "⚠️ Systemic Risk":

    st.markdown("## ⚠️ Systemic Risk — Volatility Contagion")
    st.markdown(
        "Systemic importance computed using **weighted PageRank + Strength + Eigenvector centrality** "
        "on the volatility correlation graph. "
        "A hub stock is a node that, if it experiences a volatility shock, "
        "propagates stress to other stocks in the market."
    )

    if hub_df is None:
        st.warning("Run `python src/analysis/systemic_risk.py` first.")
    else:
        # Top metrics
        top1 = hub_df.iloc[0]
        top2 = hub_df.iloc[1]
        top3 = hub_df.iloc[2]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("#1 Hub Stock",    f"Stock {int(top1['stock_id'])}", f"score {top1['avg_score']:.3f}")
        c2.metric("#2 Hub Stock",    f"Stock {int(top2['stock_id'])}", f"score {top2['avg_score']:.3f}")
        c3.metric("#3 Hub Stock",    f"Stock {int(top3['stock_id'])}", f"score {top3['avg_score']:.3f}")
        c4.metric("Total stocks", f"{len(hub_df)}", "in network")

        col_rank, col_ts = st.columns([1, 2])

        with col_rank:
            st.markdown("**Systemic Importance Ranking**")
            top20 = hub_df.head(20)[["rank","stock_id","avg_score","std_score","max_score"]].copy()
            top20["avg_score"] = top20["avg_score"].map("{:.3f}".format)
            top20["std_score"] = top20["std_score"].map("{:.3f}".format)
            top20["max_score"] = top20["max_score"].map("{:.3f}".format)
            st.dataframe(
                top20.rename(columns={
                    "rank":"#","stock_id":"Stock",
                    "avg_score":"Avg Score","std_score":"Std","max_score":"Max"
                }),
                hide_index=True,
                use_container_width=True,
                height=500,
            )

        with col_ts:
            st.markdown("**Systemic Importance Over Time — Top 5 Hub Stocks**")
            if sr_df is not None:
                top5_ids  = hub_df.head(5)["stock_id"].tolist()
                sr_top5   = sr_df[sr_df["stock_id"].isin(top5_ids)].copy()

                fig_ts = go.Figure()
                colors = px.colors.qualitative.T10
                for i, sid in enumerate(top5_ids):
                    sub = sr_top5[sr_top5["stock_id"] == sid].sort_values("time_id")
                    fig_ts.add_trace(go.Scatter(
                        x=sub["time_id"], y=sub["systemic_importance"],
                        mode="lines",
                        name=f"Stock {int(sid)}",
                        line=dict(width=2, color=colors[i % len(colors)]),
                    ))
                fig_ts.update_layout(
                    paper_bgcolor="#0e1117", plot_bgcolor="#13161d",
                    xaxis=dict(title="Time ID", gridcolor="#2a2d35", color="#aaa"),
                    yaxis=dict(title="Systemic Importance Score", gridcolor="#2a2d35", color="#aaa"),
                    legend=dict(font=dict(color="#ccc"), bgcolor="#1c1f26"),
                    height=250, margin=dict(l=50, r=20, t=20, b=50),
                )
                st.plotly_chart(fig_ts, use_container_width=True)

            # Contagion matrix
            st.markdown("**Volatility Contagion Matrix (top-20 stocks)**")
            cm_path = RESULTS_DIR / "contagion_matrix.png"
            if cm_path.exists():
                st.image(str(cm_path), use_container_width=True)
            else:
                st.info("contagion_matrix.png not found in results/")

        # Bar chart ranking completo
        st.markdown("**Full Systemic Importance Distribution**")
        fig_bar = px.bar(
            hub_df.head(40),
            x="stock_id", y="avg_score",
            error_y="std_score",
            color="avg_score",
            color_continuous_scale="Teal",
            labels={"stock_id": "Stock ID", "avg_score": "Avg Systemic Score"},
        )
        fig_bar.update_layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#13161d",
            xaxis=dict(gridcolor="#2a2d35", color="#aaa", type="category"),
            yaxis=dict(gridcolor="#2a2d35", color="#aaa"),
            coloraxis_showscale=False,
            height=320, margin=dict(l=50, r=20, t=20, b=50),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3 — Model Performance
# ---------------------------------------------------------------------------

elif tab_choice == "📊 Model Performance":

    st.markdown("## 📊 Model Performance")

    if preds_df is None:
        st.warning("Run `python src/models/baseline.py` first.")
    else:
        y_true = preds_df["target"].values
        y_lgb  = preds_df["lgb_pred"].values
        y_ens  = preds_df["ensemble_pred"].values

        results = {
            "LightGBM + NN features": (y_lgb,  y_true),
            "Ensemble (LGB×0.62 + GNN×0.38)": (y_ens, y_true),
        }
        if "gnn_pred" in preds_df.columns:
            results["GraphSAGE"] = (preds_df["gnn_pred"].values, y_true)

        # metrics
        rows = []
        for name, (pred, true) in results.items():
            rows.append({
                "Model"  : name,
                "RMSPE"  : f"{rmspe(true, pred):.5f}",
                "R²"     : f"{r2(true, pred):.4f}",
                "MAE"    : f"{np.mean(np.abs(true-pred)):.6f}",
            })
        perf_df = pd.DataFrame(rows)

        st.markdown("**Test Set Metrics**")
        st.dataframe(perf_df, hide_index=True, use_container_width=True)

        # RMSPE per stock (distribuzione errori)
        st.markdown("**RMSPE Distribution by Stock**")
        stock_rmspe = (
            preds_df
            .groupby("stock_id")
            .apply(lambda d: rmspe(d["target"].values, d["ensemble_pred"].values))
            .reset_index()
            .rename(columns={0: "rmspe"})
            .sort_values("rmspe", ascending=False)
        )

        fig_stock = px.bar(
            stock_rmspe,
            x="stock_id", y="rmspe",
            color="rmspe",
            color_continuous_scale="RdYlGn_r",
            labels={"stock_id": "Stock ID", "rmspe": "RMSPE"},
        )
        fig_stock.update_layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#13161d",
            xaxis=dict(gridcolor="#2a2d35", color="#aaa", type="category",
                       tickmode="array",
                       tickvals=stock_rmspe["stock_id"].tolist()[::5]),
            yaxis=dict(gridcolor="#2a2d35", color="#aaa"),
            coloraxis_showscale=False,
            height=340, margin=dict(l=50, r=20, t=20, b=60),
        )
        st.plotly_chart(fig_stock, use_container_width=True)

        # RMSPE nel tempo (train vs test split)
        st.markdown("**RMSPE Over Time (test window)**")
        time_rmspe = (
            preds_df
            .groupby("time_id")
            .apply(lambda d: rmspe(d["target"].values, d["ensemble_pred"].values))
            .reset_index()
            .rename(columns={0: "rmspe"})
        )
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=time_rmspe["time_id"], y=time_rmspe["rmspe"],
            mode="lines", fill="tozeroy",
            line=dict(color="#00d4aa", width=1.5),
            fillcolor="rgba(0,212,170,0.1)",
            name="RMSPE",
        ))
        fig_time.add_hline(
            y=time_rmspe["rmspe"].mean(),
            line_dash="dash", line_color="#ff6b6b",
            annotation_text=f"Avg {time_rmspe['rmspe'].mean():.4f}",
            annotation_font_color="#ff6b6b",
        )
        fig_time.update_layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#13161d",
            xaxis=dict(title="Time ID", gridcolor="#2a2d35", color="#aaa"),
            yaxis=dict(title="RMSPE", gridcolor="#2a2d35", color="#aaa"),
            height=300, margin=dict(l=50, r=20, t=20, b=50),
        )
        st.plotly_chart(fig_time, use_container_width=True)

        # Feature importance
        imp_path = PROCESSED_DIR / "lgb_importance.csv"
        if imp_path.exists():
            st.markdown("**LightGBM Feature Importance**")
            imp_df  = pd.read_csv(imp_path).sort_values("importance", ascending=True).tail(20)
            fig_imp = px.bar(
                imp_df, x="importance", y="feature",
                orientation="h", color="importance",
                color_continuous_scale="Teal",
            )
            fig_imp.update_layout(
                paper_bgcolor="#0e1117", plot_bgcolor="#13161d",
                xaxis=dict(gridcolor="#2a2d35", color="#aaa"),
                yaxis=dict(color="#aaa"),
                coloraxis_showscale=False,
                height=500, margin=dict(l=140, r=20, t=20, b=50),
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance file not found. It gets saved automatically when you run baseline.py.")