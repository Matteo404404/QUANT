
***

# Optiver Realized Volatility — GNN + LightGBM Research Pipeline

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-ff69b4.svg)](https://lightgbm.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A novel end-to-end pipeline for realized volatility prediction combining Graph Neural Networks on dynamic correlation graphs, gradient boosting with time-ID nearest neighbour features, cross-stock correlation features, and a systemic risk analysis layer.**

## 📑 Table of Contents
- [Results](#-results)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
-[Key Technical Contributions](#-key-technical-contributions)
- [Quickstart](#-quickstart)
-[Data Setup](#-data-setup)
- [Dashboard](#-dashboard)
- [Dependencies](#-dependencies)
- [References](#-references)

---

## 📊 Results

| Model | RMSPE ↓ | R² ↑ | Notes |
| :--- | :---: | :---: | :--- |
| **GraphSAGE** (correlation graph) | 0.29448 | 0.7714 | GNN on dynamic 112-stock graph |
| **LightGBM** + NN features (K=10) | 0.27873 | 0.8357 | 1st place Kaggle trick |
| **Ensemble** (LGB×0.62 + GNN×0.38) | 0.26919 | 0.8380 | Optimal weighted blend |
| **Stacking meta-learner** | 0.23410 | 0.8770 | Ridge on [LGB + NN_v2 + cross-stock] |
| **LightGBM + NN K=50 + cross-stock** | **0.22042** | **0.8863** | **Final model — 74 features** |
| *Kaggle 1st place (nyanp)* | *~0.179* | *—* | *Full test set + 20+ model ensemble* |

*Each improvement step is fully reproducible and documented below.*

---

## 🏗 Architecture

```text
                        Raw LOB Data (Optiver)
                                  │
                                  ▼
    ┌────────────────────────────────────────────────────────────┐
    │ Feature Engineering                                        │
    │  - LOB microstructure: spread, imbalance, depth, VWAP      │
    │  - Realized volatility: full / first30s / last30s          │
    │  - Trade flow: volume, count, price impact, order count    │
    │  - Time-ID NN features (K=50, dual metric) ← Kaggle #1     │
    │  - Cross-stock features: top-10 correlated peers           │
    └──────────────────────┬─────────────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
      ┌─────────────┐          ┌──────────────────────────┐
      │ LightGBM    │          │ GraphSAGE                │
      │ 74 features │          │ Dynamic correlation      │
      │ 5-fold CV   │          │ graph (window=50)        │
      │ Target      │          │ 112 nodes, ~12k edges    │
      │ ratio       │          │ 2-layer + residual       │
      │ transform   │          │                          │
      └──────┬──────┘          └────────────┬─────────────┘
             │                              │
             └──────────────┬───────────────┘
                            ▼
                   Optimal Ensemble
              w_LGB=0.623 | w_GNN=0.377
                     RMSPE = 0.269
                            │
                            ▼
                 Ridge Meta-Learner (L2)[lgb_pred + nn_v2 + cross-stock]
                     RMSPE = 0.220
                            │
                            ▼
              ┌─────────────────────────┐
              │   Systemic Risk Layer   │
              │   PageRank + Strength   │
              │   + Eigenvector         │
              │   Hub: Stock 67,20,123  │
              └─────────────────────────┘
```

---

## 📁 Project Structure

```text
QUANT/
├── data/
│   ├── raw/                             # Optiver raw data (not tracked by git)
│   └── processed/
│       ├── features.parquet             # LOB features (428k rows, 27 cols)
│       ├── features_with_nn.parquet     # + NN K=10 features (33 cols)
│       ├── features_with_nn_v2.parquet  # + NN K=50 dual metric (50 cols)
│       ├── features_with_cross.parquet  # + cross-stock features (55 cols)
│       ├── systemic_risk.parquet        # Centrality scores (time_id, stock_id)
│       ├── hub_stocks.csv               # Systemic importance ranking
│       ├── lgb_oof.parquet              # LightGBM OOF predictions
│       ├── lgb_test.parquet             # LightGBM test predictions
│       └── meta_test.parquet            # Meta-learner test predictions
│
├── src/
│   ├── metrics.py                       # Shared metric functions (RMSPE, R², MAE)
│   ├── clearing.py                      # Eisenberg-Noe (2001) clearing mechanism
│   ├── risk_measures.py                 # Aggregate Shortfall, DebtRank, MBC
│   ├── network_generator.py             # Synthetic financial network generators
│   ├── dataset.py                       # PyG dataset builder for systemic risk GNN
│   │
│   ├── features/
│   │   ├── lob_features.py              # LOB + trade feature engineering
│   │   ├── nn_features.py               # Time-ID NN features K=10
│   │   ├── nn_features_v2.py            # Time-ID NN features K=50, dual metric
│   │   ├── cross_stock_features.py      # Cross-stock correlation features
│   │   └── correlation_graph.py         # Dynamic PyG graph builder
│   │
│   ├── models/
│   │   ├── baseline.py                  # LightGBM 5-fold + target ratio
│   │   ├── gnn_volatility.py            # GraphSAGE for node-level vol prediction
│   │   ├── gnn_sage_gat.py              # GraphSAGE + GAT (pure PyTorch, no MessagePassing)
│   │   ├── penn.py                      # PENN (Gonon et al. 2024) for systemic risk
│   │   ├── mlp_baseline.py              # Graph-agnostic MLP baseline
│   │   └── stacking.py                  # Ridge meta-learner (L2 stacking)
│   │
│   └── analysis/
│       └── systemic_risk.py             # Systemic importance + contagion matrix
│
├── experiments/
│   ├── compare_models.py                # Ensemble optimisation + final table
│   ├── train_gnn_vol.py                 # GNN volatility training (SAGE on correlation graph)
│   ├── topology_generalisation.py       # OOD test: ER-trained model on CP/BA topologies
│   └── mixed_topology_training.py       # Mixed-topology training (SAGE vs MLP)
│
├── scripts/
│   ├── train_sage_as.py                 # Train GraphSAGE on Aggregate Shortfall
│   ├── train_penn_dr.py                 # Train PENN on Default Rate
│   └── train_mlp_smoke.py              # MLP smoke test (quick sanity check)
│
├── tests/
│   ├── test_clearing.py                 # Tests for Eisenberg-Noe clearing
│   ├── test_risk_measures.py            # Tests for AS, DebtRank, MBC
│   └── test_network_generator.py        # Tests for network generators
│
├── dashboard/
│   └── app.py                           # Streamlit interactive dashboard
│
├── train.py                             # Shared training loop for systemic risk GNNs
│
├── results/
│   ├── contagion_matrix.png
│   └── systemic_timeseries.png
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🔬 Key Technical Contributions

### 1. Time-ID Nearest Neighbour Features
Replication and extension of the **1st place Kaggle solution** (nyanp, 2021).
For each `time_id`, we find the `K=50` most similar market sessions using a **dual metric KNN** (Canberra + Euclidean) on a pivot table of shape `(time_id × stock_id)`. The realized volatility and targets of neighbouring sessions become additional features — capturing the **market regime** of each day.

```python
# Core idea
pivot   = features_df.pivot_table(index="time_id", columns="stock_id", values="rv_full")

nn_canb = NearestNeighbors(n_neighbors=50, metric="canberra").fit(scaler.fit_transform(pivot))
nn_eucl = NearestNeighbors(n_neighbors=50, metric="euclidean").fit(scaler.fit_transform(pivot))

# Features: weighted mean, std, p10/p25/p75, distance ratio — per both metrics
```

**Top feature importances (LightGBM, final model):**

| Feature | Importance | Description |
| :--- | :---: | :--- |
| `cs_tgt_mean` | 5154 | Mean target of top-10 correlated stocks |
| `cs_rv_mean` | 3606 | Mean RV of top-10 correlated stocks |
| `rv_full` | 2934 | Current session realized volatility |
| `nn_rv_mean` | 1756 | Mean RV of K=50 nearest time sessions |
| `rv_last30` | 1747 | RV of the last 30 seconds of session |
| `cs_corr_mean` | 1601 | Avg Pearson correlation with peer stocks |
| `nn_dist_1_e` | 1587 | Euclidean distance to nearest session |

### 2. Cross-Stock Correlation Features
Novel tabular features that make the stock correlation structure **explicit** for LightGBM — the same information that GraphSAGE learns implicitly through message passing.

For each `(stock_id, time_id)`, we find the top-10 most correlated stocks on a rolling 50-session window and compute:

```python
cs_rv_mean   = mean(rv_full of top-10 correlated stocks)
cs_rv_std    = std(rv_full of top-10 correlated stocks)
cs_rv_max    = max(rv_full of top-10 correlated stocks)
cs_tgt_mean  = mean(target of top-10 correlated stocks)
cs_corr_mean = mean(Pearson correlation with top-10 peers)
```
> 💡 This single addition dropped RMSPE from **0.279 → 0.220** (+0.059), the largest single improvement in the entire pipeline. `cs_tgt_mean` became the most important feature overall.

### 3. Dynamic Correlation Graph + GraphSAGE
Each `time_id` has its own **correlation graph** built on a rolling window of 50 sessions. Stocks are nodes; edges connect stocks with Pearson correlation `> 0.3`.

*   **112 nodes** (stocks), **~12,400 edges** per graph
*   **GraphSAGE** *(Hamilton et al. 2017)* — 2 layers, hidden=128, residual connections
*   **Node features:** 20-dimensional LOB microstructure vector
*   **Target:** normalized realized volatility (`y_mean`/`y_std` denormalization)
*   Contributes **orthogonal signal** to LightGBM — optimal blend weight 0.38

### 4. Systemic Risk Layer
Novel contribution: **network centrality metrics** applied to the volatility correlation graph to identify which stocks propagate volatility shocks most strongly across the market.

**Composite systemic importance score:**
$$ \text{Systemic Score} = 0.4 \times \text{PageRank}_{w} + 0.3 \times \text{Strength}_{w} + 0.3 \times \text{Eigenvector centrality} $$

Top hub stocks: **67, 20, 123, 17, 104** — consistently central across market regimes. High `std_score` (~0.15) confirms systemic importance is **regime-dependent**.

> 💡 **Note:** DebtRank *(Battiston 2012)* was tested but saturates on dense equity correlation graphs (12k+ edges per graph). The composite centrality approach is more appropriate for fully-connected financial networks.

### Systemic Risk Simulation Layer

Separate from the Optiver pipeline, the project includes a **full financial network simulation** framework for studying systemic risk with GNNs on synthetic interbank networks:

- **Eisenberg-Noe clearing** (`src/clearing.py`): fixed-point and fictitious default solvers for computing clearing payment vectors in interbank liability networks.
- **Risk measures** (`src/risk_measures.py`): Aggregate Shortfall, DebtRank (Battiston et al. 2012), and Minimum Bailout Capital via stochastic optimization.
- **Network generators** (`src/network_generator.py`): three topology types (Erdos-Renyi, Core-Periphery, Barabasi-Albert) with lognormal liabilities and leverage-scaled assets.
- **PyG dataset builder** (`src/dataset.py`): generates synthetic networks, computes risk labels, packages as PyG Data objects.
- **PENN model** (`src/models/penn.py`): Permutation-Equivariant Neural Network from Gonon et al. 2024, the state-of-the-art for this task.
- **Topology generalization experiments** (`experiments/topology_generalisation.py`, `experiments/mixed_topology_training.py`): tests whether GNNs trained on one network topology generalize to unseen topologies.

The simulation layer has its own test suite (`tests/test_clearing.py`, `tests/test_risk_measures.py`, `tests/test_network_generator.py`) with analytically verified test cases.

### 5. Target Ratio Transformation
Instead of predicting `rv_target` directly, we predict the ratio:

$$ \text{Target Ratio} = \frac{\text{rv\_target}}{\text{rv\_current}} $$

This stabilises the regression target (ratio $\in[0.1, 10]$ vs absolute values spanning 3 orders of magnitude) and improves LightGBM convergence. Clipped to `[0.1, 10.0]` to handle outlier sessions.

### 6. Optimal Ensemble + Ridge Stacking
**Level 1** — optimal weighted blend via `scipy.optimize.minimize` on RMSPE:

```python
def loss(w):
    return rmspe(y_true, w * pred_lgb + (1 - w) * pred_gnn)

result = minimize(loss, x0=[0.5], bounds=[(0.0, 1.0)])
# → w_LGB = 0.623, w_GNN = 0.377
```

**Level 2** — Ridge meta-learner trained on OOF predictions with GroupKFold on `time_id`:

```python
meta_features =[lgb_pred, rv_full, nn_tgt_wmean_c, nn_tgt_wmean_e,
                 nn_dist_ratio_c, nn_dist_ratio_e,
                 cs_rv_mean, cs_rv_std, cs_tgt_mean, cs_corr_mean]
                 
model = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100])
```

---

## 🚀 Quickstart

```bash
# 0. Clone and install
git clone https://github.com/YOUR_USERNAME/QUANT.git
cd QUANT
pip install -r requirements.txt

# 1. Feature engineering — base LOB features
python src/features/lob_features.py

# 2. Time-ID nearest neighbour features
python src/features/nn_features.py          # K=10 (fast)
python src/features/nn_features_v2.py       # K=50, dual metric (~7 min)

# 3. Cross-stock correlation features
python src/features/cross_stock_features.py # (~20 sec)

# 4. Train GNN
python src/models/gnn_volatility.py

# 5. Train LightGBM on full 74-feature set
python src/models/baseline.py               # RMSPE 0.220

# 6. Ensemble + comparison
python experiments/compare_models.py        # RMSPE 0.269 (LGB+GNN blend)

# 7. Ridge meta-learner stacking
python src/models/stacking.py               # RMSPE 0.222

# 8. Systemic risk analysis
python src/analysis/systemic_risk.py        # hub stocks + contagion matrix

# 9. Launch dashboard
streamlit run dashboard/app.py
```

---

## 📂 Data Setup

The raw Optiver data is **not included** in this repository (Kaggle competition rules).
To reproduce:

1. Download from [Kaggle — Optiver Realized Volatility Prediction](https://www.kaggle.com/c/optiver-realized-volatility-prediction/data)
2. Place files in `data/raw/` with the original structure:

```text
data/raw/
├── book_train.parquet/
├── trade_train.parquet/
└── train.csv
```

3. Run the pipeline from Step 1 above — all processed files are generated automatically.

**Dataset stats:**
*   **112** anonymous US equity stocks
*   **3,830** ten-minute trading sessions (`time_id`)
*   **Level 2 order book:** 10 bid/ask levels at 1-second resolution
*   **Trade data:** price, size, order count
*   `428,932` rows × `55` raw features → `74` engineered features
*   **Train:** `364,648` rows (85% of time_ids) | **Test:** `64,284` rows

---

## 📈 Dashboard

The Streamlit dashboard provides three interactive views:

**🔮 Volatility Predictions**
*   Interactive network graph: nodes = stocks (colour = prediction error), edges = top-30 strongest correlations
*   Predicted vs actual scatter plot per session
*   Largest prediction errors table

**⚠️ Systemic Risk**
*   Systemic importance ranking (all 112 stocks)
*   Time series of top-5 hub stocks across 3,830 sessions
*   Volatility contagion matrix heatmap

**📊 Model Performance**
*   RMSPE / R² / MAE comparison table
*   RMSPE distribution by stock
*   RMSPE over time (test window)
*   LightGBM feature importance (top-20)

```bash
streamlit run dashboard/app.py
# → http://localhost:8501
```

---

## 🛠 Dependencies

Install all dependencies using the provided requirements file:

```bash
pip install -r requirements.txt
```

*(Key libraries: `lightgbm>=4.0`, `torch>=2.0`, `torch-geometric>=2.3`, `networkx>=3.1`, `streamlit>=1.30`, `plotly>=5.18`, `pandas>=2.0`, `numpy>=1.24`, `scikit-learn>=1.3`, `seaborn>=0.13`, `scipy>=1.11`, `pyarrow`, `tqdm`)*

---

## 📚 References

1. **nyanp** (2021). [*1st Place Solution — Nearest Neighbors*](https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/274970). Kaggle Optiver Realized Volatility Prediction.
2. **Hamilton et al.** (2017).[*Inductive Representation Learning on Large Graphs*](https://arxiv.org/abs/1706.02216). NeurIPS 2017.
3. **Battiston et al.** (2012).[*DebtRank: Too Central to Fail?*](https://www.nature.com/articles/srep00541) Scientific Reports 2, 541.
4. **Cao et al.** (2021). [*Multivariate Realized Volatility Forecasting with Graph Neural Network*](https://arxiv.org/abs/2112.09015). arXiv:2112.09015.
5. **pksha** (2021).[*3rd Place Solution — life is volatile*](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/writeups/pksha-life-is-volatile-tentative-3rd-place-s). Kaggle Optiver Realized Volatility Prediction.
6. **Eisenberg & Noe** (2001). [*Systemic Risk in Financial Systems*](https://doi.org/10.1287/mnsc.47.2.236.9835). Management Science, 47(2), 236-249.
7. **Gonon et al.** (2024). [*Computing Systemic Risk Measures with Graph Neural Networks*](https://arxiv.org/abs/2410.07222). arXiv:2410.07222.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details. 
*Note: The Optiver dataset is subject to [Kaggle competition rules](https://www.kaggle.com/c/optiver-realized-volatility-prediction/rules) and is not redistributable.*