# SONAR — Supply-chain Optimization and Network Analysis for Resilience

A dynamic decision-support tool that simulates shipping disruptions and tariff shocks, identifies resilient rerouting options, and quantifies route risk using a novel Resilience Score — powered by UNCTAD maritime data and XGBoost.

---

## Project Overview

Global shipping is defined by two persistent threats: **sudden geopolitical disruptions** (wars, chokepoint closures) and **trade policy shocks** (tariffs, sanctions). Most businesses rely on a static "Plan A" route, leaving them exposed to cost spikes and delays.

SONAR lets users ask *"What if the Suez Canal closes tomorrow?"* or *"What happens to my China→Germany electronics route if the US imposes a 25% tariff?"* — and receive an optimal rerouted path, cost premium, lead time impact, and a Resilience Score in seconds.

### Key Results

| Metric | Value |
|---|---|
| Trade corridors modelled | 375,522 (origin × destination × product × year) |
| Countries in routing graph | 222 |
| Missing freight rates imputed by ML | 136,023 |
| XGBoost improvement over baseline | +19.1% RMSE reduction |
| Top predictive feature | `historical_mean_rate` (bilateral history) |
| #1 novel contribution | Resilience Score formula combining redundancy, connectivity, chokepoint exposure & fleet availability |
| Tests passing | 31 / 31 |

---

## Dataset

Source: [UNCTAD Trade & Development Datahub](https://unctadstat.unctad.org/datacentre/)

| File | Rows | Description |
|---|---|---|
| `transport_cost_by_product.csv` | 62,587 | **Core** — ad-valorem freight rates by origin→destination×product, 2016–2021 |
| `bilateral_shipping_connectivity_index.csv` | 1,146 | Pairwise LSCI connectivity between country pairs, quarterly |
| `country_shipping_connectivity_index.csv` | 185 | Country-level LSCI index, Q1 of each year |
| `container_port_throughput.csv` | 137 | TEU container volumes by country, 2016–2021 |
| `merchant_fleet.csv` | 207 | % of world merchant fleet by country, 2016–2021 |
| `seaborne_trade.csv` | 479 | Trade volumes (loaded/discharged) in kt, 2016–2021 |
| `vessel_percent_of_global_fleet.csv` | 167 | Fleet % by flag of registration, 2019–2021 |

**5 HS Product Codes modelled:**
- `8517` — Telephones & Electronics
- `2106` — Dried Food Preparations
- `3304` — Cosmetics & Toiletries
- `9404` — Mattresses & Household Goods
- `6109` — Clothing (T-shirts etc.)

---

## Architecture

```
Raw CSVs (7 files)
      │
      ▼
Feature Pipeline (375K rows × 25 features)
      │
      ▼
XGBoost Regressor  ──→  Imputed rates for 136K missing corridors
      │
      ▼
NetworkX DiGraph (5 graphs: 2021 × 5 products)
      │
      ▼
Streamlit Dashboard  ←──  Scenario Engine (chokepoints + tariffs)
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place raw data files

Put all 7 CSV files in `data/raw/` (symlinks already set up if running from the IS5126 repo):

```
data/raw/
├── bilateral_shipping_connectivity_index.csv
├── container_port_throughput.csv
├── country_shipping_connectivity_index.csv
├── merchant_fleet.csv
├── seaborne_trade.csv
├── transport_cost_by_product.csv
└── vessel_percent_of_global_fleet.csv
```

### 3. Run the training pipeline

```bash
cd sentinel-trade

# Step 1: Feature engineering + ML training + edge imputation (~2–3 min)
python src/models/train_xgb.py

# Step 2: Build 5 latest-year NetworkX graphs (~5 sec)
python src/graph/builder.py
```

Expected output from Step 1:
```
Train: 159,724  Val: 38,970  Test: 40,805
RMSE: 0.3313  |  Baseline RMSE: 0.4097  |  Improvement: 19.1%
Top feature: historical_mean_rate  (gain: 7.38)
Graph edges saved → data/processed/graph_edges_full.parquet  (375,522 edges)
```

Expected output from Step 2:
```
(2021, 8517) → 222 nodes,  18,020 edges
...
Graphs cached → data/processed/graphs_latest.pkl
```

### 4. Launch the dashboard

```bash
streamlit run app/main.py
```

Open your browser at `http://localhost:8501`.

---

## Dashboard Guide

### Home Page
Shows summary statistics: total corridors, countries, years covered.

### Route Explorer (`01_route_explorer.py`)

The primary simulation page.

**Controls (left sidebar):**

| Control | Description |
|---|---|
| Origin / Destination | Select any of ~222 countries |
| Product | One of 5 HS product codes |
| Chokepoint toggles | Block Suez Canal, Panama Canal, Strait of Hormuz, or Strait of Malacca |
| Tariff sliders | Apply % tariffs to US, EU, China, or ASEAN trade |
| Routes to display | Show top 1 or top 3 alternatives |

**Outputs:**
- **Plotly globe** with baseline route (blue) and rerouted scenario (orange dashed)
- **Results table**: route path, freight rate, lead time, hops, Resilience Score
- **Cost premium callout**: e.g. "+22% cost, +12 days" after Suez blockage
- **Resilience Score gauge** (0–100)
- **Score component breakdown**: Redundancy, Connectivity, Chokepoint, Fleet

### Resilience Analysis (`02_resilience_analysis.py`)

- Heatmap of Resilience Scores across top corridors × products
- Drill-down bar chart showing score components for any corridor
- Apply scenario filters to see how chokepoints/tariffs affect the full network

### Model Explainability (`03_model_explainability.py`)

- **Feature importance (single edge)**: select any corridor to see which features drove the predicted rate
- **Global importance**: XGBoost gain importance across all features
- **Model performance table**: metrics vs baseline with academic context

---

## Demo Script (Presentation)

Run these 5 scenarios in sequence for a compelling industry pitch:

**1. Baseline routing**
> Origin: China | Destination: Germany | Product: Telephones | Year: 2021
> → Optimal route, ~RS 78/100 (High Resilience)

**2. Suez Canal blockage**
> Toggle: ☑ Suez Canal
> → Observe rerouting, cost premium, lead time increase

**3. US–China trade war**
> US Tariff: 25% | China Tariff: 25%
> → Trans-Pacific cost spike; model finds lowest-tariff transshipment path

**4. Double chokepoint stress test**
> Toggle: ☑ Strait of Hormuz + ☑ Strait of Malacca
> → Shows remaining viable paths or "no viable route" message for isolated corridors

**5. Model explainability**
> Navigate to Model Explainability → select the rerouted edge → show which features explain the predicted rate

---

## Resilience Score Formula

The RS is a novel composite 0–100 index integrating four risk dimensions:

```
RS = 100 × (0.47 × Alt + 0.28 × Chk + 0.17 × Bil + 0.07 × Fleet)
```

| Component | Weight | Definition |
|---|---|---|
| **Alt** | 47% | Alternative path redundancy: `max(0, 1 − premium)` where premium = (cost_k2 − cost_k1)/cost_k1. Zero alternatives → Alt = 0. |
| **Chk** | 28% | Chokepoint avoidance: `1 − (chokepoint countries on path / 7 total)` |
| **Bil** | 17% | Bilateral LSCI quality along route edges, normalised by 95th-percentile. |
| **Fleet** | 7% | Average merchant fleet % of transit nations, normalised by global median. |

Sensitivity analysis (±10% weight perturbation) confirms score orderings are stable within ±5 points.

---

## ML Model Details

**Task:** Predict ad-valorem freight rate (cost as % of cargo value) for unobserved (origin, destination, product, year) combinations.

**Target transformation:** `log1p(freight_rate)` — handles right-skewed distribution.

**Train/Val/Test split (temporal — no leakage):**
- Train: 2016–2019 (159,724 observed rows)
- Validation: 2020 (38,970 rows, used for early stopping)
- Test: 2021 (40,805 rows, held-out final evaluation)

**Key features (top 5 by gain):**

| Feature | Source | Gain |
|---|---|---|
| `historical_mean_rate` | transport_cost (pre-2020 group mean) | 7.38 |
| `year_int` | time | 1.99 |
| `product_cat` | HS code | 1.30 |
| `dest_lsci` | country_shipping_connectivity_index | 0.54 |
| `bilateral_lsci` | bilateral_shipping_connectivity_index | 0.26 |

The appearance of `bilateral_lsci` in the top features **validates the decision to join all 7 UNCTAD datasets** — connectivity between the specific origin-destination pair adds genuine predictive signal beyond the route's own history.

**Why is R² moderate (0.34)?**
The imputation task predicts freight rates for routes that have never been observed or were only seen in pre-COVID years. The model still captures meaningful signal as evidenced by the 19.1% RMSE improvement over the naïve median baseline. Extreme outlier rates (up to 98.6% ad-valorem) inflate RMSE; the median absolute error (MAE = 0.093) better reflects typical prediction accuracy.

---

## Chokepoint Reference

| Chokepoint | Node(s) Removed | Affected Trade |
|---|---|---|
| Suez Canal | Egypt | Europe ↔ Asia |
| Panama Canal | Panama | US East Coast ↔ Asia |
| Strait of Hormuz | Iran, Oman | Global energy / Gulf exports |
| Strait of Malacca | Singapore, Malaysia, Indonesia | South China Sea → Indian Ocean |

---

## Running Tests

```bash
pytest tests/ -v
```

All 31 tests should pass:
- `tests/test_feature_pipeline.py` — 10 tests: data integrity, leakage checks, null-fill verification
- `tests/test_routing.py` — 11 tests: Dijkstra correctness, scenario immutability, tariff multipliers
- `tests/test_resilience.py` — 10 tests: score bounds, component logic, sensitivity stability

---

## File Structure

```
sentinel-trade/
├── data/
│   ├── raw/                        # 7 original UNCTAD CSVs (read-only)
│   └── processed/
│       ├── features_long.parquet   # 375K rows × 31 cols (generated)
│       ├── graph_edges_full.parquet# Complete edge matrix with ML imputations (generated)
│       ├── graphs_latest.pkl       # 5 pre-built NetworkX graphs for 2021 (generated)
│       └── model_artifacts/
│           ├── xgb_model.json      # Trained XGBoost model
│           ├── shap_explainer.pkl  # Feature importance data
│           └── normalization_constants.pkl
├── notebooks/
│   ├── 01_eda_and_data_quality.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_ml_model_training.ipynb
│   ├── 04_graph_engine_validation.ipynb
│   └── 05_resilience_score_validation.ipynb
├── src/
│   ├── data/
│   │   ├── loaders.py              # One loader per CSV; applies NAME_CANONICAL
│   │   └── feature_pipeline.py    # Full melt → join → engineer pipeline
│   ├── models/
│   │   ├── train_xgb.py           # Training script (run once)
│   │   └── predictor.py           # Inference wrapper used by app
│   ├── graph/
│   │   ├── builder.py             # Build + cache NetworkX DiGraphs
│   │   ├── routing.py             # Dijkstra, Yen's K-paths, apply_scenario()
│   │   └── chokepoints.py         # CHOKEPOINTS map, tariff multipliers
│   ├── scoring/
│   │   └── resilience.py          # ResilienceScorer class + sensitivity analysis
│   └── viz/
│       └── globe.py               # Plotly globe factory, gauge, heatmap
├── app/
│   ├── main.py                    # Streamlit entry point
│   └── pages/
│       ├── 01_route_explorer.py   # Globe + scenario simulation
│       ├── 02_resilience_analysis.py
│       └── 03_model_explainability.py
├── tests/
│   ├── test_feature_pipeline.py
│   ├── test_routing.py
│   └── test_resilience.py
├── config.py                      # Central constants (paths, names, weights)
└── requirements.txt
```

---

## Team & Contributions

Built as part of IS5126 Applied Analytics, NUS School of Computing.
Data source: UNCTAD Trade & Development Datahub (public domain).
