# SONAR — Technical Overview
### Supply-chain Optimization and Network Analysis for Resilience

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Data Sources](#2-data-sources)
3. [Feature Engineering Pipeline](#3-feature-engineering-pipeline)
4. [ML Model — Freight Rate Imputation](#4-ml-model--freight-rate-imputation)
5. [Graph Construction](#5-graph-construction)
6. [Routing Engine](#6-routing-engine)
7. [Resilience Score](#7-resilience-score)
8. [Scenario Simulation](#8-scenario-simulation)
9. [Dashboard Guide](#9-dashboard-guide)

---

## 1. System Overview

SONAR is an interactive supply-chain intelligence platform built on UNCTAD maritime data (2016–2021). It answers a core question for procurement and logistics analysts:

> **Given a trade corridor A → B for product P, which shipping route is best — and how vulnerable is that route to real-world disruptions?**

The system is built in five sequential stages:

```
Raw UNCTAD Data
      ↓
Feature Engineering  (7 datasets merged, 25 features)
      ↓
XGBoost Imputation   (fills in unobserved freight rates)
      ↓
Graph Construction   (5 directed weighted graphs: 2021 × 5 products)
      ↓
Routing + Scoring    (Yen's K-shortest + 5-factor Resilience Score)
      ↓
Streamlit Dashboard  (3-page interactive app)
```

---

## 2. Data Sources

All data is sourced from UNCTAD (United Nations Conference on Trade and Development). Seven datasets are used:

| Dataset | File | Key Fields |
|---|---|---|
| Transport Cost by Product | `transport_cost_by_product.csv` | origin, destination, product_code, year, freight_rate |
| Bilateral LSCI | `bilateral_shipping_connectivity_index.csv` | origin, destination, year, bilateral_lsci |
| Country LSCI | `country_shipping_connectivity_index.csv` | country, year, country_lsci |
| Container Port Throughput | `container_port_throughput.csv` | country, year, teu |
| Merchant Fleet | `merchant_fleet.csv` | country, year, fleet_pct |
| Seaborne Trade | `seaborne_trade.csv` | country, year, loaded_kt, discharged_kt |
| Vessel % of Global Fleet | `vessel_percent_of_global_fleet.csv` | country, year, vessel_pct |

**Coverage**: 6 years (2016–2021) × 5 HS product codes × ~200 country-pairs.

**What is LSCI?** The Liner Shipping Connectivity Index measures how well a country is integrated into global liner shipping networks. It considers the number of ships, their container-carrying capacity, the number of services, and the size of the largest vessel deployed. Higher LSCI = better maritime connectivity (Singapore ≈ 557, China ≈ 550, Netherlands ≈ 397, Panama ≈ 204; landlocked countries = 0).

**The 5 HS Product Codes tracked:**

| HS Code | Product Category |
|---|---|
| 8517 | Telephones & Electronics |
| 2106 | Dried Food Preparations |
| 3304 | Cosmetics & Toiletries |
| 9404 | Mattresses & Household |
| 6109 | Clothing (T-shirts etc.) |

---

## 3. Feature Engineering Pipeline

**Script**: `src/data/feature_pipeline.py`

The raw transport cost table is the backbone — it defines which origin–destination–product–year combinations exist. The other six datasets are joined in to enrich each row.

### 3.1 Join Strategy

Each row represents one trade corridor observation `(origin, destination, product_code, year)`. The pipeline builds this table by:

1. Starting from `transport_cost` (observed bilateral freight rates)
2. Left-joining `bilateral_lsci` on `(origin, destination, year)`
3. Left-joining `country_lsci` twice — once for origin, once for destination
4. Left-joining `port_throughput` (TEU) twice — origin and destination
5. Left-joining `merchant_fleet` twice — origin and destination fleet ownership
6. Left-joining `seaborne_trade` (loaded/discharged cargo volume in kt)
7. Left-joining `vessel_pct` (country's share of the global fleet by type)

All joins are `LEFT` to preserve every observation row even when supporting data is absent (e.g. landlocked countries have no bilateral LSCI).

### 3.2 Null Handling

| Feature | Strategy |
|---|---|
| `bilateral_lsci` | Fill 0.0 (no connectivity); add `bilateral_lsci_is_imputed` flag |
| `origin_teu`, `dest_teu` | Fill 0.0; add `*_is_imputed` flags |
| `fleet_pct`, `vessel_pct` | Fill 0.0 |
| `loaded_kt`, `discharged_kt` | Fill 0.0 |
| `origin_lsci`, `dest_lsci` | Fill with year median (not 0 — avoids penalising countries with missing LSCI) |
| `historical_mean_rate` | Fill with product-level median if corridor has no history; add `_is_imputed` flag |

### 3.3 Derived / Engineered Features

| Feature | Formula | Purpose |
|---|---|---|
| `historical_mean_rate` | Mean of 2016–2019 observed rates per (OD, product) | Strongest predictor; encodes the route's baseline pricing |
| `lsci_asymmetry` | `|origin_lsci - dest_lsci|` | Captures imbalanced connectivity (high cost to low-LSCI destinations) |
| `trade_imbalance` | `(origin_loaded - dest_loaded) / (sum + ε)` | Directional cargo imbalance; drives backhaul discount |
| `teu_log_product` | `log1p(origin_teu) × log1p(dest_teu)` | Port capacity interaction on a log scale |
| `fleet_supply` | `origin_fleet_pct + dest_fleet_pct` | Total fleet availability on both ends of the corridor |
| `post_covid` | `1 if year ≥ 2020 else 0` | Binary structural break capturing the 2020–2021 freight rate surge |
| `year_int` | Year as integer | Captures long-run temporal trends |
| `product_cat` | Integer-encoded HS code | Allows product-specific effects |

**Total features passed to XGBoost**: 25 (see `ML_FEATURES` in `config.py`)

### 3.4 Normalization Constants

Several constants computed from the training data are saved to `normalization_constants.pkl` and used later by the Resilience Scorer and routing engine:

| Constant | Used by | Purpose |
|---|---|---|
| `teu_p95` | Ports component | 95th percentile TEU throughput — normalises port capacity to [0, 1] |
| `weather_severity_median` | Weather component | Median weather severity — fallback for countries with no weather data |
| `rel_median` | Reliability component | Median on-time delivery rate — fallback for countries absent from disruption dataset |
| `sec_median_gri` | Security component | Median Geopolitical Risk Index — fallback for countries without GRI data |
| `median_lsci` | Routing engine | Median country LSCI — fallback for lead-time estimation |

---

## 4. ML Model — Freight Rate Imputation

**Script**: `src/models/train_xgb.py`

### 4.1 Why Imputation?

The UNCTAD transport cost dataset has sparse coverage. Many bilateral corridors are not observed in every year — for example, a Bolivian→German electronics corridor may have no recorded freight rate in 2021. Without imputation, these routes would be absent from the graph entirely, creating artificial dead-ends in the network.

The XGBoost model learns patterns from corridors that *are* observed, then fills in rates for those that are not, producing a complete edge matrix.

### 4.2 Target Variable

The model predicts `log1p(freight_rate)`. The log transformation is applied because:
- Freight rates are right-skewed (range: 0.01–~20 USD/kg)
- Log scale prevents large rates (during COVID-19 spikes) from dominating gradient updates
- Predictions are inverse-transformed with `expm1()` back to the original scale

### 4.3 Temporal Train/Val/Test Split

| Split | Years | Purpose |
|---|---|---|
| Train | 2016–2019 | Model learns pre-COVID pricing patterns |
| Validation | 2020 | Early stopping; no hyperparameter tuning on this set |
| Test | 2021 | Final evaluation (held out until model is frozen) |

This is a **temporal split** (not random). Using future data to predict past rates would be data leakage — all `historical_mean_rate` features are computed from 2016–2019 only.

### 4.4 XGBoost Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 800 | Generous upper bound; early stopping handles actual count |
| `max_depth` | 6 | Moderate depth — prevents overfitting on sparse corridors |
| `learning_rate` | 0.05 | Small step size for robust convergence |
| `subsample` | 0.8 | Row subsampling; reduces variance |
| `colsample_bytree` | 0.8 | Feature subsampling per tree |
| `min_child_weight` | 5 | Requires 5 samples to form a leaf — regularises sparse routes |
| `reg_alpha` | 0.1 | L1 regularization |
| `reg_lambda` | 1.0 | L2 regularization |
| `early_stopping_rounds` | 50 | Stops if val RMSE doesn't improve for 50 rounds |

### 4.5 Model Performance (Test Set, 2021)

| Metric | Value | Notes |
|---|---|---|
| RMSE | 0.331 | On original scale; driven by outlier routes |
| MAE | 0.093 | More robust — median error is small |
| R² | 0.339 | Moderate fit; inherently hard task (predicting unobserved corridors) |
| MAPE | 50.3% | High % error on very small rates (0.01–0.05); expected |
| Baseline RMSE | 0.410 | Naïve median-per-product benchmark |
| Improvement | +19.1% | XGBoost meaningfully outperforms the baseline |

The R² is moderate by design: the model is predicting freight rates for corridors that have *never been observed*, using only structural signals (connectivity, port capacity, fleet ownership). This is a much harder task than typical regression on in-distribution data.

### 4.6 Top Predictive Features (XGBoost Gain Importance)

1. `historical_mean_rate` — the route's pre-2020 average rate is the strongest prior
2. `year_int` — captures temporal trends (including COVID freight spike)
3. `dest_lsci` — destination market's shipping integration
4. `bilateral_lsci` — direct connection quality between the two countries
5. `dest_teu` — destination port capacity
6. `product_cat` — product-specific pricing patterns
7. `post_covid` — structural break in 2020–2021
8. `dest_fleet_pct` — fleet availability at destination

The `is_imputed` flags add genuine signal: they tell the model whether the bilateral connectivity data itself is missing (e.g. landlocked countries), which correlates with route difficulty.

### 4.7 Output: Complete Edge Matrix

After training, the model predicts all missing freight rates and concatenates them with the observed rows. The resulting `graph_edges_full.parquet` contains every `(origin, destination, product_code, year)` combination with:
- `freight_rate` — observed or imputed
- `is_predicted` — boolean flag indicating imputed edges
- `bilateral_lsci` — connectivity score for that corridor

---

## 5. Graph Construction

**Script**: `src/graph/builder.py`

### 5.1 Graph Structure

From `graph_edges_full.parquet`, **5 directed weighted graphs** are built — one per product code for `LATEST_YEAR` (2021). Historical years are available for model training but only the most recent year is used for dashboard routing and scoring.

Each graph is a `NetworkX DiGraph` where:

**Nodes** = countries, annotated with:
- `lsci` — country's LSCI score for that year
- `fleet_pct` — country's merchant fleet ownership share
- `lat`, `lon` — geographic centroid (for distance calculations)

**Edges** = bilateral trade corridor, annotated with:
- `weight` = `freight_rate` (observed or ML-imputed) — the routing cost
- `bilateral_lsci` — connectivity quality for that specific corridor
- `is_predicted` — whether this edge's cost was ML-imputed

### 5.2 Graph Scale (Typical, 2021)

- ~120–150 nodes (countries with at least one trade route)
- ~4,000–6,000 directed edges per graph
- Sparse: most country-pairs do not trade the specific product

### 5.3 Caching

All 5 graphs are serialised to `graphs_latest.pkl` after the first build. The Streamlit app loads from cache on startup (~1 second), avoiding graph reconstruction on every session. At startup, resilience scores are pre-computed for all top corridors and stored in `session_state.heatmap_baseline` so the Resilience Analysis page loads instantly.

---

## 6. Routing Engine

**Script**: `src/graph/routing.py`

### 6.1 Core Algorithm: Yen's K-Shortest Paths

The routing engine uses **Yen's K-Shortest Paths** algorithm (via `nx.shortest_simple_paths`), which finds paths in order of increasing total edge weight (freight cost). This is the standard algorithm for finding diverse alternatives on a weighted graph.

However, naive Yen's on this graph produces pathological results: the ML model assigns artificially low freight rates to non-existent corridors (e.g. China → Bolivia → Singapore), so the cheapest "paths" by cost ordering are often geographically absurd phantom routes.

Two mechanisms fix this:

### 6.2 Hub-Qualified Subgraph

Before searching, intermediate nodes are filtered to only **maritime hubs** — countries with LSCI ≥ 100:

```
Hub nodes: LSCI ≥ 100
  Singapore = 557, China = 550, Malaysia = 478, Republic of Korea = 536,
  Netherlands = 397, Panama = 204, Morocco = 202, Philippines = 160, ...

Excluded (LSCI < 100):
  Bahamas = 74, Jamaica = 95, Bolivia = 0, Serbia = 0, Paraguay = 0
```

The routing algorithm only routes *through* hub-qualified countries (endpoints are always included regardless of LSCI). This eliminates landlocked and minor island nations from ever appearing as transshipment stops.

### 6.3 Geographic Detour Filter

A second filter rejects paths with an implausibly large detour:

```
detour_ratio = total_haversine_distance(path) / direct_haversine_distance(origin → destination)

Reject if detour_ratio > 3.0
```

Real transshipment routes (e.g. China → Singapore → Europe) produce ratios of 1.0–2.0. A ratio above 3.0 indicates the algorithm is routing backwards across continents.

### 6.4 Phase-Based Search Strategy

Rather than relying on Yen's cost ordering (which would rank phantoms first), the engine uses a **5-phase explicit hop search**:

| Phase | Graph | Max Hops | Geo Filter | Activates |
|---|---|---|---|---|
| 1 | Hub subgraph | 1 (direct) | — | Always |
| 2 | Hub subgraph | 2 (one hub) | Yes | If Phase 1 < k routes |
| 3 | Hub subgraph | 3 (two hubs) | Yes | If Phase 2 < k routes |
| 4 | Hub subgraph | 3 | No | **Only if Phases 1–3 found nothing** |
| 5 | Full graph | 3 | No | **Only if Phase 4 found nothing** |

This ensures a valid direct route found in Phase 1 is never displaced by a cheaper phantom found later. The fallback phases (4 & 5) only activate for extremely remote corridors with no viable hub-qualified route.

**Why ≤ 2 hops as the default?** UNCTAD (2023) estimates that direct service or a single transshipment hub covers >95% of all containerised trade lanes. Three or more intermediate ports are rare and typically only used for landlocked or island destinations.

### 6.5 Lead Time Estimation

Lead time replaces the previous per-hop heuristic with a physics-based model:

```
lead_time = Σ(haversine_distance(leg_i) / ship_speed) + (intermediate_stops × port_handling)

Where:
  ship_speed    = 666.7 km/day  (15 knots × 24h)
  port_handling = 0.75 days/stop (loading + customs + berthing)
```

Example: China → Germany (direct via Suez)
- Haversine distance ≈ 10,000 km
- Sailing time ≈ 15.0 days
- No intermediate stops
- **Lead time ≈ 15 days**

### 6.6 Multi-Criteria Route Selection

For the Route Explorer, a pool of up to 20 candidate routes is generated, then the best route for each criterion is independently selected:

| Criterion | Selection Logic |
|---|---|
| **Cheapest** | `candidates[0]` — pool is already cost-sorted by Yen's algorithm |
| **Fastest** | `min(candidates, key=lead_time_days)` |
| **Most Resilient** | `max(candidates, key=rs_score)` — each candidate is scored by the RS formula |

The three winners are displayed side-by-side on the globe, coloured green (resilient), blue (cheap), and orange (fast).

---

## 7. Resilience Score

**Script**: `src/scoring/resilience.py`

### 7.1 Formula

```
RS = 100 × (Reliability × Redundancy × Weather × Ports × Security)^0.20
```

Each component is normalised to [0, 1]. With equal exponents (0.20 each), the formula reduces to the **5th root of the product** of all five components. The result maps directly to [0, 100] — no floor adjustment is applied.

### 7.2 Component Definitions

Each factor uses a single real-world input — no internal sub-weights. All components are normalised to [0, 1]; higher = better.

**Reliability (weight: 0.20)**

```
Reliability = mean(on_time_delivery_rate across path countries)
```

Per-country OTD rates from the disruption dataset (10K shipments, 6 trade lanes). Countries not in the dataset use the dataset median OTD rate. Directly answers: *"Does cargo arrive when promised?"*

**Redundancy (weight: 0.20)**

```
Redundancy = mean(LSCI / LSCI_p95 across path countries)
```

LSCI = UNCTAD Liner Shipping Connectivity Index, published annually per country. Measures the number of shipping services, companies, and vessel sizes calling at each country. High LSCI → many carriers → real alternatives exist if the primary service fails. Normalised to the 95th-percentile LSCI in the dataset. Directly answers: *"If this route is disrupted, how many alternatives exist?"*

**Weather (weight: 0.20)**

```
Weather = 1 − mean(weather_severity across path countries)
```

Weather severity is mapped from 48 meteorological condition strings to a [0, 1] scale per IMO/WMO hazard classifications (129K observations, 211 countries). Directly answers: *"How calm is the route?"*

**Ports (weight: 0.20)**

```
Ports = mean(TEU / TEU_p95 across path countries)
```

TEU throughput (UNCTAD, annual by country) proxies port infrastructure size and capacity to absorb demand shocks. Normalised to the 95th-percentile TEU in the dataset. Directly answers: *"Can the ports on this route handle volume without saturating?"*

**Security (weight: 0.20)**

```
Security = 1 − mean(geopolitical_risk_index across path countries)
```

GRI (Geopolitical Risk Index) from the disruption dataset captures conflict, political instability, and trade friction per country. Directly answers: *"How politically stable is the corridor?"*

### 7.3 Aggregation Rationale — Why Geometric Mean?

The additive (arithmetic) model is **compensatory**: a perfect score in Ports and Weather can offset a near-zero score in Security, producing a "Moderate Resilience" result for a route through an active warzone. That misrepresents how supply chains actually fail.

The geometric mean is **non-compensatory**: each component enters as an exponent, so any dimension approaching 0 drives the total score toward 0 regardless of the other four. This accurately models systemic supply chain risk — a catastrophic failure in one node breaks the entire chain.

**Numerical illustration:**

| Scenario | Reliability | Redundancy | Weather | Ports | Security | Additive RS | Geometric RS |
|---|---|---|---|---|---|---|---|
| All strong | 0.90 | 0.88 | 0.85 | 0.92 | 0.87 | 88.4 | 88.4 |
| One critical failure | 0.90 | 0.88 | 0.85 | 0.92 | **0.05** | 72.0 | **42.1** |
| Two weak dimensions | 0.90 | **0.30** | 0.85 | 0.92 | **0.30** | 65.4 | **52.8** |

The geometric mean collapses appropriately when any dimension is critically low; the arithmetic mean masks it as "moderate".

### 7.4 Score Scale

The geometric mean maps naturally to [0, 100] — no floor adjustment is needed:

```
final_score = clip(geometric_mean × 100, 0, 100)
```

| Geometric mean | Score | Meaning |
|---|---|---|
| 1.00 | 100 | All dimensions perfect |
| 0.87 | 87 | All dimensions at typical real-world levels |
| 0.70 | 70 | Noticeably below average across the board |
| any dimension → 0 | → 0 | Critical failure in one dimension |

### 7.5 Score Interpretation

| Range | Label | Meaning |
|---|---|---|
| 75–100 | High Resilience | Strong across all five dimensions — reliable delivery, redundant paths, calm weather, major ports, low geopolitical risk |
| 50–74 | Moderate Resilience | Acceptable overall but with identifiable weak dimensions — monitor actively |
| 25–49 | Low Resilience | One or more dimensions significantly below average — consider route diversification |
| 0–24 | Critical Risk | Multiple dimensions failing simultaneously — immediate rerouting recommended |

---

## 8. Scenario Simulation

**Script**: `src/graph/routing.py → apply_scenario()`

Scenario simulation modifies a copy of the base graph in real-time. The original cached graph is never mutated.

### 8.1 Chokepoint Closure

Blocking a chokepoint removes its associated **maritime waypoints** from the internal waypoint graph and reprices all affected trade edges:

| Chokepoint | Waypoints Blocked | Detour Route |
|---|---|---|
| Suez Canal | `SUEZ_S`, `SUEZ_N` | Cape of Good Hope (`IND_OCEAN_S → CAPE_GOOD_HOPE → S_ATLANTIC`) |
| Panama Canal | `PANAMA_ATL`, `PANAMA_PAC` | Cape Horn (`S_PACIFIC → CAPE_HORN → S_ATLANTIC`) |
| Strait of Hormuz | `HORMUZ` | Arabian Sea re-routing |
| Strait of Malacca | `MALACCA` | Lombok / Sunda Strait alternative |

Country nodes are **not** removed. When Suez is blocked, `SUEZ_S` and `SUEZ_N` are removed from the maritime waypoint graph. `_apply_detour_penalty()` then:
1. Runs Dijkstra on the normal waypoint graph and the blocked waypoint graph for each trade edge
2. Detects which edges' normal maritime path crosses the blocked waypoints
3. Multiplies those edge weights by the detour ratio: `alternative_km / normal_km`

Example: Suez blocked → China→Germany edge × 1.35 (Cape of Good Hope route is ~35% longer). Egypt remains a valid trading partner and transshipment hub — routes transiting it become expensive rather than structurally impossible, reflecting real-world rerouting economics.

### 8.2 Tariff Surcharges

Tariffs are modelled as a cost multiplier on all edges touching countries in the affected region:

```
edge_weight_new = edge_weight_base × max(multiplier_origin, multiplier_destination)

Where: multiplier = 1 + (tariff_pct / 100)
```

For example, a 25% US tariff → all edges touching the United States get weight × 1.25. Routes that avoid the US entirely are unaffected — the routing engine will naturally route around expensive nodes.

**Supported regions:** United States, European Union (27 members), China, ASEAN (10 members).

### 8.3 Combined Scenarios

Chokepoint closures and tariff surcharges can be combined. The graph modification is applied atomically:
1. Reprice edges crossing blocked chokepoint waypoints (detour ratio)
2. Apply tariff multipliers to affected edges

The rerouted result is displayed alongside the baseline, showing cost premium, lead time delta, and whether the path changed.

---

## 9. Dashboard Guide

The Streamlit app has three pages accessible from the sidebar.

---

### Page 1 — Route Explorer (`🗺`)

**What it does:** Finds the optimal shipping route from any origin to any destination for a chosen product, under three independent criteria simultaneously.

**How to use it:**

1. **Set origin and destination** in the sidebar — choose from ~150 countries in the graph
2. **Select a product** (5 HS categories available)
3. Optionally **activate a chokepoint scenario** — tick one or more blocked waterways
4. Optionally **set tariff rates** — adjust US/EU/China/ASEAN tariff sliders
5. The globe and cards update automatically

**What you see:**

- **Globe map** — three coloured routes rendered simultaneously:
  - 🟢 Green = Most Resilient route
  - 🔵 Blue = Cheapest route
  - 🟠 Orange = Fastest route
  - Blocked chokepoints marked with ✕ at the actual strait/canal coordinates (not country centroids)

- **3-column route cards** — one per criterion, each showing:
  - The highlighted metric (RS score / cost / lead time)
  - Full path with hop count
  - Metrics table: freight cost, lead time, hops, chokepoint exposure, RS score, ML-predicted flag
  - Resilience gauge (0–100 dial)
  - Expandable resilience breakdown (component-level contributions)

- **Trade-off summary table** — all three routes side-by-side for direct comparison

**Key questions this answers:**
- *Which single route minimises freight cost?*
- *Which route is fastest assuming 15-knot container ships?*
- *Which route is most resilient to a future disruption?*
- *If I block the Suez Canal, how does the optimal route change and how much more expensive does it get?*
- *With 25% US tariffs, does it still make sense to route through US ports?*

---

### Page 2 — Resilience Analysis (`📊`)

**What it does:** Computes and visualises Resilience Scores across all top global trade corridors simultaneously — a portfolio view rather than a single route.

**How to use it:**

1. **Select products** (multi-select; all 5 by default)
2. **Adjust Top N corridors** slider (10–50 pre-defined corridors by seaborne importance)
3. Optionally **activate chokepoint or tariff scenarios** in the sidebar
4. Scores update automatically

**What you see:**

- **Heatmap** — corridors (rows) × products (columns), colour-coded by RS score from purple (critical) to green (high). Instantly reveals which product-corridor combinations are most exposed.

- **Summary metrics** — four KPIs:
  - Average RS across all shown corridors
  - Count of High Resilience corridors (RS ≥ 75)
  - Count of Critical Risk corridors (RS < 25)
  - Count of corridors with no viable route

- **Detailed scores table** — sortable, with full corridor, product, score, and label

- **Drill-down component bar chart** — select any corridor + product to see exactly how its RS score breaks down into Reliability, Redundancy, Weather, Ports, and Security contributions (max 20 pts each)

**Key questions this answers:**
- *Which of our top trade corridors are most at risk overall?*
- *Which product is most exposed across all corridors?*
- *Under a Strait of Malacca closure, which corridors lose viability entirely?*
- *How does a China tariff change the resilience landscape for electronics vs. clothing?*

---

### Page 3 — Model Explainability (`🔍`)

**What it does:** Explains *why* the XGBoost model predicted a specific freight rate for a given corridor — and validates the model's overall reliability.

**Tab 1 — Single Edge Explanation:**
1. Select origin, destination, product, and year
2. Click "Explain Prediction"
3. A horizontal bar chart shows the top 15 features by XGBoost gain importance for that specific prediction, with the actual feature values shown on hover
4. Indicates whether the rate is observed in UNCTAD data or ML-imputed

**Tab 2 — Global Feature Importance:**
Shows overall XGBoost gain importance across all training data. Use this to understand which signals drive the model systemically (not just for one route). Key finding: `historical_mean_rate` dominates — routes that have historically been expensive tend to remain expensive.

**Tab 3 — Model Performance:**
Documents test-set metrics, design decisions (log-transform rationale, temporal split strategy, COVID structural break), and compares against a naïve median baseline.

**Key questions this answers:**
- *Why is this corridor flagged as expensive — is it because of low LSCI or high historical rates?*
- *Is this freight rate observed data or a model estimate? How confident should I be?*
- *What features does the ML model rely on most heavily?*

---

## Appendix: File Structure

```
sentinel-trade/
├── app/
│   ├── main.py                          # Home page + session state bootstrap
│   └── pages/
│       ├── 01_route_explorer.py         # Multi-criteria routing dashboard
│       ├── 02_resilience_analysis.py    # Corridor heatmap + drill-down
│       └── 03_model_explainability.py   # XGBoost feature importance + metrics
├── src/
│   ├── data/
│   │   ├── loaders.py                   # Raw dataset loaders + name normalisation
│   │   └── feature_pipeline.py          # Feature engineering (25 features)
│   ├── models/
│   │   ├── train_xgb.py                 # XGBoost training + imputation
│   │   └── predictor.py                 # Inference + per-edge explanation
│   ├── graph/
│   │   ├── builder.py                   # Graph construction (5 DiGraphs, 2021)
│   │   ├── routing.py                   # Yen's K-shortest + multi-criteria selection + detour repricing
│   │   └── chokepoints.py               # Chokepoint waypoint definitions + tariff multipliers
│   ├── scoring/
│   │   └── resilience.py                # 5-factor Resilience Score (Reliability/Redundancy/Weather/Ports/Security)
│   └── viz/
│       └── globe.py                     # Plotly Scattergeo visualisations
├── data/
│   ├── raw/                             # 7 UNCTAD CSV files
│   └── processed/
│       ├── features_long.parquet        # Engineered feature table
│       ├── graph_edges_full.parquet     # Complete edge matrix (observed + imputed)
│       ├── graphs_latest.pkl            # 5 serialised NetworkX DiGraphs (2021)
│       └── model_artifacts/
│           ├── xgb_model.json           # Trained XGBoost model
│           ├── shap_explainer.pkl       # Feature importance dict
│           └── normalization_constants.pkl  # RS normalisation constants
├── notebooks/
│   ├── 01_data_exploration.ipynb        # EDA on raw datasets
│   ├── 02_feature_engineering.ipynb     # Feature pipeline walkthrough
│   ├── 03_model_training.ipynb          # XGBoost training + evaluation
│   ├── 04_graph_engine_validation.ipynb # Graph routing + Suez Canal scenario
│   └── 05_resilience_score.ipynb        # RS formula + sensitivity analysis
└── config.py                            # All constants: paths, products, RS weights, chokepoint waypoints
```

---

*SONAR — built on UNCTAD maritime data 2016–2021. All routing decisions use the most recent available year (2021). Freight rates shown are normalised indices, not absolute USD/kg values.*
