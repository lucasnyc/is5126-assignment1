# Hotel Analytics — Intelligent Analytics Platform

An end-to-end hotel review analytics application built with **Streamlit**, **SQLite**, **scikit-learn**, and **Hugging Face Transformers**.  
The platform transforms raw TripAdvisor-style review data into interactive dashboards that help hotel operators understand guest experience, benchmark market position, and generate AI-powered improvement plans.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Data](#data)
4. [Dashboards](#dashboards)
5. [Notebooks](#notebooks)
6. [Setup & Installation](#setup--installation)
7. [Running the Application](#running-the-application)
8. [Key Technologies](#key-technologies)

---

## Project Overview

This project provides a multi-page Streamlit application for hotel performance analytics. It covers the full analytics pipeline:

| Stage | Description |
|---|---|
| **Data Preparation** | Load raw JSON reviews into a normalised SQLite database |
| **Exploratory Analysis** | Univariate, bivariate, and multivariate visual analysis |
| **Competitive Benchmarking** | Compare a hotel's aspect ratings against top/bottom performers |
| **Performance Profiling** | Track aspect score trends and contributor activity over time |
| **Intelligent Platform** | AI-powered guest experience insights, cluster classification, and LLM improvement plans |

---

## Repository Structure

```
student-name-hotel-analytics/
│
├── app/
│   ├── streamlit_app.py                  # Main entry point — registers all pages
│   └── pages/
│       ├── Variable_Exploration/         # Data schema, distributions, missing values
│       ├── Univariate_Analysis/          # KDE & histogram plots per variable
│       ├── Bivariate_Analysis/           # Scatter, bar, and time series analyses
│       ├── Multivariate_Analysis/        # Correlation heatmap
│       └── Intelligent_Analytics_Platform/
│           ├── Guest_Experience_Dashboard.py      # Rating trends, reviews, aspect analysis
│           ├── Market_Position_Dashboard.py       # Competitive benchmarking & cluster view
│           └── Improvement_Plan_by_LLM.py         # LLM-generated improvement plan
│
├── data/
│   ├── review.json                       # Raw review data (source)
│   ├── reviews_sample.db                 # SQLite database (built from review.json)
│   ├── data_schema.sql                   # DDL for reviews + authors tables
│   ├── db_indexing.sql                   # Index definitions for query performance
│   ├── consolidated_data_variables_info.csv  # Variable metadata
│   ├── cluster_performance.csv           # KMeans cluster peer averages
│   ├── kmeans.pkl                        # Trained KMeans model (3 clusters)
│   └── scaler.pkl                        # StandardScaler fitted on aspect ratings
│
├── notebooks/
│   ├── 01_data_preparation.ipynb         # JSON → SQLite pipeline
│   ├── 02_exploratory_analysis.ipynb     # EDA across all variables
│   ├── 03_competitive_benchmarking.ipynb # Market positioning analysis
│   └── 04_performance_profiling.ipynb    # Temporal trends & profiling
│
├── profiling/
│   ├── code_profiling.txt                # Python profiling results
│   └── query_results.txt                 # SQL query benchmarks
│
├── reports/                              # Generated analysis reports
│
├── requirements.txt                      # Python dependencies
└── README.md
```

---

## Data

All review data is stored in a **SQLite** database (`data/reviews_sample.db`) with two primary tables:

- **`reviews`** — one row per review: `offering_id`, `overall`, `service`, `cleanliness`, `value`, `location_rating`, `sleep_quality`, `rooms`, `title`, `text`, `review_date`, `author_id`
- **`authors`** — reviewer profile: `author_id`, `author_name`, `author_location`, `author_num_reviews`, etc.

The raw source is `data/review.json`. Run `notebooks/01_data_preparation.ipynb` to (re)build the database from scratch.

**Cluster model files** (`kmeans.pkl`, `scaler.pkl`, `cluster_performance.csv`) are pre-trained and stored in `data/`. They classify hotels into three segments:

| Cluster | Description |
|---|---|
| **Bang for Buck** | Strong value-for-money; affordability relative to quality |
| **Location Trap** | Prime location drives ratings; service and comfort lag |
| **Premium Experience** | Top-tier across all aspects |

---

## Dashboards

### 🧠 Guest Experience Dashboard
- **Overall Rating Distribution** — rating histogram with percentile breakdown
- **Top and Bottom Reviews** — highest/lowest rated reviews with text preview
- **Aspect Contribution Over Time** — per-aspect trend lines (all hotels vs selected hotel)
- **Textual Feedback Analytics** — keyword frequency from review text
- **Top Reviewer Details** — most active contributors and their rating patterns

### 🧠 Market Position Dashboard
- **Overall Rating** — monthly/quarterly/yearly average rating trend (all hotels + selected hotel overlay)
- **Position Among Top and Bottom Performers** — horizontal bar charts, aspect profile comparison, and gap analysis scorecard
- **Aspect Contribution Over Time** — dual-line aspect trends with individual mini charts
- **⭐ Hotel Cluster Classification** — K-Means cluster assignment with peer comparison chart and scatter overview

### 🧠 Hotel Improvement Plan (LLM)
- Identifies the hotel's weakest aspect automatically
- Displays the lowest-scoring reviews for that aspect
- Sends a structured prompt to a local Hugging Face model
- Generates a four-section improvement plan: Root Cause Summary, Quick Wins, Medium-Term Improvements, and Success Metrics (KPIs)
- Supports: `google/flan-t5-base`, `google/flan-t5-large`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, `sshleifer/distilbart-cnn-12-6`

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_data_preparation.ipynb` | Parse `review.json`, normalise into SQLite, apply schema and indexes |
| `02_exploratory_analysis.ipynb` | Distributions, missing values, correlations, and summary statistics |
| `03_competitive_benchmarking.ipynb` | Hotel ranking, aspect gap analysis, top/bottom performer profiling |
| `04_performance_profiling.ipynb` | Time-series trends, contributor activity, query performance benchmarks |

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd is5126-assignment1
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r student-name-hotel-analytics/requirements.txt
```

> **Note:** `torch` and `transformers` are only required for the **Improvement Plan** page. If you do not need LLM features, you can skip them — the rest of the app will still work.

### 4. Build the database (if not already present)

Run `notebooks/01_data_preparation.ipynb` to generate `data/reviews_sample.db` from `data/review.json`.

---

## Running the Application

From the **repository root**:

```bash
python -m streamlit run .\student-name-hotel-analytics\app\streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Key Technologies

| Technology | Role |
|---|---|
| [Streamlit](https://streamlit.io) | Multi-page interactive web application |
| [pandas](https://pandas.pydata.org) | Data manipulation and aggregation |
| [matplotlib](https://matplotlib.org) / [seaborn](https://seaborn.pydata.org) | Static charts and visualisations |
| [SQLite](https://www.sqlite.org) | Lightweight embedded database |
| [scikit-learn](https://scikit-learn.org) | K-Means clustering, StandardScaler, CountVectorizer |
| [Hugging Face Transformers](https://huggingface.co/transformers) | Local LLM inference for improvement plans |
| [PyTorch](https://pytorch.org) | Model backend for Transformers |
