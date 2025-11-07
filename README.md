# Stock Prices Prediction Pipeline

Event-aware stock-return prediction around earnings (Ridge/XGBoost) with LOEO validation and gated/decayed features.

## Optimisation

The pipeline is optimised for clarity, reproducibility, and credibility. To keep the repository lightweight, sample data is provided so the notebook can be executed locally without large downloads. Heavy, full-scale datasets are intentionally excluded because some computations and feature-engineering steps are expensive and time-consuming. Additional complementary Python code (scraping / API sourcing, EDA notebooks, and full datasets to run end-to-end) can be shared on request.


## Structure

* notebooks/01_main_modeling_pipeline.ipynb & 02_clustering_pipeline_neighboring_companies.ipynb — clean, end-to-end run
* data/sample/ — tiny demo CSVs (e.g., first 500 rows): master_data_df_demo.csv, ep_sentiment_new_demo.csv
* figures/ — key plots and diagrams supporting the README/report
* reports/ — final report (VPAnalytics_Final_Report_v1.0.pdf)
* methodology/ — full documentation of chosen methodological approaches (Boxes A1–A6)
* requirements.txt — package list
* .gitignore — excludes large artefacts and temporary files

## Data

A summary of data sources (financial, corporate, and macro-economic) is provided in **Box A1** within the methodology/ folder.
This repository includes small **demo samples** in data/sample/ so that the notebook executes quickly and transparently.
If you wish to experiment with full-scale data, custom integrations (e.g., from APIs or proprietary databases) can be adapted to the same schema used in the demo files.

## Technical Overview

The main notebook builds the return-prediction pipeline and includes:
- **(a)** pre-modelling transformations that de-bias, standardise, and prune the feature space to prevent mixed, redundant signals and noise; and
- **(b)** two ready-to-run validation paths:
  - **Walk-Forward Cross-Validation (WFCV)** for year-round predictions, and
  - **Leave-One-Earnings-Out (LOEO)** for earnings-window predictions.

A complementary notebook (02_clustering_pipeline_neighboring_companies.ipynb) identifies structurally similar companies within the NASDAQ-100 universe so that some models can be trained on meaningful peer sets in addition to the focal stock. The flow includes:
- **K-means regime split** – the universe is first split into broad risk / behaviour regimes (e.g. defensive / compounder vs. high-beta / momentum), so peers are selected within comparable regimes.
- **Distance-based peer selection** – within each regime, nearest neighbours are chosen by Euclidean distance in the retained 6-feature, z-scored space, producing stable peer cohorts.

*Before deployment, the candidate neighbours are filtered to technology and communications names, and a fixed top-5 peer set is kept per target (Google, NVIDIA, Apple).*

This peer selection layer is what enables the three modelling paths described in the report:
* **Path A – Peers only**: train on the peer panel only.
* **Path B – Target only**: train only on the focal stock (e.g. Google).
* **Path C – Target + Peers**: train on the union of the target and its peer cohort.

## Figures folder

A selection of supporting visuals is highlighted inside the figures/ folder.
These cover essential layers of the project, including:
1. **Conceptual framing** — 1_problem_breakdown_fishbone_diagram.png
   Shows the decomposition of rapid stock-price moves around earnings into external, market, and company drivers.
2. **Pipeline flow** — 2_predictive_modeling__pipeline_diagram.png
   End-to-end view of the Python modelling pipeline, including pre-modelling transforms and the two validation / prediction paths (WFCV, LOEO).
3. **Feature dynamics** — 3_gated_decayed_features_google_eps.png
   Illustrates how event-aware / decayed features are constructed around earnings to avoid signal leakage and overweighting.
4. **Clustering & Nearest Neighbors** —
   * 4a_neighboring_companies_PCA_scatterplot.png — PCA 2D map positioning Google, Apple, NVIDIA, and their nearest peers.
   * 4a_neighboring_companies_PCA_loadings.png — PCA loadings highlighting the strongest differentiating features between companies.
5. **Model & validation performance** — three companion charts:
   * 5a_directional_hit_rate_across_models_year_round_pred.png — directional accuracy across model variants
   * 5b_performance_metrics_across_models_google_t10.png — tabular / summary view of key metrics for the Google T+10 horizon
   * 5c_walk_forward_xgb_lasts_folds_google.png — walk-forward cross-validation behaviour over the last folds
6. **Predictions vs. actual / time-series** — two application views:
* 6a_actual_vs_predicted_time_series_plot_xgb_pathB_google_t10.png — time-series comparison for Path B XGBoost
* 5b_actual_vs_predicted_around_earnings_google_t10.png — zoomed earnings-window comparison

Together, these figures provide a visual overview of the modelling architecture, validation logic, and predictive patterns.

## Methodology folder

The methodology/ folder contains detailed technical appendices supporting the modelling design.
It documents both the **pre-modelling transformations** and the **two validation paths** (WFCV and LOEO) described above.
These methodological approaches are thoroughly explained and justified in **Boxes A2–A6**, each focusing on a distinct stage of the pipeline.
Refer to the separate methodology/ directory for full details.

## Setup

pip install -r requirements.txt
#/ then launch Jupyter
jupyter lab

## Summary
This repository presents the final modelling pipeline for predicting short-term returns for NVIDIA, Apple, and Google.
It documents event-aware feature engineering, walk-forward validation, and trading strategies (Google, T+10 horizon).





