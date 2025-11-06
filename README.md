# Stock Prices Prediction Pipeline

Event-aware stock-return prediction around earnings (Ridge/XGBoost) with LOEO validation and gated/decayed features.

## Optimisation

The pipeline is optimised for clarity, reproducibility, and credibility. To keep the repository lightweight, sample data is provided so the notebook can be executed locally without large downloads. Heavy, full-scale datasets are intentionally excluded because some computations and feature-engineering steps are expensive and time-consuming.
Complementary Python assets — including a Nasdaq-100 clustering pipeline, API-based scraping/sourcing pipes, exploratory notebooks, and the complete datasets required to run the project end-to-end — are available on request.

## Structure

* notebooks/01_main_modeling_pipeline.ipynb — clean, end-to-end run
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

The predictive modelling pipeline comprises:
- **(a)** pre-modelling transformations that de-bias, standardise, and prune the feature space to prevent mixed, redundant signals and noise; and
- **(b)** two ready-to-run validation paths:
  - **Walk-Forward Cross-Validation (WFCV)** for year-round predictions and
  - **Leave-One-Earnings-Out (LOEO)** for earnings-window predictions.

## Figures folder

A selection of supporting visuals is highlighted inside the figures/ folder.
These cover essential layers of the project, including:
- **Conceptual framing** — problem_breakdown_fishbone_diagram.png
- **Pipeline flow** — predictive_modeling_pipeline_diagram.png
- **Feature dynamics** — gated_decayed_features_google_eps.png
- **Model performance** — actual_vs_predicted, directional_hit_rate, performance_metrics
- **Time series / validation visuals** — walk_forward_xgb_last_folds_google.png
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
This report presents the final modelling pipeline for predicting short-term returns for NVIDIA, Apple, and Google.
It documents event-aware feature engineering, walk-forward validation, and trading strategies (Google, T+10 horizon).


