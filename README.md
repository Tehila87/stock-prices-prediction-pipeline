***# Stock Prices Prediction Pipeline***



Event-aware stock-return prediction around earnings (Ridge/XGBoost) with LOEO validation and gated/decayed features.



***## Optimisation***

The pipeline is optimised for clarity, speed, and credibility—not full reproducibility with heavy data.  

A full version of the pipeline, including data checkpoints and the complete datasets required to run the code end-to-end, is available on request.



***## Structure***

\- `notebooks/01\_main\_modeling\_pipeline.ipynb` – clean, end-to-end run

\- `data/sample/` – tiny demo CSVs (e.g., first 500 rows): `master\_data\_df\_demo.csv`, `ep\_sentiment\_new\_demo.csv`

\- `data/raw/` – full data (local only): `master\_data\_df.csv`, `ep\_sentiment\_new.csv` (available on request)

\- `figures/` – key plots used in the README/report

\- `reports/` – e.g., `final\_report.pdf`

\- `requirements.txt` – package list

\- `.gitignore`– excludes data/artefacts



***## Data***

This repo includes **\*\*small samples\*\*** in `data/sample/` so the main notebook runs quickly.



To run on **\*\*full data\*\***, place the following files locally:

\- `data/raw/master\_data\_df.csv`

\- `data/raw/ep\_sentiment\_new.csv`



Then set the flag in the notebook (e.g., `USE\_SAMPLE = False`).



***## Technical Overview***

The predictive modelling pipeline comprises:

\- **\*\*(a)\*\*** pre-modelling transformations that de-bias, standardise, and prune the feature space to prevent mixed, redundant signals and noise; and

\- **\*\*(b)\*\*** two ready-to-run validation paths: **\*\*Walk-Forward Cross-Validation (WFCV)\*\*** for year-round predictions and **\*\*Leave-One-Earnings-Out (LOEO)\*\*** for earnings-window predictions.



***## Setup***

```bash

pip install -r requirements.txt

\# then launch Jupyter

jupyter lab



