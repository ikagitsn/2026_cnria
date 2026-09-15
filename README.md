# Hygrothermal pipeline — reproduction guide

Code and data for the paper *End-to-end IoT-to-ML pipeline for
hygrothermal characterization and prediction of a vernacular CEB+Typha
building in a tropical-sahelian climate* (CNRIA 2026).

Every number, table and figure in the paper is produced by the scripts
below from the two raw sensor exports. Random seeds are fixed
(`random_state=42`), all paths are relative, and matplotlib runs on the
`Agg` backend, so results are identical on Windows, macOS and Linux.

## Install

```
python -m pip install -r requirements.txt
```

Python 3.11 or later. One script reaches the network (Open-Meteo
historical archive); responses are cached in `data_cache/`, so reruns
work offline once the cache is populated.

## Data

| File | Rows | Description |
|---|---|---|
| `sensor_export_1.csv` | 2,237 | sensor export, node `node_01` |
| `sensor_export_2.csv` | 1,548 | second export, strict subset after dedup |

Both are 10-minute exports of interior temperature and relative
humidity, an NH₃ air-quality index, a derived heat index (discarded at
load) and near-envelope exterior temperature. Timestamps are UTC.

## Reproducing the paper

Run from this directory, in any order — each script is independent.

| Command | Produces | Used in |
|---|---|---|
| `python completeness.py` | `outputs/completeness.{csv,tex}`, `seasonal_summary.{csv,tex}` | Table I, Sec. III-B and IV-A |
| `python regen_acquisition.py` | `outputs/fig_acquisition.pdf` | Fig. 1 |
| `python regen_pipeline.py` | `outputs/fig_pipeline.pdf` | Fig. 2 |
| `python regen_heatmap.py` | `outputs/heatmap_calendaire_broken.png` | Fig. 3 |
| `python regen_damping.py` | `outputs/amortissement_CEB.png`, `peak_lags.csv` | Fig. 4, Sec. IV-B |
| `python pipeline_EDB_v2.py` | `outputs/v2_importance_features.png`, `resultats_v2_openmeteo.csv`, `resultats_par_fold.csv` | Fig. 5, Table II, Sec. IV-D and IV-E |
| `python checks.py` | `outputs/gain_intervals.csv` | Sec. III-F and IV-D |
| `python ablation.py` | `outputs/ablation.csv`, `within_regime.csv` | Sec. IV-E |

`pipeline_EDB_v2.py` is the only one that needs the network on first
run. It takes a few minutes: it fits quantile models at three quantiles,
for two targets, across four horizons and four walk-forward folds, in
both the level and the increment formulation.

### Rebuilding the PDF

`paper.tex` and `refs.bib` are included. Run the eight commands above
first, so that `outputs/` holds the five figures, then:

```
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper && pdflatex paper
```

Four `pdflatex` passes are needed, not the usual three: `hyperref`
requires one more before the citation labels settle. `\graphicspath`
already looks inside `outputs/`, so no file needs moving.

### Headline numbers

| Quantity | Value | Source |
|---|---|---|
| Consolidated observations | 2,237 | `completeness.py` |
| Overall completeness | 37.2 % | `completeness.py` |
| Damping ratio σ_int/σ_ext | 0.54 | `regen_damping.py` |
| Median peak lag | +1.75 h | `regen_damping.py` |
| Diurnal amplitude reduction | 52.5 % | `regen_damping.py` |
| MAE, interior T at 15 min | 0.09 °C | `pipeline_EDB_v2.py` |
| MAE, interior T at 1 h | 0.23 °C | `pipeline_EDB_v2.py` |

## Notes on method

**Increment formulation.** The models predict `y(t+h) - y(t)` and the
level is reconstructed by adding `y(t)`. All metrics are computed on the
reconstructed level, so they stay comparable with the baselines.
Regressing the level directly is markedly worse — the comparison is
printed by the same script.

**Baselines.** Persistence `y(t)`, a linear autoregressive model refitted
on each training fold, and the seasonal naive `y(t+h-24h)`. All three are
scored on exactly the same walk-forward test folds as the model.

**Validity checks.** `checks.py` verifies two properties rather than
assuming them. It tests for temporal leakage by recomputing every
feature on the series truncated at `t` and comparing it with the value
obtained on the full series — all 36 features are identical, so none
reads past `t`. It then attaches a 95 % interval to each reported gain
by pooling per-sample absolute errors across the fold test sets and
resampling model and baseline jointly (paired bootstrap, 2,000
replicates). Three of the eight gains do not clear zero; the table in
`gain_intervals.csv` flags which.

`ablation.py` adds two more, both aimed at the weakness of permutation
importance under correlated predictors. It refits the model with each
feature group removed — dropping the single 6-hour rolling mean of
exterior temperature costs more accuracy than dropping all fourteen
Open-Meteo exogenous variables — and recomputes the ranking separately
inside the rainy and the dry block, so that a predictor which merely
encodes the season cannot masquerade as an informative one.

**Excluded variables.** The node records five channels; two are dropped
at load and never reach a feature set. The derived heat index correlates
with the targets at r > 0.93, and the NH₃ air-quality index has no
physical bearing on interior temperature. Dropping them at load rather
than filtering later is deliberate: it makes their exclusion impossible
to undo by accident downstream.

## What is not in this repository

`outputs/` and `data_cache/` are excluded on purpose (see `.gitignore`).
Nothing is lost: every script recreates its own directory on start, and
the eight commands above regenerate all figures, tables and CSVs from the
two exports. `data_cache/` only holds a SQLite cache of Open-Meteo
responses; the archive API is deterministic for past dates, so a rebuilt
cache returns the same values.

## Layout

```
consolidate_data.py    merge and deduplicate the exports
openmeteo_fetch.py     ERA5-Land retrieval and validation
completeness.py        data completeness audit
pipeline_EDB_v2.py     features, models, walk-forward evaluation
checks.py              leakage test and bootstrap intervals
ablation.py            feature ablation and within-regime ranking
regen_acquisition.py   Fig. 1
regen_pipeline.py      Fig. 2
regen_heatmap.py       Fig. 3
regen_damping.py       Fig. 4
common.py              shared palette and daily aggregation
run_all.py             runs all of the above in order
paper.tex, refs.bib    manuscript source
outputs/               created on first run, not versioned
```

`run_all.py` executes the whole chain; `--only <key>` and
`--skip <key>` select individual steps.
