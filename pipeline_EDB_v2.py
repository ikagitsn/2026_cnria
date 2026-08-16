"""
Machine-learning pipeline for interior T and RH at the EDB
======================================================================
Exterior temperature gaps are filled from the Open-Meteo Historical
Archive (ERA5) rather than imputed, using the official
openmeteo-requests client (FlatBuffers, HTTP cache, automatic retry).
The reanalysis also supplies exogenous predictors (exterior RH,
radiation, wind, dew point) and is validated against the on-site
sensor before use.

Instrumented site:
    latitude  = 14.7957198
    longitude = -16.9674297
    elevation = 87.42 m

Requirements:
    pip install -r requirements.txt
    (ou : pip install openmeteo-requests requests-cache retry-requests
                     pandas numpy scikit-learn matplotlib)

Usage: python pipeline_EDB_v2.py
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend, writes PNG directly
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

from openmeteo_fetch import fetch_openmeteo_historical, validate_against_observations

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================
# All paths are relative to the script directory, so the pipeline
# behaves identically on Windows, macOS and Linux.
SCRIPT_DIR     = Path(__file__).resolve().parent
CSV_PATH       = SCRIPT_DIR / "sensor_export_1.csv"
OUTPUT_DIR     = SCRIPT_DIR / "outputs"
CACHE_DIR      = SCRIPT_DIR / "data_cache"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Site EDB
SITE_LAT       = 14.7957198
SITE_LON       = -16.9674297
SITE_ELEV      = 87.42

RESAMPLE_FREQ  = "15min"
GAP_THRESHOLD  = pd.Timedelta("6 hours")

HORIZONS_STEPS = [1, 4, 24, 96]
HORIZON_LABELS = {1: "15min", 4: "1h", 24: "6h", 96: "24h"}

CV_N_SPLITS    = 4
N_ESTIMATORS   = 400
LEARNING_RATE  = 0.05
MAX_DEPTH      = 6
RANDOM_STATE   = 42


def load_data(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    # The node also logs an NH3 index and a derived heat index. Neither
    # enters this study: the heat index correlates with the targets at
    # r > 0.93, and the NH3 index has no physical bearing on interior
    # temperature. Both are dropped here rather than filtered later, so
    # that neither can slip into a feature set.
    df = df[["Date", "temperature", "humidity", "temperaturext"]]
    return df


def resample_regular(df, freq=RESAMPLE_FREQ):
    df = df.set_index("Date")
    return df.resample(freq).mean()


def enrich_with_openmeteo(df, lat, lon, elev):
    start_date = df.index.min().strftime("%Y-%m-%d")
    end_date   = df.index.max().strftime("%Y-%m-%d")

    print(f"   Fetching Open-Meteo Archive: {start_date} -> {end_date}")
    om = fetch_openmeteo_historical(
        latitude=lat, longitude=lon, elevation=elev,
        start_date=start_date, end_date=end_date,
        cache_dir=CACHE_DIR,
    )

    metrics = validate_against_observations(
        om, df["temperaturext"], om_var="temperature_2m",
    )
    print(f"   Validation (n={metrics.get('n', 0)} points overlap):")
    for k in ["MAE", "RMSE", "bias_mean", "correlation"]:
        v = metrics.get(k)
        if v is not None:
            print(f"      {k:<15} = {v:+.3f}")

    target_idx = df.index
    om_15min = om.reindex(om.index.union(target_idx)).interpolate("time")
    om_15min = om_15min.reindex(target_idx)

    df = df.copy()
    text_obs = df["temperaturext"]
    df["temperaturext"] = text_obs.fillna(om_15min["temperature_2m"])

    df["om_humidity_ext"]  = om_15min["relative_humidity_2m"]
    df["om_dewpoint_ext"]  = om_15min["dew_point_2m"]
    df["om_shortwave"]     = om_15min["shortwave_radiation"]
    df["om_cloud_cover"]   = om_15min["cloud_cover"]
    df["om_wind_speed"]    = om_15min["wind_speed_10m"]
    df["om_precipitation"] = om_15min["precipitation"]

    return df, metrics


def segment_regimes(df, gap_threshold=GAP_THRESHOLD):
    valid = df.dropna(subset=["temperature"]).copy()
    diffs = valid.index.to_series().diff()
    valid["segment"] = (diffs > gap_threshold).cumsum().values
    df = df.join(valid["segment"])
    return df


def make_features(df):
    df = df.copy()

    h   = df.index.hour + df.index.minute / 60.0
    doy = df.index.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * h / 24)
    df["hour_cos"] = np.cos(2 * np.pi * h / 24)
    df["doy_sin"]  = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"]  = np.cos(2 * np.pi * doy / 365)
    df["is_dry_season"] = df.index.month.isin([11, 12, 1, 2, 3, 4]).astype(int)

    for lag in [1, 4, 8, 24]:
        df[f"text_lag{lag}"] = df["temperaturext"].shift(lag)

    for lag in [1, 4, 8]:
        df[f"hrext_lag{lag}"] = df["om_humidity_ext"].shift(lag)

    for lag in [1, 4, 8]:
        df[f"sw_lag{lag}"] = df["om_shortwave"].shift(lag)

    for lag in [1, 4, 96]:
        df[f"temp_lag{lag}"] = df["temperature"].shift(lag)
        df[f"hr_lag{lag}"]   = df["humidity"].shift(lag)

    for window, name in [(4, "1h"), (24, "6h")]:
        df[f"text_roll_{name}_mean"] = df["temperaturext"].rolling(window).mean()
        df[f"text_roll_{name}_std"]  = df["temperaturext"].rolling(window).std()

    for window, name in [(4, "1h"), (24, "6h")]:
        df[f"sw_roll_{name}_mean"] = df["om_shortwave"].rolling(window).mean()

    return df


def make_targets(df, horizons):
    targets = {}
    for h in horizons:
        targets[f"temp_h{h}"] = df["temperature"].shift(-h)
        targets[f"hr_h{h}"]   = df["humidity"].shift(-h)
    return df, pd.DataFrame(targets, index=df.index)


def fit_predict_quantile(X_tr, y_tr, X_te, alpha):
    model = HistGradientBoostingRegressor(
        loss="quantile", quantile=alpha,
        max_iter=N_ESTIMATORS, learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH, random_state=RANDOM_STATE,
    )
    model.fit(X_tr, y_tr)
    return model.predict(X_te), model


def walk_forward_evaluate(X, y, n_splits=CV_N_SPLITS, baselines=None,
                          ar_cols=None, delta_ref=None):
    """Evaluate the model AND the baselines on exactly the same folds.

    Scoring a baseline on the whole series while the model is scored on
    the test folds makes the two MAEs incomparable, so everything here
    is restricted to the fold's own test index.

    baselines : dict name -> Series indexed like X, already shifted to
                predict y(t+h).
    ar_cols   : columns of X used to fit a linear autoregressive model,
                refitted on each training fold.
    delta_ref : if given (the y(t) series aligned on X), the model learns
                the increment y(t+h) - y(t) instead of the level, and the
                prediction is reconstructed by adding y(t) back. Metrics
                stay on the LEVEL, so they remain comparable with the
                baselines.
    """
    baselines = baselines or {}
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        if delta_ref is None:
            fit_tr, off_te = y_tr, 0.0
        else:
            fit_tr = y_tr - delta_ref.reindex(y_tr.index)
            off_te = delta_ref.reindex(y_te.index).to_numpy()

        p50, _ = fit_predict_quantile(X_tr, fit_tr, X_te, 0.5)
        p05, _ = fit_predict_quantile(X_tr, fit_tr, X_te, 0.05)
        p95, _ = fit_predict_quantile(X_tr, fit_tr, X_te, 0.95)
        p50, p05, p95 = p50 + off_te, p05 + off_te, p95 + off_te

        row = {
            "fold": fold,
            "n_test": len(y_te),
            "MAE": mean_absolute_error(y_te, p50),
            "RMSE": np.sqrt(mean_squared_error(y_te, p50)),
            "coverage_90": float(((y_te >= p05) & (y_te <= p95)).mean()),
            "MPIW_90": float(np.mean(p95 - p05)),
        }

        for name, series in baselines.items():
            pred = series.reindex(y_te.index)
            ok = pred.notna()
            row[f"MAE_{name}"] = (mean_absolute_error(y_te[ok], pred[ok])
                                  if ok.sum() >= 20 else np.nan)

        if ar_cols:
            cols = [c for c in ar_cols if c in X.columns]
            lr = LinearRegression().fit(X_tr[cols], y_tr)
            row["MAE_ar"] = mean_absolute_error(y_te, lr.predict(X_te[cols]))

        scores.append(row)
    return pd.DataFrame(scores)


def evaluate_baseline(y_obs_h, y_pred_baseline):
    mask = y_obs_h.notna() & y_pred_baseline.notna()
    if mask.sum() < 50:
        return np.nan
    return mean_absolute_error(y_obs_h[mask], y_pred_baseline[mask])


def plot_predictions(X, y, title, fname):
    n = len(X)
    cut = int(n * 0.8)
    models = {}
    for q in [0.05, 0.5, 0.95]:
        m = HistGradientBoostingRegressor(
            loss="quantile", quantile=q,
            max_iter=N_ESTIMATORS, learning_rate=LEARNING_RATE,
            max_depth=MAX_DEPTH, random_state=RANDOM_STATE,
        )
        m.fit(X.iloc[:cut], y.iloc[:cut])
        models[q] = m

    idx = X.iloc[cut:].index
    p05 = models[0.05].predict(X.iloc[cut:])
    p50 = models[0.5].predict(X.iloc[cut:])
    p95 = models[0.95].predict(X.iloc[cut:])

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(idx, p05, p95, color="#7f77dd", alpha=0.25, label="Intervalle 90 %")
    ax.plot(idx, y.iloc[cut:].values, color="#16a085", lw=1.3, label="Observe")
    ax.plot(idx, p50, color="#c0392b", lw=1.0, alpha=0.9, label="Predit (mediane)")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / fname, dpi=130)
    plt.close()


def main():
    print("\n" + "=" * 70)
    print(" EDB PIPELINE - with the Open-Meteo Historical Archive")
    print("=" * 70 + "\n")

    print("[1/8] Loading sensor data")
    df = load_data(CSV_PATH)
    print(f"   {len(df):,} obs brutes\n")

    print(f"[2/8] Reechantillonnage a {RESAMPLE_FREQ}")
    df = resample_regular(df)
    print(f"   {len(df):,} regular steps\n")

    print("[3/8] Open-Meteo enrichment (lat=14.7957, lon=-16.9674, elev=87.42m)")
    df, om_metrics = enrich_with_openmeteo(df, SITE_LAT, SITE_LON, SITE_ELEV)
    n_text_filled = df["temperaturext"].notna().sum()
    print(f"   T_ext available after gap-filling: {n_text_filled:,}/{len(df):,}\n")

    print("[4/8] Seasonal regime segmentation")
    df = segment_regimes(df)
    valid = df.dropna(subset=["temperature"])
    print(f"   {valid['segment'].nunique()} segments, {len(valid):,} valid obs\n")

    print("[5/8] Feature engineering (CEB + Open-Meteo)")
    df_feat = make_features(df)

    print("[6/8] Multi-horizon targets")
    df_feat, targets = make_targets(df_feat, HORIZONS_STEPS)
    # "temperature" and "humidity" are the CURRENT values y(t).
    # Excluding them denied the model its most recent observation: its
    # earliest input was y(t-15min), while the persistence baseline uses
    # y(t). No leakage is involved, since the target is y(t+h) and y(t)
    # is known at prediction time. The heat index stays excluded from
    # load, being correlated with the targets at r > 0.93.
    feature_cols = [c for c in df_feat.columns if c != "segment"]
    X_full = df_feat[feature_cols]
    print(f"   {len(feature_cols)} features\n")

    print(f"[7/8] Walk-forward evaluation ({CV_N_SPLITS} folds)")
    print("   Baselines scored on the SAME test folds as the model.")
    print("   persist = y(t); ar = linear regression on the AR lags;")
    print("   seas    = y(t+h-24h), the originally reported baseline.\n")
    print("   MAE level = target y(t+h); MAE delta = target y(t+h)-y(t),")
    print("   prediction reconstructed, metrics on the level.\n")
    print(f"   {'Horizon':<8} {'Cible':<6} {'MAEniv':>7} {'MAEdlt':>7} "
          f"{'persist':>8} {'ar':>7} {'saison':>7} "
          f"{'niv/pers':>9} {'dlt/pers':>9} {'Cov.niv':>8} {'Cov.dlt':>8}")
    print("   " + "-" * 104)

    rows = []
    per_fold = []
    final_models = {}
    for h in HORIZONS_STEPS:
        for var in ["temp", "hr"]:
            target_col = f"{var}_h{h}"
            mask = X_full.notna().all(axis=1) & targets[target_col].notna()
            X = X_full[mask]
            y = targets.loc[mask, target_col]
            if len(X) < 200:
                continue

            y_full = (df_feat["temperature"] if var == "temp"
                      else df_feat["humidity"])
            # Persistence: last observed value, y(t).
            # Seasonal: same time the previous day, y(t+h-24h).
            baselines = {
                "persist": y_full,
                "saison": y_full.shift(96 - h),
            }
            ar_cols = [f"{var}_lag{l}" for l in (1, 4, 96)]

            ref = y_full.reindex(X.index)

            scores = walk_forward_evaluate(X, y, baselines=baselines,
                                           ar_cols=ar_cols)
            # Same protocol, but the model learns the increment.
            scores_d = walk_forward_evaluate(X, y, baselines=baselines,
                                             ar_cols=ar_cols, delta_ref=ref)
            # Per-fold results: some folds straddle the regime change and
            # others do not, and the mean alone hides that spread.
            per_fold.append(scores.assign(horizon=HORIZON_LABELS[h],
                                          target=var, formulation="level"))
            per_fold.append(scores_d.assign(horizon=HORIZON_LABELS[h],
                                            target=var, formulation="delta"))
            mae  = scores["MAE"].mean()
            rmse = scores["RMSE"].mean()
            cov  = scores["coverage_90"].mean()
            mpiw = scores["MPIW_90"].mean()
            mae_d = scores_d["MAE"].mean()
            cov_d = scores_d["coverage_90"].mean()
            m_per = scores["MAE_persist"].mean()
            m_ar  = scores["MAE_ar"].mean()
            m_sai = scores["MAE_saison"].mean()

            def gain(ref_mae, val=None):
                val = mae if val is None else val
                return (100 * (ref_mae - val) / ref_mae
                        if ref_mae and np.isfinite(ref_mae) else np.nan)

            label = HORIZON_LABELS[h]
            print(f"   {label:<8} {var:<6} {mae:>7.3f} {mae_d:>7.3f} "
                  f"{m_per:>8.3f} {m_ar:>7.3f} {m_sai:>7.3f} "
                  f"{gain(m_per):>+7.1f}% {gain(m_per, mae_d):>+8.1f}% "
                  f"{cov:>7.1%} {cov_d:>7.1%}")

            rows.append({
                "horizon": label, "target": var, "n_obs": int(len(X)),
                "MAE": mae, "RMSE": rmse,
                "coverage_90": cov, "MPIW_90": mpiw,
                "MAE_delta": mae_d, "RMSE_delta": scores_d["RMSE"].mean(),
                "coverage_90_delta": cov_d,
                "MPIW_90_delta": scores_d["MPIW_90"].mean(),
                "MAE_persistence": m_per, "MAE_linear_AR": m_ar,
                "MAE_seasonal_naive": m_sai,
                "gain_vs_persistence_pct": gain(m_per),
                "gain_delta_vs_persistence_pct": gain(m_per, mae_d),
                "gain_vs_AR_pct": gain(m_ar),
                "gain_vs_seasonal_pct": gain(m_sai),
            })
            final_models[(h, var)] = (X, y)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_DIR / "resultats_v2_openmeteo.csv", index=False)
    if per_fold:
        pd.concat(per_fold, ignore_index=True).to_csv(
            OUTPUT_DIR / "resultats_par_fold.csv", index=False)
        print(f"\n   -> {OUTPUT_DIR / 'resultats_par_fold.csv'}")
    print()

    print("[8/8] Figures and feature importance")
    if (4, "temp") in final_models:
        X, y = final_models[(4, "temp")]
        plot_predictions(X, y,
                         "Prediction T interieure h+1h (v2 Open-Meteo) - bande = IC 90 %",
                         "v2_prediction_T_1h.png")
        print("   ok v2_prediction_T_1h.png")

    if (4, "hr") in final_models:
        X, y = final_models[(4, "hr")]
        plot_predictions(X, y,
                         "Prediction HR interieure h+1h (v2 Open-Meteo) - bande = IC 90 %",
                         "v2_prediction_HR_1h.png")
        print("   ok v2_prediction_HR_1h.png")

    if (4, "temp") in final_models:
        X, y = final_models[(4, "temp")]
        cut = int(len(X) * 0.8)
        m = HistGradientBoostingRegressor(
            max_iter=N_ESTIMATORS, learning_rate=LEARNING_RATE,
            max_depth=MAX_DEPTH, random_state=RANDOM_STATE,
        )
        m.fit(X.iloc[:cut], y.iloc[:cut])
        n_sample = min(800, len(X) - cut)
        idx = np.random.RandomState(RANDOM_STATE).choice(
            np.arange(cut, len(X)), size=n_sample, replace=False)
        pi = permutation_importance(m, X.iloc[idx], y.iloc[idx],
                                    n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
        imp = pd.DataFrame({
            "feature": X.columns,
            "importance": pi.importances_mean,
            "std": pi.importances_std,
        }).sort_values("importance", ascending=False).head(15)

        def categorize(name):
            if name.startswith("temp_") or name.startswith("hr_lag"): return "Autoregressive"
            if name.startswith("text_") or name == "temperaturext":   return "Exterior temperature"
            if name.startswith("om_") or name.startswith("hrext_") or name.startswith("sw_"):
                return "Open-Meteo exogenous"
            if name.startswith("hour_") or name.startswith("doy_") or name == "is_dry_season":
                return "Temporal cycles"
            return "Other"
        imp["cat"] = imp["feature"].apply(categorize)
        color_map = {
            "Autoregressive": "#c0392b",
            "Exterior temperature": "#7f77dd",
            "Open-Meteo exogenous": "#16a085",
            "Temporal cycles": "#e67e22",
            "Other": "#888",
        }
        colors = imp["cat"].map(color_map)

        fig, ax = plt.subplots(figsize=(10, 6.5))
        ax.barh(imp["feature"][::-1], imp["importance"][::-1],
                xerr=imp["std"][::-1], color=colors[::-1], alpha=0.85)
        # No title: the caption in the paper carries it.
        ax.set_xlabel("Decrease in $R^2$")
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=v, label=k) for k, v in color_map.items()
                   if k in imp["cat"].values]
        ax.legend(handles=handles, loc="lower right", fontsize=9)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "v2_importance_features.png", dpi=130)
        plt.close()
        imp.to_csv(OUTPUT_DIR / "v2_importance_features.csv", index=False)
        print("   ok v2_importance_features.png")

    print(f"\nAll results written to: {OUTPUT_DIR}\n")
    print("-" * 70)
    print("Summary:")
    print(results_df.to_string(index=False))
    print("-" * 70)


if __name__ == "__main__":
    main()
