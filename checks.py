"""Two validity checks on the forecasting setup.

1. Temporal leakage. Every feature at time t must be computable from
   data up to and including t. The test recomputes the feature matrix on
   a truncated series and compares the row at t with the row obtained on
   the full series: if any feature peeked past t, the two rows differ.

2. Uncertainty on the reported gains. Averaging four folds gives a point
   estimate with no spread attached. Here the per-sample absolute errors
   of the model and of each baseline are pooled across the fold test
   sets, and a paired bootstrap gives a 95 % interval on the difference
   in MAE. Pairing matters: model and baseline are scored on the very
   same samples, so resampling them jointly removes the between-sample
   variance that would otherwise dominate.

Writes outputs/gain_intervals.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

import pipeline_EDB_v2 as P

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

N_BOOT = 2000
RNG = np.random.default_rng(P.RANDOM_STATE)


# ----------------------------------------------------------------- 1
def check_leakage(df_feat: pd.DataFrame, probes=(400, 800, 1200, 1600)):
    """Recompute features on truncated history and compare at t."""
    base = P.make_features(df_feat)
    cols = [c for c in base.columns if c != "segment"]

    print("  probe    identical   differing columns")
    ok_all = True
    for t in probes:
        if t >= len(df_feat):
            continue
        trunc = P.make_features(df_feat.iloc[: t + 1])
        a = base[cols].iloc[t]
        b = trunc[cols].iloc[t]
        same = []
        for c in cols:
            va, vb = a[c], b[c]
            if (pd.isna(va) and pd.isna(vb)) or np.isclose(
                    float(va), float(vb), equal_nan=True):
                same.append(True)
            else:
                same.append(False)
        bad = [c for c, s in zip(cols, same) if not s]
        ok_all &= not bad
        print(f"  t={t:<6} {sum(same):3}/{len(cols)}     "
              f"{', '.join(bad) if bad else '-'}")
    return ok_all


# ----------------------------------------------------------------- 2
def per_sample_errors(X, y, baselines, delta_ref):
    """Absolute errors of model and baselines, pooled over test folds."""
    tscv = TimeSeriesSplit(n_splits=P.CV_N_SPLITS)
    err = {"model": []}
    err.update({k: [] for k in baselines})
    for tr, te in tscv.split(X):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        ref_tr = delta_ref.reindex(y_tr.index)
        ref_te = delta_ref.reindex(y_te.index).to_numpy()
        p50, _ = P.fit_predict_quantile(X_tr, y_tr - ref_tr, X_te, 0.5)
        err["model"].append(np.abs(y_te.to_numpy() - (p50 + ref_te)))
        for name, series in baselines.items():
            pred = series.reindex(y_te.index).to_numpy()
            err[name].append(np.abs(y_te.to_numpy() - pred))
    return {k: np.concatenate(v) for k, v in err.items()}


def bootstrap_gain(e_model, e_base, n_boot=N_BOOT):
    """Paired bootstrap CI on the percentage gain in MAE."""
    ok = np.isfinite(e_model) & np.isfinite(e_base)
    a, b = e_model[ok], e_base[ok]
    n = len(a)
    point = 100 * (b.mean() - a.mean()) / b.mean()
    idx = RNG.integers(0, n, size=(n_boot, n))
    gains = 100 * (b[idx].mean(1) - a[idx].mean(1)) / b[idx].mean(1)
    lo, hi = np.percentile(gains, [2.5, 97.5])
    return point, lo, hi


def main() -> None:
    print("=" * 66)
    print(" VALIDITY CHECKS: temporal leakage and uncertainty on gains")
    print("=" * 66)

    df = P.load_data(P.CSV_PATH)
    df = P.resample_regular(df)
    df, _ = P.enrich_with_openmeteo(df, P.SITE_LAT, P.SITE_LON, P.SITE_ELEV)
    df = P.segment_regimes(df)

    print("\n--- 1. TEMPORAL LEAKAGE ---")
    print("  A feature is clean if its value at t is unchanged when the")
    print("  series is truncated at t.\n")
    clean = check_leakage(df)
    print(f"\n  verdict: {'no leakage detected' if clean else 'LEAKAGE'}")

    df_feat = P.make_features(df)
    df_feat, targets = P.make_targets(df_feat, P.HORIZONS_STEPS)
    feature_cols = [c for c in df_feat.columns if c != "segment"]
    X_full = df_feat[feature_cols]

    print("\n--- 2. 95 % BOOTSTRAP INTERVALS ON THE GAINS ---")
    print(f"  paired resampling, {N_BOOT} replicates, pooled test folds\n")
    print(f"  {'horizon':<8} {'target':<6} {'vs':<10} "
          f"{'gain %':>8} {'95 % CI':>18}")
    print("  " + "-" * 56)

    rows = []
    for h in P.HORIZONS_STEPS:
        for var in ("temp", "hr"):
            col = f"{var}_h{h}"
            mask = X_full.notna().all(axis=1) & targets[col].notna()
            X, y = X_full[mask], targets.loc[mask, col]
            if len(X) < 200:
                continue
            y_full = df_feat["temperature"] if var == "temp" \
                else df_feat["humidity"]
            baselines = {"persistence": y_full,
                         "seasonal": y_full.shift(96 - h)}
            err = per_sample_errors(X, y, baselines,
                                    y_full.reindex(X.index))
            for name in baselines:
                g, lo, hi = bootstrap_gain(err["model"], err[name])
                sig = "" if lo <= 0 <= hi else "  *"
                print(f"  {P.HORIZON_LABELS[h]:<8} {var:<6} {name:<10} "
                      f"{g:+8.1f} {f'[{lo:+.1f}, {hi:+.1f}]':>18}{sig}")
                rows.append({"horizon": P.HORIZON_LABELS[h], "target": var,
                             "baseline": name, "gain_pct": round(g, 2),
                             "ci_low": round(lo, 2), "ci_high": round(hi, 2),
                             "excludes_zero": not (lo <= 0 <= hi)})

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "gain_intervals.csv", index=False)
    print("\n  * interval excludes zero")
    print(f"  -> {OUTPUT_DIR / 'gain_intervals.csv'}")


if __name__ == "__main__":
    main()
