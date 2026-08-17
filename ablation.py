"""Ablation and within-regime tests on the feature set.

Permutation importance is unreliable when predictors are correlated, and
a variable that separates the two measurement windows can look important
merely by acting as a season indicator. Two tests address this.

1. Ablation. Feature groups are removed one at a time and the model is
   refitted; a group that carries information costs accuracy when it
   goes. This is a direct check, immune to the correlation problem that
   affects permutation importance.

2. Within-regime importance. The ranking is recomputed separately on the
   rainy and the dry block. A predictor that only encodes which season
   the sample belongs to loses its rank once the regime is held fixed;
   one that carries physics keeps it.

Both are applied to interior temperature at h+1h, the model the
importance figure describes, under the increment formulation used in the
paper. Writes outputs/ablation.csv and outputs/within_regime.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

import pipeline_EDB_v2 as P

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

HORIZON = 4          # 1 h, the model the importance figure describes
TARGET = "temp"

GROUPS = {
    "moisture (Open-Meteo)": lambda c: c.startswith("om_humidity")
    or c.startswith("om_dewpoint") or c.startswith("hrext_"),
    "radiation (Open-Meteo)": lambda c: c.startswith("sw_")
    or c.startswith("om_shortwave") or c.startswith("om_cloud"),
    "all Open-Meteo exogenous": lambda c: c.startswith("om_")
    or c.startswith("hrext_") or c.startswith("sw_"),
    "T_ext 6 h rolling mean": lambda c: c == "text_roll_6h_mean",
    "all T_ext rolling": lambda c: c.startswith("text_roll_"),
    "temporal cycles": lambda c: c.startswith("hour_")
    or c.startswith("doy_") or c == "is_dry_season",
}


def walk_forward_mae(X, y, ref):
    """Mean MAE over the folds, increment formulation, level metrics."""
    tscv = TimeSeriesSplit(n_splits=P.CV_N_SPLITS)
    maes = []
    for tr, te in tscv.split(X):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        r_tr = ref.reindex(y_tr.index)
        r_te = ref.reindex(y_te.index).to_numpy()
        p50, _ = P.fit_predict_quantile(X_tr, y_tr - r_tr, X_te, 0.5)
        maes.append(mean_absolute_error(y_te, p50 + r_te))
    return float(np.mean(maes))


def main() -> None:
    print("=" * 66)
    print(" ABLATION AND WITHIN-REGIME TESTS")
    print("=" * 66)

    df = P.load_data(P.CSV_PATH)
    df = P.resample_regular(df)
    df, _ = P.enrich_with_openmeteo(df, P.SITE_LAT, P.SITE_LON, P.SITE_ELEV)
    df = P.segment_regimes(df)
    feat = P.make_features(df)
    feat, targets = P.make_targets(feat, P.HORIZONS_STEPS)

    cols = [c for c in feat.columns if c != "segment"]
    col_t = f"{TARGET}_h{HORIZON}"
    mask = feat[cols].notna().all(axis=1) & targets[col_t].notna()
    X_full, y = feat.loc[mask, cols], targets.loc[mask, col_t]
    ref = feat.loc[mask, "temperature"]

    base = walk_forward_mae(X_full, y, ref)
    print(f"\n--- 1. ABLATION, interior T at h+1h ---")
    print(f"  full feature set ({len(cols)} features): "
          f"MAE {base:.4f} degC\n")
    print(f"  {'group removed':<26} {'n':>3} {'MAE':>8} {'change':>9}")
    print("  " + "-" * 50)

    rows = []
    for name, pred in GROUPS.items():
        keep = [c for c in cols if not pred(c)]
        dropped = len(cols) - len(keep)
        if dropped == 0:
            continue
        mae = walk_forward_mae(X_full[keep], y, ref)
        delta = 100 * (mae - base) / base
        flag = "  <-- matters" if delta > 5 else ""
        print(f"  {name:<26} {dropped:>3} {mae:>8.4f} {delta:>+8.1f}%{flag}")
        rows.append({"group": name, "n_removed": dropped,
                     "MAE": round(mae, 4), "MAE_full": round(base, 4),
                     "change_pct": round(delta, 2)})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "ablation.csv", index=False)

    # ---------------- within-regime importance -------------------
    print("\n--- 2. IMPORTANCE WITHIN EACH REGIME ---")
    print("  Ranks recomputed with the season held fixed.\n")
    month = X_full.index.month
    blocks = {"rainy": np.isin(month, [6, 7, 8, 9, 10]),
              "dry": ~np.isin(month, [6, 7, 8, 9, 10])}

    out = []
    for label, sel in blocks.items():
        Xb, yb, rb = X_full[sel], y[sel], ref[sel]
        if len(Xb) < 250:
            print(f"  {label}: only {len(Xb)} samples, skipped")
            continue
        cut = int(len(Xb) * 0.75)
        m = HistGradientBoostingRegressor(
            max_iter=P.N_ESTIMATORS, learning_rate=P.LEARNING_RATE,
            max_depth=P.MAX_DEPTH, random_state=P.RANDOM_STATE)
        m.fit(Xb.iloc[:cut], (yb - rb).iloc[:cut])
        pi = permutation_importance(
            m, Xb.iloc[cut:], (yb - rb).iloc[cut:], n_repeats=5,
            random_state=P.RANDOM_STATE, n_jobs=-1)
        imp = (pd.DataFrame({"feature": Xb.columns,
                             "importance": pi.importances_mean})
               .sort_values("importance", ascending=False))
        print(f"  {label} ({len(Xb)} samples) — top 5:")
        for _, r in imp.head(5).iterrows():
            print(f"      {r['feature']:<22} {r['importance']:+.4f}")
        dew = imp[imp["feature"] == "om_dewpoint_ext"]
        if len(dew):
            rank = int(imp.reset_index(drop=True)
                       .index[imp.reset_index(drop=True)["feature"]
                              == "om_dewpoint_ext"][0]) + 1
            print(f"      om_dewpoint_ext: rank {rank}/{len(imp)}, "
                  f"{dew['importance'].iloc[0]:+.5f}")
        print()
        imp["regime"] = label
        out.append(imp)

    if out:
        pd.concat(out).to_csv(OUTPUT_DIR / "within_regime.csv", index=False)
    print(f"  -> {OUTPUT_DIR / 'ablation.csv'}")
    print(f"  -> {OUTPUT_DIR / 'within_regime.csv'}")


if __name__ == "__main__":
    main()
