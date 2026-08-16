"""Data completeness audit for the EDB monitoring campaign.

Quantifies how much of the nominal 10-minute series was actually
measured, per variable and per seasonal window, how fragmented each
window is, and what fraction of the 15-minute modelling grid is backed
by a real sample rather than interpolated or substituted from
reanalysis.

Two framings are reported deliberately:
  * seasonal blocks, split at the single longest interruption, which is
    the framing used in the paper;
  * contiguous sub-windows, split wherever the gap exceeds 6 h, which
    exposes the fragmentation hidden inside the rainy-season block.

Also computes the per-regime summary statistics reported in the paper.
Writes outputs/completeness.csv, completeness.tex, seasonal_summary.csv
and seasonal_summary.tex.
"""

from pathlib import Path

import pandas as pd

from consolidate_data import read_csv_robust

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

NOMINAL_STEP = pd.Timedelta("10min")
MODEL_STEP = pd.Timedelta("15min")
FRAG_THRESHOLD = pd.Timedelta("6h")

# ASHRAE 55 comfort envelope adapted to naturally ventilated buildings
# in a tropical climate.
COMFORT_T = (24.0, 30.0)     # degC
COMFORT_RH = (30.0, 70.0)    # %
RAINY_MONTHS = (6, 7, 8, 9, 10)

# Only the three variables the study actually uses. The node also
# records an NH3 index and a derived heat index; neither is used, so
# reporting their completeness would pad the table without informing
# any result.
VARIABLES = {
    "temperature":   r"$T_\mathrm{int}$",
    "humidity":      r"$\mathrm{RH}_\mathrm{int}$",
    "temperaturext": r"$T_\mathrm{ext}$",
}


def load() -> pd.DataFrame:
    frames = []
    for name in ("sensor_export_1.csv", "sensor_export_2.csv"):
        df = read_csv_robust(SCRIPT_DIR / name)
        frames.append(df)
        print(f"  {name:22} {len(df):5} rows")
    merged = (pd.concat(frames, ignore_index=True)
                .sort_values("Date")
                .drop_duplicates(subset=["Date"], keep="first")
                .reset_index(drop=True))
    print(f"  consolidated           {len(merged):5} unique rows")
    return merged


def seasonal_blocks(df: pd.DataFrame):
    """Split at the single longest interruption."""
    d = df["Date"].reset_index(drop=True)
    gaps = d.diff()
    i = int(gaps.idxmax())
    big = gaps.iloc[i]
    return ([(d.iloc[0], d.iloc[i - 1]), (d.iloc[i], d.iloc[-1])], big)


def fragments(df, start, end, threshold=FRAG_THRESHOLD):
    seg = df[(df["Date"] >= start) & (df["Date"] <= end)]["Date"]
    blocks = (seg.diff() > threshold).cumsum()
    spans = [(g.min(), g.max()) for _, g in seg.groupby(blocks)]
    covered = sum(((b - a) for a, b in spans),
                  pd.Timedelta(0)) / pd.Timedelta("1D")
    return len(spans), covered


def audit(df, blocks, names):
    rows = []
    for (start, end), name in zip(blocks, names):
        seg = df[(df["Date"] >= start) & (df["Date"] <= end)]
        span = end - start
        slots = int(span / NOMINAL_STEP) + 1
        nfrag, covered = fragments(df, start, end)
        for col, label in VARIABLES.items():
            n = int(seg[col].notna().sum())
            rows.append({
                "block": name,
                "start": start.strftime("%Y-%m-%d"),
                "end": end.strftime("%Y-%m-%d"),
                "span_days": round(span / pd.Timedelta("1D"), 1),
                "fragments": nfrag,
                "covered_days": round(covered, 1),
                "variable": col,
                "label": label,
                "measured": n,
                "expected_slots": slots,
                "completeness_pct": round(100.0 * n / slots, 1),
            })
    return pd.DataFrame(rows)


def model_grid(df, blocks):
    out = {}
    for col in ("temperature", "humidity", "temperaturext"):
        hit = tot = 0
        for start, end in blocks:
            grid = set(pd.date_range(start.floor("15min"),
                                     end.ceil("15min"), freq=MODEL_STEP))
            s = df.loc[df[col].notna(), "Date"]
            s = s[(s >= start) & (s <= end)]
            hit += len(set(pd.Series(s).dt.floor("15min")) & grid)
            tot += len(grid)
        out[col] = (hit, tot, round(100.0 * hit / tot, 1))
    return out


def seasonal_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-regime statistics: the seasonal summary table of the paper.

    Lives here rather than in a separate module because this script
    already loads and segments the record. Comfort is the fraction of
    samples inside the adapted ASHRAE 55 envelope, over samples where
    both temperature and humidity are present.
    """
    d = df.copy()
    d["regime"] = d["Date"].dt.month.map(
        lambda m: "Rainy" if m in RAINY_MONTHS else "Dry season")

    rows = []
    for label, sub in [("Global", d)] + list(d.groupby("regime")):
        ok = sub["temperature"].notna() & sub["humidity"].notna()
        s = sub[ok]
        comfort = (
            (s["temperature"].between(*COMFORT_T))
            & (s["humidity"].between(*COMFORT_RH))
        ).mean() * 100 if len(s) else float("nan")
        rows.append({
            "regime": label,
            "n": int(len(sub)),
            "T_mean": round(sub["temperature"].mean(), 1),
            "T_std": round(sub["temperature"].std(), 1),
            "RH_mean": round(sub["humidity"].mean(), 1),
            "RH_std": round(sub["humidity"].std(), 1),
            "comfort_pct": round(comfort, 1),
        })
    order = {"Global": 0, "Rainy": 1, "Dry season": 2}
    return (pd.DataFrame(rows)
            .sort_values("regime", key=lambda c: c.map(order))
            .reset_index(drop=True))


def summary_to_latex(s: pd.DataFrame) -> str:
    L = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Summary statistics by seasonal regime}",
        r"\label{tab:stats}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Regime & $n$ & $T_\text{int}$ (\textdegree{}C)",
        r"       & $\text{RH}_\text{int}$ (\%) & Comfort (\%) \\",
        r"\midrule",
    ]
    for _, r in s.iterrows():
        n = f"{int(r['n']):,}".replace(",", "{,}")
        L.append(f"{r['regime']:<10} & {n} & "
                 f"${r['T_mean']:.1f}\\pm{r['T_std']:.1f}$ & "
                 f"${r['RH_mean']:.1f}\\pm{r['RH_std']:.1f}$ & "
                 f"{r['comfort_pct']:.1f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def to_latex(tab, names):
    L = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Data completeness per variable and per seasonal window,"
        r" against the nominal 10-minute sampling interval.}",
        r"\label{tab:completeness}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Rainy (16 Jul--19 Aug)}"
        r" & \multicolumn{2}{c}{Dry (26 Nov--4 Dec)} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Variable & $n$ & \% & $n$ & \% \\",
        r"\midrule",
    ]
    for col, label in VARIABLES.items():
        cells = []
        for name in names:
            r = tab[(tab["block"] == name) & (tab["variable"] == col)].iloc[0]
            cells += [f"{int(r['measured']):,}".replace(",", "{,}"),
                      f"{r['completeness_pct']:.1f}"]
        L.append(f"{label} & " + " & ".join(cells) + r" \\")

    a = tab[tab["block"] == names[0]].iloc[0]
    b = tab[tab["block"] == names[1]].iloc[0]
    L += [
        r"\midrule",
        rf"Window span (days) & \multicolumn{{2}}{{c}}{{{a['span_days']:.1f}}}"
        rf" & \multicolumn{{2}}{{c}}{{{b['span_days']:.1f}}} \\",
        rf"Contiguous fragments & \multicolumn{{2}}{{c}}{{{a['fragments']}}}"
        rf" & \multicolumn{{2}}{{c}}{{{b['fragments']}}} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(L)


def main() -> None:
    print("=" * 64)
    print(" DATA COMPLETENESS AUDIT — EDB campaign")
    print("=" * 64)

    df = load()
    blocks, big_gap = seasonal_blocks(df)
    names = ["rainy", "dry"]

    print(f"\n  coverage     : {df['Date'].min():%Y-%m-%d %H:%M} -> "
          f"{df['Date'].max():%Y-%m-%d %H:%M} UTC")
    print(f"  interruption : {big_gap / pd.Timedelta('1D'):.1f} days "
          f"({blocks[0][1]:%d %b} -> {blocks[1][0]:%d %b})")

    tab = audit(df, blocks, names)
    grid = model_grid(df, blocks)

    print("\n--- SEASONAL WINDOWS ---")
    for name in names:
        r = tab[tab["block"] == name].iloc[0]
        n = int(df[(df["Date"] >= pd.Timestamp(r["start"], tz="UTC"))
                   & (df["Date"] <= pd.Timestamp(r["end"], tz="UTC")
                      + pd.Timedelta("1D"))].shape[0])
        print(f"  {name:6} {r['start']} -> {r['end']}  "
              f"extent {r['span_days']:5.1f} d | "
              f"{r['fragments']:2} fragments covering {r['covered_days']:5.1f} d | "
              f"{n:5} records")

    equiv = len(df) * NOMINAL_STEP / pd.Timedelta("1D")
    total_span = tab.groupby("block")["span_days"].first().sum()
    print(f"\n  total window extent    : {total_span:.1f} days")
    print(f"  full-rate equivalent   : {equiv:.1f} days")
    print(f"  overall completeness   : "
          f"{100.0 * equiv / total_span:.1f} %")

    print("\n--- COMPLETENESS BY VARIABLE (against the 10-min grid) ---")
    show = tab[["block", "variable", "measured", "expected_slots",
                "completeness_pct"]]
    print(show.to_string(index=False))

    print("\n--- 15-MIN GRID: share backed by a real sample ---")
    for col, (h, t, pct) in grid.items():
        print(f"  {col:15} {h:5} / {t:5} slots = {pct:5.1f} %"
              f"   (remainder interpolated or ERA5)")

    miss = int(df["temperaturext"].isna().sum())
    print(f"\n  T_ext missing  : {miss} values "
          f"({100.0 * miss / len(df):.1f} % of records)")

    summary = seasonal_summary(df)
    print("\n--- STATISTICS BY REGIME (seasonal summary table) ---")
    print(summary.to_string(index=False))
    summary.to_csv(OUTPUT_DIR / "seasonal_summary.csv", index=False)
    (OUTPUT_DIR / "seasonal_summary.tex").write_text(
        summary_to_latex(summary), encoding="utf-8")

    tab.to_csv(OUTPUT_DIR / "completeness.csv", index=False)
    (OUTPUT_DIR / "completeness.tex").write_text(
        to_latex(tab, names), encoding="utf-8")
    print(f"\n  -> {OUTPUT_DIR / 'completeness.csv'}")
    print(f"  -> {OUTPUT_DIR / 'completeness.tex'}")


if __name__ == "__main__":
    main()
