"""Thermal signature figure: 7-day sample, daily peak lag, damping.

The peak lag is computed on the 15-minute grid from the timestamp of
each daily maximum, after a centred 1-hour rolling mean that stops a
single noisy sample from capturing the peak. Picking peaks on an hourly
grid instead would quantise every daily estimate to a whole hour and
bias the median upwards, so the finer grid matters here. A
cross-correlation estimate over the whole record is printed alongside as
an independent check.

Only this figure is written; the other figures have their own scripts.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import COLORS, daily_aggregates
from consolidate_data import read_csv_robust

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

GRID = "15min"
SMOOTH = "1h"            # centred rolling window before peak picking
MIN_COVERAGE_H = 18      # hours of data required for a day to count


def load() -> pd.DataFrame:
    frames = [read_csv_robust(SCRIPT_DIR / n)
              for n in ("sensor_export_1.csv", "sensor_export_2.csv")]
    df = (pd.concat(frames, ignore_index=True)
            .sort_values("Date")
            .drop_duplicates("Date")
            .set_index("Date"))
    df["hour"] = df.index.hour
    df["month"] = df.index.month
    df["season"] = np.where(df["month"].isin([6, 7, 8, 9, 10]),
                            "hivernage", "saison_seche")
    return df


def peak_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Daily lag between exterior and interior temperature maxima."""
    sub = df.dropna(subset=["temperature", "temperaturext"])
    g = (sub[["temperature", "temperaturext"]]
         .resample(GRID).mean())

    win = int(pd.Timedelta(SMOOTH) / pd.Timedelta(GRID))
    sm = g.rolling(win, center=True, min_periods=max(2, win // 2)).mean()
    sm = sm.dropna()

    slots_needed = int(MIN_COVERAGE_H * pd.Timedelta("1h")
                       / pd.Timedelta(GRID))

    rows = []
    for date, day in sm.groupby(sm.index.date):
        if len(day) < slots_needed:
            continue
        t_ext = day["temperaturext"].idxmax()
        t_int = day["temperature"].idxmax()
        rows.append({
            "date": date,
            "lag_h": (t_int - t_ext) / pd.Timedelta("1h"),
            "h_ext": t_ext.hour + t_ext.minute / 60.0,
            "h_int": t_int.hour + t_int.minute / 60.0,
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def xcorr_lag(df: pd.DataFrame, max_lag_h: float = 12.0) -> float:
    """Lag maximising the cross-correlation, as an independent estimate."""
    sub = df.dropna(subset=["temperature", "temperaturext"])
    g = sub[["temperature", "temperaturext"]].resample(GRID).mean()
    g = g.interpolate(limit=4).dropna()
    if len(g) < 100:
        return float("nan")
    a = (g["temperature"] - g["temperature"].mean()).to_numpy()
    b = (g["temperaturext"] - g["temperaturext"].mean()).to_numpy()
    step_h = pd.Timedelta(GRID) / pd.Timedelta("1h")
    span = int(max_lag_h / step_h)
    best, best_r = 0, -np.inf
    for k in range(-span, span + 1):
        if k >= 0:
            x, y = a[k:], b[:len(b) - k] if k else b
        else:
            x, y = a[:len(a) + k], b[-k:]
        n = min(len(x), len(y))
        if n < 100:
            continue
        r = float(np.corrcoef(x[:n], y[:n])[0, 1])
        if r > best_r:
            best_r, best = r, k
    return best * step_h


def plot(df: pd.DataFrame, daily: pd.DataFrame, lags: pd.DataFrame,
         out: Path) -> None:
    fig = plt.figure(figsize=(14, 6.6))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.22,
                          left=0.06, right=0.98, top=0.94, bottom=0.13,
                          height_ratios=[1.25, 1])

    # ---- (a) seven-day sample -------------------------------------
    ax = fig.add_subplot(gs[0, :])
    sub = df.dropna(subset=["temperaturext"])
    sample = sub[sub.index >= sub.index.max() - pd.Timedelta(days=7)]
    ax.plot(sample.index, sample["temperaturext"], color=COLORS["T_ext"],
            lw=1.3, label="Outdoor temperature")
    ax.plot(sample.index, sample["temperature"], color=COLORS["T_int"],
            lw=1.5, label="Indoor temperature")
    ax.fill_between(sample.index, sample["temperature"],
                    sample["temperaturext"],
                    where=(sample["temperaturext"] > sample["temperature"]),
                    color=COLORS["T_ext"], alpha=0.10)
    ax.set_title("(a) 7-day sample showing time lag and damping",
                 fontweight="bold", fontsize=11, loc="left")
    ax.set_ylabel("Temperature (°C)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%Hh"))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # ---- (b) daily peak lag ---------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    med = lags["lag_h"].median()
    ax.bar(range(len(lags)), lags["lag_h"], color=COLORS["T_int"],
           alpha=0.8, edgecolor="white")
    ax.axhline(med, color="red", lw=1.8, linestyle="--",
               label=f"Median: {med:+.2f} h")
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.set_xticks(range(len(lags)))
    ax.set_xticklabels([pd.Timestamp(d).strftime("%d/%m")
                        for d in lags["date"]], rotation=45, fontsize=7)
    ax.set_title("(b) Daily peak time lag (indoor minus outdoor)",
                 fontweight="bold", fontsize=11, loc="left")
    ax.set_ylabel("Peak lag (hours)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # ---- (c) diurnal amplitude ------------------------------------
    # The word "damping" is reserved for sigma_int/sigma_ext, the single
    # indicator reported in the paper. The least-squares slope is a
    # different statistic -- it carries a positive intercept and so sits
    # well below the amplitude ratio -- and is labelled as such.
    ax = fig.add_subplot(gs[1, 1])
    d = daily.dropna(subset=["Text_amplitude", "T_amplitude"])
    ax.scatter(d["Text_amplitude"], d["T_amplitude"], color=COLORS["T_int"],
               s=40, alpha=0.7, edgecolor="white")
    m, b = np.polyfit(d["Text_amplitude"], d["T_amplitude"], 1)
    r = float(np.corrcoef(d["Text_amplitude"], d["T_amplitude"])[0, 1])
    xr = np.linspace(d["Text_amplitude"].min(), d["Text_amplitude"].max(), 100)
    ax.plot(xr, m * xr + b, color="red", lw=1.5,
            label=f"Least-squares fit: slope {m:.2f} ($r$ = {r:.2f})")
    lim = [0, max(d["Text_amplitude"].max(), d["T_amplitude"].max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=0.6, alpha=0.4, label="1:1 (no damping)")

    sub = df.dropna(subset=["temperature", "temperaturext"])
    ratio = sub["temperature"].std() / sub["temperaturext"].std()
    ax.text(0.03, 0.97,
            f"Damping $\\sigma_{{int}}/\\sigma_{{ext}}$ = {ratio:.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa",
                      edgecolor="#bdc3c7", linewidth=0.8))

    ax.set_title("(c) Diurnal amplitude reduction",
                 fontweight="bold", fontsize=11, loc="left")
    ax.set_xlabel("Outdoor amplitude (°C)")
    ax.set_ylabel("Indoor amplitude (°C)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()


def main() -> None:
    print("=" * 62)
    print(" THERMAL SIGNATURE FIGURE — sub-hour peak lag")
    print("=" * 62)

    df = load()
    daily = daily_aggregates(df)
    lags = peak_lags(df)

    print(f"\n  {len(df):,} observations | {len(daily)} aggregated days")
    print(f"  grid {GRID}, centred smoothing {SMOOTH}, "
          f"at least {MIN_COVERAGE_H} h of data per day")

    print(f"\n--- DAILY PEAK LAG ({len(lags)} days retained) ---")
    for _, r in lags.iterrows():
        print(f"  {pd.Timestamp(r['date']):%d/%m}  outdoor peak {r['h_ext']:5.2f} h"
              f"   indoor peak {r['h_int']:5.2f} h   lag {r['lag_h']:+5.2f} h")

    med, mean = lags["lag_h"].median(), lags["lag_h"].mean()
    q1, q3 = lags["lag_h"].quantile([0.25, 0.75])
    print(f"\n  median       : {med:+.2f} h")
    print(f"  mean         : {mean:+.2f} h  (sd {lags['lag_h'].std():.2f})")
    print(f"  interquartile: {q1:+.2f} .. {q3:+.2f} h")
    print(f"  range        : {lags['lag_h'].min():+.2f} .. "
          f"{lags['lag_h'].max():+.2f} h")

    xc = xcorr_lag(df)
    print(f"\n  independent cross-correlation estimate : {xc:+.2f} h")

    sub = df.dropna(subset=["temperature", "temperaturext"])
    ratio = sub["temperature"].std() / sub["temperaturext"].std()
    d = daily.dropna(subset=["Text_amplitude", "T_amplitude"])
    ai, ae = d["T_amplitude"].mean(), d["Text_amplitude"].mean()
    m, _ = np.polyfit(d["Text_amplitude"], d["T_amplitude"], 1)
    print("\n--- DAMPING (three distinct quantities) ---")
    print(f"  sigma_int/sigma_ext        : {ratio:.4f}   "
          f"(n = {len(sub)} concurrent samples)  <-- reported indicator")
    print(f"  mean amplitude ratio       : {ai / ae:.4f}   "
          f"({ai:.2f} / {ae:.2f} degC, i.e. {(1 - ai / ae) * 100:.1f} % "
          f"reduction)")
    print(f"  least-squares slope        : {m:.4f}   "
          f"(a different statistic: non-zero intercept)")

    plot(df, daily, lags, OUTPUT_DIR / "amortissement_CEB.png")
    lags.to_csv(OUTPUT_DIR / "peak_lags.csv", index=False)
    print(f"\n  -> {OUTPUT_DIR / 'amortissement_CEB.png'}")
    print(f"  -> {OUTPUT_DIR / 'peak_lags.csv'}")


if __name__ == "__main__":
    main()
