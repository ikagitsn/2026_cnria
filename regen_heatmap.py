"""Regenerate heatmap_calendaire.png only, on a true calendar axis.

Two defects in the original routine are fixed here.

1. The x axis was categorical. `pivot_table` kept only the dates that
   carried data and `imshow` drew them as equal-width columns, so the
   99-day interruption occupied exactly the width of one day. Across the
   record, 112 missing days were compressed into 4 columns.

2. Each panel built its own column set, so the three stacked panels were
   not time-aligned: interior temperature spanned 30 columns from 16
   July, exterior temperature only 22 from 25 July, because of the 391
   exterior values missing that month. Reading (a) against (b) at the
   same abscissa compared different dates.

Both are fixed by reindexing every panel onto one complete daily
calendar. Days without data are rendered in neutral grey so that "no
data" cannot be mistaken for a low value on the colour scale.

Only this figure is written; the other figures have their own scripts.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regen_damping import load

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

NODATA = "#e8e8e8"

PANELS = [
    ("temperature",   "(a) Indoor temperature (°C)",       "RdYlBu_r"),
    ("temperaturext", "(b) Outdoor temperature (°C)",      "RdYlBu_r"),
    ("humidity",      "(c) Indoor relative humidity (%)",  "BrBG"),
]


def hourly(df: pd.DataFrame) -> pd.DataFrame:
    h = df.resample("h").mean(numeric_only=True)
    h["hour"] = h.index.hour
    h["date"] = h.index.normalize()
    return h


def grid(df_h: pd.DataFrame, var: str, calendar: pd.DatetimeIndex):
    """hour x day matrix on the shared calendar, NaN where no data."""
    piv = df_h.pivot_table(index="hour", columns="date", values=var,
                           aggfunc="mean")
    return piv.reindex(index=range(24), columns=calendar)


def windows(df_h: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """The two measurement windows, split at the longest interruption."""
    days = (df_h.loc[df_h["temperature"].notna(), "date"]
            .drop_duplicates().sort_values().reset_index(drop=True))
    i = int(days.diff().iloc[1:].values.argmax()) + 1
    return [(days.iloc[0], days.iloc[i - 1]), (days.iloc[i], days.iloc[-1])]


def plot_broken(df_h: pd.DataFrame, out: Path) -> None:
    """Two windows side by side, at true scale, with an explicit break.

    Keeps the honesty of a proportional axis inside each window while
    recovering the legibility lost when 99 empty days are drawn to
    scale. The break is marked, labelled with its duration, and the
    colour scale is shared across both windows of a row so that the two
    sides remain comparable.
    """
    (r0, r1), (d0, d1) = windows(df_h)
    cal_r = pd.date_range(r0, r1, freq="D")
    cal_d = pd.date_range(d0, d1, freq="D")
    gap_days = (d0 - r1).days

    fig, axes = plt.subplots(
        3, 2, figsize=(15, 8.4),
        gridspec_kw={"width_ratios": [len(cal_r), len(cal_d)],
                     "wspace": 0.06, "hspace": 0.55})

    for row, (var, title, cmap_name) in enumerate(PANELS):
        full = grid(df_h, var, pd.date_range(r0, d1, freq="D"))
        vmin = np.nanmin(full.values)
        vmax = np.nanmax(full.values)
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(NODATA)

        for col, cal in ((0, cal_r), (1, cal_d)):
            ax = axes[row, col]
            m = grid(df_h, var, cal)
            im = ax.imshow(np.ma.masked_invalid(m.values), aspect="auto",
                           cmap=cmap, interpolation="nearest",
                           origin="lower", vmin=vmin, vmax=vmax)
            ax.set_yticks([0, 6, 12, 18, 23])
            step = max(1, len(cal) // (8 if col == 0 else 4))
            ticks = list(range(0, len(cal), step))
            ax.set_xticks(ticks)
            ax.set_xticklabels([cal[i].strftime("%d %b") for i in ticks],
                               fontsize=8)
            ax.set_xlim(-0.5, len(cal) - 0.5)

            if col == 0:
                ax.set_ylabel("Hour (UTC)")
                ax.set_title(title, fontweight="bold", fontsize=11,
                             loc="left")
                ax.spines["right"].set_visible(False)
            else:
                ax.set_yticklabels([])
                ax.spines["left"].set_visible(False)
                cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
                cb.ax.tick_params(labelsize=8)

        # break marks on the two facing edges
        kw = dict(marker=[(-0.5, -1), (0.5, 1)], markersize=9,
                  linestyle="none", color="k", mec="k", mew=1,
                  clip_on=False)
        axes[row, 0].plot([1, 1], [0, 1], transform=axes[row, 0].transAxes,
                          **kw)
        axes[row, 1].plot([0, 0], [0, 1], transform=axes[row, 1].transAxes,
                          **kw)

    axes[0, 0].annotate(
        f"{gap_days} days without data",
        xy=(1.03, 1.16), xycoords="axes fraction", ha="center", va="center",
        fontsize=9, style="italic", color="#555",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#bbb"),
    )

    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  window 1 : {r0:%d %b} -> {r1:%d %b}  ({len(cal_r)} days)")
    print(f"  window 2 : {d0:%d %b} -> {d1:%d %b}  ({len(cal_d)} days)")
    print(f"  break     : {gap_days} days, marked and labelled")
    print(f"  -> {out}")


def main() -> None:
    print("=" * 62)
    print(" CALENDAR HEATMAP — true calendar axis")
    print("=" * 62)

    df = load()
    df_h = hourly(df)

    calendar = pd.date_range(df_h["date"].min(), df_h["date"].max(), freq="D")
    print(f"\n  shared calendar : {calendar[0]:%d %b %Y} -> "
          f"{calendar[-1]:%d %b %Y}  ({len(calendar)} days)")

    fig, axes = plt.subplots(3, 1, figsize=(15, 8.4),
                             gridspec_kw={"hspace": 0.55})

    for ax, (var, title, cmap_name) in zip(axes, PANELS):
        m = grid(df_h, var, calendar)
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(NODATA)

        filled = int(np.isfinite(m.values).any(axis=0).sum())
        print(f"  {title:36} {filled:3}/{len(calendar)} days with data")

        im = ax.imshow(np.ma.masked_invalid(m.values), aspect="auto",
                       cmap=cmap, interpolation="nearest", origin="lower")
        ax.set_title(title, fontweight="bold", fontsize=11, loc="left")
        ax.set_ylabel("Hour (UTC)")
        ax.set_yticks([0, 6, 12, 18, 23])

        step = max(1, len(calendar) // 12)
        ticks = list(range(0, len(calendar), step))
        ax.set_xticks(ticks)
        ax.set_xticklabels([calendar[i].strftime("%d %b") for i in ticks],
                           fontsize=8)
        ax.set_xlim(-0.5, len(calendar) - 0.5)

        cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.025)
        cb.ax.tick_params(labelsize=8)

    # Mark the main interruption on the top panel only.
    # Days that actually carry an observation. df_h comes from a resample,
    # whose index is continuous by construction, so the gap must be read
    # from where the values are, not from the index.
    days = (df_h.loc[df_h["temperature"].notna(), "date"]
            .drop_duplicates().sort_values().reset_index(drop=True))
    # diff() leaves NaT at position 0; skip it before taking the argmax.
    i = int(days.diff().iloc[1:].values.argmax()) + 1
    g0, g1 = days.iloc[i - 1], days.iloc[i]
    x0 = calendar.get_loc(g0) + 0.5
    x1 = calendar.get_loc(g1) - 0.5
    axes[0].annotate(
        f"no data — {(g1 - g0).days} days",
        xy=((x0 + x1) / 2, 12), ha="center", va="center",
        fontsize=9, style="italic", color="#555",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#bbb", alpha=0.9),
    )
    print(f"\n  main interruption : {g0:%d %b} -> {g1:%d %b} "
          f"({(g1 - g0).days} days), now drawn to scale")

    out = OUTPUT_DIR / "heatmap_calendaire.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  -> {out}")

    print("\n--- BROKEN-AXIS VARIANT ---")
    plot_broken(df_h, OUTPUT_DIR / "heatmap_calendaire_broken.png")


if __name__ == "__main__":
    main()
