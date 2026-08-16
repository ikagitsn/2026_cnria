"""Generate fig_pipeline.pdf — the six-layer architecture diagram.

The figure is drawn at its final printed size (7.16 in, the IEEE
two-column text width) so that no scaling is applied by
\\includegraphics and the type stays at the size chosen here. It is
written as PDF because a line diagram should stay vector; pdflatex
picks the PDF ahead of any bitmap of the same name.

Content follows Section III-D and the caption in paper.tex: six
independently testable layers from raw sensor CSV to diagnostics and
forecasts, with the ERA5/Open-Meteo branch feeding both the gap-filled
exterior temperature into L2 and six exogenous variables into L4.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TEXT_WIDTH_IN = 7.16
FIG_HEIGHT_IN = 2.45

INK = "#2c3e50"
EDGE = "#7f8c8d"
API = "#16a085"
IO = "#34495e"

# Bullet strings are kept to ~15 characters: the layer boxes are only
# 0.78 in wide once the diagram is placed at IEEE text width.
LAYERS = [
    ("L1", "Ingestion",   ["merge exports", "schema check",
                           "dedup, UTC"],            "#c0392b"),
    ("L2", "Enrichment",  ["fetch ERA5", "validate",
                           "align to grid"],         API),
    ("L3", "Preprocessing", ["resample 15 min", "seasonal split",
                             "winsorise, fill"],     "#8e44ad"),
    ("L4", "Features",    ["CEB-aware lags", "rolling stats",
                           "cyclic encoding"],       "#2980b9"),
    ("L5", "Modelling",   ["quantile HGB", "walk-forward CV",
                           "4 horizons"],            "#d35400"),
    ("L6", "Visualisation", ["diagnostics", "importance",
                             "auto report"],         "#27ae60"),
]


def box(ax, x0, x1, y0, y1, edge, fill="#ffffff", lw=1.1, r=0.9):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={r}",
        linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2))


def arrow(ax, xy_from, xy_to, color=INK, style="-|>", dashed=False, lw=1.1,
          rad=0.0):
    ax.add_patch(FancyArrowPatch(
        xy_from, xy_to, arrowstyle=style, mutation_scale=9,
        linewidth=lw, color=color, zorder=3,
        linestyle=(0, (3, 2)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=0, shrinkB=0))


def main() -> None:
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, FIG_HEIGHT_IN))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 34)
    ax.axis("off")

    # ---- geometry -------------------------------------------------
    n = len(LAYERS)
    io_w, gap = 12.0, 1.5
    chain_w = 100 - 2 * io_w - 2 * gap
    lw_ = (chain_w - (n - 1) * gap) / n
    y0, y1 = 7.0, 21.5

    # ---- input / output -------------------------------------------
    box(ax, 0, io_w, y0 + 1.6, y1 - 1.6, IO, "#ecf0f1")
    ax.text(io_w / 2, (y0 + y1) / 2 + 1.6, "Sensor CSV", ha="center",
            va="center", fontsize=6.8, fontweight="bold", color=IO)
    ax.text(io_w / 2, (y0 + y1) / 2 - 1.7, "exports\n2,237 records",
            ha="center", va="center", fontsize=5.4, color=IO, linespacing=1.4)

    ox0 = 100 - io_w
    box(ax, ox0, 100, y0 + 1.6, y1 - 1.6, IO, "#ecf0f1")
    ax.text(ox0 + io_w / 2, (y0 + y1) / 2 + 1.6, "Outputs", ha="center",
            va="center", fontsize=6.8, fontweight="bold", color=IO)
    ax.text(ox0 + io_w / 2, (y0 + y1) / 2 - 1.9,
            "diagnostics\nforecasts + PI", ha="center", va="center",
            fontsize=5.4, color=IO, linespacing=1.4)

    # ---- layer boxes ----------------------------------------------
    centres = []
    x = io_w + gap
    for tag, name, bullets, colour in LAYERS:
        box(ax, x, x + lw_, y0, y1, colour, "#ffffff")
        cx = x + lw_ / 2
        ax.text(cx, y1 - 1.9, tag, ha="center", va="center",
                fontsize=6.0, fontweight="bold", color=colour)
        # long names would overrun the box at the nominal size
        ax.text(cx, y1 - 4.2, name, ha="center", va="center",
                fontsize=6.9 if len(name) <= 11 else 5.8,
                fontweight="bold", color=colour)
        ax.plot([x + 1.5, x + lw_ - 1.5], [y1 - 5.9] * 2, color=colour,
                lw=0.6, alpha=0.5, zorder=3)
        for k, b in enumerate(bullets):
            ax.text(cx, y1 - 7.8 - k * 2.4, b, ha="center", va="center",
                    fontsize=5.0, color=INK)
        centres.append((x, x + lw_))
        x += lw_ + gap

    # ---- main flow arrows -----------------------------------------
    ym = (y0 + y1) / 2
    arrow(ax, (io_w, ym), (centres[0][0], ym))
    for i in range(n - 1):
        arrow(ax, (centres[i][1], ym), (centres[i + 1][0], ym))
    arrow(ax, (centres[-1][1], ym), (ox0, ym))

    # ---- ERA5 / Open-Meteo branch ---------------------------------
    ax0, ax1 = centres[1][0] - 1.0, centres[3][1] + 1.0
    box(ax, ax0, ax1, 25.0, 32.0, API, "#eafaf6", lw=1.0)
    ax.text((ax0 + ax1) / 2, 29.9, "ERA5-Land via Open-Meteo Historical "
            "Archive API", ha="center", va="center", fontsize=6.2,
            fontweight="bold", color=API)
    ax.text((ax0 + ax1) / 2, 26.9,
            "gap-filled $T_{ext}$ (391 values)  ·  6 exogenous variables",
            ha="center", va="center", fontsize=5.6, color=API)

    c2 = sum(centres[1]) / 2
    c4 = sum(centres[3]) / 2
    arrow(ax, (c2, 25.0), (c2, y1), color=API, dashed=True, lw=1.0)
    arrow(ax, (c4, 25.0), (c4, y1), color=API, dashed=True, lw=1.0)

    # ---- reproducibility note -------------------------------------
    ax.text(50, 2.8,
            "fixed seeds  ·  relative paths  ·  headless backend  —  "
            "identical on Windows, macOS and Linux",
            ha="center", va="center", fontsize=5.8, style="italic",
            color=EDGE)

    out = OUTPUT_DIR / "fig_pipeline.pdf"
    plt.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(OUTPUT_DIR / "fig_pipeline.png", dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"  -> {out}")
    print(f"  -> {OUTPUT_DIR / 'fig_pipeline.png'}  (apercu)")


if __name__ == "__main__":
    main()
