"""Generate fig_pipeline.pdf — the six-layer architecture diagram.

Drawn at its final printed size (7.16 in, the IEEE two-column text
width) so that \\includegraphics applies no scaling and the type stays
at the size chosen here. Written as PDF because a line diagram should
stay vector; pdflatex picks the PDF ahead of any bitmap of the same
name, so no stale PNG can silently win.

Layer names and counts follow Section III-E of the paper. Keep them in
step: a figure that says 28 features while the text says 36 is exactly
the kind of mismatch a reader notices first.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
# Embed TrueType (Type 42) fonts instead of matplotlib's default Type 3:
# IEEE camera-ready compliance checks flag Type 3 fonts.
matplotlib.rcParams["pdf.fonttype"] = 42

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TEXT_WIDTH_IN = 7.16
FIG_HEIGHT_IN = 1.95

INK = "#1a1a1a"
GREY = "#595959"
FILL = "#ededed"
RULE = "#bfbfbf"

LAYERS = [
    ("L1", "Ingestion",    "CSV consolid."),
    ("L2", "Enrichment",   "ERA5 fetch"),
    ("L3", "Preprocessing", "15-min resample"),
    ("L4", "Features",     "36 predictors"),
    ("L5", "Modeling",     "HistGB quantile"),
    ("L6", "Visualisation", "figures + report"),
]


def box(ax, x0, x1, y0, y1, lw=1.3, fill=FILL, edge=INK, dashed=False):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0,rounding_size=1.2",
        linewidth=lw, edgecolor=edge, facecolor=fill,
        linestyle=(0, (4, 2.5)) if dashed else "solid", zorder=2))


def main() -> None:
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, FIG_HEIGHT_IN))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")

    n = len(LAYERS)
    left_pad, right_pad, gap = 6.5, 7.0, 1.6
    chain = 100 - left_pad - right_pad
    bw = (chain - (n - 1) * gap) / n
    y0, y1 = 9.0, 20.0
    ym = (y0 + y1) / 2

    # ---- input / output labels ------------------------------------
    ax.text(left_pad - 1.4, ym, "raw\nIoT\nCSV", ha="right", va="center",
            fontsize=5.5, style="italic", color=GREY, linespacing=1.5)
    ax.text(100 - right_pad + 1.4, ym, "figures\n+ tables\n+ CSV",
            ha="left", va="center", fontsize=5.5, style="italic",
            color=GREY, linespacing=1.5)

    # ---- layer boxes ----------------------------------------------
    centres = []
    x = left_pad
    for tag, name, sub in LAYERS:
        box(ax, x, x + bw, y0, y1)
        ax.text(x + 1.3, y1 - 2.2, tag, ha="left", va="center",
                fontsize=5.3, fontweight="bold", color=GREY)
        # long names would touch the box edges at the nominal size
        ax.text(x + bw / 2, ym - 0.4, name, ha="center", va="center",
                fontsize=6.5 if len(name) <= 12 else 5.7,
                fontweight="bold", color=INK)
        ax.text(x + bw / 2, y0 + 2.0, sub, ha="center", va="center",
                fontsize=4.8, style="italic", color=GREY)
        centres.append((x, x + bw))
        x += bw + gap

    # ---- flow arrows ----------------------------------------------
    for a, b in zip(centres, centres[1:]):
        ax.add_patch(FancyArrowPatch(
            (a[1] + 0.15, ym), (b[0] - 0.15, ym),
            arrowstyle="-|>", mutation_scale=7, linewidth=1.2,
            color=INK, shrinkA=0, shrinkB=0, zorder=3))
    ax.plot([left_pad - 1.1, left_pad], [ym, ym], color=INK, lw=1.2)
    ax.plot([100 - right_pad, 100 - right_pad + 1.1], [ym, ym],
            color=INK, lw=1.2)

    # ---- ERA5 branch, entering at L2 ------------------------------
    # The box spans L1-L2 and the arrow drops straight into L2, so the
    # branch reads as entering at one layer rather than floating.
    cx = sum(centres[1]) / 2
    bx0, bx1 = centres[0][0] + 1.0, centres[1][1] - 1.0
    box(ax, bx0, bx1, 23.5, 29.0, lw=1.3, fill="#ffffff", edge=GREY,
        dashed=True)
    ax.text((bx0 + bx1) / 2, 26.25, "ERA5  /  Open-Meteo  API",
            ha="center", va="center", fontsize=6.0, style="italic",
            color=GREY)
    ax.add_patch(FancyArrowPatch(
        (cx, 23.5), (cx, y1 + 0.15),
        arrowstyle="-|>", mutation_scale=7, linewidth=1.2, color=GREY,
        linestyle=(0, (3.5, 2.2)), shrinkA=0, shrinkB=0, zorder=3))

    # ---- footer ---------------------------------------------------
    ax.plot([2, 98], [5.2, 5.2], color=RULE, lw=0.8)
    ax.text(50, 2.9,
            "Six independently testable layers   ·   reproducible on "
            "Windows / macOS / Linux   ·   random_state = 42",
            ha="center", va="center", fontsize=5.2, style="italic",
            color=GREY)

    out = OUTPUT_DIR / "fig_pipeline.pdf"
    plt.savefig(out, bbox_inches="tight", pad_inches=0.02)
    # The PNG is for slides and documents rather than the paper, so it
    # gets an opaque white ground: a transparent one turns the grey
    # strokes invisible on a dark background.
    plt.savefig(OUTPUT_DIR / "fig_pipeline.png", dpi=600,
                bbox_inches="tight", pad_inches=0.05,
                facecolor="white", edgecolor="none", transparent=False)
    plt.close()
    print(f"  -> {out}")
    print(f"  -> {OUTPUT_DIR / 'fig_pipeline.png'}")


if __name__ == "__main__":
    main()
