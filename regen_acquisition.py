"""Generate fig_acquisition.pdf, the acquisition-chain schematic (Fig. 1).

Vector redraw of the design export "Hardware Architecture
Figure-selection.png", whose text came out at 3 to 5 pt once printed at
the two-column width. Drawn at its final printed size (7.16 in) with the
palette, strokes and box style of fig_pipeline.pdf, since the two figures
share a page. Box titles are set at 8 pt and nothing goes below 6.5 pt.

Only the three channels analysed in the paper appear; the design export
also showed the NH3 channel, which the paper does not use. The pipeline
box cites Fig. 2, so this figure must stay before fig_pipeline in the
LaTeX source.
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

# Palette shared with regen_pipeline.py.
INK = "#1a1a1a"
GREY = "#595959"
FILL = "#ededed"
RULE = "#bfbfbf"

# Type sizes in points, at the printed size.
TITLE_PT = 8.0
SUB_PT = 6.5
HEAD_PT = 6.5
NOTE_PT = 6.5

# Geometry in inches. The axes map one-to-one onto the figure, so a length
# here is a length on paper.
MARGIN = 0.04
PAD = 0.07          # inside a panel, around its boxes
GAP_BOX = 0.16      # between two boxes of the same panel
GAP_PANEL = 0.14    # between two panels
CSV_LANE = 0.36     # room for the "CSV dumps" label before the pipeline box
N_COLUMNS = 7       # sensors, board, link, IoT Core, Lambda, DynamoDB, pipeline
BOX_W = (TEXT_WIDTH_IN - 2 * MARGIN - 3 * GAP_PANEL - 8 * PAD
         - 3 * GAP_BOX - CSV_LANE) / N_COLUMNS

RULE_Y = 0.22       # footer rule
PANEL_Y0 = 0.30
NOTE_H = 0.30       # notes under the boxes
BOX_Y0 = PANEL_Y0 + PAD + NOTE_H
BOX_H = 0.70        # tall enough for two stacked sensor boxes
BOX_Y1 = BOX_Y0 + BOX_H
HEAD_H = 0.17       # panel heading above the boxes
PANEL_Y1 = BOX_Y1 + HEAD_H
FIG_HEIGHT_IN = PANEL_Y1 + MARGIN
MID_Y = BOX_Y0 + BOX_H / 2


def rounded(ax, x0, y0, w, h, edge=INK, fill=FILL, lw=1.3, radius=0.035,
            dashed=False, zorder=2):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=fill,
        linestyle=(0, (4, 2.5)) if dashed else "solid", zorder=zorder))


def labelled_box(ax, x0, y0, h, title, sub_lines):
    """Box with a bold title over grey detail lines, centred vertically."""
    rounded(ax, x0, y0, BOX_W, h)
    title_h = TITLE_PT * 1.15 / 72
    sub_h = SUB_PT * 1.25 / 72 * len(sub_lines)
    gap = 0.025
    top = y0 + h / 2 + (title_h + gap + sub_h) / 2
    cx = x0 + BOX_W / 2
    ax.text(cx, top, title, ha="center", va="top", fontsize=TITLE_PT,
            fontweight="bold", color=INK, zorder=4)
    ax.text(cx, top - title_h - gap, "\n".join(sub_lines), ha="center",
            va="top", fontsize=SUB_PT, color=GREY, linespacing=1.25,
            zorder=4)


def panel(ax, x0, x1, heading, notes=None):
    rounded(ax, x0, PANEL_Y0, x1 - x0, PANEL_Y1 - PANEL_Y0, edge=RULE,
            fill="none", lw=0.8, radius=0.05, zorder=1)
    ax.text(x0 + PAD, PANEL_Y1 - HEAD_H / 2, heading, ha="left",
            va="center", fontsize=HEAD_PT, fontweight="bold", color=GREY)
    if notes:
        ax.text(x0 + PAD, BOX_Y0 - 0.05, "\n".join(notes), ha="left",
                va="top", fontsize=NOTE_PT, style="italic", color=GREY,
                linespacing=1.25)


def arrow(ax, start, end, color=INK, dashed=False):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=7, linewidth=1.2,
        color=color, linestyle=(0, (3.5, 2.2)) if dashed else "solid",
        shrinkA=0, shrinkB=0, zorder=3))


def draw():
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, FIG_HEIGHT_IN))
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, TEXT_WIDTH_IN)
    ax.set_ylim(0, FIG_HEIGHT_IN)
    ax.axis("off")

    # ---- horizontal positions, left to right ------------------------
    p1_x0 = MARGIN
    sensors_x0 = p1_x0 + PAD
    board_x0 = sensors_x0 + BOX_W + GAP_BOX
    p1_x1 = board_x0 + BOX_W + PAD

    p2_x0 = p1_x1 + GAP_PANEL
    link_x0 = p2_x0 + PAD
    p2_x1 = link_x0 + BOX_W + PAD

    p3_x0 = p2_x1 + GAP_PANEL
    core_x0 = p3_x0 + PAD
    lambda_x0 = core_x0 + BOX_W + GAP_BOX
    table_x0 = lambda_x0 + BOX_W + GAP_BOX
    p3_x1 = table_x0 + BOX_W + PAD

    p4_x0 = p3_x1 + GAP_PANEL
    pipeline_x0 = p4_x0 + PAD + CSV_LANE
    p4_x1 = pipeline_x0 + BOX_W + PAD

    # ---- panels -------------------------------------------------------
    panel(ax, p1_x0, p1_x1, "EDB BUILDING, THIÈS",
          ["sampled every 10 min", "no local buffering"])
    panel(ax, p2_x0, p2_x1, "NETWORK", ["outage:", "sample lost"])
    panel(ax, p3_x0, p3_x1, "AWS CLOUD",
          ["the cloud chain stores samples unchanged"])
    panel(ax, p4_x0, p4_x1, "OFFLINE ANALYSIS")

    # ---- field: two sensors wired to the board ------------------------
    half = (BOX_H - 0.06) / 2
    labelled_box(ax, sensors_x0, BOX_Y0 + half + 0.06, half, "DHT22",
                 ["indoor T, RH"])
    labelled_box(ax, sensors_x0, BOX_Y0, half, "DS18B20", ["near-wall T"])
    labelled_box(ax, board_x0, BOX_Y0, BOX_H, "Arduino",
                 ["Nano 33 IoT", "UTC, JSON"])
    for yc in (BOX_Y0 + half / 2, BOX_Y0 + half + 0.06 + half / 2):
        arrow(ax, (sensors_x0 + BOX_W, yc), (board_x0, yc))

    # ---- link, cloud chain and pipeline --------------------------------
    labelled_box(ax, link_x0, BOX_Y0, BOX_H, "MQTT/TLS",
                 ["campus Wi-Fi", "QoS 0, X.509"])
    labelled_box(ax, core_x0, BOX_Y0, BOX_H, "IoT Core",
                 ["MQTT broker", "SQL rule"])
    labelled_box(ax, lambda_x0, BOX_Y0, BOX_H, "Lambda",
                 ["validation", "PutItem"])
    labelled_box(ax, table_x0, BOX_Y0, BOX_H, "DynamoDB",
                 ["key node_01", "sort key Date"])
    labelled_box(ax, pipeline_x0, BOX_Y0, BOX_H, "Pipeline",
                 ["six layers", "(Fig. 2)"])

    chain = [board_x0, link_x0, core_x0, lambda_x0, table_x0, pipeline_x0]
    for a, b in zip(chain, chain[1:]):
        arrow(ax, (a + BOX_W, MID_Y), (b, MID_Y))
    ax.text(pipeline_x0 - CSV_LANE / 2, MID_Y + 0.04, "CSV\ndumps",
            ha="center", va="bottom", fontsize=NOTE_PT, style="italic",
            color=GREY, linespacing=1.15)

    # ---- ERA5 branch, entering the pipeline from below -----------------
    era_w, era_h = 1.06, 0.18
    era_x0 = p4_x1 - PAD - era_w
    era_y0 = PANEL_Y0 + PAD
    rounded(ax, era_x0, era_y0, era_w, era_h, edge=GREY, fill="#ffffff",
            lw=1.0, dashed=True)
    ax.text(era_x0 + era_w / 2, era_y0 + era_h / 2, "ERA5 / Open-Meteo",
            ha="center", va="center", fontsize=NOTE_PT, style="italic",
            color=GREY, zorder=4)
    px = pipeline_x0 + BOX_W / 2
    arrow(ax, (px, era_y0 + era_h), (px, BOX_Y0), color=GREY, dashed=True)

    # ---- footer --------------------------------------------------------
    ax.plot([MARGIN, TEXT_WIDTH_IN - MARGIN], [RULE_Y, RULE_Y],
            color=RULE, lw=0.8)
    ax.text(TEXT_WIDTH_IN / 2, RULE_Y / 2,
            "single sensor node node_01   ·   MQTT topic "
            "edb/node_01/telemetry   ·   exports sensor_export_1.csv and "
            "sensor_export_2.csv",
            ha="center", va="center", fontsize=NOTE_PT, style="italic",
            color=GREY)
    return fig


def main() -> None:
    fig = draw()
    out = OUTPUT_DIR / "fig_acquisition.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    # PNG for slides and documents, on an opaque white ground as for Fig. 1.
    fig.savefig(OUTPUT_DIR / "fig_acquisition.png", dpi=600,
                bbox_inches="tight", pad_inches=0.05,
                facecolor="white", edgecolor="none", transparent=False)
    plt.close(fig)
    print(f"  -> {out}")
    print(f"  -> {OUTPUT_DIR / 'fig_acquisition.png'}")


if __name__ == "__main__":
    main()
