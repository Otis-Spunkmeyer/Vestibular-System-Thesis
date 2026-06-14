"""
TC4 SNS Network Visualizer — McNeal & Hunt 2026 style diagram

Renders the TC4 SNS in the same visual style as McNeal & Hunt 2026, Fig. 3:
  - White background
  - Horizontal left-to-right flow
  - Rounded-rectangle nodes, colored by functional role
  - Color-coded background regions with dashed borders
  - Upper half = CCW pathway, lower half = CW pathway
  - Green edges = excitatory, red edges = inhibitory

Requires: matplotlib, networkx
Install:  pip install matplotlib networkx
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

from sns_tc4_controller import build_tc4_network

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
# Node positions  — horizontal left-to-right layout
# Upper half (y > 0) = CCW pathway
# Lower half (y < 0) = CW  pathway
# y = 0 line  = shared nodes (coact, ib_input)
# =====================================================================
NODE_POSITIONS = {
    # --- Stage 1: bilateral inputs ---
    "bs_ccw": (0.5,  4.5),   "bs_cw":  (0.5, -4.5),
    "ss_ccw": (0.5,  2.5),   "ss_cw":  (0.5, -2.5),

    # --- Stage 2: TC4 sensory integration (CCW) ---
    "sub_diff_ccw": (2.5,  5.0),
    "wg_ccw":       (2.5,  2.5),
    "wp_ccw":       (4.0,  5.0),
    "error_ccw":    (5.5,  3.5),

    # --- Stage 2: TC4 sensory integration (CW) ---
    "sub_diff_cw":  (2.5, -5.0),
    "wg_cw":        (2.5, -2.5),
    "wp_cw":        (4.0, -5.0),
    "error_cw":     (5.5, -3.5),

    # --- Stage 3: derivative estimation (CCW) ---
    "deriv_fast_ccw": (7.5,  5.0),
    "deriv_slow_ccw": (7.5,  2.5),
    "deriv_out_ccw":  (9.5,  3.5),

    # --- Stage 3: derivative estimation (CW) ---
    "deriv_fast_cw":  (7.5, -2.5),
    "deriv_slow_cw":  (7.5, -5.0),
    "deriv_out_cw":   (9.5, -3.5),

    # --- Stage 4: co-activation node (shared) ---
    "coact_node": (11.0,  0.0),

    # --- Stage 5: Kp gain stage ---
    "kp_mod_ccw": (11.0,  6.0),  "kp_prod_ccw": (12.5,  6.0),
    "kp_mod_cw":  (11.0, -6.0),  "kp_prod_cw":  (12.5, -6.0),

    # --- Stage 5: Kd gain stage ---
    "kd_mod_ccw": (11.0,  3.5),  "kd_prod_ccw": (12.5,  3.5),
    "kd_mod_cw":  (11.0, -3.5),  "kd_prod_cw":  (12.5, -3.5),

    # --- Stage 5: Kc gain stage ---
    "kc_mod_ccw": (11.0,  1.5),  "kc_prod_ccw": (12.5,  1.5),
    "kc_mod_cw":  (11.0, -1.5),  "kc_prod_cw":  (12.5, -1.5),

    # --- Stage 6: Type-Ib input node (shared) ---
    "ib_input": (14.0,  0.0),

    # --- Stage 5: Kt gain stage ---
    "kt_mod_ccw": (14.0,  3.5),  "kt_prod_ccw": (15.5,  3.5),
    "kt_mod_cw":  (14.0, -3.5),  "kt_prod_cw":  (15.5, -3.5),

    # --- Stage 7: bilateral motor output ---
    "ta_ccw": (17.0,  3.5),
    "ta_cw":  (17.0, -3.5),
}

# =====================================================================
# Short display labels for each neuron
# =====================================================================
NODE_LABELS = {
    "bs_ccw": "BS+", "bs_cw": "BS−",
    "ss_ccw": "SS+", "ss_cw": "SS−",
    "sub_diff_ccw": "BS−SS+", "sub_diff_cw": "BS−SS−",
    "wg_ccw": "Wg·BS+",      "wg_cw": "Wg·BS−",
    "wp_ccw": "Wp·ΔΘ+",      "wp_cw": "Wp·ΔΘ−",
    "error_ccw": "e+",        "error_cw": "e−",
    "deriv_fast_ccw": "fast+", "deriv_fast_cw": "fast−",
    "deriv_slow_ccw": "slow+", "deriv_slow_cw": "slow−",
    "deriv_out_ccw": "d+",    "deriv_out_cw": "d−",
    "coact_node": "|d|",
    "kp_mod_ccw": "bp+", "kp_mod_cw": "bp−",
    "kp_prod_ccw": "Kp+", "kp_prod_cw": "Kp−",
    "kd_mod_ccw": "bd+", "kd_mod_cw": "bd−",
    "kd_prod_ccw": "Kd+", "kd_prod_cw": "Kd−",
    "kc_mod_ccw": "bc+", "kc_mod_cw": "bc−",
    "kc_prod_ccw": "Kc+", "kc_prod_cw": "Kc−",
    "kt_mod_ccw": "bt+", "kt_mod_cw": "bt−",
    "kt_prod_ccw": "Kt+", "kt_prod_cw": "Kt−",
    "ib_input": "+Ib",
    "ta_ccw": "+a_CCW", "ta_cw": "+a_CW",
}

# =====================================================================
# Node colors  (matches McNeal & Hunt Fig 3 color scheme)
# =====================================================================
NODE_COLORS = {
    # Input — orange
    "bs_ccw": "#F39C12",  "bs_cw":  "#F39C12",
    "ss_ccw": "#F39C12",  "ss_cw":  "#F39C12",

    # TC4 sensory processing — light green  (NEW over McNeal & Hunt)
    "sub_diff_ccw": "#A9DFBF",  "sub_diff_cw": "#A9DFBF",
    "wg_ccw":       "#A9DFBF",  "wg_cw":       "#A9DFBF",
    "wp_ccw":       "#A9DFBF",  "wp_cw":       "#A9DFBF",

    # Error nodes — salmon (matches M&H error formation pink)
    "error_ccw": "#F1948A",  "error_cw": "#F1948A",

    # Differential neurons — sky blue (matches M&H differential tan/blue)
    "deriv_fast_ccw": "#85C1E9",  "deriv_fast_cw": "#85C1E9",
    "deriv_slow_ccw": "#85C1E9",  "deriv_slow_cw": "#85C1E9",

    # Derivative output + co-activation — orange (matches M&H deriv output)
    "deriv_out_ccw": "#F0A500",  "deriv_out_cw": "#F0A500",
    "coact_node":    "#F0A500",

    # Modulator neurons (bias-gated) — yellow (matches M&H Kp/Kc/Kd/Kt yellow)
    "kp_mod_ccw": "#F9E79F", "kp_mod_cw": "#F9E79F",
    "kd_mod_ccw": "#F9E79F", "kd_mod_cw": "#F9E79F",
    "kc_mod_ccw": "#F9E79F", "kc_mod_cw": "#F9E79F",
    "kt_mod_ccw": "#F9E79F", "kt_mod_cw": "#F9E79F",

    # Product neurons — light purple (matches M&H gain output)
    "kp_prod_ccw": "#C39BD3", "kp_prod_cw": "#C39BD3",
    "kd_prod_ccw": "#C39BD3", "kd_prod_cw": "#C39BD3",
    "kc_prod_ccw": "#C39BD3", "kc_prod_cw": "#C39BD3",
    "kt_prod_ccw": "#C39BD3", "kt_prod_cw": "#C39BD3",

    # Type-Ib input — light gray (matches M&H Ib feedback gray)
    "ib_input": "#BDC3C7",

    # Motor output — red
    "ta_ccw": "#E74C3C",  "ta_cw": "#E74C3C",
}

# =====================================================================
# Background regions  (drawn before nodes)
# Each tuple: (x_left, y_bottom, width, height, fill_color, edge_color, label, label_y)
# =====================================================================
BACKGROUND_REGIONS = [
    # TC4 Sensory Integration (new over McNeal & Hunt) — light green
    (-0.2, -5.8, 6.6, 11.6,
     "#D5F5E3", "#27AE60",
     "TC4 Sensory Integration\n(new over McNeal & Hunt)", -5.5),

    # Error formation sub-zone — slightly darker green, dashed
    (4.8, -4.2, 1.5, 8.4,
     "#A9DFBF", "#1E8449",
     "Error\nformation", 3.8),

    # Differential calculations — light bisque / tan
    (6.8, -5.8, 3.4, 11.6,
     "#FDEBD0", "#E67E22",
     "Differential\nCalculations", -5.5),

    # Derivative Gain Circuits — light blue
    (10.2, -6.8, 6.0, 13.6,
     "#D6EAF8", "#2E86C1",
     "Derivative Gain Circuits", -6.5),

    # Type Ib feedback sub-zone — light gray
    (13.2, -4.2, 3.0, 8.4,
     "#E8E8E8", "#7F8C8D",
     "Type Ib\nfeedback", -3.8),
]

# =====================================================================
# Helper: draw a single rounded-rectangle node
# =====================================================================
def _draw_node(ax, x, y, label, color, node_w=0.9, node_h=0.55):
    box = FancyBboxPatch(
        (x - node_w / 2, y - node_h / 2), node_w, node_h,
        boxstyle="round,pad=0.07",
        facecolor=color,
        edgecolor="#2C3E50",
        linewidth=1.0,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, label,
            ha="center", va="center",
            fontsize=5.8, fontweight="bold",
            color="#1A1A1A", zorder=4)


# =====================================================================
# Helper: draw a directed arrow between two node centers
# =====================================================================
def _draw_arrow(ax, x1, y1, x2, y2, color, lw, rad=0.0):
    # shrinkA/B (pixels) clip the arrow to the node border
    ax.annotate(
        "",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=10,   # pixels back from source center
            shrinkB=10,   # pixels back from target center
        ),
        zorder=2,
    )


# =====================================================================
# Main draw function
# =====================================================================
def draw_network(neurons, synapses, output_file=None):
    """
    Render the TC4 SNS network in McNeal & Hunt 2026 diagram style.

    Args:
        neurons:     dict from build_tc4_network()
        synapses:    list from build_tc4_network()
        output_file: save path (defaults to SCRIPT_DIR/tc4_network_diagram.png)
    """
    if output_file is None:
        output_file = os.path.join(SCRIPT_DIR, "tc4_network_diagram.png")

    fig, ax = plt.subplots(figsize=(22, 14))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # --- Background regions ---
    for (rx, ry, rw, rh, fc, ec, rlabel, label_y) in BACKGROUND_REGIONS:
        rect = mpatches.FancyBboxPatch(
            (rx, ry), rw, rh,
            boxstyle="round,pad=0.1",
            facecolor=fc, edgecolor=ec,
            linewidth=1.5, linestyle="--",
            zorder=0,
        )
        ax.add_patch(rect)
        ax.text(rx + rw / 2, label_y,
                rlabel,
                ha="center", va="center",
                fontsize=7.5, color=ec,
                fontweight="bold", zorder=1,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))

    # --- Bilateral axis labels ---
    ax.text(-0.5, 4.0, "Active when\nBS is CCW (+)",
            ha="right", va="center", fontsize=8,
            style="italic", color="#555555")
    ax.annotate("", xy=(-0.2, 5.5), xytext=(-0.2, 1.5),
                arrowprops=dict(arrowstyle="-[,widthB=2.5", color="#555555", lw=1.2),
                zorder=1)

    ax.text(-0.5, -4.0, "Active when\nBS is CW (−)",
            ha="right", va="center", fontsize=8,
            style="italic", color="#555555")
    ax.annotate("", xy=(-0.2, -1.5), xytext=(-0.2, -5.5),
                arrowprops=dict(arrowstyle="-[,widthB=2.5", color="#555555", lw=1.2),
                zorder=1)

    # --- Synapses (drawn behind nodes) ---
    for syn in synapses:
        if syn.pre not in NODE_POSITIONS or syn.post not in NODE_POSITIONS:
            continue
        x1, y1 = NODE_POSITIONS[syn.pre]
        x2, y2 = NODE_POSITIONS[syn.post]

        is_inhibitory = syn.Es < 0
        color = "#C0392B" if is_inhibitory else "#27AE60"
        lw    = 1.8 if is_inhibitory else 1.4

        # Slight curve for connections that would otherwise overlap
        rad = 0.0
        if syn.pre == "deriv_out_ccw" and syn.post == "coact_node":
            rad = -0.25
        elif syn.pre == "deriv_out_cw" and syn.post == "coact_node":
            rad =  0.25
        elif "coact" in syn.pre and "kc_prod" in syn.post:
            rad = 0.15 if "ccw" in syn.post else -0.15
        elif syn.pre == "ib_input":
            rad = 0.15 if "ccw" in syn.post else -0.15

        _draw_arrow(ax, x1, y1, x2, y2, color, lw, rad=rad)

    # --- Nodes ---
    for name in NODE_POSITIONS:
        if name not in neurons:
            continue
        x, y = NODE_POSITIONS[name]
        label = NODE_LABELS.get(name, name)
        color = NODE_COLORS.get(name, "#EEEEEE")
        _draw_node(ax, x, y, label, color)

    # --- Gain stage group labels (matching M&H's Kp, Kc, Kd, Kt labels) ---
    gain_label_positions = {
        "Kp": (11.75,  6.55),  "Kd": (11.75,  4.05),
        "Kc": (11.75,  2.05),  "Kt": (14.75,  4.05),
    }
    for label, (lx, ly) in gain_label_positions.items():
        ax.text(lx, ly, label,
                ha="center", va="center",
                fontsize=9, fontweight="bold", color="#2C3E50",
                bbox=dict(facecolor="#F9E79F", edgecolor="#B7950B",
                          boxstyle="round,pad=0.3", linewidth=1.2))

    # --- Ib input annotation ---
    ax.annotate("+Ib\n(tension\nproxy)",
                xy=(14.0, 0.0), xytext=(14.0, -1.8),
                ha="center", fontsize=7, color="#7F8C8D",
                arrowprops=dict(arrowstyle="->", color="#7F8C8D", lw=1.0))

    # --- Motor output arrows (exiting diagram) ---
    for side, (_, my) in [("CCW", (17.0, 3.5)), ("CW", (17.0, -3.5))]:
        ax.annotate(f"+a_{side}",
                    xy=(18.2, my), xytext=(17.6, my),
                    fontsize=8, fontweight="bold", va="center",
                    arrowprops=dict(arrowstyle="-|>", color="#E74C3C", lw=2.0))

    # --- Legend ---
    legend_patches = [
        mpatches.Patch(color="#A9DFBF", label="TC4 Sensory (new)"),
        mpatches.Patch(color="#F1948A", label="Error node"),
        mpatches.Patch(color="#85C1E9", label="Derivative fast/slow"),
        mpatches.Patch(color="#F0A500", label="Deriv output / |d|"),
        mpatches.Patch(color="#F9E79F", label="Bias modulator (bias-gated)"),
        mpatches.Patch(color="#C39BD3", label="Gain product (Kp/Kd/Kc/Kt)"),
        mpatches.Patch(color="#BDC3C7", label="Type-Ib input"),
        mpatches.Patch(color="#E74C3C", label="Motor output"),
        mpatches.Patch(color="#F39C12", label="Input (bilateral)"),
        mpatches.Patch(color="#27AE60", label="Excitatory synapse"),
        mpatches.Patch(color="#C0392B", label="Inhibitory synapse"),
    ]
    ax.legend(handles=legend_patches,
              loc="lower right", fontsize=7.5,
              framealpha=0.95, edgecolor="#AAAAAA",
              title="Legend", title_fontsize=8)

    # --- Title ---
    ax.set_title(
        "TC4 SNS Controller  —  McNeal & Hunt 2026 architecture + TC4 dual-channel sensory integration\n"
        f"38 neurons  |  46 synapses  "
        r"$\bullet$  $e = -W_g \cdot BS - W_p \cdot (BS - SS)$  "
        r"$\bullet$  split derivative: $K_d$ (directional) + $K_c$ (co-activation)  "
        r"$\bullet$  $K_t$ = Type-Ib",
        fontsize=10, pad=12,
    )

    ax.set_xlim(-1.5, 19.5)
    ax.set_ylim(-7.5, 7.5)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Network diagram saved to {output_file}")
    return output_file


# =====================================================================
# Synapse summary table
# =====================================================================
def print_synapse_table(neurons, synapses):
    """Print a readable table of all synapses."""
    print(f"\n{'Synapse name':<35} {'Pre':<20} {'Post':<20} {'gs_max (µS)':>12} {'Type':<6}")
    print("-" * 95)
    for syn in sorted(synapses, key=lambda s: s.name):
        syn_type = "INH" if syn.Es < 0 else "EXC"
        print(f"{syn.name:<35} {syn.pre:<20} {syn.post:<20} {syn.gs_max:>12.5f} {syn_type:<6}")
    print(f"\nTotal: {len(neurons)} neurons, {len(synapses)} synapses")


# =====================================================================
# Entry point
# =====================================================================
if __name__ == "__main__":
    neurons, synapses, bias_neuron_map = build_tc4_network()
    print_synapse_table(neurons, synapses)
    draw_network(neurons, synapses)
