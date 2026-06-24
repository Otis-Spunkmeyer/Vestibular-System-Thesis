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

from sns_tc4_controller import build_neurons, build_synapses, build_tc4_network, Er

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
# Node positions  — horizontal left-to-right layout
# Upper half (y > 0) = CCW pathway
# Lower half (y < 0) = CW  pathway
# y = 0 line  = shared nodes (coact, ib_input)
# =====================================================================
NODE_POSITIONS = {
    # --- Stage 1: bilateral inputs ---
    "bs_ccw":        (0.5,  3.3),   "bs_cw":        (0.5, -3.3),
    "ss_ccw":        (0.5,  1.3),   "ss_cw":        (0.5, -1.3),
    # Single shared node — theta_ref = 0, so no bilateral split needed.
    # Both synapses (→ error_ccw and → error_cw) branch from this one position.
    "theta_ref_ccw": (0.5,  0.0),   "theta_ref_cw": (0.5,  0.0),

    # --- Stage 2: TC4 sensory integration ---
    "sub_diff_ccw": (2.0,  4.2),   "sub_diff_cw": (2.0, -4.2),
    "wg_ccw":       (2.0,  2.2),   "wg_cw":       (2.0, -2.2),
    "wp_ccw":       (3.3,  4.2),   "wp_cw":       (3.3, -4.2),
    "error_ccw":    (4.7,  2.8),   "error_cw":    (4.7, -2.8),

    # --- Stage 3: derivative estimation ---
    "deriv_fast_ccw": (6.0,  4.2),  "deriv_fast_cw": (6.0, -4.2),
    "deriv_slow_ccw": (6.0,  2.2),  "deriv_slow_cw": (6.0, -2.2),
    "d_accel_ccw":    (7.3,  3.7),  "d_accel_cw":    (7.3, -3.7),
    "d_decel_ccw":    (7.3,  2.0),  "d_decel_cw":    (7.3, -2.0),

    # --- Stage 4: co-activation node (shared) ---
    "coact_node": (8.4, 0.0),

    # --- Stage 5 Kp: bias at y=5.2, CCW/CW at ±0.8 ---
    "kp_bias":    (9.5,  5.2),
    "kp_mod_ccw": (10.7,  6.0),  "kp_prod_ccw": (11.9,  6.0),
    "kp_mod_cw":  (10.7,  4.4),  "kp_prod_cw":  (11.9,  4.4),

    # --- Stage 5 Kd: bias at y=1.6, CCW/CW at ±0.8 ---
    "kd_bias":    (9.5,  1.6),
    "kd_mod_ccw": (10.7,  2.4),  "kd_prod_ccw": (11.9,  2.4),
    "kd_mod_cw":  (10.7,  0.8),  "kd_prod_cw":  (11.9,  0.8),

    # --- Stage 5 Kc: bias at y=-2.0, CCW/CW at ±0.8 ---
    "kc_bias":    (9.5, -2.0),
    "kc_mod_ccw": (10.7, -1.2),  "kc_prod_ccw": (11.9, -1.2),
    "kc_mod_cw":  (10.7, -2.8),  "kc_prod_cw":  (11.9, -2.8),

    # --- Stage 5 Kt: bias at y=1.6, CCW/CW at ±0.8 (same spread as Kd row) ---
    "kt_bias":    (12.5,  1.6),
    "kt_mod_ccw": (13.7,  2.4),  "kt_prod_ccw": (14.9,  2.4),
    "kt_mod_cw":  (13.7,  0.8),  "kt_prod_cw":  (14.9,  0.8),

    # --- Stage 6: Type-Ib input (between kt_mod and kt_prod columns, midpoint y) ---
    "ib_input": (14.3,  1.6),

    # --- Stage 7: ta at mean y of all inputs — kd→ta is horizontal, kp/kc symmetric ---
    "ta_ccw": (16.0,  2.4),
    "ta_cw":  (16.0,  0.8),
}

# =====================================================================
# Short display labels for each neuron
# =====================================================================
NODE_LABELS = {
    "bs_ccw": "BS+", "bs_cw": "BS−",
    "ss_ccw": "SS+", "ss_cw": "SS−",
    "theta_ref_ccw": "θref=0",   # single shared node; theta_ref_cw is skipped in drawing
    "sub_diff_ccw": "BS−SS+", "sub_diff_cw": "BS−SS−",
    "wg_ccw": "Wg·BS+",      "wg_cw": "Wg·BS−",
    "wp_ccw": "Wp·ΔΘ+",      "wp_cw": "Wp·ΔΘ−",
    "error_ccw": "e+",        "error_cw": "e−",
    "deriv_fast_ccw": "fast+", "deriv_fast_cw": "fast−",
    "deriv_slow_ccw": "slow+", "deriv_slow_cw": "slow−",
    "d_accel_ccw": "d_a+", "d_accel_cw": "d_a−",
    "d_decel_ccw": "d_d+", "d_decel_cw": "d_d−",
    "coact_node": "|d|",
    "kp_bias": "Kp",
    "kp_mod_ccw": "bp+", "kp_mod_cw": "bp−",
    "kp_prod_ccw": "Kp·e+", "kp_prod_cw": "Kp·e−",
    "kd_bias": "Kd",
    "kd_mod_ccw": "bd+", "kd_mod_cw": "bd−",
    "kd_prod_ccw": "Kd·d+", "kd_prod_cw": "Kd·d−",
    "kc_bias": "Kc",
    "kc_mod_ccw": "bc+", "kc_mod_cw": "bc−",
    "kc_prod_ccw": "Kc·|d|+", "kc_prod_cw": "Kc·|d|−",
    "kt_bias": "Kt",
    "kt_mod_ccw": "bt+", "kt_mod_cw": "bt−",
    "kt_prod_ccw": "Kt·T+", "kt_prod_cw": "Kt·T−",
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
    "theta_ref_ccw": "#F39C12", "theta_ref_cw": "#F39C12",

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
    "d_accel_ccw": "#F0A500",  "d_accel_cw": "#F0A500",
    "d_decel_ccw": "#F0A500",  "d_decel_cw": "#F0A500",
    "coact_node":  "#F0A500",

    # Shared gain neurons (bias parameter) — bright yellow (matches M&H Kp/Kd/Kc/Kt)
    "kp_bias": "#F4D03F", "kd_bias": "#F4D03F",
    "kc_bias": "#F4D03F", "kt_bias": "#F4D03F",

    # Intermediate (mod) neurons — pale yellow (receive I_app=R + inhibitory from bias)
    "kp_mod_ccw": "#FCF3CF", "kp_mod_cw": "#FCF3CF",
    "kd_mod_ccw": "#FCF3CF", "kd_mod_cw": "#FCF3CF",
    "kc_mod_ccw": "#FCF3CF", "kc_mod_cw": "#FCF3CF",
    "kt_mod_ccw": "#FCF3CF", "kt_mod_cw": "#FCF3CF",

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
    # TC4 Sensory Integration — light green
    (-0.2, -5.6, 5.3, 11.2,
     "#D5F5E3", "#27AE60",
     "TC4 Sensory Integration\n(new over McNeal & Hunt)", -5.3),

    # Error formation sub-zone
    (4.3, -3.1, 0.9, 6.2,
     "#A9DFBF", "#1E8449",
     "Error\nformation", 2.9),

    # Differential calculations — fast/slow, d_accel/d_decel, coact
    (5.7, -4.5, 3.1, 9.0,
     "#FDEBD0", "#E67E22",
     "Differential\nCalculations", -4.3),

    # Derivative Gain Circuits — Kp/Kd/Kc/Kt gain stages
    (9.1, -3.2, 6.3, 9.7,
     "#D6EAF8", "#2E86C1",
     "Derivative Gain Circuits", -3.0),

    # Type Ib feedback sub-zone — ib_input + Kt pathway
    (12.1, -0.4, 3.3, 3.2,
     "#E8E8E8", "#7F8C8D",
     "Type Ib\nfeedback", -0.2),
]

# =====================================================================
# Helper: draw a single rounded-rectangle node
# =====================================================================
def _draw_node(ax, x, y, label, color, node_w=0.68, node_h=0.42):
    box = FancyBboxPatch(
        (x - node_w / 2, y - node_h / 2), node_w, node_h,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor="#2C3E50",
        linewidth=0.8,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, label,
            ha="center", va="center",
            fontsize=5.2, fontweight="bold",
            color="#1A1A1A", zorder=4)


# =====================================================================
# Helper: draw a directed arrow between two node centers
# =====================================================================
def _draw_arrow(ax, x1, y1, x2, y2, color, lw, rad=0.0, label=None):
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
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.12, label,
                ha="center", va="bottom",
                fontsize=4.5, color=color, zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=0.5))


# =====================================================================
# Main draw function
# =====================================================================
def draw_network(neurons, synapses, output_file=None, log=None):
    """
    Render the TC4 SNS network in McNeal & Hunt 2026 diagram style.

    Args:
        neurons:     dict from build_neurons()
        synapses:    list from build_synapses()
        output_file: save path (defaults to SCRIPT_DIR/tc4_network_diagram.png)
        log:         optional voltage log from run_segment() — overlays final U values
    """
    if output_file is None:
        output_file = os.path.join(SCRIPT_DIR, "tc4_network_diagram.png")

    fig, ax = plt.subplots(figsize=(24, 16))
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
    ax.text(-0.5, 3.3, "Active when\nBS is CCW (+)",
            ha="right", va="center", fontsize=7.5,
            style="italic", color="#555555")
    ax.annotate("", xy=(-0.2, 5.5), xytext=(-0.2, 1.0),
                arrowprops=dict(arrowstyle="-[,widthB=2.2", color="#555555", lw=1.2),
                zorder=1)

    ax.text(-0.5, -3.3, "Active when\nBS is CW (−)",
            ha="right", va="center", fontsize=7.5,
            style="italic", color="#555555")
    ax.annotate("", xy=(-0.2, -1.0), xytext=(-0.2, -5.5),
                arrowprops=dict(arrowstyle="-[,widthB=2.2", color="#555555", lw=1.2),
                zorder=1)

    # --- Synapses (drawn behind nodes) ---
    for syn in synapses:
        if syn.pre not in NODE_POSITIONS or syn.post not in NODE_POSITIONS:
            continue
        x1, y1 = NODE_POSITIONS[syn.pre]
        x2, y2 = NODE_POSITIONS[syn.post]

        # Gate synapses (ΔE=0, Es=-60) are drawn in purple to distinguish from
        # standard inhibitory (Es=-100). Excitatory = green, standard inh = red.
        if syn.Es == Er:           # gating synapse (Mult. Syn 2, ΔE=0)
            color, lw = "#8E44AD", 1.6
        elif syn.Es < 0:           # standard inhibitory (ΔE=-40, Es=-100)
            color, lw = "#C0392B", 1.8
        else:                      # excitatory (ΔE=+194)
            color, lw = "#27AE60", 1.4

        # Slight curve for connections that would otherwise overlap
        rad = 0.0
        if syn.pre == "ib_input":
            rad = 0.1 if "ccw" in syn.post else -0.1
        # theta_ref is now a single shared node — curve the two branches so they
        # don't overlap (CCW branch curves up, CW branch curves down)
        if syn.pre in ("theta_ref_ccw", "theta_ref_cw"):
            rad = 0.25 if "ccw" in syn.post else -0.25

        # Extract "synN" label from the standardized synapse name "synN_pre_to_post"
        syn_label = syn.name.split("_")[0]  # e.g. "syn34" from "syn34_kp_mod_ccw_to_kp_prod_ccw"

        _draw_arrow(ax, x1, y1, x2, y2, color, lw, rad=rad, label=syn_label)

    # --- Nodes ---
    # Input neurons whose labels encode a signed angle via bilateral split; when a log
    # is present we show the true angle (U value) rather than the neuron voltage offset.
    INPUT_NEURON_NAMES = {
        "bs_ccw", "bs_cw", "ss_ccw", "ss_cw", "theta_ref_ccw", "theta_ref_cw",
    }
    for name in NODE_POSITIONS:
        if name not in neurons:
            continue
        # theta_ref_cw shares the same position as theta_ref_ccw (single shared node);
        # skip drawing it so only one box appears at that position.
        if name == "theta_ref_cw":
            continue
        x, y = NODE_POSITIONS[name]
        label = NODE_LABELS.get(name, name)
        # When a simulation log is supplied, append the final steady-state U value so
        # you can read each neuron's operating point directly on the diagram.
        if log is not None and name in log:
            U = log[name][-1] - Er   # convert stored voltage V → membrane potential U
            label += f"\nU={U:+.2f}"
        color = NODE_COLORS.get(name, "#EEEEEE")
        _draw_node(ax, x, y, label, color)

    # --- Ib input annotation ---
    ax.annotate("+Ib\n(tension\nproxy)",
                xy=(14.3, 1.6), xytext=(14.3, 0.2),
                ha="center", fontsize=6.5, color="#7F8C8D",
                arrowprops=dict(arrowstyle="->", color="#7F8C8D", lw=1.0))

    # --- Motor output arrows (exiting diagram) ---
    for side, my in [("CCW", 2.4), ("CW", 0.8)]:
        ax.annotate(f"+a_{side}",
                    xy=(17.0, my), xytext=(16.5, my),
                    fontsize=8, fontweight="bold", va="center",
                    arrowprops=dict(arrowstyle="-|>", color="#E74C3C", lw=2.0))

    # --- Legend ---
    legend_patches = [
        mpatches.Patch(color="#A9DFBF", label="TC4 Sensory (new)"),
        mpatches.Patch(color="#F1948A", label="Error node"),
        mpatches.Patch(color="#85C1E9", label="Derivative fast/slow"),
        mpatches.Patch(color="#F0A500", label="Deriv output / |d|"),
        mpatches.Patch(color="#F4D03F", label="Gain neuron (Kp/Kd/Kc/Kt) — I_app=b_alpha"),
        mpatches.Patch(color="#FCF3CF", label="Intermediate (mod) — I_app=R"),
        mpatches.Patch(color="#C39BD3", label="Product neuron (signal × gain)"),
        mpatches.Patch(color="#BDC3C7", label="Type-Ib input"),
        mpatches.Patch(color="#E74C3C", label="Motor output"),
        mpatches.Patch(color="#F39C12", label="Input (bilateral)"),
        mpatches.Patch(color="#27AE60", label="Excitatory synapse"),
        mpatches.Patch(color="#C0392B", label="Inhibitory synapse"),
        mpatches.Patch(color="#8E44AD", label="Gating synapse (ΔE=0, Es=Er)"),
    ]
    ax.legend(handles=legend_patches,
              loc="lower right", fontsize=7.5,
              framealpha=0.95, edgecolor="#AAAAAA",
              title="Legend", title_fontsize=8)

    # --- Title ---
    ax.set_title(
        "TC4 SNS Controller  —  McNeal & Hunt 2026 architecture + TC4 dual-channel sensory integration\n"
        r"46 neurons  |  64 synapses  "
        r"$\bullet$  $e = \theta_{ref} - W_g \cdot BS - W_p \cdot (BS - SS)$  "
        r"$\bullet$  split derivative: $K_d$ (directional) + $K_c$ (co-activation)  "
        r"$\bullet$  FSA gain: bias $\to$ mod $\to$ prod (double-inhibitory)",
        fontsize=9, pad=12,
    )

    ax.set_xlim(-1.5, 17.6)
    ax.set_ylim(-6.2, 7.2)
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
    # Use the decomposed API (same pattern as kp_error_segment.py) rather than the
    # backward-compat wrapper, so the neuron/synapse objects are available separately
    # for the table printer and the diagram renderer.
    neurons  = build_neurons()
    synapses = build_synapses()
    print_synapse_table(neurons, synapses)
    draw_network(neurons, synapses)
