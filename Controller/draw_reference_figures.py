"""
Reference figures for SNS Controller Design Review
Segments 1 & 2: Neuron model, Synapse model, Activation function, Substitution

Run:  python draw_reference_figures.py
Output: reference_seg1_neuron.png
        reference_seg2_activation.png
        reference_seg2_synapse.png
        reference_seg2_substitution.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Palette ────────────────────────────────────────────────────────────
BLACK  = '#111111'
GREY   = '#555555'
RED    = '#C0392B'
BLUE   = '#2471A3'
GREEN  = '#1E8449'
ORANGE = '#E67E22'
BG     = '#FAFAFA'

# ── Low-level drawing helpers ──────────────────────────────────────────
def W(ax, xs, ys, lw=1.8, c=BLACK):
    ax.plot(xs, ys, color=c, lw=lw, solid_capstyle='round', zorder=2)

def capacitor(ax, x, y0, y1, label='', val=''):
    m = (y0+y1)/2;  s = y1-y0
    pw = s*0.22;    g = s*0.07
    W(ax, [x,x], [y0, m-g])
    W(ax, [x,x], [m+g, y1])
    W(ax, [x-pw, x+pw], [m-g, m-g], lw=3.5)
    W(ax, [x-pw, x+pw], [m+g, m+g], lw=3.5)
    if label:
        ax.text(x-pw-0.14, m, label, ha='right', va='center',
                fontsize=13, style='italic', fontweight='bold')
    if val:
        ax.text(x+pw+0.14, m, val, ha='left', va='center',
                fontsize=11, color=GREY)

def resistor(ax, x, y0, y1, label='', val='', variable=False):
    m = (y0+y1)/2;  s = y1-y0
    rh = s*0.30;    rw = s*0.16
    W(ax, [x,x], [y0, m-rh/2])
    W(ax, [x,x], [m+rh/2, y1])
    fc = '#FFF3CD' if variable else 'white'
    ax.add_patch(mpatches.Rectangle(
        (x-rw/2, m-rh/2), rw, rh, fc=fc, ec=BLACK, lw=1.8, zorder=3))
    if variable:
        ax.text(x, m, '~', ha='center', va='center', fontsize=15, color=GREY)
    if label:
        ax.text(x-rw/2-0.14, m, label, ha='right', va='center',
                fontsize=13, style='italic', fontweight='bold')
    if val:
        ax.text(x+rw/2+0.14, m, val, ha='left', va='center',
                fontsize=11, color=GREY)

def battery(ax, x, y0, y1, label='', color=RED):
    m = (y0+y1)/2;  s = y1-y0
    lp = s*0.22;    sp = s*0.13;    g = s*0.08
    W(ax, [x,x], [y0, m-g])
    W(ax, [x,x], [m+g, y1])
    W(ax, [x-lp, x+lp], [m+g, m+g], lw=3.5)   # + plate (long)
    W(ax, [x-sp, x+sp], [m-g, m-g], lw=1.8)    # − plate (short)
    ax.text(x+lp+0.10, m+g, '+', fontsize=10, va='center', color=BLACK)
    ax.text(x+sp+0.10, m-g, '−', fontsize=11, va='center', color=BLACK)
    if label:
        ax.text(x-lp-0.14, m, label, ha='right', va='center',
                fontsize=13, style='italic', fontweight='bold', color=color)

def ground(ax, x, y):
    for i, w in enumerate([0.24, 0.15, 0.07]):
        ax.plot([x-w, x+w], [y - i*0.11, y - i*0.11], color=BLACK, lw=1.8-i*0.3)

def current_arrow(ax, x, y_tip, length=0.7, label=''):
    ax.annotate('', xy=(x, y_tip), xytext=(x, y_tip+length),
                arrowprops=dict(arrowstyle='->', color=RED, lw=2.4,
                                mutation_scale=18))
    if label:
        ax.text(x+0.15, y_tip+length/2, label, fontsize=13,
                va='center', color=RED, fontweight='bold')

def node_dot(ax, x, y):
    ax.add_patch(plt.Circle((x, y), 0.05, color=BLACK, zorder=5))

def rail(ax, x_lo, x_hi, y):
    W(ax, [x_lo, x_hi], [y, y], lw=2.0)


# ══════════════════════════════════════════════════════════════════════
# FIGURE 1 — Segment 1: Neuron circuit
# ══════════════════════════════════════════════════════════════════════
def fig_neuron():
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor(BG); fig.patch.set_facecolor(BG)

    TOP = 4.2;  BOT = 0.3
    x_cm = 1.2;  x_gm = 2.8;  x_app = 4.2

    # ── rails ──
    rail(ax, 0.5, 5.0, TOP)
    rail(ax, 0.5, 5.0, BOT)

    # ── ground ──
    ground(ax, 2.0, BOT - 0.05)

    # ── Cm branch ──
    capacitor(ax, x_cm, BOT, TOP, label='$C_m$', val='1 nF')

    # ── Gm branch (resistor + Er battery in series) ──
    # Split the vertical space:  bottom third = Er, top two thirds = Gm
    y_mid_gm = BOT + (TOP-BOT)*0.38
    resistor(ax, x_gm, y_mid_gm, TOP, label='$G_m$', val='1 µS')
    battery( ax, x_gm, BOT, y_mid_gm, label='$E_r$', color=RED)

    # ── I_app arrow ──
    current_arrow(ax, x_app, TOP, length=0.75, label='$I_{app}$')
    W(ax, [x_app, x_app], [TOP, TOP + 0.75], lw=1.8, c=RED)

    # ── node dot and label ──
    node_dot(ax, x_gm, TOP)
    node_dot(ax, x_cm, TOP)
    node_dot(ax, x_app, TOP)
    ax.text(x_app + 0.28, TOP + 0.06, '$V$', fontsize=15,
            fontweight='bold', va='bottom')

    # ── equation box ──
    eq_x = 0.55;  eq_y = TOP + 1.1
    eq_text = (
        r"$C_m \dfrac{dV}{dt} \;=\; G_m(E_r - V) \;+\; I_{syn} \;+\; I_{app}$"
    )
    ax.text(2.75, eq_y, eq_text, ha='center', va='center',
            fontsize=14, color=BLACK,
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec=BLUE, lw=1.5))

    # ── parameter table ──
    params = [
        ('$E_r$',   '−60 mV',  'Resting potential'),
        ('$R$',     '20 mV',   'Operating range'),
        ('$E_{hi}$','−40 mV',  'Full activation voltage'),
        ('$E_{lo}$','−60 mV',  'Zero activation (= $E_r$)'),
        ('$U$',     '$V - E_r$','Activation above rest (0 → 20 mV)'),
        ('$\\tau$', '$C_m/G_m$ = 1 ms', 'Membrane time constant'),
    ]
    ax.text(5.3, TOP + 0.1, 'Parameter values', fontsize=11,
            fontweight='bold', va='top', color=BLACK)
    for i, (sym, val, desc) in enumerate(params):
        y = TOP - 0.42 - i*0.55
        ax.text(5.3, y, f'{sym} = {val}', fontsize=10, va='center',
                color=BLACK)
        ax.text(5.3, y - 0.22, f'  {desc}', fontsize=9, va='center',
                color=GREY, style='italic')

    # ── annotations ──
    ax.annotate('Eq. 2\n$I_{leak}=G_m(E_r-V)$',
                xy=(x_gm, (BOT+TOP)/2), xytext=(x_gm+0.9, (BOT+TOP)/2),
                fontsize=9, color=GREY, ha='left', va='center',
                arrowprops=dict(arrowstyle='->', color=GREY, lw=0.9))
    ax.annotate('Eq. 1',
                xy=(2.75, eq_y - 0.35), xytext=(2.75, eq_y - 0.68),
                fontsize=9, color=BLUE, ha='center',
                arrowprops=dict(arrowstyle='->', color=BLUE, lw=0.9))

    ax.set_xlim(0.2, 8.0);  ax.set_ylim(-0.3, TOP + 1.9)
    ax.set_aspect('equal');  ax.axis('off')
    ax.set_title('Segment 1 — Non-Spiking Neuron Circuit Model\n'
                 'Szczecinski & Quinn 2017, Eq. 1 & 2',
                 fontsize=13, fontweight='bold', pad=8)

    out = os.path.join(SCRIPT_DIR, 'reference_seg1_neuron.png')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f'Saved: {out}')
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# FIGURE 2 — Segment 2a: Activation function a(V_pre)
# ══════════════════════════════════════════════════════════════════════
def fig_activation():
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_facecolor(BG); fig.patch.set_facecolor(BG)

    Elo = -60;  Ehi = -40;  R = 20
    V = np.linspace(-75, -25, 500)
    a = np.clip((V - Elo) / R, 0, 1)

    # ── shaded regions ──
    ax.axvspan(-75, Elo, alpha=0.12, color='#AED6F1', label='Silent  a = 0')
    ax.axvspan(Elo, Ehi, alpha=0.12, color='#A9DFBF', label='Linear  a = (V−Eₗₒ)/R')
    ax.axvspan(Ehi, -25, alpha=0.12, color='#F9E79F', label='Saturated  a = 1')

    ax.plot(V, a, color=BLUE, lw=2.8, zorder=3)

    # ── key vertical lines ──
    for xv, lbl, yoff in [(Elo, '$E_{lo} = -60$ mV', 0.55),
                           (Ehi, '$E_{hi} = -40$ mV', 0.55)]:
        ax.axvline(xv, color=GREY, lw=1.2, ls='--', zorder=1)
        ax.text(xv, yoff, lbl, ha='center', va='bottom', fontsize=11,
                color=GREY, style='italic')

    # ── R brace ──
    ax.annotate('', xy=(Ehi, -0.12), xytext=(Elo, -0.12),
                arrowprops=dict(arrowstyle='<->', color=BLACK, lw=1.5))
    ax.text((Elo+Ehi)/2, -0.17, '$R = E_{hi} - E_{lo} = 20$ mV',
            ha='center', va='top', fontsize=11, color=BLACK)

    # ── piecewise equations ──
    ax.text(-73, 0.08, '$a = 0$',
            fontsize=12, color='#2471A3', style='italic')
    ax.text(-52, 0.50, r'$a = \dfrac{V_{pre} - E_{lo}}{R}$',
            fontsize=12, color=GREEN, style='italic', ha='center')
    ax.text(-27, 0.92, '$a = 1$',
            fontsize=12, color=ORANGE, style='italic', ha='right')

    # ── Eq. 4 label ──
    ax.text(-25, 0.2,
            'Szczecinski & Quinn 2017\nEq. 4 (linear region)',
            ha='right', fontsize=10, color=GREY,
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=GREY, lw=1))

    ax.set_xlabel('$V_{pre}$  (mV)', fontsize=13)
    ax.set_ylabel('$a(V_{pre})$  — synapse activation  (0 → 1)', fontsize=12)
    ax.set_title('Segment 2 — Piecewise-Linear Activation Function\n'
                 'Controls how much current the synapse passes',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(-75, -24);  ax.set_ylim(-0.25, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(axis='y', ls=':', alpha=0.4)
    ax.spines[['top','right']].set_visible(False)

    out = os.path.join(SCRIPT_DIR, 'reference_seg2_activation.png')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f'Saved: {out}')
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# FIGURE 3 — Segment 2b: Neuron + Synapse circuit
# ══════════════════════════════════════════════════════════════════════
def fig_synapse():
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_facecolor(BG); fig.patch.set_facecolor(BG)

    TOP = 4.2;  BOT = 0.3
    x_cm = 1.0;  x_gm = 2.6;  x_syn = 4.4;  x_app = 5.9

    rail(ax, 0.3, 6.7, TOP)
    rail(ax, 0.3, 6.7, BOT)
    ground(ax, 3.2, BOT - 0.05)

    # ── Cm branch ──
    capacitor(ax, x_cm, BOT, TOP, label='$C_m$', val='1 nF')

    # ── Gm + Er branch ──
    y_split = BOT + (TOP-BOT)*0.38
    resistor(ax, x_gm, y_split, TOP, label='$G_m$', val='1 µS')
    battery( ax, x_gm, BOT, y_split, label='$E_r$')

    # ── Synapse branch (variable Gs + Es) ──
    y_split_s = BOT + (TOP-BOT)*0.38
    resistor(ax, x_syn, y_split_s, TOP,
             label='$G_s$', val='$g_s \\cdot a(V_{pre})$', variable=True)
    battery( ax, x_syn, BOT, y_split_s, label='$E_s$', color=GREEN)

    # ── I_app arrow ──
    current_arrow(ax, x_app, TOP, length=0.75, label='$I_{app}$')
    W(ax, [x_app, x_app], [TOP, TOP+0.75], lw=1.8, c=RED)

    # ── node dots ──
    for xd in [x_cm, x_gm, x_syn, x_app]:
        node_dot(ax, xd, TOP)
    ax.text(x_app+0.28, TOP+0.06, '$V_{post}$', fontsize=14,
            fontweight='bold', va='bottom')

    # ── V_pre control arrow ──
    ax.annotate('$V_{pre}$ controls\nconductance\n(Eq. 4)',
                xy=(x_syn - 0.12, (TOP+BOT)/2 + 0.1),
                xytext=(x_syn - 1.5, (TOP+BOT)/2 + 1.2),
                fontsize=10, ha='center', color=GREEN,
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.6,
                                connectionstyle='arc3,rad=-0.2'),
                bbox=dict(boxstyle='round,pad=0.3', fc='#D5F5E3', ec=GREEN, lw=1))

    # ── excitatory / inhibitory labels ──
    ax.text(x_syn + 1.35, TOP*0.85,
            'Excitatory:\n$E_s = +134$ mV\n$\\Delta E_s = +194$ mV\n→ current IN',
            fontsize=10, ha='left', va='top', color=GREEN,
            bbox=dict(boxstyle='round,pad=0.4', fc='#D5F5E3', ec=GREEN, lw=1))
    ax.text(x_syn + 1.35, TOP*0.40,
            'Inhibitory:\n$E_s = -100$ mV\n$\\Delta E_s = -40$ mV\n→ current OUT',
            fontsize=10, ha='left', va='top', color=RED,
            bbox=dict(boxstyle='round,pad=0.4', fc='#FADBD8', ec=RED, lw=1))

    # ── full equation ──
    ax.text(3.5, TOP+1.75,
            r'$C_m\,\dfrac{dV}{dt} = G_m(E_r - V_{post})'
            r' + g_s \cdot a(V_{pre}) \cdot (E_s - V_{post}) + I_{app}$',
            ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.55', fc='white', ec=BLUE, lw=1.5))
    ax.text(3.5, TOP+1.18,
            r'$\leftarrow$ Eq. 2 (leak) $\quad\quad\quad$'
            r'$\leftarrow$ Eq. 3 with $G_s$ from Eq. 4 $\quad\quad$'
            r'$\leftarrow$ applied',
            ha='center', va='center', fontsize=9, color=GREY)

    ax.set_xlim(0.0, 9.0);  ax.set_ylim(-0.35, TOP+2.2)
    ax.set_aspect('equal');  ax.axis('off')
    ax.set_title('Segment 2 — Neuron with Synapse (Eq. 3 + Eq. 4)\n'
                 'Synapse adds a third parallel branch — conductance set by $V_{pre}$',
                 fontsize=13, fontweight='bold', pad=8)

    out = os.path.join(SCRIPT_DIR, 'reference_seg2_synapse.png')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f'Saved: {out}')
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# FIGURE 4 — Segment 2c: Equation substitution walkthrough
# ══════════════════════════════════════════════════════════════════════
def fig_substitution():
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_facecolor(BG); fig.patch.set_facecolor(BG)
    ax.axis('off')

    title_kw  = dict(fontsize=13, fontweight='bold', color=BLACK, ha='left')
    eq_kw     = dict(fontsize=13, ha='left', va='top')
    note_kw   = dict(fontsize=10, ha='left', va='top', color=GREY, style='italic')
    arrow_kw  = dict(arrowstyle='->', lw=1.4, color=GREY)

    def box(ax, text, x, y, fc, ec, fs=12):
        ax.text(x, y, text, ha='center', va='center', fontsize=fs,
                bbox=dict(boxstyle='round,pad=0.45', fc=fc, ec=ec, lw=1.5))

    # ── Title ──
    ax.text(0.03, 0.97, 'Equation Substitution: Eq. 4  →  Eq. 3',
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            color=BLACK, ha='left')
    ax.text(0.03, 0.91, 'Szczecinski & Quinn 2017',
            transform=ax.transAxes, fontsize=10, color=GREY)

    steps = [
        # (y_pos, label, eq_text, note, fg_color, bg_color)
        (0.80, 'STEP 1 — Eq. 3:  synaptic current',
         r'$I_{syn} \;=\; G_{s,i} \;\cdot\; (E_{s,i} - V)$',
         r'$G_{s,i}$ is not a constant — it changes with $V_{pre}$  (→ Eq. 4)',
         BLACK, 'white', BLUE),

        (0.60, 'STEP 2 — Eq. 4:  what $G_{s,i}$ actually is  (linear region)',
         r'$G_{s,i} \;=\; g_{s,i} \;\cdot\; \dfrac{V_{pre} - E_{lo}}{E_{hi} - E_{lo}} \;=\; g_{s,i} \;\cdot\; \dfrac{V_{pre} - E_{lo}}{R}$',
         r'$g_{s,i}$ = fixed maximum conductance    |    the fraction = how open the synapse is',
         BLACK, 'white', GREEN),

        (0.38, 'STEP 3 — Substitute Eq. 4 into Eq. 3',
         r'$I_{syn} \;=\; g_{s,i} \;\cdot\; \dfrac{V_{pre}-E_{lo}}{R} \;\cdot\; (E_{s,i} - V)$',
         r'[fixed max gs]  ×  [Eq.4: how open — 0→1]  ×  [driving force]',
         BLACK, '#EAF2FF', BLUE),

        (0.20, 'STEP 4 — Shorthand form  (used in code and lectures)',
         r'$I_{syn} \;=\; g_{s,i} \;\cdot\; a(V_{pre}) \;\cdot\; (E_{s,i} - V)$'
         '\n'
         r'where  $a(V_{pre}) = \text{clip}\!\left(\dfrac{V_{pre}-E_{lo}}{R},\;0,\;1\right)$'
         '\n'
         r'"$a$" is not in the paper — it is shorthand for the Eq. 4 fraction.',
         '',
         BLACK, '#EAFAF1', GREEN),
    ]

    for (yf, lbl, eq, note, tc, fc, ec) in steps:
        ax.text(0.03, yf+0.045, lbl, transform=ax.transAxes,
                fontsize=11, fontweight='bold', color=ec, va='top')
        ax.text(0.05, yf-0.01, eq, transform=ax.transAxes,
                **eq_kw, color=tc,
                bbox=dict(boxstyle='round,pad=0.5', fc=fc, ec=ec, lw=1.4))
        ax.text(0.05, yf-0.095, note, transform=ax.transAxes, **note_kw)

    # ── limit table ──
    ax.text(0.72, 0.56, 'Limit check for $a(V_{pre})$',
            transform=ax.transAxes, fontsize=11, fontweight='bold', color=BLACK)
    rows = [
        ('$V_{pre} = E_{lo}$', r'$a = \frac{0}{R} = 0$', 'Synapse fully closed'),
        ('$V_{pre} = E_{hi}$', r'$a = \frac{R}{R} = 1$', 'Synapse fully open'),
        ('$V_{pre}$ between',  r'$0 < a < 1$',           'Partial activation'),
    ]
    for i,(cond,res,desc) in enumerate(rows):
        y = 0.49 - i*0.10
        ax.text(0.72, y, cond, transform=ax.transAxes, fontsize=10, va='center')
        ax.text(0.83, y, res,  transform=ax.transAxes, fontsize=10, va='center', color=BLUE)
        ax.text(0.92, y, desc, transform=ax.transAxes, fontsize=9,  va='center', color=GREY)

    ax.set_title('Segment 2 — How $g_s \\cdot a(V_{pre}) \\cdot (E_s - V)$ arises from Eq. 3 + Eq. 4',
                 fontsize=13, fontweight='bold', pad=10)

    out = os.path.join(SCRIPT_DIR, 'reference_seg2_substitution.png')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f'Saved: {out}')
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# FIGURE 5 — Segment 3: Steady-state derivation + master equation
# ══════════════════════════════════════════════════════════════════════
def fig_steady_state():
    fig = plt.figure(figsize=(15, 12))
    fig.patch.set_facecolor(BG)

    # ── Layout: left = derivation steps, right = worked example ──
    gs_layout = fig.add_gridspec(1, 2, width_ratios=[1.15, 0.85],
                                  wspace=0.08, left=0.03, right=0.97,
                                  top=0.91, bottom=0.03)
    ax_L = fig.add_subplot(gs_layout[0])
    ax_R = fig.add_subplot(gs_layout[1])
    for ax in [ax_L, ax_R]:
        ax.set_facecolor(BG)
        ax.axis('off')

    fig.suptitle('Segment 3 — Steady-State Analysis: Deriving the Master Equation  U*\n'
                 'Szczecinski & Quinn 2017, Eq. 6 → 11 → 12 → 13',
                 fontsize=13, fontweight='bold', y=0.98)

    # ── colour scheme for each step ──
    step_colors = [BLUE, GREEN, ORANGE, RED]
    step_bgs    = ['#EAF2FF', '#EAFAF1', '#FEF9E7', '#FDEDEC']

    steps = [
        ('Eq. 6  —  Membrane equation in U-space\n'
         '(substitute U = V−Er, Gm = 1, ΔEs = Es−Er, Gs = gs/R·Upre)',
         r'$C_m\,\dfrac{dU}{dt} \;=\; -U \;+\; \sum_i \dfrac{g_{s,i}}{R}'
         r'\cdot U_{pre,i} \cdot (\Delta E_{s,i} - U) \;+\; I_{app}$'),

        ('Eq. 11  —  Set dU/dt = 0  (capacitor fully charged at steady state)',
         r'$0 \;=\; -U^* \;+\; \sum_i \dfrac{g_{s,i}}{R}'
         r'\cdot U_{pre,i} \cdot (\Delta E_{s,i} - U^*) \;+\; I_{app}$'),

        ('Eq. 12  —  Expand  (ΔEs − U*)  and collect all U* terms on left',
         r'$U^*\!\left(1 + \sum_i \dfrac{g_{s,i}}{R} U_{pre,i}\right)'
         r'\;=\; \sum_i \dfrac{g_{s,i}}{R} U_{pre,i} \Delta E_{s,i}'
         r'\;+\; I_{app}$'),

        ('Eq. 13  —  Divide both sides  →  MASTER EQUATION',
         r'$U^* \;=\; \dfrac{\sum_i \dfrac{g_{s,i}}{R}'
         r'\cdot U_{pre,i} \cdot \Delta E_{s,i} \;+\; I_{app}}'
         r'{1 \;+\; \sum_i \dfrac{g_{s,i}}{R} \cdot U_{pre,i}}$'),
    ]

    y_positions = [0.90, 0.68, 0.46, 0.20]

    for (lbl, eq), yf, sc, sbg in zip(steps, y_positions, step_colors, step_bgs):
        ax_L.text(0.02, yf + 0.06, lbl, transform=ax_L.transAxes,
                  fontsize=10, fontweight='bold', color=sc, va='top')
        ax_L.text(0.04, yf, eq, transform=ax_L.transAxes,
                  fontsize=11.5, va='top', ha='left',
                  bbox=dict(boxstyle='round,pad=0.5', fc=sbg, ec=sc, lw=1.6))

    # ── arrows between steps ──
    for y_top, y_bot in zip(y_positions[:-1], y_positions[1:]):
        ax_L.annotate('', xy=(0.50, y_bot + 0.10), xytext=(0.50, y_top - 0.02),
                      xycoords='axes fraction', textcoords='axes fraction',
                      arrowprops=dict(arrowstyle='->', color=GREY, lw=1.6))

    # ── annotate master equation terms ──
    ax_L.text(0.04, 0.06,
              'Numerator  =  weighted sum of "where each synapse pulls the voltage"  +  applied drive\n'
              'Denominator  =  1 (leak)  +  total synaptic conductance loading',
              transform=ax_L.transAxes,
              fontsize=9.5, color=GREY, style='italic', va='top')

    # ══ RIGHT PANEL — Worked example ══════════════════════════════════
    ax_R.text(0.50, 0.97, 'Worked Example', transform=ax_R.transAxes,
              fontsize=12, fontweight='bold', color=BLACK, ha='center', va='top')
    ax_R.text(0.50, 0.91,
              'One excitatory synapse, no $I_{app}$\n'
              r'$g_s = 0.115\,\mu S,\quad \Delta E_s = 194\,\text{mV}$'
              '\n'
              r'$R = 20\,\text{mV},\quad U_{pre} = R = 20\,\text{mV}$  (full activation)',
              transform=ax_R.transAxes, fontsize=10.5, ha='center', va='top',
              bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=BLUE, lw=1.3))

    # plug-in steps
    calc_steps = [
        (r'$\dfrac{g_s}{R} \cdot U_{pre} \;=\; \dfrac{0.115}{20} \cdot 20 \;=\; 0.115$',
         'effective conductance at full activation'),
        (r'Numerator $= 0.115 \times 194 + 0 = 22.31$',
         'synapse pulls toward ΔEs = 194 mV'),
        (r'Denominator $= 1 + 0.115 = 1.115$',
         'leak + one synapse loading'),
        (r'$U^* = \dfrac{22.31}{1.115} \approx 20.0\,\text{mV} = R$',
         'output ≈ input  →  gain ≈ 1  ✓'),
    ]

    calc_colors = [BLUE, GREEN, ORANGE, RED]
    calc_bgs    = ['#EAF2FF', '#EAFAF1', '#FEF9E7', '#FDEDEC']

    y = 0.72
    for (eq, note), cc, cbg in zip(calc_steps, calc_colors, calc_bgs):
        ax_R.text(0.50, y, eq, transform=ax_R.transAxes,
                  fontsize=11, ha='center', va='top',
                  bbox=dict(boxstyle='round,pad=0.4', fc=cbg, ec=cc, lw=1.4))
        ax_R.text(0.50, y - 0.075, note, transform=ax_R.transAxes,
                  fontsize=9, ha='center', va='top', color=GREY, style='italic')
        y -= 0.155

    # ── ksyn interpretation ──
    ax_R.text(0.50, 0.13,
              'This is the transmission synapse design:\n'
              r'$k_{syn} = U^*/U_{pre} \approx 1$ when $g_s = 0.115\,\mu S$'
              '\n'
              r'(Eq. 17: $k_{syn} = g_s \cdot \Delta E_s \;/\; R(1+g_s)$)',
              transform=ax_R.transAxes, fontsize=10, ha='center', va='top',
              bbox=dict(boxstyle='round,pad=0.45', fc='#F4F6F7', ec=GREY, lw=1.2))

    out = os.path.join(SCRIPT_DIR, 'reference_seg3_steady_state.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f'Saved: {out}')
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# FIGURE 6 — Segment 4: FSA primitive subnetworks
# ══════════════════════════════════════════════════════════════════════
def fig_fsa_subnetworks():
    from matplotlib.patches import Circle

    EXC  = '#C0392B'
    INH  = '#2980B9'
    NF   = '#F8F9FA'
    NE   = '#2C3E50'

    fig, axes = plt.subplots(2, 2, figsize=(18, 15))
    fig.patch.set_facecolor(BG)
    fig.suptitle('Segment 4 — FSA Primitive Subnetworks\n'
                 'Analytically designed via master equation (Szczecinski & Quinn 2017, Eq. 13)',
                 fontsize=13, fontweight='bold', y=0.99)

    for ax in axes.flat:
        ax.set_facecolor(BG)
        ax.axis('off')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

    def neuron(ax, x, y, label, r=0.55, fc=NF):
        ax.add_patch(Circle((x, y), r, fc=fc, ec=NE, lw=1.8, zorder=3))
        ax.text(x, y, label, ha='center', va='center', fontsize=8,
                fontweight='bold', zorder=4)

    def synapse(ax, x1, y1, x2, y2, color, label='', r=0.55):
        dx, dy = x2 - x1, y2 - y1
        dist = (dx**2 + dy**2) ** 0.5
        sx, sy = x1 + dx/dist*r, y1 + dy/dist*r
        ex, ey = x2 - dx/dist*r, y2 - dy/dist*r
        ax.annotate('', xy=(ex, ey), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.0))
        if label:
            ax.text((sx+ex)/2 + 0.2, (sy+ey)/2 + 0.25, label,
                    fontsize=7.5, color=color, va='center')

    def input_arrow(ax, x, y, label):
        ax.annotate('', xy=(x, y), xytext=(x - 1.1, y),
                    arrowprops=dict(arrowstyle='->', color=GREY, lw=1.5))
        ax.text(x - 1.25, y + 0.22, label, fontsize=8.5, color=GREY, ha='center')

    def step(ax, text, y, color, bg, fs=9):
        ax.text(5.0, y, text, ha='center', va='top', fontsize=fs,
                bbox=dict(boxstyle='round,pad=0.4', fc=bg, ec=color, lw=1.4))

    def arr(ax, y0, y1):
        ax.annotate('', xy=(5.0, y1), xytext=(5.0, y0),
                    arrowprops=dict(arrowstyle='->', color=GREY, lw=1.4))

    # ── A: Transmission ───────────────────────────────────────────────
    ax = axes[0, 0]
    ax.text(0.3, 9.75, r'A — Transmission   (solve for $g_s$)',
            fontsize=11, fontweight='bold', color=BLUE)

    input_arrow(ax, 1.8, 8.5, r'$U_{pre}$')
    neuron(ax, 2.5, 8.5, r'$U_{pre}$')
    neuron(ax, 7.5, 8.5, r'$U_{post}$')
    synapse(ax, 2.5, 8.5, 7.5, 8.5, EXC, r'$g_s^{trans}$  (exc)')

    step(ax,
         r'Eq. 13, $I_{app}=0$, one exc. synapse:' '\n'
         r'$U^* = \dfrac{g_s/R \cdot U_{pre} \cdot \Delta E_s}{1 + g_s/R \cdot U_{pre}}$',
         7.55, BLUE, '#EAF2FF')
    arr(ax, 6.4, 5.95)
    step(ax,
         r'Set $U^* = k\cdot U_{pre}$ at design point $U_{pre}=R/2$, solve for $g_s$:' '\n'
         r'$g_s = \dfrac{k \cdot R}{\Delta E_s - k \cdot R/2}$',
         5.85, GREEN, '#EAFAF1')
    arr(ax, 4.7, 4.25)
    step(ax,
         r'TC4:   $k=0.30 \to g_s=0.031\,\mu S$  (wg_node,  Wg $\times$ BS)' '\n'
         r'$k=0.70 \to g_s=0.075\,\mu S$  (wp_node,  Wp $\times$ (BS$-$SS))' '\n'
         r'$k=1.00 \to g_s=0.115\,\mu S$  (unity transmission)',
         4.15, ORANGE, '#FEF9E7', fs=8.5)

    # ── B: Addition ───────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.text(0.3, 9.75, r'B — Addition   (solve for $g_e$)',
            fontsize=11, fontweight='bold', color=GREEN)

    input_arrow(ax, 1.8, 8.7, r'$U_a$')
    input_arrow(ax, 1.8, 7.7, r'$U_b$')
    neuron(ax, 2.5, 8.7, r'$U_a$')
    neuron(ax, 2.5, 7.7, r'$U_b$')
    neuron(ax, 7.5, 8.2, r'$U_{sum}$')
    synapse(ax, 2.5, 8.7, 7.5, 8.2, EXC, r'$g_e$  (+)')
    synapse(ax, 2.5, 7.7, 7.5, 8.2, EXC, r'$g_e$  (+)')

    step(ax,
         r'Eq. 13, $I_{app}=0$, two identical exc. synapses:' '\n'
         r'$U^* = \dfrac{(g_e/R)\,U_a\Delta E_e + (g_e/R)\,U_b\Delta E_e}'
         r'{1 + (g_e/R)\,U_a + (g_e/R)\,U_b}$',
         6.85, BLUE, '#EAF2FF', fs=8.5)
    arr(ax, 5.85, 5.5)
    step(ax,
         r'Factor $(g_e\Delta E_e/R)$ from numerator — same form as transmission:' '\n'
         r'$U^* = \dfrac{(g_e\Delta E_e/R)(U_a+U_b)}{1+(g_e/R)(U_a+U_b)}$'
         r'  with  $S = U_a+U_b$',
         5.4, GREEN, '#EAFAF1', fs=8.5)
    arr(ax, 4.55, 4.2)
    step(ax,
         r'Want $U^*=S$, i.e. $k=1$.  Set $U^*=k\cdot S$, cancel $S$, cross-multiply:' '\n'
         r'$1+(g_e/R)\cdot S = g_e\Delta E_e/R$  $\Rightarrow$  at $S=R$: $g_e = R\,/\,(\Delta E_e-R)$',
         4.1, ORANGE, '#FEF9E7', fs=8.5)
    arr(ax, 3.3, 2.95)
    step(ax,
         r'$g_e = 20\,/\,(194-20) = 20/174 = 0.115\,\mu S$   both synapses identical  ✓',
         2.85, RED, '#FDEDEC', fs=8.5)

    # ── C: Subtraction ────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.text(0.3, 9.75, r'C — Subtraction   (solve for $g_{inh}$)',
            fontsize=11, fontweight='bold', color=ORANGE)

    input_arrow(ax, 1.8, 8.7, r'$U_a$')
    input_arrow(ax, 1.8, 7.7, r'$U_b$')
    neuron(ax, 2.5, 8.7, r'$U_a$')
    neuron(ax, 2.5, 7.7, r'$U_b$')
    neuron(ax, 7.5, 8.2, r'$U_{diff}$')
    synapse(ax, 2.5, 8.7, 7.5, 8.2, EXC, r'$g_e$  (+)')
    synapse(ax, 2.5, 7.7, 7.5, 8.2, INH, r'$g_{inh}$  (−)')

    step(ax,
         r'Eq. 13, $I_{app}=0$, exc ($\Delta E_e=+194$) + inh ($\Delta E_{inh}=-40$):' '\n'
         r'$U^* = \dfrac{(g_e/R)\,U_a(+194) + (g_{inh}/R)\,U_b(-40)}'
         r'{1 + (g_e/R)\,U_a + (g_{inh}/R)\,U_b}$',
         6.85, BLUE, '#EAF2FF', fs=8.5)
    arr(ax, 5.85, 5.5)
    step(ax,
         r'Want $U^*=0$ when $U_a=U_b$ (equal inputs cancel). Set numerator $= 0$:' '\n'
         r'$g_e\cdot\Delta E_e + g_{inh}\cdot\Delta E_{inh} = 0$',
         5.4, GREEN, '#EAFAF1', fs=8.5)
    arr(ax, 4.8, 4.45)
    step(ax,
         r'Rearrange: $g_{inh} = -g_e\cdot\Delta E_e\;/\;\Delta E_{inh} = g_e\cdot\Delta E_e\;/\;|\Delta E_{inh}|$' '\n'
         r'$= 0.115\times194\;/\;|-40| = 0.115\times194\;/\;40$',
         4.35, ORANGE, '#FEF9E7', fs=8.5)
    arr(ax, 3.55, 3.2)
    step(ax,
         r'$g_{inh} = 22.31\;/\;40 = 0.558\,\mu S$   TC4: sub_diff = BS $-$ SS  ✓',
         3.1, RED, '#FDEDEC', fs=8.5)

    # ── D: Differentiation ────────────────────────────────────────────
    ax = axes[1, 1]
    ax.text(0.3, 9.75, r'D — Differentiation   (solve for $C_m$)',
            fontsize=11, fontweight='bold', color=RED)

    input_arrow(ax, 1.25, 8.7, r'$U_{in}$')
    input_arrow(ax, 1.25, 7.7, r'$U_{in}$')
    neuron(ax, 2.1, 8.7, r'$U_{fast}$', r=0.55, fc='#FDEBD0')
    neuron(ax, 2.1, 7.7, r'$U_{slow}$', r=0.55, fc='#D6EAF8')
    neuron(ax, 7.5, 8.2, r'$U_{out}$')
    ax.text(2.75, 9.0, r'$\tau_f$', fontsize=8.5, color='#A04000', fontweight='bold')
    ax.text(2.75, 7.5, r'$\tau_s$', fontsize=8.5, color='#1A5276', fontweight='bold')
    synapse(ax, 2.1, 8.7, 7.5, 8.2, EXC, r'$g_e$  (+)', r=0.55)
    synapse(ax, 2.1, 7.7, 7.5, 8.2, INH, r'$g_{inh}$  (−)', r=0.55)

    step(ax,
         r'Eq. 13: same synapses $\Rightarrow U_{fast}^*=U_{slow}^* \Rightarrow U_{out}^*=0$ at SS.' '\n'
         r'$g_e$/$g_{inh}$ use standard values.  Design parameter is $C_m$, not $g_s$.',
         6.85, BLUE, '#EAF2FF', fs=8.5)
    arr(ax, 5.9, 5.55)
    step(ax,
         r'$H(j\omega) \approx j\omega(\tau_s-\tau_f)$ for $\omega \ll 1/\tau_s$  — true differentiator.' '\n'
         r'Bandwidth rule: derivative valid up to $f_{max} = 1/(2\pi\tau_s)$.' '\n'
         r'Select $\tau_s$ so $1/\tau_s \gg 2\pi f_{max}$;  set $\tau_f \ll \tau_s$.',
         5.45, GREEN, '#EAFAF1', fs=8.5)
    arr(ax, 4.4, 4.05)
    step(ax,
         r'TC4: $f_{max}\approx5\,\mathrm{Hz}$  →  need $\tau_s \ll 32\,\mathrm{ms}$.' '\n'
         r'Choose $\tau_s=8\,\mathrm{ms}$, $\tau_f=0.1\,\mathrm{ms}$  ($80\times$ apart).' '\n'
         r'$C_m = \tau\cdot G_m$:  $C_m^{fast}=0.1\,\mathrm{nF}$,  $C_m^{slow}=8\,\mathrm{nF}$  ✓',
         3.95, ORANGE, '#FEF9E7', fs=8.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(SCRIPT_DIR, 'reference_seg4_fsa_subnetworks.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f'Saved: {out}')
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# FIGURE 7 — Segment 5: Peterka 2002 sensory reweighting + Eq. 15
# ══════════════════════════════════════════════════════════════════════
def fig_peterka_eq15():
    PURPLE = '#7D3C98'
    PURPLEBG = '#F5EEF8'

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor(BG)
    gs_layout = fig.add_gridspec(2, 1, height_ratios=[1, 1.8], hspace=0.07,
                                  left=0.02, right=0.98, top=0.93, bottom=0.02)
    ax_D = fig.add_subplot(gs_layout[0])
    ax_A = fig.add_subplot(gs_layout[1])
    for ax in [ax_D, ax_A]:
        ax.set_facecolor(BG)
        ax.axis('off')
    ax_D.set_xlim(0, 16); ax_D.set_ylim(0, 5)
    ax_A.set_xlim(0, 16); ax_A.set_ylim(0, 10)

    fig.suptitle('Segment 5 — Peterka 2002 Sensory Reweighting Model\n'
                 r'$H_{CL}(s)=\theta_{b}(s)/\theta_{s}(s)$  —  TC4: $W_g=0.30$, $W_p=0.70$, $W_v=0$',
                 fontsize=13, fontweight='bold', y=0.98)

    # ── Drawing helpers ──────────────────────────────────────────────
    def blk(ax, cx, cy, w, h, txt, fc, ec, fs=8.5):
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx-w/2, cy-h/2), w, h, boxstyle='round,pad=0.15',
            fc=fc, ec=ec, lw=1.6, zorder=3))
        ax.text(cx, cy, txt, ha='center', va='center', fontsize=fs, zorder=4)

    def harr(ax, x1, x2, y, lbl='', lc=BLACK):
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color=lc, lw=1.8))
        if lbl:
            ax.text((x1+x2)/2, y+0.22, lbl, ha='center', fontsize=8.5, color=lc)

    # ════ BLOCK DIAGRAM ════════════════════════════════════════════════
    ax_D.text(0.1, 3.25, r'$\theta_{s}$', fontsize=11, color=BLUE,
              fontweight='bold', va='center')
    harr(ax_D, 0.55, 2.0, 3.25)

    blk(ax_D, 3.3, 2.8, 2.6, 2.1,
        'Sensory\nWeights\n' r'$W_g\cdot\theta_b$' '\n' r'$W_p\cdot(\theta_b-\theta_s)$',
        '#EAF2FF', BLUE, fs=8)

    harr(ax_D, 4.6, 5.9, 2.8, r'$e$')

    blk(ax_D, 7.3, 2.8, 2.6, 1.4,
        r'$C(s)\cdot e^{-\tau_d s}$' '\n' r'$(K_p+K_d s)\cdot e^{-\tau_d s}$',
        '#EAFAF1', GREEN, fs=8)

    harr(ax_D, 8.6, 9.9, 2.8, r'$T$')

    blk(ax_D, 11.1, 2.8, 2.2, 1.4,
        r'$P(s)$' '\n' r'$1/(Js^2\!-\!mgh)$',
        '#FDEDEC', RED, fs=8)

    harr(ax_D, 12.2, 13.4, 2.8)
    ax_D.text(13.55, 2.8, r'$\theta_b$', fontsize=11, color=RED,
              fontweight='bold', va='center')

    # Feedback path
    ax_D.plot([13.4, 13.4], [2.8, 1.0], color=BLACK, lw=1.8)
    ax_D.plot([13.4, 2.0],  [1.0, 1.0], color=BLACK, lw=1.8)
    ax_D.annotate('', xy=(2.0, 1.7), xytext=(2.0, 1.0),
                  arrowprops=dict(arrowstyle='->', color=BLACK, lw=1.8))
    ax_D.text(7.5, 0.5, r'$\theta_b \to BS$ (body sway — fed back into sensory block)',
              ha='center', fontsize=8.5, color=GREY, style='italic')

    # θ_surface also enters sensory block (SS input)
    ax_D.plot([0.55, 0.55], [3.25, 1.45], color=BLUE, lw=1.4, ls='dashed')
    ax_D.plot([0.55, 2.0],  [1.45, 1.45], color=BLUE, lw=1.4, ls='dashed')
    ax_D.annotate('', xy=(2.0, 1.7), xytext=(2.0, 1.45),
                  arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.4))
    ax_D.text(1.15, 1.8, r'$SS$', fontsize=9, color=BLUE)

    # TC4 weight box
    ax_D.add_patch(mpatches.FancyBboxPatch(
        (13.8, 1.4), 2.0, 2.7, boxstyle='round,pad=0.15',
        fc='#FEF9E7', ec=ORANGE, lw=1.5))
    for i, line in enumerate(['TC4',
                               r'$W_g=0.30$', r'$W_p=0.70$',
                               r'$W_v=0$  (eyes closed)',
                               r'$W_g+W_p=1.00$']):
        fw = 'bold' if i == 0 else 'normal'
        ax_D.text(14.8, 3.85 - i*0.48, line, ha='center', fontsize=8.5,
                  color=ORANGE if i == 0 else BLACK, fontweight=fw)

    # ════ ALGEBRAIC DERIVATION (two columns) ══════════════════════════
    ax_A.plot([8.0, 8.0], [0.4, 9.8], color='#CCCCCC', lw=1.0, ls='dashed')
    ax_A.text(4.0, 9.85, 'Derivation', fontsize=10.5, fontweight='bold',
              color=BLACK, ha='center')
    ax_A.text(12.0, 9.85, 'TC4 Result & Interpretation',
              fontsize=10.5, fontweight='bold', color=BLACK, ha='center')

    def stepL(y, hdr, txt, sc, sbg):
        ax_A.text(0.3, y+0.3, hdr, fontsize=8.5, fontweight='bold', color=sc, va='top')
        ax_A.text(4.0, y, txt, ha='center', va='top', fontsize=9,
                  bbox=dict(boxstyle='round,pad=0.4', fc=sbg, ec=sc, lw=1.4))

    def stepR(y, hdr, txt, sc, sbg):
        ax_A.text(8.3, y+0.3, hdr, fontsize=8.5, fontweight='bold', color=sc, va='top')
        ax_A.text(12.0, y, txt, ha='center', va='top', fontsize=9,
                  bbox=dict(boxstyle='round,pad=0.4', fc=sbg, ec=sc, lw=1.4))

    def arrL(y0, y1):
        ax_A.annotate('', xy=(4.0, y1), xytext=(4.0, y0),
                      arrowprops=dict(arrowstyle='->', color=GREY, lw=1.4))

    def arrR(y0, y1):
        ax_A.annotate('', xy=(12.0, y1), xytext=(12.0, y0),
                      arrowprops=dict(arrowstyle='->', color=GREY, lw=1.4))

    # ── Left column: setup ──────────────────────────────────────────
    stepL(9.2, 'Step 1 — Sensory error  ($W_v=0$ in TC4)',
          r'$e = -W_g\theta_b - W_p(\theta_b-\theta_s)$' '\n'
          r'$= -(W_g+W_p)\,\theta_b \;+\; W_p\,\theta_s$',
          BLUE, '#EAF2FF')
    arrL(8.1, 7.75)
    stepL(7.65, 'Step 2 — Plant equation (inverted pendulum, linearised)',
          r'$(Js^2-mgh)\,\theta_b = T = C(s)\,e^{-\tau_d s}\cdot e$',
          GREEN, '#EAFAF1')
    arrL(6.85, 6.5)
    stepL(6.4, r'Step 3 — Substitute $e$, collect $\theta_b$ on left',
          r'$[Js^2-mgh+(W_g+W_p)\,Ce^{-\tau_d s}]\,\theta_b = W_p\,Ce^{-\tau_d s}\,\theta_s$',
          ORANGE, '#FEF9E7')

    # ── Left column annotation: SNS mapping ─────────────────────────
    ax_A.text(4.0, 5.4,
              r'$e = -0.30\,\theta_b - 0.70(\theta_b-\theta_s)$  ← SNS: wg_node + wp_node',
              ha='center', fontsize=8.5, color=GREY, style='italic',
              bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#CCCCCC', lw=1))

    # ── Right column: result ────────────────────────────────────────
    stepR(9.2, 'Step 4 — Divide both sides  →  Eq. 15',
          r'$H_{CL}(s) = \dfrac{W_p\cdot C(s)\cdot e^{-\tau_d s}}'
          r'{Js^2-mgh+(W_g+W_p)\cdot C(s)\cdot e^{-\tau_d s}}$',
          RED, '#FDEDEC')
    arrR(7.7, 7.35)
    stepR(7.25, r'TC4 annotation  ($W_p=0.70$,  $W_g+W_p=1.00$,  $C=K_p+K_d s$)',
          r'$H_{CL}(s) = \dfrac{0.70\,(K_p+K_d s)\,e^{-\tau_d s}}'
          r'{Js^2-mgh + 1.00\,(K_p+K_d s)\,e^{-\tau_d s}}$',
          PURPLE, PURPLEBG)
    arrR(5.9, 5.55)
    stepR(5.45, 'Physical interpretation',
          'Numerator factor 0.70: surface perturbation only 70% effective\n'
          r'  (graviception $W_g=0.30$ partially overrides it)' '\n'
          r'Denominator: $W_g+W_p=1$ $\Rightarrow$ no net gain loss, system is stabilisable' '\n'
          r'McNeal & Hunt: PD only ($K_i=0$), delay $\tau_d=90\,\mathrm{ms}$',
          GREY, '#F0F0F0')

    out = os.path.join(SCRIPT_DIR, 'reference_seg5_peterka_eq15.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f'Saved: {out}')
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# FIGURE 8 — Segment 6: McNeal & Hunt split-derivative architecture
# ══════════════════════════════════════════════════════════════════════
def fig_mcneal_split_deriv():
    from matplotlib.patches import Circle
    PURPLE   = '#7D3C98'
    PURPLEBG = '#F5EEF8'
    EXC_C    = '#C0392B'
    INH_C    = '#2980B9'

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.patch.set_facecolor(BG)
    fig.suptitle('Segment 6 — McNeal & Hunt 2026 Split-Derivative Architecture\n'
                 'Bilateral half-wave rectification  +  bias-gated gain stages  +  Type-Ib feedback',
                 fontsize=13, fontweight='bold', y=0.99)

    for ax in axes.flat:
        ax.set_facecolor(BG)
        ax.axis('off')

    def bx(ax, cx, cy, w, h, txt, fc, ec, fs=8.5):
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx-w/2, cy-h/2), w, h, boxstyle='round,pad=0.15',
            fc=fc, ec=ec, lw=1.6, zorder=3))
        ax.text(cx, cy, txt, ha='center', va='center', fontsize=fs, zorder=4)

    def harr(ax, x1, x2, y, lbl='', lc=BLACK):
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color=lc, lw=1.8))
        if lbl:
            ax.text((x1+x2)/2, y+0.2, lbl, ha='center', fontsize=8, color=lc)

    def varr(ax, x, y1, y2, lc=BLACK):
        ax.annotate('', xy=(x, y2), xytext=(x, y1),
                    arrowprops=dict(arrowstyle='->', color=lc, lw=1.8))

    def circ(ax, x, y, lbl, r=0.45, fc='#F8F9FA'):
        ax.add_patch(Circle((x, y), r, fc=fc, ec='#2C3E50', lw=1.8, zorder=3))
        ax.text(x, y, lbl, ha='center', va='center', fontsize=7.5,
                fontweight='bold', zorder=4)

    def syn_arrow(ax, x1, y1, x2, y2, color, lbl='', r=0.45):
        dx, dy = x2-x1, y2-y1
        dist = (dx**2+dy**2)**0.5
        sx, sy = x1+dx/dist*r, y1+dy/dist*r
        ex, ey = x2-dx/dist*r, y2-dy/dist*r
        ax.annotate('', xy=(ex, ey), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.0))
        if lbl:
            ax.text((sx+ex)/2+0.2, (sy+ey)/2+0.2, lbl, fontsize=7.5, color=color)

    def stbx(ax, cx, y, txt, sc, sbg, fs=8.5):
        ax.text(cx, y, txt, ha='center', va='top', fontsize=fs,
                bbox=dict(boxstyle='round,pad=0.4', fc=sbg, ec=sc, lw=1.4))

    # ══ Panel A — 7-Stage Pipeline ════════════════════════════════════
    ax = axes[0, 0]
    ax.set_xlim(0, 11); ax.set_ylim(0, 10)
    ax.text(0.3, 9.75, 'A — 7-Stage Network Pipeline  (46 neurons, 64 synapses)',
            fontsize=11, fontweight='bold', color=BLUE)

    stages = [
        (1.0,  'Stage 1\nBilateral\nInput\n(6 n)',     BLUE,   '#EAF2FF', 1.5),
        (2.85, 'Stage 2\nTC4 Sensory\n(8 n × 2)',      GREEN,  '#EAFAF1', 1.4),
        (4.65, 'Stage 3\nDerivative\n(8 n)',           ORANGE, '#FEF9E7', 1.4),
        (6.2,  'Stage 4\nCo-act.\n(1 n)',              RED,    '#FDEDEC', 1.1),
        (7.85, 'Stage 5\nBias Gain\n(20 n)',           PURPLE, PURPLEBG,  1.6),
        (9.4,  'Stage 6\nType-Ib\n(1 n)',              GREY,   '#F0F0F0', 1.1),
        (10.55,'Stage 7\nOutput\n(2 n)',                BLACK,  '#F4F6F7', 1.0),
    ]

    for x, lbl, ec, fc, w in stages:
        bx(ax, x, 6.5, w, 2.2, lbl, fc, ec, fs=7.5)

    arrow_segs = [
        (1.0+0.75, 2.85-0.7),
        (2.85+0.7, 4.65-0.7),
        (4.65+0.7, 6.2-0.55),
        (6.2+0.55, 7.85-0.8),
        (7.85+0.8, 9.4-0.55),
        (9.4+0.55, 10.55-0.5),
    ]
    for x1, x2 in arrow_segs:
        harr(ax, x1, x2, 6.5)

    ax.annotate('', xy=(1.0-0.75, 6.5), xytext=(0.05, 6.5),
                arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.4))
    ax.text(0.05, 6.85, 'BS\nSS', fontsize=7.5, color=BLUE, ha='left')
    ax.annotate('', xy=(10.98, 6.5), xytext=(10.55+0.5, 6.5),
                arrowprops=dict(arrowstyle='->', color=BLACK, lw=1.4))
    ax.text(10.95, 6.7, r'$T_a$', fontsize=8, color=BLACK, ha='left')

    labels_below = [
        (1.0,  'bs_ccw, bs_cw\nss_ccw, ss_cw\ntheta_ref_ccw/cw'),
        (2.85, 'sub_diff\nwg, wp, error'),
        (4.65, 'deriv_fast\nderiv_slow\nd_accel, d_decel'),
        (6.2,  'coact_node'),
        (7.85, 'kp/kd/kc/kt\nmod + prod'),
        (9.4,  'ib_input'),
        (10.55,'ta_ccw\nta_cw'),
    ]
    for x, lbl in labels_below:
        ax.text(x, 5.2, lbl, ha='center', fontsize=6.2, color=GREY, va='top')

    ax.text(5.5, 3.4, '← Stages 2–7 are bilateral  (CCW and CW mirrored, identical structure)',
            ha='center', fontsize=8, color=GREY, style='italic')

    # ══ Panel B — Bilateral Split ══════════════════════════════════════
    ax = axes[0, 1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.text(0.3, 9.75, 'B — Bilateral Half-Wave Rectification',
            fontsize=11, fontweight='bold', color=GREEN)

    ax.text(1.7, 9.2, r'$\theta_A$ (rad)', ha='center', fontsize=9, color=BLACK, va='top')
    bx(ax, 1.7, 8.25, 2.6, 0.85,
       r'$I = \mathrm{clip}(g_A\theta_A,\;\pm10\,\mathrm{nA})$', '#EAF2FF', BLUE, fs=7.5)
    ax.text(1.7, 7.7, r'$g_A = 180/\pi \approx 57.3$ nA/rad',
            ha='center', fontsize=7.5, color=GREY)
    varr(ax, 1.7, 7.82, 7.3)
    bx(ax, 1.7, 6.9, 2.4, 0.75, 'split_to_bilateral( )', '#EAFAF1', GREEN, fs=8)
    ax.plot([1.7, 1.7], [6.52, 6.1],  color=BLACK, lw=1.8)
    ax.plot([1.7, 0.7], [6.1,  6.1],  color=BLACK, lw=1.8)
    ax.plot([1.7, 2.7], [6.1,  6.1],  color=BLACK, lw=1.8)
    varr(ax, 0.7, 6.1, 5.35, BLUE)
    varr(ax, 2.7, 6.1, 5.35, RED)
    bx(ax, 0.7, 4.9, 1.8, 0.85, 'bs_ccw\nmax(0, I)', '#EAF2FF', BLUE, fs=8)
    bx(ax, 2.7, 4.9, 1.8, 0.85, 'bs_cw\nmax(0, −I)', '#FDEDEC', RED, fs=8)
    ax.text(0.7, 4.3, 'I ≥ 0 only', ha='center', fontsize=7.5, color=BLUE, style='italic')
    ax.text(2.7, 4.3, 'I ≥ 0 only', ha='center', fontsize=7.5, color=RED, style='italic')
    ax.text(1.7, 3.7, '→ into Stage 2\n(both sides proceed identically)',
            ha='center', fontsize=7.5, color=GREY, va='top')

    ax.text(6.5, 9.2, 'Signal representation', ha='center', fontsize=8.5,
            fontweight='bold', color=BLACK, va='top')
    t_p = np.linspace(0, 4*np.pi, 300)
    sig_p  = np.sin(t_p)
    ccw_p  = np.clip(sig_p,  0, None)
    cw_p   = np.clip(-sig_p, 0, None)
    x0, xw = 4.5, 5.0
    xsc = xw / (4*np.pi)
    for j, (data, color, lbl) in enumerate([(sig_p,  BLACK, r'$I$ (signed)'),
                                             (ccw_p,  BLUE,  'CCW'),
                                             (cw_p,   RED,   'CW')]):
        yc = 8.3 - j * 1.8
        xs_p = x0 + t_p * xsc
        ys_p = yc + data * 0.65
        ax.fill_between(xs_p, yc, ys_p, alpha=0.2, color=color)
        ax.plot(xs_p, ys_p, color=color, lw=1.4)
        ax.plot([x0, x0+xw], [yc, yc], color=GREY, lw=0.7, ls=':')
        ax.text(x0-0.3, yc, lbl, ha='right', fontsize=8, color=color, va='center')

    ax.text(6.5, 2.8,
            'Only one side active at any instant.\n'
            'coact_node = CCW + CW = |d(t)|  (natural rectification).',
            ha='center', fontsize=8.5, color=GREY, style='italic',
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#CCCCCC', lw=1))

    # ══ Panel C — Motor Command Equation ══════════════════════════════
    ax = axes[1, 0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.text(0.3, 9.75, 'C — Motor Command Equation',
            fontsize=11, fontweight='bold', color=ORANGE)

    ax.text(5.0, 9.2,
            r'$a(t) \;=\; K_p\cdot e(t) \;+\; K_d\cdot d(t) \;+\; K_c\cdot|d(t)| \;+\; K_t\cdot I_b(t)$',
            ha='center', va='top', fontsize=10.5,
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec=BLACK, lw=2.0))

    terms = [
        (r'$K_p\!\cdot\!e$',  'Proportional',     'kp_mod\nkp_prod',
         'corrects\ncurrent lean',        r'$b_p=4.26$ nA', BLUE,   '#EAF2FF'),
        (r'$K_d\!\cdot\!d$',  'Directional\nDeriv.', 'kd_mod\nkd_prod',
         'phase lead,\nsigned direction', r'$b_d=5.01$ nA', GREEN,  '#EAFAF1'),
        (r'$K_c\!\cdot\!|d|$','Co-activation',    'kc_mod\nkc_prod',
         'symmetric\nstiffening',          r'$b_c=2.48$ nA', ORANGE, '#FEF9E7'),
        (r'$K_t\!\cdot\!I_b$','Type-Ib\nFeedback','kt_mod\nkt_prod',
         'Golgi tendon\npropr. damp',      r'$b_t=5.42$ nA', RED,    '#FDEDEC'),
    ]
    for i, (sym, name, neurons_lbl, desc, bias, color, bg) in enumerate(terms):
        x = 1.1 + i * 2.3
        ax.text(x, 8.3, sym, ha='center', va='top', fontsize=9.5, color=color,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', fc=bg, ec=color, lw=1.8))
        ax.text(x, 7.35, name, ha='center', fontsize=8, color=color,
                fontweight='bold', va='top')
        ax.text(x, 6.5,  neurons_lbl, ha='center', fontsize=7.5, color=GREY,
                style='italic', va='top')
        ax.text(x, 5.75, desc, ha='center', fontsize=7.5, color=GREY, va='top')
        ax.text(x, 4.75, bias, ha='center', fontsize=8.5, color=color, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc=bg, ec=color, lw=1.2))

    stbx(ax, 5.0, 3.9,
         r'$d(t)$ from Stage 3: $d = U_{fast} - U_{slow}$  (derivative subnetwork)' '\n'
         r'$|d(t)|$ from Stage 4: coact_node = deriv_out_ccw + deriv_out_cw  (natural rectification)',
         GREY, '#F0F0F0', fs=8)
    stbx(ax, 5.0, 2.5,
         r'Net torque: $T_a = L_{moment}(F_{CCW} - F_{CW})$  '
         r'(Hill model placeholder — see TODO in sns_tc4_controller.py)',
         BLACK, '#F4F6F7', fs=8.5)

    # ══ Panel D — Bias-Gated Gain Stage ═══════════════════════════════
    ax = axes[1, 1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.text(0.3, 9.75, r'D — Bias-Gated Gain Stage  (how $K_p, K_d, K_c, K_t$ are set)',
            fontsize=11, fontweight='bold', color=RED)

    circ(ax, 1.5, 7.5, 'signal', fc='#EAF2FF')
    circ(ax, 1.5, 5.3, 'mod',    fc='#FEF9E7')
    circ(ax, 5.0, 6.4, 'prod',   fc='#F5EEF8')

    ax.annotate('', xy=(1.5, 5.75), xytext=(1.5, 6.55),
                arrowprops=dict(arrowstyle='->', color=ORANGE, lw=2.2))
    ax.text(2.1, 6.2, r'$I_{app}=b_\alpha$', fontsize=9, color=ORANGE, va='center')

    ax.annotate('', xy=(1.05, 7.5), xytext=(0.05, 7.5),
                arrowprops=dict(arrowstyle='->', color=GREY, lw=1.5))
    ax.text(0.0, 7.8, 'signal\n(e, d, |d|,\nor Ib)', fontsize=7.5,
            color=GREY, ha='left', va='top')

    syn_arrow(ax, 1.5, 7.5, 5.0, 6.4, EXC_C, r'$g_e$ (exc)')
    syn_arrow(ax, 1.5, 5.3, 5.0, 6.4, ORANGE, r'$g_e$ (exc)')

    ax.annotate('', xy=(6.2, 6.4), xytext=(5.45, 6.4),
                arrowprops=dict(arrowstyle='->', color=GREY, lw=1.5))
    ax.text(6.3, 6.55, '→ ta_ccw\nor ta_cw', fontsize=7.5, color=GREY, va='center')

    stbx(ax, 5.5, 5.2,
         r'Mod neuron SS:  $U^*_{mod} \propto b_\alpha$' '\n'
         r'(bias current sets the operating point of the gain stage)',
         ORANGE, '#FEF9E7', fs=8.5)
    stbx(ax, 5.5, 3.8,
         r'Prod neuron:  $U^*_{prod} \propto U_{signal} \times U_{mod}$' '\n'
         r'Effective gain $K \propto b_\alpha$  —  tunable without changing $g_s$',
         RED, '#FDEDEC', fs=8.5)

    ax.text(5.5, 9.25, 'Bias values  (McNeal & Hunt Table IV, Split + kt)',
            ha='center', fontsize=8.5, fontweight='bold', color=BLACK, va='top')
    for i, (sym, val, desc, color) in enumerate([
            (r'$b_p$', '4.26 nA', 'proportional',       BLUE),
            (r'$b_d$', '5.01 nA', 'directional deriv.',  GREEN),
            (r'$b_c$', '2.48 nA', 'co-activation',       ORANGE),
            (r'$b_t$', '5.42 nA', 'Type-Ib tension',     RED)]):
        y = 8.65 - i * 0.52
        ax.text(3.8,  y, sym,         ha='left', fontsize=9,   color=color,
                fontweight='bold', va='center')
        ax.text(4.55, y, f'= {val}',  ha='left', fontsize=8.5, color=BLACK, va='center')
        ax.text(6.1,  y, desc,        ha='left', fontsize=8,   color=GREY,
                va='center', style='italic')

    stbx(ax, 5.5, 2.6,
         r'Only $b_p, b_d, b_c, b_t$ are free parameters.  All $g_s$ fixed analytically (FSA).' '\n'
         r'Optimized via PSO to match Peterka 2002 frequency-response data.',
         GREY, '#F0F0F0', fs=8.5)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(SCRIPT_DIR, 'reference_seg6_split_deriv.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f'Saved: {out}')
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# FIGURE 9 — Segment 6b: Bias transistor analogy + why split derivative
# ══════════════════════════════════════════════════════════════════════
def fig_seg6b_bias_split():
    PURPLE = '#7D3C98'

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle('Segment 6b — Bias-Gated Gain: Transistor Analogy  +  Why Split Derivative Is Necessary',
                 fontsize=13, fontweight='bold', y=0.99)

    for ax in axes.flat:
        ax.set_facecolor(BG)
        ax.axis('off')

    def dbl_arr(ax, x1, y1, x2, y2, lc=GREY, lw=1.5):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='<->', color=lc, lw=lw))

    def sarr(ax, x1, y1, x2, y2, lc=BLACK, lw=1.5):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=lc, lw=lw))

    def stbx(ax, cx, y, txt, sc, sbg, fs=9):
        ax.text(cx, y, txt, ha='center', va='top', fontsize=fs,
                bbox=dict(boxstyle='round,pad=0.4', fc=sbg, ec=sc, lw=1.4))

    # ══ Panel A — Transistor operating curve ══════════════════════════
    ax = axes[0, 0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.text(0.3, 9.75, 'A — Transistor: why the bias operating point matters',
            fontsize=11, fontweight='bold', color=BLUE)

    # Axis lines
    sarr(ax, 0.8, 1.5, 9.5, 1.5, BLACK, 1.5)
    sarr(ax, 0.8, 1.5, 0.8, 9.2, BLACK, 1.5)
    ax.text(9.7, 1.5, r'$I_B$  (base current, input)',  fontsize=9, va='center')
    ax.text(0.8, 9.5, r'$I_C$  (collector current, output)', fontsize=9, ha='left')

    # Transfer curve
    x_cut    = np.array([0.8, 2.0]);       y_cut    = np.array([1.5, 1.5])
    x_active = np.linspace(2.0, 7.0, 60); y_active = 1.5 + (x_active-2.0)*1.2
    x_sat    = np.linspace(7.0, 9.2, 30); y_sat    = 7.5 + (x_sat-7.0)*0.25

    ax.plot(x_cut,    y_cut,    color=RED,    lw=2.5)
    ax.plot(x_active, y_active, color=GREEN,  lw=2.5)
    ax.plot(x_sat,    y_sat,    color=ORANGE, lw=2.5)

    ax.text(1.4, 2.1, 'Cutoff\n(off)',       fontsize=8, color=RED,    ha='center', fontweight='bold')
    ax.text(4.0, 7.2, 'Active region',        fontsize=8, color=GREEN,  ha='center', fontweight='bold')
    ax.text(4.0, 6.7, r'$I_C = \beta \times I_B$', fontsize=8, color=GREEN, ha='center')
    ax.text(8.5, 8.7, 'Saturation\n(fully on)', fontsize=8, color=ORANGE, ha='center', fontweight='bold')

    # Bias operating point
    bx_pt, by_pt = 4.5, 1.5 + (4.5-2.0)*1.2   # = 4.5, 4.5
    ax.plot(bx_pt, by_pt, 'o', color=BLUE, markersize=10, zorder=5)
    ax.plot([bx_pt, bx_pt], [1.5, by_pt], color=BLUE, lw=1.2, ls='--')
    ax.plot([0.8, bx_pt],   [by_pt, by_pt], color=BLUE, lw=1.2, ls='--')
    ax.text(bx_pt, 1.1, r'$I_{B,bias}$', ha='center', fontsize=8.5, color=BLUE)
    ax.text(bx_pt+0.35, by_pt+0.25, 'Operating\npoint', fontsize=7.5, color=BLUE)

    # Signal swing spans
    sw_x = 1.1;  sw_y = sw_x * 1.2
    dbl_arr(ax, bx_pt-sw_x, 1.28, bx_pt+sw_x, 1.28, GREY, 1.5)
    ax.text(bx_pt, 0.85, r'small $\Delta I_B$ (input signal swing)',
            ha='center', fontsize=7.5, color=GREY)
    dbl_arr(ax, 0.6, by_pt-sw_y, 0.6, by_pt+sw_y, GREY, 1.5)
    ax.text(1.05, by_pt, r'large $\Delta I_C$ (amplified output)',
            ha='left', fontsize=7.5, color=GREY, va='center')

    # Without bias annotation
    ax.text(2.3, 3.5,
            'Without bias:\noperating point\nat cutoff (off).\nNegative signal\nhalf invisible.',
            fontsize=7.5, color=RED, ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='#FDEDEC', ec=RED, lw=1))

    # ══ Panel B — SNS bias equivalent ════════════════════════════════
    ax = axes[0, 1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.text(0.3, 9.75, r'B — SNS equivalent: $b_\alpha$ sets the operating point',
            fontsize=11, fontweight='bold', color=GREEN)

    sarr(ax, 0.8, 1.5, 7.2, 1.5, BLACK, 1.5)
    sarr(ax, 0.8, 1.5, 0.8, 8.8, BLACK, 1.5)
    ax.text(7.5, 1.5, r'$b_\alpha$ (nA)', fontsize=9.5, va='center')
    ax.text(0.8, 9.1, r'$U_{mod}$ (mV)', fontsize=9.5, ha='center')

    b_vals = np.linspace(0, 6.0, 100)
    x_b  = 0.8 + b_vals          # 0 nA→x=0.8, 6 nA→x=6.8
    y_um = 1.5 + b_vals          # 0 mV→y=1.5, 6 mV→y=7.5
    ax.plot(x_b, y_um, color=GREEN, lw=2.5)

    # Axis tick labels
    ax.text(0.8, 1.1, '0', ha='center', fontsize=8, color=GREY)
    ax.text(6.8, 1.1, '6 nA', ha='center', fontsize=8, color=GREY)
    ax.text(0.5, 1.5, '0', ha='right', fontsize=8, color=GREY, va='center')
    ax.text(0.5, 7.5, '6 mV', ha='right', fontsize=8, color=GREY, va='center')

    ax.text(4.5, 7.8, r'$U_{mod} = b_\alpha \;/\; G_m$',
            fontsize=9.5, color=GREEN, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', fc='#EAFAF1', ec=GREEN, lw=1))
    ax.text(4.5, 7.1, r'($G_m = 1\,\mu S$  $\Rightarrow$  slope = 1)',
            fontsize=8, color=GREY, ha='center')

    # b_p operating point
    bp = 4.26
    xp, yp = 0.8 + bp, 1.5 + bp
    ax.plot(xp, yp, 'o', color=ORANGE, markersize=10, zorder=5)
    ax.plot([xp, xp], [1.5, yp], color=ORANGE, lw=1.2, ls='--')
    ax.plot([0.8, xp], [yp, yp], color=ORANGE, lw=1.2, ls='--')
    ax.text(xp, 1.1, r'$b_p{=}4.26$', ha='center', fontsize=8, color=ORANGE)
    ax.text(0.5, yp, '4.26 mV', ha='right', fontsize=8, color=ORANGE, va='center')

    # Comparison table
    rows = [
        ('Transistor',   'SNS mod neuron',    BLACK),
        ('base bias',    r'$b_\alpha$ (nA)',   GREY),
        ('cutoff → off', 'b=0 → silent',      RED),
        ('active region','b=4.26 → active',   GREEN),
        ('AC signal',    r'$U_{signal}$',       BLUE),
        ('amplified out',r'$U^*_{prod}$',       ORANGE),
    ]
    ax.text(8.5, 9.2, 'Transistor  ↔  SNS', ha='center', fontsize=9,
            fontweight='bold', color=BLACK)
    ax.plot([7.2, 9.8], [8.8, 8.8], color='#CCCCCC', lw=1)
    for i, (left, right, color) in enumerate(rows):
        y = 8.5 - i * 0.52
        ax.text(7.5, y, left,  ha='left',  fontsize=8, color=GREY,  va='center')
        ax.text(9.5, y, right, ha='right', fontsize=8, color=color, va='center')
        ax.text(8.5, y, '↔',   ha='center', fontsize=8, color='#CCCCCC', va='center')

    # ══ Panel C — Single neuron failure ═══════════════════════════════
    ax = axes[1, 0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.text(0.3, 9.75, 'C — Why a single derivative neuron fails',
            fontsize=11, fontweight='bold', color=RED)

    t = np.linspace(0, 4*np.pi, 500)
    theta = np.sin(t)
    d_dt  = np.cos(t)

    x0, xw = 1.2, 8.2
    xsc = xw / (4*np.pi)
    xs_p = x0 + t * xsc

    for yc, data, color, lbl, sublbl in [
            (8.8, theta,              BLUE,  r'$\theta(t)$',    'body lean (signed)'),
            (7.3, d_dt,               GREEN, r'$d(t)$',          'true derivative (signed)'),
            (5.6, np.clip(d_dt,0,None), RED, 'single\nneuron',  'clips at U=0')]:
        ax.fill_between(xs_p, yc, yc + data*0.5, alpha=0.2, color=color)
        ax.plot(xs_p, yc + data*0.5, color=color, lw=1.5)
        ax.plot([x0, x0+xw], [yc, yc], color=GREY, lw=0.7, ls=':')
        ax.text(x0-0.15, yc+0.15, lbl,    ha='right', fontsize=8,   color=color, va='center')
        ax.text(x0-0.15, yc-0.28, sublbl, ha='right', fontsize=6.5, color=GREY,  va='center')

    # Shade the lost negative regions
    d_neg  = np.clip(d_dt, None, 0)
    ax.fill_between(xs_p, 5.6, 5.6 + d_neg*0.5,
                    alpha=0.45, color=RED, hatch='//', label='lost info')

    # Annotations for lost info
    for t_mid in [np.pi, 3*np.pi]:
        xm = x0 + t_mid * xsc
        ax.text(xm, 5.1, 'LOST', ha='center', fontsize=7, color=RED,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='#FDEDEC', ec=RED, lw=0.8))

    stbx(ax, 5.0, 4.4,
         'SNS voltages cannot go below $E_r$ (U = 0 floor).\n'
         'Negative derivative = leaning is slowing down or reversing.\n'
         'Without this info the controller cannot apply timely phase lead.',
         RED, '#FDEDEC', fs=8.5)

    # ══ Panel D — Split solution + natural |d| ════════════════════════
    ax = axes[1, 1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.text(0.3, 9.75, 'D — Split solution: full information  +  free |d(t)|',
            fontsize=11, fontweight='bold', color=GREEN)

    t = np.linspace(0, 4*np.pi, 500)
    d_dt   = np.cos(t)
    d_ccw  = np.clip(d_dt,  0, None)
    d_cw_v = np.clip(-d_dt, 0, None)
    d_abs  = np.abs(d_dt)

    x0, xw = 1.4, 7.8
    xsc = xw / (4*np.pi)
    xs_p = x0 + t * xsc

    rows_d = [
        (9.1, d_dt,   GREY,  r'$d(t)$',               'signed input'),
        (7.6, d_ccw,  BLUE,  'deriv_out_ccw',          'max(0, d)   CCW side'),
        (6.1, d_cw_v, RED,   'deriv_out_cw',           'max(0, −d)  CW side'),
        (4.5, d_abs,  GREEN, 'coact_node',              'CCW + CW  =  |d(t)|'),
    ]
    for yc, data, color, lbl, sublbl in rows_d:
        ax.fill_between(xs_p, yc, yc + data*0.55, alpha=0.22, color=color)
        ax.plot(xs_p, yc + data*0.55, color=color, lw=1.5)
        ax.plot([x0, x0+xw], [yc, yc], color=GREY, lw=0.7, ls=':')
        ax.text(x0-0.2, yc+0.15, lbl,    ha='right', fontsize=7.5, color=color, va='center')
        ax.text(x0-0.2, yc-0.28, sublbl, ha='right', fontsize=6.5, color=GREY,  va='center')

    # + and = between rows
    ax.text(9.6, 6.85, '+', fontsize=20, color=GREY,  ha='center', va='center')
    ax.text(9.6, 5.3,  '=', fontsize=20, color=GREEN, ha='center', va='center')

    stbx(ax, 5.0, 3.3,
         'Both CCW and CW are non-negative — SNS compatible.\n'
         r'Together they encode the full signed $d(t)$.' '\n'
         r'Their sum = $|d(t)|$ with no extra circuitry.' '\n'
         r'$K_d$ uses the directional side  $|$  $K_c$ uses coact_node.',
         GREEN, '#EAFAF1', fs=8.5)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(SCRIPT_DIR, 'reference_seg6b_bias_split.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f'Saved: {out}')
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# FIGURE 10 — Segment 7: Full TC4 assembly
# ══════════════════════════════════════════════════════════════════════
def fig_tc4_assembly():
    PURPLE   = '#7D3C98'
    PURPLEBG = '#F5EEF8'

    fig = plt.figure(figsize=(20, 13))
    fig.patch.set_facecolor(BG)
    gs_fig = fig.add_gridspec(2, 2, width_ratios=[1.7, 1.0], height_ratios=[1.0, 1.0],
                               wspace=0.04, hspace=0.04,
                               left=0.01, right=0.99, top=0.93, bottom=0.02)
    ax_L = fig.add_subplot(gs_fig[:, 0])
    ax_P = fig.add_subplot(gs_fig[0, 1])
    ax_S = fig.add_subplot(gs_fig[1, 1])

    for ax in [ax_L, ax_P, ax_S]:
        ax.set_facecolor(BG); ax.axis('off')
    ax_L.set_xlim(0, 14.5); ax_L.set_ylim(0, 20)
    ax_P.set_xlim(0, 10);   ax_P.set_ylim(0, 10)
    ax_S.set_xlim(0, 10);   ax_S.set_ylim(0, 10)

    fig.suptitle('Segment 7 — Full TC4 SNS Controller Assembly\n'
                 '46 neurons · 64 synapses · 4 bias parameters · Peterka 2002 TC4',
                 fontsize=13, fontweight='bold', y=0.98)

    def sarr(ax, x1, y1, x2, y2, lc=BLACK, lw=1.5):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=lc, lw=lw))

    def stage_box(ax, cx, cy, w, h, title, neurons, fc, ec, fs_t=8.5, fs_n=6.8):
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx-w/2, cy-h/2), w, h, boxstyle='round,pad=0.15',
            fc=fc, ec=ec, lw=2.0, zorder=3))
        ax.text(cx, cy+h/2-0.22, title, ha='center', va='top',
                fontsize=fs_t, fontweight='bold', color=ec, zorder=4)
        for i, n in enumerate(neurons):
            ax.text(cx, cy+h/2-0.62-i*0.46, n, ha='center', va='top',
                    fontsize=fs_n, color=GREY, zorder=4)

    def harr(ax, x1, x2, y, lbl='', lc=BLACK):
        sarr(ax, x1, y, x2, y, lc, 1.6)
        if lbl:
            ax.text((x1+x2)/2, y+0.25, lbl, ha='center', fontsize=7, color=lc)

    # Stage x-centers and y-lanes
    x1, x2, x3      = 1.1, 3.4, 5.8
    x4, x6           = 7.9, 9.2    # shared nodes (coact, ib)
    x5, x7           = 11.2, 13.2
    y_ccw, y_cw      = 16.0, 6.0
    y_mid             = 11.0
    BW_main, BW_sm   = 2.0, 1.65   # box widths

    # ── Stage 1 ──────────────────────────────────────────────────────
    for yc, s in [(y_ccw, 'ccw'), (y_cw, 'cw')]:
        stage_box(ax_L, x1, yc, BW_sm, 2.4, 'Stage 1\nInput (×2)',
                  [f'bs_{s}', f'ss_{s}', 'theta_ref'], '#EAF2FF', BLUE, fs_t=8, fs_n=7)
        ax_L.text(0.1, yc+0.45, r'$\theta_B$', fontsize=8, color=BLUE,  va='center', ha='left')
        ax_L.text(0.1, yc-0.45, r'$\theta_S$', fontsize=8, color=GREEN, va='center', ha='left')
        sarr(ax_L, 0.55, yc+0.45, x1-BW_sm/2, yc+0.45, BLUE,  1.2)
        sarr(ax_L, 0.55, yc-0.45, x1-BW_sm/2, yc-0.45, GREEN, 1.2)

    # ── Stage 2 ──────────────────────────────────────────────────────
    for yc, s in [(y_ccw, 'ccw'), (y_cw, 'cw')]:
        stage_box(ax_L, x2, yc, BW_main, 4.2, 'Stage 2\nTC4 Sensory (×2)',
                  ['sub_diff', 'wg  (k=0.30)', 'wp  (k=0.70)', 'error'],
                  '#EAFAF1', GREEN)

    # ── Stage 3 ──────────────────────────────────────────────────────
    for yc, s in [(y_ccw, 'ccw'), (y_cw, 'cw')]:
        stage_box(ax_L, x3, yc, BW_main, 4.2, 'Stage 3\nDerivative (×2)',
                  ['fast  Cm=0.1nF', 'slow  Cm=8nF', 'd_accel', 'd_decel'],
                  '#FEF9E7', ORANGE)

    # ── Stage 4 (shared) ─────────────────────────────────────────────
    stage_box(ax_L, x4, y_mid, BW_sm, 2.2, 'Stage 4\nCo-act.',
              ['coact_node'], '#FDEDEC', RED, fs_t=8, fs_n=7.5)

    # ── Stage 5 ──────────────────────────────────────────────────────
    for yc in [y_ccw, y_cw]:
        stage_box(ax_L, x5, yc, BW_main, 4.2, 'Stage 5\nBias Gain (×2)',
                  ['Kp: mod + prod', 'Kd: mod + prod',
                   'Kc: mod + prod', 'Kt: mod + prod'],
                  PURPLEBG, PURPLE)
        ax_L.text(x5, yc-4.2/2-0.38,
                  r'$b_p,b_d,b_c,b_t$ → mod neurons',
                  fontsize=6.5, color=PURPLE, ha='center', style='italic')

    # ── Stage 6 (shared) ─────────────────────────────────────────────
    stage_box(ax_L, x6, y_mid, BW_sm, 2.2, 'Stage 6\nType-Ib',
              ['ib_input'], '#F0F0F0', GREY, fs_t=8, fs_n=7.5)
    sarr(ax_L, x6, y_mid-2.2/2-0.7, x6, y_mid-2.2/2, GREY, 1.2)
    ax_L.text(x6, y_mid-2.2/2-0.9, 'tension\n(Hill)',
              fontsize=6.5, color=GREY, ha='center', va='top')

    # ── Stage 7 ──────────────────────────────────────────────────────
    for yc, s in [(y_ccw, 'ccw'), (y_cw, 'cw')]:
        stage_box(ax_L, x7, yc, 1.5, 2.2, 'Stage 7\nOutput',
                  [f'ta_{s}'], '#F4F6F7', BLACK, fs_t=8, fs_n=7.5)
        sarr(ax_L, x7+1.5/2, yc, x7+1.5/2+0.55, yc, BLACK, 1.5)
        ax_L.text(x7+1.5/2+0.65, yc,
                  r'$A_{CCW}$' if s == 'ccw' else r'$A_{CW}$',
                  fontsize=8.5, color=BLACK, va='center')

    # ── Pipeline arrows (horizontal) ─────────────────────────────────
    for yc, s in [(y_ccw, 'ccw'), (y_cw, 'cw')]:
        harr(ax_L, x1+BW_sm/2, x2-BW_main/2, yc)
        harr(ax_L, x2+BW_main/2, x3-BW_main/2, yc, f'error_{s}', GREEN)
        harr(ax_L, x3+BW_main/2, x5-BW_main/2, yc, f'd_{s}', ORANGE)
        harr(ax_L, x5+BW_main/2, x7-1.5/2, yc)

    # ── Stage 3 → Stage 4 (coact) ────────────────────────────────────
    sarr(ax_L, x3, y_ccw-4.2/2, x4, y_mid+2.2/2, RED, 1.3)
    ax_L.text((x3+x4)/2-0.3, 13.8, 'd_ccw', fontsize=6.5, color=RED, ha='center')
    sarr(ax_L, x3, y_cw+4.2/2,  x4, y_mid-2.2/2, RED, 1.3)
    ax_L.text((x3+x4)/2-0.3, 8.3,  'd_cw',  fontsize=6.5, color=RED, ha='center')

    # ── Stage 4 → Stage 5 (|d| → kc) ────────────────────────────────
    sarr(ax_L, x4+BW_sm/2, y_mid, x5-BW_main/2, y_ccw, RED, 1.2)
    ax_L.text((x4+BW_sm/2+x5-BW_main/2)/2+0.2, 13.5, '|d|', fontsize=7, color=RED)
    sarr(ax_L, x4+BW_sm/2, y_mid, x5-BW_main/2, y_cw,  RED, 1.2)
    ax_L.text((x4+BW_sm/2+x5-BW_main/2)/2+0.2, 8.6,  '|d|', fontsize=7, color=RED)

    # ── Stage 6 → Stage 5 (Ib → kt) ─────────────────────────────────
    sarr(ax_L, x6+BW_sm/2, y_mid, x5-BW_main/2, y_ccw, GREY, 1.0)
    ax_L.text((x6+BW_sm/2+x5-BW_main/2)/2+0.1, 13.1, 'Ib', fontsize=6.5, color=GREY)
    sarr(ax_L, x6+BW_sm/2, y_mid, x5-BW_main/2, y_cw,  GREY, 1.0)
    ax_L.text((x6+BW_sm/2+x5-BW_main/2)/2+0.1, 9.0,  'Ib', fontsize=6.5, color=GREY)

    # ── Net torque note ───────────────────────────────────────────────
    ax_L.text(7.0, 2.0, r'Net torque:  $T_a = L_m(F_{CCW} - F_{CW})$',
              fontsize=8.5, color=BLACK, ha='center',
              bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=BLACK, lw=1.2))

    # ── Lane labels ───────────────────────────────────────────────────
    for yc, lbl, clr in [(y_ccw, 'CCW side', BLUE), (y_mid, 'Shared', GREY),
                          (y_cw,  'CW side',  RED)]:
        ax_L.text(0.08, yc, lbl, fontsize=8, color=clr, ha='center', va='center',
                  fontweight='bold', rotation=90)

    # ══ Top-right — Parameter table ════════════════════════════════════
    ax_P.text(5.0, 9.8, 'Complete Parameter Table',
              fontsize=11, fontweight='bold', color=BLACK, ha='center', va='top')
    col_x = [0.2, 4.2, 7.5]
    for h, x in zip(['Parameter', 'Value', 'Segment'], col_x):
        ax_P.text(x, 9.3, h, ha='left', fontsize=8.5, fontweight='bold', color=BLACK, va='top')
    ax_P.plot([0.1, 9.9], [9.0, 9.0], color='#CCCCCC', lw=1)

    rows_p = [
        (r'$E_r$',               '−60 mV',                    'Seg 1',  BLUE),
        (r'$R = E_{hi}-E_r$',    '20 mV',                     'Seg 1',  BLUE),
        (r'$G_m$',               '1 µS',                      'Seg 1',  BLUE),
        (r'$g_{s,exc}$',         '0.115 µS',                  'Seg 4B', GREEN),
        (r'$\Delta E_{exc}$',    '+194 mV',                   'Seg 4B', GREEN),
        (r'$g_{s,inh}$',         '0.558 µS',                  'Seg 4C', ORANGE),
        (r'$\Delta E_{inh}$',    '−40 mV',                    'Seg 4C', ORANGE),
        (r'$G_{S,Wg}$',          '0.031 µS   (k=0.30)',        'Seg 4A', BLUE),
        (r'$G_{S,Wp}$',          '0.075 µS   (k=0.70)',        'Seg 4A', BLUE),
        (r'$C_{m,fast}$',        '0.1 nF   (τ=0.1 ms)',        'Seg 4D', ORANGE),
        (r'$C_{m,slow}$',        '8 nF   (τ=8 ms)',            'Seg 4D', ORANGE),
        (r'$b_p, b_d, b_c, b_t$','4.26, 5.01, 2.48, 5.42 nA', 'Seg 6',  PURPLE),
        (r'$W_p,\;W_g$',         '0.70,  0.30',               'Seg 5',  RED),
        (r'$\tau_d$',            '90 ms',                     'Seg 5',  RED),
        (r'$J$',                 r'63 kg·m²',                  'Plant',  GREY),
        (r'$mgh$',               r'686 N·m/rad',               'Plant',  GREY),
        (r'$\Delta t$',          '0.2 ms',                    'Sim.',   GREY),
    ]
    for i, (name, val, src, color) in enumerate(rows_p):
        y = 8.7 - i * 0.48
        ax_P.text(col_x[0], y, name, ha='left', fontsize=7.5, color=color, va='center')
        ax_P.text(col_x[1], y, val,  ha='left', fontsize=7,   color=BLACK, va='center')
        ax_P.text(col_x[2], y, src,  ha='left', fontsize=7,   color=GREY,  va='center')

    # ══ Bottom-right — Segment cross-reference ════════════════════════
    ax_S.text(5.0, 9.8, 'Segment Cross-Reference',
              fontsize=11, fontweight='bold', color=BLACK, ha='center', va='top')

    seg_items = [
        ('Seg 1–2', 'Neuron circuit, synapse + activation function', BLUE),
        ('Seg 3',   'Steady-state master equation  (Eq. 13)',        GREEN),
        ('Seg 4',   'FSA subnetwork design  (Trans/Add/Sub/Diff)',    ORANGE),
        ('Seg 5',   'Peterka reweighting + closed-loop Eq. 15',      RED),
        ('Seg 6',   'Split-derivative + bias-gated gain',            PURPLE),
        ('Seg 7',   'Full TC4 assembly  ← this figure',              BLACK),
    ]
    for i, (seg, desc, color) in enumerate(seg_items):
        y = 9.0 - i * 1.38
        ax_S.add_patch(mpatches.FancyBboxPatch(
            (0.2, y-0.6), 9.6, 1.1, boxstyle='round,pad=0.1',
            fc='white', ec=color, lw=1.5))
        ax_S.text(0.6, y, seg,  ha='left', fontsize=9,   color=color,
                  fontweight='bold', va='center')
        ax_S.text(2.2, y, desc, ha='left', fontsize=8,   color=BLACK, va='center')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(SCRIPT_DIR, 'reference_seg7_assembly.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f'Saved: {out}')
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# FIGURES 11-13 — Segment 7 node-level derivations
# Shared helpers for the three node-derivation figures
# ══════════════════════════════════════════════════════════════════════

def _nd(ax, x, y, lbl, fc, NE='#2C3E50', w=1.9, h=0.75, fs=8):
    """Draw a labeled neuron box."""
    ax.add_patch(mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.1',
        fc=fc, ec=NE, lw=1.8, zorder=3))
    ax.text(x, y, lbl, ha='center', va='center', fontsize=fs,
            fontweight='bold', zorder=4)


def _syn(ax, x1, y1, x2, y2, clr, lbl='', hw=0.95, hh=0.375):
    """Arrow between two rectangular nodes, stopping at their edges."""
    dx, dy = x2 - x1, y2 - y1
    d = max((dx ** 2 + dy ** 2) ** 0.5, 0.01)
    ndx, ndy = dx / d, dy / d
    r = min(hw / max(abs(ndx), 1e-9), hh / max(abs(ndy), 1e-9)) + 0.06
    sx, sy = x1 + ndx * r, y1 + ndy * r
    ex, ey = x2 - ndx * r, y2 - ndy * r
    ax.annotate('', xy=(ex, ey), xytext=(sx, sy),
                arrowprops=dict(arrowstyle='->', color=clr, lw=2.0))
    if lbl:
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        ax.text(mx - ndy * 0.35, my + ndx * 0.35, lbl,
                fontsize=7, color=clr, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.1', fc='white', ec=clr, lw=0.6))


def _stp(ax, y, txt, sc, sbg, cx=5.0, fs=8.5):
    """Centered step box."""
    ax.text(cx, y, txt, ha='center', va='top', fontsize=fs,
            bbox=dict(boxstyle='round,pad=0.4', fc=sbg, ec=sc, lw=1.4))


def _arrd(ax, y0, y1, cx=5.0):
    ax.annotate('', xy=(cx, y1), xytext=(cx, y0),
                arrowprops=dict(arrowstyle='->', color=GREY, lw=1.3))


def _ptitle(ax, title, color):
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.3, 11.35), 9.4, 0.6, boxstyle='round,pad=0.1',
        fc=color + '22', ec=color, lw=1.5, zorder=2, clip_on=False))
    ax.text(5, 11.65, title, ha='center', va='center',
            fontsize=9.5, fontweight='bold', color=color, zorder=3)


# ──────────────────────────────────────────────────────────────────────
# FIGURE 11 — Segment 7a: TC4 Sensory Integration
# ──────────────────────────────────────────────────────────────────────
def fig_node_sensory():
    C_INP  = '#E59866'   # orange  – input neurons
    C_SUB  = '#A9DFBF'   # green   – subtraction / addition output
    C_TRN  = '#AED6F1'   # blue    – transmission
    C_ERR  = '#F1948A'   # pink    – error neuron
    EXC    = '#1E8449'
    INH    = '#C0392B'

    fig, axes = plt.subplots(1, 4, figsize=(20, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        'Segment 7a — TC4 Sensory Integration: Eq. 13 instantiated at each node\n'
        'Stage 2  |  CCW side shown  |  CW is an identical mirror',
        fontsize=12, fontweight='bold', y=0.99)
    for ax in axes:
        ax.set_facecolor(BG); ax.axis('off')
        ax.set_xlim(0, 10);   ax.set_ylim(0, 12)

    # ── Panel 0 : sub_diff  (BS − SS) ─────────────────────────────────
    ax = axes[0]
    _ptitle(ax, 'sub_diff  =  BS − SS', GREEN)
    _nd(ax, 2.5, 10.6, 'BS',       C_INP)
    _nd(ax, 2.5,  9.2, 'SS',       C_INP)
    _nd(ax, 7.5,  9.9, 'sub_diff', C_SUB)
    _syn(ax, 2.5, 10.6, 7.5, 9.9, EXC, 'exc')
    _syn(ax, 2.5,  9.2, 7.5, 9.9, INH, 'inh')
    ax.text(5, 8.8, r'Goal: $U^*_{diff} \approx U_{BS} - U_{SS}$',
            ha='center', fontsize=8, color=GREY, style='italic')

    _stp(ax, 8.4,
         r'Eq. 13,  $I_{app}=0$,  2 synapses  ($\Delta E_e=+194$,  $\Delta E_{inh}=-40$):' '\n'
         r'$U^* = \dfrac{(g_e/R)\,U_{BS}(+194)+(g_{inh}/R)\,U_{SS}(-40)}'
         r'{1+(g_e/R)\,U_{BS}+(g_{inh}/R)\,U_{SS}}$',
         BLUE, '#EAF2FF', fs=7.5)
    _arrd(ax, 7.0, 6.7)
    _stp(ax, 6.65,
         r'Want $U^*=0$ when $U_{BS}=U_{SS}$  →  set numerator $=0$:' '\n'
         r'$g_e(+194)+g_{inh}(-40)=0$  →  $g_{inh}=g_e\times 194/40$',
         GREEN, '#EAFAF1', fs=8)
    _arrd(ax, 5.3, 5.0)
    _stp(ax, 4.95,
         r'$g_{inh}=0.115\times 4.85=0.558\,\mu S$',
         ORANGE, '#FEF9E7', fs=8.5)
    _arrd(ax, 4.25, 3.95)
    _stp(ax, 3.9,
         r'$U^*_{diff} = U_{BS}-U_{SS}$' '\n'
         r'$= \theta_{ankle}$ (proprioceptive ankle angle)  ✓',
         RED, '#FDEDEC', fs=8.5)

    # ── Panel 1 : wg  (0.30 × BS) ─────────────────────────────────────
    ax = axes[1]
    _ptitle(ax, r'wg  =  0.30 $\times$ BS', BLUE)
    _nd(ax, 2.5, 9.9, 'BS', C_INP)
    _nd(ax, 7.5, 9.9, 'wg', C_TRN)
    _syn(ax, 2.5, 9.9, 7.5, 9.9, EXC, r'$G_{S,Wg}$ = ?')
    ax.text(5, 9.1, r'Goal: $U^*_{wg} = 0.30\,U_{BS}$  (transmission,  $k=0.30$)',
            ha='center', fontsize=8, color=GREY, style='italic')

    _stp(ax, 8.65,
         r'Eq. 13,  $I_{app}=0$,  1 exc. synapse:' '\n'
         r'$U^* = \dfrac{(g_s/R)\,U_{BS}\,\Delta E_e}{1+(g_s/R)\,U_{BS}}$',
         BLUE, '#EAF2FF', fs=8)
    _arrd(ax, 7.5, 7.2)
    _stp(ax, 7.15,
         r'Want $U^*=k\,U_{BS}$ at design point $U_{BS}=R/2$.' '\n'
         r'Solve Eq. 13 for $g_s$:' '\n'
         r'$g_s = \dfrac{k\cdot R}{\Delta E_e - k\cdot R/2}$',
         GREEN, '#EAFAF1', fs=8)
    _arrd(ax, 5.65, 5.35)
    _stp(ax, 5.3,
         r'$g_s = \dfrac{0.30\times 20}{194-0.30\times10} = \dfrac{6}{191}$',
         ORANGE, '#FEF9E7', fs=8.5)
    _arrd(ax, 4.55, 4.25)
    _stp(ax, 4.2,
         r'$G_{S,Wg} = 0.031\,\mu S$  →  wg outputs $W_g\cdot\theta_{body}$  ✓',
         RED, '#FDEDEC', fs=8.5)

    # ── Panel 2 : wp  (0.70 × sub_diff) ──────────────────────────────
    ax = axes[2]
    _ptitle(ax, r'wp  =  0.70 $\times$ sub\_diff', BLUE)
    _nd(ax, 2.5, 9.9, 'sub_diff', C_SUB)
    _nd(ax, 7.5, 9.9, 'wp',       C_TRN)
    _syn(ax, 2.5, 9.9, 7.5, 9.9, EXC, r'$G_{S,Wp}$ = ?')
    ax.text(5, 9.1, r'Goal: $U^*_{wp} = 0.70\,(U_{BS}-U_{SS})$',
            ha='center', fontsize=8, color=GREY, style='italic')

    _stp(ax, 8.65,
         r'Same transmission form,  $k=0.70$:' '\n'
         r'$U^* = \dfrac{(g_s/R)\,U_{sub}\,\Delta E_e}{1+(g_s/R)\,U_{sub}}$',
         BLUE, '#EAF2FF', fs=8)
    _arrd(ax, 7.5, 7.2)
    _stp(ax, 7.15,
         r'$g_s = \dfrac{k\cdot R}{\Delta E_e - k\cdot R/2}$  with  $k=0.70$:' '\n'
         r'$g_s = \dfrac{0.70\times20}{194-0.70\times10} = \dfrac{14}{187}$',
         GREEN, '#EAFAF1', fs=8)
    _arrd(ax, 5.65, 5.35)
    _stp(ax, 5.3,
         r'$G_{S,Wp} = 0.075\,\mu S$',
         ORANGE, '#FEF9E7', fs=8.5)
    _arrd(ax, 4.55, 4.25)
    _stp(ax, 4.2,
         r'wp outputs $W_p\cdot(BS-SS) = 0.70\,\theta_{ankle}$  ✓',
         RED, '#FDEDEC', fs=8.5)

    # ── Panel 3 : error  (wg + wp) ────────────────────────────────────
    ax = axes[3]
    _ptitle(ax, 'error  =  wg + wp', ORANGE)
    _nd(ax, 2.5, 10.6, 'wg',    C_TRN)
    _nd(ax, 2.5,  9.2, 'wp',    C_TRN)
    _nd(ax, 7.5,  9.9, 'error', C_ERR)
    _syn(ax, 2.5, 10.6, 7.5, 9.9, EXC, 'exc')
    _syn(ax, 2.5,  9.2, 7.5, 9.9, EXC, 'exc')
    ax.text(5, 8.8, r'Goal: $U^*_{err} = U_{wg}+U_{wp}$  (unity-gain addition)',
            ha='center', fontsize=8, color=GREY, style='italic')

    _stp(ax, 8.4,
         r'Eq. 13,  $I_{app}=0$,  2 identical exc. synapses.' '\n'
         r'Let $S=U_{wg}+U_{wp}$.  Want $U^*=S$  ($k=1$):' '\n'
         r'$g_e = \dfrac{R}{\Delta E_e - R} = \dfrac{20}{194-20}$',
         BLUE, '#EAF2FF', fs=8)
    _arrd(ax, 6.8, 6.5)
    _stp(ax, 6.45,
         r'$g_e = 20/174 = 0.115\,\mu S$  (standard exc. conductance)',
         ORANGE, '#FEF9E7', fs=8.5)
    _arrd(ax, 5.75, 5.45)
    _stp(ax, 5.4,
         r'error $= W_g\cdot BS + W_p\cdot(BS-SS)$' '\n'
         r'$= -(W_g+W_p)\,\theta_{body} + W_p\,\theta_{surf}$' '\n'
         r'$= -\theta_b + 0.70\,\theta_s = e(t)$  ✓',
         RED, '#FDEDEC', fs=8.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(SCRIPT_DIR, 'reference_seg7a_sensory.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f'Saved: {out}')
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# FIGURE 12 — Segment 7b: Derivative + Co-activation
# ──────────────────────────────────────────────────────────────────────
def fig_node_derivative():
    C_ERR  = '#F1948A'   # pink   – error input
    C_FAST = '#85C1E9'   # steel  – fast integrator
    C_SLOW = '#A9CCE3'   # muted  – slow integrator
    C_DOUT = '#F0B27A'   # amber  – derivative output
    C_COAT = '#C39BD3'   # violet – co-activation
    EXC    = '#1E8449'
    INH    = '#C0392B'

    fig, axes = plt.subplots(1, 3, figsize=(18, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        'Segment 7b — Derivative Estimation & Co-activation: Eq. 13 at each node\n'
        'Stages 3–4  |  CCW side shown  |  τ = Cm/Gm  (Gm = 1 µS for all neurons)',
        fontsize=12, fontweight='bold', y=0.99)
    for ax in axes:
        ax.set_facecolor(BG); ax.axis('off')
        ax.set_xlim(0, 10);   ax.set_ylim(0, 12)

    # ── Panel 0 : fast & slow neurons  (design Cm, not gs) ────────────
    ax = axes[0]
    _ptitle(ax, 'fast / slow  (error integrators)', BLUE)
    _nd(ax, 1.8, 9.9, 'error', C_ERR)
    _nd(ax, 6.5, 10.7, r'fast  $\tau$=0.1ms',  C_FAST, w=2.2)
    _nd(ax, 6.5,  9.0, r'slow  $\tau$=8ms',   C_SLOW, w=2.2)
    _syn(ax, 1.8, 9.9, 6.5, 10.7, EXC, 'exc', hw=1.1, hh=0.375)
    _syn(ax, 1.8, 9.9, 6.5,  9.0, EXC, 'exc', hw=1.1, hh=0.375)
    ax.text(5, 8.15,
            r'Both get same synapse from error — SS is identical.' '\n'
            r'Design targets the time constant $\tau = C_m/G_m$.',
            ha='center', fontsize=7.5, color=GREY, style='italic')

    _stp(ax, 7.75,
         r'Eq. 13 (steady-state):  $U^*_{fast} = U^*_{slow} = U_{error}$' '\n'
         r'(identical at SS — both converge to the same value)',
         BLUE, '#EAF2FF', fs=8)
    _arrd(ax, 6.45, 6.15)
    _stp(ax, 6.1,
         r'Time domain:  $C_m\,\dot{U} = -G_m U + I_{syn}$' '\n'
         r'$\Rightarrow$  transfer function  $H(s) = \dfrac{1}{1+\tau s}$,  $\tau = C_m/G_m$',
         GREEN, '#EAFAF1', fs=8)
    _arrd(ax, 4.85, 4.55)
    _stp(ax, 4.5,
         r'Choose $C_m$ to set bandwidth:' '\n'
         r'fast:  $\tau_f = 0.1\,\mathrm{ms}$  →  $C_m = \tau_f G_m = 0.1\,\mathrm{nF}$' '\n'
         r'slow:  $\tau_s = 8\,\mathrm{ms}$ →  $C_m = 8\,\mathrm{nF}$',
         ORANGE, '#FEF9E7', fs=8)
    _arrd(ax, 3.1, 2.8)
    _stp(ax, 2.75,
         r'fast tracks $e(t)$ quickly;  slow lags  →  difference $\approx \dot{e}$' '\n'
         r'$U_{fast}-U_{slow} \approx (\tau_s-\tau_f)\dfrac{de}{dt} = 7.9\,\mathrm{ms}\cdot\dot{e}$  ✓',
         RED, '#FDEDEC', fs=8)

    # ── Panel 1 : deriv_out  (fast − slow) ────────────────────────────
    ax = axes[1]
    _ptitle(ax, r'deriv\_out  =  fast $-$ slow', GREEN)
    _nd(ax, 2.5, 10.6, r'fast',    C_FAST)
    _nd(ax, 2.5,  9.2, r'slow',    C_SLOW)
    _nd(ax, 7.5,  9.9, 'deriv_out', C_DOUT, w=2.1)
    _syn(ax, 2.5, 10.6, 7.5, 9.9, EXC, 'exc')
    _syn(ax, 2.5,  9.2, 7.5, 9.9, INH, 'inh')
    ax.text(5, 8.8, r'Goal: $U^*_{out} = U_{fast} - U_{slow}$',
            ha='center', fontsize=8, color=GREY, style='italic')

    _stp(ax, 8.4,
         r'Eq. 13,  subtraction form  ($\Delta E_e=+194$,  $\Delta E_{inh}=-40$):' '\n'
         r'$U^* = \dfrac{(g_e/R)\,U_{fast}(+194)+(g_{inh}/R)\,U_{slow}(-40)}'
         r'{1+(g_e/R)\,U_{fast}+(g_{inh}/R)\,U_{slow}}$',
         BLUE, '#EAF2FF', fs=7.5)
    _arrd(ax, 7.0, 6.7)
    _stp(ax, 6.65,
         r'Same cancellation condition as sub\_diff:' '\n'
         r'$g_{inh} = g_e\times194/40 = 0.558\,\mu S$',
         GREEN, '#EAFAF1', fs=8)
    _arrd(ax, 5.7, 5.4)
    _stp(ax, 5.35,
         r'At SS: $U^*_{out}=0$ (fast=slow at SS)' '\n'
         r'Transiently: $U_{out}\approx(\tau_s-\tau_f)\,\dot{e} = d(t)$',
         ORANGE, '#FEF9E7', fs=8)
    _arrd(ax, 4.3, 4.0)
    _stp(ax, 3.95,
         r'$g_e=0.115\,\mu S$,  $g_{inh}=0.558\,\mu S$' '\n'
         r'deriv\_out $\approx d(t) = \dot{e}(t)$  ✓',
         RED, '#FDEDEC', fs=8.5)

    # ── Panel 2 : coact_node  (CCW + CW = |d|) ────────────────────────
    ax = axes[2]
    _ptitle(ax, r'coact\_node  =  $|d(t)|$', ORANGE)
    _nd(ax, 2.5, 10.6, r'$d_{CCW}$', C_DOUT)
    _nd(ax, 2.5,  9.2, r'$d_{CW}$',  C_DOUT)
    _nd(ax, 7.5,  9.9, 'coact', C_COAT, w=1.6)
    _syn(ax, 2.5, 10.6, 7.5, 9.9, EXC, 'exc')
    _syn(ax, 2.5,  9.2, 7.5, 9.9, EXC, 'exc')
    ax.text(5, 8.8,
            r'CCW = max$(0,\,d)$;  CW = max$(0,\,-d)$' '\n'
            r'Goal: coact $= U_{CCW}+U_{CW} = |d(t)|$',
            ha='center', fontsize=7.5, color=GREY, style='italic')

    _stp(ax, 8.1,
         r'Eq. 13,  2 identical exc. synapses  ($S=U_{CCW}+U_{CW}$):' '\n'
         r'Want $U^*=S$  ($k=1$)  →  $g_e=R/(\Delta E_e-R)=20/174$',
         BLUE, '#EAF2FF', fs=8)
    _arrd(ax, 6.85, 6.55)
    _stp(ax, 6.5,
         r'$g_e = 0.115\,\mu S$  (standard addition)',
         ORANGE, '#FEF9E7', fs=8.5)
    _arrd(ax, 5.8, 5.5)
    _stp(ax, 5.45,
         r'Since CCW and CW carry opposite halves of $d(t)$,' '\n'
         r'only one is non-zero at any instant:' '\n'
         r'coact $= U_{CCW}+U_{CW} = |d(t)|$  (natural rectification)  ✓',
         RED, '#FDEDEC', fs=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(SCRIPT_DIR, 'reference_seg7b_derivative.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f'Saved: {out}')
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# FIGURE 13 — Segment 7c: Bias-Gated Gain Stages
# ──────────────────────────────────────────────────────────────────────
def fig_node_gain():
    PURPLE   = '#7D3C98'
    PURPLEBG = '#F5EEF8'
    C_SIG  = '#F1948A'    # pink   – signal input (e or d)
    C_MOD  = '#F9E79F'    # yellow – bias mod neuron
    C_PROD = '#FDEBD0'    # cream  – prod/output neuron
    C_MOT  = '#FADBD8'    # rose   – motor output sum
    EXC    = '#1E8449'

    fig, axes = plt.subplots(1, 3, figsize=(18, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        'Segment 7c — Bias-Gated Gain Stages: Eq. 13 at each node\n'
        r'Stage 5  |  Template applies to all four pathways: $K_p, K_d, K_c, K_t$',
        fontsize=12, fontweight='bold', y=0.99)
    for ax in axes:
        ax.set_facecolor(BG); ax.axis('off')
        ax.set_xlim(0, 10);   ax.set_ylim(0, 12)

    # ── Panel 0 : mod neuron  (bias current → voltage) ────────────────
    ax = axes[0]
    _ptitle(ax, r'mod neuron  ($b_\alpha \to U_{mod}$)', BLUE)

    # Circuit: just the mod neuron with Iapp arrow
    _nd(ax, 5.0, 10.3, r'$\mathrm{mod}_\alpha$', C_MOD, w=2.0)
    ax.annotate('', xy=(3.95, 10.3), xytext=(2.3, 10.3),
                arrowprops=dict(arrowstyle='->', color=BLUE, lw=2.0))
    ax.text(2.2, 10.3, r'$I_{app}=b_\alpha$', fontsize=8.5,
            ha='right', va='center', color=BLUE)
    ax.annotate('', xy=(8.0, 10.3), xytext=(6.05, 10.3),
                arrowprops=dict(arrowstyle='->', color=BLACK, lw=1.8))
    ax.text(8.2, 10.3, r'$U_{mod}$', fontsize=8.5, ha='left', va='center')
    ax.text(5, 9.35,
            r'No synapses — only applied bias current $b_\alpha$',
            ha='center', fontsize=8, color=GREY, style='italic')

    _stp(ax, 8.95,
         r'Eq. 13  with no synapses  ($\Sigma g_{s,i}=0$):' '\n'
         r'$U^* = \dfrac{I_{app}}{G_m + 0} = \dfrac{b_\alpha}{G_m}$',
         BLUE, '#EAF2FF', fs=8)
    _arrd(ax, 7.55, 7.25)
    _stp(ax, 7.2,
         r'With $G_m=1\,\mu S$:  $U^*_{mod} = b_\alpha\;[\mathrm{mV}]$' '\n'
         r'(bias current in nA directly sets voltage in mV)',
         GREEN, '#EAFAF1', fs=8)
    _arrd(ax, 6.1, 5.8)

    bias_tbl = (
        r'$b_p=4.26$ nA  →  $U_{mod}=4.26$ mV  (21% of R)' '\n'
        r'$b_d=5.01$ nA  →  $U_{mod}=5.01$ mV  (25% of R)' '\n'
        r'$b_c=2.48$ nA  →  $U_{mod}=2.48$ mV  (12% of R)' '\n'
        r'$b_t=5.42$ nA  →  $U_{mod}=5.42$ mV  (27% of R)'
    )
    _stp(ax, 5.75, bias_tbl, ORANGE, '#FEF9E7', fs=7.5)
    _arrd(ax, 3.85, 3.55)
    _stp(ax, 3.5,
         r'$U_{mod}$ sets a constant DC voltage offset' '\n'
         r'that raises the prod neuron operating point  ✓',
         RED, '#FDEDEC', fs=8)

    # ── Panel 1 : prod neuron  (signal + mod → gain-scaled output) ────
    ax = axes[1]
    _ptitle(ax, r'prod neuron  (signal + mod)', GREEN)
    _nd(ax, 2.5, 10.6, r'signal', C_SIG)
    _nd(ax, 2.5,  9.2, r'$\mathrm{mod}_\alpha$', C_MOD)
    _nd(ax, 7.5,  9.9, r'prod', C_PROD)
    _syn(ax, 2.5, 10.6, 7.5, 9.9, EXC, 'exc')
    _syn(ax, 2.5,  9.2, 7.5, 9.9, EXC, 'exc')
    ax.text(5, 8.8,
            r'Goal: $U^*_{prod} \propto U_{signal}$,  scale set by $b_\alpha$',
            ha='center', fontsize=8, color=GREY, style='italic')

    _stp(ax, 8.4,
         r'Eq. 13,  2 exc. synapses  ($S=U_{signal}+U_{mod}$):' '\n'
         r'$U^*_{prod} = \dfrac{(g_e/R)\,S\,\Delta E_e}{1+(g_e/R)\,S}$' '\n'
         r'Same addition design:  $g_e=0.115\,\mu S$',
         BLUE, '#EAF2FF', fs=8)
    _arrd(ax, 6.9, 6.6)
    _stp(ax, 6.55,
         r'$U^*_{prod} = K\,(U_{signal}+U_{mod})$  where $K<1$' '\n'
         r'$= K\,U_{signal}+K\,b_\alpha$',
         GREEN, '#EAFAF1', fs=8)
    _arrd(ax, 5.55, 5.25)
    _stp(ax, 5.2,
         r'$K\,b_\alpha$ is a constant offset  →  shifts operating point.' '\n'
         r'Larger $b_\alpha$:  prod neuron runs higher  →  greater signal response.',
         ORANGE, '#FEF9E7', fs=8)
    _arrd(ax, 4.1, 3.8)
    _stp(ax, 3.75,
         r'prod output $\approx K_\alpha\cdot U_{signal}$  where gain $\propto b_\alpha$' '\n'
         r'The 4 free parameters $b_p, b_d, b_c, b_t$ control all gains  ✓',
         RED, '#FDEDEC', fs=8)

    # ── Panel 2 : motor sum  (all 4 prod neurons → a(t)) ─────────────
    ax = axes[2]
    _ptitle(ax, r'Motor sum:  $a(t) = K_p e + K_d d + K_c|d| + K_t I_b$', PURPLE)

    src_labels = [r'$K_p$ prod', r'$K_d$ prod', r'$K_c$ prod', r'$K_t$ prod']
    src_ys     = [11.0, 10.1, 9.2, 8.3]
    src_c      = [C_PROD, C_PROD, C_PROD, C_PROD]
    for lbl, yy, cc in zip(src_labels, src_ys, src_c):
        _nd(ax, 2.8, yy, lbl, cc, w=1.85, h=0.65)
        _syn(ax, 2.8, yy, 7.2, 9.65, EXC, '', hw=0.925, hh=0.325)
    _nd(ax, 7.2, 9.65, r'$a(t)$', C_MOT, w=1.5, h=0.65)

    _stp(ax, 7.8,
         r'Eq. 13,  4 exc. synapses into $a(t)$  neuron:' '\n'
         r'$U^* = \dfrac{(g_e/R)(U_{Kp}+U_{Kd}+U_{Kc}+U_{Kt})\,\Delta E_e}'
         r'{1+(g_e/R)(U_{Kp}+U_{Kd}+U_{Kc}+U_{Kt})}$',
         BLUE, '#EAF2FF', fs=7.5)
    _arrd(ax, 6.35, 6.05)
    _stp(ax, 6.0,
         r'Same 4-input addition:  $g_e=0.115\,\mu S$ per synapse' '\n'
         r'$U^*_{a} \approx K_p\,e + K_d\,d + K_c\,|d| + K_t\,I_b$',
         GREEN, '#EAFAF1', fs=8)
    _arrd(ax, 5.0, 4.7)
    _stp(ax, 4.65,
         r'Bilateral split: $a_{CCW}$ driven by CCW signal lane;' '\n'
         r'$a_{CW}$ driven by CW lane.' '\n'
         r'Only one lane active at a time  →  directional torque.',
         ORANGE, '#FEF9E7', fs=8)
    _arrd(ax, 3.45, 3.15)
    _stp(ax, 3.1,
         r'Motor drive fully determined by 4 bias parameters:' '\n'
         r'$b_p=4.26$,  $b_d=5.01$,  $b_c=2.48$,  $b_t=5.42$ nA  ✓',
         RED, '#FDEDEC', fs=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(SCRIPT_DIR, 'reference_seg7c_gain.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f'Saved: {out}')
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# PROOF FIGURES — Mathematical proofs that the TC4 network implements
# a(t) = Kp·e + Kd·d + Kc·|d| + Kt·Ib
# ══════════════════════════════════════════════════════════════════════

def _proof_fig(title):
    """Create a two-panel proof figure and return (fig, ax_net, ax_proof)."""
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(BG)
    gs_f = fig.add_gridspec(1, 2, width_ratios=[0.85, 1.15],
                             wspace=0.05, left=0.02, right=0.98,
                             top=0.92, bottom=0.03)
    ax_n = fig.add_subplot(gs_f[0])
    ax_p = fig.add_subplot(gs_f[1])
    for ax in [ax_n, ax_p]:
        ax.set_facecolor(BG); ax.axis('off')
    ax_n.set_xlim(0, 10);  ax_n.set_ylim(0, 10)
    ax_p.set_xlim(0, 12);  ax_p.set_ylim(0, 12)   # extra height to prevent step/QED overlap
    fig.suptitle(title, fontsize=12, fontweight='bold', y=0.97)
    return fig, ax_n, ax_p


def _thm_box(ax, y_top, title_line, body_lines):
    """Blue theorem header box at the top of the proof panel."""
    n = len(body_lines)
    h = 0.55 + n * 0.48
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.2, y_top - h), 11.6, h, boxstyle='round,pad=0.25',
        fc='#EAF2FF', ec=BLUE, lw=2.0))
    ax.text(6.0, y_top - 0.22, title_line, ha='center', va='top',
            fontsize=10, fontweight='bold', color=BLUE)
    for i, line in enumerate(body_lines):
        ax.text(6.0, y_top - 0.62 - i * 0.46, line,
                ha='center', va='top', fontsize=8.5, color=BLACK)
    return y_top - h - 0.55  # 0.25 clears round,pad + 0.30 breathing room


def _qed_box(ax, text, y_bot=0.1):
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.4, y_bot), 11.2, 1.05, boxstyle='round,pad=0.2',
        fc='#FDEDEC', ec=RED, lw=2.0))
    ax.text(6.0, y_bot + 0.52, text, ha='center', va='center',
            fontsize=9, color=RED, fontweight='bold')


def _pstp(ax, y, txt, sc, sbg, fs=8):
    """Step box centred in proof panel (cx=6.0)."""
    _stp(ax, y, txt, sc, sbg, cx=6.0, fs=fs)


def _parr(ax, y0, y1):
    _arrd(ax, y0, y1, cx=6.0)


# ──────────────────────────────────────────────────────────────────────
# PROOF FIGURE 1 — Error signal e(t)
# ──────────────────────────────────────────────────────────────────────
def fig_proof_error():
    C_INP = '#E59866'; C_MID = '#A9DFBF'; C_TRN = '#AED6F1'; C_ERR = '#F1948A'
    EXC = '#1E8449'; INH = '#C0392B'

    fig, ax_n, ax_p = _proof_fig(
        'Proof 1 of 4 — TC4 Sensory Integration computes the correct error signal')

    # ── Network fragment ─────────────────────────────────────────────
    ax_n.add_patch(mpatches.FancyBboxPatch(
        (0.3, 0.4), 9.4, 9.3, boxstyle='round,pad=0.2',
        fc='#F0FFF4', ec=GREEN, lw=1.5, ls='--', zorder=0))
    ax_n.text(5, 9.5, 'Stage 2 — TC4 Sensory Integration  (CCW side)',
              ha='center', fontsize=8.5, fontweight='bold', color=GREEN)

    _nd(ax_n, 1.2, 7.5, r'$\theta_b$',    C_INP,     w=1.3, h=0.7)
    _nd(ax_n, 1.2, 4.5, r'$\theta_s$',    C_INP,     w=1.3, h=0.7)
    _nd(ax_n, 1.2, 2.0, r'$\theta_{ref}=0$', '#E8E8E8', w=1.5, h=0.7)
    _nd(ax_n, 4.0, 7.5, 'sub_diff',       C_MID,     w=1.85)
    _nd(ax_n, 4.0, 2.5, 'wg',             C_TRN,     w=1.5)
    _nd(ax_n, 7.2, 6.0, 'wp',             C_TRN,     w=1.5)
    _nd(ax_n, 9.2, 3.8, r'$e(t)$',        C_ERR,     w=1.3, h=0.7)

    _syn(ax_n, 1.2, 7.5, 4.0, 7.5, EXC,  hw=0.65, hh=0.35)
    _syn(ax_n, 1.2, 4.5, 4.0, 7.5, INH,  hw=0.65, hh=0.35)
    _syn(ax_n, 1.2, 7.5, 4.0, 2.5, EXC,  hw=0.65, hh=0.35)
    _syn(ax_n, 4.0, 7.5, 7.2, 6.0, EXC,  hw=0.925, hh=0.375)
    _syn(ax_n, 4.0, 2.5, 9.2, 3.8, EXC,  hw=0.75,  hh=0.35)
    _syn(ax_n, 7.2, 6.0, 9.2, 3.8, EXC,  hw=0.75,  hh=0.35)
    _syn(ax_n, 1.2, 2.0, 9.2, 3.8, EXC,  hw=0.75,  hh=0.35)

    lkw = dict(fontsize=6.5, ha='center')
    ax_n.text(2.6,  8.05, r'$g_e$=0.115',       color=EXC, **lkw)
    ax_n.text(1.95, 5.7,  r'$g_{inh}$=0.558',   color=INH, **lkw)
    ax_n.text(2.2,  3.5,  r'$G_{Wg}$=0.031',    color=EXC, **lkw)
    ax_n.text(5.3,  7.25, r'$G_{Wp}$=0.075',    color=EXC, **lkw)
    ax_n.text(6.4,  2.7,  r'$g_e$=0.115',       color=EXC, **lkw)
    ax_n.text(8.85, 5.15, r'$g_e$=0.115',       color=EXC, **lkw)
    ax_n.text(5.5,  2.6,  r'$g_e$=0.115',       color='#AAAAAA', fontsize=6.5, ha='center')
    ax_n.text(1.2,  1.3,  r'(always 0 in TC4)',  color=GREY, fontsize=6, ha='center',
              style='italic')

    ax_n.text(5, 0.65,
              r'$W_g=0.30$,  $W_p=0.70$,  $R=20$ mV,  $\Delta E_e=+194$ mV,  '
              r'$\Delta E_{inh}=-40$ mV',
              ha='center', fontsize=7.5, color=GREY)

    # ── Proof ────────────────────────────────────────────────────────
    y = _thm_box(ax_p, 11.8, 'THEOREM 1', [
        r'The TC4 sensory integration stage computes  [Peterka 2002, TC4]:',
        r'$U_{err} = \theta_{ref} + (W_g+W_p)\,\theta_b - W_p\,\theta_s$  '
        r'($\theta_{ref}=0$)  $=\;\theta_b - 0.70\,\theta_s$',
    ])
    ax_p.text(0.4, y, 'Proof  (FSA design — Szczecinski & Quinn 2017, Eq. 13):',
              fontsize=9, fontweight='bold', color=BLACK, va='top')
    y -= 0.25

    _pstp(ax_p, y,
          r'(1)  sub\_diff  [Eq. 13, $g_e$=0.115, $g_{inh}$=0.558]' '\n'
          r'Cancellation: set numerator$=0$ when $U_{BS}=U_{SS}$  '
          r'$\Rightarrow$  $g_{inh}=g_e\cdot\Delta E_e/|\Delta E_{inh}|=0.558\,\mu S$' '\n'
          r'$\Rightarrow$  $U_{diff}=U_{\theta_b}-U_{\theta_s}$',
          BLUE, '#EAF2FF')
    y -= 1.5; _parr(ax_p, y, y - 0.3); y -= 0.45

    _pstp(ax_p, y,
          r'(2)  wg  [Eq. 13, transmission $k=0.30$]' '\n'
          r'$g_s=k\cdot R/(\Delta E_e-k\cdot R/2)=6/191=0.031\,\mu S$  '
          r'$\Rightarrow$  $U_{wg}=0.30\,U_{\theta_b}$',
          BLUE, '#EAF2FF')
    y -= 1.1; _parr(ax_p, y, y - 0.3); y -= 0.45

    _pstp(ax_p, y,
          r'(3)  wp  [Eq. 13, transmission $k=0.70$]' '\n'
          r'$g_s=14/187=0.075\,\mu S$  '
          r'$\Rightarrow$  $U_{wp}=0.70\,U_{diff}=0.70\,(\theta_b-\theta_s)$',
          GREEN, '#EAFAF1')
    y -= 1.1; _parr(ax_p, y, y - 0.3); y -= 0.45

    _pstp(ax_p, y,
          r'(4)  error  [Eq. 13, 3 exc.: wg + wp + $\theta_{ref}$,  $k=1$]' '\n'
          r'$g_e=R/(\Delta E_e-R)=0.115\,\mu S$;  $\theta_{ref}=0$ always  '
          r'$\Rightarrow$  $U_{err}=U_{wg}+U_{wp}+0$' '\n'
          r'Substitute (1)(2)(3):  $U_{err}=0.30\,\theta_b+0.70\,(\theta_b-\theta_s)'
          r'=\theta_b-0.70\,\theta_s$',
          ORANGE, '#FEF9E7')
    y -= 1.4

    _qed_box(ax_p,
             r'$\therefore$  $U_{err}=(W_g+W_p)\,\theta_b-W_p\,\theta_s$'
             r'$=\theta_b-0.70\,\theta_s=e(t)$   □',
             y_bot=y - 0.45)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(SCRIPT_DIR, 'reference_proof1_error.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f'Saved: {out}')
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# PROOF FIGURE 2 — Derivative estimate d(t)
# ──────────────────────────────────────────────────────────────────────
def fig_proof_derivative():
    C_ERR  = '#F1948A'; C_FAST = '#85C1E9'; C_SLOW = '#A9CCE3'; C_DOUT = '#F0B27A'
    EXC = '#1E8449'; INH = '#C0392B'

    fig, ax_n, ax_p = _proof_fig(
        r'Proof 2 of 4 — Derivative stage computes  $d(t)\approx(\tau_s-\tau_f)\,\dot{e}(t)$')

    # ── Network fragment ─────────────────────────────────────────────
    ax_n.add_patch(mpatches.FancyBboxPatch(
        (0.3, 0.4), 9.4, 9.3, boxstyle='round,pad=0.2',
        fc='#FFF8F0', ec=ORANGE, lw=1.5, ls='--', zorder=0))
    ax_n.text(5, 9.5, 'Stage 3 — Differential Calculations  (CCW side)',
              ha='center', fontsize=8.5, fontweight='bold', color=ORANGE)

    _nd(ax_n, 1.5, 5.5, r'$e(t)$',   C_ERR,  w=1.4, h=0.7)
    _nd(ax_n, 4.8, 7.8, r'fast  $\tau_f$=0.1ms',  C_FAST, w=2.2, h=0.8)
    _nd(ax_n, 4.8, 3.2, r'slow  $\tau_s$=8ms',   C_SLOW, w=2.2, h=0.8)
    _nd(ax_n, 8.2, 5.5, 'deriv_out', C_DOUT, w=1.9, h=0.7)

    _syn(ax_n, 1.5, 5.5, 4.8, 7.8, EXC, hw=0.7, hh=0.35)
    _syn(ax_n, 1.5, 5.5, 4.8, 3.2, EXC, hw=0.7, hh=0.35)
    _syn(ax_n, 4.8, 7.8, 8.2, 5.5, EXC, hw=1.1, hh=0.4)
    _syn(ax_n, 4.8, 3.2, 8.2, 5.5, INH, hw=1.1, hh=0.4)

    ax_n.text(3.0, 7.0, r'$g_e$=0.115', fontsize=6.5, ha='center', color=EXC)
    ax_n.text(3.0, 4.0, r'$g_e$=0.115', fontsize=6.5, ha='center', color=EXC)
    ax_n.text(6.6, 7.3, r'exc',         fontsize=6.5, ha='center', color=EXC)
    ax_n.text(6.6, 3.7, r'inh',         fontsize=6.5, ha='center', color=INH)

    ax_n.text(5, 1.7,
              r'$G_m=1\,\mu S$  for all neurons   $\Rightarrow$   $\tau=C_m/G_m$',
              ha='center', fontsize=7.5, color=GREY)
    ax_n.text(5, 1.1,
              r'fast: $C_m=0.1\,\mathrm{nF}$, $\tau_f=0.1\,\mathrm{ms}$   '
              r'slow: $C_m=8\,\mathrm{nF}$, $\tau_s=8\,\mathrm{ms}$',
              ha='center', fontsize=7.5, color=GREY)

    # ── Proof ────────────────────────────────────────────────────────
    y = _thm_box(ax_p, 11.8, 'THEOREM 2', [
        r'The derivative stage computes:',
        r'$d(t)\approx(\tau_s-\tau_f)\,\dot{e}(t)=7.9\,\mathrm{ms}\cdot\dot{e}(t)$',
    ])
    ax_p.text(0.4, y, 'Proof  (Eq. 13 + Laplace analysis):',
              fontsize=9, fontweight='bold', color=BLACK, va='top')
    y -= 0.25

    _pstp(ax_p, y,
          r'(1)  fast and slow  [Eq. 13, identical exc. synapse from $e(t)$]' '\n'
          r'$U^*_{fast}=U^*_{slow}=U_{err}$ at steady state  — '
          r'same SS; difference arises only transiently',
          BLUE, '#EAF2FF')
    y -= 1.1; _parr(ax_p, y, y - 0.3); y -= 0.45

    _pstp(ax_p, y,
          r'(2)  Time domain:  $C_m\dot{U}=-G_m U+I_{syn}$' '\n'
          r'Laplace:  $H(s)=\dfrac{1}{1+\tau s}$,  $\tau=C_m/G_m$' '\n'
          r'Choose $C_m$:  fast $\tau_f=0.1\,\mathrm{ms}$ ($C_m=0.1\,\mathrm{nF}$);  '
          r'slow $\tau_s=8\,\mathrm{ms}$ ($C_m=8\,\mathrm{nF}$)',
          BLUE, '#EAF2FF')
    y -= 1.5; _parr(ax_p, y, y - 0.3); y -= 0.45

    _pstp(ax_p, y,
          r'(3)  d\_accel / d\_decel  [Eq. 13, subtraction, same as sub\_diff]' '\n'
          r'$g_{inh}=0.558\,\mu S$  $\Rightarrow$  $U_{d\_accel}=U_{fast}-U_{slow}$',
          GREEN, '#EAFAF1')
    y -= 1.1; _parr(ax_p, y, y - 0.3); y -= 0.45

    _pstp(ax_p, y,
          r'(4)  Transfer function of the combined stage:' '\n'
          r'$D(s)=E(s)\cdot\dfrac{(\tau_s-\tau_f)\,s}{(1+\tau_f s)(1+\tau_s s)}$'
          r'  $[\tau_s-\tau_f=7.9\,\mathrm{ms}]$' '\n'
          r'For $\omega\ll100\,\mathrm{rad/s}$:  $D(s)\approx 7.9\,\mathrm{ms}\cdot s\cdot E(s)$',
          ORANGE, '#FEF9E7')
    y -= 1.4

    _qed_box(ax_p,
             r'$\therefore$  $d(t)\approx(\tau_s-\tau_f)\,\dot{e}(t)=7.9\,\mathrm{ms}'
             r'\cdot\dot{e}(t)$  (valid below 100 rad/s)   □',
             y_bot=y - 0.45)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(SCRIPT_DIR, 'reference_proof2_derivative.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f'Saved: {out}')
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# PROOF FIGURE 3 — Bias-gated gain stages
# ──────────────────────────────────────────────────────────────────────
def fig_proof_gain():
    PURPLE   = '#7D3C98'
    C_SIG  = '#F1948A'; C_MOD = '#F9E79F'; C_PROD = '#D5DBDB'; C_COAT = '#C39BD3'
    EXC = '#1E8449'

    fig, ax_n, ax_p = _proof_fig(
        r'Proof 3 of 4 — Bias-gated gain stages implement $K_\alpha$-scaled outputs')

    # ── Network fragment: template for one gain pathway ────────────
    ax_n.add_patch(mpatches.FancyBboxPatch(
        (0.3, 0.4), 9.4, 9.3, boxstyle='round,pad=0.2',
        fc='#F5EEF8', ec=PURPLE, lw=1.5, ls='--', zorder=0))
    ax_n.text(5, 9.5, 'Stage 5 — Bias-gated Gain  (template, one pathway)',
              ha='center', fontsize=8.5, fontweight='bold', color=PURPLE)

    # signal input + bias
    _nd(ax_n, 1.5, 7.5, 'signal\n(e or d)',  C_SIG,  w=1.7, h=1.0)
    _nd(ax_n, 1.5, 3.5, r'bias $b_\alpha$', '#FFFFFF', w=1.7, h=0.8,
        NE=PURPLE)

    # mod neuron
    _nd(ax_n, 5.0, 3.5, r'$\mathrm{mod}_\alpha$', C_MOD, w=1.8, h=0.8)
    ax_n.annotate('', xy=(4.05, 3.5), xytext=(2.35, 3.5),
                  arrowprops=dict(arrowstyle='->', color=PURPLE, lw=2.0))
    ax_n.text(3.2, 3.85, r'$I_{app}=b_\alpha$', fontsize=7, color=PURPLE, ha='center')

    # prod neuron
    _nd(ax_n, 8.0, 5.5, r'$\mathrm{prod}_\alpha$', C_PROD, w=1.8, h=0.8)
    _syn(ax_n, 1.5, 7.5, 8.0, 5.5, EXC, hw=0.85, hh=0.5)
    _syn(ax_n, 5.0, 3.5, 8.0, 5.5, EXC, hw=0.9,  hh=0.4)

    ax_n.text(4.5, 7.1,  r'exc $g_e$=0.115', fontsize=6.5, ha='center', color=EXC)
    ax_n.text(6.7, 3.8,  r'exc $g_e$=0.115', fontsize=6.5, ha='center', color=EXC)
    ax_n.annotate('', xy=(9.4, 5.5), xytext=(8.9, 5.5),
                  arrowprops=dict(arrowstyle='->', color=BLACK, lw=1.5))
    ax_n.text(9.5, 5.5, r'$K_\alpha\cdot$signal', fontsize=7.5, va='center')

    ax_n.text(5, 1.8,
              r'4 pathways:  $K_p$ (e),  $K_d$ (d),  $K_c$ (|d|),  $K_t$ ($I_b$)',
              ha='center', fontsize=7.5, color=GREY)
    ax_n.text(5, 1.2,
              r'Bias values:  $b_p$=4.26,  $b_d$=5.01,  $b_c$=2.48,  $b_t$=5.42 nA',
              ha='center', fontsize=7.5, color=GREY)

    # ── Proof ────────────────────────────────────────────────────────
    y = _thm_box(ax_p, 11.8, 'THEOREM 3', [
        r'Each bias-gated gain stage implements an output proportional to its signal input,',
        r'with effective gain $K_\alpha\propto b_\alpha$  (the 4 free control parameters)',
    ])
    ax_p.text(0.4, y, 'Proof  (Eq. 13 applied to mod and prod neurons):',
              fontsize=9, fontweight='bold', color=BLACK, va='top')
    y -= 0.25

    _pstp(ax_p, y,
          r'(1)  mod neuron  [Eq. 13, no synapses, $I_{app}=b_\alpha$]' '\n'
          r'$\Sigma g_{s,i}=0$  $\Rightarrow$  '
          r'$U^*_{mod}=I_{app}/G_m=b_\alpha/1\,\mu S=b_\alpha\,[\mathrm{mV}]$' '\n'
          r'mod neuron converts bias current directly to a DC voltage offset',
          BLUE, '#EAF2FF')
    y -= 1.5; _parr(ax_p, y, y - 0.3); y -= 0.45

    _pstp(ax_p, y,
          r'(2)  prod neuron  [Eq. 13, 2 exc. synapses: signal + mod]' '\n'
          r'Same addition design ($k=1$):  $g_e=0.115\,\mu S$' '\n'
          r'$U^*_{prod}=K\,(U_{signal}+U_{mod})$   where $K<1$',
          BLUE, '#EAF2FF')
    y -= 1.5; _parr(ax_p, y, y - 0.3); y -= 0.45

    _pstp(ax_p, y,
          r'(3)  Substitute (1):  $U_{mod}=b_\alpha$ (constant)' '\n'
          r'$U^*_{prod}=K\,U_{signal}+K\,b_\alpha$' '\n'
          r'$K\,b_\alpha$ is a fixed offset that raises the operating point;'
          r'  effective gain on signal $\propto b_\alpha$',
          GREEN, '#EAFAF1')
    y -= 1.5; _parr(ax_p, y, y - 0.3); y -= 0.45

    _pstp(ax_p, y,
          r'(4)  Four independent pathways — each bias sets one gain:' '\n'
          r'$K_p\propto b_p=4.26$ nA,  $K_d\propto b_d=5.01$ nA,' '\n'
          r'$K_c\propto b_c=2.48$ nA,  $K_t\propto b_t=5.42$ nA',
          ORANGE, '#FEF9E7')
    y -= 1.4

    _qed_box(ax_p,
             r'$\therefore$  prod$_\alpha\approx K_\alpha\cdot\mathrm{signal}$;'
             r'  gain is independently tunable via $b_p, b_d, b_c, b_t$   □',
             y_bot=y - 0.45)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(SCRIPT_DIR, 'reference_proof3_gain.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f'Saved: {out}')
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# PROOF FIGURE 4 — Final control equation  a(t)
# ──────────────────────────────────────────────────────────────────────
def fig_proof_final():
    PURPLE = '#7D3C98'
    C_INP  = '#E59866'; C_ERR = '#F1948A'; C_DOUT = '#F0B27A'
    C_COAT = '#C39BD3'; C_PROD = '#D5DBDB'; C_MOT = '#E74C3C'
    EXC = '#1E8449'
    TNBG   = '#F5EEF8'

    # Custom taller figure so all 5 proof steps fit without crowding
    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor(BG)
    gs_f = fig.add_gridspec(1, 2, width_ratios=[0.85, 1.15],
                             wspace=0.05, left=0.02, right=0.98,
                             top=0.93, bottom=0.03)
    ax_n = fig.add_subplot(gs_f[0]); ax_p = fig.add_subplot(gs_f[1])
    for ax in [ax_n, ax_p]: ax.set_facecolor(BG); ax.axis('off')
    ax_n.set_xlim(0, 10); ax_n.set_ylim(0, 10)
    ax_p.set_xlim(0, 12); ax_p.set_ylim(0, 12)   # taller proof panel
    fig.suptitle(
        r'Proof 4 of 4 — Final Theorem: TC4 network implements $a(t)=K_p e+K_d d+K_c|d|+K_t I_b$',
        fontsize=12, fontweight='bold', y=0.98)

    # ── Compact flow diagram ─────────────────────────────────────────
    ax_n.add_patch(mpatches.FancyBboxPatch(
        (0.2, 0.4), 9.6, 9.3, boxstyle='round,pad=0.2',
        fc='#FAFAFA', ec='#999999', lw=1.2, ls='--', zorder=0))
    ax_n.text(5, 9.5, 'TC4 Network — signal flow summary',
              ha='center', fontsize=8.5, fontweight='bold', color=BLACK)

    # Inputs
    _nd(ax_n, 1.0, 8.2, r'$\theta_b$',  C_INP, w=1.2, h=0.6)
    _nd(ax_n, 1.0, 7.0, r'$\theta_s$',  C_INP, w=1.2, h=0.6)
    _nd(ax_n, 1.0, 3.0, r'$I_b$',       '#CCCCCC', w=1.2, h=0.6)

    # Stage boxes
    _nd(ax_n, 4.0, 7.6, 'Stage 2\n(Thm 1)',  '#EAFAF1', w=2.0, h=1.0, fs=7)
    _nd(ax_n, 4.0, 5.5, 'Stage 3\n(Thm 2)',  '#EAF2FF', w=2.0, h=1.0, fs=7)
    _nd(ax_n, 4.0, 3.0, 'Stage 5\n(Thm 3)',  TNBG,      w=2.0, h=0.8, fs=7)

    # Intermediate signals
    _nd(ax_n, 7.0, 8.0, r'$e(t)$',   C_ERR,  w=1.3, h=0.6)
    _nd(ax_n, 7.0, 6.2, r'$d(t)$',   C_DOUT, w=1.3, h=0.6)
    _nd(ax_n, 7.0, 4.8, r'$|d(t)|$', C_COAT, w=1.4, h=0.6)

    # Motor output
    _nd(ax_n, 9.0, 5.5, r'$a(t)$',   C_MOT,  w=1.3, h=0.6, NE='#7B241C')

    # Arrows
    for x1, y1, x2, y2 in [(1.0,8.2,4.0,7.6),(1.0,7.0,4.0,7.6),
                            (4.0,7.6,7.0,8.0),(4.0,7.6,5.0,5.5)]:
        _syn(ax_n, x1, y1, x2, y2, EXC, hw=0.6, hh=0.35)
    for x1, y1, x2, y2 in [(5.0,5.5,7.0,6.2),(5.0,5.5,7.0,4.8)]:
        _syn(ax_n, x1, y1, x2, y2, EXC, hw=0.75, hh=0.35)
    for x1, y1, x2, y2 in [(7.0,8.0,9.0,5.5),(7.0,6.2,9.0,5.5),
                            (7.0,4.8,9.0,5.5),(1.0,3.0,4.0,3.0)]:
        _syn(ax_n, x1, y1, x2, y2, EXC, hw=0.65, hh=0.35)
    _syn(ax_n, 4.0, 3.0, 9.0, 5.5, EXC, hw=1.0, hh=0.4)

    ax_n.text(5.7, 5.0, r'bilateral  →  $|d|$', fontsize=6.5, color=GREY,
              ha='center', style='italic')

    ax_n.text(5, 1.6,
              r'$K_p\propto b_p$=4.26,  $K_d\propto b_d$=5.01,  '
              r'$K_c\propto b_c$=2.48,  $K_t\propto b_t$=5.42 nA',
              ha='center', fontsize=7, color=GREY)

    # ── Proof (ylim 0-12 gives room for all 5 steps) ─────────────────
    y = _thm_box(ax_p, 11.8, 'MAIN THEOREM (TC4 Control Law)', [
        r'The TC4 SNS network computes:',
        r'$a(t)=K_p\,e(t)+K_d\,d(t)+K_c\,|d(t)|+K_t\,I_b(t)$',
        r'[McNeal & Hunt 2026, Eq. 1;  Peterka 2002 TC4 sensory weights]',
    ])
    ax_p.text(0.4, y,
              'Proof  (by composition of Theorems 1–3 and FSA Lemmas):',
              fontsize=9, fontweight='bold', color=BLACK, va='top')
    y -= 0.25

    _pstp(ax_p, y,
          r'(1)  [By Theorem 1]  Stage 2 computes:' '\n'
          r'$e(t)=(W_g+W_p)\,\theta_b-W_p\,\theta_s=\theta_b-0.70\,\theta_s$  '
          r'[Peterka 2002 TC4]',
          BLUE, '#EAF2FF')
    y -= 1.1; _parr(ax_p, y, y - 0.3); y -= 0.45

    _pstp(ax_p, y,
          r'(2)  [By Theorem 2]  Stage 3 computes:' '\n'
          r'$d(t)\approx7.9\,\mathrm{ms}\cdot\dot{e}(t)$  '
          r'(valid for $\omega\ll100\,\mathrm{rad/s}$)',
          BLUE, '#EAF2FF')
    y -= 1.1; _parr(ax_p, y, y - 0.3); y -= 0.45

    _pstp(ax_p, y,
          r'(3)  [By bilateral construction]  Stage 4 computes:' '\n'
          r'$d_{CCW}=\max(0,d)$,  $d_{CW}=\max(0,-d)$' '\n'
          r'coact $= d_{CCW}+d_{CW}=|d(t)|$  [addition,  $g_e$=0.115$\,\mu S$]',
          GREEN, '#EAFAF1')
    y -= 1.5; _parr(ax_p, y, y - 0.3); y -= 0.45

    _pstp(ax_p, y,
          r'(4)  [By Theorem 3]  Stage 5 gain stages produce:' '\n'
          r'$U_{Kp}\approx K_p e$,  $U_{Kd}\approx K_d d$,  '
          r'$U_{Kc}\approx K_c|d|$,  $U_{Kt}\approx K_t I_b$',
          GREEN, '#EAFAF1')
    y -= 1.1; _parr(ax_p, y, y - 0.3); y -= 0.45

    _pstp(ax_p, y,
          r'(5)  Motor sum  [Eq. 13,  4 exc. synapses,  $g_e$=0.115$\,\mu S$ each]' '\n'
          r'Addition subnetwork:  $U_a=U_{Kp}+U_{Kd}+U_{Kc}+U_{Kt}$',
          ORANGE, '#FEF9E7')
    y -= 1.1

    # QED box — follows last step dynamically
    qed_bot = y - 0.45
    ax_p.add_patch(mpatches.FancyBboxPatch(
        (0.4, qed_bot), 11.2, 1.05, boxstyle='round,pad=0.2',
        fc='#FDEDEC', ec=RED, lw=2.0))
    ax_p.text(6.0, qed_bot + 0.52,
              r'$\therefore$  $a(t)=K_p\,e(t)+K_d\,d(t)+K_c\,|d(t)|+K_t\,I_b(t)$'
              r'   [4 free gain parameters: $b_p, b_d, b_c, b_t$]   □',
              ha='center', fontsize=9, va='center', color=RED, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(SCRIPT_DIR, 'reference_proof4_final.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f'Saved: {out}')
    plt.close()


# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    fig_neuron()
    fig_activation()
    fig_synapse()
    fig_substitution()
    fig_steady_state()
    fig_fsa_subnetworks()
    fig_peterka_eq15()
    fig_mcneal_split_deriv()
    fig_seg6b_bias_split()
    fig_tc4_assembly()
    fig_node_sensory()
    fig_node_derivative()
    fig_node_gain()
    fig_proof_error()
    fig_proof_derivative()
    fig_proof_gain()
    fig_proof_final()
    print('\nAll reference figures saved.')
