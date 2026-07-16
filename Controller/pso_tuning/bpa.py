"""Isometric force model for a Festo phi10mm braided pneumatic actuator (BPA).

From Bolen et al. 2026, "Isometric Force Characterization of Braided Pneumatic
Actuators" (Hunt lab), for the 10 mm diameter Festo fluidic muscle.

BPA physics differs fundamentally from a Hill muscle, which is why the MuJoCo
<muscle> was the wrong model for this rig:
  - Force ONLY in tension, ONLY when pressurised. Deflated (P=0) -> F=0. There is
    NO passive spring (a Hill muscle has a passive force-length curve; the BPA does
    not).
  - Maximum force at RESTING (= maximum, uninflated) length lrest, decreasing
    monotonically as the actuator contracts. (A Hill muscle peaks mid-range.)
  - Force is ~linear in pressure.

Equations (phi10mm):
  Eq.5  F620(lrest)     = 303.5 * arctan(19.03 * (lrest - 0.0075))          [N, at 620 kPa]
  Eq.8  Fstar(e*, P*)   = c0*(exp(-c1*e*) - 1) + P* * exp(-c2*e*^2), clipped >= 0
        c0, c1, c2 = 0.5682, 4.254, 0.5597
  Eq.9  F(e*,P*,lrest)  = Fstar(e*, P*) * F620(lrest)
  with contraction   e   = (lrest - l)/lrest
       relative      e*  = e / EPS620
       relative pres P*  = P / P620

Static (isometric, S=0) model: hysteresis/velocity terms are omitted, which the
paper (sec 5.4) endorses for a force-map control approach.
"""
import numpy as np

# ---- coefficients (Bolen table 2, phi10mm) ----
_C0, _C1, _C2 = 0.5682, 4.254, 0.5597
P620 = 620.0            # kPa, characterisation pressure and eps620 reference

# ---- constants tied to the PHYSICAL BUILD (not free parameters) ----
# LREST: the BPA resting (= maximum, uninflated) length -- a HARDWARE MOUNTING choice
# that sets the operating-point engagement. The upright tendon length is 0.265 m; the
# tendon reaches 0.2832 m at the joint limit (+-0.3 rad).
#   - LREST = 0.2832 (rest at the joint limit) keeps the actuator taut across the whole
#     range, BUT at upright it sits at eps* ~ 0.43, where Bolen Eq.8 needs ~330 kPa just
#     to develop force. Under a small (2 deg) sway the controller barely engages -> the
#     body rides the platform (flat gain ~1.0) and P_MAX is inert. Poor operating point.
#   - LREST ~ 0.267 (rest just above upright) puts upright near rest length, so the BPA
#     engages readily at the operating sways. Measured: reaches the Peterka gain
#     magnitude (peak ~3.0-3.6, no delay) vs ~1.0 at 0.2832. It goes slack if the joint
#     tilts more than a few degrees toward that muscle's lengthening side, which is fine
#     for balancing about upright (a BPA antagonist only ever pulls with one side).
# Recommendation: mount the BPAs at a resting length ~0.267 m. MUST match the hardware.
LREST = 0.267           # m

# EPS620: maximum contraction at 620 kPa (Bolen Fig 3b). VARIES per muscle -- measure
# eps620 on each real actuator (Bolen sec 4.3 stresses it is not predictable a priori).
# Across the joint range the actuator sees eps up to ~0.136 < EPS620, so it always
# produces force.
EPS620 = 0.15


def F620(lrest=LREST):
    """Max isometric force (N) at 620 kPa for resting length lrest (Bolen Eq.5)."""
    return 303.5 * np.arctan(19.03 * (lrest - 0.0075))


def Fstar(eps_star, P_star):
    """Normalised force F* in [0,1] vs relative contraction and pressure (Bolen Eq.8)."""
    f = _C0 * (np.exp(-_C1 * eps_star) - 1.0) + P_star * np.exp(-_C2 * eps_star ** 2)
    return max(0.0, f)


def bpa_force(l, lrest, P_kpa):
    """Isometric BPA tension (N) at current length l, resting length lrest, pressure P.

    Tension is always >= 0 (a BPA pulls only). Returns 0 if deflated (P<=0), if the
    actuator is slack (l >= lrest, cannot be stretched past rest to make force), or if
    it has contracted past eps620.
    """
    if P_kpa <= 0.0:
        return 0.0
    eps = (lrest - l) / lrest
    if eps <= 0.0:
        return 0.0                      # at or beyond rest length -> slack
    eps_star = eps / EPS620
    if eps_star >= 1.0:
        return 0.0                      # contracted past eps620 -> no force
    return Fstar(eps_star, P_kpa / P620) * F620(lrest)


if __name__ == "__main__":
    print("F620(LREST=%.4f) = %.1f N" % (LREST, F620()))
    for P in (0, 200, 400, 620):
        print("  at rest length,   P=%3d kPa -> %.1f N" % (P, bpa_force(LREST - 1e-6, LREST, P)))
    print("  at upright 0.265, P=620 kPa -> %.1f N" % bpa_force(0.265, LREST, 620))
    print("  slack (l>lrest),  P=620 kPa -> %.1f N" % bpa_force(0.30, LREST, 620))
