"""
test_subnetworks.py  -  Empirical verification of all SNS functional subnetworks.

Each test simulates the subnetwork, then checks the final voltage against the
analytical steady-state prediction from Eq. 13 (derived by setting dV/dt = 0):

    U* = [Σ (gs_i/R · U_pre_i · ΔEs_i) + I_app] / [Gm + Σ (gs_i/R · U_pre_i)]

    U     = V − Er          activation above rest (range 0–R = 20 mV)
    U_pre = V_pre − Elo     presynaptic activation (0 when silent, R when saturated)
    ΔEs   = Es − Er         driving force (+194 mV excitatory, −40 mV inhibitory)

Run:
    python test_subnetworks.py            # prints PASS / FAIL for each assertion
    python test_subnetworks.py --plot     # also shows matplotlib traces
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulate import simulate
from sns_subnetworks import (
    make_transmission_network,
    make_addition_network,
    make_subtraction_network,
    make_multiplication_network,
    make_division_network,
    make_derivative_network,
    make_integral_network,
    Er, R, GM_DEFAULT as Gm,
    gs_exc, gs_inh, Es_exc, Es_inh,
)

PLOT = "--plot" in sys.argv

DT    = 0.1    # ms per step
STEPS = 1000   # 100 ms - >> τ_slow = 10 ms, good enough for all steady-state tests

# Driving forces (used throughout Eq. 13)
dEe   = Es_exc - Er   # +194 mV  excitatory
dEinh = Es_inh - Er   # − 40 mV  inhibitory


# -- shared helpers ------------------------------------------------------------

def run(neurons, synapses, I_app, steps=STEPS):
    """Return the full voltage log and a {name: final_U} dict."""
    log = simulate(neurons, synapses, I_app, dt_ms=DT, steps=steps)
    final_U = {k: log[k][-1] - Er for k in log}
    return log, final_U


def eq13(*terms, I_app=0.0):
    """
    Eq. 13 analytical steady-state.

    Each term is (gs, U_pre, dEs):
        gs    - max conductance of that synapse (µS)
        U_pre - presynaptic activation, already clamped to [0, R] (mV)
        dEs   - driving force Es − Er for that synapse (mV)
    I_app in nA; Gm in µS -> result in mV.
    """
    numer = sum(gs / R * U_pre * dEs for gs, U_pre, dEs in terms) + I_app
    denom = Gm + sum(gs / R * U_pre for gs, U_pre, _ in terms)
    return numer / denom


def u_iso(I_app):
    """Steady-state activation of an isolated (no-synapse) neuron: U* = I_app / Gm."""
    return I_app / Gm


def pwl(U):
    """PWL clamping of presynaptic activation to [0, R]."""
    return max(0.0, min(R, U))


def check(label, sim, expected, tol=0.05):
    err = abs(sim - expected)
    tag = "PASS" if err < tol else "FAIL"
    print(f"  [{tag}]  {label}")
    print(f"          sim={sim:+.4f} mV   expected={expected:+.4f} mV   err={err:.4f} mV")
    assert err < tol, f"FAILED '{label}': |{sim:.4f} − {expected:.4f}| = {err:.4f} > tol {tol}"


def show(log, title, hi=()):
    if not PLOT:
        return
    try:
        import matplotlib.pyplot as plt
        t = np.arange(len(next(iter(log.values())))) * DT
        fig, ax = plt.subplots()
        for name, v in log.items():
            ax.plot(t, v - Er, label=name, lw=(2.5 if name in hi else 1.0))
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("U = V − Er  (mV)")
        ax.set_title(title)
        ax.legend(fontsize=7)
        plt.tight_layout()
        plt.show()
    except ImportError:
        pass


# -- Test 1 - Transmission ----------------------------------------------------

def test_transmission():
    """
    Claim:  U_post* = target_gain x U_pre  at the FSA design point U_pre = R/2.

    Derivation
    ----------
    Eq. 13 for one excitatory synapse, no I_app:

        U_post* = (gs/R · U_pre · dEe) / (Gm + gs/R · U_pre)

    We want U_post* = gain x U_pre.  Set U_pre = R/2 and solve for gs:

        gain · R/2 · (Gm + gs/2) = gs/2 · dEe
        gain · Gm · R/2          = gs/2 · (dEe − gain · R/2)
        gs                       = gain · Gm · R / (dEe − gain · R/2)

    That is exactly the formula in make_transmission_network().  By construction,
    plugging this gs back into Eq. 13 at U_pre = R/2 returns gain x R/2.
    """
    print("\n-- Test 1: Transmission ------------------------------------------")
    for gain in [0.30, 0.70]:
        pre, post, syn = make_transmission_network(gain)
        neurons  = {pre.name: pre, post.name: post}

        I_pre = R / 2                      # design-point drive: U_pre* = 10 mV
        _, fu = run(neurons, [syn], {pre.name: I_pre})

        U_pre  = u_iso(I_pre)              # = 10 mV (isolated neuron, no synapses)
        U_an   = eq13((syn.gs_max, U_pre, dEe))   # Eq. 13 prediction
        U_tgt  = gain * U_pre              # equals U_an by construction

        check(f"gain={gain:.2f}: sim vs analytical Eq.13", fu[post.name], U_an, tol=0.01)
        check(f"gain={gain:.2f}: sim vs target {gain:.2f}x10", fu[post.name], U_tgt, tol=0.01)


# -- Test 2 - Addition --------------------------------------------------------

def test_addition():
    """
    Claim:  U_sum* ≈ U_a + U_b  (exactly via Eq. 13; approximately linear for small inputs).

    Derivation
    ----------
    Two excitatory synapses (both gs_exc) on the sum neuron, no I_app:

        U_sum* = (gs_exc/R · dEe · U_a + gs_exc/R · dEe · U_b)
               / (Gm + gs_exc/R · U_a + gs_exc/R · U_b)
               = K (U_a + U_b) / (1 + (K/dEe)(U_a + U_b))

    where K = gs_exc · dEe / R = 0.115 x 194 / 20 = 1.115.

    gs_exc was chosen so K ≈ 1 (unity gain at maximum drive, from
    gs_exc = Gm·R/(dEe−R) = 20/174 ≈ 0.115 µS), so U_sum ≈ U_a + U_b
    when (U_a + U_b) is small relative to R.
    """
    print("\n-- Test 2: Addition ----------------------------------------------")
    p1, p2, s, syn1, syn2 = make_addition_network()
    neurons = {p1.name: p1, p2.name: p2, s.name: s}

    I1, I2 = 1.0, 0.5
    log, fu = run(neurons, [syn1, syn2], {p1.name: I1, p2.name: I2})

    U1, U2 = u_iso(I1), u_iso(I2)
    U_an   = eq13((gs_exc, U1, dEe), (gs_exc, U2, dEe))

    check("U_sum vs Eq.13", fu[s.name], U_an, tol=0.01)
    approx_err_pct = 100 * (fu[s.name] / (U1 + U2) - 1)
    print(f"          approximate addition: {U1}+{U2}={U1+U2:.2f},  actual {fu[s.name]:.4f}  ({approx_err_pct:+.1f}% linearity error)")

    show(log, "Addition", hi=[s.name])


# -- Test 3 - Subtraction -----------------------------------------------------

def test_subtraction():
    """
    Claim:  U_diff* ≈ U_a − U_b.

    Derivation
    ----------
    One excitatory (gs_exc) and one inhibitory (gs_inh) synapse on the diff neuron.
    Eq. 13 numerator when U_a = U_b = U:

        gs_exc · dEe · U/R  +  gs_inh · dEinh · U/R
        = (U/R) · (gs_exc · dEe + gs_inh · dEinh)
        = (U/R) · (0.115 x 194 + 0.558 x (−40))
        = (U/R) · (22.31 − 22.32) ≈ 0

    Exact cancellation is by design: gs_inh = gs_exc · dEe / |dEinh|.
    When U_a > U_b the excitatory term dominates -> U_diff > 0.
    When U_a < U_b the inhibitory term dominates -> U_diff < 0 (floored at Er).
    """
    print("\n-- Test 3: Subtraction -------------------------------------------")

    def build():
        p1, p2, d, s1, s2 = make_subtraction_network()
        return {p1.name: p1, p2.name: p2, d.name: d}, [s1, s2], d.name

    # Case A - equal inputs: numerator ≈ 0 -> U_diff ≈ 0
    ns, syns, dname = build()
    _, fu = run(ns, syns, {list(ns)[0]: 1.0, list(ns)[1]: 1.0})
    check("A (I_a=I_b=1.0): U_diff -> 0", fu[dname], 0.0, tol=0.05)

    # Case B - unequal inputs: U_diff > 0, matches Eq. 13
    ns, syns, dname = build()
    Ia, Ib = 2.0, 1.0
    log, fu = run(ns, syns, {list(ns)[0]: Ia, list(ns)[1]: Ib})
    Ua, Ub  = u_iso(Ia), u_iso(Ib)
    U_an    = eq13((gs_exc, Ua, dEe), (gs_inh, Ub, dEinh))
    check("B (I_a=2.0, I_b=1.0): U_diff vs Eq.13", fu[dname], U_an, tol=0.05)
    assert fu[dname] > 0, f"FAIL: U_diff should be positive, got {fu[dname]:.4f}"
    print(f"  [PASS]  B: U_diff > 0 confirmed")

    show(log, "Subtraction (I_a=2, I_b=1)", hi=[dname])


# -- Test 4 - Multiplication ---------------------------------------------------

def test_multiplication():
    """
    Claim:  U_prod increases with the gain input (U_pre2) for a fixed signal (U_pre1).

    Topology (FSA Fig. 4D, double-inhibition):

        pre2_gain  ->[INH]->  mul_mod  ← I_app = R·Gm = 20 nA
                                ↓[INH]
        pre1_signal ->[EXC]->  mul_prod

    Derivation (two-step Eq. 13)
    ----------------------------
    Step 1 - mod neuron:
        I_app = 20 nA drives U_mod = R when pre2 is silent.
        Inhibitory pre2 reduces U_mod below R:

        U_mod* = (gs_inh/R · U_pre2 · dEinh + R·Gm) / (Gm + gs_inh/R · U_pre2)

        U_pre2 = 0  ->  U_mod* = R   (max inhibition of prod)
        U_pre2 = R  ->  U_mod* ≈ 0   (mod silenced, no inhibition of prod)

    Step 2 - prod neuron:
        U_prod* = (gs_exc/R · U_pre1 · dEe + gs_inh/R · U_mod_eff · dEinh)
                / (Gm + gs_exc/R · U_pre1 + gs_inh/R · U_mod_eff)

        where U_mod_eff = pwl(U_mod*) clamps to [0, R].

    As U_pre2 rises: U_mod falls -> less inhibition -> more signal passes -> U_prod rises.
    """
    print("\n-- Test 4: Multiplication ----------------------------------------")
    I_pre1 = 5.0       # fixed signal input
    I_mod  = R         # constant applied to mod neuron (nA; Gm=1 µS -> U_mod_baseline = R)

    # Four gain levels: expect U_prod to increase monotonically
    cases = [
        (0.0,  "gain=0  (mod at R, prod fully suppressed)"),
        (14.0, "gain=14 (partial relief, U_prod > 0)"),
        (16.0, "gain=16 (more relief)"),
        (18.0, "gain=18 (mod near-silent, full pass-through)"),
    ]

    U_prod_sim = []
    for I_pre2, label in cases:
        p1, p2, mod, prod, syn1, syn2, syn3 = make_multiplication_network()
        neurons  = {p1.name: p1, p2.name: p2, mod.name: mod, prod.name: prod}
        synapses = [syn1, syn2, syn3]
        I_app    = {p1.name: I_pre1, p2.name: I_pre2, mod.name: I_mod}
        _, fu    = run(neurons, synapses, I_app)

        # Two-step analytical
        U_pre2    = u_iso(I_pre2)
        U_pre1    = u_iso(I_pre1)
        U_mod_ss  = eq13((gs_inh, U_pre2, dEinh), I_app=I_mod)   # Step 1
        U_mod_eff = pwl(U_mod_ss)                                  # clamp activation
        U_prod_an = eq13((gs_exc, U_pre1, dEe), (gs_inh, U_mod_eff, dEinh))  # Step 2

        check(label, fu[prod.name], U_prod_an, tol=0.1)
        U_prod_sim.append(fu[prod.name])

    # Monotonicity: U_prod strictly increases with gain input
    for i in range(len(U_prod_sim) - 1):
        lo, hi = U_prod_sim[i], U_prod_sim[i + 1]
        assert lo < hi, f"FAIL monotonicity: case {i} ({lo:.3f}) >= case {i+1} ({hi:.3f})"
    print(f"  [PASS]  U_prod increases monotonically: "
          f"{' < '.join(f'{v:.2f}' for v in U_prod_sim)}")


# -- Test 5 - Division --------------------------------------------------------

def test_division():
    """
    Characterises actual behaviour of make_division_network().

    Topology:
        pre2 ->[INH]-> mod ->[EXC]-> quot   AND   pre1 ->[EXC]-> quot

    Without I_app on mod, any positive pre2 drives V_mod below Elo,
    clamping Gs_mod = 0 (PWL floor).  The mod->quot excitatory pathway
    is therefore permanently silent for pre2 ≥ 0.

    Result: U_quot depends only on pre1 (same whether pre2 = 0 or 5 nA).

    Eq. 13 (with mod silent, contribution = 0):
        U_quot* = (gs_exc/R · U_pre1 · dEe) / (Gm + gs_exc/R · U_pre1)

    Note: true divisive gain control would require I_app = R on mod so that
    pre2 can reduce mod below a non-zero baseline - analogous to the
    multiplication fix.
    """
    print("\n-- Test 5: Division (characterisation) ---------------------------")
    I_pre1 = 1.0
    U_pre1 = u_iso(I_pre1)
    U_quot_an = eq13((gs_exc, U_pre1, dEe))   # mod contributes nothing

    for I_pre2, label in [(0.0, "pre2=0 (mod at rest)"),
                          (5.0, "pre2=5 (mod hyperpolarised)")]:
        p1, p2, mod, quot, s1, s2, s3 = make_division_network()
        neurons  = {p1.name: p1, p2.name: p2, mod.name: mod, quot.name: quot}
        _, fu    = run(neurons, [s1, s2, s3], {p1.name: I_pre1, p2.name: I_pre2})
        check(label, fu[quot.name], U_quot_an, tol=0.05)

    print("  [NOTE]  pre2 has no measurable effect: mod is silenced before it can gate quot")
    print("          True division requires I_app = R on mod (known architectural gap)")


# -- Test 6 - Derivative ------------------------------------------------------

def test_derivative():
    """
    Claim:  output ≈ 8 ms · d(input)/dt  (step response: positive transient, decays to 0).

    Derivation
    ----------
    Both fast (τ_f = 2 ms) and slow (τ_s = 10 ms) receive the same step input.
    Each responds as  U(t) = U_∞ (1 − e^{−t/τ}).  The output is their difference:

        U_out(t) = U_fast(t) − U_slow(t) = U_∞ [e^{−t/τ_s} − e^{−t/τ_f}]

    In the Laplace domain:
        H(s) = 1/(1+τ_f·s) − 1/(1+τ_s·s) = (τ_s − τ_f)·s / [(1+τ_f·s)(1+τ_s·s)]

    For ω << 1/τ_f = 500 rad/s:  H(s) ≈ (10−2)·s = 8 ms · s  ->  d/dt scaled by 8 ms.

    Steady-state: both neurons converge to U_∞.  Eq. 13 numerator of out-neuron:
        gs_exc · dEe · U_∞ + gs_inh · dEinh · U_∞ ≈ 0  (by the cancellation condition)
    -> U_out* -> 0 as t -> ∞.
    """
    print("\n-- Test 6: Derivative --------------------------------------------")
    STEPS_D = 2000   # 200 ms - need >> τ_slow x 10 = 100 ms for full convergence

    fast, slow, out, syn1, syn2 = make_derivative_network()
    neurons  = {fast.name: fast, slow.name: slow, out.name: out}
    synapses = [syn1, syn2]

    log, _ = run(neurons, synapses, {fast.name: 5.0, slow.name: 5.0}, steps=STEPS_D)

    U_out  = log[out.name]  - Er
    U_fast = log[fast.name] - Er
    U_slow = log[slow.name] - Er

    # 1. Steady-state converges to 0
    check("steady-state U_out -> 0  (derivative of constant = 0)", U_out[-1], 0.0, tol=0.05)

    # 2. Transient peak is positive (fast outpaces slow on step-up)
    n20 = int(20 / DT)
    peak = float(U_out[:n20].max())
    assert peak > 0, f"FAIL: transient peak should be positive, got {peak:.4f}"
    print(f"  [PASS]  transient peak = {peak:.4f} mV > 0 in first 20 ms")

    # 3. Fast leads slow during transient
    assert U_fast[:n20].mean() > U_slow[:n20].mean(), "FAIL: fast should lead slow"
    print(f"  [PASS]  fast mean ({U_fast[:n20].mean():.3f}) > slow mean ({U_slow[:n20].mean():.3f}) in first 20 ms")

    show(log, "Derivative (step input to fast + slow)", hi=[out.name])


# -- Test 7 - Integral (bistable hold) ----------------------------------------

def test_integral():
    """
    Claim:  after being driven above zero, the network holds a non-zero voltage
            indefinitely with no applied current.

    Derivation
    ----------
    Mutual excitation (u1 ↔ u2, both gs_exc) has two Eq. 13 fixed points.
    Setting U_u1* = U_u2* = U* (symmetric solution, no I_app):

        U* = (gs_exc/R · U* · dEe) / (Gm + gs_exc/R · U*)

    Solutions:
        U* = 0                             (unstable - any perturbation grows)
        U* = dEe − Gm · R / gs_exc ≈ R    (stable - mutual excitation sustains it)

    Once driven above zero the system locks to the non-zero attractor (U* ≈ R)
    and remains there when I_app is removed.  This is bistable short-term memory.
    """
    print("\n-- Test 7: Integral (bistable hold) ------------------------------")
    DRIVE = 500    # 50 ms drive
    HOLD  = 2000   # 200 ms hold with no I_app

    u1, u2, syn1, syn2 = make_integral_network()
    neurons  = {u1.name: u1, u2.name: u2}
    synapses = [syn1, syn2]

    # Phase 1: drive u1 - neurons update V in-place so state carries into phase 2
    log_d, fu_d = run(neurons, synapses, {u1.name: 1.0}, steps=DRIVE)
    assert fu_d[u1.name] > 0.5, f"FAIL: u1 should activate during drive, U={fu_d[u1.name]:.3f}"
    assert fu_d[u2.name] > 0.5, f"FAIL: u2 should activate during drive, U={fu_d[u2.name]:.3f}"
    print(f"  [PASS]  end of drive: U_u1={fu_d[u1.name]:.3f} mV, U_u2={fu_d[u2.name]:.3f} mV")

    # Phase 2: no I_app - neurons keep their current voltage (simulate modifies V in-place)
    log_h, fu_h = run(neurons, synapses, {}, steps=HOLD)
    assert fu_h[u1.name] > 0.5, f"FAIL: u1 should hold state, U={fu_h[u1.name]:.3f}"
    assert fu_h[u2.name] > 0.5, f"FAIL: u2 should hold state, U={fu_h[u2.name]:.3f}"
    print(f"  [PASS]  end of hold:  U_u1={fu_h[u1.name]:.3f} mV, U_u2={fu_h[u2.name]:.3f} mV (bistable hold confirmed)")

    if PLOT:
        combined = {k: np.concatenate([log_d[k], log_h[k]]) for k in log_d}
        show(combined, "Integral: drive (0–50 ms) then hold (50–250 ms)",
             hi=[u1.name, u2.name])


# -- run all -------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_transmission,
        test_addition,
        test_subtraction,
        test_multiplication,
        test_division,
        test_derivative,
        test_integral,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"\n  *** {e}")
            failed += 1

    print(f"\n{'='*56}")
    print(f"  {passed} passed   {failed} failed")
    print(f"{'='*56}")
    if failed:
        sys.exit(1)
