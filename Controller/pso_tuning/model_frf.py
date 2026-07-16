"""
Step 2 - the model's frequency response.

Runs the SNS + MuJoCo balance loop headless (mode 3: synthetic IMU ->
complementary filter) under the PRTS support-surface perturbation, and returns
the body-sway / stimulus frequency response at the PRTS harmonic frequencies.
"""
import os
import sys
import numpy as np
import mujoco

# make the sibling Controller/ modules importable
_CTRL = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _CTRL not in sys.path:
    sys.path.insert(0, _CTRL)
import balance
import bpa
from prts_generator import generate_prts
from complementary_filter import ComplementaryFilter

_XML = os.path.join(_CTRL, "mujoco_lessons", "e6_bilateral.xml")

DT = 1e-4                       # MUST match <option timestep> in e6_bilateral.xml
R, Er = 20.0, -60.0             # SNS operating range (mV) and resting potential
DEG_MAX, IB_SCALE, IB_CLIP = 10.0, 10.0 / 1500.0, 10.0

# Ceiling of the SNS-output -> BPA-pressure map (kPa). The SNS output range
# [Er, Er+R] = [-60, -40] mV maps linearly to [0, P_MAX] kPa of actuator pressure.
# CALIBRATED so the SNS effective static K_P at the published gains lands near
# Peterka's target of 1.3*mgh = 150 N*m/rad (see calibrate step). Not physically
# fixed -- it is the controller's authority scaling for this rig.
P_MAX = 620.0

# Peterka Fig 5C stimulus condition: 2 deg peak-to-peak support-surface tilt.
# Must match the experiment that produced the target data -- human gain is
# strongly amplitude-dependent (sensory reweighting), so the 2 deg curve is not
# the 4 deg curve.  The model is amplitude-dependent too (rectifying Ib
# feedback, clipping adapters, saturating SNS neurons), so this is not a free
# parameter: halving the drive from 4.25 to 2.0 deg pp moves the model gain by
# ~5% on average (17% worst case).
PRTS_PP_DEG = 2.0

# Published gains (b_p, b_d, b_c, b_t) from McNeal & Hunt table IV, row "Split, k_t on"
# (rho = 0.33, rho_hat = 0.22) -- the paper's own best split-derivative configuration.
# NOT hand-tuned starting values: this is the result the extension has to beat, and the
# seed for particle 0 in pso.py.
# Note b_d != b_c is the paper's FINDING, not drift. Setting b_c = b_d collapses the two
# derivative pathways into a conventional single derivative gain.
BASELINE_GAINS = (4.26, 5.01, 2.48, 5.42)

# Long-loop sensorimotor delay (tau_d).  Peterka lumps the whole long-loop latency into
# one e^(-s*tau_d) on the weighted sensory error.  His fitted EFFECTIVE delay is
# amplitude-dependent (it absorbs a compensation mechanism, so it is longer than a raw
# neural transport delay): larger stimuli -> shorter effective delay.
#   lowest amplitude (0.5 deg pp): tau_d ~ 0.191 s
#   highest        (8   deg pp)  : tau_d ~ 0.105 s
# (Approximate endpoints; confirm against Peterka 2002's fitted table.)  0.090 s was used
# previously to match peterka_controller.py / sns_tc4_controller.py, but that is NOT
# Peterka's effective value -- it is closer to a raw transport delay.  Interpolated in
# log-amplitude, the same way bs_wt tracks amplitude; 2 deg pp -> ~0.148 s.
_TAU_LO = (0.5, 0.191)          # (PRTS pp deg, tau_d s) at Peterka's smallest amplitude
_TAU_HI = (8.0, 0.105)          # (PRTS pp deg, tau_d s) at Peterka's largest amplitude


def tau_d_for(pp_deg):
    """Peterka effective sensorimotor delay (s) for a PRTS amplitude (deg pp).

    Log-amplitude interpolation between the reported endpoints, clamped outside them.
    """
    (a_lo, t_lo), (a_hi, t_hi) = _TAU_LO, _TAU_HI
    t = (np.log(pp_deg) - np.log(a_lo)) / (np.log(a_hi) - np.log(a_lo))
    lo, hi = min(t_lo, t_hi), max(t_lo, t_hi)
    return float(np.clip(t_lo + t * (t_hi - t_lo), lo, hi))


SENSORY_DELAY_S = tau_d_for(PRTS_PP_DEG)   # ~0.148 s at 2 deg pp

# Delay on the Ib tension channels.  McNeal & Hunt fig.2 shows a tau_d block on the
# tension path as well as the joint-angle path, so both afferent groups carry the same
# lumped delay.  Kept as its own constant only so the two can be separated for
# diagnostics; it is NOT a free parameter.
# (Note: sec.III-B-1 of that paper says the delay is applied "only to theta_A", which
# contradicts fig.2.  Fig.2 is taken as authoritative here.  A previous version of this
# file set 0.035 s on the reasoning that Ib is a fast spinal reflex -- that value appears
# in neither the paper nor Hilts and was invented; do not reintroduce it without a source.)
IB_DELAY_S = SENSORY_DELAY_S

# ---- Peterka's sensory reweighting: graviceptive weight w_g vs PRTS amplitude ----
#
# Eyes closed, support-surface stimulation. Peterka reports only the two ENDPOINTS of
# his amplitude series:
#     lowest amplitude   : w_p = 0.70, w_g = 0.30   (proprioception dominates)
#     highest, 8 deg pp  : w_p = 0.24, w_g = 0.76   (graviception dominates)
# The intermediate amplitudes are NOT tabulated in the paper, so w_g at 2 deg pp -- the
# amplitude of the Fig 5C target data -- has to be interpolated. Reasoning:
#
#   1. His amplitudes form a geometric series (0.5, 1, 2, 4, 8 deg pp; each a doubling).
#      Choosing amplitudes that way implies the effect was expected to scale with LOG
#      amplitude, as sensory phenomena generally do (Weber-Fechner). So we interpolate
#      linearly in log(amplitude), not in amplitude.
#   2. 2 deg pp happens to be the exact geometric mean of the two reported endpoints:
#      sqrt(0.5 * 8) = 2. So it lands exactly halfway in log-amplitude, and w_g is simply
#      the average of the endpoints: (0.30 + 0.76)/2 = 0.53.
#   3. Landing at the midpoint is the best case for this kind of estimate. If the true
#      reweighting is sigmoidal in log-amplitude (typical for a saturating shift between
#      two bounded weights), a sigmoid is most nearly linear through its centre -- so the
#      midpoint is exactly where log-linear interpolation is most trustworthy. The 1 and
#      4 deg pp points would be the ones to distrust, not 2.
#
# ASSUMPTION TO CHECK: that the "lowest amplitude" quoted above is 0.5 deg pp. If the
# series actually starts at 1 deg pp, fix _WG_LO and w_g(2 deg) becomes ~0.45, not 0.53.
# The endpoints are named so that correction is a one-line change.
#
# This is an ESTIMATE, not data. If a fit turns out to be sensitive to it, promote bs_wt
# to a searched parameter rather than leaning on this number.
_WG_LO = (0.5, 0.30)            # (PRTS pp deg, w_g) at Peterka's smallest amplitude
_WG_HI = (8.0, 0.76)            # (PRTS pp deg, w_g) at Peterka's largest amplitude


def bs_wt_for(pp_deg):
    """Graviceptive weight w_g for a PRTS amplitude (deg pp); bf_wt = 1 - w_g.

    Linear in log-amplitude between Peterka's two reported endpoints, clamped outside
    them -- his series spans 0.5-8 deg pp and there is no basis for extrapolating past it.
    """
    (a_lo, w_lo), (a_hi, w_hi) = _WG_LO, _WG_HI
    t = (np.log(pp_deg) - np.log(a_lo)) / (np.log(a_hi) - np.log(a_lo))
    return float(np.clip(w_lo + t * (w_hi - w_lo), w_lo, w_hi))


FMIN, FMAX = 0.01, 3.0          # analysis band (Hz)

# Minimum fraction of in-band body-sway energy that must sit at stimulus
# frequencies for a run to count as a real measurement.
#
# A delayed loop can settle into a bounded limit cycle. It is finite, so the
# isfinite() check in model_frf() passes, and the oscillation then leaks into the
# PRTS bins -- which carry as little as 2% of the stimulus power up at 0.7 Hz --
# and manufactures a convincing resonant gain peak out of noise. Measured on the
# baseline gains: a healthy stimulus-driven run sits at 99.2%, while a 1.74 Hz
# limit cycle sits at 5.7%. 0.5 is nowhere near either, so it separates them
# cleanly without tuning.
MIN_DRIVEN_FRAC = 0.5


# ---- adapters between the physical world and the SNS (from g2_integrate.py) ----
def _a2na(a_rad):
    return float(np.clip(np.degrees(a_rad) / DEG_MAX * (R / 2), -R / 2, R / 2))

def _t2ib(force_n):
    return float(np.clip(abs(force_n) * IB_SCALE, 0.0, IB_CLIP))

def _v2p(v_mv, p_max=P_MAX):
    """SNS output (mV) -> BPA pressure command (kPa). [Er, Er+R] -> [0, p_max]."""
    return float(np.clip((v_mv - Er) / R, 0.0, 1.0)) * p_max


def _make_cf(alpha=0.98):
    """Build the complementary filter WITHOUT its interactive input() prompt."""
    cf = ComplementaryFilter.__new__(ComplementaryFilter)
    cf.alpha, cf.angle = alpha, 0.0
    return cf


def _prts_periods(n_periods, pp_deg=PRTS_PP_DEG):
    """Continuous n-period PRTS drive (position + velocity), zero-mean.

    Built at unit velocity amplitude then rescaled to exactly `pp_deg`
    peak-to-peak.  generate_prts is linear in v_amp, so the rescale is exact and
    equivalent to picking the right v_amp -- but peak-to-peak is the quantity
    Peterka reports, so it is the one worth naming.
    """
    _, pos1 = generate_prts(v_amp=np.deg2rad(1.0), dt=DT)
    pos1 = pos1 * (np.deg2rad(pp_deg) / (pos1.max() - pos1.min()))
    vel = np.tile(np.diff(pos1, prepend=pos1[0]) / DT, n_periods)
    pos = np.cumsum(vel) * DT
    pos -= pos.mean()
    return pos, vel, len(pos1)


def _significant(x, fmin=FMIN, fmax=FMAX, frac=0.02):
    """FFT of x; keep bins in [fmin, fmax] whose magnitude clears frac*peak."""
    X = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(len(x), DT)
    band = (f >= fmin) & (f <= fmax)
    sig = band & (np.abs(X) > frac * np.abs(X)[band].max())
    return X, f, sig


def analysis_freqs(n_periods=2):
    """The in-band PRTS harmonic frequencies (gain-independent, no simulation)."""
    pos, _, plen = _prts_periods(n_periods)
    _, f, sig = _significant(pos[-plen:])
    return np.sort(f[sig])


def model_frf(gains, ctrlr_mode=3, bs_wt=None, n_periods=2, cf_alpha=0.98,
              delay_s=SENSORY_DELAY_S, ib_delay_s=IB_DELAY_S,
              min_driven=MIN_DRIVEN_FRAC, p_max=P_MAX):
    """Simulate the closed loop; return (freqs, gain, phase_deg) or (None,)*3.

    Returns (None,)*3 for an unusable run: one that diverged, or one the stimulus
    isn't driving (a limit cycle -- see MIN_DRIVEN_FRAC).  frf_cost turns that
    into its `fail` score.

    `delay_s` is the long-loop sensorimotor delay on the bf/bs channels;
    `ib_delay_s` is the shorter spinal Ib reflex latency on the tension channels.
    Pass 0.0 to either to remove it -- both zero reproduces the pre-delay model.
    `min_driven` is the stimulus-driven energy floor; pass 0.0 to disable.

    `bs_wt` defaults to bs_wt_for(PRTS_PP_DEG): the sensory split has to track the
    stimulus amplitude, so leaving it None keeps the two from drifting apart. Pass a
    number only to override deliberately (e.g. to search it).
    """
    if bs_wt is None:
        bs_wt = bs_wt_for(PRTS_PP_DEG)
    m = mujoco.MjModel.from_xml_path(_XML)
    d = mujoco.MjData(m)
    assert abs(m.opt.timestep - DT) < 1e-12, "DT must match the XML timestep"

    aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ankle_joint")
    pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "platform_joint")
    QA, VA = m.jnt_qposadr[aid], m.jnt_dofadr[aid]
    QP, VP = m.jnt_qposadr[pid], m.jnt_dofadr[pid]

    pos, vel, plen = _prts_periods(n_periods)
    net = balance.generate_sns(gains, ctrlr_mode=ctrlr_mode, bs_wt=bs_wt)
    sns = net.compile(backend="numpy", dt=DT * 1000.0, debug=False)
    inp = np.zeros(net.get_num_inputs())
    cf = _make_cf(cf_alpha)

    mujoco.mj_resetData(m, d)
    d.qpos[QA] = 0.0
    mujoco.mj_forward(m, d)

    x = np.empty(len(pos))
    y = np.empty(len(pos))

    # Afferent delay lines: two pathways, two latencies. bf/bs carry the long
    # loop delay_s; the Ib tension channels carry the shorter spinal ib_delay_s.
    # One shared ring buffer read at two offsets.
    #
    # Ring buffer, NOT np.roll: this runs 1.2M times per simulation, and rolling
    # the whole buffer each step would copy ~4.3e9 floats per call.
    # Slot (bi - n) % N holds the sample written n steps ago, so reading at
    # offset n is an exact n-step delay. n=0 reads the sample just written, which
    # is how delay_s=0 degrades to "no delay" without a special case. N needs the
    # +1 so that the longest delay still reads a past sample rather than itself.
    n_long = int(round(delay_s / DT))
    n_ib = int(round(ib_delay_s / DT))
    N = max(n_long, n_ib) + 1
    sbuf = np.zeros((N, 4))
    bi = 0

    for k in range(len(pos)):
        d.qpos[QP] = pos[k]                       # drive the platform kinematically
        d.qvel[VP] = vel[k]
        ba = d.qpos[QP] + d.qpos[QA]              # body-in-space angle
        bv = d.qvel[VP] + d.qvel[VA]              # body-in-space angular velocity
        acc = [np.sin(ba) * 9.81, np.cos(ba) * 9.81]   # synthetic IMU accel
        # cf.update integrates, so it must run every step -- delay its OUTPUT,
        # never its input.
        sense = (_a2na(d.qpos[QA]),                        # bf: proprioceptive ankle angle
                 _a2na(cf.update(bv, bv, acc, acc, DT)),   # bs: graviceptive estimate
                 _t2ib(d.actuator_force[0]),               # Flx Ib tension feedback
                 _t2ib(d.actuator_force[1]))               # Ext Ib tension feedback
        sbuf[bi] = sense
        il = (bi - n_long) % N                    # long-loop read offset
        ii = (bi - n_ib) % N                      # spinal Ib read offset
        inp[0] = sbuf[il, 0]                      # bf  <- tau_d ago
        inp[1] = sbuf[il, 1]                      # bs  <- tau_d ago
        inp[2] = sbuf[ii, 2]                      # Flx Ib <- reflex latency ago
        inp[3] = sbuf[ii, 3]                      # Ext Ib <- reflex latency ago
        bi = (bi + 1) % N
        out = sns(inp)
        # SNS output -> BPA pressure -> tendon tension (N). d.actuator_length is the
        # tendon length from the previous step; the ~0.1 ms staleness matches the
        # latency the old activation path already had.
        # Mapping is DIRECT (out[0]->Flx, out[1]->Ext). The crossed mapping inherited
        # from g2_integrate.py (out[1]->Flx) is destabilising here: measured static
        # K_P = -265 N*m/rad crossed vs +139 direct. It only "worked" with the Hill
        # <muscle> because that model's ~200 N passive force stabilised the joint
        # regardless of the active drive's sign; the BPA has no passive force and
        # exposes the error.
        d.ctrl[0] = bpa.bpa_force(d.actuator_length[0], bpa.LREST, _v2p(out[0], p_max))
        d.ctrl[1] = bpa.bpa_force(d.actuator_length[1], bpa.LREST, _v2p(out[1], p_max))
        mujoco.mj_step(m, d)
        x[k] = np.degrees(d.qpos[QP])                     # stimulus: surface tilt
        y[k] = np.degrees(d.qpos[QP] + d.qpos[QA])        # response: body sway
        if not np.isfinite(y[k]):
            return None, None, None                       # blew up

    x, y = x[-plen:], y[-plen:]                   # analyse the last period only
    X, f, sig = _significant(x)
    Y = np.fft.rfft(y - y.mean())

    # Reject runs the stimulus isn't actually driving (see MIN_DRIVEN_FRAC).
    # Without this a limit cycle survives the isfinite() check above and returns
    # a fabricated resonant peak that the optimizer will happily chase.
    band = (f >= FMIN) & (f <= FMAX)
    e_tot = float(np.sum(np.abs(Y[band]) ** 2))
    if e_tot <= 0.0:
        return None, None, None                   # no response at all
    if float(np.sum(np.abs(Y[sig]) ** 2)) / e_tot < min_driven:
        return None, None, None                   # self-oscillating, not measuring

    H = Y[sig] / X[sig]                           # transfer function
    order = np.argsort(f[sig])
    freqs = f[sig][order]
    gain = np.abs(H)[order]
    phase = np.degrees(np.unwrap(np.angle(H)[order]))
    return freqs, gain, phase


if __name__ == "__main__":
    f, g, p = model_frf((4.26, 5.01, 2.48, 5.42))
    print("harmonics:", np.round(f, 3))
    print("gain:", np.round(g, 3))
    print("phase:", np.round(p, 1))
