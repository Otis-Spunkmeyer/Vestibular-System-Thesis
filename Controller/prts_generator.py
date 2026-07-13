import numpy as np

def generate_prts(v_amp, dt, shift=7):
    """
    Generate a Pseudo-Random Ternary Sequence (PRTS) for postural control experiments.

    Parameters:
    v_amp (float): velocity amplitude in rad/s for non-zero clock steps
    dt    (float): simulation time step in seconds
    shift (int)  : phase offset — cycles the 242-element sequence to select starting point
    """
    clock_dt = 0.25
    n_states = 3**5 - 1                    # 242 clock periods × 0.25 s = 60.5 s

    # 5-stage GF(3) shift register (Peterka 2002 initial state)
    reg = np.array([2, 0, 2, 0, 2], dtype=np.int8)
    out = np.empty(n_states, dtype=np.int8)

    for n in range(n_states):
        m       = (int(reg[2]) - int(reg[3]) - int(reg[4])) % 3
        out[n]  = m
        reg[1:] = reg[:4].copy()           # shift right: oldest falls off
        reg[0]  = m

    # Map {0, 1, 2} → {0, +v_amp, -v_amp}
    vel_seq = np.where(out == 2, -1, out.astype(np.float64)) * v_amp
    vel_seq = np.roll(vel_seq, shift)      # select starting phase within the 242-element cycle

    samples_per_clock = int(clock_dt / dt)
    vel_expanded      = np.repeat(vel_seq, samples_per_clock)
    pos = np.cumsum(vel_expanded) * dt
    t   = np.arange(len(pos)) * dt
    return t, pos
