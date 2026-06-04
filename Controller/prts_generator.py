import numpy as np

def generate_prts(v_amp, dt):
    """
    Generate a Pseudo-Random Ternary Sequence (PRTS) for postural control experiments.

    Parameters:
    v_amp (float): Amplitude of the PRTS signal (degrees).
    dt (float): Time step for the PRTS signal (seconds).

"""
    # PRTS parameters
    clock_dt = 0.25 # time step for the PRTS signal (seconds)
    n_states =3**5-1 # number of states in the PRTS sequence (3^n - 1, where n is the number of bits)
    reg = [2, 0, 2, 0, 2] # initial state from peterka 2002 fig 2A
    seq= []
    for i in range(n_states):
        
        # update the register using the feedback taps (2, 0, 2, 0, 2)
        new_bit = (reg[2] + reg[3] + reg[4]) % 3 # sum the tapped bits and take mod 3
        seq.append(new_bit) # output is the last bit of the register
        reg = [new_bit] + reg[:-1] # shift the register and add the new bit at the front

    vel_seq = [0 if x == 0 else (v_amp if x == 1 else -v_amp) for x in seq]
    samples_per_clock = int(clock_dt / dt)
    vel_expanded = np.repeat(vel_seq, samples_per_clock)
    pos = np.cumsum(vel_expanded) * dt
    t   = np.arange(len(pos)) * dt
    return t, pos