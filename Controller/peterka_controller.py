import numpy as np
import matplotlib.pyplot as plt
from sns_neurons import Neuron, Synapse
from sns_subnetworks import (make_addition_network,make_subtraction_network,make_division_network,make_multiplication_network,make_derivative_network)
from prts_generator import generate_prts

# Plant Parameters (McNeal 2026 table I / Peterka 2002)
J = 63.0 # moment of inertia (kg*m^2)
m = 77.8 # mass (kg)
h = 0.9 # height of COM (m)
g = 9.81 # gravity (m/s^2)
mgh = m * g * h # destabilizing torque per unit angle (N*m/rad)
b = 351.0 # damping coefficient (N*m*s/rad)

# Controller gains (starting point from McNeal 2026 table IV, TC4)
Kp = 1100.0 # proportional gain (N*m/rad)
Kd = 270.0 # derivative gain (N*m*s/rad)

# Sensory weights (Peterka 2002 TC4: eyes closed, surface tilt)
wp = 0.70 # weight on proprioceptive weight
wg = 0.30 # weight on graviceptive weight

# Time Parameters
tau_d = 0.090 # sensorimotor delay (s)
dt = 2e-4 # time step (s)
T = 60.5 # one PRTS cycle (Peterka 2002)

# Time vector
t, FS = generate_prts(v_amp=1.0, dt=dt) #v_amp is the amplitude of the PRTS signal in degrees/s
steps = len(t) # number of time steps in the simulation

#Storage arrays
BS_log = np.zeros(steps) # body sway angle (rad)
Ta_log = np.zeros(steps) # active torque (N*m)
FS_log = np.zeros(steps) # sensory stimulus reference

# Control loop: e = wp*(BS-FS) + wg*BS,  Ta = Kp*e + Kd*de/dt
sub_net  = make_subtraction_network()    # BF = BS - FS
mul_wp   = make_multiplication_network() # wp * BF
mul_wg   = make_multiplication_network() # wg * BS
add_err  = make_addition_network()       # e = wp*BF + wg*BS
mul_kp   = make_multiplication_network() # Kp * e
deriv    = make_derivative_network()     # de/dt
mul_kd   = make_multiplication_network() # Kd * de/dt
add_ta   = make_addition_network()       # Ta = Kp*e + Kd*de/dt

# Time delay buffer

k_delay = round(tau_d /dt) # number of time steps corresponding to the sensorimotor delay
BS_delay = np.zeros(k_delay) # buffer to implement time delay for body sway angle