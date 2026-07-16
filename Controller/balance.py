"""
Balance model SNS
"""

#import the necessary modules
import sns_toolbox
from sns_toolbox.neurons import NonSpikingNeuron
from sns_toolbox.networks import Network
from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.renderer import render
from sns_toolbox.color_utilities import set_text_color
from graphviz import Digraph
import numpy as np
import matplotlib.pyplot as plt


#Create class
class BalanceSNS():
    def __init__(self):
        pass

def generate_sns(gains:tuple, ctrlr_mode:int=3, analysis_outputs:bool=False, bs_wt:float=0.5,
                 ib_cm:float=4000.0, mult_kp_gs:float=2.2,
                 mult_kd_gs:float=75.0, mult_kc_gs:float=75.0,
                 t1_cm:float=0.21, t2_cm:float=8.0,
                 mult_inter_gs:float=20.0, mult_inter_erev:float=-60.1,
                 e_lo_mv:float=-60.0, e_hi_mv:float=-40.0) -> object:

    """ 
        SNS stability depends on dt, Cm, Gm, g_{max,i}, and ΔE_{s,i} according to

                0 < (dt/Cm) * I_tot < 2

        where I_tot is the total incoming current, based on the need of the post-synaptic 
        neuron to leak current. If dt is too large, the neuron doesn't have time to leak.

        ctrlr_mode describes the level of development of the controller:
            1: available inputs == kp, kd (kc fused to kd)
            2: available inputs == kp, kd, kc
            3: available inputs == kp, kd, kc, kt

        bs_wt (0 to 1) sets the sensory-reweighting split between the graviceptive (bs) and
        proprioceptive (bf) channels feeding the bilateral error neurons (03_err_CCW/15_err_CW),
        with bf_wt = 1 - bs_wt computed automatically.

        NOTE (superseded): this docstring used to state that bs_wt has NO observable effect,
        because bf and bs shared the single external input 01_bf_input. That precondition no
        longer holds - 37_bs_input is now its own external input (see add_input below), and
        pso_tuning/model_frf.py drives bf with the surface-relative ankle angle and bs with a
        complementary-filter body-in-space estimate. Those are the "genuinely distinct signals"
        the old note said were required, so bs_wt IS now behaviorally live.

        bs_wt IS AMPLITUDE-DEPENDENT - it is not a constant to be set once. It maps directly to
        Peterka's w_g, with bf_wt = w_p. For the eyes-closed support-surface condition he reports
        the weights shifting with stimulus amplitude: proprioception dominates for small tilts,
        graviception takes over for large ones. This sensory reweighting IS Peterka's central
        result, not a nuisance parameter. The 0.3 seen elsewhere in this repo is his SMALLEST
        amplitude value, so it is only correct there.

        See pso_tuning/model_frf.py: bs_wt_for(pp_deg), which interpolates w_g from the reported
        endpoints and is the single place that relationship is defined.

        Reassuringly, the network reproduces the right TREND without being asked: measured on the
        MuJoCo loop, bs_wt 0.3 -> 0.7 -> 1.0 drops the model gain span from 0.68-1.10 to
        0.34-0.70 to 0.12-0.37. Higher graviceptive weight -> lower gain is exactly the direction
        Peterka's amplitude series shows.

        ib_cm sets the membrane capacitance (nF) of the type Ib feedback neurons
        (25_kt_x_t/26_kt_x_t). With sns-toolbox's default Gm = 1 uS, tau = Cm/Gm, so Cm in nF
        IS tau in ms. This is NOT a free parameter -- it implements the low-pass on the positive
        torque feedback. See Implementation note 4 before changing it.

        Implementation notes
            The parameter values generally follow those of Hilts, 2018 Thesis with the following exceptions:
            1: The kd gain circuit was split to allow kc to be tuned independently. Since the method was to 
            simply duplicate the kd gain circuit, kc neuronal/synaptic parameter values are identical to the 
            kd gain circuit 
            2: g_s for the derivative output synapse was raised from 54.0 to 75.0 to increase the contribution,
            based on trial and error
            3: The membrane capacitance of the gain_interneuron prototype was raised to 5 nF as a workaround 
            for the problem of kp tonic inputs greater than ~9 nA diverging and crashing the simulations.
            4: The membrane capacitance of the type Ib feedback neurons (ib_cm = 4000.0 nF) was
               "lowered to 4000 nF to address the issue of the kt neuron requiring very low tonic
               inputs and being insensitive to system fluctuations".
               DO NOT reduce this toward a "biologically plausible" 5-20 ms. It is not modelling a
               membrane; it is IMPLEMENTING Peterka's low-passed positive torque feedback. Hilts
               2019 sec.III-A: "neurons behave as RC low-pass filters where the cutoff frequency is
               determined by the membrane capacitance and conductance ... It also is used to filter
               the positive torque feedback signal." His eq.6 is H_T = Kt*wc/(s + wc) with a fitted
               wc = 0.209 rad/s (tau = 4.8 s). Gm = 1 uS, so ib_cm = 4000 nF -> tau = 4.0 s ->
               wc = 0.25 rad/s. That IS the filter, to within the fit.
               Consequence, and it is intended: the Ib channel is a ~0.04 Hz path, so kt barely
               moves the response inside the 0.017-0.744 Hz PRTS band. Measured across kt = 0..10:
               max |delta gain| ~ 0.004-0.016, under 2% of a ~0.9 gain, and in some configurations
               it rounds to bit-identical. That looks like a dead parameter and is not one --
               cf. Hilts fig.7, where Kt = 0/0.06/0.1 visibly changes the response only below
               ~0.1 Hz, consistent with this corner. kt likely acts mostly as a DC operating-point
               shift, which a mean-subtracted FRF cannot see by construction.
               If you want to SEE kt work, look below 0.1 Hz or at the operating point -- not at
               the in-band FRF, and do not reach for ib_cm.
               (Two mistakes recorded so they are not repeated: a previous version of this note
               claimed 4000 nF was an ~800x-out-of-biology error that "silently disabled kt", and
               set ib_cm = 5.0 -- that deleted the H_T filter. The "disabled" evidence was a
               bit-identical FRF measured at one operating point (bs_wt=0.3); at bs_wt=0 the same
               ib_cm=4000 network shows kt live. The parameter was never dead.)
            5: The reversal potential of the mult_inter synapse prototype was raised from 60.0 to 60.1 to reflect
            an old workaround that prevented dividing by zero

    """
    kp, kd, kc, kt = gains

    # access to tonic stims is controlled by the value of ctrlr_mode
    if ctrlr_mode == 1:
        kc = kd; kt = 0
    elif ctrlr_mode == 2:
        kt = 0
    
    # Validate and compute sensory reweighting fractions
    if not (0 <= bs_wt <= 1):
        raise ValueError("bs_wt must be between 0 and 1")
    bf_wt = 1 - bs_wt

    """
    Neuron prototypes. 

    In all cases, Gm is set to the defaul value of 1 uS, based on Daun-Gruhn et al., 2009, Daun-Gruhn ,2010.
    """

    # Error (subtraction) neuron prototypes
    input_ref_neuron = NonSpikingNeuron(name = 'input_ref',     # Name displayed in a render of the network
                                    color='white',              # Fill color for the rendered neuron
                                    membrane_capacitance=20.0,  # Membrane capacitance in nF (time constant in AnimatLab)
                                    resting_potential=-50.0)    # Membrane resting potential in mV

    TH_err_neuron = NonSpikingNeuron(name = 'TH_err',           # Name displayed in a render of the network
                                    color='blue',               # Fill color for the rendered neuron
                                    resting_potential=-60.0,    # Membrane resting potential in mV
                                    membrane_capacitance=1.0)   # Membrane capacitance in nF (time constant in AnimatLab)

    # Derivative (differential) neuron prototypes
    t1_neuron = NonSpikingNeuron(name = 't1',                   # Name displayed in a render of the network
                                    color='blue',               # Fill color for the rendered neuron
                                    resting_potential=-60.0,    # Membrane resting potential in mV
                                    membrane_capacitance=t1_cm)  # nF; see t1_cm arg (was 0.21)

    t2_gt_t1_neuron = NonSpikingNeuron(name = 't2_gt_t1',       # Name displayed in a render of the network
                                    color='blue',               # Fill color for the rendered neuron
                                    resting_potential=-60.0,    # Membrane resting potential in mV
                                    membrane_capacitance=t2_cm)  # nF; see t2_cm arg (was 8.0)

    neg_dErr_dt_neuron = NonSpikingNeuron(name = 'neg_dErr_dt', # Name displayed in a render of the network
                                    color='blue',               # Fill color for the rendered neuron
                                    resting_potential=-60.0,    # Membrane resting potential in mV
                                    membrane_capacitance=1.0)   # Membrane capacitance in nF (time constant in AnimatLab)

    pos_dErr_dt_neuron = NonSpikingNeuron(name = 'pos_dErr_dt', # Name displayed in a render of the network
                                    color='blue',               # Fill color for the rendered neuron
                                    resting_potential=-60.0,    # Membrane resting potential in mV
                                    membrane_capacitance=1.0)   # Membrane capacitance in nF (time constant in AnimatLab)

    # Gain (multiplication) neuron prototypes
    gain_inter_neuron = NonSpikingNeuron(name = 'gain_inter',   # Name displayed in a render of the network
                                    color='blue',               # Fill color for the rendered neuron
                                    resting_potential=-60.0,    # Membrane resting potential in mV
                                    membrane_capacitance=5.0,#5.0 fix   # Membrane capacitance in nF (time constant in AnimatLab)
                                    bias=20.0)                  # Tonic applied current in nA

    gain_output_neuron = NonSpikingNeuron(name = 'gain_output', # Name displayed in a render of the network
                                    color='blue',               # Fill color for the rendered neuron
                                    resting_potential=-60.0,    # Membrane resting potential in mV
                                    membrane_capacitance=1.0)   # Membrane capacitance in nF (time constant in AnimatLab)

    Kp_gain_neuron = NonSpikingNeuron(name = 'Kp_gain',         # Name displayed in a render of the network
                                    color='blue',               # Fill color for the rendered neuron
                                    resting_potential=-60.0,    # Membrane resting potential in mV
                                    membrane_capacitance=1.0,   # Membrane capacitance in nF (time constant in AnimatLab)
                                    bias=kp)                  # Tonic applied current in nA

    Kd_gain_neuron = NonSpikingNeuron(name = 'Kd_gain',         # Name displayed in a render of the network
                                    color='blue',               # Fill color for the rendered neuron
                                    resting_potential=-60.0,    # Membrane resting potential in mV
                                    membrane_capacitance=1.0,   # Membrane capacitance in nF (time constant in AnimatLab)
                                    bias=kd)                  # Tonic applied current in nA
    
    Kc_gain_neuron = NonSpikingNeuron(name = 'Kc_gain',         # Name displayed in a render of the network
                                    color='blue',               # Fill color for the rendered neuron
                                    resting_potential=-60.0,    # Membrane resting potential in mV
                                    membrane_capacitance=1.0,   # Membrane capacitance in nF (time constant in AnimatLab)
                                    bias=kc)                  # Tonic applied current in nA

    Kt_gain_neuron = NonSpikingNeuron(name = 'Kt_gain',         # Name displayed in a render of the network
                                    color='blue',               # Fill color for the rendered neuron
                                    resting_potential=-60.0,    # Membrane resting potential in mV
                                    membrane_capacitance=1.0,   # Membrane capacitance in nF (time constant in AnimatLab)
                                    bias=kt)                   # Tonic applied current in nA

    # Feedback neurons
    Ib_fdbck_neuron = NonSpikingNeuron(name = 'Ib_fdbck',       # Name displayed in a render of the network
                                    color='blue',               # Fill color for the rendered neuron
                                    resting_potential=-60.0,    # Membrane resting potential in mV
                                    membrane_capacitance=ib_cm) # Membrane capacitance in nF (see ib_cm arg)

    # PD sum/output neurons
    PD_sum_neuron = NonSpikingNeuron(name = 'PD_sum',           # Name displayed in a render of the network
                                    color='blue',               # Fill color for the rendered neuron
                                    resting_potential=-60.0,    # Membrane resting potential in mV
                                    membrane_capacitance=20.0)  # Membrane capacitance in nF (time constant in AnimatLab)

    """
    Synapse Prototypes
    ==================
    """
    # Synaptic threshold (e_lo) and saturation (e_hi), in mV; the 20 mV operating range
    # the whole network is designed around (Hilts 2019 sec.III-A).
    # e_lo == E_r == -60 is a DESIGN IDENTITY, not a coincidence: a synapse driven by a
    # neuron at rest sits on its threshold and half-wave rectifies. That is exactly how
    # 06_pos_dErr_dt / 07_neg_dErr_dt split d(err)/dt into its positive and negative
    # halves, from which the controller forms d = pos - neg and |d| = pos + neg. The
    # |d| term is the co-activation pathway (McNeal & Hunt eq.1), so the rectification
    # is the architecture. Measuring the kd path against an ideal linear derivative and
    # calling the missing ~32 deg a "phase deficit" is measuring the feature.
    # Exposed as arguments for diagnostics only -- these are not tuning knobs.
    e_hi = e_hi_mv
    e_lo = e_lo_mv

    # Addition synapses 
    add_syn = NonSpikingSynapse(max_conductance = 0.115,        # Maximum synaptic conductance, g_s, in uS
                                e_lo=e_lo, e_hi=e_hi,           # Synaptic threshold and saturation in mV
                                reversal_potential = 134.0)     # Synaptic reversal potential in mV

    # Subtraction synapses
    sub_pos = NonSpikingSynapse(max_conductance = 0.115,        # Maximum synaptic conductance, g_s, in uS
                                e_lo=e_lo, e_hi=e_hi,           # Synaptic threshold and saturation in mV
                                reversal_potential = 134.0)     # Synaptic reversal potential in mV

    sub_neg = NonSpikingSynapse(max_conductance = 0.55775,      # Maximum synaptic conductance, g_s, in uS
                                e_lo=e_lo, e_hi=e_hi,           # Synaptic threshold and saturation in mV
                                reversal_potential = -100.0)    # Synaptic reversal potential in mV

    # Tunable addition/subtraction synapses for bf/bs sensory reweighting
    bf_add = NonSpikingSynapse(max_conductance = bf_wt*0.115,    # Maximum synaptic conductance, g_s, in uS
                                e_lo=e_lo, e_hi=e_hi,           # Synaptic threshold and saturation in mV
                                reversal_potential = 134.0)     # Synaptic reversal potential in mV

    bs_add = NonSpikingSynapse(max_conductance = bs_wt*0.115,    # Maximum synaptic conductance, g_s, in uS
                                e_lo=e_lo, e_hi=e_hi,           # Synaptic threshold and saturation in mV
                                reversal_potential = 134.0)     # Synaptic reversal potential in mV

    bf_sub = NonSpikingSynapse(max_conductance = bf_wt*0.55775,  # Maximum synaptic conductance, g_s, in uS
                                e_lo=e_lo, e_hi=e_hi,           # Synaptic threshold and saturation in mV
                                reversal_potential = -100.0)    # Synaptic reversal potential in mV

    bs_sub = NonSpikingSynapse(max_conductance = bs_wt*0.55775,  # Maximum synaptic conductance, g_s, in uS
                                e_lo=e_lo, e_hi=e_hi,           # Synaptic threshold and saturation in mV
                                reversal_potential = -100.0)    # Synaptic reversal potential in mV

    # Addition synapses
    kp_asyn = NonSpikingSynapse(max_conductance = 0.115,        # Maximum synaptic conductance, g_s, in uS
                                e_lo=e_lo, e_hi=e_hi,           # Synaptic threshold and saturation in mV
                                reversal_potential = 134.0)     # Synaptic reversal potential in mV
    
    kd_asyn = NonSpikingSynapse(max_conductance = 0.115,        # Maximum synaptic conductance, g_s, in uS
                                e_lo=e_lo, e_hi=e_hi,           # Synaptic threshold and saturation in mV
                                reversal_potential = 134.0)     # Synaptic reversal potential in mV
    
    # gain (multiplication) synapses
    mult_kp = NonSpikingSynapse(max_conductance = mult_kp_gs,   # g_s in uS; see mult_kp_gs arg (was 2.2)
                                e_lo=e_lo, e_hi=e_hi,           # Synaptic threshold and saturation in mV
                                reversal_potential = 134.0)     # Synaptic reversal potential in mV

    mult_kd = NonSpikingSynapse(max_conductance = mult_kd_gs,   # g_s in uS; see mult_kd_gs arg (was 75.0, orig 54.0)
                                e_lo=e_lo, e_hi=e_hi,           # Synaptic threshold and saturation in mV
                                reversal_potential = 134.0)     # Synaptic reversal potential in mV

    mult_inter = NonSpikingSynapse(max_conductance = mult_inter_gs,  # g_s in uS; see mult_inter_gs arg (was 20.0)
                                e_lo=e_lo, e_hi=e_hi,           # Synaptic threshold and saturation in mV
                                reversal_potential = mult_inter_erev)  # mV; see arg (was -60.1, see note 5)
    
    mult_kc = NonSpikingSynapse(max_conductance = mult_kc_gs,   # g_s in uS; see mult_kc_gs arg (was 75.0, orig 54.0)
                                e_lo=e_lo, e_hi=e_hi,           # Synaptic threshold and saturation in mV
                                reversal_potential = 134.0)     # Synaptic reversal potential in mV

    """
    Assemble Network
    ================
    """
    # Define the network
    net = Network(name = 'Balance Model SNS')

    # Add error calculation neurons to the network
    net.add_neuron(input_ref_neuron,    name='01_bf_input'      , color='lightblue')  # Ankle angle, Theta (proprioceptive/bf; PRTS injected here)
    net.add_neuron(input_ref_neuron,    name='02_bf_ref'        , color='lightblue')  # bf reference angle (0)
    net.add_neuron(TH_err_neuron,       name='03_err_CCW'       , color='lightblue')  # CCW error calc output
    net.add_neuron(TH_err_neuron,       name='15_err_CW'        , color='lightblue')  # CW error calc output
    net.add_neuron(TH_err_neuron,       name='33_bf_err_CCW'    , color='lightblue')  # bf (proprioceptive) channel CCW error
    net.add_neuron(TH_err_neuron,       name='34_bf_err_CW'     , color='yellow')     # bf (proprioceptive) channel CW error
    net.add_neuron(TH_err_neuron,       name='35_bs_err_CCW'    , color='lightblue')  # bs (graviceptive) channel CCW error
    net.add_neuron(TH_err_neuron,       name='36_bs_err_CW'     , color='yellow')     # bs (graviceptive) channel CW error
    net.add_neuron(input_ref_neuron,    name='37_bs_input'      , color='lightgreen') # Body-in-space angle (graviceptive; IMU-fused or sim proxy)
    net.add_neuron(input_ref_neuron,    name='38_bs_ref'        , color='lightgreen') # bs reference angle (0)

    # Add differential calculation neurons to the network
    net.add_neuron(t1_neuron,           name='04_t1'            , color='blue')       # CCW differntial term t1
    net.add_neuron(t2_gt_t1_neuron,     name='05_t2_gt_t1'      , color='blue')       # CCW differential term t2
    net.add_neuron(pos_dErr_dt_neuron,  name='06_pos_dErr_dt'   , color='blue')       # CCW differential pos val
    net.add_neuron(neg_dErr_dt_neuron,  name='07_neg_dErr_dt'   , color='blue')       # CCW differential neg val

    net.add_neuron(t1_neuron,           name='17_t1'            , color='blue')       # CW differntial term t1
    net.add_neuron(t2_gt_t1_neuron,     name='18_t2_gt_t1'      , color='blue')       # CW differntial term t2
    net.add_neuron(pos_dErr_dt_neuron,  name='19_pos_dErr_dt'   , color='blue')       # CW differential pos val
    net.add_neuron(neg_dErr_dt_neuron,  name='20_neg_dErr_dt'   , color='blue')       # CW differential neg val

    # Add Kp gain neurons to the network
    net.add_neuron(Kp_gain_neuron,      name='08_kp'            , color='green')      # Kp Gain
    net.add_neuron(gain_inter_neuron,   name='09_prop_gain'     , color='blue')       # Kp mult interneuron
    net.add_neuron(gain_output_neuron,  name='10_kp_x_err'      , color='blue')       # CCW Kp output
    net.add_neuron(gain_output_neuron,  name='16_kp_x_err'      , color='blue')       # CW Kp output

    # Add Kd gain neurons to the network
    net.add_neuron(Kd_gain_neuron,      name='11_kd'            , color='green')      # Kd Gain
    net.add_neuron(gain_inter_neuron,   name='12_deriv_gain'    , color='blue')       # Kd mult interneuron
    net.add_neuron(gain_output_neuron,  name='13_kd_x_err'      , color='blue')       # CCW Kd output
    net.add_neuron(gain_output_neuron,  name='21_kd_x_err'      , color='blue')       # CW Kd output

    # Add Kc gain neurons to the network
    net.add_neuron(Kc_gain_neuron,      name='27_kc'            , color='green')      # Kd Gain
    net.add_neuron(gain_inter_neuron,   name='28_deriv_gain'    , color='blue')       # Kd mult interneuron
    net.add_neuron(gain_output_neuron,  name='29_kc_x_err'      , color='blue')       # CCW Kd output
    net.add_neuron(gain_output_neuron,  name='30_kc_x_err'      , color='blue')       # CW Kd output

    # Add Kt gain neurons to the network
    net.add_neuron(Kt_gain_neuron,      name='23_kt'            , color='green')      # Kt Gain
    net.add_neuron(gain_inter_neuron,   name='24_int_gain'      , color='blue')       # Kt mult interneuron

    # Add Kt output/Ib feedback neurons to the network
    net.add_neuron(Ib_fdbck_neuron,     name='25_kt_x_t'        , color='blue')       # CCW Kt output
    net.add_neuron(Ib_fdbck_neuron,     name='26_kt_x_t'        , color='blue')       # CW Kt output

    # Add PD sum/model output neurons to the network
    net.add_neuron(PD_sum_neuron,       name='14_PD_output'     , color='yellow')     # CCW PD output
    net.add_neuron(PD_sum_neuron,       name='22_PD_output'     , color='yellow')     # CW PD output


    """
    Add Synapses to Network
    =======================

    Synapse naming convention: 'syn_<synapse #>_<Upre #>_<Upost #>'
    """
    # Add error calculation synapses to the network
    net.add_connection(sub_pos,     '01_bf_input'       , '33_bf_err_CCW'   , name='syn_01_01_03') #synapse 1
    net.add_connection(sub_neg,     '02_bf_ref'         , '33_bf_err_CCW'   , name='syn_02_02_03') #synapse 2
    net.add_connection(sub_neg,     '01_bf_input'       , '34_bf_err_CW'    , name='syn_18_01_15') #synapse 18
    net.add_connection(sub_pos,     '02_bf_ref'         , '34_bf_err_CW'    , name='syn_19_02_15') #synapse 19

    # Graviceptive (bs) channel error calc - reads dedicated 37_bs_input (IMU-fused or sim proxy)
    net.add_connection(sub_pos,     '37_bs_input'       , '35_bs_err_CCW'   , name='syn_50_37_35') #synapse 50
    net.add_connection(sub_neg,     '38_bs_ref'         , '35_bs_err_CCW'   , name='syn_51_38_35') #synapse 51
    net.add_connection(sub_neg,     '37_bs_input'       , '36_bs_err_CW'    , name='syn_52_37_36') #synapse 52
    net.add_connection(sub_pos,     '38_bs_ref'         , '36_bs_err_CW'    , name='syn_53_38_36') #synapse 53

    # Recombine bf/bs channel errors into the final bilateral error, weighted by bf_wt/bs_wt
    net.add_connection(bf_add,      '33_bf_err_CCW'     , '03_err_CCW'      , name='syn_60_33_03') #synapse 60
    net.add_connection(bs_add,      '35_bs_err_CCW'     , '03_err_CCW'      , name='syn_61_35_03') #synapse 61
    net.add_connection(bf_add,      '34_bf_err_CW'      , '15_err_CW'       , name='syn_62_34_15') #synapse 62
    net.add_connection(bs_add,      '36_bs_err_CW'      , '15_err_CW'       , name='syn_63_36_15') #synapse 63

    net.add_connection(bf_sub,      '33_bf_err_CCW'     , '15_err_CW'       , name='syn_64_33_15') #synapse 64
    net.add_connection(bs_sub,      '35_bs_err_CCW'     , '15_err_CW'       , name='syn_65_35_15') #synapse 65
    net.add_connection(bf_sub,      '34_bf_err_CW'      , '03_err_CCW'      , name='syn_66_34_03') #synapse 66
    net.add_connection(bs_sub,      '36_bs_err_CW'      , '03_err_CCW'      , name='syn_67_36_03') #synapse 67

    # Add differential calculation routing synapses to the network
    net.add_connection(add_syn,     '03_err_CCW'        , '04_t1'           , name='syn_03_03_04') #synapse 3
    net.add_connection(add_syn,     '03_err_CCW'        , '05_t2_gt_t1'     , name='syn_04_03_05') #synapse 4
    net.add_connection(add_syn,     '15_err_CW'         , '17_t1'           , name='syn_21_15_17') #synapse 21
    net.add_connection(add_syn,     '15_err_CW'         , '18_t2_gt_t1'     , name='syn_20_15_18') #synapse 20

    # Add Kp gain routing synapses to the network
    net.add_connection(mult_kp,     '03_err_CCW'        , '10_kp_x_err'     , name='syn_05_03_10') #synapse 5
    net.add_connection(mult_kp,     '15_err_CW'         , '16_kp_x_err'     , name='syn_22_15_16') #synapse 22

    # Add CCW differential calculation synapses
    net.add_connection(sub_pos,     '04_t1'             , '07_neg_dErr_dt'  , name='syn_06_04_07') #synapse 6
    net.add_connection(sub_neg,     '04_t1'             , '06_pos_dErr_dt'  , name='syn_07_04_06') #synapse 7
    net.add_connection(sub_neg,     '05_t2_gt_t1'       , '07_neg_dErr_dt'  , name='syn_08_05_07') #synapse 8
    net.add_connection(sub_pos,     '05_t2_gt_t1'       , '06_pos_dErr_dt'  , name='syn_09_05_06') #synapse 9

    # Add CW differential calculation synapses
    net.add_connection(sub_neg,     '17_t1'             , '19_pos_dErr_dt'  , name='syn_26_17_19') #synapse 26
    net.add_connection(sub_pos,     '17_t1'             , '20_neg_dErr_dt'  , name='syn_25_17_20') #synapse 25
    net.add_connection(sub_pos,     '18_t2_gt_t1'       , '19_pos_dErr_dt'  , name='syn_24_18_19') #synapse 24
    net.add_connection(sub_neg,     '18_t2_gt_t1'       , '20_neg_dErr_dt'  , name='syn_23_18_20') #synapse 23

    # Add Kp gain multiplication interneuronal synapses to the network
    net.add_connection(mult_inter,  '08_kp'             , '09_prop_gain'    , name='syn_16_08_09') #synapse 16
    net.add_connection(mult_inter,  '09_prop_gain'      , '10_kp_x_err'     , name='syn_17_09_10') #synapse 17
    net.add_connection(mult_inter,  '09_prop_gain'      , '16_kp_x_err'     , name='syn_32_09_16') #synapse 32

    # Add Kd gain multiplication interneuronal synapses to the network
    net.add_connection(mult_inter,  '11_kd'             , '12_deriv_gain'   , name='syn_14_11_12') #synapse 14
    net.add_connection(mult_inter,  '12_deriv_gain'     , '13_kd_x_err'     , name='syn_15_12_13') #synapse 15
    net.add_connection(mult_inter,  '12_deriv_gain'     , '21_kd_x_err'     , name='syn_31_12_21') #synapse 31

    # Add Kc gain multiplication interneuronal synapses to the network
    net.add_connection(mult_inter,  '27_kc'             , '28_deriv_gain'   , name='syn_42_27_28') #synapse 42
    net.add_connection(mult_inter,  '28_deriv_gain'     , '29_kc_x_err'     , name='syn_43_28_29') #synapse 43
    net.add_connection(mult_inter,  '28_deriv_gain'     , '30_kc_x_err'     , name='syn_44_28_30') #synapse 44

    # Add Kd/Kc gain routing (ie. derivative calculation output) synapses to the network
    net.add_connection(mult_kc,    '06_pos_dErr_dt'    , '29_kc_x_err'     , name='syn_10_06_21') #synapse 10
    net.add_connection(mult_kd,    '07_neg_dErr_dt'    , '13_kd_x_err'     , name='syn_11_07_13') #synapse 11
    net.add_connection(mult_kd,    '20_neg_dErr_dt'    , '21_kd_x_err'     , name='syn_27_20_21') #synapse 27
    net.add_connection(mult_kc,    '19_pos_dErr_dt'    , '30_kc_x_err'     , name='syn_28_19_13') #synapse 28

    # Add output gain sum synapses to the network
    net.add_connection(kp_asyn,     '10_kp_x_err'       , '14_PD_output'    , name='syn_12_10_14') #synapse 12
    net.add_connection(kd_asyn,     '13_kd_x_err'       , '14_PD_output'    , name='syn_13_13_14') #synapse 13
    net.add_connection(add_syn,     '29_kc_x_err'       , '22_PD_output'    , name='syn_45_29_14') #synapse 45
    net.add_connection(add_syn,     '30_kc_x_err'       , '14_PD_output'    , name='syn_46_30_22') #synapse 46
    net.add_connection(kp_asyn,     '16_kp_x_err'       , '22_PD_output'    , name='syn_29_16_22') #synapse 29
    net.add_connection(kd_asyn,     '21_kd_x_err'       , '22_PD_output'    , name='syn_30_21_22') #synapse 30

    # Add Kt gain multiplication interneuronal synapses to the network
    net.add_connection(mult_inter,  '23_kt'             , '24_int_gain'     , name='syn_33_23_24') #synapse 33
    net.add_connection(mult_inter,  '24_int_gain'       , '25_kt_x_t'       , name='syn_34_24_25') #synapse 34
    net.add_connection(mult_inter,  '24_int_gain'       , '26_kt_x_t'       , name='syn_38_24_26') #synapse 38

    # Add Kt gain subraction calculation synapses (Ask Alex about this circuit)
    # net.add_connection(sub_pos,     '14_PD_output'      , '25_kt_x_t'       , name='syn_35_14_25') #synapse 35
    net.add_connection(sub_neg,     '22_PD_output'      , '25_kt_x_t'       , name='syn_36_22_25') #synapse 36
    net.add_connection(sub_neg,     '14_PD_output'      , '26_kt_x_t'       , name='syn_39_14_26') #synapse 39
    # net.add_connection(sub_pos,     '22_PD_output'      , '26_kt_x_t'       , name='syn_40_22_26') #synapse 40

    # Add Ib feedback synapses
    net.add_connection(add_syn,     '25_kt_x_t'         , '03_err_CCW'     , name='syn_37_25_03') #synapse 37
    net.add_connection(add_syn,     '26_kt_x_t'         , '15_err_CW'       , name='syn_41_26_15') #synapse 41
    
 
    """
    Inputs and Outputs
    ==================
    """
    # Add ankle angle input to the network. Note: adapters are external to this SNS model
    net.add_input(dest="01_bf_input")

    # Add graviceptive (bs) input — body-in-space angle from IMU sensor fusion or sim proxy
    net.add_input(dest='37_bs_input', name='BS_angle', color='lightgreen')

    # Add Ib Feedback inputs to the network. Note: adapters are external to this SNS model
    net.add_input(dest='25_kt_x_t', name='Flx_Ib', color='white')
    net.add_input(dest='26_kt_x_t', name='Ext_Ib', color='white')

    # Add muscle actuation outputs
    net.add_output('14_PD_output')
    net.add_output('22_PD_output')

    # Analysis mode records key membrane conductance values that correspond to neurons that 
    # represent the output of a calculation, giving a baked-in way to gain better data 
    # visibility of a run.  
    if analysis_outputs:
        # Add data analysis outputs
        net.add_output('25_kt_x_t')
        net.add_output('26_kt_x_t')

        # Add signal tracking outputs
        net.add_output('10_kp_x_err')
        net.add_output('13_kd_x_err')
        net.add_output('29_kc_x_err')#'30_kc_x_err'

        net.add_output('01_bf_input')

        # Add bf/bs sensory reweighting tracking outputs
        net.add_output('33_bf_err_CCW')
        net.add_output('34_bf_err_CW')
        net.add_output('35_bs_err_CCW')
        net.add_output('36_bs_err_CW')
        net.add_output('03_err_CCW')   # combined weighted CCW error neuron
        net.add_output('15_err_CW')    # combined weighted CW error neuron

        # Derivative-circuit taps. Appended last so the indices above stay stable.
        # Needed to measure the differentiator itself: without these you can only
        # see 13_kd_x_err, which is AFTER the multiplication, and any phase loss
        # inside t1/t2 or the pos/neg split is invisible.
        net.add_output('04_t1')
        net.add_output('05_t2_gt_t1')
        net.add_output('06_pos_dErr_dt')
        net.add_output('07_neg_dErr_dt')
    # net.add_output('02_bf_ref')

    # Add outputs for measuring Kp vs Kd
    # net.add_output('10_kp_x_err')
    # net.add_output('13_kd_x_err')
    # render(net)

    return net

def renderSNS(net):
    clusters = {
        'Sensory Reweighting': ['01_bf_input', '02_bf_ref',
                                '33_bf_err_CCW', '34_bf_err_CW',
                                '35_bs_err_CCW', '36_bs_err_CW'],
        'Combined Error':      ['03_err_CCW', '15_err_CW'],
        'CCW Derivative':      ['04_t1', '05_t2_gt_t1',
                                '06_pos_dErr_dt', '07_neg_dErr_dt'],
        'CW Derivative':       ['17_t1', '18_t2_gt_t1',
                                '19_pos_dErr_dt', '20_neg_dErr_dt'],
        'Kp Gain':             ['08_kp', '09_prop_gain',
                                '10_kp_x_err', '16_kp_x_err'],
        'Kd Gain':             ['11_kd', '12_deriv_gain',
                                '13_kd_x_err', '21_kd_x_err'],
        'Kc Gain':             ['27_kc', '28_deriv_gain',
                                '29_kc_x_err', '30_kc_x_err'],
        'Kt / Ib Feedback':    ['23_kt', '24_int_gain',
                                '25_kt_x_t', '26_kt_x_t'],
        'Output':              ['14_PD_output', '22_PD_output'],
    }

    name_to_idx = {pop['name']: str(i) for i, pop in enumerate(net.populations)}

    graph = Digraph(engine='dot')
    graph.attr(rankdir='LR', ranksep='1.5', nodesep='0.8',
               splines='curved', size='40,24!', overlap='false')
    graph.attr('node', fontsize='9')
    graph.attr('edge', fontsize='7')

    clustered = {n for names in clusters.values() for n in names}

    for cluster_name, neuron_names in clusters.items():
        with graph.subgraph(name='cluster_' + cluster_name) as c:
            c.attr(label=cluster_name, style='rounded,dashed', color='gray')
            for nname in neuron_names:
                if nname in name_to_idx:
                    i = int(name_to_idx[nname])
                    pop = net.populations[i]
                    color_cell = pop['color']
                    color_font = set_text_color(color_cell)
                    shape = 'ellipse' if pop['number'] == 1 else 'rect'
                    style = 'filled' if pop['number'] == 1 else 'filled,rounded'
                    c.node(name_to_idx[nname], label=nname, style=style,
                           shape=shape, fillcolor=color_cell, fontcolor=color_font)

    for i, pop in enumerate(net.populations):
        if pop['name'] not in clustered:
            color_cell = pop['color']
            color_font = set_text_color(color_cell)
            shape = 'ellipse' if pop['number'] == 1 else 'rect'
            style = 'filled' if pop['number'] == 1 else 'filled,rounded'
            graph.node(str(i), label=pop['name'], style=style, shape=shape,
                       fillcolor=color_cell, fontcolor=color_font)

    for i, inp in enumerate(net.inputs):
        color_cell = inp['color']
        color_font = set_text_color(color_cell)
        graph.node('In'+str(i), label=inp['name'], style='filled',
                   shape='invhouse', fillcolor=color_cell, fontcolor=color_font)
        graph.edge('In'+str(i), str(inp['destination']))

    for i, out in enumerate(net.outputs):
        color_cell = out['color']
        color_font = set_text_color(color_cell)
        graph.node('Out'+str(i), label=out['name'], style='filled',
                   shape='house', fillcolor=color_cell, fontcolor=color_font)
        graph.edge(str(out['source']), 'Out'+str(i))

    for conn in net.connections:
        src = str(conn['source'])
        dst = str(conn['destination'])
        params = conn['params']
        label = conn['name']
        if params['reversal_potential'] > 0:
            arrowhead = 'invempty'
        elif params['reversal_potential'] < 0:
            arrowhead = 'dot'
        else:
            arrowhead = 'odot'
        graph.edge(src, dst, dir='forward', arrowhead=arrowhead,
                   arrowtail=arrowhead, label=label)

    graph.format = 'png'
    graph.render(filename=net.params['name'], view=True, cleanup=True)

if __name__ == "__main__":

    #build the network (gains: kp, kd, kc, kt)
    net = generate_sns(gains=(4.26, 5.01, 2.48, 5.42))

    # Show the network (This technique bypasses issues with using render after importing the toolbox elsewhere.)
    renderSNS(net)

    # # Set simulation parameters
    # dt = 0.1 # Simulation timestep in ms
    # t_max = 300 # Max simulation time in ms

    # #compile the network
    # model = net.compile(backend='numpy', dt=dt, debug=False)

    # # Initialize a vector of timesteps
    # t = np.arange(0, t_max, dt)
    # num_steps = len(t)

    # # Initialize vectors which store the input to our network, and for data to be written to during simulation from outputs
    # inputs = np.ones([1,3])  # Input vector must be 2d, even if second dimension is 1
    # inputs[0][0] = -10 #[nA]
    # data = np.zeros([num_steps,net.get_num_outputs()])
    
    # for i in range(num_steps):
    #     #step the model forward one step
    #     data[i,:] = model(inputs[0,:])

    # data = data.transpose()

    # plt.figure()
    # plt.title('Motor Neurons')
    # plt.plot(t,data[:][0], label='Extensor')
    # plt.plot(t,data[:][1], label='Flexor')
    # plt.legend()

    # plt.figure()
    # plt.title('Input Neuron')
    # plt.plot(t,data[:][2], label='01_input')
    # # plt.plot(t,data[:][3], label='03_err')
    # # plt.plot(t,data[:][4], label='10_kp_gain')
    # plt.legend()

    # plt.show()

    

