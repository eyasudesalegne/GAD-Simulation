import numpy as np

class PatientProfile:
    """
    Holds patient clinical profile parameters relevant to GAD severity.
    """
    def __init__(self, anxiety_score, stress_level, cortisol, hr_variability, sleep_quality, demographics=None):
        self.anxiety_score = np.clip(anxiety_score, 0, 21)
        self.stress_level = np.clip(stress_level, 0, 100)
        self.cortisol = np.clip(cortisol, 5, 25)
        self.hr_variability = np.clip(hr_variability, 0, 100)
        self.sleep_quality = np.clip(sleep_quality, 0, 10)
        self.demographics = demographics or {}

def calculate_gad_severity(patient_profile):
    """
    Compute a severity score [0,1] for Generalized Anxiety Disorder from clinical inputs.
    Weighted combination of normalized clinical scores.
    """
    norm_anxiety = patient_profile.anxiety_score / 21.0
    norm_stress = patient_profile.stress_level / 100.0
    norm_cortisol = (patient_profile.cortisol - 5) / (25 - 5)
    norm_hrv = 1 - (patient_profile.hr_variability / 100.0)
    norm_sleep = 1 - (patient_profile.sleep_quality / 10.0)

    severity = (0.4 * norm_anxiety +
                0.2 * norm_stress +
                0.15 * norm_cortisol +
                0.15 * norm_hrv +
                0.1 * norm_sleep)
    return np.clip(severity, 0, 1)

def set_simulation_parameters(sim, severity):
    """
    Adjust simulation baseline parameters according to GAD severity.
    Neuromodulator baseline levels and synaptic scaling adjusted.
    """
    if not hasattr(sim, '_baseline_parameters'):
        sim._baseline_parameters = {}

    if hasattr(sim, 'neuromodulators'):
        nm_state = sim.neuromodulators.concentrations
        if 'DA' not in sim._baseline_parameters:
            sim._baseline_parameters['DA'] = nm_state['DA'][0]
        if '5-HT' not in sim._baseline_parameters:
            sim._baseline_parameters['5-HT'] = nm_state['5-HT'][0]
        if 'NE' not in sim._baseline_parameters:
            sim._baseline_parameters['NE'] = nm_state['NE'][0]
        if 'ACh' not in sim._baseline_parameters:
            sim._baseline_parameters['ACh'] = nm_state['ACh'][0]
        if 'Cort' not in sim._baseline_parameters:
            sim._baseline_parameters['Cort'] = nm_state['Cort'][0]

        baseline_da = sim._baseline_parameters['DA']
        baseline_5ht = sim._baseline_parameters['5-HT']
        baseline_ne = sim._baseline_parameters['NE']
        baseline_ach = sim._baseline_parameters['ACh']
        baseline_cort = sim._baseline_parameters['Cort']

        nm_state['DA'][:] = baseline_da - 0.1 * severity
        nm_state['5-HT'][:] = baseline_5ht - 0.1 * severity
        nm_state['NE'][:] = baseline_ne + 0.2 * severity
        nm_state['ACh'][:] = baseline_ach
        nm_state['Cort'][:] = baseline_cort + 0.3 * severity

    if not hasattr(sim, 'synaptic_scaling'):
        sim.synaptic_scaling = {}

    if 'A_plus' not in sim.synaptic_scaling:
        sim.synaptic_scaling['A_plus'] = 0.01
    if 'A_minus' not in sim.synaptic_scaling:
        sim.synaptic_scaling['A_minus'] = 0.012

    sim.synaptic_scaling['A_plus'] = 0.01 * (1 + 0.5 * severity)
    sim.synaptic_scaling['A_minus'] = 0.012 * (1 + 0.5 * severity)

    for region in sim.regions.values():
        if not hasattr(region, 'base_noise_amp'):
            region.base_noise_amp = [pop.noise_amp for pop in region.populations]
        factor = 1 + 0.2 * severity
        for idx, pop in enumerate(region.populations):
            pop.noise_amp = region.base_noise_amp[idx] * factor

    if not hasattr(sim, 'p_base_default'):
        sim.p_base_default = 0.3

    sim.p_base = sim.p_base_default * (1 + 0.3 * severity)
