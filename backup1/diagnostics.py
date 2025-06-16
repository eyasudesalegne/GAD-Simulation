import numpy as np

class PatientProfile:
    """
    Encapsulates patient diagnostic data.
    """
    def __init__(self, anxiety_score, stress_level, cortisol, hr_variability, sleep_quality, demographics=None):
        self.anxiety_score = anxiety_score
        self.stress_level = stress_level
        self.cortisol = cortisol
        self.hr_variability = hr_variability
        self.sleep_quality = sleep_quality
        self.demographics = demographics or {}

def calculate_gad_severity(patient_profile):
    """
    Compute normalized composite severity score between 0 and 1 based on clinical inputs.
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
    Safely update simulation parameters in response to GAD severity.
    Uses stored baseline parameters to avoid cumulative scaling.
    """
    # Update neuromodulator baselines
    if hasattr(sim, 'neuromodulators'):
        neuromod_state = sim.neuromodulators.state
        neuromod_state['DA'] = 0.2 - 0.1 * severity
        neuromod_state['5-HT'] = 0.2 - 0.1 * severity
        neuromod_state['NE'] = 0.2 + 0.2 * severity
        neuromod_state['ACh'] = 0.2
        neuromod_state['Cort'] = 0.2 + 0.3 * severity

    # Adjust global synaptic plasticity scaling
    if hasattr(sim, 'synaptic_scaling'):
        sim.synaptic_scaling['A_plus'] = 0.01 * (1 + 0.5 * severity)
        sim.synaptic_scaling['A_minus'] = 0.012 * (1 + 0.5 * severity)

    # Adjust noise amplitude per region using baseline
    for region in sim.regions.values():
        if not hasattr(region, 'base_noise_amp'):
            region.base_noise_amp = region.noise_amp  # store baseline on first call
        factor = 1 + 0.2 * severity
        region.noise_amp = region.base_noise_amp * factor

    # Adjust connection probability safely using baseline
    if hasattr(sim, 'p_base_default'):
        sim.p_base = sim.p_base_default * (1 + 0.3 * severity)
    else:
        # fallback: set and store default baseline if missing
        if not hasattr(sim, 'p_base_default'):
            sim.p_base_default = 0.3
        sim.p_base = sim.p_base_default * (1 + 0.3 * severity)

    print(f"Simulation parameters updated for GAD severity: {severity:.2f}")
