# brain_sim/treatments.py

def set_gad_severity(sim, severity):
    """
    Modulates network parameters to simulate different levels of Generalized Anxiety Disorder (GAD).
    """
    severity = max(0, min(severity, 1))
    for region_name in ['BLA', 'LA', 'CeA']:
        if region_name in sim.regions:
            r = sim.regions[region_name]
            r.a *= (1 + 0.5 * severity)
            r.d += 1e-3 * severity
    for region_name in ['vmPFC', 'mPFC', 'dACC', 'rACC']:
        if region_name in sim.regions:
            sim.regions[region_name].b *= (1 - 0.3 * severity)
    sim.neuromodulators.state['Cort'] += 0.5 * severity
    sim.neuromodulators.state['5-HT'] -= 0.3 * severity
    sim.neuromodulators.state['DA'] -= 0.2 * severity
    print(f"GAD severity applied: {severity:.2f}")


def apply_treatment(sim, treatment_name, intensity, dosage=0, frequency=0):
    intensity = max(0, min(intensity, 1))
    treatments = {
        'SSRI': apply_ssri,
        'SNRI': apply_snri,
        'Benzodiazepine': apply_benzo,
        'CBT': apply_cbt,
        'Exposure': apply_exposure,
        'rTMS': apply_rtms,
        'Mindfulness': apply_mindfulness,
        'SleepTherapy': apply_sleep_therapy
    }
    func = treatments.get(treatment_name)
    if func:
        func(sim, intensity, dosage, frequency)
    else:
        print(f"Unknown treatment: {treatment_name}")


def apply_ssri(sim, intensity, dosage=20, frequency=1):
    rec_dose, rec_freq = 20, 1
    scale = intensity * (dosage/rec_dose) * (frequency/rec_freq)
    sim.neuromodulators.state['5-HT'] += 0.5 * scale
    for r in ['BLA', 'LA', 'CeA']:
        if r in sim.regions:
            sim.regions[r].a *= (1 - 0.3 * scale)
    print(f"SSRI: {dosage}mg × {frequency}/week, intensity={intensity}, scale={scale:.2f}")


def apply_snri(sim, intensity, dosage=75, frequency=1):
    rec_dose, rec_freq = 75, 1
    scale = intensity * (dosage/rec_dose) * (frequency/rec_freq)
    sim.neuromodulators.state['5-HT'] += 0.3 * scale
    sim.neuromodulators.state['NE'] += 0.3 * scale
    print(f"SNRI: {dosage}mg × {frequency}/week, intensity={intensity}, scale={scale:.2f}")


def apply_benzo(sim, intensity, dosage=0.5, frequency=2):
    rec_dose, rec_freq = 0.5, 2
    scale = intensity * (dosage/rec_dose) * (frequency/rec_freq)
    for region in sim.regions.values():
        region.b *= (1 + 0.5 * scale)
    sim.neuromodulators.state['GABA'] += 0.5 * scale  # GABA modulation added
    print(f"Benzodiazepine: {dosage}mg × {frequency}/week, intensity={intensity}, scale={scale:.2f}")


def apply_cbt(sim, intensity, dosage=8, frequency=1):
    rec_dose, rec_freq = 8, 1
    scale = intensity * (dosage/rec_dose) * (frequency/rec_freq)
    for r in ['vmPFC', 'mPFC', 'dACC', 'rACC']:
        if r in sim.regions:
            sim.regions[r].b *= (1 + 0.4 * scale)
    print(f"CBT: {dosage} sessions, intensity={intensity}, scale={scale:.2f}")


def apply_exposure(sim, intensity, dosage=8, frequency=1):
    rec_dose, rec_freq = 8, 1
    scale = intensity * (dosage/rec_dose) * (frequency/rec_freq)
    print(f"Exposure Therapy: {dosage} sessions, intensity={intensity}, scale={scale:.2f}")


def apply_rtms(sim, intensity, dosage=3000, frequency=5):
    rec_dose, rec_freq = 3000, 5
    scale = intensity * (dosage/rec_dose) * (frequency/rec_freq)
    for r in ['DLPFC', 'vmPFC', 'mPFC']:
        if r in sim.regions:
            sim.regions[r].a *= (1 - 0.3 * scale)
    print(f"rTMS: {dosage} pulses/day × {frequency}/week, intensity={intensity}, scale={scale:.2f}")


def apply_mindfulness(sim, intensity, dosage=20, frequency=7):
    rec_dose, rec_freq = 20, 7
    scale = intensity * (dosage/rec_dose) * (frequency/rec_freq)
    for r in ['BLA', 'LA', 'CeA']:
        if r in sim.regions:
            sim.regions[r].a *= (1 - 0.2 * scale)
    sim.neuromodulators.state['GABA'] += 0.2 * scale  # GABA modulation added
    print(f"Mindfulness: {dosage} min/day × {frequency}/week, intensity={intensity}, scale={scale:.2f}")


def apply_sleep_therapy(sim, intensity, dosage=7, frequency=7):
    rec_dose, rec_freq = 7, 7
    scale = intensity * (dosage/rec_dose) * (frequency/rec_freq)
    sim.neuromodulators.state['Cort'] -= 0.3 * scale
    sim.neuromodulators.state['5-HT'] += 0.2 * scale
    sim.neuromodulators.state['GABA'] += 0.1 * scale  # GABA modulation added
    print(f"Sleep Therapy: {dosage} hrs/night × {frequency}/week, intensity={intensity}, scale={scale:.2f}")
