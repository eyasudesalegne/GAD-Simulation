import numpy as np
from brain_sim.logger import logger

class Treatment:
    """
    Base class for treatments affecting neuromodulators and neuronal parameters.
    """

    def __init__(self, name, sim, intensity=1.0, dosage=1.0, frequency=1.0):
        self.name = name
        self.sim = sim
        self.intensity = intensity
        self.dosage = dosage
        self.frequency = frequency

        self.concentration = 0.0
        self.absorption_rate = 0.1
        self.metabolism_rate = 0.05
        self.elimination_rate = 0.03

        self.Emax = 1.0
        self.EC50 = 0.5
        self.hill_coefficient = 2.0

        self.tolerance = 0.0
        self.tolerance_rate = 0.001
        self.sensitization = 0.0
        self.sensitization_rate = 0.0005

        self.adherence = 1.0

        self.last_dose_time = -np.inf

    def dose(self, current_time):
        # frequency is interpreted in Hz (per second)
        if current_time - self.last_dose_time >= 1.0 / self.frequency:
            self.concentration += self.dosage * self.adherence * self.intensity
            self.last_dose_time = current_time
            logger.info(f"{self.name} dosed at time {current_time:.2f}, concentration: {self.concentration:.3f}")

    def update_pk(self, dt):
        absorption = self.absorption_rate * (1 - self.concentration)
        metabolism = self.metabolism_rate * self.concentration
        elimination = self.elimination_rate * self.concentration

        d_conc = absorption - metabolism - elimination
        self.concentration = max(0.0, self.concentration + d_conc * dt)

    def compute_effect(self):
        effective_conc = self.concentration * (1 - self.tolerance + self.sensitization)
        denom = self.EC50 ** self.hill_coefficient + effective_conc ** self.hill_coefficient + 1e-12
        response = self.Emax * (effective_conc ** self.hill_coefficient) / denom
        return response

    def update_tolerance(self, dt):
        self.tolerance += self.tolerance_rate * self.concentration * dt
        self.tolerance = min(self.tolerance, 0.9)
        self.sensitization += self.sensitization_rate * (1 - self.tolerance) * dt
        self.sensitization = min(self.sensitization, 0.5)

    def apply(self, dt, current_time):
        self.dose(current_time)
        self.update_pk(dt)
        effect = self.compute_effect()
        self.update_tolerance(dt)
        return effect

# Specific treatment subclasses modifying simulation parameters

class SSRI(Treatment):
    def apply(self, dt, current_time):
        effect = super().apply(dt, current_time)
        sim = self.sim
        if hasattr(sim, 'neuromodulators'):
            sim.neuromodulators.concentrations['5-HT'] += 0.5 * effect
        for r in ['BLA_E', 'LA_E', 'CeA_I']:
            if r in sim.regions:
                for pop in sim.regions[r].populations:
                    pop.a *= (1 - 0.3 * effect)
        return effect

class SNRI(Treatment):
    def apply(self, dt, current_time):
        effect = super().apply(dt, current_time)
        sim = self.sim
        if hasattr(sim, 'neuromodulators'):
            sim.neuromodulators.concentrations['5-HT'] += 0.3 * effect
            sim.neuromodulators.concentrations['NE'] += 0.3 * effect
        return effect

class Benzodiazepine(Treatment):
    def apply(self, dt, current_time):
        effect = super().apply(dt, current_time)
        sim = self.sim
        for region in sim.regions.values():
            for pop in region.populations:
                pop.b *= (1 + 0.5 * effect)
        if hasattr(sim, 'neuromodulators'):
            sim.neuromodulators.concentrations['GABA'] += 0.5 * effect
        return effect

class CBT(Treatment):
    def apply(self, dt, current_time):
        effect = super().apply(dt, current_time)
        sim = self.sim
        for r in ['vmPFC_E', 'mPFC_E', 'dACC_E', 'rACC_E']:
            if r in sim.regions:
                for pop in sim.regions[r].populations:
                    pop.b *= (1 + 0.4 * effect)
        return effect

class Exposure(Treatment):
    def apply(self, dt, current_time):
        effect = super().apply(dt, current_time)
        # No direct parameter change defined here
        return effect

class rTMS(Treatment):
    def apply(self, dt, current_time):
        effect = super().apply(dt, current_time)
        sim = self.sim
        for r in ['DLPFC_E', 'vmPFC_E', 'mPFC_E']:
            if r in sim.regions:
                for pop in sim.regions[r].populations:
                    pop.a *= (1 - 0.3 * effect)
        return effect

class Mindfulness(Treatment):
    def apply(self, dt, current_time):
        effect = super().apply(dt, current_time)
        sim = self.sim
        for r in ['BLA_E', 'LA_E', 'CeA_I']:
            if r in sim.regions:
                for pop in sim.regions[r].populations:
                    pop.a *= (1 - 0.2 * effect)
        if hasattr(sim, 'neuromodulators'):
            sim.neuromodulators.concentrations['GABA'] += 0.2 * effect
        return effect

class SleepTherapy(Treatment):
    def apply(self, dt, current_time):
        effect = super().apply(dt, current_time)
        sim = self.sim
        if hasattr(sim, 'neuromodulators'):
            sim.neuromodulators.concentrations['Cort'] -= 0.3 * effect
            sim.neuromodulators.concentrations['5-HT'] += 0.2 * effect
            sim.neuromodulators.concentrations['GABA'] += 0.1 * effect
        return effect


class TreatmentManager:
    def __init__(self, sim):
        self.sim = sim
        self.treatments = []

    def add_treatment(self, treatment):
        self.treatments.append(treatment)
        logger.info(f"Treatment '{treatment.name}' added.")

    def step(self, dt, current_time):
        total_effects = {}
        for treatment in self.treatments:
            effect = treatment.apply(dt, current_time)
            total_effects[treatment.name] = effect
        return total_effects


def apply_treatment(sim, treatment_name, intensity=1.0, dosage=1.0, frequency=1.0):
    treatment_classes = {
        "SSRI": SSRI,
        "SNRI": SNRI,
        "Benzodiazepine": Benzodiazepine,
        "CBT": CBT,
        "Exposure": Exposure,
        "rTMS": rTMS,
        "Mindfulness": Mindfulness,
        "SleepTherapy": SleepTherapy,
    }
    if not hasattr(sim, "treatment_manager"):
        sim.treatment_manager = TreatmentManager(sim)

    cls = treatment_classes.get(treatment_name)
    if cls is None or treatment_name == "None":
        return None  # No treatment or unknown

    treatment = cls(treatment_name, sim, intensity, dosage, frequency)
    sim.treatment_manager.add_treatment(treatment)
    return treatment
