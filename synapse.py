import numpy as np
from brain_sim.logger import logger

class Synapse:
    """
    Represents a synaptic connection between two individual neurons,
    with short-term plasticity and STDP modulated by neuromodulators.
    """

    def __init__(self, pre_region, post_region, weight, delay_steps, is_inhibitory=False,
                 neuromod=None, plasticity_factor=1.0, pre_neuron=None, post_neuron=None,
                 pre_population=None, post_population=None):
        self.pre = pre_region
        self.post = post_region
        self.delay_steps = delay_steps
        self.queue = [[] for _ in range(delay_steps + 1)]  # Spike queue for delays
        self.is_inhibitory = is_inhibitory
        self.neuromod = neuromod
        self.plasticity_factor = plasticity_factor

        self.pre_neuron = pre_neuron  # index of presynaptic neuron
        self.post_neuron = post_neuron  # index of postsynaptic neuron

        # Add direct references to populations for easy access
        self.pre_population = pre_population
        self.post_population = post_population

        # Initialize weight with some variability
        weight_variation = np.random.uniform(0.9, 1.1)
        self.mu = weight * weight_variation
        self.sigma = 0.1 * self.mu

        if not is_inhibitory:
            self.w_AMPA = 0.7 * self.mu
            self.w_NMDA = 0.3 * self.mu
            self.release_prob_base = np.random.uniform(0.85, 0.95)
        else:
            self.w_GABA = self.mu
            self.release_prob_base = 1.0

        # Short-term plasticity scalars (per synapse)
        self.syn_depression = 1.0
        self.syn_facilitation = 1.0

        self.depression_decay = 0.9
        self.recovery_rate = 0.1
        self.facilitation_decay = 0.9
        self.facilitation_increment = 0.05

        # STDP parameters
        self.A_plus = 0.01 * self.plasticity_factor
        self.A_minus = 0.012 * self.plasticity_factor
        self.tau_plus = 20e-3
        self.tau_minus = 20e-3
        self.obs_variance = 1e-4
        self.homeo_rate = 1e-4
        self.target_weight = self.mu

        self.plasticity_threshold = 0.5
        self.metaplasticity_rate = 1e-3

    def propagate(self):
        """
        Propagate spikes arriving after delay for this neuron-to-neuron synapse,
        modulated by short-term plasticity and neuromodulators.
        """
        arrivals = self.queue.pop(0)

        for neuron_id in arrivals:
            if neuron_id != self.pre_neuron:
                continue

            release_prob = self.release_prob_base
            if self.neuromod:
                D1, D2 = self.neuromod.get_dopamine_receptors()
                mod_factor = 1 + 0.3 * D1[0] - 0.2 * D2[0] if len(D1) > 0 else 1
                release_prob *= np.clip(mod_factor, 0.0, 1.0)

            if np.random.rand() < release_prob:
                eff_amp = 1.0 * self.syn_depression * self.syn_facilitation

                if self.is_inhibitory:
                    if self.neuromod:
                        D1_post, D2_post = self.neuromod.get_dopamine_receptors()
                        mod_factor = 1.0 - 0.3 * D2_post[0] + 0.1 * D1_post[0] if len(D1_post) > 0 else 1.0
                        self.post_population.I[self.post_neuron] -= self.w_GABA * mod_factor * eff_amp
                    else:
                        self.post_population.I[self.post_neuron] -= self.w_GABA * eff_amp
                else:
                    self.post_population.I[self.post_neuron] += eff_amp * self.w_AMPA
                    v_post = self.post_population.v[self.post_neuron]
                    nmda_factor = self.w_NMDA / (1 + np.exp(-v_post / 0.01))
                    self.post_population.I[self.post_neuron] += eff_amp * nmda_factor

                # Facilitation increment
                self.syn_facilitation += self.facilitation_increment
                self.syn_facilitation = np.clip(self.syn_facilitation, 1.0, 2.0)

                # Depression decay
                self.syn_depression *= self.depression_decay

        self.queue.append([])

        # Recovery of depression variable
        self.syn_depression += self.recovery_rate * (1 - self.syn_depression)
        self.syn_depression = np.clip(self.syn_depression, 0.0, 1.0)

    def update_plasticity(self, dt_pre_arr, dt_post_arr):
        """
        STDP update (can be implemented similarly as before).
        """
        pass

    def remodel_topology(self, activity_threshold=0.01):
        # Pruning/growing synapse based on activity (optional)
        return True
