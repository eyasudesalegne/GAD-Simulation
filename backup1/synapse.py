import numpy as np

class Synapse:
    """
    Delayed synaptic connection with heterogeneous properties.
    Features include:
      - Variability in synaptic weights via random scaling.
      - Variability in plasticity rules modulated by a plasticity_factor.
      - Incorporation of AMPA and NMDA components for excitatory synapses.
      - GABAergic inhibition modulated by dopamine.
      - Bayesian STDP plasticity with Kalman filter updates.
      - Homeostatic scaling toward target weights.
      - Optional short-term depression with stochastic release.
    """
    def __init__(self, pre, post, weight, delay_steps, is_inhibitory=False, neuromod=None, plasticity_factor=1.0):
        self.pre = pre
        self.post = post
        self.delay_steps = delay_steps  # Already includes delay variability from connection setup
        self.queue = [[] for _ in range(delay_steps + 1)]
        self.is_inhibitory = is_inhibitory
        self.neuromod = neuromod  # Reference to neuromodulator hub
        
        # Introduce variability in synaptic weight by applying a ±10% random scaling factor
        weight_variation = np.random.uniform(0.9, 1.1)
        adjusted_weight = weight * weight_variation
        
        self.plasticity_factor = plasticity_factor
        
        if not is_inhibitory:
            self.mu = adjusted_weight  # Base synaptic weight with variability
            self.sigma = 0.1 * self.mu
            self.w_AMPA = 0.7 * self.mu
            self.w_NMDA = 0.3 * self.mu
            # Vary release probability slightly (±5%)
            self.release_prob = np.random.uniform(0.85, 0.95)
        else:
            self.w_GABA = adjusted_weight
        
        # Modulate plasticity parameters with plasticity_factor for heterogeneous plasticity rules
        self.A_plus = 0.01 * self.plasticity_factor
        self.A_minus = 0.012 * self.plasticity_factor
        self.tau_plus = 20e-3
        self.tau_minus = 20e-3
        self.obs_variance = 1e-4

        self.homeo_rate = 1e-4
        self.target_weight = adjusted_weight

        # Short-term depression parameters with variability
        self.use_depression = True
        self.syn_depression = np.ones(post.neuron_count)
        self.depression_decay = np.random.uniform(0.85, 0.95)  # Variability in depression decay
        self.recovery_rate = np.random.uniform(0.08, 0.12)       # Variability in recovery rate

    def propagate(self):
        arrivals = self.queue.pop(0)
        for idx in arrivals:
            if self.is_inhibitory:
                # Inhibitory synapses are modulated by the current dopamine level
                mod_factor = 1.0 - 0.5 * self.neuromod.state.get('DA', 0)
                self.post.I[idx] -= self.w_GABA * mod_factor
            else:
                if np.random.rand() < self.release_prob:
                    if self.use_depression:
                        eff_amp = self.syn_depression[idx]
                        self.syn_depression[idx] *= self.depression_decay
                    else:
                        eff_amp = 1.0
                    # Sum AMPA and NMDA contributions with effective amplitude
                    self.post.I[idx] += eff_amp * self.w_AMPA
                    self.post.I[idx] += eff_amp * self.w_NMDA / (1 + np.exp(-self.post.v[idx] / 0.01))
        self.queue.append([])

        if self.use_depression:
            self.syn_depression += self.recovery_rate * (1 - self.syn_depression)

    def update_plasticity(self, dt_pre, dt_post):
        delta_t = dt_post - dt_pre

        if delta_t > 0:
            measurement_update = self.A_plus * np.exp(-delta_t / self.tau_plus)
        else:
            measurement_update = -self.A_minus * np.exp(delta_t / self.tau_minus)

        # Modulate the update with dopamine influence
        mod_factor = 1.0 + 0.5 * self.neuromod.state.get('DA', 0)
        measurement_update *= mod_factor

        new_weight_obs = self.mu + measurement_update
        K = (self.sigma ** 2) / (self.sigma ** 2 + self.obs_variance)

        new_mu = self.mu + K * (new_weight_obs - self.mu)
        new_sigma = (1 - K) * self.sigma

        if new_mu > 0:
            scaling = 1 - self.homeo_rate * (new_mu - self.target_weight)
            new_mu *= scaling

        self.mu = new_mu
        self.sigma = new_sigma
        self.w_AMPA = 0.7 * self.mu
        self.w_NMDA = 0.3 * self.mu
