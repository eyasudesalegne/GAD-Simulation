import numpy as np

class Synapse:
    def __init__(self, pre, post, weight, delay_steps, is_inhibitory=False, neuromod=None, plasticity_factor=1.0):
        self.pre = pre
        self.post = post
        self.delay_steps = delay_steps
        self.queue = [[] for _ in range(delay_steps + 1)]
        self.is_inhibitory = is_inhibitory
        self.neuromod = neuromod
        self.plasticity_factor = plasticity_factor

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

        self.use_depression = True
        self.use_facilitation = True

        # Use neuron_count instead of count for BrainRegion
        self.syn_depression = np.ones(post.neuron_count)
        self.syn_facilitation = np.ones(post.neuron_count)

        self.depression_decay = 0.9
        self.recovery_rate = 0.1
        self.facilitation_decay = 0.9
        self.facilitation_increment = 0.05

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
        arrivals = self.queue.pop(0)
        for idx in arrivals:
            release_prob = self.release_prob_base
            if self.neuromod:
                D1, D2 = self.neuromod.get_dopamine_receptors()
                release_prob *= (1 + 0.3 * D1[idx] - 0.2 * D2[idx])
                release_prob = np.clip(release_prob, 0.0, 1.0)

            if np.random.rand() < release_prob:
                eff_amp = 1.0
                if self.use_depression:
                    eff_amp *= self.syn_depression[idx]
                    self.syn_depression[idx] *= self.depression_decay
                if self.use_facilitation:
                    eff_amp *= self.syn_facilitation[idx]
                    self.syn_facilitation[idx] *= self.facilitation_decay

                if self.is_inhibitory:
                    D1_post, D2_post = self.neuromod.get_dopamine_receptors()
                    mod_factor = 1.0 - 0.3 * D2_post[idx] + 0.1 * D1_post[idx]
                    self.post.I[idx] -= self.w_GABA * mod_factor * eff_amp
                else:
                    self.post.I[idx] += eff_amp * self.w_AMPA
                    self.post.I[idx] += eff_amp * self.w_NMDA / (1 + np.exp(-self.post.v[idx] / 0.01))

                if self.use_facilitation:
                    self.syn_facilitation[idx] += self.facilitation_increment
                    self.syn_facilitation[idx] = np.clip(self.syn_facilitation[idx], 1.0, 2.0)

        self.queue.append([])

        if self.use_depression:
            self.syn_depression += self.recovery_rate * (1 - self.syn_depression)
            self.syn_depression = np.clip(self.syn_depression, 0.0, 1.0)

    def update_plasticity(self, dt_pre_arr, dt_post_arr):
        if self.neuromod:
            D1, D2 = self.neuromod.get_dopamine_receptors()
            HT1A, HT2A = self.neuromod.get_serotonin_receptors()
        else:
            D1 = D2 = HT1A = HT2A = np.zeros_like(dt_pre_arr)

        delta_t = dt_post_arr - dt_pre_arr

        positive_mask = delta_t > 0
        negative_mask = ~positive_mask

        measurement_update = np.zeros_like(delta_t)
        measurement_update[positive_mask] = self.A_plus * np.exp(-delta_t[positive_mask] / self.tau_plus)
        measurement_update[negative_mask] = -self.A_minus * np.exp(delta_t[negative_mask] / self.tau_minus)

        mod_factor = 1.0 + 0.6 * D1 - 0.3 * D2 + 0.2 * HT2A - 0.2 * HT1A
        measurement_update *= mod_factor

        self.plasticity_threshold += self.metaplasticity_rate * (np.abs(measurement_update) - self.plasticity_threshold)
        adaptive_learning_rate = np.where(measurement_update > 0, self.A_plus, self.A_minus)
        adaptive_learning_rate *= 1 / (1 + np.exp(-(np.abs(measurement_update) - self.plasticity_threshold) * 10))

        new_weight_obs = self.mu + measurement_update
        K = (self.sigma ** 2) / (self.sigma ** 2 + self.obs_variance)

        new_mu = self.mu + K * (new_weight_obs - self.mu) * adaptive_learning_rate
        new_sigma = (1 - K) * self.sigma

        scaling = 1 - self.homeo_rate * (new_mu - self.target_weight)
        scaling = np.clip(scaling, 0.0, 1.0)

        new_mu = np.where(new_mu > 0, new_mu * scaling, 0.0)

        self.mu = new_mu
        self.sigma = new_sigma
        self.w_AMPA = 0.7 * self.mu
        self.w_NMDA = 0.3 * self.mu

    def remodel_topology(self, activity_threshold=0.01):
        # Use neuron_count instead of count
        recent_activity = len(self.pre.spikes) / max(1.0, self.pre.neuron_count)

        if self.mu < activity_threshold:
            if recent_activity < activity_threshold:
                self.mu = 0
                self.w_AMPA = 0
                self.w_NMDA = 0
                return False
            else:
                self.mu += 0.001
                self.w_AMPA = 0.7 * self.mu
                self.w_NMDA = 0.3 * self.mu

        return True
