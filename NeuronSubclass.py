import numpy as np
import logging

logger = logging.getLogger(__name__)

class NeuronSubclass:
    def __init__(self, neuron_type, count, base_params, neuromodulator, compartment_idx):
        self.neuron_type = neuron_type
        self.count = count
        self.compartment_idx = compartment_idx
        self.neuromodulator = neuromodulator

        # Initialize intrinsic parameters with Gaussian variability and clipping
        self.a = np.random.normal(base_params['a_mean'], base_params['a_std'], count)
        self.b = np.random.normal(base_params['b_mean'], base_params['b_std'], count)
        self.c = np.random.normal(base_params['c_mean'], base_params['c_std'], count)
        self.d = np.random.normal(base_params['d_mean'], base_params['d_std'], count)

        self.a = np.clip(self.a, 0.005, 0.04)
        self.b = np.clip(self.b, 0.1, 0.4)
        self.c = np.clip(self.c, -75, -50)
        self.d = np.clip(self.d, 0.5, 10)

        # Initial membrane potentials around resting potential c
        self.v = np.random.normal(self.c, 1.0, count)
        self.u = self.b * self.v

        # Input current vector and noise parameters
        self.I = np.zeros(count)
        self.noise_amp = base_params.get('noise_amp', 0.03)  # Increased default noise_amp from 0.02 to 0.03
        self.noise_rate = base_params.get('noise_rate', 0.4)  # Increased noise_rate from 0.3 to 0.4

        self.spikes = []
        self.refractory = np.zeros(count)
        self.last_spike_time = np.full(count, -np.inf)
        self.excitability_adaptation = np.zeros(count)

    def apply_neuromodulation(self, global_time):
        # Fetch receptor activation arrays for this compartment
        D1 = self.neuromodulator.get_receptor_activation('DA', 'D1')[self.compartment_idx]
        D2 = self.neuromodulator.get_receptor_activation('DA', 'D2')[self.compartment_idx]
        HT1A = self.neuromodulator.get_receptor_activation('5-HT', '5HT1A')[self.compartment_idx]
        HT2A = self.neuromodulator.get_receptor_activation('5-HT', '5HT2A')[self.compartment_idx]

        # Calculate recent firing rate (spikes per neuron per second over last 0.5s)
        recent_spikes = [sp for sp in self.spikes if sp[0] > global_time - 0.5]
        recent_rate = len(recent_spikes) / (self.count * 0.5) if self.count > 0 else 0

        sensitivity = np.clip(1.0 + 0.5 * (recent_rate - 0.01), 0.5, 1.5)

        # Apply sensitivity scaling to neuromodulator receptor activations
        D1 *= sensitivity
        D2 *= sensitivity
        HT1A *= sensitivity
        HT2A *= sensitivity

        # Modulate intrinsic neuron parameters by neuromodulation factors
        old_a = self.a.copy()
        old_b = self.b.copy()
        old_noise_amp = self.noise_amp

        self.a = np.clip(self.a * (1 + 0.1 * D1 - 0.07 * D2), 0.005, 0.04)
        self.b = np.clip(self.b * (1 - 0.05 * HT1A + 0.08 * HT2A), 0.1, 0.4)
        self.noise_amp = np.clip(self.noise_amp * (1 + 0.05 * D1 - 0.03 * D2), 0.005, 0.05)

        logger.debug(f"Neuromodulation @ time {global_time:.3f}s: a range {old_a.min():.3f}-{old_a.max():.3f} -> {self.a.min():.3f}-{self.a.max():.3f}")
        logger.debug(f"Neuromodulation @ time {global_time:.3f}s: b range {old_b.min():.3f}-{old_b.max():.3f} -> {self.b.min():.3f}-{self.b.max():.3f}")
        logger.debug(f"Neuromodulation @ time {global_time:.3f}s: noise_amp {old_noise_amp:.4f} -> {self.noise_amp:.4f}")

    def step(self, dt, global_time):
        self.apply_neuromodulation(global_time)

        # Add noise current: Poisson spikes + Gaussian noise
        poisson_spikes = np.random.poisson(self.noise_rate * dt, self.count)
        self.I += poisson_spikes * self.noise_amp + np.random.normal(0, 0.01, self.count)

        # Decay excitability adaptation variable slowly
        self.excitability_adaptation *= 0.995

        active = self.refractory <= 0

        # Update membrane potential v using Izhikevich eq (scaled by dt)
        dv = ((0.04 * self.v[active] ** 2 + 5 * self.v[active] + 140 - self.u[active] + self.I[active]) * dt / 0.001)
        dv = np.clip(dv, -1.0, 1.0)
        self.v[active] += dv

        # Update recovery variable u
        du = (self.a[active] * (self.b[active] * self.v[active] - self.u[active]) * dt)
        du = np.clip(du, -1.0, 1.0)
        self.u[active] += du

        # Clamp membrane potentials and recovery variable
        self.v = np.clip(self.v, -90, 40)
        self.u = np.clip(self.u, -90, 40)

        # Decrease refractory counters
        self.refractory = np.maximum(0, self.refractory - dt)

        # Detect spikes with refractory window of 15 ms
        fired = np.where((self.v >= 30) & ((global_time - self.last_spike_time) > 0.015))[0]

        if len(fired) > 0:
            logger.debug(f"Step {global_time:.3f}s: Neurons fired: {len(fired)} out of {self.count}")

        for idx in fired:
            self.spikes.append((global_time, idx))
            self.v[idx] = self.c[idx] + np.random.normal(0, 1.0)  # Reset membrane potential after spike
            self.refractory[idx] = 0.01
            self.u[idx] += self.d[idx] + np.random.normal(0, 1.0)

            # Increase adaptation and decrease 'a' for firing neuron (spike-frequency adaptation)
            self.excitability_adaptation[idx] += 0.005
            self.a[idx] = np.clip(self.a[idx] * (1 - 0.1 * self.excitability_adaptation[idx]), 0.005, 0.04)

            self.last_spike_time[idx] = global_time

        # Reset input current vector for next step
        self.I.fill(0)

    def reset(self):
        self.spikes.clear()
        self.refractory.fill(0)
        self.last_spike_time.fill(-np.inf)
        self.excitability_adaptation.fill(0)
