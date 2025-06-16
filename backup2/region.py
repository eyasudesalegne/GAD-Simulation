import numpy as np

class NeuronSubclass:
    def __init__(self, neuron_type, count, base_params, neuromodulator, compartment_idx):
        self.neuron_type = neuron_type
        self.count = count
        self.compartment_idx = compartment_idx
        self.neuromodulator = neuromodulator

        self.a = np.random.normal(base_params['a_mean'], base_params['a_std'], count)
        self.b = np.random.normal(base_params['b_mean'], base_params['b_std'], count)
        self.c = np.random.normal(base_params['c_mean'], base_params['c_std'], count)
        self.d = np.random.normal(base_params['d_mean'], base_params['d_std'], count)

        self.a = np.clip(self.a, 0.005, 0.04)
        self.b = np.clip(self.b, 0.1, 0.4)
        self.c = np.clip(self.c, -75, -50)
        self.d = np.clip(self.d, 0.5, 10)

        self.v = np.random.normal(self.c, 1.0, count)
        self.u = self.b * self.v

        self.I = np.zeros(count)
        self.noise_amp = base_params.get('noise_amp', 0.02)
        self.noise_rate = base_params.get('noise_rate', 0.3)

        self.spikes = []
        self.refractory = np.zeros(count)
        self.last_spike_time = np.full(count, -np.inf)

        self.excitability_adaptation = np.zeros(count)

    def apply_neuromodulation(self, global_time):
        D1 = self.neuromodulator.get_receptor_activation('DA', 'D1')[self.compartment_idx]
        D2 = self.neuromodulator.get_receptor_activation('DA', 'D2')[self.compartment_idx]
        HT1A = self.neuromodulator.get_receptor_activation('5-HT', '5HT1A')[self.compartment_idx]
        HT2A = self.neuromodulator.get_receptor_activation('5-HT', '5HT2A')[self.compartment_idx]

        recent_spikes = [sp for sp in self.spikes if sp[0] > global_time - 0.5]
        recent_rate = len(recent_spikes) / (self.count * 0.5)

        receptor_sensitivity = np.clip(1.0 + 0.5 * (recent_rate - 0.01), 0.5, 1.5)

        D1 *= receptor_sensitivity
        D2 *= receptor_sensitivity
        HT1A *= receptor_sensitivity
        HT2A *= receptor_sensitivity

        self.a = np.clip(self.a * (1 + 0.1 * D1 - 0.07 * D2), 0.005, 0.04)
        self.b = np.clip(self.b * (1 - 0.05 * HT1A + 0.08 * HT2A), 0.1, 0.4)
        self.noise_amp = np.clip(self.noise_amp * (1 + 0.05 * D1 - 0.03 * D2), 0.005, 0.05)

    def step(self, dt, global_time):
        self.apply_neuromodulation(global_time)

        poisson_spikes = np.random.poisson(self.noise_rate * dt, self.count)
        self.I += poisson_spikes * self.noise_amp + np.random.normal(0, 0.01, self.count)

        self.excitability_adaptation *= 0.995

        active = self.refractory <= 0

        dv = ((0.04 * self.v[active] ** 2 + 5 * self.v[active] + 140 - self.u[active] + self.I[active]) * dt / 0.001)
        dv = np.clip(dv, -1.0, 1.0)
        self.v[active] += dv

        du = (self.a[active] * (self.b[active] * self.v[active] - self.u[active]) * dt)
        du = np.clip(du, -1.0, 1.0)
        self.u[active] += du

        self.v = np.clip(self.v, -90, 40)
        self.u = np.clip(self.u, -90, 40)

        self.refractory = np.maximum(0, self.refractory - dt)

        fired = np.where((self.v >= 30) & ((global_time - self.last_spike_time) > 0.015))[0]

        for idx in fired:
            self.spikes.append((global_time, idx))
            self.v[idx] = self.c[idx] + np.random.normal(0, 1.0)
            self.refractory[idx] = 0.01
            self.u[idx] += self.d[idx] + np.random.normal(0, 1.0)

            self.excitability_adaptation[idx] += 0.005
            self.a[idx] = np.clip(self.a[idx] * (1 - 0.1 * self.excitability_adaptation[idx]), 0.005, 0.04)

            self.last_spike_time[idx] = global_time

        self.I.fill(0)

    def reset(self):
        self.spikes.clear()
        self.refractory.fill(0)
        self.last_spike_time.fill(-np.inf)
        self.excitability_adaptation.fill(0)

class BrainRegion:
    def __init__(self, name):
        self.name = name
        self.populations = []

    def add_population(self, population):
        self.populations.append(population)

    def step(self, dt, t):
        for pop in self.populations:
            pop.step(dt, t)

    def reset(self):
        for pop in self.populations:
            pop.reset()
