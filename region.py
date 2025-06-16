import numpy as np
from brain_sim.logger import logger  # Assuming logger supports debug/info levels

class NeuronSubclass:
    """
    Represents a population (subclass) of neurons within a brain region.
    Uses Izhikevich neuron model parameters with neuromodulator influences.
    """

    def __init__(self, neuron_type, count, base_params, neuromodulator, compartment_idx, region_centroid=None):
        self.neuron_type = neuron_type
        self.count = count
        self.compartment_idx = compartment_idx
        self.neuromodulator = neuromodulator

        # Initialize neuron parameters with Gaussian variation and clip to plausible ranges
        self.a = np.random.normal(base_params['a_mean'], base_params['a_std'], count)
        self.b = np.random.normal(base_params['b_mean'], base_params['b_std'], count)
        self.c = np.random.normal(base_params['c_mean'], base_params['c_std'], count)
        self.d = np.random.normal(base_params['d_mean'], base_params['d_std'], count)

        self.a = np.clip(self.a, 0.005, 0.04)
        self.b = np.clip(self.b, 0.1, 0.4)
        self.c = np.clip(self.c, -75, -50)
        self.d = np.clip(self.d, 0.5, 10)

        self.v = np.random.normal(self.c, 1.0, count)  # Initial membrane potentials near resting
        self.u = self.b * self.v  # Recovery variable

        self.I = np.zeros(count)  # Input currents

        # Increased default noise amplitude and rate to improve baseline spiking
        self.noise_amp = base_params.get('noise_amp', 0.05)  # increased from 0.03 to 0.05
        self.noise_rate = base_params.get('noise_rate', 0.5)  # increased from 0.4 to 0.5

        self.spikes = []  # List of (time, neuron_idx) spike events
        self.refractory = np.zeros(count)  # Refractory timers
        self.last_spike_time = np.full(count, -np.inf)  # Track last spike times

        self.excitability_adaptation = np.zeros(count)  # Adaptation variable

        # Generate neuron coordinates near region centroid (for potential spatial analyses)
        if region_centroid is not None:
            # Gaussian scatter around centroid with std dev of 2 mm (example value)
            self.coordinates = np.random.normal(loc=region_centroid, scale=2.0, size=(count, 3))
        else:
            # If no centroid given, place at origin + small random jitter
            self.coordinates = np.random.normal(loc=0.0, scale=2.0, size=(count, 3))

    def apply_neuromodulation(self, global_time):
        """
        Modulates intrinsic neuron parameters based on neuromodulator receptor activations and recent firing rate.
        """
        D1 = self.neuromodulator.get_receptor_activation('DA', 'D1')[self.compartment_idx]
        D2 = self.neuromodulator.get_receptor_activation('DA', 'D2')[self.compartment_idx]
        HT1A = self.neuromodulator.get_receptor_activation('5-HT', '5HT1A')[self.compartment_idx]
        HT2A = self.neuromodulator.get_receptor_activation('5-HT', '5HT2A')[self.compartment_idx]

        # Compute recent firing rate (last 0.5 seconds)
        recent_spikes = [sp for sp in self.spikes if sp[0] > global_time - 0.5]
        recent_rate = len(recent_spikes) / (self.count * 0.5) if self.count > 0 else 0

        receptor_sensitivity = np.clip(1.0 + 0.5 * (recent_rate - 0.01), 0.5, 1.5)

        D1_mod = D1 * receptor_sensitivity
        D2_mod = D2 * receptor_sensitivity
        HT1A_mod = HT1A * receptor_sensitivity
        HT2A_mod = HT2A * receptor_sensitivity

        old_a = self.a.copy()
        old_b = self.b.copy()
        old_noise_amp = self.noise_amp

        # Modulate neuron parameters
        self.a = np.clip(self.a * (1 + 0.1 * D1_mod - 0.07 * D2_mod), 0.005, 0.04)
        self.b = np.clip(self.b * (1 - 0.05 * HT1A_mod + 0.08 * HT2A_mod), 0.1, 0.4)
        self.noise_amp = np.clip(self.noise_amp * (1 + 0.05 * D1_mod - 0.03 * D2_mod), 0.005, 0.07)  # slight increase cap

        logger.debug(f"Neuromodulation @ {global_time:.3f}s: a range {old_a.min():.4f}-{old_a.max():.4f} -> {self.a.min():.4f}-{self.a.max():.4f}")
        logger.debug(f"Neuromodulation @ {global_time:.3f}s: b range {old_b.min():.4f}-{old_b.max():.4f} -> {self.b.min():.4f}-{self.b.max():.4f}")
        logger.debug(f"Neuromodulation @ {global_time:.3f}s: noise_amp {old_noise_amp:.4f} -> {self.noise_amp:.4f}")

    def step(self, dt, global_time):
        """
        Advances the state of all neurons by one timestep.
        """
        self.apply_neuromodulation(global_time)

        # Generate Poisson noise spikes and Gaussian noise, add to input currents
        poisson_spikes = np.random.poisson(self.noise_rate * dt, self.count)
        self.I += poisson_spikes * self.noise_amp + np.random.normal(0, 0.015, self.count)  # increased Gaussian noise std

        # Decay excitability adaptation factor
        self.excitability_adaptation *= 0.995

        # Identify neurons not in refractory period
        active = self.refractory <= 0

        # Izhikevich model differential equations for membrane potential and recovery variable
        dv = ((0.04 * self.v[active] ** 2 + 5 * self.v[active] + 140 - self.u[active] + self.I[active]) * dt / 0.001)
        dv = np.clip(dv, -1.0, 1.0)
        self.v[active] += dv

        du = (self.a[active] * (self.b[active] * self.v[active] - self.u[active]) * dt)
        du = np.clip(du, -1.0, 1.0)
        self.u[active] += du

        # Clip voltages to physiological ranges
        self.v = np.clip(self.v, -90, 40)
        self.u = np.clip(self.u, -90, 40)

        # Decrement refractory timers
        self.refractory = np.maximum(0, self.refractory - dt)

        # Find neurons firing spikes, with refractory guard of 15 ms
        fired = np.where((self.v >= 30) & ((global_time - self.last_spike_time) > 0.015))[0]

        if len(fired) > 0:
            logger.debug(f"Step {global_time:.3f}s: Neurons fired: {len(fired)} out of {self.count}")

        for idx in fired:
            self.spikes.append((global_time, idx))
            self.v[idx] = self.c[idx] + np.random.normal(0, 1.0)
            self.refractory[idx] = 0.01
            self.u[idx] += self.d[idx] + np.random.normal(0, 1.0)

            # Increase adaptation and reduce 'a' parameter (excitability decrease)
            self.excitability_adaptation[idx] += 0.005
            self.a[idx] = np.clip(self.a[idx] * (1 - 0.1 * self.excitability_adaptation[idx]), 0.005, 0.04)

            self.last_spike_time[idx] = global_time

        # Clear input currents for next timestep
        self.I.fill(0)

    def reset(self):
        """
        Reset neuron state: clear spikes, log current accumulation, and reset timers.
        """
        logger.info(f"Resetting neuron subclass '{self.neuron_type}' with {self.count} neurons.")
        logger.info(f"Current accumulation before reset: mean={np.mean(self.I):.4f}, std={np.std(self.I):.4f}")

        self.spikes.clear()
        self.refractory.fill(0)
        self.last_spike_time.fill(-np.inf)
        self.excitability_adaptation.fill(0)
        self.I.fill(0)


class BrainRegion:
    """
    Represents a brain region containing multiple neuron populations.
    Keeps anatomical centroid and global index offset.
    """

    def __init__(self, name, centroid=None):
        self.name = name
        self.populations = []
        self.centroid = np.array(centroid) if centroid is not None else np.zeros(3)
        self.global_offset = 0  # Set by simulation for global indexing

    @property
    def neuron_count(self):
        """
        Total neurons across all populations in this region.
        """
        return sum(pop.count for pop in self.populations)

    @property
    def spikes(self):
        """
        Aggregate spikes from all populations as list of (time, global_neuron_idx).
        """
        aggregated_spikes = []
        neuron_offset = 0
        for pop in self.populations:
            for t, local_id in pop.spikes:
                global_id = neuron_offset + local_id
                aggregated_spikes.append((t, global_id))
            neuron_offset += pop.count
        return aggregated_spikes

    def add_population(self, population):
        """
        Add a neuron population to this region.
        """
        self.populations.append(population)
        logger.info(f"Added population '{population.neuron_type}' with {population.count} neurons to region '{self.name}'")

    def step(self, dt, t):
        """
        Step all neuron populations by dt.
        """
        for pop in self.populations:
            pop.step(dt, t)

    def reset(self):
        """
        Reset all neuron populations in the region.
        """
        for pop in self.populations:
            pop.reset()
        logger.info(f"Reset all populations in region '{self.name}'")
