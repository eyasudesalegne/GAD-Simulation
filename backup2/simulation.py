import numpy as np
from brain_sim.neuromodulator import Neuromodulator
from brain_sim.region import BrainRegion
from brain_sim.synapse import Synapse
from brain_sim.connectome_data import connectome
from brain_sim.logger import (
    logger,
    log_region_added,
    log_synapse_created,
    log_simulation_step,
    log_neuromodulator_update,
    log_spike_count,
    log_region_firing_rates,
    log_region_burst_counts,
    log_average_synaptic_weight
)

import brain_sim.analytics as analytics  

class SimulationWithAnalytics:
    def __init__(self, dt=0.001, duration=10.0):
        self.dt = dt
        self.duration = duration
        self.steps = int(duration / dt)

        self.conduction_velocity = 10.0
        self.weight_scale = 1.0
        self.p_base_default = 0.3
        self.lambda_dist = 5.0

        self.regions = {}
        self.synapses = []
        self.spikes = []
        self.connections = []

        self.neuromodulators = Neuromodulator(dt, duration)

        self.external_inputs = {
            "LA_E": lambda t: 0.02 * np.sin(2 * np.pi * 6 * t),
            "mPFC_E": lambda t: 0.01 * np.sin(2 * np.pi * 10 * t),
            "BLA_E": lambda t: 0.05 if (t % 0.2) < 0.05 else 0.0,
            "DG_E": lambda t: 0.03 * np.sin(2 * np.pi * 8 * t) + 0.01 * np.sin(2 * np.pi * 12 * t),
        }

        self.analytics_results = {}

        logger.info(f"🧠 Simulation initialized with dt={dt}, duration={duration}s, steps={self.steps}")
        self.build_from_connectome(connectome)

    def add_region(self, region):
        region_offset = sum(r.neuron_count for r in self.regions.values())
        region.global_offset = region_offset
        self.regions[region.name] = region
        log_region_added(region.name, region.neuron_count)

    def build_from_connectome(self, conn_dict):
        region_names = set(conn_dict.keys())
        for targets in conn_dict.values():
            region_names.update(targets.keys())

        for name in sorted(region_names):
            if name not in self.regions:
                try:
                    base_name, neuron_suffix = name.rsplit("_", 1)
                except ValueError:
                    base_name, neuron_suffix = name, "E"
                is_exc = (neuron_suffix == "E")
                density = 4000 if is_exc else 1000
                volume = 1000
                neuron_type = "excitatory" if is_exc else "inhibitory"
                region = BrainRegion(name)
                region.neuron_count = int(density * volume / 1000)
                self.add_region(region)

        syn_count = 0
        for src, targets in conn_dict.items():
            for tgt, conn_params in targets.items():
                if self.connect(src, tgt, conn_params):
                    syn_count += 1

        logger.info(f"✅ Added {len(self.regions)} regions and {syn_count} synapses.")

    def connect(self, src_name, tgt_name, conn_params):
        if src_name not in self.regions or tgt_name not in self.regions:
            logger.warning(f"⚠️ Connection skipped: {src_name} or {tgt_name} not found.")
            return False

        src_region = self.regions[src_name]
        tgt_region = self.regions[tgt_name]

        weight = conn_params.get("weight", 0.5)
        plasticity_factor = conn_params.get("plasticity", 1.0)
        delay_jitter = conn_params.get("delay_jitter", 5)

        dist = np.linalg.norm(getattr(src_region, 'centroid', np.zeros(3)) - getattr(tgt_region, 'centroid', np.zeros(3)))
        myelination_factor = np.random.uniform(0.5, 1.5)
        delay_ms = (dist / (self.conduction_velocity * myelination_factor)) * 1000
        delay_ms += np.random.normal(0, delay_jitter)
        delay_ms = max(1.0, delay_ms)
        delay_steps = max(1, int(delay_ms / (self.dt * 1000)))

        prob = self.p_base_default * np.exp(-dist / self.lambda_dist)
        prob *= np.random.uniform(0.9, 1.1)
        DA = self.neuromodulators.concentrations.get('DA', np.array([0]))[0]
        prob *= (1 + 0.5 * DA)

        if np.random.rand() < prob:
            is_inhibitory = (weight < 0)
            syn_weight = abs(weight) * self.weight_scale * np.random.uniform(0.8, 1.2)
            syn = Synapse(src_region, tgt_region, syn_weight, delay_steps, is_inhibitory, self.neuromodulators, plasticity_factor)
            self.synapses.append(syn)
            self.connections.append({
                'src': src_name,
                'tgt': tgt_name,
                'weight': weight,
                'plasticity': plasticity_factor,
                'delay_ms': delay_ms,
                'myelination_factor': myelination_factor
            })
            log_synapse_created(src_name, tgt_name, syn_weight, delay_ms)
            return True
        return False

    def apply_external_drives(self, t):
        for region_name, func in self.external_inputs.items():
            if region_name in self.regions:
                region = self.regions[region_name]
                if hasattr(region, 'I'):
                    region.I += func(t)

    def step_regions(self, dt, t):
        for region in self.regions.values():
            region.step(dt, t)

    def collect_spikes(self):
        count = 0
        for region in self.regions.values():
            for spike_time, local_id in getattr(region, 'spikes', []):
                self.spikes.append((spike_time, local_id + getattr(region, 'global_offset', 0)))
                count += 1
            if hasattr(region, 'spikes'):
                region.spikes.clear()
        return count

    def propagate_synapses(self):
        for syn in self.synapses:
            syn.propagate()

    def log_metrics(self, current_step):
        log_neuromodulator_update(current_step, self.neuromodulators.get_state())
        step_spikes = sum(1 for t, _ in self.spikes if int(t / self.dt) == current_step)
        log_spike_count(current_step, step_spikes)
        region_rates = self.compute_region_firing_rates(window=200)
        log_region_firing_rates(current_step, region_rates)
        burst_counts = self.compute_burst_statistics(window=200)
        log_region_burst_counts(current_step, burst_counts)
        avg_weight = np.mean([syn.mu if not syn.is_inhibitory else getattr(syn, 'w_GABA', 0) for syn in self.synapses])
        log_average_synaptic_weight(current_step, avg_weight)

    def step(self, dt, t):
        current_step = int(t / dt)
        self.apply_external_drives(t)
        self.step_regions(dt, t)
        step_spikes = self.collect_spikes()
        self.propagate_synapses()
        global_firing_rate = step_spikes / (sum(r.neuron_count for r in self.regions.values()) * dt) if sum(r.neuron_count for r in self.regions.values()) > 0 else 0
        reward_signal = 1.0 if step_spikes > 0.05 * sum(r.neuron_count for r in self.regions.values()) else 0.0
        self.neuromodulators.step({}, reward_signal)
        if current_step % 200 == 0:
            self.log_metrics(current_step)
        if current_step % 1000 == 0 or current_step == self.steps - 1:
            log_simulation_step(current_step, self.steps)

    def run(self):
        logger.info("🚀 Running full simulation with integrated analytics...")
        t = 0.0
        for _ in range(self.steps):
            self.step(self.dt, t)
            t += self.dt

        logger.info(f"✅ Simulation complete. Total spikes: {len(self.spikes)}")
        self.analyze_and_store_results()

    def analyze_and_store_results(self):
        logger.info("🔍 Running integrated analytics...")
        self.analytics_results['simple'] = {
            'firing_rates': analytics.simple_firing_rates(self),
            'spike_counts': analytics.simple_spike_counts(self),
        }
        self.analytics_results['intermediate'] = {
            'isi_cv': analytics.intermediate_isi_cv(self),
            'fano_factor': analytics.intermediate_fano_factor(self),
        }
        self.analytics_results['advanced'] = {
            'burst_data': analytics.advanced_burst_detection(self),
            'synaptic_weights_trends': analytics.advanced_synaptic_plasticity_trends(self),
            'neuromod_correlation': analytics.advanced_neuromodulator_correlation(self),
            'spike_entropy': analytics.advanced_spike_entropy(self),
        }
        logger.info("🔍 Analytics complete.")

    def plot_all_analytics(self):
        analytics.simple_plot(self)
        analytics.intermediate_plot(self)
        analytics.advanced_plot(self)

    def compute_region_firing_rates(self, window=200):
        rates = {}
        time_window = window * self.dt
        current_time = self.steps * self.dt
        window_start_time = current_time - time_window
        spikes_in_window = [(t, nid) for (t, nid) in self.spikes if t >= window_start_time]
        region_spike_counts = {name: 0 for name in self.regions}
        for t, nid in spikes_in_window:
            for name, region in self.regions.items():
                offset = getattr(region, 'global_offset', 0)
                count = getattr(region, 'neuron_count', 1)
                if offset <= nid < offset + count:
                    region_spike_counts[name] += 1
                    break
        for name, count in region_spike_counts.items():
            n_neurons = self.regions[name].neuron_count
            rates[name] = count / (n_neurons * time_window) if n_neurons > 0 else 0
        return rates

    def compute_burst_statistics(self, window=200):
        bursts_per_region = {name: 0 for name in self.regions}
        time_window = window * self.dt
        current_time = self.steps * self.dt
        window_start_time = current_time - time_window
        neuron_spikes = {}
        for t, gid in self.spikes:
            if t >= window_start_time:
                neuron_spikes.setdefault(gid, []).append(t)
        burst_threshold = 0.01
        neuron_to_region = {}
        for region_name, region in self.regions.items():
            offset = getattr(region, 'global_offset', 0)
            count = getattr(region, 'neuron_count', 1)
            for nid in range(offset, offset + count):
                neuron_to_region[nid] = region_name
        for gid, times in neuron_spikes.items():
            times = np.sort(times)
            if len(times) < 2:
                continue
            isis = np.diff(times)
            bursts = 0
            consecutive = 0
            for isi in isis:
                if isi < burst_threshold:
                    consecutive += 1
                else:
                    if consecutive >= 2:
                        bursts += 1
                    consecutive = 0
            if consecutive >= 2:
                bursts += 1
            region_name = neuron_to_region.get(gid, None)
            if region_name:
                bursts_per_region[region_name] += bursts
        return bursts_per_region

    def reset(self):
        self.spikes.clear()
        self.connections.clear()
        self.synapses.clear()
        for region in self.regions.values():
            region.reset()
