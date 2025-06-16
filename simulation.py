import numpy as np
from brain_sim.neuromodulator import Neuromodulator
from brain_sim.region import BrainRegion, NeuronSubclass
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

# Import your external connectome validation function
from brain_sim.validate_connectome import validate_connectome

# Your constants (no changes)
NEURON_COUNTS = {
    "excitatory": 400,
    "inhibitory": 100
}

CONNECTION_PROBABILITIES = {
    "exc_to_exc": 0.15,
    "exc_to_inh": 0.30,
    "inh_to_exc": 0.40,
    "inh_to_inh": 0.25,
    "long_range": 0.1
}

LAMBDA_DIST = 30.0
CONDUCTION_VELOCITY = 50.0  # m/s

DELAY = {
    "min_ms": 1.0,
    "max_ms": 50.0,
    "jitter_std": 10.0
}

NOISE = {
    "amplitude": 0.03,
    "rate": 0.3
}

EXTERNAL_DRIVES = {
    "LA_E": 0.02,
    "mPFC_E": 0.01,
    "BLA_E": 0.05,
    "DG_E": 0.03
}

MAX_SYNAPSES_PER_PRE = 50
MAX_POST_SAMPLES = 400

REGION_MNI_COORDS = {
    "LA": [-25, -5, -15], "BLA": [-30, -6, -20], "CeA": [-20, -4, -10],
    "DLPFC": [-40, 30, 30], "vmPFC": [0, 50, -10], "OFC": [25, 30, -15],
    "mPFC": [0, 50, 20], "dACC": [0, 20, 35], "rACC": [0, 40, 10],
    "DG": [-20, -40, -10], "CA3": [-30, -30, -10], "CA1": [-35, -25, -15],
    "CA2": [-32, -27, -12], "Subiculum": [-25, -35, -20], "AI": [32, 20, 0],
    "PI": [38, -10, 10], "dPAG": [0, -30, -20], "lPAG": [5, -32, -18],
    "vlPAG": [-5, -28, -22], "aBNST": [0, 0, -5], "pBNST": [0, -10, -5]
}

class SimulationWithAnalytics:
    def __init__(self, dt=0.001, duration=10.0):
        self.dt = dt
        self.duration = duration
        self.steps = int(duration / dt)

        self.conduction_velocity = CONDUCTION_VELOCITY
        self.weight_scale = 1.0
        self.p_base_default = CONNECTION_PROBABILITIES['long_range']
        self.lambda_dist = LAMBDA_DIST

        self.regions = {}
        self.synapses = []
        self.spikes = []
        self.connections = []

        self.neuromodulators = Neuromodulator(dt, duration)

        self.external_inputs = EXTERNAL_DRIVES

        self.analytics_results = {}

        logger.info(f"🧠 Simulation initialized with dt={dt}, duration={duration}s, steps={self.steps}")

        # Collect all unique region names from connectome keys and targets
        region_names = set(connectome.keys())
        for targets in connectome.values():
            region_names.update(targets.keys())

        # Validate connectome dictionary against these region names
        logger.info("Validating connectome data against region list...")
        validate_connectome(connectome, valid_regions=region_names, require_bidirectional=True)

        # Build the simulation from the validated connectome
        self.build_from_connectome(connectome)

        # Report initial info about regions and distances
        self.report_initialization()

    def validate_connectome_vs_regions(self, conn_dict):
        connectome_regions = set(conn_dict.keys())
        for targets in conn_dict.values():
            connectome_regions.update(targets.keys())

        sim_regions = set(self.regions.keys())

        missing_in_sim = connectome_regions - sim_regions
        missing_in_connectome = sim_regions - connectome_regions

        if missing_in_sim:
            logger.warning(f"⚠️ Regions referenced in connectome but missing in simulation: {missing_in_sim}")
        else:
            logger.info("✅ All regions referenced in connectome exist in simulation.")

        if missing_in_connectome:
            logger.warning(f"⚠️ Regions present in simulation but missing in connectome: {missing_in_connectome}")
        else:
            logger.info("✅ All simulation regions are referenced in connectome.")

        return missing_in_sim, missing_in_connectome

    def report_initialization(self):
        logger.info(f"Simulation has {len(self.regions)} regions:")
        for region in self.regions.values():
            types = set(pop.neuron_type for pop in region.populations)
            types_str = ", ".join(types)
            logger.info(f" - Region {region.name}: {region.neuron_count} neurons ({types_str})")

        names = list(self.regions.keys())
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                r1 = self.regions[names[i]]
                r2 = self.regions[names[j]]
                dist = np.linalg.norm(np.array(r1.centroid) - np.array(r2.centroid))
                logger.info(f"Distance {r1.name} -> {r2.name}: {dist:.2f} mm")

    def add_region(self, region):
        region_offset = sum(r.neuron_count for r in self.regions.values())
        region.global_offset = region_offset
        self.regions[region.name] = region
        log_region_added(region.name, region.neuron_count)

    def build_from_connectome(self, conn_dict):
        base_params_exc = {
            'a_mean': 0.02, 'a_std': 0.005,
            'b_mean': 0.2, 'b_std': 0.05,
            'c_mean': -65, 'c_std': 2,
            'd_mean': 8, 'd_std': 2,
            'noise_amp': NOISE['amplitude'],
            'noise_rate': NOISE['rate']
        }
        base_params_inh = {
            'a_mean': 0.1, 'a_std': 0.02,
            'b_mean': 0.2, 'b_std': 0.05,
            'c_mean': -65, 'c_std': 2,
            'd_mean': 2, 'd_std': 1,
            'noise_amp': NOISE['amplitude'],
            'noise_rate': NOISE['rate']
        }

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
                density = NEURON_COUNTS['excitatory'] if is_exc else NEURON_COUNTS['inhibitory']
                volume = 1000
                neuron_type = "excitatory" if is_exc else "inhibitory"

                centroid = REGION_MNI_COORDS.get(base_name, [0, 0, 0])
                logger.info(f"Assigning centroid {centroid} to region {name}")

                region = BrainRegion(name, centroid=centroid)
                neuron_count = int(density * volume / 1000)
                pop = NeuronSubclass(neuron_type, neuron_count,
                                     base_params_exc if is_exc else base_params_inh,
                                     self.neuromodulators, compartment_idx=0,
                                     region_centroid=centroid)
                region.add_population(pop)

                self.add_region(region)

        total_synapses = 0
        for src, targets in conn_dict.items():
            for tgt, conn_params in targets.items():
                total_synapses += self.connect(src, tgt, conn_params)

        logger.info(f"✅ Added {len(self.regions)} regions and {total_synapses} synapses.")

    def connect(self, src_name, tgt_name, conn_params):
        if src_name not in self.regions or tgt_name not in self.regions:
            logger.warning(f"⚠️ Connection skipped: {src_name} or {tgt_name} not found.")
            return 0

        src_region = self.regions[src_name]
        tgt_region = self.regions[tgt_name]

        weight = conn_params.get("weight", 0.5)
        plasticity_factor = conn_params.get("plasticity", 1.0)
        delay_jitter = DELAY['jitter_std']

        dist = np.linalg.norm(src_region.centroid - tgt_region.centroid)
        myelination_factor = np.random.uniform(0.8, 3.0)
        delay_ms_raw = dist / (self.conduction_velocity * myelination_factor)
        delay_ms_noisy = delay_ms_raw + np.random.normal(0, delay_jitter)

        logger.info(f"Distance (mm): {dist:.2f} for connection {src_name} -> {tgt_name}")
        logger.info(f"Raw delay (ms): {delay_ms_noisy:.2f} for connection {src_name} -> {tgt_name}")

        delay_ms = np.clip(delay_ms_noisy, DELAY['min_ms'], DELAY['max_ms'])

        logger.info(f"Capped delay (ms): {delay_ms:.2f} for connection {src_name} -> {tgt_name}")

        delay_steps = max(1, int(delay_ms / self.dt))

        prob_base = self.p_base_default * np.exp(-dist / self.lambda_dist)
        prob_base *= np.random.uniform(0.9, 1.1)
        DA = self.neuromodulators.concentrations.get('DA', np.array([0.2]))[0]
        prob_base *= (1 + 0.5 * DA)

        synapse_count = 0
        src_pop = src_region.populations[0]
        tgt_pop = tgt_region.populations[0]

        max_syn_per_pre = MAX_SYNAPSES_PER_PRE
        max_post_samples = min(tgt_pop.count, MAX_POST_SAMPLES)

        for pre_idx in range(src_pop.count):
            post_indices_sampled = np.random.choice(tgt_pop.count, max_post_samples, replace=False)
            created_for_pre = 0
            for post_idx in post_indices_sampled:
                if np.random.rand() < prob_base:
                    syn_weight = abs(weight) * self.weight_scale * np.random.uniform(0.8, 1.2)
                    is_inhibitory = (weight < 0)

                    syn = Synapse(src_region, tgt_region, syn_weight, delay_steps,
                                  is_inhibitory, self.neuromodulators, plasticity_factor,
                                  pre_neuron=pre_idx, post_neuron=post_idx,
                                  pre_population=src_pop, post_population=tgt_pop)
                    self.synapses.append(syn)
                    synapse_count += 1
                    created_for_pre += 1
                    log_synapse_created(src_name, tgt_name, syn_weight, delay_ms)

                    if created_for_pre >= max_syn_per_pre:
                        break

        return synapse_count

    def apply_external_drives(self, t):
        for region_name, drive in self.external_inputs.items():
            region = self.regions.get(region_name, None)
            if region is None:
                logger.warning(f"External drive skipped: Region {region_name} not found.")
                continue
            if not region.populations:
                logger.warning(f"External drive skipped: Region {region_name} has no populations.")
                continue
            drive_value = drive(t) if callable(drive) else drive
            for pop in region.populations:
                if hasattr(pop, 'I'):
                    pop.I += drive_value
                else:
                    logger.warning(f"Population in region {region_name} missing input current attribute.")

    def step_regions(self, dt, t):
        for region in self.regions.values():
            region.step(dt, t)

    def collect_spikes(self):
        self.spikes.clear()
        for region in self.regions.values():
            offset = region.global_offset
            for t_spike, local_id in region.spikes:
                self.spikes.append((t_spike, local_id + offset))
        return len(self.spikes)

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
        logger.info(f"Starting step {current_step}")
        self.apply_external_drives(t)
        logger.info(f"Applied external drives at step {current_step}")
        self.step_regions(dt, t)
        logger.info(f"Stepped regions at step {current_step}")
        self.collect_spikes()
        logger.info(f"Collected spikes at step {current_step}")
        self.propagate_synapses()
        logger.info(f"Propagated synapses at step {current_step}")

        reward_signal = 1.0 if len(self.spikes) > 0.05 * sum(r.neuron_count for r in self.regions.values()) else 0.0
        self.neuromodulators.step({}, reward_signal)
        logger.info(f"Updated neuromodulators at step {current_step}")

        if current_step % 200 == 0:
            self.log_metrics(current_step)
        if current_step % 1000 == 0 or current_step == self.steps - 1:
            log_simulation_step(current_step, self.steps)
        logger.info(f"Completed step {current_step}")

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

    def compute_region_firing_rates(self, window=200):
        rates = {}
        time_window = window * self.dt
        current_time = self.steps * self.dt
        window_start_time = current_time - time_window
        spikes_in_window = [(t, nid) for (t, nid) in self.spikes if t >= window_start_time]
        region_spike_counts = {name: 0 for name in self.regions}
        for t_spike, nid in spikes_in_window:
            for name, region in self.regions.items():
                offset = region.global_offset
                count = region.neuron_count
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
        for t_spike, gid in self.spikes:
            if t_spike >= window_start_time:
                neuron_spikes.setdefault(gid, []).append(t_spike)
        burst_threshold = 0.01
        neuron_to_region = {}
        for region_name, region in self.regions.items():
            offset = region.global_offset
            count = region.neuron_count
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
