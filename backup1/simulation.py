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

class Simulation:
    """
    Brain network simulation integrating region-specific Izhikevich neurons,
    neuromodulation, heterogeneous synapses, and realistic connectome.
    Includes detailed logging and progress reporting hooks.
    """

    def __init__(self, dt=0.001, duration=10.0):
        self.dt = dt
        self.duration = duration
        self.steps = int(duration / dt)

        # Connectivity parameters (modifiable externally)
        self.conduction_velocity = 10.0  # m/s typical for myelinated axons
        self.weight_scale = 1.0
        self.p_base_default = 0.3  # baseline connection probability
        self.lambda_dist = 5.0     # spatial decay constant in mm

        self.regions = {}   # name -> BrainRegion instance
        self.synapses = []  # list of Synapse instances
        self.spikes = []    # list of (time, global neuron id) tuples
        self.connections = []  # metadata for connections

        # Global neuromodulator system
        self.neuromodulators = Neuromodulator(dt, duration)

        # Define region-specific external drives (can be extended/overridden)
        self.external_inputs = {
            "LA_E": lambda t: 0.02 * np.sin(2 * np.pi * 6 * t),
            "mPFC_E": lambda t: 0.01 * np.sin(2 * np.pi * 10 * t),
            "BLA_E": lambda t: 0.05 if (t % 0.2) < 0.05 else 0.0,
            "DG_E": lambda t: 0.03 * np.sin(2 * np.pi * 8 * t) + 0.01 * np.sin(2 * np.pi * 12 * t),
        }

        logger.info(f"🧠 Simulation initialized with dt={dt}, duration={duration}s, steps={self.steps}")
        self.build_from_connectome(connectome)

    def add_region(self, region):
        """
        Add a brain region to the simulation and assign global neuron offset.
        """
        region_offset = sum(r.neuron_count for r in self.regions.values())
        region.global_offset = region_offset
        self.regions[region.name] = region
        log_region_added(region.name, region.neuron_count)

    def build_from_connectome(self, conn_dict):
        """
        Build network regions and synapses from connectome dictionary.
        """
        # Collect unique region names from source and targets
        region_names = set(conn_dict.keys())
        for targets in conn_dict.values():
            region_names.update(targets.keys())

        # Create BrainRegion objects for all regions not already added
        for name in sorted(region_names):
            if name not in self.regions:
                try:
                    base_name, neuron_suffix = name.rsplit("_", 1)
                except ValueError:
                    # fallback if region name format unexpected
                    base_name, neuron_suffix = name, "E"
                is_exc = (neuron_suffix == "E")
                density = 4000 if is_exc else 1000  # neurons per mm^3 approx
                volume = 1000  # mm^3 - fixed or could be region-specific
                neuron_type = "excitatory" if is_exc else "inhibitory"
                region = BrainRegion(name, density, volume, neuron_type, self.neuromodulators)
                self.add_region(region)

        # Connect regions based on connectome entries
        syn_count = 0
        for src, targets in conn_dict.items():
            for tgt, conn_params in targets.items():
                created = self.connect(src, tgt, conn_params)
                if created:
                    syn_count += 1

        logger.info(f"✅ Added {len(self.regions)} regions and {syn_count} synapses.")

    def connect(self, src_name, tgt_name, conn_params):
        """
        Create synaptic connection between regions with parameters.
        Returns True if connection created.
        """
        if src_name not in self.regions or tgt_name not in self.regions:
            logger.warning(f"⚠️ Connection skipped: {src_name} or {tgt_name} not found.")
            return False

        src_region = self.regions[src_name]
        tgt_region = self.regions[tgt_name]

        weight = conn_params.get("weight", 0.5)
        plasticity_factor = conn_params.get("plasticity", 1.0)
        delay_jitter = conn_params.get("delay_jitter", 5)  # ms

        # Compute anatomical distance between region centroids
        dist = np.linalg.norm(src_region.centroid - tgt_region.centroid)
        # Random factor simulating myelination variation
        myelination_factor = np.random.uniform(0.5, 1.5)
        delay_ms = (dist / (self.conduction_velocity * myelination_factor)) * 1000  # convert s to ms
        delay_ms += np.random.normal(0, delay_jitter)
        delay_ms = max(1.0, delay_ms)  # min 1 ms delay
        delay_steps = max(1, int(delay_ms / (self.dt * 1000)))  # convert ms to steps

        # Connection probability with spatial decay and DA modulation
        prob = self.p_base_default * np.exp(-dist / self.lambda_dist)
        prob *= np.random.uniform(0.9, 1.1)  # small random variation
        DA = self.neuromodulators.state.get('DA', 0)
        prob *= (1 + 0.5 * DA)  # dopamine increases connection probability

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

    def step(self, dt, t):
        """
        Perform one simulation timestep:
        - Apply external drives
        - Step each brain region neurons
        - Propagate synapses
        - Update neuromodulators
        - Record spikes
        - Log relevant rich parameters every 200 steps
        """
        step_spikes = 0
        total_neurons = sum(r.neuron_count for r in self.regions.values())
        current_step = int(t / dt)

        for region in self.regions.values():
            # Add external drive if defined for this region
            if region.name in self.external_inputs:
                region.I += self.external_inputs[region.name](t)

            region.step(dt, t)

            # Collect spikes with global neuron ID offset
            for spike_time, local_id in region.spikes:
                self.spikes.append((spike_time, local_id + region.global_offset))
                step_spikes += 1
            region.spikes.clear()

        # Propagate synaptic inputs with delays
        for syn in self.synapses:
            syn.propagate()

        # Calculate global firing rate (spikes per neuron per second)
        global_firing_rate = step_spikes / (total_neurons * dt) if total_neurons > 0 else 0

        # Reward signal logic (e.g., above threshold firing triggers reward)
        reward_signal = 1.0 if step_spikes > (0.05 * total_neurons) else 0.0

        # Update neuromodulator states with adaptive decay and external inputs
        self.neuromodulators.update(current_step, global_firing_rate, reward_signal)

        # Log detailed parameters every 200 steps to reduce overhead
        if current_step % 200 == 0:
            # Log neuromodulator levels
            log_neuromodulator_update(current_step, self.neuromodulators.state)

            # Log total spikes this step
            log_spike_count(current_step, step_spikes)

            # Compute region-specific firing rates and log
            region_rates = self.compute_region_firing_rates(window=200)
            log_region_firing_rates(current_step, region_rates)

            # Compute burst statistics and log
            burst_counts = self.compute_burst_statistics(window=200)
            log_region_burst_counts(current_step, burst_counts)

            # Compute and log average synaptic weight
            avg_weight = np.mean([syn.mu if not syn.is_inhibitory else syn.w_GABA for syn in self.synapses])
            log_average_synaptic_weight(current_step, avg_weight)

        # Log simulation progress every 1000 steps to reduce clutter
        if current_step % 1000 == 0 or current_step == self.steps - 1:
            log_simulation_step(current_step, self.steps)

    def run(self):
        """
        Run full simulation from t=0 to t=duration with dt timestep.
        Logs progress and performs spike pattern analysis after.
        """
        logger.info("🚀 Running full simulation...")
        t = 0.0
        for step in range(self.steps):
            self.step(self.dt, t)
            t += self.dt

        logger.info(f"✅ Simulation complete. Total spikes: {len(self.spikes)}")
        self.analyze_spike_patterns()

    def run_incremental(self, chunk_size=200):
        """
        Run simulation in chunks to support UI responsiveness.
        Yields progress as fraction (0 to 1).
        """
        logger.info("🚀 Running simulation incrementally...")
        t = 0.0
        for start_step in range(0, self.steps, chunk_size):
            end_step = min(start_step + chunk_size, self.steps)
            for step in range(start_step, end_step):
                self.step(self.dt, t)
                t += self.dt
            log_simulation_step(end_step, self.steps)
            yield end_step / self.steps

        logger.info(f"✅ Incremental simulation complete. Total spikes: {len(self.spikes)}")
        self.analyze_spike_patterns()

    def analyze_spike_patterns(self):
        """
        Analyze spike trains for burst counts, ISI stats, and log results.
        """
        logger.info("🔍 Analyzing spike patterns...")
        neuron_spikes = {}
        for t, gid in self.spikes:
            neuron_spikes.setdefault(gid, []).append(t)

        all_isis = []
        burst_counts = []
        burst_threshold = 0.01  # 10 ms for burst detection

        for gid, times in neuron_spikes.items():
            times = np.sort(times)
            if len(times) < 2:
                continue
            isis = np.diff(times)
            all_isis.extend(isis)

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
            burst_counts.append(bursts)

        if all_isis:
            mean_isi = np.mean(all_isis)
            std_isi = np.std(all_isis)
        else:
            mean_isi = std_isi = 0

        total_bursts = sum(burst_counts)
        avg_bursts_per_neuron = total_bursts / len(neuron_spikes) if neuron_spikes else 0

        logger.info(f"⏱️ Mean ISI: {mean_isi:.4f} s, Std ISI: {std_isi:.4f} s")
        logger.info(f"🔥 Total bursts detected: {total_bursts}, Average bursts per neuron: {avg_bursts_per_neuron:.2f}")

    def compute_region_firing_rates(self, window=200):
        """
        Compute approximate firing rates (Hz) for each region over the last 'window' steps.
        """
        rates = {}
        # Compute time window size in seconds
        time_window = window * self.dt

        # Current simulation time is based on total steps * dt
        current_time = self.steps * self.dt
        window_start_time = current_time - time_window

        # Filter spikes in time window
        spikes_in_window = [(t, nid) for (t, nid) in self.spikes if t >= window_start_time]

        # Count spikes per region
        region_spike_counts = {name: 0 for name in self.regions}
        for t, nid in spikes_in_window:
            for name, region in self.regions.items():
                if region.global_offset <= nid < region.global_offset + region.neuron_count:
                    region_spike_counts[name] += 1
                    break

        # Compute firing rate: spikes / (neurons * time window)
        for name, count in region_spike_counts.items():
            n_neurons = self.regions[name].neuron_count
            rate = count / (n_neurons * time_window) if n_neurons > 0 else 0
            rates[name] = rate

        return rates

    def compute_burst_statistics(self, window=200):
        """
        Compute burst counts per region over last 'window' steps.
        """
        bursts_per_region = {name: 0 for name in self.regions}

        time_window = window * self.dt
        current_time = self.steps * self.dt
        window_start_time = current_time - time_window

        # Group spikes per neuron in window
        neuron_spikes = {}
        for t, gid in self.spikes:
            if t >= window_start_time:
                neuron_spikes.setdefault(gid, []).append(t)

        burst_threshold = 0.01  # 10 ms ISI for burst detection

        # Map neurons to regions
        neuron_to_region = {}
        for region_name, region in self.regions.items():
            for nid in range(region.global_offset, region.global_offset + region.neuron_count):
                neuron_to_region[nid] = region_name

        # Count bursts per neuron and accumulate per region
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
