import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from brain_sim.neuromodulator import Neuromodulator
from brain_sim.region import BrainRegion, NeuronSubclass
from brain_sim.synapse import Synapse  # Your synapse module
from brain_sim.connectome_data import connectome

def setup_network_from_connectome(connectome):
    region_names = set(connectome.keys())
    for targets in connectome.values():
        region_names.update(targets.keys())
    region_names = sorted(region_names)
    region_to_idx = {name: i for i, name in enumerate(region_names)}

    diffusion_matrix = np.eye(len(region_names)) * 0.5
    neuromod = Neuromodulator(dt=0.01, duration=5.0, compartments=region_names, diffusion_matrix=diffusion_matrix)

    brain_regions = {}
    base_params_exc = {
        'a_mean': 0.02, 'a_std': 0.005,
        'b_mean': 0.2, 'b_std': 0.02,
        'c_mean': -65, 'c_std': 3,
        'd_mean': 8, 'd_std': 1,
        'noise_amp': 0.02, 'noise_rate': 0.3
    }
    base_params_inh = {
        'a_mean': 0.1, 'a_std': 0.01,
        'b_mean': 0.25, 'b_std': 0.02,
        'c_mean': -65, 'c_std': 3,
        'd_mean': 2, 'd_std': 0.5,
        'noise_amp': 0.015, 'noise_rate': 0.4
    }

    for region_name in region_names:
        region = BrainRegion(region_name)
        neuron_type = 'excitatory' if region_name.endswith('_E') else 'inhibitory'
        base_params = base_params_exc if neuron_type == 'excitatory' else base_params_inh

        neuron_count = 400
        compartment_idx = region_to_idx[region_name]

        population = NeuronSubclass(neuron_type, neuron_count, base_params, neuromod, compartment_idx)
        region.add_population(population)
        brain_regions[region_name] = region

    synapses = []
    for pre_region, targets in connectome.items():
        for post_region, params in targets.items():
            pre_pop = brain_regions[pre_region].populations[0]
            post_pop = brain_regions[post_region].populations[0]
            weight = params['weight']
            plasticity = params['plasticity']
            delay_jitter = params['delay_jitter']
            delay_steps = int(np.round(delay_jitter))

            is_inhibitory = pre_region.endswith('_I')

            syn = Synapse(pre_pop, post_pop, weight, delay_steps, is_inhibitory, neuromod, plasticity)
            brain_regions[pre_region].add_synapse(syn)
            synapses.append(syn)

    return neuromod, brain_regions, region_to_idx, synapses

def run_simulation(neuromod, brain_regions, region_to_idx, synapses, dt=0.01, duration=5.0):
    steps = int(duration / dt)
    times = []
    da_levels = []
    ht_levels = []
    spike_counts = []

    # Spike time buffers per population for plasticity
    spike_buffers = {syn: {'pre': deque(), 'post': deque()} for syn in synapses}
    max_plasticity_window = 0.05  # 50 ms plasticity window

    for step in range(steps):
        t = step * dt

        external_inputs = {
            'DA': np.random.uniform(0.1, 0.3, len(region_to_idx)),
            '5-HT': np.random.uniform(0.1, 0.3, len(region_to_idx)),
            'NE': np.random.uniform(0.05, 0.2, len(region_to_idx)),
            'ACh': np.random.uniform(0.05, 0.2, len(region_to_idx)),
            'GABA': np.random.uniform(0.05, 0.2, len(region_to_idx))
        }
        stress_input = 0.1

        neuromod.step(external_inputs, stress_input)

        for region in brain_regions.values():
            region.step(dt, t)

        # Process synapses: propagate and update plasticity using spike timing
        for syn in synapses:
            syn.propagate()

            # Gather recent spikes within plasticity window
            pre_spikes = [sp for sp in syn.pre.spikes if t - max_plasticity_window <= sp[0] <= t]
            post_spikes = [sp for sp in syn.post.spikes if t - max_plasticity_window <= sp[0] <= t]

            # Update plasticity for every pre-post spike pair in window
            for pre_spike in pre_spikes:
                for post_spike in post_spikes:
                    dt_pre = pre_spike[0]
                    dt_post = post_spike[0]
                    syn.update_plasticity(dt_pre, dt_post)

            # Optionally prune/grow synapses periodically
            if step % 100 == 0:
                syn.remodel_topology()

        sample_region = "LA_E"
        idx = region_to_idx[sample_region]

        times.append(t)
        da_levels.append(neuromod.concentrations['DA'][idx])
        ht_levels.append(neuromod.concentrations['5-HT'][idx])

        total_spikes = sum(len(pop.spikes) for pop in brain_regions[sample_region].populations)
        spike_counts.append(total_spikes)

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(times, da_levels, label='Dopamine LA_E')
    plt.plot(times, ht_levels, label='Serotonin LA_E')
    plt.xlabel('Time (s)')
    plt.ylabel('Neuromodulator Level')
    plt.title('Neuromodulator Dynamics in LA_E')
    plt.legend()

    plt.figure(figsize=(6, 4))
    plt.plot(times, np.cumsum(spike_counts))
    plt.xlabel('Time (s)')
    plt.ylabel('Cumulative Spikes')
    plt.title('Spiking Activity in LA_E')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    neuromod, brain_regions, region_to_idx, synapses = setup_network_from_connectome(connectome)
    run_simulation(neuromod, brain_regions, region_to_idx, synapses)
