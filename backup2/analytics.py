import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.stats import entropy

def smooth_rate(rate, window=50):
    window = min(window, len(rate))
    kernel = np.ones(window) / window
    return np.convolve(rate, kernel, mode='same')

####################
# SIMPLE ANALYSIS (ENRICHED)
####################
def simple_firing_rates(sim):
    rates = {}
    for region_name, region in sim.regions.items():
        spike_times = [t for t, nid in sim.spikes if region.global_offset <= nid < region.global_offset + region.neuron_count]
        rate = len(spike_times) / sim.duration if sim.duration > 0 else 0
        rates[region_name] = rate
    return rates

def simple_spike_counts(sim):
    counts = {}
    for region_name, region in sim.regions.items():
        counts[region_name] = sum(1 for _, nid in sim.spikes if region.global_offset <= nid < region.global_offset + region.neuron_count)
    return counts

def simple_plot(sim):
    rates = simple_firing_rates(sim)
    counts = simple_spike_counts(sim)

    names = sorted(rates.keys())
    firing_vals = [rates[n] for n in names]
    count_vals = [counts[n] for n in names]

    fig, ax = plt.subplots(2,1, figsize=(12,8))
    ax[0].bar(names, firing_vals)
    ax[0].set_title("Simple: Mean Firing Rates per Region")
    ax[0].set_ylabel("Firing rate (Hz)")
    ax[0].tick_params(axis='x', rotation=90)

    ax[1].bar(names, count_vals)
    ax[1].set_title("Simple: Total Spike Counts per Region")
    ax[1].set_ylabel("Spike count")
    ax[1].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()

####################
# INTERMEDIATE ANALYSIS (ENRICHED)
####################
def intermediate_isi_cv(sim):
    stats = {}
    for region_name, region in sim.regions.items():
        spike_times = sorted([t for t, nid in sim.spikes if region.global_offset <= nid < region.global_offset + region.neuron_count])
        if len(spike_times) < 2:
            stats[region_name] = {'mean_isi': None, 'cv_isi': None}
            continue
        isis = np.diff(spike_times)
        mean_isi = np.mean(isis)
        cv_isi = np.std(isis) / mean_isi if mean_isi > 0 else None
        stats[region_name] = {'mean_isi': mean_isi, 'cv_isi': cv_isi}
    return stats

def intermediate_firing_rate_dynamics(sim, bin_size=0.05):
    total_neurons = sum(r.neuron_count for r in sim.regions.values())
    bins = np.arange(0, sim.duration + bin_size, bin_size)
    spike_times = np.array([t for t, _ in sim.spikes])
    counts, _ = np.histogram(spike_times, bins=bins)
    rate = counts / (bin_size * total_neurons)
    smooth = smooth_rate(rate, window=5)
    return bins[:-1], rate, smooth

def intermediate_fano_factor(sim, bin_size=0.05):
    total_neurons = sum(r.neuron_count for r in sim.regions.values())
    bins = np.arange(0, sim.duration + bin_size, bin_size)
    spike_times = np.array([t for t, _ in sim.spikes])
    counts, _ = np.histogram(spike_times, bins=bins)
    mean_count = np.mean(counts)
    var_count = np.var(counts)
    fano = var_count / mean_count if mean_count > 0 else None
    return fano

def intermediate_plot(sim):
    isi_stats = intermediate_isi_cv(sim)
    names, means = zip(*[(k, v['mean_isi'] if v['mean_isi'] else 0) for k, v in isi_stats.items()])
    _, cvs = zip(*[(k, v['cv_isi'] if v['cv_isi'] else 0) for k, v in isi_stats.items()])

    bins, rate, smooth = intermediate_firing_rate_dynamics(sim)
    fano = intermediate_fano_factor(sim)

    fig, ax = plt.subplots(3,1, figsize=(12,10))

    ax[0].bar(names, means)
    ax[0].set_title("Intermediate: Mean ISI per Region")
    ax[0].set_ylabel("Mean ISI (s)")
    ax[0].tick_params(axis='x', rotation=90)

    ax[1].plot(bins, rate, alpha=0.5, label='Raw')
    ax[1].plot(bins, smooth, label='Smoothed')
    ax[1].set_title("Intermediate: Population Firing Rate Over Time")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Firing rate (Hz)")
    ax[1].legend()

    ax[2].bar(["Fano Factor"], [fano if fano is not None else 0])
    ax[2].set_title("Intermediate: Population Spike Count Variability (Fano Factor)")
    plt.tight_layout()
    plt.show()

####################
# ADVANCED ANALYSIS (ENRICHED)
####################
def advanced_burst_detection(sim, isi_threshold=0.01):
    burst_data = {}
    for region_name, region in sim.regions.items():
        neuron_spikes = {}
        for t, nid in sim.spikes:
            if region.global_offset <= nid < region.global_offset + region.neuron_count:
                neuron_spikes.setdefault(nid, []).append(t)
        burst_times = []
        burst_lengths = []
        for spikes in neuron_spikes.values():
            spikes = sorted(spikes)
            isis = np.diff(spikes)
            bursts_idx = np.where(isis < isi_threshold)[0]
            for idx in bursts_idx:
                burst_times.append(spikes[idx])
                length = 1
                # Count consecutive ISIs below threshold as burst length
                while idx+length < len(isis) and isis[idx+length] < isi_threshold:
                    length += 1
                burst_lengths.append(length+1)
        burst_data[region_name] = {'burst_count': len(burst_times), 'burst_lengths': burst_lengths}
    return burst_data

def advanced_synaptic_plasticity_trends(sim, window_steps=100):
    weights_trend = []
    for start in range(0, sim.steps - window_steps, window_steps):
        avg_weight = np.mean([syn.mu for syn in sim.synapses]) if sim.synapses else 0
        weights_trend.append(avg_weight)
    return np.array(weights_trend)

def advanced_neuromodulator_correlation(sim, nm_name='DA', bin_size=0.05):
    bins = np.arange(0, sim.duration + bin_size, bin_size)
    spike_times = np.array([t for t, _ in sim.spikes])
    counts, _ = np.histogram(spike_times, bins=bins)
    rate = counts / bin_size

    nm_data = sim.neuromodulators.history.get(nm_name, [])
    if not nm_data:
        return None
    times, vals = zip(*nm_data)
    vals_interp = np.interp(bins[:-1], times, vals)
    corr = np.corrcoef(rate, vals_interp)[0,1] if len(rate) == len(vals_interp) else None
    return corr, bins[:-1], rate, vals_interp

def advanced_entropy(spike_counts):
    spike_probs = spike_counts / np.sum(spike_counts) if np.sum(spike_counts) > 0 else np.ones_like(spike_counts)/len(spike_counts)
    spike_probs = spike_probs[spike_probs > 0]
    return -np.sum(spike_probs * np.log2(spike_probs))

def advanced_spike_entropy(sim, bin_size=0.05):
    bins = np.arange(0, sim.duration + bin_size, bin_size)
    spike_times = np.array([t for t, _ in sim.spikes])
    counts, _ = np.histogram(spike_times, bins=bins)
    return advanced_entropy(counts)

def advanced_plot(sim):
    bursts = advanced_burst_detection(sim)
    names = sorted(bursts.keys())
    burst_counts = [bursts[n]['burst_count'] for n in names]
    burst_lengths_avg = [np.mean(bursts[n]['burst_lengths']) if bursts[n]['burst_lengths'] else 0 for n in names]

    weights = advanced_synaptic_plasticity_trends(sim)

    corr_result = advanced_neuromodulator_correlation(sim, 'DA')

    entropy_val = advanced_spike_entropy(sim)

    fig, ax = plt.subplots(4,1, figsize=(14,16))

    ax[0].bar(names, burst_counts)
    ax[0].set_title("Advanced: Burst Counts per Region")
    ax[0].tick_params(axis='x', rotation=90)

    ax[1].bar(names, burst_lengths_avg)
    ax[1].set_title("Advanced: Average Burst Length per Region")
    ax[1].tick_params(axis='x', rotation=90)

    ax[2].plot(weights)
    ax[2].set_title("Advanced: Synaptic Weight Trends Over Time")

    if corr_result:
        corr, times, rates, vals = corr_result
        ax[3].plot(times, rates, label='Firing Rate')
        ax[3].plot(times, vals, label='DA Level')
        ax[3].set_title(f"Advanced: DA & Firing Rate Correlation = {corr:.2f}, Spike Train Entropy = {entropy_val:.2f}")
        ax[3].legend()
    else:
        ax[3].text(0.5, 0.5, "Neuromodulator data unavailable", ha='center', va='center')
        ax[3].axis('off')

    plt.tight_layout()
    plt.show()

####################
# RUN ALL
####################
def run_all_analytics(sim):
    print("Running Simple Analysis...")
    simple_plot(sim)
    print("Running Intermediate Analysis...")
    intermediate_plot(sim)
    print("Running Advanced Analysis...")
    advanced_plot(sim)
