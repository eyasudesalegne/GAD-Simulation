import numpy as np
import matplotlib.pyplot as plt

def smooth_rate(rate, window=50):
    window = min(window, len(rate))
    kernel = np.ones(window) / window
    return np.convolve(rate, kernel, mode='same')

# SIMPLE ANALYSIS
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

def create_simple_plot(sim, ax=None):
    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    else:
        if isinstance(ax, (np.ndarray, list)):
            ax1, ax2 = ax[0], ax[1]
        else:
            fig = ax.figure
            ax1 = ax
            ax2 = fig.add_subplot(212)
            ax = [ax1, ax2]

    rates = simple_firing_rates(sim)
    counts = simple_spike_counts(sim)

    names = sorted(rates.keys())
    firing_vals = [rates[n] for n in names]
    count_vals = [counts[n] for n in names]

    ax[0].clear()
    ax[1].clear()

    ax[0].bar(names, firing_vals, color='skyblue')
    ax[0].set_title("Simple: Mean Firing Rates per Region")
    ax[0].set_ylabel("Firing rate (Hz)")
    ax[0].tick_params(axis='x', rotation=90)

    ax[1].bar(names, count_vals, color='orange')
    ax[1].set_title("Simple: Total Spike Counts per Region")
    ax[1].set_ylabel("Spike count")
    ax[1].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    return ax[0].figure

# INTERMEDIATE ANALYSIS
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
    rate = counts / (bin_size * total_neurons) if total_neurons > 0 else np.zeros_like(counts)
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

def create_intermediate_plot(sim, ax=None):
    if ax is None:
        fig, ax = plt.subplots(3, 1, figsize=(12, 10))
    else:
        if isinstance(ax, (np.ndarray, list)):
            ax1, ax2, ax3 = ax[0], ax[1], ax[2]
        else:
            fig = ax.figure
            ax1 = ax
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)
            ax = [ax1, ax2, ax3]

    isi_stats = intermediate_isi_cv(sim)
    names, means = zip(*[(k, v['mean_isi'] if v['mean_isi'] else 0) for k, v in isi_stats.items()])
    _, cvs = zip(*[(k, v['cv_isi'] if v['cv_isi'] else 0) for k, v in isi_stats.items()])

    bins, rate, smooth = intermediate_firing_rate_dynamics(sim)
    fano = intermediate_fano_factor(sim)

    ax[0].clear()
    ax[1].clear()
    ax[2].clear()

    ax[0].bar(names, means, color='lightgreen')
    ax[0].set_title("Intermediate: Mean ISI per Region")
    ax[0].set_ylabel("Mean ISI (s)")
    ax[0].tick_params(axis='x', rotation=90)

    ax[1].plot(bins, rate, alpha=0.5, label='Raw')
    ax[1].plot(bins, smooth, label='Smoothed')
    ax[1].set_title("Intermediate: Population Firing Rate Over Time")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Firing rate (Hz)")
    ax[1].legend()

    ax[2].bar(["Fano Factor"], [fano if fano is not None else 0], color='salmon')
    ax[2].set_title("Intermediate: Population Spike Count Variability (Fano Factor)")

    plt.tight_layout()
    return ax[0].figure

# ADVANCED ANALYSIS
def advanced_burst_detection(sim, isi_threshold=0.02):  # Increased threshold from 0.01 to 0.02
    print(f"DEBUG: Total spikes in simulation: {len(sim.spikes)}")
    burst_data = {}
    for region_name, region in sim.regions.items():
        print(f"DEBUG: Processing region {region_name}")
        neuron_spikes = {}
        for t, nid in sim.spikes:
            if region.global_offset <= nid < region.global_offset + region.neuron_count:
                neuron_spikes.setdefault(nid, []).append(t)
        print(f"DEBUG: Number of neurons with spikes in {region_name}: {len(neuron_spikes)}")
        burst_times = []
        burst_lengths = []
        for spikes in neuron_spikes.values():
            spikes = sorted(spikes)
            isis = np.diff(spikes)
            bursts_idx = np.where(isis < isi_threshold)[0]
            for idx in bursts_idx:
                burst_times.append(spikes[idx])
                length = 1
                while idx + length < len(isis) and isis[idx + length] < isi_threshold:
                    length += 1
                burst_lengths.append(length + 1)
        burst_data[region_name] = {'burst_count': len(burst_times), 'burst_lengths': burst_lengths}
    return burst_data

def advanced_synaptic_plasticity_trends(sim, window_steps=100):
    print(f"DEBUG: Total synapses in simulation: {len(sim.synapses)}")
    weights_trend = []
    for start in range(0, sim.steps - window_steps, window_steps):
        avg_weight = np.mean([syn.mu for syn in sim.synapses]) if sim.synapses else 0
        weights_trend.append(avg_weight)
    return np.array(weights_trend)

def advanced_neuromodulator_correlation(sim, nm_name='DA', bin_size=0.05):
    nm_data = sim.neuromodulators.history.get(nm_name, [])
    print(f"DEBUG: Neuromodulator '{nm_name}' history length: {len(nm_data)}")
    if not nm_data:
        return None
    bins = np.arange(0, sim.duration + bin_size, bin_size)
    spike_times = np.array([t for t, _ in sim.spikes])
    counts, _ = np.histogram(spike_times, bins=bins)
    rate = counts / bin_size

    times, vals = zip(*nm_data)
    vals_interp = np.interp(bins[:-1], times, vals)
    corr = np.corrcoef(rate, vals_interp)[0, 1] if len(rate) == len(vals_interp) else None
    return corr, bins[:-1], rate, vals_interp

def advanced_entropy(spike_counts):
    spike_probs = spike_counts / np.sum(spike_counts) if np.sum(spike_counts) > 0 else np.ones_like(spike_counts) / len(spike_counts)
    spike_probs = spike_probs[spike_probs > 0]
    return -np.sum(spike_probs * np.log2(spike_probs))

def advanced_spike_entropy(sim, bin_size=0.05):
    bins = np.arange(0, sim.duration + bin_size, bin_size)
    spike_times = np.array([t for t, _ in sim.spikes])
    counts, _ = np.histogram(spike_times, bins=bins)
    return advanced_entropy(counts)

def create_advanced_plot(sim, ax=None):
    if ax is None:
        fig, ax = plt.subplots(4, 1, figsize=(14, 16))
    else:
        if isinstance(ax, (np.ndarray, list)):
            ax1, ax2, ax3, ax4 = ax[0], ax[1], ax[2], ax[3]
        else:
            fig = ax.figure
            ax1 = ax
            ax2 = fig.add_subplot(412)
            ax3 = fig.add_subplot(413)
            ax4 = fig.add_subplot(414)
            ax = [ax1, ax2, ax3, ax4]

    bursts = advanced_burst_detection(sim)
    names = sorted(bursts.keys())
    burst_counts = [bursts[n]['burst_count'] for n in names]
    burst_lengths_avg = [np.mean(bursts[n]['burst_lengths']) if bursts[n]['burst_lengths'] else 0 for n in names]

    weights = advanced_synaptic_plasticity_trends(sim)

    corr_result = advanced_neuromodulator_correlation(sim, 'DA')

    entropy_val = advanced_spike_entropy(sim)

    ax[0].clear()
    ax[1].clear()
    ax[2].clear()
    ax[3].clear()

    ax[0].bar(names, burst_counts, color='mediumslateblue')
    ax[0].set_title("Advanced: Burst Counts per Region")
    ax[0].tick_params(axis='x', rotation=90)

    ax[1].bar(names, burst_lengths_avg, color='cadetblue')
    ax[1].set_title("Advanced: Average Burst Length per Region")
    ax[1].tick_params(axis='x', rotation=90)

    ax[2].plot(weights, color='darkorange')
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
    return ax[0].figure

def run_all_analytics(sim):
    print("Running Simple Analysis...")
    fig1 = create_simple_plot(sim)
    print("Running Intermediate Analysis...")
    fig2 = create_intermediate_plot(sim)
    print("Running Advanced Analysis...")
    fig3 = create_advanced_plot(sim)
    return fig1, fig2, fig3
