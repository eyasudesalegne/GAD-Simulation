import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from brain_sim.logger import logger

def compute_rate(spikes, duration, dt, N):
    """
    Compute instantaneous firing rate (Hz) over time from spike data.

    spikes: list of (time, neuron_id) tuples
    duration: total simulation time in seconds
    dt: bin size in seconds (time resolution)
    N: total number of neurons
    """
    bins = np.arange(0, duration + dt, dt)
    times = np.array([t for t, _ in spikes])
    counts, _ = np.histogram(times, bins)
    rate = counts / (dt * N)  # Normalize by bin width and neuron count
    return rate, bins[:-1]  # return rate and bin edges (start of bins)

def compute_cross_correlation(spike_times_A, spike_times_B, duration, dt, max_lag=0.5):
    """
    Compute cross-correlation function of spike times between two regions.

    spike_times_A, spike_times_B: arrays of spike times (seconds)
    duration: total simulation time (seconds)
    dt: bin size for histogram
    max_lag: maximum lag time (seconds) for correlation window
    """
    bins = np.arange(0, duration + dt, dt)
    countsA, _ = np.histogram(spike_times_A, bins)
    countsB, _ = np.histogram(spike_times_B, bins)

    countsA_centered = countsA - np.mean(countsA)
    countsB_centered = countsB - np.mean(countsB)

    corr = np.correlate(countsA_centered, countsB_centered, mode='full')
    lags = np.arange(-len(countsA) + 1, len(countsA)) * dt

    # Limit to max_lag window around zero lag
    lag_mask = (lags >= -max_lag) & (lags <= max_lag)
    return corr[lag_mask], lags[lag_mask]

def compute_region_spike_times(spikes, sim, region_name):
    """
    Extract spike times corresponding to a single region.
    """
    if region_name not in sim.regions:
        logger.warning(f"Region '{region_name}' not found in simulation.")
        return np.array([])

    region = sim.regions[region_name]
    offset = region.global_offset
    size = region.neuron_count

    return np.array([t for t, nid in spikes if offset <= nid < offset + size])

def plot_raster(sim, ax, max_neurons_per_region=100):
    """
    Raster plot with neurons grouped and colored by region.
    Limits neurons per region plotted for clarity.
    """
    ax.clear()
    if not sim.spikes:
        ax.text(0.5, 0.5, "No spikes to plot.", ha='center', va='center')
        return ax

    region_names = sorted(sim.regions.keys(), key=lambda n: sim.regions[n].global_offset)
    cmap = cm.get_cmap('tab20', len(region_names))
    region_colors = {name: cmap(i) for i, name in enumerate(region_names)}

    spike_x = []
    spike_y = []
    spike_colors = []

    for t, nid in sim.spikes:
        for idx, name in enumerate(region_names):
            region = sim.regions[name]
            if region.global_offset <= nid < region.global_offset + region.neuron_count:
                local_id = nid - region.global_offset
                if local_id < max_neurons_per_region:
                    spike_x.append(t)
                    spike_y.append(idx * max_neurons_per_region + local_id)
                    spike_colors.append(region_colors[name])
                break

    ax.scatter(spike_x, spike_y, marker='|', s=10, c=spike_colors)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron index (grouped by region)")
    ax.set_title("Raster Plot")
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    # Add horizontal separators between regions
    for i in range(len(region_names) + 1):
        ax.axhline(i * max_neurons_per_region, color='gray', linestyle='--', linewidth=0.5)

    # Add legend
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=region_colors[name], label=name) for name in region_names]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', title='Regions')

    return ax

def plot_spike_heatmap(sim, bin_size=0.05, ax=None):
    """
    Heatmap of spike counts over time per region.

    bin_size: time bin size in seconds for counting spikes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.clear()
    region_names = sorted(sim.regions.keys(), key=lambda n: sim.regions[n].global_offset)
    num_regions = len(region_names)
    num_bins = int(sim.duration / bin_size)
    heatmap = np.zeros((num_regions, num_bins))

    for t, nid in sim.spikes:
        for idx, name in enumerate(region_names):
            region = sim.regions[name]
            if region.global_offset <= nid < region.global_offset + region.neuron_count:
                bin_idx = int(t / bin_size)
                if bin_idx < num_bins:
                    heatmap[idx, bin_idx] += 1
                break

    im = ax.imshow(heatmap, aspect='auto', cmap='hot', extent=[0, sim.duration, 0, num_regions])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Region")
    ax.set_yticks(np.arange(num_regions) + 0.5)
    ax.set_yticklabels(region_names)
    ax.set_title("Spike Heatmap by Region")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label("Spike Count")

    return ax

def analyze_burst_statistics(sim, burst_threshold=0.01, ax=None):
    """
    Analyze burst counts per neuron using ISI thresholding.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.clear()

    neuron_spikes = {}
    for t, nid in sim.spikes:
        neuron_spikes.setdefault(nid, []).append(t)

    burst_counts = []

    for nid, times in neuron_spikes.items():
        times = np.sort(times)
        if len(times) < 2:
            burst_counts.append(0)
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
        burst_counts.append(bursts)

    avg_bursts = np.mean(burst_counts) if burst_counts else 0

    ax.hist(burst_counts, bins=30, color='skyblue', edgecolor='black')
    ax.set_xlabel("Burst Count per Neuron")
    ax.set_ylabel("Number of Neurons")
    ax.set_title(f"Burst Count Distribution (Threshold: {burst_threshold}s)\nAverage bursts/neuron: {avg_bursts:.2f}")

    return avg_bursts, ax

def analyze_isi(sim, bin_size=0.001, ax=None):
    """
    Compute and plot inter-spike interval (ISI) distribution.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.clear()

    neuron_spikes = {}
    for t, nid in sim.spikes:
        neuron_spikes.setdefault(nid, []).append(t)

    all_isis = []
    for nid, times in neuron_spikes.items():
        times = np.sort(times)
        if len(times) < 2:
            continue
        isis = np.diff(times)
        all_isis.extend(isis)

    if all_isis:
        mean_isi = np.mean(all_isis)
        std_isi = np.std(all_isis)
    else:
        mean_isi = std_isi = 0

    ax.hist(all_isis, bins=int((max(all_isis) / bin_size) if all_isis else 50), color='lightgreen', edgecolor='black')
    ax.set_xlabel("Inter-Spike Interval (s)")
    ax.set_ylabel("Count")
    ax.set_title(f"ISI Distribution\nMean: {mean_isi:.4f}s, Std: {std_isi:.4f}s")

    return (mean_isi, std_isi), ax

def plot_all_metrics(sim, bin_size=0.001, ax=None):
    """
    Plot overall firing rate over time.
    Can be expanded for a dashboard.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    ax.clear()

    total_neurons = sum(r.neuron_count for r in sim.regions.values())
    rate, bins = compute_rate(sim.spikes, sim.duration, bin_size, total_neurons)

    time_axis = bins + bin_size / 2
    ax.plot(time_axis, rate)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Firing Rate (Hz)")
    ax.set_title("Population Firing Rate Over Time")

    return ax

def plot_cross_correlation(sim, regionA, regionB, bin_size=0.005, max_lag=0.5, ax=None):
    """
    Plot cross-correlation of spike trains between two regions.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.clear()

    spikesA = compute_region_spike_times(sim.spikes, sim, regionA)
    spikesB = compute_region_spike_times(sim.spikes, sim, regionB)

    if len(spikesA) == 0 or len(spikesB) == 0:
        ax.text(0.5, 0.5, "No spikes found for one or both regions.", ha='center', va='center')
        return ax

    corr, lags = compute_cross_correlation(spikesA, spikesB, sim.duration, bin_size, max_lag)
    ax.plot(lags, corr)
    ax.set_xlabel("Lag (s)")
    ax.set_ylabel("Correlation")
    ax.set_title(f"Cross-Correlation: {regionA} vs {regionB}")

    return ax

def plot_neuromodulators(sim, ax=None):
    """
    Plot neuromodulator levels over simulation time.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.clear()

    if not hasattr(sim, 'neuromodulators') or not sim.neuromodulators.history:
        ax.text(0.5, 0.5, "Neuromodulator data unavailable.", ha='center', va='center')
        return ax

    for nm, values in sim.neuromodulators.history.items():
        times, vals = zip(*values)
        ax.plot(times, vals, label=nm)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Level")
    ax.set_title("Neuromodulator Levels Over Time")
    ax.legend()

    return ax

def plot_power_spectrum(sim, bin_size=0.001, ax=None):
    """
    Power spectrum of population firing rate.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.clear()

    total_neurons = sum(r.neuron_count for r in sim.regions.values())
    rate, _ = compute_rate(sim.spikes, sim.duration, bin_size, total_neurons)

    fft_vals = np.fft.fft(rate)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.fftfreq(len(rate), d=bin_size)

    mask = freqs >= 0
    ax.plot(freqs[mask], power[mask])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title("Power Spectrum of Firing Rate")

    return ax

def plot_autocorrelation(sim, bin_size=0.001, ax=None):
    """
    Autocorrelation of population firing rate.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.clear()

    total_neurons = sum(r.neuron_count for r in sim.regions.values())
    rate, _ = compute_rate(sim.spikes, sim.duration, bin_size, total_neurons)

    mean_rate = np.mean(rate)
    autocorr = np.correlate(rate - mean_rate, rate - mean_rate, mode='full')
    lags = np.arange(-len(rate) + 1, len(rate)) * bin_size

    ax.plot(lags, autocorr)
    ax.set_xlabel("Lag (s)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelation of Firing Rate")

    return ax

def plot_synaptic_weight_distribution(sim, ax=None):
    """
    Histogram of synaptic weights in the network.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.clear()

    if not hasattr(sim, 'synapses') or not sim.synapses:
        ax.text(0.5, 0.5, "No synapse data available.", ha='center', va='center')
        return ax

    weights = []
    for syn in sim.synapses:
        if syn.is_inhibitory:
            weights.append(syn.w_GABA)
        else:
            weights.append(syn.mu)

    ax.hist(weights, bins=30, color='purple', edgecolor='black')
    ax.set_xlabel("Synaptic Weight")
    ax.set_ylabel("Count")
    ax.set_title("Synaptic Weight Distribution")

    return ax
