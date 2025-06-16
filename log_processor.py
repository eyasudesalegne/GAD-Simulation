import re
from collections import defaultdict

def process_simulation_logs(raw_log_text, max_chars=4000):
    """
    Processes raw simulation logs to produce a concise, structured summary for API input.

    Args:
        raw_log_text (str): The full multiline raw log text.
        max_chars (int): Max characters allowed in output.

    Returns:
        str: Filtered, organized, and trimmed log summary text.
    """

    lines = raw_log_text.splitlines()

    # Containers for filtered info
    progress_lines = []
    neuromod_lines = []
    treatment_lines = []
    warning_lines = []
    error_lines = []

    spike_counts = []
    firing_rates = defaultdict(list)  # region -> list of rates
    burst_counts = defaultdict(list)  # region -> list of counts
    syn_weights = []

    # Regex patterns for log lines
    patterns = {
        'progress': re.compile(r'Simulation progress|Simulation complete|Step \d+'),
        'neuromod': re.compile(r'Neuromodulator update|DA=|5-HT=|NE=|ACh=|Cort='),
        'treatment': re.compile(r'dosed|Treatment|dosage|intensity|frequency'),
        'warning': re.compile(r'WARNING|Warning|warn'),
        'error': re.compile(r'ERROR|Error|exception|failed', re.I),
        'spikes': re.compile(r'Spikes at step (\d+): (\d+)'),
        'firing_rates': re.compile(r'\[Step (\d+)\] Region firing rates: (.+)'),
        'burst_counts': re.compile(r'\[Step (\d+)\] Region burst counts: (.+)'),
        'syn_weights': re.compile(r'\[Step (\d+)\] Average synaptic weight: ([0-9.]+)'),
    }

    for line in lines:
        l = line.strip()
        if not l:
            continue

        if patterns['error'].search(l):
            error_lines.append(l)
            continue
        if patterns['warning'].search(l):
            warning_lines.append(l)
            continue
        if patterns['treatment'].search(l):
            treatment_lines.append(l)
            continue
        if patterns['progress'].search(l):
            progress_lines.append(l)
            continue
        if patterns['neuromod'].search(l):
            neuromod_lines.append(l)
            continue

        # Parse spikes counts
        m_spikes = patterns['spikes'].search(l)
        if m_spikes:
            step = int(m_spikes.group(1))
            count = int(m_spikes.group(2))
            spike_counts.append((step, count))
            continue

        # Parse firing rates per region
        m_rates = patterns['firing_rates'].search(l)
        if m_rates:
            step = int(m_rates.group(1))
            rates_str = m_rates.group(2)
            # Parse "RegionName: rate Hz, ..."
            for pair in rates_str.split(','):
                try:
                    region, rate = pair.strip().split(':')
                    rate_val = float(rate.strip().split()[0])
                    firing_rates[region.strip()].append((step, rate_val))
                except Exception:
                    continue
            continue

        # Parse burst counts per region
        m_bursts = patterns['burst_counts'].search(l)
        if m_bursts:
            step = int(m_bursts.group(1))
            bursts_str = m_bursts.group(2)
            for pair in bursts_str.split(','):
                try:
                    region, count_str = pair.strip().split(':')
                    count_val = int(count_str.strip())
                    burst_counts[region.strip()].append((step, count_val))
                except Exception:
                    continue
            continue

        # Parse synaptic weights
        m_weights = patterns['syn_weights'].search(l)
        if m_weights:
            step = int(m_weights.group(1))
            w = float(m_weights.group(2))
            syn_weights.append((step, w))
            continue

    # Summarize spike counts
    spike_summary = "No spike count data found."
    if spike_counts:
        avg_spikes = sum(c for _, c in spike_counts) / len(spike_counts)
        spike_summary = f"Average spikes per logged step: {avg_spikes:.1f} (from {len(spike_counts)} entries)"

    # Summarize firing rates (average over last 5 entries per region)
    firing_summary_lines = []
    for region, data in firing_rates.items():
        last_rates = [r for _, r in data[-5:]]
        avg_rate = sum(last_rates) / len(last_rates) if last_rates else 0
        firing_summary_lines.append(f"{region}: {avg_rate:.3f} Hz")
    firing_summary = "\n".join(firing_summary_lines) if firing_summary_lines else "No firing rate data found."

    # Summarize burst counts (average over last 5 entries per region)
    burst_summary_lines = []
    for region, data in burst_counts.items():
        last_counts = [c for _, c in data[-5:]]
        avg_count = sum(last_counts) / len(last_counts) if last_counts else 0
        burst_summary_lines.append(f"{region}: {avg_count:.1f}")
    burst_summary = "\n".join(burst_summary_lines) if burst_summary_lines else "No burst count data found."

    # Summarize synaptic weights (average)
    syn_weight_summary = "No synaptic weight data found."
    if syn_weights:
        avg_weight = sum(w for _, w in syn_weights) / len(syn_weights)
        syn_weight_summary = f"Average synaptic weight: {avg_weight:.4f} (from {len(syn_weights)} entries)"

    # Build final organized summary
    summary_parts = []

    if error_lines:
        summary_parts.append("=== Errors ===")
        summary_parts.extend(error_lines[:5])

    if warning_lines:
        summary_parts.append("=== Warnings ===")
        summary_parts.extend(warning_lines[:5])

    if progress_lines:
        summary_parts.append("=== Simulation Progress ===")
        summary_parts.extend(progress_lines[-10:])

    if neuromod_lines:
        summary_parts.append("=== Neuromodulator Levels ===")
        summary_parts.extend(neuromod_lines[-10:])

    if treatment_lines:
        summary_parts.append("=== Treatments Applied ===")
        summary_parts.extend(treatment_lines)

    summary_parts.append("=== Spike Summary ===")
    summary_parts.append(spike_summary)

    summary_parts.append("=== Firing Rates (avg over last 5 logs) ===")
    summary_parts.append(firing_summary)

    summary_parts.append("=== Burst Counts (avg over last 5 logs) ===")
    summary_parts.append(burst_summary)

    summary_parts.append("=== Synaptic Weights ===")
    summary_parts.append(syn_weight_summary)

    final_text = "\n".join(summary_parts)

    # Trim if too long, keep recent info
    if len(final_text) > max_chars:
        final_text = "[...trimmed...]\n" + final_text[-max_chars:]

    return final_text
