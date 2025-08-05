import sys
import threading
from datetime import datetime
import numpy as np

class Logger:
    LEVELS = {
        'DEBUG': '[DEBUG]',
        'INFO': '[INFO]',
        'WARNING': '[WARNING]',
        'ERROR': '[ERROR]'
    }

    # Add a level order for filtering
    LEVEL_ORDER = {
        'DEBUG': 10,
        'INFO': 20,
        'WARNING': 30,
        'ERROR': 40
    }

    def __init__(self, level='INFO'):
        self.lock = threading.Lock()
        self.gui_callback = None
        self.level = level.upper()

    def set_level(self, level):
        self.level = level.upper()

    def set_gui_callback(self, callback):
        self.gui_callback = callback

    def _format_message(self, message, level='INFO'):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        level_tag = self.LEVELS.get(level.upper(), '[INFO]')
        return f"{timestamp} {level_tag} {message}"

    def log(self, message, level='INFO'):
        if self.LEVEL_ORDER[level.upper()] < self.LEVEL_ORDER[self.level]:
            return  # Ignore logs below the set level

        formatted = self._format_message(message, level)
        with self.lock:
            print(formatted, file=sys.stdout)
            sys.stdout.flush()
            if self.gui_callback:
                try:
                    self.gui_callback(formatted)
                except Exception as e:
                    print(f"[Logger] Error sending log to GUI callback: {e}", file=sys.stderr)

    def debug(self, message):
        self.log(message, 'DEBUG')

    def info(self, message):
        self.log(message, 'INFO')

    def warning(self, message):
        self.log(message, 'WARNING')

    def error(self, message):
        self.log(message, 'ERROR')


logger = Logger(level='DEBUG')  # Default to DEBUG level to see all logs


def log_simulation_step(step, total_steps):
    percent = (step / total_steps) * 100 if total_steps else 0
    logger.info(f"Simulation progress: Step {step}/{total_steps} ({percent:.1f}%)")

def log_region_added(region_name, neuron_count):
    logger.info(f"Region added: {region_name} with {neuron_count} neurons")

def log_synapse_created(src, tgt, weight, delay_ms):
    logger.info(f"Synapse created: {src} → {tgt} | weight={weight:.3f}, delay={delay_ms:.2f} ms")

def log_neuromodulator_update(step, neuromod_state):
    state_str = ", ".join(f"{k}={v if not hasattr(v, '__len__') else np.mean(v):.3f}" for k, v in neuromod_state.items())
    logger.info(f"Neuromodulator update at step {step}: {state_str}")

def log_spike_count(step, spike_count):
    logger.info(f"Spikes at step {step}: {spike_count}")

def log_region_firing_rates(step, region_rates):
    rates_str = ", ".join(f"{region}: {rate:.4f} Hz" for region, rate in region_rates.items())
    logger.info(f"[Step {step}] Region firing rates: {rates_str}")

def log_region_burst_counts(step, burst_counts):
    bursts_str = ", ".join(f"{region}: {count}" for region, count in burst_counts.items())
    logger.info(f"[Step {step}] Region burst counts: {bursts_str}")

def log_average_synaptic_weight(step, avg_weight):
    logger.info(f"[Step {step}] Average synaptic weight: {avg_weight:.4f}")
