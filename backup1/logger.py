import sys
import threading
from datetime import datetime

class Logger:
    """
    Thread-safe logger to handle progress and debug messages.
    Supports console output and optional GUI callbacks.
    """

    LEVELS = {
        'INFO': '[INFO]',
        'WARNING': '[WARNING]',
        'ERROR': '[ERROR]'
    }

    def __init__(self):
        self.lock = threading.Lock()
        self.gui_callback = None  # Function to send logs to GUI

    def set_gui_callback(self, callback):
        """
        Set a callable that receives log messages for GUI display.
        callback: function(str) -> None
        """
        self.gui_callback = callback

    def _format_message(self, message, level='INFO'):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        level_tag = self.LEVELS.get(level.upper(), '[INFO]')
        return f"{timestamp} {level_tag} {message}"

    def log(self, message, level='INFO'):
        formatted = self._format_message(message, level)
        with self.lock:
            # Print to console
            print(formatted, file=sys.stdout)
            sys.stdout.flush()
            # Send to GUI callback if set
            if self.gui_callback:
                try:
                    self.gui_callback(formatted)
                except Exception as e:
                    print(f"[Logger] Error sending log to GUI callback: {e}", file=sys.stderr)

    def info(self, message):
        self.log(message, 'INFO')

    def warning(self, message):
        self.log(message, 'WARNING')

    def error(self, message):
        self.log(message, 'ERROR')

# Singleton instance to be imported and used globally
logger = Logger()

# === Logging helper functions ===

def log_simulation_step(step, total_steps):
    percent = (step / total_steps) * 100
    logger.info(f"Simulation progress: Step {step}/{total_steps} ({percent:.1f}%)")

def log_region_added(region_name, neuron_count):
    logger.info(f"Region added: {region_name} with {neuron_count} neurons")

def log_synapse_created(src, tgt, weight, delay_ms):
    logger.info(f"Synapse created: {src} → {tgt} | weight={weight:.3f}, delay={delay_ms:.2f} ms")

def log_neuromodulator_update(step, neuromod_state):
    state_str = ", ".join(f"{k}: {v:.3f}" for k, v in neuromod_state.items())
    logger.info(f"Neuromodulator update at step {step}: {state_str}")

def log_spike_count(step, spike_count):
    logger.info(f"Spikes at step {step}: {spike_count}")

def log_region_firing_rates(step, region_rates):
    """
    region_rates: dict {region_name: firing_rate_hz}
    """
    rates_str = ", ".join(f"{region}: {rate:.4f} Hz" for region, rate in region_rates.items())
    logger.info(f"[Step {step}] Region firing rates: {rates_str}")

def log_region_burst_counts(step, burst_counts):
    """
    burst_counts: dict {region_name: burst_count}
    """
    bursts_str = ", ".join(f"{region}: {count}" for region, count in burst_counts.items())
    logger.info(f"[Step {step}] Region burst counts: {bursts_str}")

def log_average_synaptic_weight(step, avg_weight):
    logger.info(f"[Step {step}] Average synaptic weight: {avg_weight:.4f}")
