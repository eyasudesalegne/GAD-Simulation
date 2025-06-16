import numpy as np
from brain_sim.logger import logger

class Neuromodulator:
    """
    Tracks key neuromodulators: Dopamine (DA), Serotonin (5-HT), Norepinephrine (NE),
    Acetylcholine (ACh), and Cortisol (Cort).
    Updates them each timestep based on external inputs, global neural activity,
    and adaptive decay dynamics.
    Records timestamped history for analysis.
    """

    def __init__(self, dt, duration):
        self.dt = dt
        self.steps = int(duration / dt)

        # Initialize neuromodulator levels to baseline (0-1 scale)
        self.state = {k: 0.2 for k in ['DA', '5-HT', 'NE', 'ACh', 'Cort']}

        # History: list of (time, value) tuples per neuromodulator
        self.history = {k: [] for k in self.state}

        # External input time series (modifiable or extendable)
        self.external_inputs = {
            'reward': self.generate_timed_array(0.1, 0.5),
            'mood': self.generate_timed_array(0.05, 0.3),
            'stress': self.generate_timed_array(0.07, 0.4),
            'arousal': self.generate_timed_array(0.12, 0.6),
            'astrocyte_modulation': self.generate_oscillation(0.5, 0.2),
            'sleep_wake': self.generate_sleep_wake_cycle()
        }

    def generate_timed_array(self, freq, amplitude):
        """
        Generate sinusoidal oscillation array for external inputs.
        """
        t = np.linspace(0, self.steps * self.dt, self.steps)
        return amplitude * np.sin(2 * np.pi * freq * t)

    def generate_oscillation(self, freq, amplitude):
        return self.generate_timed_array(freq, amplitude)

    def generate_sleep_wake_cycle(self):
        """
        Simulate a simplified sleep-wake cycle with 16 hours awake (value=1) and 8 hours asleep (value=0).
        """
        cycle = np.zeros(self.steps)
        seconds_per_day = 86400
        awake_seconds = 57600  # 16 hours
        for i in range(self.steps):
            time_in_day = (i * self.dt) % seconds_per_day
            cycle[i] = 1 if time_in_day < awake_seconds else 0
        return cycle

    def update(self, step, global_firing_rate=0.0, reward_signal=0.0):
        """
        Update neuromodulator levels at the given simulation step.
        Uses adaptive decay modulated by global firing rate and adds external input effects.

        Parameters:
            step (int): current simulation step index
            global_firing_rate (float): firing rate normalized to [0, ...]
            reward_signal (float): transient reward input (0 or 1)
        """
        step = min(max(step, 0), self.steps - 1)
        t = step * self.dt

        for k in self.state:
            # Aggregate scaled external inputs affecting neuromodulator k
            ext_input = 0.0
            for mod_name, arr in self.external_inputs.items():
                # Scale inputs by 0.1 to keep effects moderate
                ext_input += arr[min(step, len(arr)-1)] * 0.1

            # Adaptive decay and activity-dependent boosts
            if k == 'DA':  # Dopamine
                activity_boost = 0.2 * reward_signal + 0.1 * global_firing_rate
                tau = 0.5
                decay = -self.state[k] * (1 + global_firing_rate) / tau

            elif k == '5-HT':  # Serotonin
                activity_boost = -0.05 * (global_firing_rate - 0.05)
                tau = 0.3
                decay = -self.state[k] * (1 + global_firing_rate) / tau

            elif k == 'NE':  # Norepinephrine
                activity_boost = 0.1 * global_firing_rate
                tau = 0.3
                decay = -self.state[k] * (1 + global_firing_rate) / tau

            elif k == 'Cort':  # Cortisol (stress hormone)
                activity_boost = 0.2 if global_firing_rate > 0.1 else 0.0
                tau = 0.4
                decay = -self.state[k] * (1 + global_firing_rate) / tau

            else:  # ACh or others
                activity_boost = 0.0
                tau = 0.5
                decay = -self.state[k] * (1 + global_firing_rate) / tau

            # Add small Gaussian noise for stochastic fluctuations
            noise = np.random.normal(0, 0.001)

            # Euler integration step
            delta = self.dt * (decay + ext_input + activity_boost + noise)
            self.state[k] += delta

            # Clamp to [0,1] bounds
            self.state[k] = np.clip(self.state[k], 0, 1)

            # Record history for analysis
            self.history[k].append((t, self.state[k]))

        # Log neuromodulator levels every 100 steps (adjust frequency as needed)
        if step % 100 == 0:
            levels_str = ", ".join(f"{nm}={val:.3f}" for nm, val in self.state.items())
            logger.info(f"Neuromodulator levels at step {step}: {levels_str}")

    def get_state(self):
        """
        Return current neuromodulator levels as a dict.
        """
        return self.state.copy()
