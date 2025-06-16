import numpy as np
from brain_sim.logger import logger

class ExternalDrive:
    """
    Uses precomputed real data as drive signal.
    Suitable for circadian or ultradian rhythm recordings.
    """
    def __init__(self, values, dt):
        self.values = values
        self.dt = dt

    def get(self, step):
        return self.values[step] if step < len(self.values) else 0

class StochasticDrive:
    """
    Poisson-driven stochastic input with rate and amplitude from biological data.
    Typical cortical spontaneous firing rates ~ 0.5-5 Hz.

    Parameters:
      rate: baseline firing rate (Hz).
      amp: amplitude scaling.
      dt: timestep in seconds.
      duration: total duration in seconds.
    """
    def __init__(self, rate=1.5, amp=0.1, dt=0.001, duration=600):
        self.dt = dt
        self.duration = duration
        steps = int(duration / dt)
        self.signal = np.random.poisson(rate * dt, steps) * amp

    def get(self, step):
        return self.signal[step] if step < len(self.signal) else 0

class BurstDrive:
    """
    Models bursty inputs with oscillatory modulation.
    """
    def __init__(self, base_rate=1.0, burst_amp=0.5, burst_freq=20.0, dt=0.001, duration=600):
        self.dt = dt
        self.duration = duration
        self.time = np.arange(0, duration, dt)
        self.signal = base_rate + burst_amp * np.sin(2 * np.pi * burst_freq * self.time)

    def get(self, step):
        return self.signal[step] if step < len(self.signal) else 0

class SleepArchitectureDrive:
    """
    Models sleep pressure and sleep-wake cycle influences.
    """
    def __init__(self, dt=0.001, duration=600):
        self.dt = dt
        self.duration = duration
        self.sleep_pressure = 0.0
        self.sleep_threshold = 0.7
        self.time = 0.0

    def step(self):
        # Simplified sleep pressure dynamics
        self.sleep_pressure += 0.001 * self.dt
        if self.sleep_pressure > 1.0:
            self.sleep_pressure = 1.0
        self.time += self.dt
        awake = self.sleep_pressure < self.sleep_threshold
        return awake

class AdaptiveDrive:
    """
    Drive modulated by neuromodulator feedback and reward signals.
    """
    def __init__(self, base_drive=0.5):
        self.base_drive = base_drive
        self.adaptation = 0.0

    def update(self, reward_signal):
        self.adaptation += 0.01 * (reward_signal - self.adaptation)
        return self.base_drive * (1 + self.adaptation)

class EnvironmentalDrive:
    """
    External environmental stimuli drive.
    """
    def __init__(self, intensity=0.2):
        self.intensity = intensity

    def get(self, condition):
        # condition can be binary or continuous
        return self.intensity * condition

class MultiFrequencyDrive:
    """
    Combines multiple sinusoidal components for rhythmic drives.
    """
    def __init__(self, freqs=[1, 3, 7], amps=[0.1, 0.05, 0.02], dt=0.001, duration=600):
        self.dt = dt
        self.duration = duration
        self.time = np.arange(0, duration, dt)
        self.signal = np.zeros_like(self.time)
        for f, a in zip(freqs, amps):
            self.signal += a * np.sin(2 * np.pi * f * self.time)

    def get(self, step):
        return self.signal[step] if step < len(self.signal) else 0
