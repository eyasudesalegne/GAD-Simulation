import numpy as np

class ExternalDrive:
    """
    Uses a precomputed array of values as the drive signal.
    """
    def __init__(self, values, dt):
        self.values = values
        self.dt = dt

    def get(self, step):
        return self.values[step] if step < len(self.values) else 0

class OscillatoryDrive:
    """
    Generates a simple sinusoidal drive at a specified frequency and amplitude.
    """
    def __init__(self, freq, amp, dt, duration):
        t = np.arange(0, duration, dt)
        self.signal = amp * np.sin(2 * np.pi * freq * t)

    def get(self, step):
        return self.signal[step] if step < len(self.signal) else 0

class MultiBandOscillatoryDrive:
    """
    Combines multiple frequency band oscillations into a single drive signal.
    Includes:
      - Theta (6 Hz)
      - Gamma (40 Hz)
      - Delta (2 Hz)
      - Alpha (10 Hz)
      - Beta (20 Hz)
    """
    def __init__(self, dt, duration):
        t = np.arange(0, duration, dt)
        self.signals = {
            "theta": 0.5 * np.sin(2 * np.pi * 6 * t),
            "gamma": 0.3 * np.sin(2 * np.pi * 40 * t),
            "delta": 0.2 * np.sin(2 * np.pi * 2 * t),
            "alpha": 0.4 * np.sin(2 * np.pi * 10 * t),
            "beta": 0.3 * np.sin(2 * np.pi * 20 * t),
        }
        self.composite_signal = sum(self.signals.values())

    def get(self, step):
        return self.composite_signal[step] if step < len(self.composite_signal) else 0

class StochasticDrive:
    """
    Generates a drive signal based on a Poisson process to introduce stochastic fluctuations.
    
    Parameters:
      rate: Expected rate (Hz) for the Poisson process.
      amp: Amplitude scaling factor.
      dt: Timestep in seconds.
      duration: Total duration of the drive in seconds.
    """
    def __init__(self, rate, amp, dt, duration):
        self.dt = dt
        self.duration = duration
        self.signal = np.random.poisson(rate * dt, int(duration/dt)) * amp

    def get(self, step):
        return self.signal[step] if step < len(self.signal) else 0

class BurstDrive:
    """
    Generates a drive signal with burst patterns.
    
    A burst is defined as a period of elevated drive amplitude that occurs periodically.
    
    Parameters:
      burst_interval: Time between burst onsets (s).
      burst_duration: Duration of each burst (s).
      amp: Amplitude during burst periods.
      dt: Timestep in seconds.
      duration: Total duration of the drive in seconds.
    """
    def __init__(self, burst_interval, burst_duration, amp, dt, duration):
        self.dt = dt
        self.duration = duration
        self.burst_interval = burst_interval
        self.burst_duration = burst_duration
        self.amp = amp
        t = np.arange(0, duration, dt)
        self.signal = np.array([amp if (time % burst_interval) < burst_duration else 0 for time in t])

    def get(self, step):
        return self.signal[step] if step < len(self.signal) else 0

class MultiFrequencyDrive:
    """
    Combines multiple sinusoidal drives with specified frequencies and amplitudes.
    
    Parameters:
      frequencies: List of frequencies (Hz).
      amplitudes: List of amplitudes corresponding to each frequency.
      dt: Timestep in seconds.
      duration: Total duration of the drive in seconds.
    """
    def __init__(self, frequencies, amplitudes, dt, duration):
        self.dt = dt
        self.duration = duration
        t = np.arange(0, duration, dt)
        self.signal = sum(amp * np.sin(2 * np.pi * freq * t) for freq, amp in zip(frequencies, amplitudes))

    def get(self, step):
        return self.signal[step] if step < len(self.signal) else 0
