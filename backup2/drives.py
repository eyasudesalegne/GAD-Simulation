import numpy as np

class BaseDrive:
    def get(self, step):
        raise NotImplementedError()

class ExternalDrive(BaseDrive):
    def __init__(self, values, dt):
        self.values = values
        self.dt = dt

    def get(self, step):
        if step < 0:
            return 0
        return self.values[step] if step < len(self.values) else 0

class StochasticDrive(BaseDrive):
    def __init__(self, rate=1.5, amp=0.1, dt=0.001, duration=600):
        self.dt = dt
        self.duration = duration
        steps = int(duration / dt)
        self.signal = np.random.poisson(rate * dt, steps) * amp

    def get(self, step):
        if step < 0:
            return 0
        return self.signal[step] if step < len(self.signal) else 0

class BurstDrive(BaseDrive):
    def __init__(self, burst_interval=0.7, burst_duration=0.2, amp=0.5, dt=0.001, duration=600):
        self.dt = dt
        self.duration = duration
        t = np.arange(0, duration, dt)
        self.signal = np.array([amp if (time % burst_interval) < burst_duration else 0 for time in t])

    def get(self, step):
        if step < 0:
            return 0
        return self.signal[step] if step < len(self.signal) else 0

class SleepArchitectureDrive(BaseDrive):
    def __init__(self, dt=0.001, duration=8*3600, cycle_duration=90*60):
        self.dt = dt
        self.duration = duration
        self.cycle_duration = cycle_duration
        steps = int(duration / dt)
        self.signal = np.zeros(steps)

        nrem_dur = 75 * 60
        rem_dur = 15 * 60

        for i in range(steps):
            time_in_cycle = (i * dt) % cycle_duration
            if time_in_cycle < nrem_dur:
                self.signal[i] = 0.8
            else:
                self.signal[i] = 0.3

    def get(self, step):
        if step < 0:
            return 0
        return self.signal[step] if step < len(self.signal) else 0

class AdaptiveDrive(BaseDrive):
    def __init__(self, base_drive, adaptation_rate=0.02):
        self.base_drive = base_drive
        self.adaptation_rate = adaptation_rate
        self.current_modulation = 1.0

    def update(self, firing_rate_hz):
        if firing_rate_hz < 1.0:
            target = 1.2
        elif firing_rate_hz > 5.0:
            target = 0.5
        else:
            target = 1.2 - (firing_rate_hz - 1.0) * (0.7 / 4.0)
        delta = target - self.current_modulation
        self.current_modulation += self.adaptation_rate * delta

    def get(self, step):
        base_val = self.base_drive.get(step)
        return base_val * self.current_modulation

class EnvironmentalDrive(BaseDrive):
    def __init__(self, base_drive, environment_signal):
        self.base_drive = base_drive
        self.environment_signal = environment_signal

    def get(self, step):
        if step < 0:
            return 0
        env_mod = self.environment_signal[step] if step < len(self.environment_signal) else 1.0
        return self.base_drive.get(step) * env_mod

class MultiFrequencyDrive(BaseDrive):
    def __init__(self, dt=0.001, duration=600):
        t = np.arange(0, duration, dt)
        self.signal = (20 * np.sin(2 * np.pi * 2 * t) +
                       10 * np.sin(2 * np.pi * 6 * t) +
                       10 * np.sin(2 * np.pi * 10 * t) +
                       5 * np.sin(2 * np.pi * 20 * t) +
                       2 * np.sin(2 * np.pi * 40 * t))
        self.signal -= self.signal.min()
        self.signal /= self.signal.max()

    def get(self, step):
        if step < 0:
            return 0
        return self.signal[step] if step < len(self.signal) else 0
