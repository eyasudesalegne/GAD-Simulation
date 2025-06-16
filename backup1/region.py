import numpy as np

# Define region-specific noise configuration: (noise_rate, noise_amp)
REGION_NOISE_CONFIG = {
    # Amygdala-related regions
    "LA_E":       (0.3, 0.01),
    "BLA_E":      (0.4, 0.015),
    "CeA_I":      (0.6, 0.02),
    "aBNST_E":    (0.4, 0.015),
    "pBNST_E":    (0.4, 0.015),
    # Prefrontal and cingulate regions
    "vmPFC_E":    (0.5, 0.015),
    "vmPFC_I":    (0.3, 0.01),
    "mPFC_E":     (0.5, 0.02),
    "mPFC_I":     (0.3, 0.015),
    "dACC_E":     (0.5, 0.02),
    "dACC_I":     (0.3, 0.015),
    "rACC_E":     (0.4, 0.015),
    "DLPFC_E":    (0.5, 0.02),
    "OFC_E":      (0.4, 0.015),
    # Hippocampal regions
    "DG_E":       (0.2, 0.01),
    "CA3_E":      (0.2, 0.01),
    "CA1_E":      (0.2, 0.01),
    "Subiculum_E":(0.2, 0.01),
    # Insular regions
    "AI_E":       (0.3, 0.01),
    "PI_E":       (0.3, 0.01),
    # PAG regions
    "dPAG_E":     (0.2, 0.01),
    "vlPAG_E":    (0.2, 0.01),
    "lPAG_E":     (0.2, 0.01)
}

class BrainRegion:
    """
    A population of enhanced Izhikevich neurons with:
      - Leak current and refractory gating.
      - Intrinsic diversity in parameters.
      - Background Poisson noise (region-specific if available).
      - Potassium-dependent hyperpolarization.
      - Dopamine/Neuromodulator gating for plasticity.
      - Homeostatic plasticity and dynamic noise modulation.
      - Per-region tuning of thresholds, excitability, noise, and feedback sensitivity.

    Note: All membrane potentials and parameters are now in millivolts (mV).
    """
    def __init__(self, name, density, volume, neuron_type, neuromodulator=None, scaling=1/10000):
        self.name = name
        self.neuron_count = max(1, int(density * volume * scaling))
        
        # Izhikevich parameters (in mV)
        if neuron_type == "excitatory":
            self.a = np.random.normal(0.018, 0.008, self.neuron_count)
            self.b = np.random.normal(0.25, 0.03, self.neuron_count)
            self.c = np.random.normal(-65, 3, self.neuron_count)    # reset potential in mV
            self.d = np.random.normal(4, 1.5, self.neuron_count)      # recovery increment in mV
        else:
            self.a = np.random.normal(0.009, 0.003, self.neuron_count)
            self.b = np.random.normal(0.25, 0.03, self.neuron_count)
            self.c = np.random.normal(-65, 3, self.neuron_count)
            self.d = np.random.normal(2, 0.8, self.neuron_count)
        
        # Additional region-specific tuning for excitatory cells:
        if neuron_type == "excitatory":
            if self.name == "LA_E":
                self.a = np.random.normal(0.015, 0.005, self.neuron_count)
                self.b = np.random.normal(0.3, 0.015, self.neuron_count)
            elif self.name == "BLA_E":
                self.a = np.random.normal(0.016, 0.005, self.neuron_count)
                self.b = np.random.normal(0.28, 0.015, self.neuron_count)
            elif self.name in {"DG_E", "CA3_E", "CA1_E", "Subiculum_E"}:
                self.a = np.random.normal(0.02, 0.006, self.neuron_count)
                self.b = np.random.normal(0.22, 0.015, self.neuron_count)
        
        # Initialize membrane potential with typical mV noise
        self.v = np.random.normal(self.c, 1.0, self.neuron_count)
        self.u = self.b * self.v + np.random.normal(0, 1.0, self.neuron_count)
        self.I = np.zeros(self.neuron_count)
        self.refractory = np.zeros(self.neuron_count)
        self.last_spike_time = np.full(self.neuron_count, -np.inf)
        self.spikes = []
        self.centroid = np.random.rand(3)
        
        # Set noise parameters from REGION_NOISE_CONFIG if available
        if self.name in REGION_NOISE_CONFIG:
            self.noise_rate, self.noise_amp = REGION_NOISE_CONFIG[self.name]
        else:
            self.noise_rate = np.random.uniform(0.2, 1.0)
            self.noise_amp = np.random.uniform(0.01, 0.03)
        
        # Region-specific target firing rate settings (in Hz)
        if self.name in ['LA_E', 'BLA_E', 'CeA_I', 'aBNST_E', 'pBNST_E']:
            self.target_rate = np.random.uniform(0.005, 0.015)
        elif self.name in ['vmPFC_E', 'vmPFC_I', 'mPFC_E', 'mPFC_I', 'dACC_E', 'dACC_I', 'rACC_E', 'DLPFC_E', 'OFC_E']:
            self.target_rate = np.random.uniform(0.01, 0.02)
        elif self.name in ['DG_E', 'CA3_E', 'CA1_E', 'Subiculum_E']:
            self.target_rate = np.random.uniform(0.005, 0.015)
        elif self.name in ['AI_E', 'PI_E']:
            self.target_rate = np.random.uniform(0.008, 0.018)
        elif self.name in ['dPAG_E', 'vlPAG_E', 'lPAG_E']:
            self.target_rate = np.random.uniform(0.005, 0.015)
        else:
            self.target_rate = np.random.uniform(0.005, 0.02)
        
        np.random.seed(None)
        
        self.base_noise_amp = self.noise_amp
        self.I_K = np.zeros(self.neuron_count)
        self.delta_I_K = 1.0
        self.neuromodulator = neuromodulator
        self.dopa_sensitivity = np.random.uniform(0.01, 0.05)
        
        # Region-specific noise adjustment factor for homeostatic feedback
        self.noise_adjustment_factor = np.random.uniform(0.1, 0.3)
        
        # Debug output
        print(f"[{self.name}] Initialized: neurons={self.neuron_count}, noise_rate={self.noise_rate:.3f}, noise_amp={self.noise_amp:.3f}, target_rate={self.target_rate:.4f}")
        print(f"   Izhikevich params (first 5): a={self.a[:5]}, b={self.b[:5]}, c={self.c[:5]}, d={self.d[:5]}")
    
    def step(self, dt, t):
        # Add background Poisson noise
        poisson_spikes = np.random.poisson(self.noise_rate * dt, self.neuron_count)
        self.I += poisson_spikes * self.noise_amp
        self.I += np.random.normal(0, 0.01, self.neuron_count)
        
        # Decay potassium current with a time constant of 100 ms
        self.I_K -= (dt / 0.100) * self.I_K
        
        active = self.refractory <= 0
        # Compute leak term: shift v by 65 mV (typical resting potential) and scale by 100 mV
        leak = (self.v[active] + 65) / 100
        
        # Izhikevich model update (scaling dt appropriately)
        dv = ((0.04 * self.v[active]**2 + 5 * self.v[active] + 140 - self.u[active] + self.I[active] - self.I_K[active]) - leak) * (dt / 0.001)
        dv = np.clip(dv, -1.0, 1.0)  # prevent numerical instability
        self.v[active] += dv
        
        du = self.a[active] * (self.b[active] * self.v[active] - self.u[active]) * dt
        du = np.clip(du, -1.0, 1.0)
        self.u[active] += du
        
        # Clamp membrane potential and recovery variable to realistic ranges (mV)
        self.v[active] = np.clip(self.v[active], -90, 40)
        self.u[active] = np.clip(self.u[active], -90, 40)
        
        self.refractory = np.maximum(0, self.refractory - dt)
        
        # Neuromodulator effects (e.g., dopamine modulation)
        if self.neuromodulator:
            DA_level = self.neuromodulator.state.get("DA", 0)
            self.a *= (1 + DA_level * self.dopa_sensitivity)
            self.noise_amp *= (1 + DA_level * 0.02)
        
        # Spike detection: threshold of 30 mV and a refractory period >15 ms
        fired = np.where((self.v >= 30) & ((t - self.last_spike_time) > 0.015))[0]
        for idx in fired:
            self.spikes.append((t, idx))
            # Reset membrane potential to c plus small noise (mV)
            self.v[idx] = self.c[idx] + np.random.normal(0, 1.0)
            self.refractory[idx] = 0.010  # 10 ms refractory period
            self.u[idx] += self.d[idx] + np.random.normal(0, 1.0)
            self.I_K[idx] += self.delta_I_K
            self.last_spike_time[idx] = t
            eta = 1e-5
            self.d[idx] = max(0, self.d[idx] + eta * (1 - self.target_rate))
        
        local_rate = len(fired) / self.neuron_count
        # Update noise amplitude using homeostatic regulation
        self.noise_amp = self.base_noise_amp * (1 + self.noise_adjustment_factor * (local_rate - self.target_rate))
        
        self.I.fill(0)
    
    def reset(self):
        self.spikes.clear()
        self.last_spike_time.fill(-np.inf)
