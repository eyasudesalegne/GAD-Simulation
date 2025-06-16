import numpy as np

def pharmacodynamic(conc, Emax=1.0, EC50=0.3, n=2.0, E0=0.0):
    numerator = Emax * (conc ** n)
    denominator = (EC50 ** n) + (conc ** n) + 1e-12
    return E0 + numerator / denominator

class HPAAxis:
    def __init__(self, dt):
        self.dt = dt

        self.CRH = 0.2
        self.ACTH = 0.2
        self.Cort = 0.2

        self.delay_CRH_ACTH_steps = int(5 * 60 / dt)
        self.delay_ACTH_Cort_steps = int(15 * 60 / dt)

        self.CRH_buffer = np.zeros(self.delay_CRH_ACTH_steps)
        self.ACTH_buffer = np.zeros(self.delay_ACTH_Cort_steps)

        self.k_prod_CRH = 0.012
        self.k_prod_ACTH = 0.025
        self.k_prod_Cort = 0.035

        self.k_decay_CRH = 0.005
        self.k_decay_ACTH = 0.008
        self.k_decay_Cort = 0.0025

        self.k_feedback = 0.55

        self.step_count = 0

    def update(self, stress_input):
        feedback = 1 / (1 + self.k_feedback * self.Cort)

        dCRH = self.dt * (self.k_prod_CRH * stress_input * feedback - self.k_decay_CRH * self.CRH)
        self.CRH += dCRH

        idx_CRH = self.step_count % self.delay_CRH_ACTH_steps
        self.CRH_buffer[idx_CRH] = self.CRH

        delayed_CRH = self.CRH_buffer[(self.step_count - self.delay_CRH_ACTH_steps) % self.delay_CRH_ACTH_steps]
        dACTH = self.dt * (self.k_prod_ACTH * delayed_CRH * feedback - self.k_decay_ACTH * self.ACTH)
        self.ACTH += dACTH

        idx_ACTH = self.step_count % self.delay_ACTH_Cort_steps
        self.ACTH_buffer[idx_ACTH] = self.ACTH

        delayed_ACTH = self.ACTH_buffer[(self.step_count - self.delay_ACTH_Cort_steps) % self.delay_ACTH_Cort_steps]
        dCort = self.dt * (self.k_prod_Cort * delayed_ACTH - self.k_decay_Cort * self.Cort)
        self.Cort += dCort

        self.CRH = np.clip(self.CRH, 0, 1)
        self.ACTH = np.clip(self.ACTH, 0, 1)
        self.Cort = np.clip(self.Cort, 0, 1)

        self.step_count += 1

        return self.CRH, self.ACTH, self.Cort

class Neuromodulator:
    def __init__(self, dt, duration, compartments=None, diffusion_matrix=None, noise_std=0.005,
                 dt_init=0.1, dt_min=0.01, dt_max=1.0):
        self.dt = dt
        self.duration = duration
        self.steps = int(duration / dt)
        self.compartments = compartments or ['default']
        self.N = len(self.compartments)
        self.diffusion_matrix = diffusion_matrix if diffusion_matrix is not None else np.zeros((self.N, self.N))
        self.noise_std = noise_std

        self.concentrations = {k: np.full(self.N, 0.2) for k in ['DA', '5-HT', 'NE', 'ACh', 'GABA', 'Cort']}

        self.hpa_axis = HPAAxis(dt)

        self.receptors = {
            'DA': {
                'D1': {'A': np.ones(self.N), 'B': np.zeros(self.N), 'I': np.zeros(self.N)},
                'D2': {'A': np.ones(self.N), 'B': np.zeros(self.N), 'I': np.zeros(self.N)},
            },
            '5-HT': {
                '5HT1A': {'A': np.ones(self.N), 'B': np.zeros(self.N), 'I': np.zeros(self.N)},
                '5HT2A': {'A': np.ones(self.N), 'B': np.zeros(self.N), 'I': np.zeros(self.N)},
            },
            'NE': {
                'Alpha1': {'A': np.ones(self.N), 'B': np.zeros(self.N), 'I': np.zeros(self.N)},
                'Beta1': {'A': np.ones(self.N), 'B': np.zeros(self.N), 'I': np.zeros(self.N)},
            },
            'ACh': {
                'M1': {'A': np.ones(self.N), 'B': np.zeros(self.N), 'I': np.zeros(self.N)},
                'Nicotinic': {'A': np.ones(self.N), 'B': np.zeros(self.N), 'I': np.zeros(self.N)},
            }
        }

        self.receptor_params = {
            'DA': {
                'D1': {'k_on': 1.1, 'k_off': 0.11, 'k_int': 0.009, 'k_rec': 0.004},
                'D2': {'k_on': 1.3, 'k_off': 0.13, 'k_int': 0.014, 'k_rec': 0.006},
            },
            '5-HT': {
                '5HT1A': {'k_on': 0.95, 'k_off': 0.095, 'k_int': 0.010, 'k_rec': 0.005},
                '5HT2A': {'k_on': 1.15, 'k_off': 0.115, 'k_int': 0.012, 'k_rec': 0.007},
            },
            'NE': {
                'Alpha1': {'k_on': 1.0, 'k_off': 0.1, 'k_int': 0.008, 'k_rec': 0.005},
                'Beta1': {'k_on': 1.1, 'k_off': 0.11, 'k_int': 0.009, 'k_rec': 0.004},
            },
            'ACh': {
                'M1': {'k_on': 1.05, 'k_off': 0.105, 'k_int': 0.009, 'k_rec': 0.005},
                'Nicotinic': {'k_on': 1.2, 'k_off': 0.12, 'k_int': 0.010, 'k_rec': 0.006},
            }
        }

        self.dt_sim = dt_init
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.t_sim = 0.0

        self.history = {k: [] for k in self.concentrations.keys()}

    def pharmacodynamic(self, conc, Emax=1.0, EC50=0.3, n=2.0, E0=0.0):
        numerator = Emax * (conc ** n)
        denominator = (EC50 ** n) + (conc ** n) + 1e-12
        return E0 + numerator / denominator

    def get_receptor_activation(self, nm, rec_type):
        states = self.receptors.get(nm, {}).get(rec_type, None)
        if states is None:
            return np.zeros(self.N)
        total = states['A'] + states['B'] + states['I'] + 1e-12
        return states['B'] / total

    def get_dopamine_receptors(self):
        D1 = self.get_receptor_activation('DA', 'D1')
        D2 = self.get_receptor_activation('DA', 'D2')
        return D1, D2

    def get_serotonin_receptors(self):
        HT1A = self.get_receptor_activation('5-HT', '5HT1A')
        HT2A = self.get_receptor_activation('5-HT', '5HT2A')
        return HT1A, HT2A

    def co_modulation(self):
        DA = self.concentrations['DA']
        HT = self.concentrations['5-HT']
        NE = self.concentrations['NE']

        DA_effect = 1 / (1 + np.exp(-10 * (DA - 0.3)))
        HT_effect = 1 / (1 + np.exp(15 * (HT - 0.5)))

        self.concentrations['5-HT'] *= DA_effect
        self.concentrations['DA'] *= HT_effect

        NE_DA_interaction = 0.5 * DA * NE
        self.concentrations['NE'] += NE_DA_interaction * self.dt
        self.concentrations['NE'] = np.clip(self.concentrations['NE'], 0, 1)

    def update_receptors(self):
        for nm, recs in self.receptors.items():
            conc = self.concentrations.get(nm, np.zeros(self.N))
            for rec_type, states in recs.items():
                p = self.receptor_params.get(nm, {}).get(rec_type, None)
                if p is None:
                    continue
                A, B, I = states['A'], states['B'], states['I']

                dB = self.dt_sim * (p['k_on'] * conc * A - p['k_off'] * B)
                dI = self.dt_sim * (p['k_int'] * B - p['k_rec'] * I)

                states['A'] = np.clip(A - dB + dI, 0, 1)
                states['B'] = np.clip(B + dB - p['k_int'] * B * self.dt_sim, 0, 1)
                states['I'] = np.clip(I + dI - p['k_rec'] * I * self.dt_sim, 0, 1)

    def drift(self, state, external_input, diffusion):
        decay = -0.1 * state
        return external_input + decay + diffusion

    def diffusion(self, state):
        return self.noise_std * np.ones_like(state)

    def step_sde(self, state, external_input):
        diffusion_term = self.diffusion_matrix @ state - np.sum(self.diffusion_matrix, axis=1) * state
        f = self.drift(state, external_input, diffusion_term)
        g = self.diffusion(state)
        dW = np.random.normal(0, np.sqrt(self.dt_sim), size=state.shape)
        return state + f * self.dt_sim + g * dW

    def adaptive_step(self, external_inputs):
        accept_step = False
        while not accept_step:
            concentrations_new = {}
            max_change = 0

            for nm, state in self.concentrations.items():
                if nm == 'Cort':
                    concentrations_new[nm] = np.full(self.N, self.hpa_axis.Cort)
                    continue
                ext_input = external_inputs.get(nm, np.zeros(self.N))
                new_state = self.step_sde(state, ext_input)
                change = np.max(np.abs(new_state - state))
                max_change = max(max_change, change)
                concentrations_new[nm] = np.clip(new_state, 0, 1)

            if max_change > 0.05 and self.dt_sim > self.dt_min:
                self.dt_sim /= 2
            else:
                accept_step = True
                if max_change < 0.005 and self.dt_sim < self.dt_max:
                    self.dt_sim *= 2

            if accept_step:
                self.concentrations = concentrations_new
                self.t_sim += self.dt_sim
                for k, v in self.concentrations.items():
                    self.history[k].append((self.t_sim, np.mean(v)))

    def step(self, external_inputs, stress_input):
        self.hpa_axis.update(stress_input)

        ext_inputs_except_cort = {k: v for k, v in external_inputs.items() if k != 'Cort'}
        self.adaptive_step(ext_inputs_except_cort)

        self.co_modulation()

        self.update_receptors()

    def get_state(self):
        return {k: v.copy() for k, v in self.concentrations.items()}
