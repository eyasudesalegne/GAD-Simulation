# Generalized Anxiety Disorder (GAD) Brain Simulation Project

---

## Overview

This project is a biologically inspired computational simulation framework designed to model Generalized Anxiety Disorder (GAD) at the neural network level. It simulates a network of interconnected brain regions composed of excitatory and inhibitory neuron populations, using realistic connectome data, neuromodulator dynamics, and synaptic plasticity. The framework supports:

- Multi-region brain network simulation with heterogeneous synaptic weights, delays, and plasticity.
- Biophysical neuron modeling using Izhikevich-type neurons modulated by neuromodulator receptor activation.
- Modeling key neuromodulators: dopamine, serotonin, norepinephrine, acetylcholine, GABA, and cortisol, including HPA axis feedback.
- Treatment simulation including pharmacological (SSRIs, SNRIs, Benzodiazepines) and behavioral interventions (CBT, rTMS, Mindfulness, Sleep Therapy).
- Patient profile-based adjustment of simulation parameters reflecting clinical severity scores.
- Detailed logging and multi-level analytics: simple firing rates, spike counts, intermediate inter-spike intervals and Fano factors, and advanced burst detection, synaptic plasticity trends, neuromodulator correlations, and spike entropy.
- Interactive graphical user interface (GUI) built with PyQt5 for configuring simulations, running experiments, viewing analytics, exporting data, and 3D brain visualization.
- Modular design enabling extension with new neuron types, treatments, and analysis methods.

This framework is intended for neuroscience researchers, computational modelers, and clinicians interested in exploring network-level mechanisms of anxiety disorders and treatment effects in silico.

---

## Features

- **Realistic Connectome:** Weighted, plastic, delayed synaptic connections between anatomically inspired brain regions.
- **Neuromodulator Modeling:** Dynamic extracellular concentrations with receptor kinetics and co-modulation effects.
- **Neuron Populations:** Excitatory and inhibitory neuron subclasses with Izhikevich dynamics modulated by neuromodulators.
- **Synaptic Plasticity:** Short-term plasticity and spike-timing-dependent plasticity (STDP) modulated by neuromodulator levels.
- **Patient Diagnostics:** Clinical profile inputs (anxiety, stress, cortisol, HRV, sleep) mapped to simulation severity parameters.
- **Treatment Simulation:** Incorporates pharmacological and behavioral treatments adjusting neuromodulator levels and neuron parameters.
- **Multi-level Analytics:** From basic firing rates to advanced burst statistics and neuromodulator correlation analyses.
- **3D Visualization:** Interactive brain mesh and neuron-level connectivity visualization using PyVista and Nilearn datasets.
- **Extensible GUI:** User-friendly PyQt5 interface to configure parameters, run simulations asynchronously, visualize results, and export data.
- **Logging:** Thread-safe logging with detailed timestamped events and GUI integration.

---

## Installation

### Requirements

- Python 3.8 or later
- See `requirements.txt` for exact dependencies:

```
numpy==1.24.3
pandas>=1.3
matplotlib>=3.5
scipy>=1.7
ipywidgets>=7.7
nilearn==0.10.2
nibabel==4.0.1
pyvista==0.42.0
PyQt5>=5.15
requests>=2.28
chardet>=5.1.0
```

### Installation Steps

1. Clone or download the repository.

2. Create a Python virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate  # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set your OpenAI API key as environment variable for log analysis feature in GUI (optional):

```bash
export OPENAI_API_KEY="your_api_key_here"  # Linux/macOS
setx OPENAI_API_KEY "your_api_key_here"    # Windows (restart terminal)
```

---

## Usage

### Running the GUI

Launch the graphical interface for simulation configuration, execution, and analysis:

```bash
python gui3.py
```

### GUI Workflow

1. **Patient Profile Inputs:** Enter clinical scores for anxiety, stress, cortisol, heart rate variability, and sleep quality.
2. **Simulation Parameters:** Set simulation duration (seconds) and timestep (`dt` in seconds).
3. **Treatment Selection:** Choose among pharmacological and behavioral treatments and specify intensity, dosage, and frequency.
4. **Run Simulation:** Click "Run Simulation" to start. Logs and progress are shown in real-time.
5. **Analyze Results:** Use the analysis selector to view simple, intermediate, or advanced plots.
6. **Export Data:** Save plots as PNG or raw spike data as CSV.
7. **3D Visualization:** Launch interactive 3D brain connectivity visualization.
8. **Save Logs:** Save simulation logs for review or sharing.
9. **Dark Mode:** Toggle between light and dark UI themes.
10. **User Guides:** Access in-app documentation on usage, project structure, and analysis charts.

---

## Project Structure

```
brain_sim/             # Core simulation package
  __init__.py          # Package initialization and imports
  connectome_data.py   # Anatomical connectome data dict with weights/delays
  region.py            # Brain region and neuron population classes
  NeuronSubclass.py    # Neuron subclass with Izhikevich dynamics
  synapse.py           # Synapse modeling with plasticity and neuromodulation
  neuromodulator.py    # Neuromodulator concentrations and receptor kinetics
  drives.py            # Various external and internal input drive models
  treatments.py        # Treatment classes affecting neuromodulators and neurons
  diagnostics.py       # Patient profile and GAD severity computations
  analytics.py         # Multi-level simulation analytics and plotting
  simulation.py        # Simulation engine integrating regions, synapses, neuromodulators
  validate_connectome.py # Connectome validation utilities
  logger.py            # Thread-safe logging with GUI support
  network_viz.py       # 3D brain mesh and network visualization using PyVista
gui3.py                # PyQt5 GUI application for running simulations and visualization
requirements.txt       # Python package dependencies list
```

---

## Key Components

### 1. Simulation Core

- `simulation.py`: Main simulation engine managing brain regions, neuron populations, synapses, neuromodulators, simulation loop, logging, and analytics.
- `region.py` & `NeuronSubclass.py`: Define neuron populations with Izhikevich model, neuromodulation effects, spike detection, and firing dynamics.
- `synapse.py`: Models synaptic connections including delays, short-term plasticity, and modulation by neuromodulators.

### 2. Connectome & Validation

- `connectome_data.py`: Brain region connectivity with synaptic weights, plasticity factors, and delays.
- `validate_connectome.py`: Utilities to check connectome integrity, parameter correctness, and bidirectionality.

### 3. Neuromodulation

- `neuromodulator.py`: Models extracellular neuromodulator concentrations and receptor kinetics for dopamine, serotonin, norepinephrine, acetylcholine, GABA, and cortisol including HPA axis feedback.

### 4. Treatments

- `treatments.py`: Classes modeling pharmacological and behavioral treatments affecting neuromodulator levels and neuron parameters (e.g., SSRIs, Benzodiazepines, CBT).

### 5. Diagnostics & Patient Profiles

- `diagnostics.py`: Clinical profile representation, calculation of GAD severity score, and adjustment of simulation parameters accordingly.

### 6. Analytics

- `analytics.py`: Implements simple, intermediate, and advanced analysis functions producing firing rates, spike counts, inter-spike interval metrics, burst detection, synaptic trends, neuromodulator correlations, and spike entropy.

### 7. Visualization

- `network_viz.py`: 3D interactive visualization of brain mesh and neuron-level connectivity using PyVista and Nilearn cortical surfaces.

### 8. GUI

- `gui3.py`: PyQt5 application with controls for inputs, simulation execution in a background thread, real-time logs, plotting tabs, 3D visualization launcher, export functionality, dark mode, and log-based AI-assisted analysis via OpenAI API.

---

## Extensibility

- Add new neuron subclasses or brain regions by extending `NeuronSubclass` and `BrainRegion`.
- Implement additional treatments by subclassing `Treatment` in `treatments.py`.
- Extend analysis functions in `analytics.py` for custom metrics.
- Enhance visualization with new plotting methods in `network_viz.py`.
- Integrate other data-driven or experimental connectomes in `connectome_data.py`.

---

## Example Usage Snippet

```python
from brain_sim.simulation import SimulationWithAnalytics
from brain_sim.diagnostics import PatientProfile, calculate_gad_severity, set_simulation_parameters

# Create simulation
sim = SimulationWithAnalytics(dt=0.001, duration=10.0)

# Define patient profile
patient = PatientProfile(anxiety_score=10, stress_level=50, cortisol=15, hr_variability=30, sleep_quality=7)
severity = calculate_gad_severity(patient)

# Adjust simulation parameters based on severity
set_simulation_parameters(sim, severity)

# Run simulation
sim.run()

# Access analytics results
print(sim.analytics_results['simple']['firing_rates'])
```

---

## Citation and License

This project is provided for research and educational purposes. Please cite appropriately if used in publications.

---

## Contact

For questions, suggestions, or contributions, please contact:

**Eyasu Desalegne Beyene**  
Email: [your_email@example.com]  
GitHub: [your_github_profile]  
