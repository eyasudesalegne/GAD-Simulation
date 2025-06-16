# brain_sim/__init__.py

__version__ = '0.1.0'

# Core data and models
from .connectome_data import connectome
from .region import BrainRegion, NeuronSubclass
from .synapse import Synapse
from .neuromodulator import Neuromodulator
from .drives import (
    ExternalDrive,
    StochasticDrive,
    BurstDrive,
    SleepArchitectureDrive,
    AdaptiveDrive,
    EnvironmentalDrive,
    MultiFrequencyDrive,
)

# Treatments and treatment manager
from .treatments import (
    Treatment,
    SSRI,
    SNRI,
    Benzodiazepine,
    CBT,
    Exposure,
    rTMS,
    Mindfulness,
    SleepTherapy,
    TreatmentManager,
)

# Analytics and diagnostics
from .analytics import (
    simple_firing_rates,
    simple_spike_counts,
    simple_plot,
    intermediate_isi_cv,
    intermediate_fano_factor,
    intermediate_plot,
    advanced_burst_detection,
    advanced_synaptic_plasticity_trends,
    advanced_neuromodulator_correlation,
    advanced_spike_entropy,
    advanced_plot,
    run_all_analytics,
)
from .diagnostics import PatientProfile, calculate_gad_severity, set_simulation_parameters

# Simulation core
from .simulation import SimulationWithAnalytics

# Logging
from .logger import logger

# Visualization (optional)
from .network_viz import load_brain_mesh, plot_complete_brain_network

__all__ = [
    "connectome",
    "BrainRegion", "NeuronSubclass",
    "Synapse",
    "Neuromodulator",
    "ExternalDrive", "StochasticDrive", "BurstDrive", "SleepArchitectureDrive",
    "AdaptiveDrive", "EnvironmentalDrive", "MultiFrequencyDrive",
    "Treatment", "SSRI", "SNRI", "Benzodiazepine", "CBT", "Exposure", "rTMS",
    "Mindfulness", "SleepTherapy", "TreatmentManager",
    "simple_firing_rates", "simple_spike_counts", "simple_plot",
    "intermediate_isi_cv", "intermediate_fano_factor", "intermediate_plot",
    "advanced_burst_detection", "advanced_synaptic_plasticity_trends",
    "advanced_neuromodulator_correlation", "advanced_spike_entropy", "advanced_plot",
    "run_all_analytics",
    "PatientProfile", "calculate_gad_severity", "set_simulation_parameters",
    "SimulationWithAnalytics",
    "logger",
    "load_brain_mesh", "plot_complete_brain_network"
]
