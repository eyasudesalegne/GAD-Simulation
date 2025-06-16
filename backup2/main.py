import argparse
from brain_sim.simulation import Simulation
from brain_sim.treatments import set_gad_severity, apply_treatment
from brain_sim.analytics import (plot_all_metrics, plot_raster,
                                 plot_spike_heatmap, analyze_burst_statistics, analyze_isi)
from brain_sim.desktop_gui import launch_dearpygui_gui
from brain_sim.diagnostics import PatientProfile, calculate_gad_severity, set_simulation_parameters
from brain_sim.network_viz import plot_complete_brain_network

def build_simulation(duration, dt):
    """Create and return a Simulation object with specified duration and dt."""
    return Simulation(dt=dt, duration=duration)

def main():
    parser = argparse.ArgumentParser(description="Run GAD brain network simulation.")

    parser.add_argument('--duration', type=float, default=10.0, help='Simulation duration in seconds.')
    parser.add_argument('--dt', type=float, default=0.001, help='Timestep in seconds.')
    parser.add_argument('--severity', type=float, default=0.0)
    parser.add_argument('--treatment', choices=[
        'SSRI','SNRI','Benzodiazepine','CBT','Exposure','rTMS','Mindfulness','SleepTherapy'
    ], default=None)
    parser.add_argument('--intensity', type=float, default=1.0)
    parser.add_argument('--dosage', type=float, default=None, help='Dosage (mg or sessions)')
    parser.add_argument('--frequency', type=int, default=None, help='Frequency (per week)')

    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--network_viz', action='store_true', help='Launch 3D network snapshot after simulation')
    parser.add_argument('--analytics', choices=["standard", "enhanced"], default="enhanced",
                        help="Select analytics view: 'standard' shows basic metrics; 'enhanced' includes heatmap, burst, and ISI analyses.")

    parser.add_argument('--anxiety', type=float)
    parser.add_argument('--stress', type=float)
    parser.add_argument('--cortisol', type=float)
    parser.add_argument('--hrv', type=float)
    parser.add_argument('--sleep', type=float)

    args = parser.parse_args()

    sim = build_simulation(args.duration, args.dt)

    if all(v is not None for v in (args.anxiety, args.stress, args.cortisol, args.hrv, args.sleep)):
        patient = PatientProfile(anxiety_score=args.anxiety,
                                 stress_level=args.stress,
                                 cortisol=args.cortisol,
                                 hr_variability=args.hrv,
                                 sleep_quality=args.sleep)
        severity = calculate_gad_severity(patient)
        print(f"Calculated GAD severity: {severity:.2f}")
        set_simulation_parameters(sim, severity)
    else:
        severity = args.severity
        print(f"Using manual GAD severity: {severity:.2f}")
        set_gad_severity(sim, severity)

    for name, region in sim.regions.items():
        print(f"Region {name} has {region.neuron_count} neurons.")

    if args.treatment:
        apply_treatment(sim, args.treatment, args.intensity,
                        dosage=args.dosage or 0, frequency=args.frequency or 0)

    if args.gui:
        launch_dearpygui_gui(sim)
    else:
        sim.run()
        print("Simulation complete.")

        if args.network_viz:
            print("Launching 3D visualization...")
            plot_complete_brain_network(sim, show_inter=True, show_intra=False)
        else:
            plot_all_metrics(sim)
            plot_raster(sim)
            if args.analytics == "enhanced":
                plot_spike_heatmap(sim)
                analyze_burst_statistics(sim)
                analyze_isi(sim)

if __name__ == '__main__':
    main()
