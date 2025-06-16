import sys
import os
import threading
import openai
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QComboBox, QTabWidget, QFileDialog, QTextEdit, QProgressBar,
    QMessageBox, QFormLayout, QSizePolicy, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from brain_sim.simulation import Simulation
from brain_sim.diagnostics import PatientProfile, calculate_gad_severity, set_simulation_parameters
from brain_sim.treatments import apply_treatment
from brain_sim.analytics import (
    plot_raster, plot_spike_heatmap, analyze_burst_statistics,
    analyze_isi, plot_all_metrics, plot_cross_correlation,
    plot_neuromodulators, plot_power_spectrum,
    plot_autocorrelation, plot_synaptic_weight_distribution
)
from brain_sim.network_viz import plot_brain_mesh
from brain_sim.logger import logger


class SimulationRunner(QThread):
    finished = pyqtSignal(object)  # emits Simulation instance
    log_message = pyqtSignal(str)
    progress_message = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params
        self._is_running = True

    def run(self):
        original_callback = logger.gui_callback
        logger.set_gui_callback(self.emit_log)

        try:
            duration = self.params['duration']
            dt = self.params['dt']
            anxiety = self.params['anxiety']
            stress = self.params['stress']
            cortisol = self.params['cortisol']
            hrv = self.params['hrv']
            sleep = self.params['sleep']
            treatment = self.params['treatment']
            intensity = self.params['intensity']
            dosage = self.params['dosage']
            frequency = self.params['frequency']

            self.log_message.emit("Building simulation...")
            sim = Simulation(dt=dt, duration=duration)

            if None not in (anxiety, stress, cortisol, hrv, sleep):
                patient = PatientProfile(anxiety, stress, cortisol, hrv, sleep)
                severity = calculate_gad_severity(patient)
                self.log_message.emit(f"Calculated GAD severity: {severity:.2f}")
                set_simulation_parameters(sim, severity)
            else:
                severity = 0.0
                self.log_message.emit("No patient profile given, using severity=0.0")

            if treatment and treatment != "None":
                self.log_message.emit(f"Applying treatment: {treatment}")
                apply_treatment(sim, treatment, intensity, dosage, frequency)

            self.progress_message.emit("Running simulation...")

            # Run simulation incrementally with chunk_size=200 for matched logging frequency
            for progress_frac in sim.run_incremental(chunk_size=200):
                if not self._is_running:
                    self.log_message.emit("Simulation stopped by user.")
                    break
                self.progress_message.emit(f"Simulation progress: {progress_frac*100:.1f}%")

            if self._is_running:
                self.log_message.emit("Simulation complete.")
        finally:
            logger.set_gui_callback(original_callback)

        self.finished.emit(sim)

    def stop(self):
        self._is_running = False

    def emit_log(self, msg):
        self.log_message.emit(msg)


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots(figsize=(6, 4), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        plt.tight_layout()

    def clear_plot(self):
        self.ax.clear()

    def plot_func(self, plot_function, sim, **kwargs):
        self.clear_plot()
        plot_function(sim, ax=self.ax, **kwargs)
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Brain Simulation GUI (PyQt5)")
        self.resize(1300, 1050)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        left_panel = QVBoxLayout()
        main_layout.addLayout(left_panel, 1)

        # Patient Profile Inputs
        patient_group = QWidget()
        patient_layout = QFormLayout()
        patient_group.setLayout(patient_layout)
        patient_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        patient_group.setMinimumWidth(350)
        patient_group.setMaximumWidth(370)
        left_panel.addWidget(QLabel("<b>Patient Profile Inputs</b>"))
        left_panel.addWidget(patient_group)

        self.input_anxiety = QDoubleSpinBox()
        self.input_anxiety.setRange(0, 21)
        self.input_anxiety.setDecimals(2)
        self.input_anxiety.setValue(0.0)

        self.input_stress = QDoubleSpinBox()
        self.input_stress.setRange(0, 100)
        self.input_stress.setDecimals(2)
        self.input_stress.setValue(0.0)

        self.input_cortisol = QDoubleSpinBox()
        self.input_cortisol.setRange(5, 25)
        self.input_cortisol.setDecimals(2)
        self.input_cortisol.setValue(5.0)

        self.input_hrv = QDoubleSpinBox()
        self.input_hrv.setRange(0, 100)
        self.input_hrv.setDecimals(2)
        self.input_hrv.setValue(0.0)

        self.input_sleep = QDoubleSpinBox()
        self.input_sleep.setRange(0, 10)
        self.input_sleep.setDecimals(2)
        self.input_sleep.setValue(0.0)

        patient_layout.addRow("Anxiety Score (0-21):", self.input_anxiety)
        patient_layout.addRow("Stress Level (0-100):", self.input_stress)
        patient_layout.addRow("Cortisol Level (5-25):", self.input_cortisol)
        patient_layout.addRow("Heart Rate Variability (0-100):", self.input_hrv)
        patient_layout.addRow("Sleep Quality (0-10):", self.input_sleep)

        # Simulation Parameters
        left_panel.addWidget(QLabel("<b>Simulation Parameters</b>"))
        sim_params_group = QWidget()
        sim_params_layout = QFormLayout()
        sim_params_group.setLayout(sim_params_layout)
        left_panel.addWidget(sim_params_group)

        self.input_duration = QDoubleSpinBox()
        self.input_duration.setRange(1.0, 300.0)
        self.input_duration.setDecimals(2)
        self.input_duration.setValue(10.0)

        self.input_dt = QDoubleSpinBox()
        self.input_dt.setRange(0.0001, 0.01)
        self.input_dt.setDecimals(5)
        self.input_dt.setValue(0.001)

        sim_params_layout.addRow("Duration (seconds):", self.input_duration)
        sim_params_layout.addRow("Time step dt (seconds):", self.input_dt)

        # Treatment Inputs
        left_panel.addWidget(QLabel("<b>Treatment</b>"))
        treatment_group = QWidget()
        treatment_layout = QFormLayout()
        treatment_group.setLayout(treatment_layout)
        left_panel.addWidget(treatment_group)

        self.combo_treatment = QComboBox()
        self.combo_treatment.addItems([
            "None", "SSRI", "SNRI", "Benzodiazepine", "CBT",
            "Exposure", "rTMS", "Mindfulness", "SleepTherapy"
        ])
        treatment_layout.addRow("Treatment Type:", self.combo_treatment)

        self.input_intensity = QDoubleSpinBox()
        self.input_intensity.setRange(0.0, 1.0)
        self.input_intensity.setDecimals(2)
        self.input_intensity.setValue(1.0)

        self.input_dosage = QDoubleSpinBox()
        self.input_dosage.setRange(0, 1000)
        self.input_dosage.setDecimals(2)
        self.input_dosage.setValue(20)

        self.input_frequency = QDoubleSpinBox()
        self.input_frequency.setRange(0, 100)
        self.input_frequency.setDecimals(2)
        self.input_frequency.setValue(1)

        treatment_layout.addRow("Intensity (0-1):", self.input_intensity)
        treatment_layout.addRow("Dosage (mg/sessions):", self.input_dosage)
        treatment_layout.addRow("Frequency (per week):", self.input_frequency)

        # Run, Stop, Reset, Analyze Buttons and Progress
        self.btn_run = QPushButton("Run Simulation")
        self.btn_stop = QPushButton("Stop Simulation")
        self.btn_reset = QPushButton("Reset GUI")
        self.btn_analyze = QPushButton("Analyze Logs")
        left_panel.addWidget(self.btn_run)
        left_panel.addWidget(self.btn_stop)
        left_panel.addWidget(self.btn_reset)
        left_panel.addWidget(self.btn_analyze)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        left_panel.addWidget(self.progress_bar)

        self.label_status = QLabel("Status: Idle")
        left_panel.addWidget(self.label_status)

        # Export Buttons
        self.btn_export_plots = QPushButton("Export All Plots as PNG")
        left_panel.addWidget(self.btn_export_plots)

        self.btn_export_data = QPushButton("Export Raw Spike Data as CSV")
        left_panel.addWidget(self.btn_export_data)

        # 3D Visualization
        self.btn_launch_3d = QPushButton("Show 3D Brain Visualization")
        left_panel.addWidget(self.btn_launch_3d)

        # Analysis result display
        self.text_analysis = QTextEdit()
        self.text_analysis.setReadOnly(True)
        self.text_analysis.setFixedHeight(150)
        left_panel.addWidget(self.text_analysis)

        left_panel.addStretch()

        # Tabs for Plots and Log
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs, 3)

        self.plot_widgets = {}

        plot_names = [
            "Analytics Dashboard",
            "Raster Plot",
            "Spike Heatmap",
            "Burst Statistics",
            "ISI Distribution",
            "Cross-Correlation",
            "Neuromodulator Levels",
            "Power Spectrum",
            "Autocorrelation",
            "Synaptic Weight Distribution"
        ]

        for name in plot_names:
            tab = QWidget()
            layout = QVBoxLayout()
            tab.setLayout(layout)
            canvas = PlotCanvas()
            layout.addWidget(canvas)
            toolbar = NavigationToolbar(canvas, self)
            layout.addWidget(toolbar)
            self.tabs.addTab(tab, name)
            self.plot_widgets[name] = canvas

        # Console Log Tab
        log_tab = QWidget()
        log_layout = QVBoxLayout()
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        log_layout.addWidget(self.text_log)
        log_tab.setLayout(log_layout)
        self.tabs.addTab(log_tab, "Console Log")

        # Buffer for logs
        self.log_buffer = []

        # Connect buttons
        self.btn_run.clicked.connect(self.on_run_simulation)
        self.btn_stop.clicked.connect(self.on_stop_simulation)
        self.btn_reset.clicked.connect(self.on_reset_gui)
        self.btn_export_plots.clicked.connect(self.on_export_plots)
        self.btn_export_data.clicked.connect(self.on_export_data)
        self.btn_launch_3d.clicked.connect(self.on_launch_3d)
        self.btn_analyze.clicked.connect(self.on_analyze_logs)

        self.simulation = None
        self.thread = None

    def buffer_log_message(self, msg: str):
        # Append log messages to buffer and GUI log with limit for performance
        self.log_buffer.append(msg)
        if len(self.log_buffer) > 1000:
            self.log_buffer = self.log_buffer[-1000:]
        self.text_log.append(msg)
        self.text_log.ensureCursorVisible()

    def update_status_label(self, msg: str):
        self.label_status.setText("Status: " + msg)

    def on_run_simulation(self):
        try:
            params = {
                'duration': float(self.input_duration.value()),
                'dt': float(self.input_dt.value()),
                'anxiety': float(self.input_anxiety.value()),
                'stress': float(self.input_stress.value()),
                'cortisol': float(self.input_cortisol.value()),
                'hrv': float(self.input_hrv.value()),
                'sleep': float(self.input_sleep.value()),
                'treatment': self.combo_treatment.currentText(),
                'intensity': float(self.input_intensity.value()),
                'dosage': float(self.input_dosage.value()),
                'frequency': float(self.input_frequency.value())
            }
        except Exception:
            QMessageBox.warning(self, "Input Error", "Please enter valid numeric values.")
            return

        if self.thread and self.thread.isRunning():
            QMessageBox.warning(self, "Warning", "A simulation is already running.")
            return

        self.btn_run.setEnabled(False)
        self.progress_bar.setRange(0, 0)  # Indeterminate while running
        self.label_status.setText("Status: Starting simulation...")
        self.text_log.clear()
        self.text_analysis.clear()
        self.log_buffer.clear()

        self.thread = SimulationRunner(params)
        self.thread.log_message.connect(self.buffer_log_message)
        self.thread.progress_message.connect(self.update_status_label)
        self.thread.finished.connect(self.on_simulation_finished)
        self.thread.start()

    def on_stop_simulation(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()
            self.label_status.setText("Status: Simulation stopped.")
            self.btn_run.setEnabled(True)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
        else:
            QMessageBox.information(self, "Info", "No simulation is running.")

    def on_reset_gui(self):
        if self.thread and self.thread.isRunning():
            QMessageBox.warning(self, "Warning", "Stop the simulation before resetting.")
            return

        self.input_anxiety.setValue(0.0)
        self.input_stress.setValue(0.0)
        self.input_cortisol.setValue(5.0)
        self.input_hrv.setValue(0.0)
        self.input_sleep.setValue(0.0)

        self.input_duration.setValue(10.0)
        self.input_dt.setValue(0.001)

        self.combo_treatment.setCurrentIndex(0)
        self.input_intensity.setValue(1.0)
        self.input_dosage.setValue(20)
        self.input_frequency.setValue(1)

        for canvas in self.plot_widgets.values():
            canvas.ax.clear()
            canvas.draw()

        self.text_log.clear()
        self.text_analysis.clear()
        self.log_buffer.clear()

        self.label_status.setText("Status: Idle")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.simulation = None

    def on_simulation_finished(self, sim):
        self.simulation = sim
        self.btn_run.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)

        self.label_status.setText("Status: Simulation finished successfully.")

        self.update_plots()

    def update_plots(self):
        if not self.simulation:
            return

        plot_raster(self.simulation, ax=self.plot_widgets["Raster Plot"].ax)
        self.plot_widgets["Raster Plot"].draw()

        plot_spike_heatmap(self.simulation, ax=self.plot_widgets["Spike Heatmap"].ax)
        self.plot_widgets["Spike Heatmap"].draw()

        analyze_burst_statistics(self.simulation, ax=self.plot_widgets["Burst Statistics"].ax)
        self.plot_widgets["Burst Statistics"].draw()

        analyze_isi(self.simulation, ax=self.plot_widgets["ISI Distribution"].ax)
        self.plot_widgets["ISI Distribution"].draw()

        plot_all_metrics(self.simulation, ax=self.plot_widgets["Analytics Dashboard"].ax)
        self.plot_widgets["Analytics Dashboard"].draw()

        plot_cross_correlation(self.simulation, 'BLA_E', 'mPFC_E', ax=self.plot_widgets["Cross-Correlation"].ax)
        self.plot_widgets["Cross-Correlation"].draw()

        plot_neuromodulators(self.simulation, ax=self.plot_widgets["Neuromodulator Levels"].ax)
        self.plot_widgets["Neuromodulator Levels"].draw()

        plot_power_spectrum(self.simulation, ax=self.plot_widgets["Power Spectrum"].ax)
        self.plot_widgets["Power Spectrum"].draw()

        plot_autocorrelation(self.simulation, ax=self.plot_widgets["Autocorrelation"].ax)
        self.plot_widgets["Autocorrelation"].draw()

        plot_synaptic_weight_distribution(self.simulation, ax=self.plot_widgets["Synaptic Weight Distribution"].ax)
        self.plot_widgets["Synaptic Weight Distribution"].draw()

    def on_export_plots(self):
        if not self.simulation:
            QMessageBox.warning(self, "No Data", "Run a simulation first!")
            return
        folder = QFileDialog.getExistingDirectory(self, "Select folder to save plots")
        if not folder:
            return
        for name, canvas in self.plot_widgets.items():
            filename = os.path.join(folder, name.replace(" ", "_") + ".png")
            canvas.fig.savefig(filename)
        QMessageBox.information(self, "Export Complete", f"Plots saved to {folder}")

    def on_export_data(self):
        if not self.simulation:
            QMessageBox.warning(self, "No Data", "Run a simulation first!")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save Spike Data CSV", filter="CSV Files (*.csv)")
        if not filename:
            return
        try:
            with open(filename, 'w') as f:
                f.write("time,neuron_id\n")
                for t, nid in self.simulation.spikes:
                    f.write(f"{t},{nid}\n")
            QMessageBox.information(self, "Export Complete", f"Spike data saved to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))

    def on_launch_3d(self):
        if not self.simulation:
            QMessageBox.warning(self, "No Data", "Run a simulation first!")
            return
        threading.Thread(target=plot_brain_mesh, args=(self.simulation,), daemon=True).start()

    def on_analyze_logs(self):
        if self.thread and self.thread.isRunning():
            QMessageBox.warning(self, "Warning", "Stop simulation before analyzing.")
            return

        logs = self.text_log.toPlainText().strip()
        if not logs:
            QMessageBox.warning(self, "Warning", "No logs to analyze.")
            return

        self.btn_analyze.setEnabled(False)
        self.text_analysis.setPlainText("Analyzing logs, please wait...")

        threading.Thread(target=self.call_chatgpt_api, args=(logs,), daemon=True).start()

    def call_chatgpt_api(self, logs):
        try:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            if not openai.api_key:
                raise ValueError("OpenAI API key not set in environment variable OPENAI_API_KEY.")

            prompt = (
                "You are a computational neuroscience expert specializing in brain network simulations related to psychiatric disorders.\n\n"
                "The project models Generalized Anxiety Disorder (GAD) by simulating multiple interconnected brain regions with populations "
                "of excitatory and inhibitory neurons.\n\n"
                "Key features of the simulation include:\n"
                "- Realistic connectome structure with heterogeneous synaptic weights and plasticity.\n"
                "- Neuromodulator systems including dopamine, serotonin, norepinephrine, acetylcholine, and cortisol affecting neuron dynamics.\n"
                "- Region-specific parameters controlling neuron properties and noise.\n"
                "- Simulation of treatment effects such as SSRIs, CBT, and benzodiazepines.\n"
                "- Spike trains, burst statistics, synaptic weights, and neuromodulator levels are logged in detail.\n\n"
                "The logs provided include time-stamped events, neuron firing rates, synaptic changes, and treatment application data.\n\n"
                "Your task is to:\n"
                "1. Provide a detailed summary of the simulation progress and final results.\n"
                "2. Identify and explain any abnormal or interesting neural activity patterns or neuromodulator fluctuations.\n"
                "3. Suggest potential improvements to the model or experimental parameters based on the data.\n"
                "4. Highlight any correlations or notable interactions between brain regions or neuromodulators.\n"
                "5. Offer insights into how treatment parameters may have influenced the simulation outcome.\n\n"
                "Please produce a clear, professional report suitable for a neuroscience research audience, focusing on key findings and interpretations from the simulation logs below.\n\n"
                "Simulation logs:\n"
                + logs
            )

            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=500,
            )
            answer = response['choices'][0]['message']['content']
        except Exception as e:
            answer = f"Error during analysis: {str(e)}"

        def update_text():
            self.text_analysis.setPlainText(answer)
            self.btn_analyze.setEnabled(True)

        QTimer.singleShot(0, update_text)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
