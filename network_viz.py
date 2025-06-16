import numpy as np
import pyvista as pv
from nilearn import datasets, surface
import matplotlib.colors as mcolors  # for color conversion

# Brain region MNI coordinates (example)
REGION_MNI_COORDS = {
    "LA": [-25, -5, -15], "BLA": [-30, -6, -20], "CeA": [-20, -4, -10],
    "DLPFC": [-40, 30, 30], "vmPFC": [0, 50, -10], "OFC": [25, 30, -15],
    "mPFC": [0, 50, 20], "dACC": [0, 20, 35], "rACC": [0, 40, 10],
    "DG": [-20, -40, -10], "CA3": [-30, -30, -10], "CA1": [-35, -25, -15],
    "CA2": [-32, -27, -12], "Subiculum": [-25, -35, -20], "AI": [32, 20, 0],
    "PI": [38, -10, 10], "dPAG": [0, -30, -20], "lPAG": [5, -32, -18],
    "vlPAG": [-5, -28, -22], "aBNST": [0, 0, -5], "pBNST": [0, -10, -5]
}

def load_brain_mesh():
    """
    Loads the fsaverage cortical mesh from nilearn datasets and merges hemispheres.
    """
    fsaverage = datasets.fetch_surf_fsaverage()
    coords_l, faces_l = surface.load_surf_mesh(fsaverage['pial_left'])
    coords_r, faces_r = surface.load_surf_mesh(fsaverage['pial_right'])
    coords = np.vstack((coords_l, coords_r))
    faces = np.vstack((faces_l, faces_r + coords_l.shape[0]))
    # PyVista expects faces in format: [n_points, p0, p1, p2, ...]
    faces_flat = np.hstack([np.concatenate(([3], face)) for face in faces])
    return pv.PolyData(coords, faces_flat)


def plot_complete_brain_network(sim, synapse_weight_threshold=0.0, show_labels=False):
    plotter = pv.Plotter(window_size=[1400, 900])
    brain_mesh = load_brain_mesh()
    plotter.add_mesh(brain_mesh, opacity=0.1, color="white")

    neuron_positions = []
    neuron_colors = []
    neuron_labels = []

    # Collect neuron positions and colors
    for region_name, region in sim.regions.items():
        for pop in region.populations:
            coords = pop.coordinates
            neuron_positions.append(coords)
            color_name = 'red' if pop.neuron_type == 'excitatory' else 'blue'
            neuron_colors.extend([color_name] * pop.count)
            if show_labels:
                neuron_labels.extend([f"{region_name}_{pop.neuron_type}_{i}" for i in range(pop.count)])

    neuron_positions = np.vstack(neuron_positions)
    print("Neuron positions shape:", neuron_positions.shape, "dtype:", neuron_positions.dtype)

    neuron_colors_rgb_float = np.array([mcolors.to_rgb(c) for c in neuron_colors])
    print("neuron_colors_rgb_float shape:", neuron_colors_rgb_float.shape, "dtype:", neuron_colors_rgb_float.dtype)
    print("Sample colors (float):", neuron_colors_rgb_float[:5])

    # Convert float RGB to uint8 RGBA
    neuron_colors_uint8 = (neuron_colors_rgb_float * 255).astype(np.uint8)
    alpha_channel = 255 * np.ones((neuron_colors_uint8.shape[0], 1), dtype=np.uint8)
    neuron_colors_rgba = np.hstack((neuron_colors_uint8, alpha_channel))
    print("neuron_colors_rgba shape:", neuron_colors_rgba.shape, "dtype:", neuron_colors_rgba.dtype)
    print("Sample colors (uint8 RGBA):", neuron_colors_rgba[:5])

    # Try method 1: scalars=RGBA uint8 with rgba=True
    try:
        print("Trying add_points with scalars=RGBA uint8 and rgba=True")
        plotter.add_points(
            neuron_positions,
            scalars=neuron_colors_rgba,
            rgba=True,
            point_size=5,
            render_points_as_spheres=True
        )
        print("Success: add_points with scalars=RGBA uint8")
    except Exception as e:
        print("Failed add_points with scalars=RGBA uint8:", e)

        # Fallback 1: float RGB without alpha and rgba
        try:
            print("Trying add_points with color=float RGB (0-1), no rgba")
            plotter.add_points(
                neuron_positions,
                color=neuron_colors_rgb_float,
                point_size=5,
                render_points_as_spheres=True
            )
            print("Success: add_points with float RGB")
        except Exception as e2:
            print("Failed add_points with float RGB:", e2)

            # Fallback 2: scalar integer + colormap
            try:
                print("Trying scalar integer + colormap fallback")
                neuron_type_map = {'excitatory': 0, 'inhibitory': 1}
                scalars = []
                for region in sim.regions.values():
                    for pop in region.populations:
                        val = neuron_type_map.get(pop.neuron_type, 0)
                        scalars.extend([val] * pop.count)
                scalars = np.array(scalars)

                plotter.add_points(
                    neuron_positions,
                    scalars=scalars,
                    cmap=['red', 'blue'],
                    point_size=5,
                    render_points_as_spheres=True
                )
                print("Success: add_points with scalars and colormap")
            except Exception as e3:
                print("Failed scalar integer + colormap fallback:", e3)
                raise e3  # re-raise if all fail

    if show_labels:
        plotter.add_point_labels(neuron_positions, neuron_labels,
                                 font_size=8, text_color='black')

    for syn in sim.synapses:
        if syn.mu < synapse_weight_threshold:
            continue
        pre_pop = getattr(syn, 'pre_population', None)
        post_pop = getattr(syn, 'post_population', None)
        pre_idx = getattr(syn, 'pre_neuron', None)
        post_idx = getattr(syn, 'post_neuron', None)
        if pre_pop is None or post_pop is None or pre_idx is None or post_idx is None:
            continue

        src_pos = pre_pop.coordinates[pre_idx]
        tgt_pos = post_pop.coordinates[post_idx]

        line = pv.Line(src_pos, tgt_pos)
        line_color = 'red' if not syn.is_inhibitory else 'blue'
        plotter.add_mesh(line, color=line_color, line_width=1, opacity=0.4)

    plotter.add_axes()
    plotter.add_text("Dynamic Neuron-level Brain Network Visualization",
                     position='upper_edge', font_size=14, color='black')
    plotter.show(interactive=True)
