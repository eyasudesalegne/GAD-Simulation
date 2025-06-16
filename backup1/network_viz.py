import numpy as np
import pyvista as pv
from nilearn import datasets, surface

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
    fsaverage = datasets.fetch_surf_fsaverage()
    coords_l, faces_l = surface.load_surf_mesh(fsaverage['pial_left'])
    coords_r, faces_r = surface.load_surf_mesh(fsaverage['pial_right'])
    coords = np.vstack((coords_l, coords_r))
    faces = np.vstack((faces_l, faces_r + coords_l.shape[0]))
    faces_flat = np.hstack([np.concatenate(([3], face)) for face in faces])
    return pv.PolyData(coords, faces_flat)

def plot_complete_brain_network(sim, show_inter=True, show_intra=False):
    plotter = pv.Plotter()

    brain_mesh = load_brain_mesh()
    plotter.add_mesh(brain_mesh, opacity=0.1, color="white")

    region_centers = {}
    for region_name, region in sim.regions.items():
        base_name = region_name.split('_')[0]
        center = np.array(REGION_MNI_COORDS.get(base_name, np.random.rand(3)*100))
        region_centers[region_name] = center
        color = 'red' if region_name.endswith('_E') else 'blue'
        plotter.add_mesh(pv.Sphere(radius=2, center=center), color=color)
        plotter.add_point_labels([center], [region_name], font_size=10, text_color='black')

    for syn in sim.synapses:
        src_pos = region_centers.get(syn.pre.name)
        tgt_pos = region_centers.get(syn.post.name)
        if src_pos is None or tgt_pos is None:
            continue
        if show_inter and syn.pre.name.split('_')[0] != syn.post.name.split('_')[0]:
            line = pv.Line(src_pos, tgt_pos)
            color = 'red' if not syn.is_inhibitory else 'blue'
            plotter.add_mesh(line, color=color, line_width=2, opacity=0.5)
        elif show_intra and syn.pre.name.split('_')[0] == syn.post.name.split('_')[0]:
            line = pv.Line(src_pos, tgt_pos)
            color = 'green'
            plotter.add_mesh(line, color=color, line_width=1, opacity=0.3)

    plotter.add_axes()
    plotter.add_text("Optimized Brain Network Visualization", position='upper_edge', font_size=12, color='black')
    plotter.show(interactive=True)

def plot_brain_mesh(sim):
    """
    Alternative 3D brain visualization compatible with GUI call.
    Visualizes brain regions as spheres and connections as lines.
    """
    plotter = pv.Plotter()

    brain_mesh = load_brain_mesh()
    plotter.add_mesh(brain_mesh, opacity=0.2, color='lightgray')

    region_centers = {}
    for region_name, region in sim.regions.items():
        base_name = region_name.split('_')[0]
        center = np.array(REGION_MNI_COORDS.get(base_name, np.random.rand(3)*100))
        region_centers[region_name] = center
        color = 'red' if region_name.endswith('_E') else 'blue'
        plotter.add_mesh(pv.Sphere(radius=2, center=center), color=color, opacity=0.9)

    for syn in sim.synapses:
        src_pos = region_centers.get(syn.pre.name)
        tgt_pos = region_centers.get(syn.post.name)
        if src_pos is None or tgt_pos is None:
            continue
        line = pv.Line(src_pos, tgt_pos)
        color = 'red' if not syn.is_inhibitory else 'blue'
        plotter.add_mesh(line, color=color, line_width=1, opacity=0.5)

    plotter.add_axes()
    plotter.add_text("3D Brain Mesh Visualization", position='upper_edge', font_size=12, color='black')
    plotter.show()
