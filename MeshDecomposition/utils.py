import random
import colorsys
import numpy as np

from scipy.spatial.transform import Rotation as R

from .Mesh import Mesh


def get_color(n_colors):
    colors = []
    for i in np.arange(0, 360, 360 / n_colors):
        hue = i / 360
        lightness = (50 + np.random.rand() * 10) / 100
        saturation = (90 + np.random.rand() * 10) / 100
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    random.shuffle(colors)
    return colors



def export_ply(file: str, mesh: Mesh, face_colors: np.ndarray):
    with open(file, 'w') as f:
        f.write(
f"""ply
format ascii 1.0
element vertex {len(mesh.verts)}
property float x
property float y
property float z
element face {mesh.num_faces}
property list uchar int vertex_indices
property uint8 red
property uint8 green
property uint8 blue
end_header
""")
        for v in mesh.verts:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for i, face in enumerate(mesh.faces):
            f.write(f"3 {face[0]} {face[1]} {face[2]} {face_colors[i][0]} {face_colors[i][1]} {face_colors[i][2]}\n")

