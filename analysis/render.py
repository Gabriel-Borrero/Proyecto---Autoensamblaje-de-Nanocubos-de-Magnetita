"""
render.py  --  visualizacion 3D.

Dibuja cada nanocubo como un cubo solido con su momento magnetico (flecha) y
produce:
  * snapshots (png)
  * animaciones (gif) de la trayectoria de Monte Carlo
  * animacion comparativa de varias densidades en paralelo
"""
import os, sys, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from parameters import a

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FACES = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
         (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
_V = np.array([[sx, sy, sz] for sx in (-.5, .5) for sy in (-.5, .5)
               for sz in (-.5, .5)])


def cube_faces(pos, R, scale=0.92):
    """Lista de caras (4x3) de todos los cubos."""
    verts = pos[:, None, :] + np.einsum("kj,nij->nki", _V * a * scale, R)
    return [verts[n][list(f)] for n in range(len(pos)) for f in FACES]


def draw_state(ax, pos, R, M, L=None, arrows=True, cmap="coolwarm",
               elev=18, azim=-60, alpha=0.95, lw=0.25):
    ax.clear()
    cols = plt.get_cmap(cmap)((M[:, 2] + 1) / 2)
    face_cols = np.repeat(cols, 6, axis=0)
    pc = Poly3DCollection(cube_faces(pos, R), facecolors=face_cols,
                          edgecolors=(0, 0, 0, 0.35), linewidths=lw, alpha=alpha)
    ax.add_collection3d(pc)
    if arrows:
        ax.quiver(pos[:, 0], pos[:, 1], pos[:, 2],
                  M[:, 0], M[:, 1], M[:, 2], length=1.15 * a, normalize=True,
                  color="k", linewidth=0.7, arrow_length_ratio=0.35)
    if L is None:
        c = pos.mean(axis=0)
        half = max(np.ptp(pos, axis=0).max() * 0.6, 2 * a)
        lim = np.array([c - half, c + half])
    else:
        lim = np.array([[0, 0, 0], [L, L, L]], dtype=float)
    ax.set_xlim(lim[0, 0], lim[1, 0])
    ax.set_ylim(lim[0, 1], lim[1, 1])
    ax.set_zlim(lim[0, 2], lim[1, 2])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.03)
    return ax


# ----------------------------------------------------------------------
def animate_one(npz_path, out_gif, title="", fps=8, box=True, spin=0.0,
                figsize=(4.6, 4.6), dpi=95):
    d = np.load(npz_path)
    L = float(d["L"]) if ("L" in d and box) else None
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    n = len(d["cycles"])
    w = PillowWriter(fps=fps)
    w.setup(fig, out_gif, dpi=dpi)
    for k in range(n):
        draw_state(ax, d["pos"][k], d["R"][k], d["M"][k], L=L,
                   azim=-60 + spin * k)
        ax.set_title(f"{title}\nciclo MC = {int(d['cycles'][k])}", fontsize=9)
        w.grab_frame()
    w.finish()
    plt.close(fig)
    return out_gif


def animate_multi(npz_paths, labels, out_gif, fps=8, dpi=95, spin=0.0,
                  suptitle="", per_panel=(3.1, 3.4)):
    data = [np.load(p) for p in npz_paths]
    n = min(len(d["cycles"]) for d in data)
    fig = plt.figure(figsize=(per_panel[0] * len(data), per_panel[1]), dpi=dpi)
    axes = [fig.add_subplot(1, len(data), i + 1, projection="3d")
            for i in range(len(data))]
    w = PillowWriter(fps=fps)
    w.setup(fig, out_gif, dpi=dpi)
    for k in range(n):
        for ax, d, lab in zip(axes, data, labels):
            L = float(d["L"]) if "L" in d else None
            draw_state(ax, d["pos"][k], d["R"][k], d["M"][k], L=L,
                       arrows=False, azim=-60 + spin * k, lw=0.15)
            ax.set_title(lab, fontsize=9)
        fig.suptitle(f"{suptitle}   ciclo MC = {int(data[0]['cycles'][k])}",
                     fontsize=11)
        w.grab_frame()
    w.finish()
    plt.close(fig)
    return out_gif


def snapshot_grid(npz_paths, labels, out_png, indices=(0, -1), dpi=150,
                  arrows=True, titles=None):
    data = [np.load(p) for p in npz_paths]
    nr, nc = len(indices), len(data)
    fig = plt.figure(figsize=(3.2 * nc, 3.4 * nr), dpi=dpi)
    for r, idx in enumerate(indices):
        for c, (d, lab) in enumerate(zip(data, labels)):
            ax = fig.add_subplot(nr, nc, r * nc + c + 1, projection="3d")
            k = idx if idx >= 0 else len(d["cycles"]) + idx
            L = float(d["L"]) if "L" in d else None
            draw_state(ax, d["pos"][k], d["R"][k], d["M"][k], L=L,
                       arrows=arrows, lw=0.2)
            t = f"{lab}\nciclo {int(d['cycles'][k])}"
            ax.set_title(t, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png

