"""
gif_densidad_clusters.py -- animacion del barrido de densidad, mejorada.

  * misma escala en los cuatro paneles: los cubos se ven del mismo tamaño y lo
    que cambia a la vista es la caja (dibujada en gris);
  * cada cubo coloreado segun el agregado al que pertenece: gris = suelto o en
    par, colores = agregados de 3 o mas (rojo = el mayor);
  * en cada panel, la fraccion ensamblada en ese instante.

Uso:  python3 analysis/gif_densidad_clusters.py
Sale: results/anim_densidad_clusters.gif
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from render import cube_faces
from analysis import cluster_labels

RES = os.path.join(os.path.dirname(__file__), "..", "results")
TAGS = ["phi0p010", "phi0p050", "phi0p120", "phi0p250"]
PAL = ["#C1443C", "#3B6EA5", "#2E7D5B", "#E0A72C", "#7B4EA3", "#1F9AA8", "#B5651D", "#D15A9A"]
SUELTO = "#C9CED6"
data = [np.load(os.path.join(RES, f"density_{t}_frames.npz")) for t in TAGS]
Lmax = max(float(d["L"]) for d in data)

def colores(pos):
    lab = cluster_labels(pos)
    u, n = np.unique(lab, return_counts=True)
    grandes = [x for _, x in sorted(zip(-n, u)) if (lab == x).sum() >= 3]
    mapa = {x: PAL[i % len(PAL)] for i, x in enumerate(grandes)}
    col = [mapa.get(l, SUELTO) for l in lab]
    frac = sum((lab == x).sum() for x in grandes) / len(pos)
    return col, frac

def caja(ax, L, o):
    c = np.array([[x, y, z] for x in (0, L) for y in (0, L) for z in (0, L)]) + o
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(np.abs(c[i] - c[j]) > 1e-6) == 1:
                ax.plot(*zip(c[i], c[j]), color="#8A93A0", lw=0.8)

fig = plt.figure(figsize=(16, 4.8), dpi=90)
fig.subplots_adjust(left=0, right=1, bottom=-0.04, top=0.74, wspace=0)
axs = [fig.add_subplot(1, 4, i + 1, projection="3d") for i in range(4)]
n = min(len(d["cycles"]) for d in data)
w = PillowWriter(fps=7)
out = os.path.join(RES, "anim_densidad_clusters.gif")
w.setup(fig, out, dpi=90)
for k in range(n):
    for ax, d in zip(axs, data):
        ax.clear()
        L = float(d["L"]); o = np.full(3, (Lmax - L) / 2)     # caja centrada
        pos = d["pos"][k] + o
        col, frac = colores(d["pos"][k])
        pc = Poly3DCollection(cube_faces(pos, d["R"][k]), facecolors=np.repeat(col, 6),
                              edgecolors=(0, 0, 0, 0.35), linewidths=0.2)
        ax.add_collection3d(pc)
        caja(ax, L, o)
        for f in (ax.set_xlim, ax.set_ylim, ax.set_zlim): f(0, Lmax)
        ax.set_box_aspect((1, 1, 1), zoom=1.08); ax.view_init(elev=16, azim=-60)
        ax.set_axis_off()
        ax.set_title(f"φ = {float(d['phi']):.2f} · caja {L:.0f} nm\nensamblado: {100*frac:.0f} %", fontsize=13)
    fig.suptitle(f"Los mismos 56 nanocubos en cajas cada vez más pequeñas (417 G) · ciclo {int(data[0]['cycles'][k])}\n"
                 "gris: sueltos o en pares · color: cada agregado de 3 o más (rojo = el mayor)", fontsize=13)
    w.grab_frame()
w.finish()
print("->", out, os.path.getsize(out) // 1024, "kB")
