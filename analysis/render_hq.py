"""
render_hq.py  --  imagenes y animacion en alta resolucion de una corrida.

El GIF que produce analizar_produccion.py es pequeno y a baja resolucion para
que salga rapido.  Este script rehace lo mismo en grande, y ademas saca una
lamina del estado final visto desde cuatro angulos, que suele ser mas util que
la animacion para juzgar si el filamento esta torcido.

Uso:
    python3 analysis/render_hq.py --tag prod_soft_2x2_H100_v1
    python3 analysis/render_hq.py --tag prod_soft_2x2_H100_v1 --dpi 220 --size 10
    python3 analysis/render_hq.py --tag prod_soft_2x2_H100_v1 --sin-gif
"""
import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

from render import draw_state
from analysis import filament_twist, _layer_lattice_angle
from render import cube_faces
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from parameters import D_LATTICE

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def laminas(d, tag, dpi, size, indices=None):
    """Estado final desde cuatro azimuts + comparacion inicial/final."""
    n = len(d["cycles"])
    idx = indices if indices is not None else [0, n // 3, 2 * n // 3, n - 1]

    fig = plt.figure(figsize=(size, size), dpi=dpi)
    for k, az in enumerate((-60, 0, 60, 120)):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        draw_state(ax, d["pos"][-1], d["R"][-1], d["M"][-1], L=None,
                   arrows=False, azim=az, elev=12, lw=0.3)
        ax.set_title(f"azimut {az}°", fontsize=9)
    fig.suptitle(f"{tag} — estado final, ciclo {int(d['cycles'][-1])}", fontsize=11)
    fig.tight_layout()
    f1 = os.path.join(RES, f"{tag}_hq_final.png")
    fig.savefig(f1, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(size * 1.6, size * 0.55), dpi=dpi)
    for k, i in enumerate(idx):
        ax = fig.add_subplot(1, len(idx), k + 1, projection="3d")
        draw_state(ax, d["pos"][i], d["R"][i], d["M"][i], L=None,
                   arrows=False, azim=-60, elev=12, lw=0.3)
        tw = filament_twist(d["pos"][i])
        ax.set_title(f"ciclo {int(d['cycles'][i])}\n"
                     f"torsion {tw['total_twist']:.0f}°", fontsize=9)
    fig.tight_layout()
    f2 = os.path.join(RES, f"{tag}_hq_evolucion.png")
    fig.savefig(f2, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return f1, f2


def perfil_angulo(pos, layer_h=None):
    """Angulo acumulado del reticulo de cada capa a lo largo del filamento."""
    layer_h = layer_h or 0.6 * D_LATTICE
    z = pos[:, 2]
    bins = np.floor((z - z.min()) / layer_h).astype(int)
    zs, th = [], []
    for b in np.unique(bins):
        sel = bins == b
        if sel.sum() < 2:
            continue
        a_ = _layer_lattice_angle(pos[sel][:, :2])
        if a_ is None:
            continue
        zs.append(z[sel].mean())
        th.append(a_)
    if len(th) < 2:
        return np.array([]), np.array([])
    d = np.diff(th)
    d = (d + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return np.array(zs) - zs[0], np.degrees(np.concatenate([[0.0], np.cumsum(d)]))


def lamina_perfil(d, tag, dpi, size):
    """
    La prueba mas limpia de torsion: el angulo del reticulo de cada capa en
    funcion de la altura.  Una helice es una RECTA con pendiente sostenida;
    una cinta plana es una linea horizontal; una fluctuacion termica es un
    camino aleatorio que no mantiene pendiente.
    """
    n = len(d["cycles"])
    idx = [0, n // 3, 2 * n // 3, n - 1]
    fig, ax = plt.subplots(figsize=(size * 1.1, size * 0.6), dpi=dpi)
    colores = ["#9aa3ad", "#3b6ea5", "#e0a72c", "#c1443c"]
    for i, c in zip(idx, colores):
        z, phi = perfil_angulo(d["pos"][i])
        if len(z):
            ax.plot(z, phi, lw=2 if i == n - 1 else 1.4, color=c,
                    label=f"ciclo {int(d['cycles'][i])}")
            if i == n - 1 and len(z) > 3:
                m, b = np.polyfit(z, phi, 1)
                ax.plot(z, m * z + b, "k--", lw=1,
                        label=f"ajuste lineal final: {m:+.3f} deg/nm")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("altura a lo largo del filamento [nm]")
    ax.set_ylabel("angulo acumulado del reticulo [deg]")
    ax.set_title(f"{tag}: perfil de torsion\n"
                 "helice = recta inclinada · cinta = linea horizontal · "
                 "ruido = camino aleatorio", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout()
    out = os.path.join(RES, f"{tag}_hq_perfil.png")
    fig.savefig(out, dpi=dpi); plt.close(fig)
    return out


def lamina_segmento(d, tag, dpi, size, n_capas=14):
    """Tramo central del filamento, con proporciones reales, coloreado por altura."""
    p, R = d["pos"][-1], d["R"][-1]
    z = p[:, 2]
    zc = np.median(z)
    h = n_capas * D_LATTICE / 2
    sel = np.abs(z - zc) < h
    p, R = p[sel], R[sel]
    cmap = plt.get_cmap("viridis")
    col = cmap((p[:, 2] - p[:, 2].min()) / max(np.ptp(p[:, 2]), 1e-9))
    fig = plt.figure(figsize=(size * 1.2, size * 0.9), dpi=dpi)
    vistas = [(-60, 12, "lateral"), (30, 12, "lateral girada 90°"),
              (-60, 89, "desde arriba")]
    for k, (az, el, nombre) in enumerate(vistas):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        pc = Poly3DCollection(cube_faces(p, R), facecolors=np.repeat(col, 6, axis=0),
                              edgecolors=(0, 0, 0, 0.35), linewidths=0.3)
        ax.add_collection3d(pc)
        c = p.mean(axis=0)
        rng = np.ptp(p, axis=0) / 2 + 12
        ax.set_xlim(c[0] - rng[0], c[0] + rng[0])
        ax.set_ylim(c[1] - rng[1], c[1] + rng[1])
        ax.set_zlim(c[2] - rng[2], c[2] + rng[2])
        ax.set_box_aspect(tuple(rng))
        ax.view_init(elev=el, azim=az)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title(nombre, fontsize=9)
    fig.suptitle(f"{tag}: tramo central de {n_capas} capas, estado final "
                 "(color = altura)", fontsize=10)
    fig.tight_layout()
    out = os.path.join(RES, f"{tag}_hq_segmento.png")
    fig.savefig(out, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return out


def gif(d, tag, dpi, size, fps, spin, cada):
    sel = list(range(0, len(d["cycles"]), max(1, cada)))
    if sel[-1] != len(d["cycles"]) - 1:
        sel.append(len(d["cycles"]) - 1)
    fig = plt.figure(figsize=(size, size), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    out = os.path.join(RES, f"{tag}_hq_anim.gif")
    w = PillowWriter(fps=fps)
    w.setup(fig, out, dpi=dpi)
    for j, k in enumerate(sel):
        draw_state(ax, d["pos"][k], d["R"][k], d["M"][k], L=None,
                   arrows=False, azim=-60 + spin * j, elev=12, lw=0.3)
        tw = filament_twist(d["pos"][k])
        ax.set_title(f"{tag}\nciclo {int(d['cycles'][k])}   "
                     f"torsion {tw['total_twist']:.0f}°", fontsize=10)
        w.grab_frame()
    w.finish()
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--size", type=float, default=8.0, help="pulgadas")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--spin", type=float, default=1.5, help="grados por cuadro")
    ap.add_argument("--cada", type=int, default=2, help="usar 1 de cada N cuadros")
    ap.add_argument("--sin-gif", action="store_true")
    a_ = ap.parse_args()

    d = np.load(os.path.join(RES, f"{a_.tag}_frames.npz"))
    print(f"{len(d['cycles'])} cuadros, {d['pos'].shape[1]} nanocubos")
    for f in (lamina_perfil(d, a_.tag, a_.dpi, a_.size),
              lamina_segmento(d, a_.tag, a_.dpi, a_.size),
              *laminas(d, a_.tag, a_.dpi, a_.size)):
        print("->", f, f"{os.path.getsize(f)/1e6:.1f} MB")
    if not a_.sin_gif:
        f = gif(d, a_.tag, a_.dpi, a_.size, a_.fps, a_.spin, a_.cada)
        print("->", f, f"{os.path.getsize(f)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
