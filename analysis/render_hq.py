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
from analysis import filament_twist

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
    for f in laminas(d, a_.tag, a_.dpi, a_.size):
        print("->", f, f"{os.path.getsize(f)/1e6:.1f} MB")
    if not a_.sin_gif:
        f = gif(d, a_.tag, a_.dpi, a_.size, a_.fps, a_.spin, a_.cada)
        print("->", f, f"{os.path.getsize(f)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
