"""
gif_hebra_hq.py -- animacion 3D en alta calidad de una hebra torciendose.

Es la vista "normal" del filamento: vertical, con la camara girando despacio,
cada cara sombreada segun su orientacion y los cubos coloreados a elegir.
Pensado para presentaciones: 150 dpi por defecto.

Uso (desde la raiz del proyecto, con el entorno activado):
    python analysis/gif_hebra_hq.py --tag prod_soft_3x3_H167_v1
    python analysis/gif_hebra_hq.py                      # las cuatro de lambda = 1
    python analysis/gif_hebra_hq.py --tag <tag> --color torsion --segmento 20
    python analysis/gif_hebra_hq.py --tag <tag> --dpi 200 --cada 1 --fps 12 --giro 0.8

Opciones:
    --color altura | torsion | dipolo
            altura  : degradado a lo largo de la hebra (el mas legible)
            torsion : rojo/azul segun cuanto giro la capa de cada cubo
            dipolo  : componente vertical del momento magnetico
    --segmento N  dibuja solo las N capas centrales (cubos mas grandes)
    --giro G      grados que gira la camara por cuadro (0 = camara quieta)
    --cada N      usa 1 de cada N cuadros guardados
Sale: results/<tag>_hebra_hq.gif
"""
import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.collections import PolyCollection

from analysis import _layer_lattice_angle, filament_twist
from render import cube_faces
from parameters import D_LATTICE

RES = os.path.join(os.path.dirname(__file__), "..", "results")
CUATRO = ["prod_soft_3x3_H167_v1", "prod_soft_3x3_H417_v1",
          "prod_soft_2x2_H100_v1", "prod_soft_2x2_H417_v1"]
LUZ = np.array([0.4, 0.5, 0.75])
LUZ = LUZ / np.linalg.norm(LUZ)


def angulo_por_cubo(pos):
    """Giro acumulado de la capa de cada cubo, en grados."""
    z = pos[:, 2]
    bins = np.floor((z - z.min()) / (0.6 * D_LATTICE)).astype(int)
    zs, th = [], []
    for b in np.unique(bins):
        sel = bins == b
        if sel.sum() < 2:
            continue
        a_ = _layer_lattice_angle(pos[sel][:, :2])
        if a_ is None:
            continue
        zs.append(z[sel].mean()); th.append(a_)
    if len(th) < 2:
        return np.zeros(len(pos))
    d = np.diff(th)
    d = (d + np.pi / 4) % (np.pi / 2) - np.pi / 4
    acum = np.degrees(np.concatenate([[0.0], np.cumsum(d)]))
    return np.interp(z, np.array(zs), acum)


def sombrear(caras, cols):
    """Oscurece cada cara segun el angulo entre su normal y la luz."""
    v = np.asarray(caras)
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    b = 0.55 + 0.45 * np.clip(np.abs(n @ LUZ), 0, 1)
    out = cols.copy()
    out[:, :3] = np.clip(out[:, :3] * b[:, None], 0, 1)
    return out


def gif(tag, cada, dpi, fps, giro, modo, segmento):
    ruta = os.path.join(RES, f"{tag}_frames.npz")
    if not os.path.exists(ruta):
        print(f"[omitida] no existe {ruta}")
        return
    d = np.load(ruta)
    cyc, P, R = d["cycles"], d["pos"], d["R"]
    M = d["M"] if "M" in d.files else np.zeros_like(P)
    sel = list(range(0, len(cyc), max(1, cada)))
    if sel[-1] != len(cyc) - 1:
        sel.append(len(cyc) - 1)
    total = np.array([filament_twist(P[k])["total_twist"] for k in range(len(cyc))])
    print(f"{tag}: {P.shape[1]} nanocubos, {len(sel)} cuadros por dibujar")

    # recorte al segmento central, si se pide
    if segmento:
        z0 = P[0][:, 2]
        centro = 0.5 * (z0.min() + z0.max())
        media = segmento * 0.5 * D_LATTICE
        keep = np.abs(z0 - centro) < media
        P, R, M = P[:, keep], R[:, keep], M[:, keep]
        print(f"  segmento central: {keep.sum()} nanocubos")

    if modo == "torsion":
        ang = {k: angulo_por_cubo(P[k]) for k in sel}
        vmax = max(20.0, np.percentile(np.abs(np.concatenate(list(ang.values()))), 98))
        norm, cmap = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), plt.get_cmap("coolwarm")
        etiqueta = "giro de la capa [°]"
    elif modo == "dipolo":
        norm, cmap = Normalize(-1, 1), plt.get_cmap("RdBu_r")
        etiqueta = "componente vertical del dipolo"
    else:
        z0 = P[0][:, 2]
        norm, cmap = Normalize(z0.min(), z0.max()), plt.get_cmap("viridis")
        etiqueta = "altura en la hebra"

    fig = plt.figure(figsize=(4.6, 9.5), dpi=dpi)
    ax = fig.add_axes([0.02, 0.02, 0.78, 0.90])
    cax = fig.add_axes([0.86, 0.25, 0.028, 0.45])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax).set_label(etiqueta, fontsize=11)
    titulo = fig.text(0.46, 0.955, "", ha="center", fontsize=13)

    ctr = P[0].mean(0)

    out = os.path.join(RES, f"{tag}_hebra_hq.gif")
    w = PillowWriter(fps=fps)
    w.setup(fig, out, dpi=dpi)
    for j, k in enumerate(sel):
        pos = P[k] - ctr
        if modo == "torsion":
            val = ang[k]
        elif modo == "dipolo":
            val = M[k][:, 2]
        else:
            val = P[0][:, 2] - ctr[2]
        # camara: giramos la escena alrededor del eje de la hebra
        t = np.radians(-60 + giro * j)
        c, sn = np.cos(t), np.sin(t)
        G = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1.0]])
        caras = np.asarray(cube_faces(pos @ G.T, np.einsum("ij,njk->nik", G, R[k])))
        prof = caras[:, :, 1].mean(1)
        orden = np.argsort(prof)
        # proyeccion oblicua: y aporta la profundidad
        xy = np.stack([caras[:, :, 0] + 0.30 * caras[:, :, 1],
                       caras[:, :, 2] + 0.12 * caras[:, :, 1]], axis=-1)
        cols = sombrear(caras, np.repeat(cmap(norm(val)), 6, axis=0))
        ax.clear()
        ax.add_collection(PolyCollection(xy[orden], facecolors=cols[orden],
                                         edgecolors=(0, 0, 0, 0.30), linewidths=0.25))
        ax.set_xlim(xy[..., 0].min() - 6, xy[..., 0].max() + 6)
        ax.set_ylim(xy[..., 1].min() - 6, xy[..., 1].max() + 6)
        ax.set_aspect("equal"); ax.set_axis_off()
        titulo.set_text(f"{tag}\nciclo {int(cyc[k])}  ·  torsión total {total[k]:+.0f}°")
        w.grab_frame()
        if j % 20 == 0:
            print(f"  cuadro {j + 1}/{len(sel)}", flush=True)
    w.finish()
    plt.close(fig)
    print(f"-> {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", help="una corrida; sin esto hace las cuatro de lambda = 1")
    ap.add_argument("--color", default="altura", choices=["altura", "torsion", "dipolo"])
    ap.add_argument("--segmento", type=int, default=0, help="solo las N capas centrales")
    ap.add_argument("--cada", type=int, default=2)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--giro", type=float, default=0.5, help="grados de cámara por cuadro")
    a = ap.parse_args()
    for t in ([a.tag] if a.tag else CUATRO):
        gif(t, a.cada, a.dpi, a.fps, a.giro, a.color, a.segmento)


if __name__ == "__main__":
    main()
