"""
gif_quiralidad.py -- animacion que hace VISIBLE la torsion de un filamento.

Cada cuadro tiene tres paneles:
  * arriba: el filamento acostado, con cada cubo coloreado segun cuanto ha
    girado su capa respecto a la de abajo (rojo = giro +, azul = giro -,
    blanco = sin girar).  Una helice se ve como un degradado de color a lo
    largo del filamento; una cinta recta, de un solo color.
  * abajo a la izquierda: el perfil de torsion (angulo acumulado frente a la
    altura) del cuadro actual, con los anteriores en gris.  Una helice seria
    una recta inclinada.
  * abajo a la derecha: la torsion total frente a los ciclos, con un punto
    que avanza.  Cada cruce por cero es un cambio de quiralidad.

Uso (desde la raiz del proyecto, con el entorno activado):
    python analysis/gif_quiralidad.py                      # las cuatro corridas a lambda = 1
    python analysis/gif_quiralidad.py --tag prod_soft_3x3_H167_v1
    python analysis/gif_quiralidad.py --tag <tag> --cada 1 --dpi 100 --fps 10

Sale: results/<tag>_quiralidad.gif
"""
import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.colors import TwoSlopeNorm
from matplotlib.collections import PolyCollection

from analysis import _layer_lattice_angle, filament_twist
from render import cube_faces
from parameters import D_LATTICE

RES = os.path.join(os.path.dirname(__file__), "..", "results")
CUATRO = ["prod_soft_3x3_H167_v1", "prod_soft_3x3_H417_v1",
          "prod_soft_2x2_H100_v1", "prod_soft_2x2_H417_v1"]


def perfil(pos, layer_h=None):
    """Angulo acumulado de cada capa y el angulo que le toca a cada cubo."""
    layer_h = layer_h or 0.6 * D_LATTICE
    z = pos[:, 2]
    bins = np.floor((z - z.min()) / layer_h).astype(int)
    zs, th, validos = [], [], []
    for b in np.unique(bins):
        sel = bins == b
        if sel.sum() < 2:
            continue
        a_ = _layer_lattice_angle(pos[sel][:, :2])
        if a_ is None:
            continue
        zs.append(z[sel].mean()); th.append(a_); validos.append(b)
    if len(th) < 2:
        return np.array([]), np.array([]), np.zeros(len(pos))
    d = np.diff(th)
    d = (d + np.pi / 4) % (np.pi / 2) - np.pi / 4          # red cuadrada: modulo 90 grados
    acum = np.degrees(np.concatenate([[0.0], np.cumsum(d)]))
    zs = np.array(zs)
    # cada cubo toma el angulo de su capa (o de la capa valida mas cercana)
    por_cubo = np.interp(z, zs, acum)
    return zs - zs[0], acum, por_cubo


def gif(tag, cada, dpi, fps):
    ruta = os.path.join(RES, f"{tag}_frames.npz")
    if not os.path.exists(ruta):
        print(f"[omitida] no existe {ruta}")
        return None
    d = np.load(ruta)
    cyc, P, R = d["cycles"], d["pos"], d["R"]
    sel = list(range(0, len(cyc), max(1, cada)))
    if sel[-1] != len(cyc) - 1:
        sel.append(len(cyc) - 1)
    print(f"{tag}: {len(cyc)} cuadros guardados, {P.shape[1]} nanocubos, se dibujan {len(sel)}")

    # torsion total de todos los cuadros (para la curva de la derecha y la escala de color)
    total = np.array([filament_twist(P[k])["total_twist"] for k in range(len(cyc))])
    perf = {k: perfil(P[k]) for k in sel}
    vmax = max(30.0, np.percentile(np.abs(np.concatenate([perf[k][2] for k in sel])), 98))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("coolwarm")
    saltos = int(np.sum(np.diff(np.sign(total[np.abs(total) > 1e-9])) != 0))

    fig = plt.figure(figsize=(16, 9), dpi=dpi)
    ax3 = fig.add_axes([0.03, 0.50, 0.87, 0.38])
    cax = fig.add_axes([0.92, 0.52, 0.012, 0.32])
    axp = fig.add_axes([0.06, 0.08, 0.42, 0.33])
    axt = fig.add_axes([0.56, 0.08, 0.40, 0.33])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("giro de la capa [°]")

    titulo = fig.text(0.5, 0.95, "", ha="center", fontsize=14)
    out = os.path.join(RES, f"{tag}_quiralidad.gif")
    w = PillowWriter(fps=fps)
    w.setup(fig, out, dpi=dpi)
    zmax = max(perf[k][0].max() if len(perf[k][0]) else 1.0 for k in sel)
    for j, k in enumerate(sel):
        zs, acum, por_cubo = perf[k]
        pos = P[k] - P[k].mean(0)
        caras = np.asarray(cube_faces(pos, R[k]))           # (6N, 4, 3)
        # proyeccion oblicua: la altura z en horizontal, x en vertical, y da profundidad
        prof = caras[:, :, 1].mean(1)
        orden = np.argsort(prof)                            # de atras hacia adelante
        xy = np.stack([caras[:, :, 2] + 0.25 * caras[:, :, 1],
                       caras[:, :, 0] + 0.45 * caras[:, :, 1]], axis=-1)
        cols = np.repeat(cmap(norm(por_cubo)), 6, axis=0)
        sombra = 0.80 + 0.20 * (prof - prof.min()) / (np.ptp(prof) + 1e-9)
        cols[:, :3] *= sombra[:, None]
        ax3.clear()
        ax3.add_collection(PolyCollection(xy[orden], facecolors=cols[orden],
                                          edgecolors=(0, 0, 0, 0.25), linewidths=0.2))
        ax3.set_xlim(xy[..., 0].min() - 10, xy[..., 0].max() + 10)
        ax3.set_ylim(xy[..., 1].min() - 10, xy[..., 1].max() + 10)
        ax3.set_aspect("equal"); ax3.set_axis_off()
        titulo.set_text(f"{tag}  ·  ciclo {int(cyc[k])}  ·  torsión total {total[k]:+.0f}°"
                      f"  ·  {saltos} cambios de quiralidad en la corrida")

        axp.clear()
        for kk in sel[max(0, j - 12):j]:
            axp.plot(perf[kk][0], perf[kk][1], color="#B8BEC8", lw=1)
        if len(zs):
            axp.plot(zs, acum, color="#C1443C", lw=2.5)
        axp.axhline(0, color="k", lw=0.8)
        axp.set_xlim(0, zmax); axp.set_ylim(-1.1 * vmax * 1.6, 1.1 * vmax * 1.6)
        axp.set_xlabel("altura a lo largo del filamento [nm]")
        axp.set_ylabel("ángulo acumulado [°]")
        axp.set_title("Perfil de torsión (una hélice sería una recta inclinada)", fontsize=12)
        axp.grid(alpha=0.3)

        axt.clear()
        axt.axhspan(0, 1e4, color="#C1443C", alpha=0.06)
        axt.axhspan(-1e4, 0, color="#3B6EA5", alpha=0.06)
        axt.plot(cyc / 1000, total, color="#9AA3AD", lw=1)
        axt.plot(cyc[:k + 1] / 1000, total[:k + 1], color="#14213D", lw=2)
        axt.plot(cyc[k] / 1000, total[k], "o", color="#C1443C", ms=9)
        axt.axhline(0, color="k", lw=0.8)
        lim = 1.1 * max(60.0, np.abs(total).max())
        axt.set_ylim(-lim, lim); axt.set_xlim(0, cyc[-1] / 1000)
        axt.set_xlabel("ciclos MC / 1000"); axt.set_ylabel("torsión total [°]")
        axt.set_title("Torsión total en el tiempo (cada cruce por cero = cambio de quiralidad)", fontsize=12)
        axt.grid(alpha=0.3)
        w.grab_frame()
        if j % 20 == 0:
            print(f"  cuadro {j + 1}/{len(sel)}", flush=True)
    w.finish()
    plt.close(fig)
    print(f"-> {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", help="una corrida; sin esto hace las cuatro de lambda = 1")
    ap.add_argument("--cada", type=int, default=2, help="usar 1 de cada N cuadros guardados")
    ap.add_argument("--dpi", type=int, default=80)
    ap.add_argument("--fps", type=int, default=8)
    a = ap.parse_args()
    for t in ([a.tag] if a.tag else CUATRO):
        gif(t, a.cada, a.dpi, a.fps)


if __name__ == "__main__":
    main()
