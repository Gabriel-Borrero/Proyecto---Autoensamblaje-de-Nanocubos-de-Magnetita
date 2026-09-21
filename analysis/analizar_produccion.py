"""
analizar_produccion.py  --  figuras y animacion de una corrida larga.

Uso:  python3 analysis/analizar_produccion.py --tag prod_w3_H417_v2 [--gif]

Produce, a partir de {tag}_energias.csv y {tag}_frames.npz:
  {tag}_fig_energias.png   las seis energias vs ciclos MC (formato Fig. S23)
  {tag}_fig_torsion.png    torsion, ancho y superenrollamiento vs ciclos
  {tag}_fig_snapshots.png  configuraciones inicial / intermedia / final
  {tag}_anim.gif           animacion con camara girando
y por pantalla las metricas de la helice final (paso, radio, quiralidad).
"""
import os, sys, csv, argparse, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis import filament_twist, cluster_stats, radial_distribution
from render import animate_one, snapshot_grid
from parameters import a

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def read_csv(tag):
    rows = list(csv.DictReader(open(os.path.join(RES, f"{tag}_energias.csv"))))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gif", action="store_true")
    a_ = ap.parse_args()
    tag = a_.tag
    d = read_csv(tag)
    c = d["cycle"] / 1000.0

    keys = [("Eall", r"$E_{all}$"), ("EvdW", r"$E_{vdW}$"), ("Ez", r"$E_z$"),
            ("Edd", r"$E_{dd}$"), ("Ea", r"$E_a$"), ("Emag", r"$E_{mag}$")]
    fig, axes = plt.subplots(len(keys), 1, figsize=(5.4, 9), sharex=True)
    for ax, (k, lab) in zip(axes, keys):
        ax.plot(c, d[k], lw=1.2, color="#8a6d1f")
        ax.set_ylabel(lab, fontsize=10); ax.grid(alpha=.3)
    axes[-1].set_xlabel("ciclos MC / 1000")
    axes[0].set_title(f"{tag}: energias por nanocubo [kcal/mol]", fontsize=10)
    plt.tight_layout(); plt.savefig(f"{RES}/{tag}_fig_energias.png", dpi=150); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4))
    ax[0].plot(c, d["twist_deg"], lw=1.4, color="#c1443c")
    ax[0].axhline(0, color="k", lw=.6)
    ax[0].set_xlabel("ciclos MC / 1000"); ax[0].set_ylabel("torsion acumulada [deg]")
    ax[0].set_title("Torsion (signo = quiralidad)"); ax[0].grid(alpha=.3)
    ax[1].plot(c, d["width_nm"], lw=1.4, color="#3b6ea5", label="ancho de la seccion")
    ax[1].plot(c, d["coil_nm"], lw=1.4, color="#48a072", label="superenrollamiento")
    ax[1].set_xlabel("ciclos MC / 1000"); ax[1].set_ylabel("nm")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3); ax[1].set_title("Geometria")
    ax[2].plot(c, d["acc_mech"], lw=1.4, label="mecanica")
    ax[2].plot(c, d["acc_mag"], lw=1.4, label="magnetica")
    ax[2].plot(c, d["Mz"], lw=1.4, label=r"$\langle M_z\rangle$")
    ax[2].set_xlabel("ciclos MC / 1000"); ax[2].set_ylim(0, 1.05)
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3); ax[2].set_title("Salud de la corrida")
    plt.tight_layout(); plt.savefig(f"{RES}/{tag}_fig_torsion.png", dpi=150); plt.close()

    npz = f"{RES}/{tag}_frames.npz"
    if os.path.exists(npz):
        f = np.load(npz)
        n = len(f["cycles"])
        snapshot_grid([npz] * 3, [tag] * 3, f"{RES}/{tag}_fig_snapshots.png",
                      indices=(0, n // 2, -1))
        tw = filament_twist(f["pos"][-1])
        cs = cluster_stats(f["pos"][-1])
        res = dict(tag=tag, torsion_total_deg=round(tw["total_twist"], 1),
                   torsion_deg_por_nm=round(tw["twist_rate"], 4),
                   paso_nm=(None if not np.isfinite(tw["pitch"]) else round(tw["pitch"], 1)),
                   quiralidad={1: "derecha", -1: "izquierda", 0: "sin resolver"}[tw["handedness"]],
                   ancho_nm=round(tw["width"], 1),
                   superenrollamiento_nm=round(tw["coil_radius"], 2),
                   cluster_max=int(cs["max_size"]),
                   vueltas=round(abs(tw["total_twist"]) / 360, 2))
        print(json.dumps(res, indent=1, ensure_ascii=False))
        json.dump(res, open(f"{RES}/{tag}_metricas.json", "w"), indent=1)
        if a_.gif:
            animate_one(npz, f"{RES}/{tag}_anim.gif",
                        title=tag, box=False, spin=1.2, fps=10)
            print("gif ->", f"{RES}/{tag}_anim.gif")


if __name__ == "__main__":
    main()
