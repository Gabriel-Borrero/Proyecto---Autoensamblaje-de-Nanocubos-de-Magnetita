"""
figuras.py  --  genera todas las figuras y animaciones.
Uso:  python3 analysis/figuras.py [figuras|gifs|todo]
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from parameters import a, D_LATTICE, kBT
from analysis import radial_distribution
from render import animate_multi, animate_one, snapshot_grid

RES = os.path.join(os.path.dirname(__file__), "..", "results")
PHIS = ["phi0p010", "phi0p050", "phi0p120", "phi0p250"]
LAB = [r"$\phi$ = 0.01", r"$\phi$ = 0.05", r"$\phi$ = 0.12", r"$\phi$ = 0.25"]
C = ["#3b6ea5", "#48a072", "#e0a72c", "#c1443c"]
load = lambda p: json.load(open(os.path.join(RES, p)))


def f1_validacion():
    rows = load("belt_energies.json")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, n_w in enumerate((3, 6, 9)):
        r = sorted([x for x in rows if x["n_w"] == n_w], key=lambda d: d["H"])
        Hs = [x["H"] for x in r]
        ax[0].plot(Hs, [x["Emag"] for x in r], "o-", color=C[k], lw=2,
                   label=f"simulacion, ancho {n_w}")
        ax[0].plot(Hs, [x["paper_Emag"] for x in r], "s--", color=C[k],
                   mfc="none", lw=1.2, label=f"Singh et al. S22, ancho {n_w}")
    ax[0].set_xlabel("campo externo H [G]")
    ax[0].set_ylabel(r"$E_{mag}$ [kcal/mol por nanocubo]")
    ax[0].set_title("Energia magnetica de belts$_{100}$")
    ax[0].legend(fontsize=7.5); ax[0].grid(alpha=.3)
    sim = np.array([x["Emag"] for x in rows]); pap = np.array([x["paper_Emag"] for x in rows])
    simv = np.array([x["EvdW"] for x in rows]); papv = np.array([x["paper_EvdW"] for x in rows])
    ax[1].plot(pap, sim, "o", ms=8, color="#c1443c", label=r"$E_{mag}$")
    ax[1].plot(papv, simv, "s", ms=8, color="#3b6ea5", label=r"$E_{vdW}$")
    lo, hi = min(pap.min(), papv.min()) - 2, max(pap.max(), papv.max()) + 2
    ax[1].plot([lo, hi], [lo, hi], "k--", lw=1)
    err = 100 * np.abs(sim - pap) / np.abs(pap)
    ax[1].set_xlabel("valor del paper [kcal/mol/NC]"); ax[1].set_ylabel("simulacion")
    ax[1].set_title(f"paridad: error medio en $E_{{mag}}$ = {err.mean():.1f} %")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{RES}/fig1_validacion_S22.png", dpi=150); plt.close()
    print("-> fig1", f"error medio {err.mean():.2f} %")


def f2_densidad():
    runs = [load(f"density_{t}.json") for t in PHIS]
    phi = [r["phi"] for r in runs]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    ax = axes[0, 0]
    ax.plot(phi, [r["final"]["assembled"] for r in runs], "o-", lw=2, color=C[3],
            label="fraccion ensamblada")
    ax.plot(phi, [abs(r["final"]["Mz"]) for r in runs], "^-", lw=2, color=C[1],
            label=r"$\langle M_z\rangle$")
    ax2 = ax.twinx()
    ax2.plot(phi, [r["final"]["mean_cluster"] for r in runs], "d--", lw=2, color="k",
             label="tamano medio de cluster")
    ax2.set_ylabel("nanocubos por cluster")
    ax.set_xscale("log"); ax.set_xlabel(r"fraccion de volumen $\phi$")
    ax.set_ylabel("parametro de orden"); ax.set_title("Estado final vs densidad")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center left"); ax.grid(alpha=.3)

    ax = axes[0, 1]
    for r, lab, c in zip(runs, LAB, C):
        s = r["structure"]
        ax.plot([x["cycle"] for x in s], [x["mean_size"] for x in s], lw=2, color=c, label=lab)
    ax.set_xlabel("ciclo MC"); ax.set_ylabel("tamano medio de cluster")
    ax.set_title("Cinetica del ensamblaje"); ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = axes[1, 0]
    for t, lab, c in zip(PHIS, LAB, C):
        d = np.load(f"{RES}/density_{t}_frames.npz")
        rr, g = radial_distribution(d["pos"][-1], L=float(d["L"]), rmax=4 * a, nbins=70)
        ax.plot(rr / a, g, lw=1.8, color=c, label=lab)
    ax.axvline(D_LATTICE / a, color="gray", ls="--", lw=1)
    ax.set_xlabel("r [unidades de a]"); ax.set_ylabel("g(r)")
    ax.set_title("Distribucion radial final (pico = contacto vdW)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = axes[1, 1]
    for (k, lab), c in zip([("Eall", r"$E_{all}$"), ("EvdW", r"$E_{vdW}$"),
                            ("Emag", r"$E_{mag}$"), ("Edd", r"$E_{dd}$")], C):
        ax.plot(phi, [r["history"][-1][k] for r in runs], "o-", lw=2, color=c, label=lab)
    ax.set_xscale("log"); ax.set_xlabel(r"$\phi$")
    ax.set_ylabel("energia [kcal/mol por nanocubo]")
    ax.set_title("Energias finales"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{RES}/fig2_densidad.png", dpi=150); plt.close()
    print("-> fig2")


def f3_campo_efectivo():
    d = load("effective_field.json")
    dens, belts = d["density"], d["belts"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    phi = [x["phi"] for x in dens]
    ax.flat[0].plot(phi, [x["H_eff_mag"] for x in dens], "o-", lw=2.4, color="#c1443c",
                    label=r"$\langle |B_{ef}| \rangle$ simulado")
    ax.flat[0].axhline(417, color="k", ls=":", lw=1.4, label="campo externo (417 G)")
    for y, txt, col in ((417, "belts (417 G)", "#48a072"), (668, "helices (668 G)", "#8a3ffc")):
        ax.flat[0].axhline(y, color=col, ls="--", lw=1)
        ax.flat[0].text(0.011, y + 12, txt, fontsize=7.5, color=col)
    ax2 = ax.flat[0].twinx()
    ax2.plot(phi, [x["mean_cluster"] for x in dens], "d--", color="gray", lw=1.6,
             label="tamano de cluster")
    ax2.set_ylabel("nanocubos por cluster")
    ax.flat[0].set_xscale("log"); ax.flat[0].set_xlabel(r"$\phi$")
    ax.flat[0].set_ylabel("campo efectivo [G]")
    ax.flat[0].set_title("Densidad -> campo efectivo")
    ax.flat[0].legend(fontsize=8, loc="lower right"); ax.flat[0].grid(alpha=.3)

    ax.flat[1].bar([f"{x['n_w']}" for x in belts], [x["H_eff_mag"] for x in belts],
                   color="#3b6ea5", alpha=.85)
    ax.flat[1].axhline(417, color="k", ls=":", lw=1.4)
    ax.flat[1].text(0.05, 460, "campo externo aplicado 417 G", fontsize=8)
    ax.flat[1].set_xlabel("ancho de la cinta (nanocubos)")
    ax.flat[1].set_ylabel("campo efectivo [G]")
    ax.flat[1].set_title("Campo dentro de un filamento ya ensamblado")
    ax.flat[1].grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig(f"{RES}/fig4_campo_efectivo.png", dpi=150); plt.close()
    print("-> fig4")


def f5_torsion():
    rows = load("twist_scan_H668.json")
    widths = sorted({r["n_w"] for r in rows})
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for n_w, c in zip(widths, C):
        r = sorted([x for x in rows if x["n_w"] == n_w], key=lambda d: d["omega"])
        om = [x["omega"] for x in r]
        ax[0].plot(om, [x["dE_kT"] for x in r], "o-", lw=2, color=c,
                   label=f"ancho {n_w} (N={r[0]['N']})")
        ax[1].plot(om, [x["Ea"] - r[0]["Ea"] for x in r], "o-", lw=2, color=c,
                   label=f"ancho {n_w}")
        ax[2].plot(om, [x["EvdW"] - r[0]["EvdW"] for x in r], "o-", lw=2, color=c)
    ax[0].set_xlabel("torsion impuesta $\\omega$ [deg/nm]")
    ax[0].set_ylabel(r"$\Delta E_{all}$ [kT por nanocubo]")
    ax[0].set_title("Coste total de torcer la cinta")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].set_xlabel("$\\omega$ [deg/nm]"); ax[1].set_ylabel(r"$\Delta E_a$ [kcal/mol/NC]")
    ax[1].set_title("Ganancia de anisotropia (motor de la helice)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[2].set_xlabel("$\\omega$ [deg/nm]"); ax[2].set_ylabel(r"$\Delta E_{vdW}$ [kcal/mol/NC]")
    ax[2].set_title("Coste de van der Waals (se opone)")
    ax[2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{RES}/fig5_torsion.png", dpi=150); plt.close()
    print("-> fig5")


def f6_helices():
    ws = [w for w in ("w2", "w4", "w6") if os.path.exists(f"{RES}/helix_{w}.json")]
    runs = {w: load(f"helix_{w}.json") for w in ws}
    keys = [("Eall", r"$E_{all}$"), ("EvdW", r"$E_{vdW}$"), ("Ez", r"$E_z$"),
            ("Edd", r"$E_{dd}$"), ("Ea", r"$E_a$"), ("Emag", r"$E_{mag}$")]
    fig, axes = plt.subplots(len(keys), 1, figsize=(5.2, 8.6), sharex=True)
    h = runs[ws[-1]]["history"]
    for ax, (k, lab) in zip(axes, keys):
        ax.plot([x["cycle"] / 1000 for x in h], [x[k] for x in h], lw=1.4, color="#8a6d1f")
        ax.set_ylabel(lab, fontsize=10); ax.grid(alpha=.3)
    axes[-1].set_xlabel("ciclos MC / 1000")
    axes[0].set_title(f"Relajacion de la cinta de ancho {runs[ws[-1]]['n_w']} "
                      f"(H = {runs[ws[-1]]['field']:.0f} G)\n"
                      "energias por nanocubo [kcal/mol]", fontsize=10)
    plt.tight_layout(); plt.savefig(f"{RES}/fig6_energias_dinamica.png", dpi=150); plt.close()
    snapshot_grid([f"{RES}/helix_{w}_frames.npz" for w in ws],
                  [f"ancho {runs[w]['n_w']}" for w in ws],
                  f"{RES}/fig7_snapshots_cintas.png", indices=(0, -1))
    print("-> fig6, fig7")


def f_snapshots_densidad():
    snapshot_grid([f"{RES}/density_{t}_frames.npz" for t in PHIS], LAB,
                  f"{RES}/fig3_snapshots_densidad.png", indices=(0, -1), arrows=False)
    print("-> fig3")


def gifs():
    animate_multi([f"{RES}/density_{t}_frames.npz" for t in PHIS], LAB,
                  f"{RES}/anim1_densidad.gif",
                  suptitle="Autoensamblaje vs densidad (H = 417 G)")
    print("-> anim1", os.path.getsize(f"{RES}/anim1_densidad.gif") // 1024, "kB")
    animate_one(f"{RES}/density_phi0p250_frames.npz", f"{RES}/anim2_phi025.gif",
                title=r"$\phi$ = 0.25: cadenas -> filamento", spin=2.0)
    print("-> anim2", os.path.getsize(f"{RES}/anim2_phi025.gif") // 1024, "kB")
    animate_one(f"{RES}/helix_w6_frames.npz", f"{RES}/anim3_cinta_w6.gif",
                title="Cinta de ancho 6 a H = 668 G", box=False, spin=2.5)
    print("-> anim3", os.path.getsize(f"{RES}/anim3_cinta_w6.gif") // 1024, "kB")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "todo"
    if what in ("figuras", "todo"):
        for f in (f1_validacion, f2_densidad, f3_campo_efectivo, f5_torsion,
                  f6_helices, f_snapshots_densidad):
            try:
                f()
            except Exception as e:
                print("fallo", f.__name__, repr(e))
    if what in ("gifs", "todo"):
        gifs()
