"""
distribucion_torsion.py  --  ¿hay dos estados quirales o solo ruido?

La pregunta decisiva no es "cuanta torsion hay al final" sino como se reparte
la torsion a lo largo de toda la corrida:

  * P(omega) UNIMODAL centrada en 0  -> cinta plana con fluctuaciones termicas.
  * P(omega) BIMODAL en +-omega_0     -> ruptura espontanea de quiralidad: el
    filamento tiene dos estados torcidos (izquierdo y derecho) y salta entre
    ellos.  Es la version "filamento" de la quiralidad transitoria que el paper
    describe para un par de cubos (Fig. S21).

Se calcula tambien el perfil de energia libre  F(omega) = -kT ln P(omega):
un pozo unico en 0 o un doble pozo con barrera.

Uso:   python3 analysis/distribucion_torsion.py
Sale:  results/fig_distribucion_torsion.png  y una tabla por pantalla.
"""
import os, sys, glob, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from parameters import kBT

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def leer(f, descartar=0.1):
    r = list(csv.DictReader(open(f)))
    w = np.array([float(x["twist_rate"]) for x in r])
    return w[int(descartar * len(w)):]


def coef_bimodalidad(x):
    """
    Coeficiente de bimodalidad de Sarle: BC = (g^2 + 1) / (k + 3(n-1)^2/((n-2)(n-3)))
    con g la asimetria y k la curtosis en exceso.  Referencia: 0.555 para una
    distribucion uniforme; por encima sugiere bimodalidad.
    """
    n = len(x)
    if n < 10 or np.std(x) == 0:
        return np.nan
    z = (x - x.mean()) / x.std()
    g = np.mean(z ** 3)
    k = np.mean(z ** 4) - 3.0
    return float((g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def cambios_de_signo(x, ventana=20, umbral=0.02):
    """Saltos entre quiralidades en la serie suavizada (ignora el ruido cerca de 0)."""
    if len(x) < ventana:
        return 0
    s = np.convolve(x, np.ones(ventana) / ventana, mode="valid")
    estado = np.where(s > umbral, 1, np.where(s < -umbral, -1, 0))
    estado = estado[estado != 0]
    return int(np.sum(np.diff(estado) != 0)) if len(estado) > 1 else 0


def main():
    fs = sorted(glob.glob(os.path.join(RES, "prod_*_energias.csv")))
    if not fs:
        print("no hay archivos prod_*_energias.csv en results/")
        return
    datos = [(os.path.basename(f).replace("_energias.csv", ""), leer(f)) for f in fs]
    datos = [(t, w) for t, w in datos if len(w) > 20]

    print(f"{'corrida':36s} {'media':>7s} {'sigma':>6s} {'<|w|>':>6s} "
          f"{'BC':>5s} {'saltos':>6s}   lectura")
    filas = []
    for tag, w in datos:
        bc = coef_bimodalidad(w)
        sal = cambios_de_signo(w)
        if bc > 0.555 and sal >= 2:
            lec = "BIMODAL: dos quiralidades, salta entre ellas"
        elif abs(w.mean()) > 2 * w.std() / np.sqrt(max(len(w) / 40, 1)):
            lec = "sesgo hacia una quiralidad"
        else:
            lec = "unimodal: fluctuacion en torno a 0"
        filas.append((tag, w, bc, sal, lec))
        print(f"{tag:36s} {w.mean():+7.3f} {w.std():6.3f} {np.abs(w).mean():6.3f} "
              f"{bc:5.2f} {sal:6d}   {lec}")
    print("\nunidades de omega: deg/nm.  BC > 0.555 sugiere bimodalidad.")

    n = len(filas)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows * 2, cols, figsize=(4.6 * cols, 5.2 * rows),
                             gridspec_kw=dict(height_ratios=[1.3, 1] * rows))
    axes = np.atleast_2d(axes)
    lim = max(np.abs(np.concatenate([f[1] for f in filas])).max(), 0.05)
    bins = np.linspace(-lim, lim, 41)
    for k, (tag, w, bc, sal, lec) in enumerate(filas):
        r, c = divmod(k, cols)
        ah, af = axes[2 * r, c], axes[2 * r + 1, c]
        color = "#c1443c" if "BIMODAL" in lec else "#3b6ea5"
        ah.hist(w, bins=bins, color=color, alpha=0.85)
        ah.axvline(0, color="k", lw=0.8)
        ah.set_title(f"{tag.replace('prod_', '')}\nBC={bc:.2f}  saltos={sal}",
                     fontsize=8.5)
        ah.set_xlabel("omega [deg/nm]", fontsize=8)
        h, e = np.histogram(w, bins=bins)
        c_ = 0.5 * (e[1:] + e[:-1])
        ok = h > 0
        F = -kBT * np.log(h[ok] / h.max())
        af.plot(c_[ok], F, "o-", color=color, ms=3, lw=1.2)
        af.axvline(0, color="k", lw=0.8)
        af.set_ylabel("F [kcal/mol]", fontsize=8)
        af.set_xlabel("omega [deg/nm]", fontsize=8)
        af.grid(alpha=.3)
    for k in range(n, rows * cols):
        r, c = divmod(k, cols)
        axes[2 * r, c].axis("off"); axes[2 * r + 1, c].axis("off")
    fig.suptitle("Distribucion de la tasa de torsion (arriba) y energia libre "
                 "F = -kT ln P (abajo)\nrojo = bimodal (dos quiralidades) · "
                 "azul = unimodal", fontsize=11)
    fig.tight_layout()
    out = os.path.join(RES, "fig_distribucion_torsion.png")
    fig.savefig(out, dpi=150)
    print("->", out)


if __name__ == "__main__":
    main()
