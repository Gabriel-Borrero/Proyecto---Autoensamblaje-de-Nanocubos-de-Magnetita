"""
Barrido energetico cinta -> helice (posiciones congeladas).

La formacion espontanea de la helice en la dinamica MC requiere ~2x10^4 ciclos
(Fig. S23 del suplemento).  Para cuantificarla con coste razonable se usa el
mismo recurso que el paper en las Figs. S22, S24-S27: se CONSTRUYEN estructuras
con una torsion impuesta omega (deg/nm), se congelan las posiciones y se
promedian los grados de libertad magneticos por Monte Carlo.

Para cada ancho de cinta (= densidad local de nanocubos = campo efectivo) se
obtiene E_all(omega).  El minimo de esa curva dice si la estructura estable es
la cinta plana (omega = 0) o una helice (omega > 0), y  Delta E / kT  da la
probabilidad relativa  P(helice)/P(cinta) = exp(-N Delta E / kT).
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from parameters import LAMBDA_S22, kBT, D_LATTICE
from builders import belt
from mcmc import System
from rotations import rotation_matrix

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)


def twisted_belt(n_w, n_l, omega_deg_nm):
    """Cinta con una torsion rigida de omega grados por nm alrededor del eje z."""
    pos = belt(n_w, 1, n_l)
    z0 = pos[:, 2].mean()
    new_pos, ors = [], []
    for p in pos:
        ang = np.radians(omega_deg_nm) * (p[2] - z0)
        Rz = rotation_matrix([0, 0, 1], ang)
        q = Rz @ np.array([p[0], p[1], 0.0])
        new_pos.append([q[0], q[1], p[2]])
        ors.append(Rz)                      # el cubo gira con su capa
    return np.array(new_pos), np.array(ors)


def scan(field=668.0, widths=(2, 3, 4, 6), omegas=(0, .25, .5, .75, 1., 1.5, 2.),
         n_l=14, cycles=150, vdw_scale=0.5 * LAMBDA_S22, seed=5):
    rows = []
    print(f"H = {field:.0f} G   vdw_scale = {vdw_scale:.2f}")
    print(f"{'ancho':>5} {'omega':>6} {'N':>4} | {'Eall':>8} {'EvdW':>8} "
          f"{'Emag':>8} {'Ez':>7} {'Edd':>7} {'Ea':>7} | {'dEall/kT':>9}")
    for n_w in widths:
        base = None
        for om in omegas:
            pos, ors = twisted_belt(n_w, n_l, om)
            s = System(pos, orientations=ors, field=[0, 0, field], seed=seed,
                       vdw_scale=vdw_scale, freeze_positions=True)
            hist, _ = s.run(cycles, record_every=10, frame_every=cycles,
                            include_vdw=False)
            half = [h for h in hist if h["cycle"] >= cycles // 2]
            m = {k: float(np.mean([h[k] for h in half]))
                 for k in ("Ez", "Ea", "Edd", "Emag")}
            evdw = sum(0.5 * s.vdw_of(i, s.pos[i], s.R[i])
                       for i in range(s.N)) / s.N
            eall = m["Emag"] + evdw
            base = eall if om == 0 else base
            d_kT = (eall - base) / kBT
            rows.append(dict(field=field, n_w=n_w, omega=om, N=int(s.N),
                             Eall=eall, EvdW=evdw, **m, dE_kT=d_kT))
            print(f"{n_w:5d} {om:6.2f} {s.N:4d} | {eall:8.2f} {evdw:8.2f} "
                  f"{m['Emag']:8.2f} {m['Ez']:7.2f} {m['Edd']:7.2f} "
                  f"{m['Ea']:7.3f} | {d_kT:9.2f}", flush=True)
    f = os.path.join(OUT, f"twist_scan_H{int(field)}.json")
    json.dump(rows, open(f, "w"))
    return rows


if __name__ == "__main__":
    scan(field=float(sys.argv[1]) if len(sys.argv) > 1 else 668.0)
