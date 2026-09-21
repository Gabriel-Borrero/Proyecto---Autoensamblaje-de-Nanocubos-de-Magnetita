"""
Validacion cuantitativa contra la Fig. S22 del suplemento.

Se congelan las posiciones en cintas perfectas belt_100 de ancho n (n = 3, 6, 9)
y se muestrean SOLO los grados de libertad magneticos a H = 167, 417 y 668 G.
Se comparan E_z, E_dd, E_a y E_mag = E_z + E_dd + E_a por nanocubo con la tabla
del paper.  E_vdW es constante (posiciones fijas) y se calcula aparte.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from parameters import FIELDS_PAPER, LAMBDA_S22
from builders import belt
from mcmc import System

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

PAPER = {  # (ancho, H): (E_all, E_vdW, E_mag) de la Fig. S22, belts_100
    (3, 167): (-46.66, -34.70, -11.96), (3, 417): (-50.68, -34.70, -15.98),
    (3, 668): (-54.78, -34.70, -20.08),
    (6, 167): (-51.08, -38.96, -12.12), (6, 417): (-54.83, -38.96, -15.87),
    (6, 668): (-58.90, -38.96, -19.94),
    (9, 167): (-52.57, -40.38, -12.19), (9, 417): (-56.20, -40.38, -15.82),
    (9, 668): (-60.26, -40.38, -19.88),
}

LEN = 40          # nanocubos de largo (el paper usa 100)
CYCLES = 320      # ciclos magneticos; se promedia la segunda mitad


def main():
    rows = []
    print(f"{'ancho':>5} {'H[G]':>6} | {'Ez':>8} {'Edd':>8} {'Ea':>8} "
          f"{'Emag':>8} | {'Emag paper':>10} {'dif %':>7} | {'EvdW':>8} {'paper':>8}")
    for n_w in (3, 6, 9):
        pos = belt(n_w, 1, LEN)
        for Hval in FIELDS_PAPER:
            t0 = time.time()
            s = System(pos, field=[0, 0, Hval], seed=3,
                       vdw_scale=LAMBDA_S22, freeze_positions=True)
            hist, _ = s.run(CYCLES, record_every=10, frame_every=CYCLES,
                            include_vdw=False)
            half = [h for h in hist if h["cycle"] >= CYCLES // 2]
            m = {k: float(np.mean([h[k] for h in half]))
                 for k in ("Ez", "Ea", "Edd", "Emag")}
            evdw = sum(0.5 * s.vdw_of(i, s.pos[i], s.R[i])
                       for i in range(s.N)) / s.N
            p_all, p_vdw, p_mag = PAPER[(n_w, int(Hval))]
            rows.append(dict(n_w=n_w, H=Hval, **m, EvdW=evdw,
                             paper_Emag=p_mag, paper_EvdW=p_vdw,
                             t=time.time() - t0))
            print(f"{n_w:5d} {Hval:6.0f} | {m['Ez']:8.2f} {m['Edd']:8.2f} "
                  f"{m['Ea']:8.2f} {m['Emag']:8.2f} | {p_mag:10.2f} "
                  f"{100*(m['Emag']-p_mag)/abs(p_mag):7.1f} | {evdw:8.2f} "
                  f"{p_vdw:8.2f}", flush=True)
    with open(os.path.join(OUT, "belt_energies.json"), "w") as fh:
        json.dump(rows, fh)


if __name__ == "__main__":
    main()
