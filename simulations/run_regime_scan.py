"""
run_regime_scan.py  --  en que regimen de parametros existe la helice.

Antes de quemar 10 h en una corrida larga conviene saber en que esquina del
espacio (escala vdW, campo) el modelo permite la torsion.  Dos pruebas cortas:

  modo "semilla" (rapido y decisivo): se parte de una cinta YA torcida
      (omega = 1 deg/nm) y se deja evolucionar libremente.  Si la torsion
      sobrevive o crece, la helice es estable en ese regimen; si se desenrolla,
      no lo es y la corrida larga seria inutil.

  modo "libre": se parte de la cinta plana y se mide cuanta torsion aparece.

Uso:  python3 simulations/run_regime_scan.py [semilla|libre] [ciclos]
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from parameters import LAMBDA_S22, kBT
from builders import belt
from mcmc import System
from analysis import filament_twist, cluster_stats
from run_twist_scan import twisted_belt

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def one(vdw_scale, field, mode="semilla", n_w=3, n_l=24, cycles=400,
        omega=1.0, seed=23):
    if mode == "semilla":
        pos, ors = twisted_belt(n_w, n_l, omega)
    else:
        pos, ors = belt(n_w, 1, n_l), None
    s = System(pos, orientations=ors, field=[0, 0, field],
               vdw_scale=vdw_scale, seed=seed)
    tw0 = filament_twist(s.pos)
    t0 = time.time()
    traj = []
    for c in range(cycles):
        s.cycle()
        s.adapt(c)
        if c % 25 == 0 or c == cycles - 1:
            tw = filament_twist(s.pos)
            traj.append((c, tw["twist_rate"], tw["coil_radius"]))
    tw = filament_twist(s.pos)
    cs = cluster_stats(s.pos)
    comp = s.components()
    keep = (tw["twist_rate"] / tw0["twist_rate"]) if abs(tw0["twist_rate"]) > 1e-6 else np.nan
    return dict(vdw_scale=vdw_scale, field=field, mode=mode, N=int(s.N),
                rate0=tw0["twist_rate"], rate=tw["twist_rate"], keep=keep,
                coil=tw["coil_radius"], max_cluster=int(cs["max_size"]),
                intact=int(cs["max_size"]) == s.N,
                acc=s.acceptance()[0],
                well_kT=2.33 * vdw_scale / kBT,
                Eall=comp["Eall"], t=time.time() - t0, traj=traj)


def main(mode="semilla", cycles=400):
    rows = []
    print(f"modo = {mode}   ciclos = {cycles}")
    print(f"{'vdW':>6} {'pozo/kT':>8} {'H[G]':>6} | {'w0':>7} {'w_fin':>7} "
          f"{'conserva':>8} {'intacta':>8} {'acc':>5} {'Eall':>8}")
    for vs in (1.0, 2.0, 0.5 * LAMBDA_S22):
        for H in (417.0, 668.0):
            r = one(vs, H, mode=mode, cycles=cycles)
            rows.append(r)
            print(f"{vs:6.2f} {r['well_kT']:8.1f} {H:6.0f} | {r['rate0']:7.3f} "
                  f"{r['rate']:7.3f} {r['keep']:8.2f} {str(r['intact']):>8} "
                  f"{r['acc']:5.2f} {r['Eall']:8.2f}", flush=True)
    json.dump(rows, open(os.path.join(RES, f"regime_{mode}.json"), "w"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "semilla",
         int(sys.argv[2]) if len(sys.argv) > 2 else 400)
