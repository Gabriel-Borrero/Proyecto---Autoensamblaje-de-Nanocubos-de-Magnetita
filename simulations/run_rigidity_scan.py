"""
run_rigidity_scan.py  --  dónde se ablanda el modo de torsión.

Las corridas largas mostraron que la torsión fluctúa alrededor de cero sin
derivar, pero la AMPLITUD de esa fluctuación es en sí misma una medida útil:
por equipartición, para un modo armónico

        <delta_phi^2> = kT / k      ->      k = kT / <delta_phi^2>

con phi en radianes.  k es la rigidez torsional efectiva del filamento.  Si en
algún régimen k se desploma, ahí el filamento está al borde de torcerse y ahí
vale la pena gastar una corrida larga; donde k es grande, no.

Lo importante: k se mide con ~2500 ciclos, no con 23 500, porque el movimiento
colectivo de torsión muestrea esa coordenada directamente.  Un punto de la
rejilla cuesta minutos en vez de horas.

Uso:
    python3 simulations/run_rigidity_scan.py                 # rejilla por defecto
    python3 simulations/run_rigidity_scan.py --procs 4       # en paralelo
    python3 simulations/run_rigidity_scan.py --cycles 4000
"""
import sys, os, json, time, argparse, itertools
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from parameters import kBT
from builders import belt_110, belt
from mcmc import System
from analysis import filament_twist, cluster_stats

RES = os.path.join(os.path.dirname(__file__), "..", "results")

# (ancho, espesor, largo, arreglo, campo [G], vdw_scale, dd_scale)
GRID = [(w, t, 20, arr, H, vs, dd)
        for (w, t, arr) in [(3, 3, "110"), (2, 2, "110"), (4, 4, "110")]
        for H in (100.0, 167.0, 417.0)
        for (vs, dd) in [(2.0, 1.0), (1.0, 1.0)]]


def one(cfg, cycles=2500, equil=0.3, seed=31, collective_every=2):
    w, t, l, arr, H, vs, dd = cfg
    if arr == "110":
        pos, ors = belt_110(w, t, l)
    else:
        pos, ors = belt(w, t, l), None
    s = System(pos, orientations=ors, field=[0, 0, H], vdw_scale=vs,
               dd_scale=dd, seed=seed)
    s.collective_every = collective_every
    if s.overlap_report():
        return None
    t0 = time.time()
    serie = []
    for c in range(cycles):
        s.cycle(c)
        s.adapt(c)
        if c % 10 == 0:
            serie.append(filament_twist(s.pos)["total_twist"])
    serie = np.array(serie)
    tail = serie[int(equil * len(serie)):]
    mitad = len(tail) // 2
    sig = float(np.std(tail))
    sig1, sig2 = float(np.std(tail[:mitad])), float(np.std(tail[mitad:]))
    var_rad = np.radians(sig) ** 2
    k = float(kBT / var_rad) if var_rad > 0 else np.inf
    cs = cluster_stats(s.pos)
    am, _ = s.acceptance()
    return dict(width=w, thick=t, length=l, arrangement=arr, field=H,
                vdw_scale=vs, dd_scale=dd, N=int(s.N), cycles=cycles,
                sigma_deg=sig, sigma_half1=sig1, sigma_half2=sig2,
                k_torsion=k, drift=float(tail.mean() - serie[0]),
                max_abs=float(np.abs(tail).max()),
                acc=float(am), acc_tw=float(s.acc_tw / max(s.try_tw, 1)),
                intact=bool(int(cs["max_size"]) == s.N),
                minutes=(time.time() - t0) / 60)


def _worker(args):
    cfg, cycles = args
    try:
        r = one(cfg, cycles=cycles)
    except Exception as e:
        print("fallo", cfg, repr(e), flush=True)
        return None
    if r:
        print(f"{r['width']}x{r['thick']} {r['arrangement']}  H={r['field']:5.0f}  "
              f"vdw={r['vdw_scale']:4.1f} dd={r['dd_scale']:3.1f} | "
              f"sigma={r['sigma_deg']:6.1f} deg ({r['sigma_half1']:5.1f}/"
              f"{r['sigma_half2']:5.1f})  k={r['k_torsion']:7.2f}  "
              f"deriva={r['drift']:7.1f}  acc={r['acc']:.2f}  "
              f"intacta={r['intact']}  {r['minutes']:.1f} min", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--cycles", type=int, default=2500)
    a_ = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    print(f"{len(GRID)} puntos, {a_.cycles} ciclos cada uno\n"
          f"sigma = amplitud de la fluctuacion de torsion; los dos numeros entre "
          f"parentesis son las dos mitades (deben parecerse)\n"
          f"k = rigidez torsional [kcal/mol/rad^2]: CUANTO MAS PEQUENA, mejor "
          f"candidato para la corrida larga\n")
    args = [(cfg, a_.cycles) for cfg in GRID]
    if a_.procs > 1:
        with Pool(a_.procs) as p:
            rows = p.map(_worker, args)
    else:
        rows = [_worker(x) for x in args]
    rows = [r for r in rows if r]
    rows.sort(key=lambda r: r["k_torsion"])
    json.dump(rows, open(os.path.join(RES, "rigidity_scan.json"), "w"), indent=1)
    print("\n== ordenados de mas blando a mas rigido ==")
    for r in rows[:10]:
        print(f"  k={r['k_torsion']:7.2f}  {r['width']}x{r['thick']} "
              f"H={r['field']:5.0f} vdw={r['vdw_scale']:4.1f} "
              f"dd={r['dd_scale']:3.1f}  sigma={r['sigma_deg']:6.1f} deg")


if __name__ == "__main__":
    main()
