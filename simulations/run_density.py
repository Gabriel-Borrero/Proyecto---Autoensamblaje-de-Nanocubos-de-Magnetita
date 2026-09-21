"""
EXPERIMENTO PRINCIPAL: autoensamblaje en funcion de la DENSIDAD.

N nanocubos con posiciones, orientaciones y dipolos aleatorios dentro de una
caja cubica de lado L, a fraccion de volumen  phi = N V_NC / L^3, bajo un campo
externo fijo a lo largo de z.  Lo unico que cambia entre corridas es L.

Mecanismo que se pone a prueba (S3 y Fig. S17 del suplemento): al subir la
densidad crece el numero de vecinos, y con el el acoplamiento dipolar, es decir
el CAMPO EFECTIVO B = H_externo + H_dipolar.  La secuencia esperada es
    disperso -> cadenas -> cintas (belts) -> filamentos gruesos/helicoidales.

Uso:  python3 simulations/run_density.py [phi1 phi2 ...]
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from parameters import LAMBDA_S22, a
from builders import random_gas, box_for_density
from mcmc import System
from analysis import cluster_stats, nematic_order, filament_twist

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)


def run_one(phi, N=56, cycles=1500, field=417.0, seed=11,
            vdw_scale=LAMBDA_S22, record_every=50):
    rng = np.random.default_rng(seed)
    L = box_for_density(N, phi)
    pos = random_gas(N, L, rng)
    box = np.array([[0.75 * a] * 3, [L - 0.75 * a] * 3])
    s = System(pos, field=[0, 0, field], box=box, vdw_scale=vdw_scale, seed=seed)
    tag = f"phi{phi:.3f}".replace(".", "p")
    print(f"[{tag}] N={N}  L={L:.1f} nm  phi={phi:.3f}  ciclos={cycles}", flush=True)
    t0 = time.time()
    hist, frames = s.run(cycles, record_every=record_every,
                         frame_every=max(1, cycles // 50), verbose=True)
    struct = []
    for f in frames:
        cs = cluster_stats(f["pos"])
        struct.append(dict(cycle=int(f["cycle"]),
                           mean_size=float(cs["weighted_mean_size"]),
                           max_size=int(cs["max_size"]),
                           assembled=float(cs["assembled_fraction"]),
                           S=float(nematic_order(f["pos"])),
                           Mz=float(f["M"][:, 2].mean())))
    cs = cluster_stats(s.pos)
    shapes = sorted(cs["shapes"], key=lambda d: -d["Rg"])
    tw = filament_twist(s.pos)
    res = dict(tag=tag, phi=phi, N=N, L=L, field=field, cycles=cycles,
               vdw_scale=vdw_scale, wall_time=time.time() - t0,
               history=[{k: float(v) for k, v in h.items()} for h in hist],
               structure=struct,
               final=dict(max_cluster=int(cs["max_size"]),
                          mean_cluster=float(cs["weighted_mean_size"]),
                          assembled=float(cs["assembled_fraction"]),
                          n_clusters=int(cs["n_clusters"]),
                          S=float(nematic_order(s.pos)),
                          asphericity=float(shapes[0]["asphericity"]) if shapes else 0.0,
                          acylindricity=float(shapes[0]["acylindricity"]) if shapes else 0.0,
                          twist=float(tw["total_twist"]),
                          Mz=float(s.M[:, 2].mean())))
    with open(os.path.join(OUT, f"density_{tag}.json"), "w") as fh:
        json.dump(res, fh)
    np.savez_compressed(os.path.join(OUT, f"density_{tag}_frames.npz"),
                        cycles=np.array([f["cycle"] for f in frames]),
                        pos=np.array([f["pos"] for f in frames]),
                        R=np.array([f["R"] for f in frames]),
                        M=np.array([f["M"] for f in frames]), L=L, phi=phi)
    print(f"[{tag}] {res['wall_time']:.0f} s | cluster max={cs['max_size']} "
          f"medio={cs['weighted_mean_size']:.1f} "
          f"ensamblado={cs['assembled_fraction']:.2f} "
          f"S={res['final']['S']:.2f}", flush=True)
    return res


if __name__ == "__main__":
    phis = [float(x) for x in sys.argv[1:]] or [0.01, 0.05, 0.12, 0.25]
    out = [run_one(p) for p in phis]
    with open(os.path.join(OUT, "density_summary.json"), "w") as fh:
        json.dump([{k: r[k] for k in ("tag", "phi", "N", "L", "final")}
                   for r in out], fh)
