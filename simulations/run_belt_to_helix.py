"""
Transicion cinta -> helice en funcion del ANCHO (= densidad local de nanocubos
= campo efectivo B, cf. Figs. S17, S22 y S23 del suplemento).

Se parte de una cinta simple-cubica perfecta belt_100 con los dipolos
aleatorios y se deja evolucionar bajo el campo externo.  La helice NO se
impone: emerge de la competencia Zeeman / anisotropia / dipolar / vdW.

Uso:  python3 simulations/run_belt_to_helix.py  [ancho] [largo] [ciclos] [H] [vdw_scale]
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from parameters import N_SURF
from builders import belt
from mcmc import System
from analysis import filament_twist, cluster_stats

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)


def run(n_w=4, n_l=14, cycles=3000, field=668.0, vdw_scale=0.5, seed=7,
        tag=None, n_surf=N_SURF, record_every=25):
    pos = belt(n_w, 1, n_l)
    sys_ = System(pos, field=[0, 0, field], n_surf=n_surf,
                  vdw_scale=vdw_scale, seed=seed)
    tag = tag or f"w{n_w}_L{n_l}_H{int(field)}"
    print(f"[{tag}] N={sys_.N}  H={field} G  vdw_scale={vdw_scale}  "
          f"ciclos={cycles}", flush=True)
    t0 = time.time()
    hist, frames = sys_.run(cycles, record_every=record_every,
                            frame_every=max(1, cycles // 60), verbose=True)
    # torsion a lo largo de la trayectoria
    tw = []
    for f in frames:
        d = filament_twist(f["pos"])
        tw.append(dict(cycle=int(f["cycle"]), twist=d["total_twist"],
                       rate=d["twist_rate"], width=d["width"],
                       coil=d["coil_radius"], handedness=d["handedness"],
                       pitch=(d["pitch"] if np.isfinite(d["pitch"]) else None)))
    cs = cluster_stats(sys_.pos)
    res = dict(tag=tag, N=int(sys_.N), n_w=n_w, n_l=n_l, field=field,
               vdw_scale=vdw_scale, cycles=cycles,
               wall_time=time.time() - t0,
               history=[{k: float(v) for k, v in h.items()} for h in hist],
               twist=tw,
               final=dict(twist=tw[-1]["twist"], width=tw[-1]["width"],
                          coil=tw[-1]["coil"], rate=tw[-1]["rate"],
                          max_cluster=int(cs["max_size"]),
                          assembled=float(cs["assembled_fraction"])))
    with open(os.path.join(OUT, f"helix_{tag}.json"), "w") as fh:
        json.dump(res, fh)
    np.savez_compressed(os.path.join(OUT, f"helix_{tag}_frames.npz"),
                        cycles=np.array([f["cycle"] for f in frames]),
                        pos=np.array([f["pos"] for f in frames]),
                        R=np.array([f["R"] for f in frames]),
                        M=np.array([f["M"] for f in frames]))
    print(f"[{tag}] listo en {res['wall_time']:.0f} s  "
          f"torsion final = {tw[-1]['twist']:.1f} deg "
          f"({tw[-1]['rate']:.2f} deg/nm), ancho = {tw[-1]['width']:.1f} nm",
          flush=True)
    return res


if __name__ == "__main__":
    args = sys.argv[1:]
    run(n_w=int(args[0]) if len(args) > 0 else 4,
        n_l=int(args[1]) if len(args) > 1 else 14,
        cycles=int(args[2]) if len(args) > 2 else 3000,
        field=float(args[3]) if len(args) > 3 else 668.0,
        vdw_scale=float(args[4]) if len(args) > 4 else 0.5)
