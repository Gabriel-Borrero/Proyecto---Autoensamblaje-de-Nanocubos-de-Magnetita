import sys, os, json
sys.path.insert(0,"src")
import numpy as np
from mcmc import System
from builders import belt
from analysis import effective_field, cluster_stats
from parameters import LAMBDA_S22

out = {"density": [], "belts": []}
for tag in ("phi0p010","phi0p050","phi0p120","phi0p250"):
    d = np.load(f"results/density_{tag}_frames.npz")
    s = System(d["pos"][-1], orientations=d["R"][-1], moments=d["M"][-1],
               field=[0,0,417.0], vdw_scale=LAMBDA_S22, seed=0)
    cs = cluster_stats(s.pos)
    e = effective_field(s)
    e.update(phi=float(d["phi"]), tag=tag, max_cluster=int(cs["max_size"]),
             mean_cluster=float(cs["weighted_mean_size"]))
    out["density"].append(e); print(tag, {k: round(v,2) for k,v in e.items() if isinstance(v,float)})
for n_w in (1,2,3,4,6,9):
    pos = belt(n_w,1,20)
    s = System(pos, field=[0,0,417.0], vdw_scale=LAMBDA_S22, seed=0,
               moments=np.tile([0,0,1.],(len(pos),1)), freeze_positions=True)
    s.run(120, record_every=60, frame_every=120, include_vdw=False)
    e = effective_field(s); e.update(n_w=n_w, N=int(s.N))
    out["belts"].append(e); print("belt", n_w, {k: round(v,2) for k,v in e.items() if isinstance(v,float)})
json.dump(out, open("results/effective_field.json","w"))
