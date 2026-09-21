"""Ejecuta una etapa de produccion: s22 | dens <phi> | helix <ancho>."""
import sys, os, json
sys.path.insert(0, "src"); sys.path.insert(0, "simulations")
import numpy as np
from parameters import LAMBDA_S22

stage = sys.argv[1]

if stage == "s22":
    import run_belt_energies as rbe
    from builders import belt
    from mcmc import System
    f = "results/belt_energies.json"
    rows = json.load(open(f)) if os.path.exists(f) else []
    have = {(r["n_w"], int(r["H"])) for r in rows}
    n_w, CY = 9, 200
    pos = belt(n_w, 1, rbe.LEN)
    for Hval in (167.0, 417.0, 668.0):
        if (n_w, int(Hval)) in have: continue
        s = System(pos, field=[0,0,Hval], seed=3, vdw_scale=LAMBDA_S22,
                   freeze_positions=True)
        hist,_ = s.run(CY, record_every=10, frame_every=CY, include_vdw=False)
        half=[h for h in hist if h["cycle"]>=CY//2]
        m={k: float(np.mean([h[k] for h in half])) for k in ("Ez","Ea","Edd","Emag")}
        ev=sum(0.5*s.vdw_of(i,s.pos[i],s.R[i]) for i in range(s.N))/s.N
        pa,pv,pm = rbe.PAPER[(n_w,int(Hval))]
        rows.append(dict(n_w=n_w,H=Hval,**m,EvdW=ev,paper_Emag=pm,paper_EvdW=pv,t=0.0))
        print(f"  {n_w} {Hval:6.0f} | Emag={m['Emag']:8.2f} paper={pm:8.2f} "
              f"dif={100*(m['Emag']-pm)/abs(pm):5.1f}%  EvdW={ev:8.2f} paper={pv:8.2f}",
              flush=True)
        json.dump(rows, open(f,"w"))

elif stage == "dens":
    import run_density as rd
    rd.run_one(float(sys.argv[2]), N=56, cycles=int(sys.argv[3]) if len(sys.argv)>3 else 1200)

elif stage == "helix":
    import run_belt_to_helix as rbh
    w = int(sys.argv[2]); cy = int(sys.argv[3]) if len(sys.argv)>3 else 1500
    rbh.run(n_w=w, n_l=14, cycles=cy, field=668.0,
            vdw_scale=0.5*LAMBDA_S22, tag=f"w{w}")
