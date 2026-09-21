"""
run_production.py  --  corrida larga reproduciendo el protocolo de la Fig. S23.

Pensado para dejarlo corriendo horas en tu maquina: guarda checkpoints, se puede
reanudar, escribe un CSV de energias y un npz de snapshots, e informa el ETA.

Ejemplos
--------
# replica del paper: belt_100 de 3 de ancho x 100 de largo = 300 nanocubos
python3 simulations/run_production.py --width 3 --length 100 --cycles 23500 \
        --field 417 --vdw-scale 2.0 --tag prod_w3

# doble hebra (dos cintas paralelas) -> doble helice
python3 simulations/run_production.py --width 2 --length 100 --double \
        --cycles 25000 --field 417 --vdw-scale 2.0 --tag prod_doble

# reanudar despues de cortar la corrida
python3 simulations/run_production.py --tag prod_w3 --resume

Notas
-----
* --vdw-scale es el factor sobre el potencial calibrado con la Fig. S28E.
  8.33 = escala "nominal" que reproduce E_vdW de la Fig. S22 (cintas rigidas);
  ~1-3 = regimen de "parametros escalados" que el paper usa para obtener helices
  (Fig. S21 reduce el acoplamiento vdW a la mitad; Fig. S23 dice explicitamente
  que usa parametros escalados).  Correr antes run_regime_scan.py.
* El coste por ciclo crece ~linealmente con N: ~0.37 s con 300 nanocubos,
  ~0.8 s con 600 y ~1.3 s con 900 en un solo nucleo.
"""
import sys, os, json, time, argparse, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from parameters import LAMBDA_S22, D_LATTICE, a as A_NM
from builders import belt, belt_110, double_belt
from mcmc import System
from analysis import filament_twist, cluster_stats

RES = os.path.join(os.path.dirname(__file__), "..", "results")


# ----------------------------------------------------------------------
def save_checkpoint(path, s, cycle, frames, t_elapsed):
    tmp = path + ".tmp"
    np.savez_compressed(
        tmp, pos=s.pos, R=s.R, M=s.M, cycle=cycle,
        delta=np.array([s.delta_s, s.delta_a, s.delta_m]),
        t_elapsed=t_elapsed,
        rng_state=json.dumps(s.rng.bit_generator.state),
        frame_cycles=np.array([f["cycle"] for f in frames], dtype=int),
        frame_pos=np.array([f["pos"] for f in frames]),
        frame_R=np.array([f["R"] for f in frames]),
        frame_M=np.array([f["M"] for f in frames]))
    os.replace(tmp + ".npz", path)


def load_checkpoint(path, s):
    d = np.load(path, allow_pickle=False)
    s.pos, s.R, s.M = d["pos"].copy(), d["R"].copy(), d["M"].copy()
    s.delta_s, s.delta_a, s.delta_m = [float(x) for x in d["delta"]]
    s.rng.bit_generator.state = json.loads(str(d["rng_state"]))
    s._build_clouds()
    s._T = None
    frames = [dict(cycle=int(c), pos=p, R=r, M=m) for c, p, r, m in
              zip(d["frame_cycles"], d["frame_pos"], d["frame_R"], d["frame_M"])]
    return int(d["cycle"]), frames, float(d["t_elapsed"])


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=3)
    ap.add_argument("--length", type=int, default=100)
    ap.add_argument("--thickness", type=int, default=1)
    ap.add_argument("--double", action="store_true",
                    help="dos cintas paralelas (semilla de doble helice)")
    ap.add_argument("--arrangement", choices=["100", "110"], default="100",
                    help="belt_100 (caras al frente) o belt_110 (cubos girados "
                         "45 grados: el precursor de la helice segun S3)")
    ap.add_argument("--dd-scale", type=float, default=1.0,
                    help="multiplica K_d; >1 imita el campo efectivo de un "
                         "filamento grueso ('parametros escalados' de la Fig. S23)")
    ap.add_argument("--dd-cutoff", type=float, default=0.0,
                    help="corte dipolar en unidades de a (0 = exacto, todos los "
                         "pares; 8 recomendado para N > 400: error 0.5 %%)")
    ap.add_argument("--mag-global-frac", type=float, default=0.1,
                    help="fraccion de pasos magneticos con reorientacion global")
    ap.add_argument("--gap", type=float, default=1.6,
                    help="separacion entre cintas, en unidades de red")
    ap.add_argument("--cycles", type=int, default=23500)
    ap.add_argument("--field", type=float, default=417.0)
    ap.add_argument("--vdw-scale", type=float, default=2.0)
    ap.add_argument("--twist-seed", type=float, default=0.0,
                    help="torsion inicial impuesta en deg/nm (0 = cinta plana)")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--tag", default="prod")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--frame-every", type=int, default=100)
    ap.add_argument("--checkpoint-every", type=int, default=250)
    ap.add_argument("--collective-every", type=int, default=5,
                    help="cada cuantos ciclos se propone una torsion colectiva "
                         "(0 = solo movimientos de una particula)")
    ap.add_argument("--resume", action="store_true")
    a_ = ap.parse_args()

    os.makedirs(RES, exist_ok=True)
    ck = os.path.join(RES, f"{a_.tag}_checkpoint.npz")
    csv_path = os.path.join(RES, f"{a_.tag}_energias.csv")

    ors = None
    if a_.arrangement == "110":
        pos, ors = belt_110(a_.width, a_.thickness, a_.length)
        if a_.double:
            p2, o2 = belt_110(a_.width, a_.thickness, a_.length)
            shift = np.array([a_.gap * D_LATTICE, 0.0, 0.0])
            pos = np.vstack([pos - shift / 2, p2 + shift / 2])
            ors = np.vstack([ors, o2])
    elif a_.double:
        pos = double_belt(a_.width, a_.thickness, a_.length,
                          gap=a_.gap * D_LATTICE)
    else:
        pos = belt(a_.width, a_.thickness, a_.length)
    if a_.twist_seed:
        from run_twist_scan import twisted_belt
        pos, ors = twisted_belt(a_.width, a_.length, a_.twist_seed)

    s = System(pos, orientations=ors, field=[0, 0, a_.field],
               vdw_scale=a_.vdw_scale, seed=a_.seed,
               dd_scale=a_.dd_scale, mag_global_frac=a_.mag_global_frac,
               dd_cutoff=(a_.dd_cutoff * A_NM if a_.dd_cutoff else None))
    s.collective_every = a_.collective_every
    ov = s.overlap_report()
    if ov:
        raise SystemExit(f"configuracion inicial con {len(ov)} solapamientos")

    start, frames, t_prev = 0, [], 0.0
    if a_.resume and os.path.exists(ck):
        start, frames, t_prev = load_checkpoint(ck, s)
        print(f"reanudando en el ciclo {start}", flush=True)
    else:
        with open(csv_path, "w", newline="") as fh:
            csv.writer(fh).writerow(
                ["cycle", "Eall", "EvdW", "Emag", "Ez", "Edd", "Ea", "Mz",
                 "twist_deg", "twist_rate", "width_nm", "coil_nm",
                 "acc_mech", "acc_mag", "delta_s", "delta_a", "delta_tw"])

    print(f"[{a_.tag}] N={s.N}  belt_{a_.arrangement}  H={a_.field:.0f} G  "
          f"vdw_scale={a_.vdw_scale}  dd_scale={a_.dd_scale}  "
          f"ciclos={a_.cycles}  ({'doble hebra' if a_.double else 'una hebra'})",
          flush=True)
    t0 = time.time()
    for c in range(start, a_.cycles):
        s.cycle(c)
        s.adapt(c)
        if c % a_.log_every == 0 or c == a_.cycles - 1:
            comp = s.components()
            tw = filament_twist(s.pos)
            am, ag = s.acceptance()
            with open(csv_path, "a", newline="") as fh:
                csv.writer(fh).writerow(
                    [c] + [f"{comp[k]:.5f}" for k in
                           ("Eall", "EvdW", "Emag", "Ez", "Edd", "Ea")]
                    + [f"{s.M[:, 2].mean():.4f}", f"{tw['total_twist']:.2f}",
                       f"{tw['twist_rate']:.4f}", f"{tw['width']:.2f}",
                       f"{tw['coil_radius']:.2f}", f"{am:.3f}", f"{ag:.3f}",
                       f"{s.delta_s:.3f}", f"{s.delta_a:.4f}",
                       f"{s.delta_tw:.4f}"])
            if c % (a_.log_every * 20) == 0:
                el = time.time() - t0 + t_prev
                done = c - start + 1
                eta = (time.time() - t0) / max(done, 1) * (a_.cycles - c) / 3600
                print(f"  ciclo {c:6d}/{a_.cycles}  Eall={comp['Eall']:8.2f}  "
                      f"<Mz>={s.M[:, 2].mean():.3f}  torsion={tw['total_twist']:8.1f} deg "
                      f"({tw['twist_rate']:+.3f} deg/nm)  acc={am:.2f}  "
                      f"t={el/3600:.2f} h  ETA={eta:.2f} h", flush=True)
        if c % a_.frame_every == 0 or c == a_.cycles - 1:
            frames.append(dict(cycle=c, pos=s.pos.copy(), R=s.R.copy(),
                               M=s.M.copy()))
        if c % a_.checkpoint_every == 0 or c == a_.cycles - 1:
            save_checkpoint(ck, s, c + 1, frames, time.time() - t0 + t_prev)

    np.savez_compressed(os.path.join(RES, f"{a_.tag}_frames.npz"),
                        cycles=np.array([f["cycle"] for f in frames]),
                        pos=np.array([f["pos"] for f in frames]),
                        R=np.array([f["R"] for f in frames]),
                        M=np.array([f["M"] for f in frames]))
    tw = filament_twist(s.pos)
    cs = cluster_stats(s.pos)
    out = dict(tag=a_.tag, N=int(s.N), width=a_.width, length=a_.length,
               double=bool(a_.double), arrangement=a_.arrangement,
               field=a_.field, vdw_scale=a_.vdw_scale, dd_scale=a_.dd_scale,
               cycles=a_.cycles, hours=(time.time() - t0 + t_prev) / 3600,
               twist_deg=tw["total_twist"], twist_rate=tw["twist_rate"],
               pitch_nm=(None if not np.isfinite(tw["pitch"]) else tw["pitch"]),
               handedness=tw["handedness"], width_nm=tw["width"],
               coil_nm=tw["coil_radius"], max_cluster=int(cs["max_size"]),
               assembled=float(cs["assembled_fraction"]))
    json.dump(out, open(os.path.join(RES, f"{a_.tag}_resumen.json"), "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
