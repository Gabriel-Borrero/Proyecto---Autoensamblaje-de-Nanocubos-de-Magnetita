"""
run_batch.py  --  lanza varias corridas de produccion en paralelo, una por nucleo.

Cada corrida es independiente (semillas y parametros distintos), asi que escalan
perfectamente con el numero de nucleos.  Pensado para dejar la maquina trabajando
una noche y comparar regimenes al dia siguiente.

Uso:
    python3 simulations/run_batch.py              # rejilla recomendada
    python3 simulations/run_batch.py --dry        # solo muestra que lanzaria
    python3 simulations/run_batch.py --procs 4
"""
import os, sys, subprocess, argparse, itertools, time

HERE = os.path.dirname(os.path.abspath(__file__))
PROD = os.path.join(HERE, "run_production.py")

# (ancho, espesor, largo, arreglo, campo [G], vdW, dd_scale, doble, ciclos)
# La ruta del suplemento es belt_100 -> belt_110 multicapa -> side-stepping ->
# helice, asi que la rejilla prioriza belts_110 gruesos y parametros escalados.
# Los largos estan ajustados para que cada corrida quepa en 4-10 h por nucleo.
GRID = [
    (3, 3, 60, "110", 417.0, 2.5, 1.0, False, 23500),   # lambda deducida de S23
    (3, 3, 60, "110", 417.0, 3.0, 1.5, False, 23500),   # extremo alto + dd fuerte
    (3, 3, 60, "110", 417.0, 2.0, 1.0, False, 23500),   # extremo bajo
    (3, 3, 60, "110", 668.0, 2.5, 1.0, False, 23500),   # campo alto
    (3, 1, 100, "100", 417.0, 2.5, 1.0, False, 23500),  # control belt_100
    (2, 2, 80, "110", 417.0, 2.5, 1.5, True, 25000),    # doble hebra
]


def cmd_for(cfg, i):
    w, t, l, arr, H, vs, dd, dbl, cyc = cfg
    tag = f"prod_{arr}_w{w}x{t}{'d' if dbl else ''}_H{int(H)}_v{vs:g}_dd{dd:g}"
    c = [sys.executable, PROD, "--width", str(w), "--thickness", str(t),
         "--length", str(l), "--arrangement", arr, "--cycles", str(cyc),
         "--field", str(H), "--vdw-scale", str(vs), "--dd-scale", str(dd),
         "--tag", tag, "--seed", str(17 + i), "--resume"]
    if w * t * l * (2 if dbl else 1) > 400:
        c += ["--dd-cutoff", "8"]
    if dbl:
        c.append("--double")
    return tag, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=len(GRID))
    ap.add_argument("--dry", action="store_true")
    a_ = ap.parse_args()

    running, queue = [], [cmd_for(c, i) for i, c in enumerate(GRID)]
    logs = os.path.join(HERE, "..", "results")
    os.makedirs(logs, exist_ok=True)
    if a_.dry:
        for tag, c in queue:
            print(tag, " ".join(c))
        return
    while queue or running:
        while queue and len(running) < a_.procs:
            tag, c = queue.pop(0)
            fh = open(os.path.join(logs, f"{tag}.log"), "a")
            print("lanzando", tag, flush=True)
            running.append((tag, subprocess.Popen(c, stdout=fh, stderr=fh), fh))
        time.sleep(10)
        for item in running[:]:
            tag, p, fh = item
            if p.poll() is not None:
                print(f"terminado {tag} (codigo {p.returncode})", flush=True)
                fh.close()
                running.remove(item)


if __name__ == "__main__":
    main()
