"""
Validacion de los pasos 1-5 (seccion 7 de la metodologia).
Ejecutar:  python3 simulations/test_validation.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from parameters import (a, K_Z, K_A1, K_d, H0, D_MIN_FF, E_MIN_FF, kBT,
                        N_SURF, N_SURF_REF, D_LATTICE)
from geometry import Nanocube, surface_elements, R_IN, R_OUT
from rotations import rotation_matrix
from energies import (energy_zeeman, energy_anisotropy, energy_dipole_dipole,
                      energy_vdw, vdw_profile, get_mesh)
from gjk import gjk_overlap

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
ok = lambda c: "OK " if c else "FALLA"
print("=" * 70)

# --- (5) rotaciones ---------------------------------------------------
R = rotation_matrix([0, 0, 1], np.pi / 2)
v = R @ np.array([1.0, 0, 0])
print(f"[rotaciones]  Rz(90) x_hat = {v.round(6)}   {ok(np.allclose(v,[0,1,0]))}")
print(f"              ortogonal={ok(np.allclose(R@R.T, np.eye(3)))} "
      f"det={np.linalg.det(R):.6f}")

# --- (1) Zeeman -------------------------------------------------------
c = Nanocube([0, 0, 0], magnetic_moment=[0, 0, 1])
e_par = energy_zeeman(c)
c2 = Nanocube([0, 0, 0], magnetic_moment=[0, 0, -1])
e_anti = energy_zeeman(c2)
c3 = Nanocube([0, 0, 0], magnetic_moment=[1, 0, 0])
print(f"[Zeeman]      paralelo={e_par:.4f}  antiparalelo={e_anti:.4f} "
      f"perpendicular={energy_zeeman(c3):.4f} kcal/mol  "
      f"{ok(np.isclose(e_par, -K_Z*H0) and np.isclose(energy_zeeman(c3),0))}")
print(f"              |E_Z|/kT = {abs(e_par)/kBT:.2f}")

# --- (2) anisotropia --------------------------------------------------
eje = Nanocube([0, 0, 0], magnetic_moment=[0, 0, 1])
diag = Nanocube([0, 0, 0], magnetic_moment=np.ones(3) / np.sqrt(3))
cara = Nanocube([0, 0, 0], magnetic_moment=np.array([1, 1, 0]) / np.sqrt(2))
print(f"[anisotropia] eje<100>={energy_anisotropy(eje):.4f}  "
      f"diagonal<111>={energy_anisotropy(diag):.4f} (esperado {K_A1/3:.4f})  "
      f"<110>={energy_anisotropy(cara):.4f}  "
      f"{ok(np.isclose(energy_anisotropy(diag), K_A1/3))}")

# --- (3)(4) dipolo-dipolo --------------------------------------------
d = D_LATTICE
ht = (Nanocube([0, 0, 0], magnetic_moment=[0, 0, 1]),
      Nanocube([0, 0, d], magnetic_moment=[0, 0, 1]))
sbs = (Nanocube([0, 0, 0], magnetic_moment=[0, 0, 1]),
       Nanocube([d, 0, 0], magnetic_moment=[0, 0, 1]))
e_ht, e_sbs = energy_dipole_dipole(*ht), energy_dipole_dipole(*sbs)
print(f"[dipolar]     cabeza-cola={e_ht:.4f} (atractiva)  "
      f"lado-lado={e_sbs:.4f} (repulsiva)  {ok(e_ht<0<e_sbs)}")
rs = np.array([1.0, 2.0, 4.0]) * d
es = [energy_dipole_dipole(Nanocube([0, 0, 0], magnetic_moment=[0, 0, 1]),
                           Nanocube([0, 0, r], magnetic_moment=[0, 0, 1]))
      for r in rs]
ratio = np.array(es[:-1]) / np.array(es[1:])
print(f"              E(r)/E(2r) = {ratio.round(3)}  (esperado 8.0) {ok(np.allclose(ratio,8,rtol=1e-6))}")
theta_c = np.degrees(np.arccos(1 / np.sqrt(3)))
print(f"              angulo critico de cono = {theta_c:.1f} deg (paper: ~54 deg)")

# --- vdW: calibracion y comparacion de mallas ------------------------
gaps = np.linspace(1.2, 14.0, 160)
prof54 = vdw_profile(gaps, n_surf=54)
prof386 = vdw_profile(gaps, n_surf=N_SURF_REF)
i54, i386 = np.argmin(prof54), np.argmin(prof386)
print(f"[vdW]         malla  54: minimo en {gaps[i54]:.2f} nm, "
      f"{prof54[i54]:.3f} kcal/mol")
print(f"              malla 386: minimo en {gaps[i386]:.2f} nm, "
      f"{prof386[i386]:.3f} kcal/mol   (paper: {D_MIN_FF} nm, {E_MIN_FF})")
sel = gaps > 2.5     # region donde vive la estructura (minimo en 2.99 nm)
dev = np.max(np.abs(prof54[sel] - prof386[sel]))
print(f"              desviacion max entre mallas (region accesible) = {dev:.3f} kcal/mol "
      f"{ok(dev < 0.25)}")
wall = np.max(prof54[(gaps > 1.9) & (gaps < 2.2)] - prof386[(gaps > 1.9) & (gaps < 2.2)])
print(f"              (la pared repulsiva con 54 elementos es {wall:.1f} kcal/mol "
      f"mas dura a 2 nm, ~{wall/0.596:.0f} kT: irrelevante, ya esta prohibida)")
s, dS = surface_elements(386)
print(f"              area de la malla 386 = {dS.sum():.1f} nm^2 "
      f"(cubo ideal 6a^2 = {6*a**2:.1f})")

# anisotropia del potencial: cara-cara vs esquina-esquina
c1 = Nanocube([0, 0, 0])
cff = Nanocube([a + D_MIN_FF, 0, 0])
crot = Nanocube([a + D_MIN_FF, 0, 0],
                orientation=rotation_matrix([0, 0, 1], np.pi / 4))
print(f"              cara-cara={energy_vdw(c1,cff):.3f}  "
      f"girado 45deg={energy_vdw(c1,crot):.3f} kcal/mol "
      f"{ok(energy_vdw(c1,cff) < energy_vdw(c1,crot))}")

# --- (5) GJK ----------------------------------------------------------
I = np.eye(3)
tests = [
    ("separados 1.3a  ", [1.3*a,0,0], I, False),
    ("contacto 0.9a   ", [0.9*a,0,0], I, True),
    ("diagonal 1.1a   ", np.array([1,1,0])/np.sqrt(2)*1.1*a, I, True),
    ("girado 45 a 1.1a", [1.1*a,0,0], rotation_matrix([0,0,1],np.pi/4), True),
    ("girado 45 a 1.5a", [1.5*a,0,0], rotation_matrix([0,0,1],np.pi/4), False),
]
for name, dvec, Rj, expected in tests:
    got = gjk_overlap(np.zeros(3), I, np.array(dvec, float), Rj)
    print(f"[GJK]         {name} -> solapa={got}  {ok(got==expected)}")
print(f"              R_in={R_IN:.2f} nm  R_out={R_OUT:.2f} nm")

# --- figura de validacion --------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(gaps, prof386, lw=2, label="386 elementos (suplemento)")
ax[0].plot(gaps, prof54, "--", lw=2, label="54 elementos (produccion)")
ax[0].plot([D_MIN_FF], [E_MIN_FF], "o", ms=9, color="crimson",
           label="Fig. S28E (2.99 nm, -2.33)")
ax[0].axhline(0, color="k", lw=.6)
ax[0].set_xlabel("separacion superficie-superficie [nm]")
ax[0].set_ylabel(r"$E_{vdW}$ [kcal/mol]")
ax[0].set_title("Potencial efectivo vdW + esterico")
ax[0].set_ylim(-3, 4); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

rr = np.linspace(1.05, 4, 200) * a
par = [energy_dipole_dipole(Nanocube([0,0,0], magnetic_moment=[0,0,1]),
                            Nanocube([0,0,r], magnetic_moment=[0,0,1])) for r in rr]
sid = [energy_dipole_dipole(Nanocube([0,0,0], magnetic_moment=[0,0,1]),
                            Nanocube([r,0,0], magnetic_moment=[0,0,1])) for r in rr]
ax[1].plot(rr/a, par, lw=2, label="cabeza-cola (atraccion)")
ax[1].plot(rr/a, sid, lw=2, label="lado a lado (repulsion)")
ax[1].plot(rr/a, -2*K_d/(rr/a)**3, "k:", lw=1, label=r"$-2K_d/r^3$")
ax[1].axhline(0, color="k", lw=.6)
ax[1].axvline(D_LATTICE/a, color="gray", ls="--", lw=1, label="vecino de equilibrio")
ax[1].set_xlabel("r [unidades de a]"); ax[1].set_ylabel(r"$E_{dd}$ [kcal/mol]")
ax[1].set_title("Acoplamiento dipolo-dipolo"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig0_validacion.png"), dpi=150)
print("=" * 70)
print("figura guardada en results/fig0_validacion.png")
