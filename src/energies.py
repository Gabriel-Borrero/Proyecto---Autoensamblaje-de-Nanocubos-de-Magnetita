"""
energies.py  --  Paso 4 de la metodologia.

E_T = sum_i (E_i^Z + E_i^A) + sum_i sum_{j>i} (E_ij^dd + E_ij^vdW)      (ec. 5)

  Zeeman        E_i^Z   = -K_Z (H . M_i)                                 (ec. 1)
  Anisotropia   E_i^A   =  K_A1 [(M'x M'y)^2+(M'x M'z)^2+(M'y M'z)^2]    (ec. 2)
                con M' = R_i^T M_i  (componentes en el marco del cubo)
  Dipolo-dipolo E_ij^dd = (K_d/r^3)[M_i.M_j - 3(M_i.r)(M_j.r)]           (ec. 3)
                con r medido en unidades de a
  vdW + esterica E_ij^vdW = E^attr + E^rep                               (ec. 4)
                E^attr = -C_a sum_{k,l in 27x27} 1/r1^6
                E^rep  = +C_r sum_{m,n in NsxNs} w_m w_n / r2^8

Nota sobre C_a y C_r
--------------------
El suplemento da la FORMA de la ec. 4 (suma de Hamaker sobre 27 elementos de
volumen y suma sobre elementos de superficie) y sus dos anclajes cuantitativos
(Fig. S28E): el minimo del potencial efectivo esta en 2.99 nm de separacion
superficie-superficie y vale -2.33 kcal/mol por nanocubo.  Como los prefactores
originales dependen de la discretizacion exacta de 386 elementos, aqui se
RECALIBRAN C_a y C_r resolviendo  E(d0) = -2.33  y  E'(d0) = 0  para la
discretizacion que se este usando.  Asi el potencial es identico al de la
Fig. S28E aunque se use una malla mas barata.
"""
import numpy as np

from parameters import (K_Z, K_A1, K_d, H, a, D_MIN_FF, E_MIN_FF,
                        N_SURF, R_CUT_VDW)
from geometry import volume_elements, surface_elements

# ----------------------------------------------------------------------
# Mallas y calibracion (cacheadas por numero de elementos de superficie)
# ----------------------------------------------------------------------
VOL_PTS, VOL_W = volume_elements()
_CACHE = {}


def get_mesh(n_surf=N_SURF):
    if n_surf not in _CACHE:
        s_pts, s_area = surface_elements(n_surf)
        w = s_area / s_area.sum()                      # pesos normalizados
        C_a, C_r = _calibrate(VOL_PTS, s_pts, w)
        _CACHE[n_surf] = dict(vol=VOL_PTS, surf=s_pts, w=w,
                              W2=np.outer(w, w), C_a=C_a, C_r=C_r,
                              area=s_area.sum())
    return _CACHE[n_surf]


def _sums_face_to_face(vol, surf, W2, gap):
    """Sumas geometricas A(d) y B(d) para dos cubos alineados cara a cara."""
    shift = np.array([a + gap, 0.0, 0.0])
    d1 = vol[:, None, :] - (vol + shift)[None, :, :]
    A = np.sum(1.0 / np.einsum("ijk,ijk->ij", d1, d1) ** 3)
    d2 = surf[:, None, :] - (surf + shift)[None, :, :]
    B = np.sum(W2 / np.einsum("ijk,ijk->ij", d2, d2) ** 4)
    return A, B


def _calibrate(vol, surf, w, d0=D_MIN_FF, e0=E_MIN_FF, h=0.02):
    W2 = np.outer(w, w)
    Ap, Bp = _sums_face_to_face(vol, surf, W2, d0 + h)
    Am, Bm = _sums_face_to_face(vol, surf, W2, d0 - h)
    A0, B0 = _sums_face_to_face(vol, surf, W2, d0)
    dA, dB = (Ap - Am) / (2 * h), (Bp - Bm) / (2 * h)
    # E = -C_a A + C_r B ;  E'(d0)=0 -> C_r = C_a dA/dB ;  E(d0)=e0
    C_a = (-e0) / (A0 - dA * B0 / dB)
    C_r = C_a * dA / dB
    return C_a, C_r


# ----------------------------------------------------------------------
# API pedagogica (una o dos particulas, objetos Nanocube)
# ----------------------------------------------------------------------
def energy_zeeman(cube, field=H):
    return -K_Z * float(np.dot(field, cube.magnetic_moment))


def energy_anisotropy(cube):
    Mx, My, Mz = cube.orientation.T @ cube.magnetic_moment
    return K_A1 * ((Mx * My) ** 2 + (Mx * Mz) ** 2 + (My * Mz) ** 2)


def energy_dipole_dipole(cube_i, cube_j):
    r_vec = (cube_j.position - cube_i.position) / a      # en unidades de a
    r = np.linalg.norm(r_vec)
    r_hat = r_vec / r
    Mi, Mj = cube_i.magnetic_moment, cube_j.magnetic_moment
    return (K_d / r ** 3) * (np.dot(Mi, Mj)
                             - 3.0 * np.dot(Mi, r_hat) * np.dot(Mj, r_hat))


def energy_vdw(cube_i, cube_j, n_surf=N_SURF, scale=1.0):
    m = get_mesh(n_surf)
    Vi = cube_i.position + m["vol"] @ cube_i.orientation.T
    Vj = cube_j.position + m["vol"] @ cube_j.orientation.T
    Si = cube_i.position + m["surf"] @ cube_i.orientation.T
    Sj = cube_j.position + m["surf"] @ cube_j.orientation.T
    d1 = Vi[:, None, :] - Vj[None, :, :]
    attr = -m["C_a"] * np.sum(1.0 / np.einsum("ijk,ijk->ij", d1, d1) ** 3)
    d2 = Si[:, None, :] - Sj[None, :, :]
    rep = m["C_r"] * np.sum(m["W2"] / np.einsum("ijk,ijk->ij", d2, d2) ** 4)
    return scale * (attr + rep)


def total_energy(cubes, field=H, n_surf=N_SURF, vdw_scale=1.0):
    E = 0.0
    for c in cubes:
        E += energy_zeeman(c, field) + energy_anisotropy(c)
    for i in range(len(cubes)):
        for j in range(i + 1, len(cubes)):
            E += energy_dipole_dipole(cubes[i], cubes[j])
            E += energy_vdw(cubes[i], cubes[j], n_surf, vdw_scale)
    return E


# ----------------------------------------------------------------------
# API vectorizada (arreglos) -- la que usa el motor de Monte Carlo
# ----------------------------------------------------------------------
def zeeman_array(M, field=H):
    return -K_Z * (M @ field)


def anisotropy_array(M, R):
    """M' = R^T M para cada particula -> (N,3)."""
    Mp = np.einsum("nji,nj->ni", R, M)
    x, y, z = Mp[:, 0], Mp[:, 1], Mp[:, 2]
    return K_A1 * ((x * y) ** 2 + (x * z) ** 2 + (y * z) ** 2)


def dd_one_vs_many(pos_i, M_i, pos_j, M_j):
    """Energia dipolar de la particula i con un conjunto j (vectorizado)."""
    rv = (pos_j - pos_i) / a
    r2 = np.einsum("ij,ij->i", rv, rv)
    r = np.sqrt(r2)
    rhat = rv / r[:, None]
    return K_d / r ** 3 * (M_j @ M_i
                           - 3.0 * np.einsum("ij,ij->i", M_j, rhat) * (rhat @ M_i))


def vdw_one_vs_many(Vi, Si, Vj, Sj, mesh, scale=1.0):
    """
    Vi (27,3), Si (Ns,3) de la particula i;
    Vj (n,27,3), Sj (n,Ns,3) de sus vecinas -> vector (n,) de energias.
    """
    if len(Vj) == 0:
        return np.zeros(0)
    d1 = Vi[None, :, None, :] - Vj[:, None, :, :]
    r1sq = np.einsum("nijk,nijk->nij", d1, d1)
    attr = -mesh["C_a"] * np.sum(1.0 / r1sq ** 3, axis=(1, 2))
    d2 = Si[None, :, None, :] - Sj[:, None, :, :]
    r2sq = np.einsum("nijk,nijk->nij", d2, d2)
    rep = mesh["C_r"] * np.sum(mesh["W2"] / r2sq ** 4, axis=(1, 2))
    return scale * (attr + rep)


def vdw_sum(Vi, Si, Vj, Sj, mesh, scale=1.0):
    """
    Version escalar y rapida: atraccion sobre las vecinas Vj (corte 2.5a) y
    repulsion sobre las vecinas Sj (corte 1.7a).  Devuelve la energia total
    de la particula i con sus vecinas.
    """
    e = 0.0
    if len(Vj):
        d1 = Vi[None, :, None, :] - Vj[:, None, :, :]
        e -= mesh["C_a"] * np.sum(1.0 / np.einsum("nijk,nijk->nij", d1, d1) ** 3)
    if len(Sj):
        d2 = Si[None, :, None, :] - Sj[:, None, :, :]
        r2 = np.einsum("nijk,nijk->nij", d2, d2)
        e += mesh["C_r"] * np.sum(mesh["W2"][None] / r2 ** 4)
    return scale * e


def vdw_profile(gaps, n_surf=N_SURF, scale=1.0):
    """Perfil E_vdW(separacion superficie-superficie) cara a cara (Fig. S28E)."""
    m = get_mesh(n_surf)
    out = []
    for g in np.atleast_1d(gaps):
        A, B = _sums_face_to_face(m["vol"], m["surf"], m["W2"], g)
        out.append(scale * (-m["C_a"] * A + m["C_r"] * B))
    return np.array(out)
