"""
builders.py  --  configuraciones iniciales.

* random_gas    : N nanocubos dispersos en una caja cubica a fraccion de
                  volumen phi = N * V_NC / L^3  (control de la DENSIDAD)
* belt          : cinta simple-cubica (belt_100) de n_w x n_t x n_l nanocubos
                  con el espaciamiento de equilibrio a + 2.99 nm
* double_belt   : dos cintas paralelas separadas un gap (semilla de doble helice)
"""
import numpy as np

from parameters import a, V_NC, D_LATTICE
from geometry import R_OUT
from rotations import random_unit_vector


def box_for_density(N, phi):
    """Lado de la caja cubica que da la fraccion de volumen phi."""
    return (N * V_NC / phi) ** (1.0 / 3.0)


def density_from_box(N, L):
    return N * V_NC / L ** 3


def random_gas(N, L, rng, min_center_dist=None, max_tries=60000):
    """
    Posiciones aleatorias sin solapamiento dentro de [0,L]^3.
    Si la densidad es demasiado alta para la insercion secuencial aleatoria,
    se cae a una red cubica con jitter (mismo phi, sin solapamientos).
    """
    # 2*R_OUT es la distancia por encima de la cual dos superelipsoides no pueden
    # solaparse en NINGUNA orientacion.  Usarla garantiza un estado inicial valido:
    # una particula que arranca solapada queda congelada (toda propuesta se rechaza).
    min_center_dist = 2.02 * R_OUT if min_center_dist is None else min_center_dist
    margin = 0.75 * a
    pos, tries = [], 0
    while len(pos) < N and tries < max_tries:
        tries += 1
        p = rng.uniform(margin, L - margin, size=3)
        if pos and np.min(np.linalg.norm(np.array(pos) - p, axis=1)) < min_center_dist:
            continue
        pos.append(p)
    if len(pos) >= N:
        return np.array(pos)

    # --- fallback: red cubica con jitter -------------------------------
    n = int(np.ceil(N ** (1.0 / 3.0)))
    while n ** 3 < N:
        n += 1
    step = (L - 2 * margin) / max(n - 1, 1)
    g = margin + np.arange(n) * step
    grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    sel = rng.choice(len(grid), size=N, replace=False)
    jit = 0.5 * max(step - min_center_dist, 0.0)
    return grid[sel] + rng.uniform(-jit, jit, size=(N, 3))


def belt(n_w=3, n_t=1, n_l=20, spacing=D_LATTICE, center=(0.0, 0.0, 0.0)):
    """Cinta belt_100: n_w de ancho (x), n_t de espesor (y), n_l de largo (z)."""
    ix, iy, iz = np.meshgrid(np.arange(n_w), np.arange(n_t), np.arange(n_l),
                             indexing="ij")
    idx = np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=1).astype(float)
    idx -= idx.mean(axis=0)
    return idx * spacing + np.asarray(center, float)


def _spacing_110(cache={}):
    """
    Espaciamiento lateral de equilibrio de dos nanocubos girados 45 grados
    alrededor del eje largo (contacto arista-arista en lugar de cara-cara).
    Se obtiene minimizando el mismo potencial vdW calibrado con la Fig. S28E.
    """
    if "d" not in cache:
        from geometry import Nanocube
        from rotations import rotation_matrix
        from energies import energy_vdw
        from gjk import gjk_overlap
        Rz = rotation_matrix([0, 0, 1], np.pi / 4)
        # solo distancias SIN solapamiento: la suma discretizada de la atraccion
        # diverge si los elementos de volumen se interpenetran.
        ds = [d for d in np.linspace(1.1 * a, 2.4 * a, 260)
              if not gjk_overlap(np.zeros(3), Rz, np.array([d, 0.0, 0.0]), Rz)]
        e = [energy_vdw(Nanocube([0, 0, 0], orientation=Rz),
                        Nanocube([d, 0, 0], orientation=Rz)) for d in ds]
        cache["d"] = float(ds[int(np.argmin(e))])
        cache["e"] = float(np.min(e))
    return cache["d"]


def belt_110(n_w=3, n_t=1, n_l=20, spacing_z=D_LATTICE, center=(0.0, 0.0, 0.0)):
    """
    Cinta belt_110: los nanocubos estan girados 45 grados alrededor del eje largo
    ("cube edge on top", arreglo tipo diamante de la Fig. 1E derecha).  A lo largo
    del campo el contacto sigue siendo cara-cara (Rz no toca las caras z), pero
    lateralmente es arista-arista, con mayor espaciamiento y menor acoplamiento
    vdW: por eso el paper reporta E_vdW menos negativa para los belts_110.

    Devuelve (posiciones, orientaciones).
    """
    from rotations import rotation_matrix
    dx = _spacing_110()
    ix, iy, iz = np.meshgrid(np.arange(n_w), np.arange(n_t), np.arange(n_l),
                             indexing="ij")
    idx = np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=1).astype(float)
    idx -= idx.mean(axis=0)
    pos = idx * np.array([dx, dx, spacing_z]) + np.asarray(center, float)
    Rz = rotation_matrix([0, 0, 1], np.pi / 4)
    return pos, np.repeat(Rz[None], len(pos), axis=0)


def double_belt(n_w=2, n_t=1, n_l=16, spacing=D_LATTICE, gap=1.6 * D_LATTICE):
    """Dos cintas paralelas (semilla del ensamble de doble hebra)."""
    b1 = belt(n_w, n_t, n_l, spacing, center=(-gap / 2, 0, 0))
    b2 = belt(n_w, n_t, n_l, spacing, center=(+gap / 2, 0, 0))
    return np.vstack([b1, b2])


def random_orientations(N, rng):
    """N matrices de rotacion uniformes (QR de una matriz gaussiana)."""
    out = np.empty((N, 3, 3))
    for k in range(N):
        Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
        Q = Q * np.sign(np.diag(R))
        if np.linalg.det(Q) < 0:
            Q[:, 2] *= -1
        out[k] = Q
    return out


def random_moments(N, rng):
    return random_unit_vector(rng, N)


def aligned_moments(N, direction=(0, 0, 1)):
    d = np.array(direction, float)
    d /= np.linalg.norm(d)
    return np.repeat(d[None], N, axis=0)
