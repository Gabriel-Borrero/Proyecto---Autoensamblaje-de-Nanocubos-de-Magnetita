"""
geometry.py  --  Paso 2 de la metodologia.

El nanocubo NC_i = (r_i, R_i, M_i):
    r_i  posicion del centro                (nm)
    R_i  matriz de rotacion del cuerpo      (3x3 ortogonal)
    M_i  vector unitario del super-dipolo   (marco de laboratorio)

La forma es el superelipsoide  x^6 + y^6 + z^6 = (a/2)^6  del suplemento (S4),
que es un cubo con esquinas redondeadas (bluntness intermedio, Fig. S28A).

Aqui tambien se construyen:
  * los 27 elementos de volumen   -> parte atractiva de E_vdW
  * los N_s elementos de superficie con su area -> parte repulsiva de E_vdW
  * la funcion soporte del superelipsoide  -> algoritmo GJK (ec. 7)
"""
import numpy as np

from parameters import a, P_SUPER, V_NC, N_VOL, N_SURF


# ----------------------------------------------------------------------
# Objeto pedagogico de una sola particula
# ----------------------------------------------------------------------
class Nanocube:
    def __init__(self, position, orientation=None, magnetic_moment=None):
        self.position = np.array(position, dtype=float)
        self.orientation = (np.eye(3) if orientation is None
                            else np.array(orientation, dtype=float))
        if magnetic_moment is None:
            self.magnetic_moment = np.array([0.0, 0.0, 1.0])
        else:
            M = np.array(magnetic_moment, dtype=float)
            self.magnetic_moment = M / np.linalg.norm(M)

    # --- utilidades -------------------------------------------------
    def body_frame(self, v):
        """Pasa un vector del laboratorio al marco del cubo: M' = R^T M."""
        return self.orientation.T @ np.asarray(v, dtype=float)

    def vertices(self, scale=1.0):
        """8 vertices del cubo equivalente (para dibujar)."""
        s = 0.5 * a * scale
        v = np.array([[sx, sy, sz] for sx in (-s, s) for sy in (-s, s)
                      for sz in (-s, s)])
        return self.position + v @ self.orientation.T

    def __repr__(self):
        return f"Nanocube(r={self.position.round(2)}, M={self.magnetic_moment.round(3)})"


# ----------------------------------------------------------------------
# Mallas de discretizacion (ecuacion 4 del suplemento)
# ----------------------------------------------------------------------
def volume_elements(n_side=3):
    """n_side^3 = 27 elementos identicos de volumen en el marco del cubo."""
    c = (np.arange(n_side) - (n_side - 1) / 2) * (a / n_side)
    pts = np.array(np.meshgrid(c, c, c, indexing="ij")).reshape(3, -1).T
    vol = np.full(len(pts), V_NC / len(pts))
    return pts, vol


def surface_elements(n=N_SURF, p=P_SUPER):
    """
    n elementos sobre la superficie del superelipsoide, con su area dS.
    Direcciones cuasi-uniformes (espiral de Fibonacci); para cada direccion u
    el radio del superelipsoide es  r(u) = (a/2) / (sum |u_i|^p)^(1/p),
    y el elemento de area es  dS = r^2 dOmega / (u . n_hat).
    """
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    u = np.stack([np.cos(theta) * np.sin(phi),
                  np.sin(theta) * np.sin(phi),
                  np.cos(phi)], axis=1)
    r = (a / 2.0) / np.sum(np.abs(u) ** p, axis=1) ** (1.0 / p)
    pts = r[:, None] * u
    nrm = np.sign(pts) * np.abs(pts) ** (p - 1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    cos_un = np.einsum("ij,ij->i", u, nrm)
    dS = r ** 2 * (4.0 * np.pi / n) / cos_un
    return pts, dS


def support_superellipsoid(direction, p=P_SUPER, half=a / 2.0):
    """
    Funcion soporte (ec. 7 del suplemento): punto de la superficie mas lejano
    en la direccion d.  Multiplicadores de Lagrange sobre x^p+y^p+z^p=(a/2)^p
    dan  x_i = (a/2) sgn(d_i) |d_i|^{1/(p-1)} / (sum |d_j|^{p/(p-1)})^{1/p}.
    """
    d = np.asarray(direction, dtype=float)
    q = 1.0 / (p - 1.0)
    ad = np.abs(d) ** q
    denom = np.sum(np.abs(d) ** (p * q)) ** (1.0 / p)
    if denom == 0.0:
        return np.zeros(3)
    return half * np.sign(d) * ad / denom


def support_nanocube(center, R, direction):
    """Funcion soporte de un nanocubo con centro y orientacion dados."""
    d_body = R.T @ np.asarray(direction, dtype=float)
    return center + R @ support_superellipsoid(d_body)


# Radios caracteristicos usados como pre-filtro rapido del GJK
R_IN = a / 2.0                                    # esfera inscrita (caras)
R_OUT = float(np.linalg.norm(support_superellipsoid(np.ones(3))))  # circunscrita
