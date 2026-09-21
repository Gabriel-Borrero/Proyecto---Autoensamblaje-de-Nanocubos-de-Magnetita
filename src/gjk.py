"""
gjk.py  --  Paso 5 de la metodologia.

Gilbert-Johnson-Keerthi: decide si dos cuerpos convexos se solapan.
NO es una energia: una propuesta que produce solapamiento se rechaza de plano.

Idea: dos convexos A y B se intersectan  <=>  el origen pertenece a la
diferencia de Minkowski A (-) B.  El GJK construye iterativamente un simplex
(hasta un tetraedro) de puntos de A(-)B usando solo la funcion soporte
    s_{A-B}(d) = s_A(d) - s_B(-d)
y comprueba si ese tetraedro encierra el origen.
"""
import numpy as np

from geometry import support_superellipsoid, R_IN, R_OUT


def support_minkowski(ci, Ri, cj, Rj, d):
    pi = ci + Ri @ support_superellipsoid(Ri.T @ d)
    pj = cj + Rj @ support_superellipsoid(-(Rj.T @ d))
    return pi - pj


def _cross(u, v):
    """Producto cruz explicito (np.cross tiene demasiado overhead aqui)."""
    return np.array([u[1] * v[2] - u[2] * v[1],
                     u[2] * v[0] - u[0] * v[2],
                     u[0] * v[1] - u[1] * v[0]])


def _tiny(v):
    return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) < 1e-20


def _same_direction(a_, b_):
    return (a_[0] * b_[0] + a_[1] * b_[1] + a_[2] * b_[2]) > 0.0


def _do_simplex(simplex, d):
    """Actualiza el simplex y la direccion de busqueda. True si contiene 0."""
    if len(simplex) == 2:
        b, aa = simplex[0], simplex[1]
        ab, ao = b - aa, -aa
        if _same_direction(ab, ao):
            d[:] = _cross(_cross(ab, ao), ab)
            if _tiny(d):
                d[:] = _cross(ab, np.array([1.0, 0.0, 0.0]))
                if _tiny(d):
                    d[:] = _cross(ab, np.array([0.0, 1.0, 0.0]))
        else:
            simplex[:] = [aa]
            d[:] = ao
        return False

    if len(simplex) == 3:
        c, b, aa = simplex
        ab, ac, ao = b - aa, c - aa, -aa
        abc = _cross(ab, ac)
        if _same_direction(_cross(abc, ac), ao):
            if _same_direction(ac, ao):
                simplex[:] = [c, aa]
                d[:] = _cross(_cross(ac, ao), ac)
            else:
                simplex[:] = [b, aa]
                return _do_simplex(simplex, d)
        elif _same_direction(_cross(ab, abc), ao):
            simplex[:] = [b, aa]
            return _do_simplex(simplex, d)
        else:
            if _same_direction(abc, ao):
                d[:] = abc
            else:
                simplex[:] = [b, c, aa]
                d[:] = -abc
        return False

    # tetraedro
    dd, c, b, aa = simplex
    ab, ac, ad, ao = b - aa, c - aa, dd - aa, -aa
    abc, acd, adb = _cross(ab, ac), _cross(ac, ad), _cross(ad, ab)
    if _same_direction(abc, ao):
        simplex[:] = [c, b, aa]
        d[:] = abc
        return False
    if _same_direction(acd, ao):
        simplex[:] = [dd, c, aa]
        d[:] = acd
        return False
    if _same_direction(adb, ao):
        simplex[:] = [b, dd, aa]
        d[:] = adb
        return False
    return True


def gjk_overlap(ci, Ri, cj, Rj, max_iter=32):
    """True si los dos nanocubos se solapan."""
    delta = np.asarray(cj, dtype=float) - np.asarray(ci, dtype=float)
    dist = np.linalg.norm(delta)
    # pre-filtros rapidos con las esferas inscrita y circunscrita
    if dist >= 2.0 * R_OUT:
        return False
    if dist <= 2.0 * R_IN:
        return True

    d = np.array([1.0, 0.0, 0.0]) if dist == 0 else -delta / dist
    simplex = [support_minkowski(ci, Ri, cj, Rj, d)]
    d = np.array(-simplex[0], dtype=float)
    for _ in range(max_iter):
        if _tiny(d):
            return True
        p = support_minkowski(ci, Ri, cj, Rj, d)
        if float(np.dot(p, d)) < 0.0:
            return False
        simplex.append(p)
        if _do_simplex(simplex, d):
            return True
    return False


def any_overlap(i, pos_i, R_i, pos, R, candidates):
    """True si la particula i (en la pose propuesta) choca con alguna candidata."""
    for j in candidates:
        if j == i:
            continue
        if gjk_overlap(pos_i, R_i, pos[j], R[j]):
            return True
    return False
