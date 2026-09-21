"""
rotations.py  --  Paso 3 de la metodologia.

Rotaciones 3D con matrices ortogonales y formula de Rodrigues.
Se distingue explicitamente entre:
  * rotar la GEOMETRIA del cubo  (cambia R, por tanto cambia E_A pero no E_Z)
  * rotar el MOMENTO magnetico   (cambia M en el marco de laboratorio)
"""
import numpy as np


def rotation_matrix(axis, angle):
    """Matriz de rotacion de Rodrigues alrededor de 'axis' por 'angle' (rad)."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def rotate_vector(vector, axis, angle):
    return rotation_matrix(axis, angle) @ np.asarray(vector, dtype=float)


def rotate_cube(cube, axis, angle):
    """Rota la orientacion del cuerpo rigido (no toca el momento magnetico)."""
    cube.orientation = rotation_matrix(axis, angle) @ cube.orientation
    return cube


def random_unit_vector(rng, n=None):
    """Muestreo uniforme sobre la esfera S^2 (Marsaglia 1972, ref. 59 del paper)."""
    if n is None:
        v = rng.normal(size=3)
        return v / np.linalg.norm(v)
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def random_small_rotation(rng, delta):
    """Rotacion aleatoria de eje uniforme y angulo uniforme en (-delta, delta)."""
    return rotation_matrix(random_unit_vector(rng), rng.uniform(-delta, delta))


def orthonormalize(R):
    """
    Re-ortonormaliza para evitar deriva numerica.  Gram-Schmidt explicito:
    ~3 us frente a ~12 us de la SVD, y se llama una vez por movimiento.
    """
    u0 = R[:, 0] / np.linalg.norm(R[:, 0])
    u1 = R[:, 1] - (R[:, 1] @ u0) * u0
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(u0, u1)
    return np.stack([u0, u1, u2], axis=1)
