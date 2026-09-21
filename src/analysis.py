"""
analysis.py  --  Paso 8: energias y estructura.

Herramientas:
  cluster_labels     union-find con criterio de contacto (centro-centro < 1.5 a)
  cluster_stats      tamano medio/maximo, fraccion ensamblada, forma (tensor de giro)
  radial_distribution g(r)
  filament_twist     torsion acumulada, quiralidad, radio y paso de la helice
  nematic_order      parametro de orden S de los vectores de enlace respecto a H
"""
import numpy as np

from parameters import a, V_NC, D_LATTICE


# ----------------------------------------------------------------------
def cluster_labels(pos, cutoff=1.5 * a):
    """Etiqueta de cluster para cada particula (percolacion por contacto)."""
    N = len(pos)
    parent = np.arange(N)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    ii, jj = np.where((d < cutoff) & (np.arange(N)[:, None] < np.arange(N)[None, :]))
    for i, j in zip(ii, jj):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    return np.array([find(i) for i in range(N)])


def gyration_shape(points):
    """Autovalores del tensor de giro (ordenados) y descriptores de forma."""
    p = points - points.mean(axis=0)
    G = p.T @ p / len(p)
    w = np.sort(np.linalg.eigvalsh(G))[::-1]
    Rg2 = w.sum()
    asph = w[0] - 0.5 * (w[1] + w[2])          # >0 -> alargado (filamento)
    acyl = w[1] - w[2]                          # >0 -> aplanado (cinta)
    return dict(eig=w, Rg=np.sqrt(Rg2),
                asphericity=asph / Rg2 if Rg2 else 0.0,
                acylindricity=acyl / Rg2 if Rg2 else 0.0)


def cluster_stats(pos, cutoff=1.5 * a, min_size=3):
    lab = cluster_labels(pos, cutoff)
    uniq, counts = np.unique(lab, return_counts=True)
    N = len(pos)
    big = uniq[counts >= min_size]
    shapes = [gyration_shape(pos[lab == u]) for u in big]
    return dict(labels=lab,
                n_clusters=len(uniq),
                sizes=np.sort(counts)[::-1],
                mean_size=float(counts.mean()),
                weighted_mean_size=float((counts ** 2).sum() / counts.sum()),
                max_size=int(counts.max()),
                assembled_fraction=float(counts[counts >= min_size].sum() / N),
                shapes=shapes)


def nematic_order(pos, axis=(0, 0, 1), cutoff=1.5 * a):
    """S = <3cos^2(theta)-1>/2 de los vectores de enlace respecto al campo."""
    n = np.asarray(axis, float)
    n /= np.linalg.norm(n)
    d = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(d, axis=-1)
    m = (r < cutoff) & (r > 1e-9)
    ii, jj = np.where(np.triu(m, 1))
    if not len(ii):
        return 0.0
    u = d[ii, jj] / r[ii, jj][:, None]
    c = u @ n
    return float((3 * c ** 2 - 1).mean() / 2)


def radial_distribution(pos, L=None, rmax=6 * a, nbins=90):
    """g(r) normalizada con la densidad media del sistema."""
    N = len(pos)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    r = d[np.triu_indices(N, 1)]
    hist, edges = np.histogram(r, bins=nbins, range=(0, rmax))
    rc = 0.5 * (edges[1:] + edges[:-1])
    if L is None:
        L = pos.max(axis=0) - pos.min(axis=0)
        V = np.prod(np.maximum(L, a))
    else:
        V = L ** 3
    rho = N / V
    shell = 4 * np.pi * rc ** 2 * (edges[1] - edges[0])
    g = hist / (0.5 * N * rho * shell)
    return rc, g


# ----------------------------------------------------------------------
def _layer_lattice_angle(q, psi_min=0.25):
    """
    Orientacion del reticulo de una capa, modulo 90 grados, via el parametro
    de orden orientacional de enlace de orden 4:  psi4 = <exp(4 i theta)>.

    Sustituye al autovector principal de la covarianza, que es DEGENERADO en
    secciones cuadradas (los dos autovalores coinciden y la direccion la elige
    el ruido termico, dando torsiones espurias de cientos de grados).  psi4
    funciona igual para una fila (n x 1) que para una seccion n x n, porque en
    ambos casos los enlaces del reticulo estan a 0 y 90 grados.
    """
    if len(q) < 2:
        return None
    d = q[:, None, :] - q[None, :, :]
    r = np.linalg.norm(d, axis=-1)
    off = r > 1e-9
    if not off.any():
        return None
    # solo los primeros vecinos: la diagonal esta a sqrt(2) r_min y sus enlaces
    # a 45 grados cancelan exactamente a los axiales en psi4.
    sel = off & (r < 1.25 * r[off].min())
    ii, jj = np.where(sel)
    if not len(ii):
        return None
    th = np.arctan2(d[ii, jj, 1], d[ii, jj, 0])
    psi = np.mean(np.exp(4j * th))
    if abs(psi) < psi_min:          # la capa perdio el orden del reticulo
        return None
    return float(np.angle(psi) / 4.0)


def filament_twist(pos, axis=2, layer_h=None, min_per_layer=2):
    """
    Torsion de una cinta o filamento alineado con el campo.

    Se agrupan los nanocubos en capas perpendiculares al eje y se mide el giro
    del RETICULO de cada capa (psi4, modulo 90 grados) respecto a la anterior.
    Los giros entre capas consecutivas son pequenos, asi que la ambiguedad
    modulo 90 grados no afecta a la suma acumulada.

      total_twist   torsion acumulada (grados)
      twist_rate    grados por nm
      handedness    +1 derecha, -1 izquierda, 0 sin resolver
      pitch         longitud de una vuelta completa (nm)
      width         anchura media de la seccion (nm)
      coil_radius   amplitud transversal del centroide (nm) -> superenrollamiento
    """
    layer_h = layer_h or 0.6 * D_LATTICE
    perp = [k for k in range(3) if k != axis]
    z = pos[:, axis]
    bins = np.floor((z - z.min()) / layer_h).astype(int)
    ang, zc, widths, cents = [], [], [], []
    for b in np.unique(bins):
        sel = np.where(bins == b)[0]
        if len(sel) < min_per_layer:
            continue
        q = pos[sel][:, perp]
        a_ = _layer_lattice_angle(q)
        if a_ is None:
            continue
        cents.append(q.mean(axis=0))
        w = np.linalg.eigvalsh((q - q.mean(0)).T @ (q - q.mean(0)) / len(q))
        widths.append(2.0 * np.sqrt(max(w[-1], 0.0)))
        ang.append(a_)
        zc.append(z[sel].mean())
    if len(ang) < 4:
        return dict(total_twist=0.0, twist_rate=0.0, handedness=0,
                    pitch=np.inf, width=0.0, coil_radius=0.0,
                    twist_per_layer=np.array([]), z=np.array(zc))
    th, zc = np.array(ang), np.array(zc)
    d = np.diff(th)
    d = (d + np.pi / 4) % (np.pi / 2) - np.pi / 4      # reticulo cuadrado: mod 90
    total = float(np.degrees(d.sum()))
    dz = zc[-1] - zc[0]
    rate = total / dz if abs(dz) > 1e-9 else 0.0
    cents = np.array(cents)
    coil = float(np.linalg.norm(cents - cents.mean(axis=0), axis=1).mean())
    return dict(total_twist=total, twist_rate=rate,
                handedness=int(np.sign(d.sum())) if abs(total) > 20 else 0,
                pitch=float(360.0 / abs(rate)) if abs(rate) > 1e-6 else np.inf,
                width=float(np.mean(widths)), coil_radius=coil,
                twist_per_layer=np.degrees(d), z=zc)


def dipole_zigzag(M, pos, axis=2):
    """Grado de configuracion zigzag: dispersion angular de M en el plano xz."""
    ang = np.degrees(np.arctan2(M[:, 0], M[:, axis]))
    return dict(mean_abs_tilt=float(np.abs(ang).mean()),
                std_tilt=float(ang.std()))


# ----------------------------------------------------------------------
# Campo efectivo y coordinacion  --  el enlace fisico densidad -> helice
# ----------------------------------------------------------------------
from parameters import G_PER_DIPOLE, H as H_EXT


def effective_field(pos, M, field=None, r_cut=6.0 * a):
    """
    Campo efectivo B que actua sobre cada nanocubo (en Gauss):

        H_eff,i = H_ext + sum_j (K_d/K_Z) [3(M_j.r)r - M_j] / (r_ij/a)^3

    Es la magnitud que el suplemento (S3, Figs. S17 y S22) identifica como la
    que controla el tipo de superestructura: chains -> belts_100 -> belts_110
    -> helices al aumentar B.  Como el termino dipolar crece con el numero de
    vecinos, la DENSIDAD de nanocubos se traduce directamente en un B mayor.
    """
    field = H_EXT if field is None else np.asarray(field, float)
    rv = (pos[None, :, :] - pos[:, None, :]) / a          # (i,j,3) en unidades de a
    r2 = np.einsum("ijk,ijk->ij", rv, rv)
    np.fill_diagonal(r2, np.inf)
    r2[r2 > (r_cut / a) ** 2] = np.inf
    r = np.sqrt(r2)
    rhat = rv / r[:, :, None]
    proj = np.einsum("ijk,jk->ij", rhat, M)
    Hd = G_PER_DIPOLE * (3.0 * proj[:, :, None] * rhat - M[None, :, :]) / r[:, :, None] ** 3
    Hd = np.nansum(np.where(np.isfinite(Hd), Hd, 0.0), axis=1)
    Heff = field[None, :] + Hd
    mag = np.linalg.norm(Heff, axis=1)
    return dict(vectors=Heff,
                mean=float(mag.mean()), median=float(np.median(mag)),
                p90=float(np.percentile(mag, 90)),
                mean_dipolar=float(np.linalg.norm(Hd, axis=1).mean()))


def coordination(pos, cutoff=1.45 * a):
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    n = (d < cutoff).sum(axis=1)
    return dict(mean=float(n.mean()), max=int(n.max()),
                fraction_isolated=float((n == 0).mean()), per_particle=n)


def bond_chirality(pos, M, cutoff=1.45 * a):
    """
    Quiralidad escalar del campo de dipolos:
        chi = < (M_i x M_j) . r_ij >   sobre pares vecinos.
    chi != 0 indica una textura dipolar con mano definida; su signo distingue
    helices derechas de izquierdas (cf. Figs. S21 y S26-S27 del suplemento).
    """
    d = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(d, axis=-1)
    m = np.triu((r < cutoff) & (r > 1e-9), 1)
    ii, jj = np.where(m)
    if not len(ii):
        return dict(chi=0.0, n_bonds=0, tilt_mean=0.0)
    u = d[ii, jj] / r[ii, jj][:, None]
    chi = np.einsum("ij,ij->i", np.cross(M[ii], M[jj]), u)
    cos = np.clip(np.einsum("ij,ij->i", M[ii], M[jj]), -1, 1)
    return dict(chi=float(chi.mean()), n_bonds=int(len(ii)),
                tilt_mean=float(np.degrees(np.arccos(np.abs(cos))).mean()),
                chi_abs=float(np.abs(chi).mean()))


def effective_field(system):
    """
    Campo efectivo (en Gauss) que siente cada nanocubo:
        B_eff = H_externo + H_dipolar,   H_dipolar = -F_i / K_Z
    donde F_i = sum_j T_ij M_j es el campo local de los dipolos vecinos y K_Z
    es la constante Zeeman (E_Z = -K_Z H . M).  Es la variable que, segun S3 del
    suplemento, controla el tipo de superestructura:
        B bajo -> cadenas cortas ; intermedio -> belts ; alto -> helices.
    """
    from parameters import K_Z
    T = system.dipolar_tensor()
    F = np.einsum("ijab,jb->ia", T, system.M)
    H_dip = -F / K_Z
    H_eff = system.field[None, :] + H_dip
    mag = np.linalg.norm(H_eff, axis=1)
    return dict(H_dip_z=float(H_dip[:, 2].mean()),
                H_eff_z=float(H_eff[:, 2].mean()),
                H_eff_mag=float(mag.mean()),
                H_eff_std=float(mag.std()),
                gain=float(np.linalg.norm(H_eff, axis=1).mean()
                           / np.linalg.norm(system.field)))
