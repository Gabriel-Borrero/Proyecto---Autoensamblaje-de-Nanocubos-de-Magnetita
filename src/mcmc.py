"""
mcmc.py  --  Pasos 6 y 7 de la metodologia.

Un ciclo MCMC (tal como se describe en S4):
  1. para cada nanocubo, en orden aleatorio:
         propuesta de traslacion  (vector aleatorio de longitud < delta_s)
         propuesta de rotacion    (eje aleatorio, angulo < delta_a)
         si GJK detecta solapamiento -> rechazo inmediato
         si no -> criterio de Metropolis con Delta E
  2. 10 pasos magneticos: para cada nanocubo, rotacion local del dipolo y Metropolis

delta_s y delta_a se ajustan +-5 % cada 10 ciclos (primeros 2000) y cada 1000
ciclos despues, para mantener una aceptacion cercana al 50 %.

Convenios de energia
--------------------
* Un movimiento MECANICO (posicion / orientacion) cambia E_A, E_dd y E_vdW,
  pero no E_Z (el dipolo vive en el marco de laboratorio).
* Un movimiento MAGNETICO cambia E_Z, E_A y E_dd, pero no E_vdW.
"""
import numpy as np

from parameters import (H, kBT, a, N_SURF, R_CUT_VDW, R_CUT_REP, DELTA_S,
                        DELTA_A, DELTA_M, N_MAG_STEPS, ADAPT_FRAC, TARGET_ACC)
from energies import (get_mesh, zeeman_array, anisotropy_array, vdw_sum,
                      K_Z, K_A1, K_d)
from rotations import random_unit_vector, rotation_matrix, orthonormalize
from geometry import R_OUT
from gjk import gjk_overlap


def _aniso(M_body):
    """K_A1 [(M'x M'y)^2 + (M'x M'z)^2 + (M'y M'z)^2]  con M' = R^T M."""
    x, y, z = M_body
    return K_A1 * ((x * y) ** 2 + (x * z) ** 2 + (y * z) ** 2)


class System:
    def __init__(self, positions, orientations=None, moments=None, field=H,
                 box=None, n_surf=N_SURF, vdw_scale=1.0, seed=0,
                 freeze_positions=False, kT=kBT, dd_cutoff=None,
                 dd_scale=1.0, mag_global_frac=0.1):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.pos = np.array(positions, dtype=float)
        self.N = len(self.pos)
        self.R = (np.repeat(np.eye(3)[None], self.N, axis=0)
                  if orientations is None else np.array(orientations, float))
        self.M = (random_unit_vector(rng, self.N) if moments is None
                  else np.array(moments, float))
        self.M /= np.linalg.norm(self.M, axis=1, keepdims=True)
        self.field = np.asarray(field, float)
        self.box = None if box is None else np.array(box, float)  # [[lo],[hi]]
        self.mesh = get_mesh(n_surf)
        self.vdw_scale = vdw_scale
        self.kT = kT
        self.freeze_positions = freeze_positions
        # dd_cutoff (nm): corte opcional del acoplamiento dipolar.  None = exacto
        # (todos los pares).  En filamentos, 8a captura > 97 % de la suma 1/r^3 y
        # baja el coste de O(N^2) a O(N n_vecinos); util a partir de N ~ 600.
        self.dd_cutoff2 = np.inf if dd_cutoff is None else float(dd_cutoff) ** 2
        # "parametros escalados" de la Fig. S23: el paper compensa el menor campo
        # efectivo de un filamento delgado reforzando el acoplamiento magnetico.
        # dd_scale multiplica K_d (1.0 = valor nominal del suplemento).
        self.K_d = K_d * float(dd_scale)
        self.dd_scale = float(dd_scale)
        # fraccion de pasos magneticos que proponen una direccion COMPLETAMENTE
        # aleatoria en lugar de una rotacion local (mejora la ergodicidad del
        # subsistema magnetico: zigzag <-> paralelo son minimos separados).
        self.mag_global_frac = float(mag_global_frac)
        self.delta_s, self.delta_a, self.delta_m = DELTA_S, DELTA_A, DELTA_M
        self.acc_mech = self.try_mech = 0
        self.acc_mag = self.try_mag = 0
        self.acc_mech_tot = self.try_mech_tot = 0   # contadores acumulados
        self.acc_mag_tot = self.try_mag_tot = 0     # (adapt() resetea los otros)
        self.acc_tw = self.try_tw = 0
        self.delta_tw = 0.05          # deg/nm, paso de la torsion colectiva
        self.collective_every = 0     # 0 = desactivado
        self._build_clouds()
        self._T = None

    # ------------------------------------------------------------------
    # nubes de elementos (se actualizan solo para la particula que se mueve)
    # ------------------------------------------------------------------
    def _clouds_of(self, pos_i, R_i):
        return (pos_i + self.mesh["vol"] @ R_i.T,
                pos_i + self.mesh["surf"] @ R_i.T)

    def _build_clouds(self):
        self.Vw = self.pos[:, None, :] + np.einsum("kj,nij->nki",
                                                   self.mesh["vol"], self.R)
        self.Sw = self.pos[:, None, :] + np.einsum("kj,nij->nki",
                                                   self.mesh["surf"], self.R)

    # ------------------------------------------------------------------
    # energias locales
    # ------------------------------------------------------------------
    def dd_of(self, i, pos_i, M_i):
        """E_dd de la particula i (pose y momento dados) con todas las demas."""
        rv = (self.pos - pos_i) / a
        r2 = np.einsum("ij,ij->i", rv, rv)
        r2[i] = np.inf
        if np.isfinite(self.dd_cutoff2):
            r2 = np.where(r2 * a ** 2 > self.dd_cutoff2, np.inf, r2)
        dot = self.M @ M_i
        pj = np.einsum("ij,ij->i", self.M, rv)
        pi = rv @ M_i
        return self.K_d * float(np.sum(r2 ** -1.5 * dot - 3.0 * pi * pj * r2 ** -2.5))

    def vdw_of(self, i, pos_i, R_i):
        """E_vdW de i con sus vecinas: atraccion hasta 2.5a, repulsion hasta 1.7a."""
        d = self.pos - pos_i
        r2 = np.einsum("ij,ij->i", d, d)
        r2[i] = np.inf
        ia = np.where(r2 < R_CUT_VDW ** 2)[0]
        if not len(ia):
            return 0.0
        ir = ia[r2[ia] < R_CUT_REP ** 2]
        Vi, Si = self._clouds_of(pos_i, R_i)
        return vdw_sum(Vi, Si, self.Vw[ia], self.Sw[ir], self.mesh, self.vdw_scale)

    def e_mech(self, i, pos_i, R_i):
        """Energia que cambia en un movimiento mecanico: E_A + E_dd + E_vdW."""
        return (_aniso(R_i.T @ self.M[i]) + self.dd_of(i, pos_i, self.M[i])
                + self.vdw_of(i, pos_i, R_i))

    def dipolar_tensor(self):
        """T_ij tal que F_i = sum_j T_ij M_j  y  E_dd(i) = M_i . F_i."""
        rv = (self.pos[None, :, :] - self.pos[:, None, :]) / a
        r2 = np.einsum("ijk,ijk->ij", rv, rv)
        np.fill_diagonal(r2, np.inf)
        if np.isfinite(self.dd_cutoff2):
            r2 = np.where(r2 * a ** 2 > self.dd_cutoff2, np.inf, r2)
        return (self.K_d * r2 ** -1.5)[:, :, None, None] * (
            np.eye(3)[None, None]
            - 3.0 * rv[:, :, :, None] * rv[:, :, None, :] / r2[:, :, None, None])

    # ------------------------------------------------------------------
    # energia total por componentes (kcal/mol por nanocubo)
    # ------------------------------------------------------------------
    def components(self, include_vdw=True):
        Ez = zeeman_array(self.M, self.field).sum()
        Ea = anisotropy_array(self.M, self.R).sum()
        F = np.einsum("ijab,jb->ia", self.dipolar_tensor(), self.M)
        Edd = 0.5 * float(np.einsum("ia,ia->", self.M, F))
        Ev = 0.0
        if include_vdw:
            for i in range(self.N):
                Ev += 0.5 * self.vdw_of(i, self.pos[i], self.R[i])
        n = self.N
        return dict(Ez=Ez / n, Ea=Ea / n, Edd=Edd / n, EvdW=Ev / n,
                    Emag=(Ez + Ea + Edd) / n, Eall=(Ez + Ea + Edd + Ev) / n)

    # ------------------------------------------------------------------
    # movimientos
    # ------------------------------------------------------------------
    def n_overlaps(self, i, pos_i, R_i):
        """Numero de nanocubos que solapan con la pose dada de i (via GJK)."""
        d = self.pos - pos_i
        r2 = np.einsum("ij,ij->i", d, d)
        r2[i] = np.inf
        n = 0
        for j in np.where(r2 < (2 * R_OUT) ** 2)[0]:
            if gjk_overlap(pos_i, R_i, self.pos[j], self.R[j]):
                n += 1
        return n

    def overlap_report(self):
        """Lista de pares solapados en el estado actual (debe estar vacia)."""
        out = []
        for i in range(self.N):
            d = self.pos - self.pos[i]
            r2 = np.einsum("ij,ij->i", d, d)
            r2[i] = np.inf
            for j in np.where(r2 < (2 * R_OUT) ** 2)[0]:
                if j > i and gjk_overlap(self.pos[i], self.R[i],
                                         self.pos[j], self.R[j]):
                    out.append((int(i), int(j)))
        return out

    def _inside_box(self, pos_i):
        if self.box is None:
            return True
        return bool(np.all(pos_i > self.box[0]) and np.all(pos_i < self.box[1]))

    def mechanical_move(self, i):
        self.try_mech += 1; self.try_mech_tot += 1
        new_pos = self.pos[i] + random_unit_vector(self.rng) * \
            self.rng.uniform(0, self.delta_s)
        if not self._inside_box(new_pos):
            return False
        new_R = orthonormalize(
            rotation_matrix(random_unit_vector(self.rng),
                            self.rng.uniform(-self.delta_a, self.delta_a))
            @ self.R[i])
        # GJK solo con candidatas cuyas esferas circunscritas se tocan.
        # Si la particula YA estaba solapada (estado inicial imperfecto), se
        # aceptan movimientos que reduzcan el numero de solapamientos; de lo
        # contrario quedaria congelada para siempre.
        n_new = self.n_overlaps(i, new_pos, new_R)
        if n_new:
            n_old = self.n_overlaps(i, self.pos[i], self.R[i])
            if n_old == 0 or n_new >= n_old:
                return False
        dE = (self.e_mech(i, new_pos, new_R)
              - self.e_mech(i, self.pos[i], self.R[i]))
        if metropolis(dE, self.kT, self.rng):
            self.pos[i], self.R[i] = new_pos, new_R
            self.Vw[i], self.Sw[i] = self._clouds_of(new_pos, new_R)
            self.acc_mech += 1; self.acc_mech_tot += 1
            return True
        return False

    def magnetic_move(self, i, F):
        """Rotacion local del dipolo; dE se evalua de forma incremental."""
        self.try_mag += 1; self.try_mag_tot += 1
        M_old = self.M[i]
        if self.rng.random() < self.mag_global_frac:
            M_new = random_unit_vector(self.rng)       # reorientacion global
        else:
            M_new = rotation_matrix(random_unit_vector(self.rng),
                                    self.rng.uniform(-self.delta_m,
                                                     self.delta_m)) @ M_old
            M_new /= np.linalg.norm(M_new)
        dM = M_new - M_old
        dE = (-K_Z * float(self.field @ dM) + float(dM @ F[i])
              + _aniso(self.R[i].T @ M_new) - _aniso(self.R[i].T @ M_old))
        if metropolis(dE, self.kT, self.rng):
            self.M[i] = M_new
            # F_j = sum_k T_jk M_k  ->  solo hay que sumar T_ji dM (O(N))
            F += self._T[:, i, :, :] @ dM
            self.acc_mag += 1; self.acc_mag_tot += 1
            return True
        return False

    # ------------------------------------------------------------------
    # ciclo completo
    # ------------------------------------------------------------------
    def cycle(self, index=0):
        if (self.collective_every and not self.freeze_positions
                and index % self.collective_every == 0):
            self.collective_twist_move()
        if not self.freeze_positions:
            for i in self.rng.permutation(self.N):
                self.mechanical_move(i)
            self._T = None
        if getattr(self, "_T", None) is None:
            self._T = self.dipolar_tensor()
        F = np.einsum("ijab,jb->ia", self._T, self.M)   # campo local, una vez
        for _ in range(N_MAG_STEPS):
            for i in self.rng.permutation(self.N):
                self.magnetic_move(i, F)

    # ------------------------------------------------------------------
    # movimiento COLECTIVO de torsion
    # ------------------------------------------------------------------
    # La torsion de la cinta es una coordenada colectiva lentisima: con
    # movimientos de una sola particula hace falta que cada nanocubo difunda un
    # parametro de red entero (~10^4 ciclos).  Este movimiento propone un
    # incremento global de torsion d(omega) alrededor del eje del filamento y lo
    # acepta o rechaza con el mismo criterio de Metropolis.  La propuesta es
    # simetrica (d(omega) uniforme en [-delta, +delta]) y la transformacion es
    # invertible, asi que el balance detallado se conserva y la distribucion
    # muestreada sigue siendo la canonica: solo se acelera, no se sesga.
    #
    # Al girar posicion, orientacion y momento de cada capa por la MISMA Rz,
    # E_A y E_Z quedan invariantes (Rz conserva la componente z de M y M'=R^T M),
    # de modo que solo hay que reevaluar E_dd y E_vdW.

    def total_vdw(self, pos=None, R=None):
        """E_vdW total (kcal/mol) de una configuracion arbitraria."""
        if pos is None:
            return sum(0.5 * self.vdw_of(i, self.pos[i], self.R[i])
                       for i in range(self.N))
        Vw_old, Sw_old, p_old, R_old = self.Vw, self.Sw, self.pos, self.R
        self.pos, self.R = pos, R
        self._build_clouds()
        e = sum(0.5 * self.vdw_of(i, pos[i], R[i]) for i in range(self.N))
        self.pos, self.R, self.Vw, self.Sw = p_old, R_old, Vw_old, Sw_old
        return e

    def total_dd(self, pos=None, M=None):
        """E_dd total (kcal/mol) de una configuracion arbitraria."""
        p = self.pos if pos is None else pos
        m = self.M if M is None else M
        rv = (p[None, :, :] - p[:, None, :]) / a
        r2 = np.einsum("ijk,ijk->ij", rv, rv)
        np.fill_diagonal(r2, np.inf)
        if np.isfinite(self.dd_cutoff2):
            r2 = np.where(r2 * a ** 2 > self.dd_cutoff2, np.inf, r2)
        dot = m @ m.T
        proj = np.einsum("ijk,ik->ij", rv, m)          # (M_i . r_ij)
        # ojo con el signo: proj.T[i,j] = r_ji . M_j = -(r_ij . M_j), de modo que
        # -3 (M_i.r)(M_j.r) = +3 proj proj^T
        return 0.5 * self.K_d * float(np.sum(r2 ** -1.5 * dot
                                        + 3.0 * proj * proj.T * r2 ** -2.5))

    def twisted_copy(self, delta_deg_nm):
        """Aplica una torsion rigida global de delta (deg/nm) sobre el eje z."""
        z0 = self.pos[:, 2].mean()
        c = self.pos[:, :2].mean(axis=0)
        ang = np.radians(delta_deg_nm) * (self.pos[:, 2] - z0)
        ca, sa = np.cos(ang), np.sin(ang)
        dx, dy = self.pos[:, 0] - c[0], self.pos[:, 1] - c[1]
        pos = self.pos.copy()
        pos[:, 0] = c[0] + ca * dx - sa * dy
        pos[:, 1] = c[1] + sa * dx + ca * dy
        Rz = np.zeros((self.N, 3, 3))
        Rz[:, 0, 0] = ca; Rz[:, 0, 1] = -sa
        Rz[:, 1, 0] = sa; Rz[:, 1, 1] = ca
        Rz[:, 2, 2] = 1.0
        return pos, Rz @ self.R, np.einsum("nij,nj->ni", Rz, self.M)

    def collective_twist_move(self):
        self.try_tw += 1
        delta = self.rng.uniform(-self.delta_tw, self.delta_tw)
        pos, R, M = self.twisted_copy(delta)
        # solapamientos: solo pares que se acercaron
        for i in range(self.N):
            d = pos - pos[i]
            r2 = np.einsum("ij,ij->i", d, d)
            r2[i] = np.inf
            for j in np.where(r2 < (2 * R_OUT) ** 2)[0]:
                if j > i and gjk_overlap(pos[i], R[i], pos[j], R[j]):
                    return False
        e_new = self.total_dd(pos, M) + self.total_vdw(pos, R)
        e_old = self.total_dd() + self.total_vdw()
        if metropolis(e_new - e_old, self.kT, self.rng):
            self.pos, self.R, self.M = pos, R, M
            self._build_clouds()
            self._T = None
            self.acc_tw += 1
            return True
        return False

    def acceptance(self):
        """Tasas de aceptacion acumuladas (mecanica, magnetica)."""
        return (self.acc_mech_tot / max(self.try_mech_tot, 1),
                self.acc_mag_tot / max(self.try_mag_tot, 1))

    def adapt(self, cycle_index):
        every = 10 if cycle_index < 2000 else 1000
        if (cycle_index + 1) % every:
            return
        if self.try_mech:
            r = self.acc_mech / self.try_mech
            f = (1 + ADAPT_FRAC) if r > TARGET_ACC else (1 - ADAPT_FRAC)
            self.delta_s = float(np.clip(self.delta_s * f, 1e-3 * a, 0.5 * a))
            self.delta_a = float(np.clip(self.delta_a * f, 1e-3, 0.8))
            self.acc_mech = self.try_mech = 0
        if self.try_mag:
            r = self.acc_mag / self.try_mag
            f = (1 + ADAPT_FRAC) if r > TARGET_ACC else (1 - ADAPT_FRAC)
            self.delta_m = float(np.clip(self.delta_m * f, 1e-3, np.pi))
            self.acc_mag = self.try_mag = 0

    def run(self, n_cycles, record_every=25, frame_every=None, verbose=False,
            include_vdw=True):
        hist, frames = [], []
        frame_every = frame_every or max(1, n_cycles // 40)
        for c in range(n_cycles):
            self.cycle(c)
            self.adapt(c)
            if c % record_every == 0 or c == n_cycles - 1:
                comp = self.components(include_vdw=include_vdw)
                comp["cycle"] = c
                comp["Mz"] = float(self.M[:, 2].mean())
                hist.append(comp)
                if verbose and (c % (record_every * 8) == 0):
                    print(f"  ciclo {c:6d}  Eall={comp['Eall']:8.3f} "
                          f"Emag={comp['Emag']:8.3f} <Mz>={comp['Mz']:.3f}",
                          flush=True)
            if c % frame_every == 0 or c == n_cycles - 1:
                frames.append(dict(cycle=c, pos=self.pos.copy(),
                                   R=self.R.copy(), M=self.M.copy()))
        return hist, frames


def metropolis(delta_E, kT=kBT, rng=None):
    """Acepta si dE <= 0, o con probabilidad exp(-dE/kT)."""
    if delta_E <= 0.0:
        return True
    rng = np.random.default_rng() if rng is None else rng
    return rng.random() < np.exp(-delta_E / kT)
