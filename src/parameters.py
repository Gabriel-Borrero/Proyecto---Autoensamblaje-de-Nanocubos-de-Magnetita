"""
parameters.py  --  Paso 1 de la metodologia.

Todas las constantes fisicas y de simulacion en un solo lugar.
Fuente: Singh, Chan, Baskin, Gelman, Repnin, Kral & Klajn,
"Self-assembly of magnetite nanocubes into helical superstructures",
Science 345, 1149 (2014), Supplementary Material, secciones S3-S4.

Unidades del modelo
-------------------
energia   : kcal/mol
longitud  : nm
campo H   : Gauss
r_dd      : en el termino dipolo-dipolo la distancia se mide en unidades de a
"""
import numpy as np

# ----------------------------------------------------------------------
# Geometria del nanocubo  (S4)
# ----------------------------------------------------------------------
a = 13.37                      # nm, arista promedio medida por TEM (Fig. S1)
P_SUPER = 6                    # superelipsoide x^6 + y^6 + z^6 = (a/2)^6
V_NC = 0.9 * a ** 3            # nm^3, V = (2a^3/sqrt(pi)) Gamma^3(7/6) ~ 0.9 a^3

# ----------------------------------------------------------------------
# Magnetismo  (ecuaciones 1-3 del suplemento)
# ----------------------------------------------------------------------
Ms = 480e3                     # A/m, magnetizacion de saturacion de magnetita bulk
ms = Ms * V_NC * 1e-27         # A m^2, momento intrinseco (~1.174e-18)
K_Z = 1.647e-2                 # kcal/mol/G   (= 2.77e-2 kT a 300 K)
K_A1 = -3.784                  # kcal/mol, anisotropia magnetocristalina cuartica
K_A2 = -0.963                  # kcal/mol, sexto orden (se desprecia, ~4x menor)
K_d = 7.973                    # kcal/mol = mu0 ms^2 / 4pi a^3 (r en unidades de a)

# Campo (en Gauss) que un dipolo vecino genera sobre otro a distancia r = a:
# E_dd = -K_Z H_j . M_i   con   H_j = (K_d/K_Z)[3(M_j.r)r - M_j]/(r/a)^3
G_PER_DIPOLE = K_d / K_Z       # 484.1 G  <- origen del "campo efectivo" B (S3)

# ----------------------------------------------------------------------
# Interaccion no magnetica  (ecuacion 4 del suplemento)
# ----------------------------------------------------------------------
A_H = 3.0                      # kcal/mol, constante de Hamaker magnetita/hexano
EPS_1 = 130.0                  # parametro de ajuste (atraccion)
EPS_2 = 290.0                  # parametro de ajuste (repulsion)
BETA = 9.56                    # nm, parametro de ajuste de forma
K_W_PAPER = 2.5e5              # nm^6 kcal/mol
N_VOL = 27                     # 3x3x3 elementos de volumen (atraccion ~ 1/r^6)
N_SURF = 54                    # elementos de superficie (repulsion ~ 1/r^8)
N_SURF_REF = 386               # discretizacion del suplemento (usada para validar)

# Anclajes cuantitativos de la Fig. S28E: el minimo del potencial efectivo vdW
# esta en una separacion superficie-superficie de 2.99 nm con profundidad 2.33
# kcal/mol por nanocubo. Con ellos se recalibran los prefactores (ver energies.py).
D_MIN_FF = 2.99                # nm
E_MIN_FF = -2.33               # kcal/mol

# Parametro de red de un belt/helice perfecta: cubo + gap de equilibrio
D_LATTICE = a + D_MIN_FF       # 16.36 nm = 1.2237 a

# ----------------------------------------------------------------------
# Termodinamica y campo
# ----------------------------------------------------------------------
T = 300.0                      # K
k_B = 0.0019872041             # kcal/(mol K)
kBT = k_B * T                  # 0.5961 kcal/mol
H0 = 417.0                     # G, campo intermedio usado en la Fig. S23
H = np.array([0.0, 0.0, H0])   # campo externo a lo largo de z
FIELDS_PAPER = (167.0, 417.0, 668.0)   # G, los tres campos de las Figs. S22/S24

# ----------------------------------------------------------------------
# Monte Carlo  (S4, ultimo parrafo)
# ----------------------------------------------------------------------
DELTA_S = 0.10 * a             # nm, paso maximo de traslacion inicial
DELTA_A = 0.15                 # rad, angulo maximo de rotacion del cubo
DELTA_M = 0.30                 # rad, angulo maximo de rotacion local del momento
N_MAG_STEPS = 10               # pasos magneticos por ciclo mecanico
ADAPT_FRAC = 0.05              # +-5 % de ajuste de delta
TARGET_ACC = 0.50              # tasa de aceptacion objetivo
R_CUT_VDW = 2.5 * a            # nm, corte de la interaccion vdW (1/r^6 -> 0.5 %)
R_CUT_REP = 1.70 * a           # nm, corte de la repulsion esterica (< 0.005 kcal/mol)

# Escala vdW "nominal" del paper: el potencial de la Fig. S28E reproduce la FORMA
# de la interaccion, pero las energias E_vdW por nanocubo de la Fig. S22
# (-34.70, -38.96, -40.38 kcal/mol para belts 3_100, 6_100, 9_100) son 8.3 veces
# mayores y con el mismo cociente entre anchos.  LAMBDA_S22 recupera esa escala.
LAMBDA_S22 = 8.33
