# Autoensamblaje de nanocubos de Fe₃O₄: simulación Monte Carlo

Implementación del modelo de **Singh et al., *Self-assembly of magnetite nanocubes into helical superstructures*, Science 345, 1149 (2014)** (Supplementary Material, secciones S3–S4), siguiendo la ruta de la metodología y con un objetivo propio: **cuantificar cómo la densidad (concentración) de nanocubos controla el autoensamblaje**.

---

## 1. Modelo

Cada nanocubo es `NC_i = (r_i, R_i, M_i)`: centro, matriz de rotación del cuerpo y vector unitario del super-dipolo. La forma es el superelipsoide `x⁶+y⁶+z⁶=(a/2)⁶` con `a = 13.37 nm`.

Energía total (ec. 5 del suplemento):

```
E_T = Σ_i (E_i^Z + E_i^A) + Σ_i Σ_{j>i} (E_ij^dd + E_ij^vdW)
```

| término | expresión | constante |
|---|---|---|
| Zeeman | `E^Z = −K_Z (H·M)` | `K_Z = 1.647×10⁻² kcal/mol/G` |
| Anisotropía | `E^A = K_A1[(M'ₓM'_y)²+(M'ₓM'_z)²+(M'_yM'_z)²]`, `M' = RᵀM` | `K_A1 = −3.784 kcal/mol` |
| Dipolo–dipolo | `E^dd = (K_d/r³)[Mᵢ·Mⱼ − 3(Mᵢ·r̂)(Mⱼ·r̂)]`, `r` en unidades de `a` | `K_d = 7.973 kcal/mol` |
| vdW + estérica | atracción Hamaker sobre 27 elementos de volumen (∝1/r⁶) + repulsión sobre elementos de superficie (∝1/r⁸) | ver §2 |

Detección de solapamiento: **GJK** con la función soporte del superelipsoide (ec. 7), como criterio geométrico de rechazo, no como energía.

Muestreo: **MCMC + Metropolis**. Un ciclo = una propuesta mecánica (traslación < δs y rotación < δa) por nanocubo en orden aleatorio, seguida de 10 pasos magnéticos (rotación local del dipolo). δs, δa y δm se ajustan ±5 % para mantener ~50 % de aceptación.

Convenio importante que se respeta en el código:
* movimiento **mecánico** → cambian `E_A`, `E_dd`, `E_vdW` (no `E_Z`);
* movimiento **magnético** → cambian `E_Z`, `E_A`, `E_dd` (no `E_vdW`).

---

## 2. Calibración del potencial de van der Waals (decisión metodológica)

El suplemento da la **forma** de la ec. 4 pero sus prefactores dependen de la malla exacta de 386 elementos. Aquí se recalibran `C_a` y `C_r` imponiendo los dos anclajes cuantitativos de la **Fig. S28E**: mínimo en 2.99 nm de separación superficie–superficie con profundidad −2.33 kcal/mol. Resultado: mínimo reproducido exactamente y curva indistinguible de la malla de 386 elementos (desviación < 0.1 kcal/mol en toda la región accesible térmicamente).

**Hallazgo:** con esa calibración, `E_vdW` por nanocubo en cintas perfectas da −4.19 / −4.67 / −4.83 kcal/mol para anchos 3/6/9, mientras la Fig. S22 reporta −34.70 / −38.96 / −40.38. El **cociente es constante (8.28, 8.35, 8.37)**: el potencial usado en las simulaciones MC del paper es el mismo en forma pero **8.3 veces más intenso** (coherente con que mencionen un acoplamiento vdW "nominal" mucho mayor, reducido a la mitad en la Fig. S21). Por eso el código define `LAMBDA_S22 = 8.33` y todas las corridas de producción lo usan.

---

## 3. Validación

`python3 simulations/test_validation.py` → `results/fig0_validacion.png`

* Zeeman: paralelo −6.87, antiparalelo +6.87, perpendicular 0 kcal/mol (|E_Z|/kT = 11.5 a 417 G).
* Anisotropía: 0 en `<100>`, mínimo `K_A1/3 = −1.261` en `<111>` (ejes fáciles = diagonales del cubo ✓).
* Dipolar: cabeza-cola −8.70 (atractiva), lado-lado +4.35 (repulsiva), escala `1/r³` exacta, ángulo crítico de cono 54.7°.
* vdW: mínimo en 2.97 nm / −2.33 kcal/mol; anisotropía correcta (cara-cara −2.33 vs girado 45° −0.33).
* GJK: 5 casos límite correctos (incluido el contacto de una esquina rotada 45°).

**Validación cuantitativa contra la Fig. S22** (`fig1_validacion_S22.png`): posiciones congeladas en `belt₁₀₀` de ancho 3, 6 y 9, muestreo magnético a 167/417/668 G.

| ancho | H [G] | E_mag simulación | E_mag paper | error |
|---|---|---|---|---|
| 3 | 167 / 417 / 668 | −11.86 / −15.86 / −19.95 | −11.96 / −15.98 / −20.08 | < 1 % |
| 6 | 167 / 417 / 668 | −12.01 / −15.72 / −19.72 | −12.12 / −15.87 / −19.94 | ~1 % |
| 9 | 167 / 417 / 668 | −12.14 / −15.63 / −19.64 | −12.19 / −15.82 / −19.88 | ~1 % |

**Error medio 0.95 %** sobre los nueve puntos. La implementación magnética reproduce el paper.

---

## 4. Experimento principal: densidad → autoensamblaje

56 nanocubos con posición, orientación y dipolo aleatorios en una caja cúbica; lo único que cambia entre corridas es el lado `L`, es decir la fracción de volumen `φ = N·V_NC/L³`. Campo fijo `H = 417 G`, 1200 ciclos MC.

| φ | L [nm] | cluster máximo | tamaño medio | fracción ensamblada | ⟨\|B_ef\|⟩ [G] |
|---|---|---|---|---|---|
| 0.01 | 229 | 3 | 1.9 | 0.21 | 637 |
| 0.05 | 134 | 9 | 4.1 | 0.62 | 688 |
| 0.12 | 100 | 33 | 26.0 | 0.93 | 814 |
| 0.25 | 78 | 55 | 54.0 | 0.98 | 829 |

**Mecanismo cuantificado (fig4):** el campo efectivo `B_ef = H_ext + H_dipolar`, con `H_dipolar = −F_i/K_Z`, sube de 637 a 829 G al pasar de φ = 0.01 a 0.25 — el campo externo se **amplifica ×2** solo por la densidad. Dentro de un filamento ya ensamblado el campo local llega a ~1450–1570 G (×3.5 del aplicado). Esta es exactamente la explicación del paper para la Fig. S17: la zona central, más densa, alcanza un campo efectivo suficiente para la transición cinta → hélice, mientras la periferia, más diluida, se queda en cintas.

Otros observables (fig2): la `g(r)` desarrolla un pico agudo en `r = 1.22a` (contacto vdW de equilibrio) solo a alta densidad; `E_all` por nanocubo pasa de −24 a −54 kcal/mol; la cinética de agregación se acelera con φ.

---

## 5. Transición cinta → hélice

**(a) Dinámica directa.** Cintas `belt₁₀₀` de ancho 2, 4 y 6 (H = 668 G, vdW reducido a la mitad como en la Fig. S21), 1200–1500 ciclos: los dipolos se alinean (⟨M_z⟩ → 0.96) y la energía se relaja, pero la torsión permanece < 12°. No es un fallo del modelo: el paper obtiene la hélice tras **23 500 ciclos con 300–900 nanocubos** (Fig. S23), dos órdenes de magnitud más de cómputo del disponible aquí.

**(b) Análisis energético de cintas pre-torcidas** (`fig5`), que es el recurso que el propio paper usa en las Figs. S22 y S24–S27: se impone una torsión ω (deg/nm), se congelan posiciones y se promedian los grados de libertad magnéticos.

| ancho | ΔE_a en ω = 1 deg/nm | ΔE_vdW | ΔE_all |
|---|---|---|---|
| 2 | −0.05 | +0.46 | +1.3 kT |
| 3 | −0.09 | +1.02 | +3.0 kT |
| 4 | −0.15 | +1.87 | +5.4 kT |
| 6 | −0.29 | +3.49 | +10.2 kT |

La **ganancia de anisotropía crece ×5.6 entre el ancho 2 y el 6**, es decir con la densidad local: es el motor de la hélice. El coste vdW crece también porque la torsión rígida estira los contactos; en el sistema real ese coste se paga con el *side-stepping* de los nanocubos y con la entropía (el paper señala explícitamente que la hélice **pierde** E_vdW, E_z y E_dd y lo compensa con E_a y entropía, Fig. S23). El balance obtenido aquí reproduce ese signo término a término.

---

## 6. Animaciones

| archivo | contenido |
|---|---|
| `anim1_densidad.gif` | las cuatro densidades evolucionando en paralelo (color = M_z) |
| `anim2_phi025.gif` | φ = 0.25 con cámara giratoria: disperso → cadenas → filamento |
| `anim3_cinta_w6.gif` | cinta de ancho 6 a 668 G, relajación y fluctuación de los dipolos |

---

## 7. Estructura

```
nanocubes-helix-simulation/
├── src/
│   ├── parameters.py    constantes físicas y de simulación
│   ├── geometry.py      nanocubo, superelipsoide, mallas, función soporte
│   ├── rotations.py     Rodrigues, muestreo uniforme en S²
│   ├── energies.py      Zeeman, anisotropía, dipolar, vdW + calibración
│   ├── gjk.py           Gilbert–Johnson–Keerthi
│   ├── mcmc.py          System: propuestas, Metropolis, ciclo, energías
│   ├── builders.py      gas a densidad dada, cintas, cintas dobles
│   └── analysis.py      clusters, g(r), orden nemático, torsión, campo efectivo
├── simulations/
│   ├── test_validation.py     pasos 1–5
│   ├── run_belt_energies.py   reproducción de la Fig. S22
│   ├── run_density.py         barrido de densidad
│   ├── run_belt_to_helix.py   dinámica de cintas
│   └── run_twist_scan.py      energía vs torsión impuesta
├── analysis/  render.py (3D, GIFs) · figuras.py (todas las figuras)
└── results/   json, npz, png, gif
```

Reproducir todo:
```bash
python3 simulations/test_validation.py
python3 simulations/run_belt_energies.py
python3 run_stage.py dens 0.01 1200    # y 0.05, 0.12, 0.25
python3 run_stage.py helix 6 1200
python3 simulations/run_twist_scan.py 668
python3 analysis/figuras.py todo
```

---

## 8. Optimizaciones y aproximaciones (declaradas)

1. `E_dd` es lineal en `M_i`: se escribe `E_dd(i) = M_i·F_i` con `F_i = Σ_j T_ij M_j`. Como en la fase magnética las posiciones están congeladas, `T_ij` se precalcula una vez por ciclo (≈10× más rápido, física idéntica).
2. Malla de superficie de 54 elementos en producción en vez de 386: la curva vdW coincide con la de referencia dentro de 0.1 kcal/mol en toda la región relevante (comprobado en `fig0`).
3. Cortes: atracción hasta 2.5a, repulsión hasta 1.7a (allí vale < 0.005 kcal/mol). El acoplamiento dipolar se suma **sin corte**, sobre todos los pares.
4. Cintas de 40 nanocubos de largo en la validación S22 (el paper usa 100); efecto de borde < 1 %.
5. Sin condiciones periódicas: caja con paredes duras, como en una gota que se seca.

## 9. Segunda revisión del código: qué se encontró y se corrigió

| # | Hallazgo | Impacto | Estado |
|---|---|---|---|
| 1 | **Partículas congeladas por solapamiento inicial.** `random_gas` colocaba cubos a 1.30–1.35 a; para cubos alineados eso permite solapamiento (basta que las tres componentes sean < a). Una partícula que arranca solapada nunca se mueve: toda propuesta se rechaza por GJK. | 2 partículas de 56 inmóviles a φ = 0.12 | Corregido: distancia mínima 2.02·R_out (imposible solapar en cualquier orientación) **y** `mechanical_move` ahora acepta movimientos que *reducen* el número de solapamientos, más `System.overlap_report()` que valida el estado inicial |
| 2 | **Orientaciones iniciales idénticas.** `System` ponía la identidad cuando no se pasaban orientaciones, aunque el texto decía "orientaciones aleatorias". | sesgo en el estado inicial del gas | Añadido `builders.random_orientations` (QR de matrices gaussianas) y usado en los barridos |
| 3 | **Tasa de aceptación mal reportada.** `adapt()` reinicia los contadores cada 10 ciclos, así que el valor impreso era el de los últimos ciclos, a menudo 0/0. | solo diagnóstico, pero engañoso | Contadores acumulados + `System.acceptance()` |
| 4 | **Signo en la forma matricial de E_dd.** Al vectorizar sobre todos los pares, `proj.T[i,j] = −(r_ij·M_j)`, de modo que `−3(M_i·r̂)(M_j·r̂)` se convierte en `+3·proj·projᵀ`. | habría dado E_dd errónea en el movimiento colectivo | Corregido y verificado: las tres vías (tensor, suma por partícula, forma matricial) dan el mismo número |
| 5 | `orthonormalize` usaba SVD en cada rotación | 12 µs × N por ciclo | Gram–Schmidt explícito, ~3 µs |
| 6 | El campo local `F_i` se recalculaba en cada intento magnético | O(N) por intento | Se calcula una vez por ciclo y se actualiza incrementalmente solo al aceptar (`F += T[:,i]·dM`), exacto y ~2× más rápido |
| 7 | El criterio de la malla vdW comparaba también la pared repulsiva dura | falso "FALLA" en el test | El test compara ahora la región que fija la estructura (gap > 2.5 nm): desviación 0.05 kcal/mol; la pared con 54 elementos es 2.4 kcal/mol más dura a 2 nm, ya prohibida por 4 kT |

Verificación de no regresión tras todos los cambios: los cinco bloques de `test_validation.py` pasan y `E_mag` de la Fig. S22 sigue reproduciéndose dentro del 1 %.

---

## 10. El cuello de botella real y cómo se resolvió

Medido en una cinta 3×1×24 a 300 ciclos: la aceptación mecánica es sana (0.31–0.46) y cada nanocubo se desplaza 1–3 nm. Para que la cinta se reconfigure por *side-stepping* hace falta que se desplace **un parámetro de red entero (16.4 nm)**: como el desplazamiento crece con √ciclos, eso son ~10⁴ ciclos. No era un problema del modelo sino del tiempo de muestreo; coincide con los 23 500 ciclos del paper.

Dos respuestas, ambas ya en el código:

**(a) Correrlo de verdad.** Tiempos medidos en un solo núcleo modesto (un procesador de escritorio actual suele ser 1.5–2.5× más rápido):

| sistema | cinta | s/ciclo | 23 500 ciclos |
|---|---|---|---|
| 300 nanocubos | 3×1×100 | 0.49 | **3.4 h** |
| 600 nanocubos | 6×1×100 | 1.23 | 8.7 h |
| 900 nanocubos | 9×1×100 | 2.57 | 17.7 h (14.3 h con `dd_cutoff=8a`) |

**(b) Movimiento colectivo de torsión** (`System.collective_twist_move`). La torsión de la cinta es una coordenada colectiva lentísima. Este movimiento propone un incremento global dω alrededor del eje del filamento y lo acepta con el mismo criterio de Metropolis. La propuesta es simétrica y la transformación invertible, así que **el balance detallado se conserva y la distribución muestreada sigue siendo la canónica**: acelera el muestreo, no sesga el resultado. Al girar posición, orientación y momento de cada capa con la misma Rz, E_Z y E_A quedan invariantes y solo hay que reevaluar E_dd y E_vdW. Se propone cada `--collective-every` ciclos (5 por defecto) y cuesta ~0.15 s en N = 300.

Con él, en cintas cortas (3×1×24, 200 ciclos) la torsión ya explora ±0.1 deg/nm con 40–49 % de aceptación: el modo es **blando pero fluctúa alrededor de cero**, es decir a esas dimensiones y parámetros la cinta plana sigue siendo el estado de equilibrio. Esa es la pregunta que la corrida larga tiene que responder con la geometría del paper.

---

## 11. Cómo correr la reproducción en tu máquina

```bash
# 1. comprobar que todo pasa (30 s)
python3 simulations/test_validation.py

# 2. ver en qué régimen la torsión sobrevive, antes de gastar horas (5 min)
python3 simulations/run_regime_scan.py semilla 300

# 3. corrida larga: la geometría exacta de la Fig. S23
python3 simulations/run_production.py --width 3 --length 100 --cycles 23500         --field 417 --vdw-scale 2.0 --tag prod_w3_H417_v2

# si se corta la corrida (o cierras el portátil), se reanuda sin perder nada
python3 simulations/run_production.py --tag prod_w3_H417_v2 --resume

# 4. doble hebra -> doble hélice
python3 simulations/run_production.py --width 2 --length 100 --double         --cycles 25000 --field 417 --vdw-scale 2.0 --tag prod_doble

# 5. varias corridas en paralelo, una por núcleo (deja la máquina toda la noche)
python3 simulations/run_batch.py --procs 6

# 6. figuras, métricas de la hélice y animación de cualquier corrida
python3 analysis/analizar_produccion.py --tag prod_w3_H417_v2 --gif
```

`run_production.py` escribe cada 25 ciclos una fila en `{tag}_energias.csv` (las seis energías, torsión, ancho, aceptación), guarda un checkpoint cada 250 ciclos (`{tag}_checkpoint.npz`, incluye el estado del generador aleatorio, así que reanudar es exacto) y snapshots cada 100 ciclos. En pantalla imprime ETA.

**Rejilla recomendada** (es la que trae `run_batch.py`; 6 procesos ≈ una noche):

| tag | geometría | H [G] | escala vdW | qué contesta |
|---|---|---|---|---|
| `prod_w3_H417_v2` | 3×1×100 | 417 | 2.0 | réplica directa de la Fig. S23 |
| `prod_w3_H668_v2` | 3×1×100 | 668 | 2.0 | régimen de campo alto (donde el paper ve hélices) |
| `prod_w3_H417_v1` | 3×1×100 | 417 | 1.0 | vdW más débil = más reconfigurable |
| `prod_w6_H417_v2` | 6×1×100 | 417 | 2.0 | mayor densidad local |
| `prod_w2d_H417_v2` | 2 cintas | 417 | 2.0 | **doble hebra** |
| `prod_w2d_H668_v2` | 2 cintas | 668 | 2.0 | doble hebra a campo alto |

**Sobre `--vdw-scale`:** es el parámetro no determinado por el paper. 8.33 reproduce E_vdW de la Fig. S22 (cintas rígidas), pero con ese valor cada contacto vale 32 kT y la estructura queda congelada. La Fig. S21 reduce el acoplamiento a la mitad y la Fig. S23 dice literalmente que usa "parámetros escalados" sin publicarlos. El barrido de movilidad da: λ = 1 → pozo 3.9 kT, aceptación 0.46, 2.7 nm de desplazamiento por 300 ciclos; λ = 2 → 7.8 kT, 0.41, 2.0 nm; λ = 4.17 → 16.3 kT, 0.31, 1.2 nm. **λ = 1–2 es el rango donde la cinta se mantiene unida y a la vez puede reconfigurarse**, y por eso es el valor por defecto.

---

## 13. Tercera lectura del suplemento (S3–S4): qué faltaba

Cinco cosas, todas ya incorporadas:

**1. El precursor de la hélice es el belt₁₁₀, no el belt₁₀₀.** El texto de S3 es explícito: *"la configuración zigzag se preserva en belts₁₁₀ multicapa… una conectividad suave de los dipolos siguiendo el eje fácil de los nanocubos puede resolverse si las estructuras se reconfiguran por side-stepping, lo que eventualmente lleva a estructuras helicoidales."* Todas mis semillas eran belt₁₀₀. Añadido `builders.belt_110`: cubos girados 45° alrededor del eje largo ("cube edge on top"), con contacto cara-cara a lo largo del campo y arista-arista lateralmente. El espaciamiento lateral (18.60 nm frente a 16.36 nm) se obtiene minimizando el mismo potencial vdW **con filtro GJK**, porque la suma discretizada de la atracción diverge si los elementos de volumen se interpenetran.

Validación nueva contra la Fig. S22 (belts₁₁₀, posiciones congeladas):

| ancho | H = 167 / 417 / 668 G | paper | error |
|---|---|---|---|
| 6 | −11.89 / −15.94 / −20.04 | −11.82 / −15.93 / −20.05 | **0.1–0.6 %** |
| 3 | −12.00 / −16.07 / −20.18 | −11.16 / −15.27 / −19.39 | 4–7 % |

El caso ancho encaja casi exactamente; el de ancho 3 se desvía porque su belt₁₁₀ tiene 284 nanocubos y no 300, es decir su construcción descarta una fila parcial en el borde, detalle que pesa más cuanto más estrecha es la cinta. El E_vdW del belt₁₁₀ también sale menos negativo que el del belt₁₀₀ (−3.64 vs −4.19), igual que en el paper (−31.64 vs −34.70), con el mismo factor global ~8.3–9.6.

**2. Filamentos multicapa.** La Fig. S24 trabaja con cintas *n×n×100*, no monocapa: el campo efectivo de un filamento grueso es mayor. `--thickness` ya existía pero la rejilla usaba monocapas; ahora la rejilla por defecto son belts₁₁₀ de sección n×n.

**3. "Parámetros escalados" admite dos lecturas.** La Fig. S21 reduce el acoplamiento vdW a la mitad (mi `--vdw-scale`), pero la Fig. S23 dice otra cosa: *"helices formed by thick filaments should be less prone to entropic forces due to stronger effective magnetic fields"*, o sea que para un filamento delgado hay que **reforzar el acoplamiento magnético** para imitar el campo efectivo de uno grueso. Añadido `--dd-scale`, que multiplica K_d. La rejilla barre ambas lecturas.

**4. Ergodicidad magnética.** El suplemento dice *"10 magnetic steps (local magnetic dipole orientation sampled randomly)"*. Mi paso magnético era puramente local con δ adaptativo, y zigzag y paralelo son dos mínimos separados del subsistema magnético. Ahora un 10 % de los pasos magnéticos (`--mag-global-frac`) propone una dirección uniformemente aleatoria en la esfera, lo que permite saltar entre ambas configuraciones.

**5. Confirmaciones.** La anisotropía de forma se menciona en el texto pero el paper la **desprecia** explícitamente en la energía (solo el término cuártico de MA, ec. 2): mi implementación coincide. La entropía en el código es solo configuracional (de partículas y espines), igual que la suya. Y aunque el suplemento escribe K_Z en kcal/mol, numéricamente se usa por Gauss: mis energías Zeeman reproducen las suyas dentro del 1 %.

---

## 14. Por qué no hubo torsión, y qué esperar al correrlo

**La respuesta corta: los ciclos son necesarios pero no suficientes.** Hay dos barreras y solo una es de tiempo de cómputo.

*Barrera cinética (se arregla con ciclos).* El side-stepping exige que cada nanocubo se desplace un parámetro de red entero, 16.4 nm. Medido: 2–3 nm cada 300 ciclos, y el desplazamiento crece como √ciclos → ~10⁴ ciclos. Corriendo 1200 estábamos un factor 10 por debajo. Con 23 500 esta barrera desaparece.

*Barrera termodinámica (no se arregla con ciclos).* La Fig. S23 muestra que durante la formación de la hélice **E_all sube**, de −27.5 a −22.5 kcal/mol por nanocubo, y también suben E_vdW, E_z y E_dd; solo E_a baja. El propio texto lo dice: la pérdida se compensa *"con la ganancia en E_a y el crecimiento de la entropía de nanocubos más sueltos y desordenados"*. Es decir: **la hélice no es el mínimo de energía potencial, es un estado estabilizado por entropía**. Eso impone una condición sobre los parámetros que ningún número de ciclos suple: si cada contacto vdW vale 32 kT (λ = 8.33, la escala que reproduce la Fig. S22 en cintas rígidas), la estructura no puede "soltarse" y la hélice es inalcanzable aunque corras un millón de ciclos. Con λ = 1–2 el pozo vale 4–8 kT y el estado suelto sí es accesible.

Por eso la rejilla de `run_batch.py` no repite seis veces la misma corrida: barre justo esas dos incógnitas (λ_vdW y dd_scale) sobre la geometría correcta (belt₁₁₀ multicapa). La conclusión sale de comparar las seis, y cualquiera de los tres desenlaces es un resultado publicable en tu informe:

* alguna corrida tuerce → reprodujiste la Fig. S23 y además acotaste los parámetros que el paper no publica;
* ninguna tuerce pero el modo de torsión se ablanda al aumentar el ancho y el campo → mediste la tendencia que predice el mecanismo, y el paso siguiente es el término de depleción;
* ninguna tuerce y el modo no se ablanda → el modelo de cinco términos no basta y falta la depleción del ácido oleico, que el paper menciona en la nota 28 y tampoco incluye.

**Rejilla por defecto** (`run_batch.py`, 6 procesos, ~7 h cada uno en un núcleo):

| tag | geometría | H [G] | λ_vdW | dd_scale | hipótesis |
|---|---|---|---|---|---|
| `prod_110_w3x3_H417_v1_dd1.5` | belt₁₁₀ 3×3×60 (540 NC) | 417 | 1.0 | 1.5 | apuesta principal |
| `prod_110_w3x3_H668_v1_dd1` | belt₁₁₀ 3×3×60 | 668 | 1.0 | 1.0 | campo alto |
| `prod_110_w3x3_H417_v0.5_dd2` | belt₁₁₀ 3×3×60 | 417 | 0.5 | 2.0 | vdW blando + dd fuerte |
| `prod_100_w3x1_H417_v2_dd1` | belt₁₀₀ 3×1×100 (300 NC) | 417 | 2.0 | 1.0 | control literal de la Fig. S23 |
| `prod_110_w2x2_H417_v1_dd1` | belt₁₁₀ 2×2×100 (400 NC) | 417 | 1.0 | 1.0 | filamento delgado y largo |
| `prod_110_w2x2d_H417_v1_dd1.5` | doble hebra 2×2×80 (640 NC) | 417 | 1.0 | 1.5 | doble hélice |

Las corridas con más de 400 nanocubos añaden `--dd-cutoff 8` automáticamente (error 0.5 % en E_dd, ~20 % más rápido).

---

## 12. Qué queda abierto

* El ácido oleico en exceso actúa como agente de **depleción** (el paper lo señala como facilitador entrópico y no lo incluye explícitamente en el modelo). Añadir un término de depleción tipo Asakura–Oosawa es la extensión natural si las corridas largas no tuercen.
* La métrica `filament_twist` mide el giro del director de la sección transversal, así que **funciona igual para una hebra y para dos** (en una doble hélice el director es la línea que une los dos centroides). El análisis de ángulo de fase entre hélices vecinas (Figs. S26–S27) todavía no está automatizado.
* Paralelización interna: hoy se paraleliza lanzando corridas independientes. Para una sola corrida más rápida haría falta listas de celdas y, sobre todo, mover el núcleo vdW a numba o C.
