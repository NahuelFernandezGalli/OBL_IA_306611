# Registro de decisiones

Este documento explica cada decisión técnica del obligatorio: qué se decidió, qué alternativas había, por qué se eligió esa y qué evidencia la respalda. Se actualiza junto con el código y sirve de base para el informe.

**Cómo se usa:**

- Cada decisión tiene un identificador (`D-XX`) que no se reutiliza.
- Una decisión que se cambia no se borra: se marca como **Reemplazada por D-YY** y se crea la nueva, explicando qué motivó el cambio.
- Cuando hay resultados que respaldan una decisión, se indica la ruta (`results/…`, `figures/…`).

## Índice

| ID | Ámbito | Decisión | Estado |
|---|---|---|---|
| [D-01](#d-01-python-311-y-un-entorno-poetry-por-proyecto) | General | Python 3.11 y un entorno Poetry por proyecto | Vigente |
| [D-02](#d-02-módulos-propios-junto-a-los-de-la-cátedra) | General | Módulos propios junto a los de la cátedra | Vigente |
| [D-03](#d-03-formato-del-registro-de-experimentos) | General | Formato del registro de experimentos | Vigente |
| [D-04](#d-04-entrenar-y-evaluar-acrobot-sin-render) | Acrobot | Entrenar y evaluar sin render | Vigente |
| [D-05](#d-05-línea-base-agente-al-azar-sobre-50-estados-iniciales) | Acrobot | Línea base: agente al azar sobre 50 estados iniciales | Vigente |
| [D-06](#d-06-representar-el-estado-con-ángulos-en-vez-de-cossin) | Acrobot | Representar el estado con ángulos en vez de cos/sin | Vigente (confirmada por el experimento C) |
| [D-07](#d-07-bins-de-igual-ancho-y-rangos-iniciales) | Acrobot | Bins de igual ancho y rangos iniciales | Reemplazada por D-12 y D-13 |
| [D-08](#d-08-detalles-de-implementación-de-q-learning) | Acrobot | Detalles de implementación de Q-Learning | Vigente |
| [D-09](#d-09-métrica-de-evaluación-política-greedy-sobre-estados-iniciales-fijos) | Acrobot | Métrica: política greedy sobre estados iniciales fijos | Vigente |
| [D-10](#d-10-formato-del-modelo-guardado) | Acrobot | Formato del modelo guardado | Vigente |
| [D-11](#d-11-metodología-de-los-barridos-5-semillas-y-presupuesto-fijo-de-episodios) | Acrobot | Barridos: 5 semillas y presupuesto fijo de episodios | Vigente |
| [D-12](#d-12-rango-de-las-velocidades-la-mitad-de-los-límites-del-entorno) | Acrobot | Rango de velocidades: la mitad de los límites del entorno | Vigente |
| [D-13](#d-13-cantidad-de-bins-8-por-ángulo-y-12-por-velocidad) | Acrobot | Cantidad de bins: 8 por ángulo y 12 por velocidad | Vigente |

---

## General

### D-01: Python 3.11 y un entorno Poetry por proyecto

- **Fecha:** 2026-11-03 · **Estado:** Vigente
- **Contexto:** la letra pide Poetry con entornos separados para cada ejercicio. Los `pyproject.toml` de la cátedra exigen `python >=3.9,<3.13`.
- **Decisión:** usar Python 3.11 y crear el `.venv` dentro de cada carpeta (`poetry.toml` → `in-project = true`).
- **Alternativas:** Python 3.12 (también es compatible, pero no aportaba nada extra) o el 3.14 del sistema (lo descartan los `pyproject.toml`).
- **Justificación:** 3.11 entra en el rango pedido y tiene buen soporte en `gymnasium` y `numpy<2`. Tener el `.venv` dentro de la carpeta hace que el kernel de cada notebook sea fácil de encontrar y que los entornos no se mezclen.

### D-02: Módulos propios junto a los de la cátedra

- **Fecha:** 2026-11-03 · **Estado:** Vigente
- **Contexto:** el código de la cátedra importa sus módulos por nombre plano (`from board import Board`, `from agent import InputAgent`).
- **Decisión:** los módulos nuevos van en la misma carpeta que los de la cátedra, sin subpaquetes. Las salidas se separan en `models/`, `results/` y `figures/`.
- **Alternativas:** un paquete `src/`, que obligaría a tocar los imports de la cátedra o el `sys.path` de los notebooks.
- **Justificación:** no se modifica el código entregado por la cátedra y los notebooks siguen funcionando tal cual.

### D-03: Formato del registro de experimentos

- **Fecha:** 2026-11-03 · **Estado:** Vigente
- **Contexto:** la letra pide un registro completo de resultados, con parámetros y tiempos de ejecución, y gráficos en el informe.
- **Decisión:** cada corrida genera `results/<run_id>.json` (configuración completa, semilla, métricas finales, tiempo). En Acrobot genera además `results/<run_id>_episodes.csv` (una fila por episodio). `results/runs.csv` funciona como índice, con una fila por corrida (o por partida, en Connect Four). El detalle está en el [README](../README.md#registro-de-experimentos).
- **Justificación:** el JSON hace que cada corrida se pueda reproducir. El CSV por episodio permite graficar curvas de aprendizaje. El índice permite comparar configuraciones sin abrir cada archivo.

## ACROBAT (Acrobot)

### D-04: Entrenar y evaluar Acrobot sin render

- **Fecha:** 2026-11-03 · **Estado:** Vigente
- **Contexto:** con `render_mode='human'`, un episodio de 700 pasos tarda unos 4.5 s. Los barridos de hiperparámetros necesitan miles de episodios.
- **Decisión:** entrenar y evaluar con `render_mode=None` (o `'rgb_array'`) y reservar `'human'` para mostrar el modelo final. Se mantiene `max_episode_steps=700`, como sugiere la cátedra.
- **Justificación:** el render no aporta nada al aprendizaje y multiplica el tiempo de cada corrida.
- **Evidencia (2026-11-19):** sin render, un episodio de 700 pasos tarda **0.031 s** (unos 45 µs por paso), contra unos 4.5 s con render: unas 140 veces más rápido. Con Q-Learning, 3000 episodios de entrenamiento tardan unos 93 s. Ver `Acrobot/results/random_20261119-182717_e5db.json` y `Acrobot/results/qlearning_base_20261119-183046_14cb.json`.

### D-05: Línea base: agente al azar sobre 50 estados iniciales

- **Fecha:** 2026-11-19 · **Estado:** Vigente
- **Contexto:** hace falta una referencia contra la cual comparar al agente, y datos sobre qué valores toman en la práctica las velocidades angulares.
- **Decisión:** `baseline.py` corre 50 episodios con acciones uniformes al azar, cada uno con la semilla `seed + episodio`, y registra pasos, éxito y estadísticas de ω1 y ω2.
- **Resultado:** 0 % de éxito. Los 50 episodios llegan al corte de 700 pasos (recompensa −700). Las velocidades se quedan muy por debajo de los límites del entorno:

  | | Límite del entorno | Mín. / Máx. observado | Percentiles 1 y 99 |
  |---|---|---|---|
  | ω1 | ±4π ≈ ±12.57 | −4.88 / 4.95 | −3.36 / 3.33 |
  | ω2 | ±9π ≈ ±28.27 | −8.61 / 9.25 | −5.88 / 5.52 |

- **Uso:** −700 es el piso de los gráficos. Los rangos medidos son el insumo de D-07.
- **Evidencia:** `Acrobot/results/random_20261119-182717_e5db.json`.

### D-06: Representar el estado con ángulos en vez de cos/sin

- **Fecha:** 2026-11-19 · **Estado:** Vigente (confirmada el 2026-11-20 por el experimento C)
- **Contexto:** la observación tiene 6 componentes, pero cos θ y sin θ describen un mismo ángulo. En una tabla, el número de celdas es el producto de los bins de cada dimensión, así que crece exponencialmente con la cantidad de dimensiones. AIMA (p. 803) ubica el límite práctico de lo tabular en unos 10⁶ estados.
- **Decisión:** la representación por defecto (`"angles"`) recupera θ1 = atan2(sin θ1, cos θ1) y θ2 igual, y discretiza 4 variables: θ1, θ2, ω1, ω2. La representación `"raw"` (las 6 componentes) queda implementada para compararla.
- **Alternativas:** discretizar las 6 componentes. Con 10 bins da 10⁶ celdas × 3 acciones, contra 10⁴ × 3 con ángulos.
- **Justificación:** se eliminan 2 dimensiones redundantes sin perder información. Con los mismos bins la tabla es 100 veces más chica, y cada celda se visita más seguido.
- **Limitación conocida:** θ = π y θ = −π son el mismo ángulo, pero caen en el primer y el último bin. El estado físico queda partido en dos celdas: no mezcla estados distintos, pero cada mitad aprende por separado.
- **Experimento C (2026-11-20):** velocidades en la mitad de los límites (D-12), 5 semillas, 5000 episodios.

  | Variante | Celdas | Pasos finales | Éxito | Pasos en la curva | Celdas visitadas |
  |---|---|---|---|---|---|
  | **Ángulos, 10 bins** | **10⁴** | **166.5 ± 12.4** | **100 %** | **236.4 ± 10.6** | 38 % |
  | Crudo, 4 bins | 4096 | 330.5 ± 173.5 | 82 % ± 34 % | 316.7 ± 21.1 | 34 % |
  | Crudo, 6 bins | 46 656 | 250.2 ± 27.1 | 100 % | 292.2 ± 13.3 | 12 % |
  | Crudo, 10 bins | 10⁶ | 284.1 ± 40.8 | 96 % ± 4 % | 370.7 ± 6.9 | 2 % |

  Con ángulos se gana en todas las métricas, con cualquier tamaño de tabla:
  - Con 10 bins, la representación cruda tiene 100 veces más celdas y en 5000 episodios visita solo el 2 %: casi toda la tabla queda en su valor inicial.
  - Con un tamaño parecido (crudo 4 bins = 4096 celdas, igual que ángulos con 8 bins) también pierde, por dos razones. Primero, los bins sobre cos y sin no son uniformes en el ángulo y varias combinaciones (cos, sin) no se pueden dar, así que se desperdician celdas. Segundo, deja solo 4 bins por velocidad, y D-13 muestra que la resolución de las velocidades es lo que más importa.
- **Evidencia:** `Acrobot/results/summary_representacion.csv`.

### D-07: Bins de igual ancho y rangos iniciales

- **Fecha:** 2026-11-19 · **Estado:** Reemplazada por D-12 (rangos, 2026-11-20) y D-13 (bins, 2026-11-20). Se mantiene el corte en bins de igual ancho.
- **Contexto:** Q-Learning tabular necesita estados discretos. Si la discretización es gruesa, estados que piden acciones distintas caen en la misma celda (*aliasing*) y el agente deja de ver un proceso de Markov. Si es fina, hay muchas celdas, cada una se visita poco y el aprendizaje es lento.
- **Decisión:** `discretizer.py` corta cada dimensión en bins de igual ancho dentro de un rango [low, high], y los valores de afuera van al bin extremo. Para la primera corrida: 10 bins por dimensión, θ ∈ [−π, π] y velocidades en sus límites teóricos (ω1 ∈ ±4π, ω2 ∈ ±9π).
- **Alternativas:** rangos empíricos (percentiles medidos en D-05 o con el agente entrenado), bins no uniformes, o distinta cantidad de bins por dimensión.
- **Justificación:** es el punto de partida más simple y no hace supuestos sobre qué velocidades se alcanzan.
- **Hallazgo (2026-11-19):** con estos rangos el agente visitó solo el **12.8 %** de las 10 000 celdas. Con el agente entrenado, el percentil 99 de |ω1| es 4.9 (de un rango de ±12.57) y el de |ω2| es 10.1 (de ±28.27). Eso deja unos 4 de los 10 bins de cada velocidad haciendo todo el trabajo. Es un indicio fuerte de que conviene achicar los rangos.
- **Motivo del reemplazo:** los experimentos A y B (D-12 y D-13) encontraron rangos y cantidades de bins mejores que los iniciales.
- **Evidencia:** `Acrobot/results/qlearning_base_20261119-183046_14cb.json`.

### D-08: Detalles de implementación de Q-Learning

- **Fecha:** 2026-11-19 · **Estado:** Vigente
- **Contexto:** la letra pide Q-Learning. La regla es la ecuación 22.7 de AIMA: `Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]`.
- **Decisión** (`qlearning.py`):
  - **Q inicial en 0.** Como todas las recompensas son ≤ 0, 0 es una estimación optimista: cada acción probada baja su valor, y el agente tiende a probar las que todavía no probó (el "prior optimista" de AIMA 22.3.1).
  - **Desempate al azar en el argmax.** Si no, ante empates (por ejemplo, en celdas sin visitar) ganaría siempre la acción 0.
  - **ε-greedy con decaimiento exponencial por episodio:** `ε = max(ε_min, ε_inicial · decayᵉᵖⁱˢᵒᵈⁱᵒ)`. Valores iniciales: ε de 1.0 a 0.05 con decay 0.995, es decir, llega al piso cerca del episodio 600.
  - **α constante** (0.1 al inicio).
  - **`terminated` contra `truncated`.** Si se llegó al objetivo, el objetivo de la actualización es solo `r` (el estado terminal vale 0). Si el episodio se cortó a los 700 pasos, s' no es terminal y se usa `r + γ max Q(s')`.
- **Alternativas:**
  - α decreciente (1/n por par estado–acción), que garantiza convergencia en lo tabular, pero aprende muy lento y la discretización ya rompe las hipótesis del teorema.
  - La función de exploración de AIMA (Fig. 22.8), descartada porque la letra pide ε-greedy.
  - Tratar el corte por tiempo como terminal, que le enseñaría al agente que esas celdas "terminan" el problema, algo falso.
- **Justificación:** es la forma estándar del algoritmo. α constante permite que la tabla siga adaptándose, y eso importa porque, con aliasing, el problema que ve el agente no es exactamente un MDP estacionario.

### D-09: Métrica de evaluación: política greedy sobre estados iniciales fijos

- **Fecha:** 2026-11-19 · **Estado:** Vigente
- **Contexto:** la recompensa durante el entrenamiento mezcla lo aprendido con la exploración (ε > 0), y cada episodio arranca de un estado inicial distinto.
- **Decisión** (`train.py`):
  - La métrica principal es la **evaluación greedy** (ε = 0, sin actualizar Q) sobre **100 estados iniciales fijos**, con semillas 1 000 000 a 1 000 099, distintas de las de entrenamiento. Se reporta: **pasos medios** (equivale a la recompensa, porque es −1 por paso), **tasa de éxito** (llegar antes de los 700 pasos), desvío y pasos medios de los episodios exitosos.
  - **Curva de aprendizaje:** cada 250 episodios se evalúa la política greedy sobre 20 de esos estados iniciales.
  - También se registra la media de pasos de los últimos 100 episodios de entrenamiento, el porcentaje de celdas visitadas y los tiempos.
- **Alternativas:** usar solo la recompensa de entrenamiento, que subestima la política por la exploración, o estados iniciales al azar en cada evaluación, que agregan ruido a la comparación entre configuraciones.
- **Justificación:** todas las configuraciones se comparan sobre exactamente los mismos estados iniciales. Como la dinámica es determinista, la diferencia se debe a la política.
- **Observación (2026-11-19):** la curva greedy oscila bastante entre evaluaciones (de 156 a 353 pasos en la corrida base, con ε ya fijo en 0.05). Por eso las comparaciones de hiperparámetros tienen que usar **varias semillas de entrenamiento** y no una sola corrida.
- **Nota:** la evaluación greedy tiene una pequeña componente aleatoria, el desempate en celdas con valores iguales. Al recargar el modelo base se obtuvieron 196.0 pasos en vez de 196.6.

### D-10: Formato del modelo guardado

- **Fecha:** 2026-11-19 · **Estado:** Vigente
- **Decisión:** `models/<run_id>.pkl` es un `pickle` con la tabla Q (`numpy`), la configuración del discretizador, los hiperparámetros y la configuración completa de la corrida. `qlearning.load_model` devuelve el agente (con ε = 0) y el discretizador. El `run_id` coincide con el de `results/`, así cada modelo se puede rastrear hasta su registro.
- **Justificación:** la tabla Q no sirve sin el discretizador que define qué celda es cada fila. Guardarlos juntos evita cargar un modelo con una discretización equivocada.
- **Primer modelo (obligatorio para la entrega):** `Acrobot/models/qlearning_base_20261119-183046_14cb.pkl`, 100 % de éxito y 196.6 pasos medios en la evaluación greedy.

### D-11: Metodología de los barridos: 5 semillas y presupuesto fijo de episodios

- **Fecha:** 2026-11-20 · **Estado:** Vigente
- **Contexto:** una sola corrida no alcanza para comparar configuraciones. La evaluación greedy oscila mucho de un punto de la curva al siguiente (D-09), y el resultado depende de la semilla.
- **Decisión** (`sweep.py`):
  - Cada variante se entrena con **5 semillas** (0 a 4) y se reporta la **media ± desvío** entre semillas.
  - Todas las variantes de un experimento tienen el **mismo presupuesto: 5000 episodios**. Es más que los 3000 de la corrida base, para no castigar a las tablas más grandes, que necesitan más visitas.
  - Se cambia **una cosa por vez** respecto de la configuración por defecto (α = 0.1, γ = 0.99, ε de 1 a 0.05 con decay 0.995).
  - Se comparan dos métricas. **Pasos finales** es la evaluación greedy al terminar, sobre 100 estados fijos. **Pasos en la curva** es la media de las 20 evaluaciones intermedias. La segunda mide la velocidad de aprendizaje y es menos ruidosa, porque promedia 20 instantáneas en vez de una.
  - Las corridas se hacen en paralelo, con 10 procesos (`multiprocessing`). Solo el proceso principal escribe en `results/`, para que dos procesos no escriban `runs.csv` a la vez.
- **Alternativas:** más semillas, que dan más precisión pero multiplican el tiempo, o un presupuesto de tiempo en vez de episodios, que depende de la máquina y de la carga.
- **Justificación:** 5 semillas permiten distinguir diferencias grandes del ruido en un tiempo razonable: unos 10 minutos por experimento de 4 variantes.
- **Dato de tiempos:** con 10 procesos en paralelo, cada corrida de 5000 episodios tarda unos 270 s, contra unos 150 s estimados si corre sola, porque los procesos compiten por la CPU.
- **Resúmenes:** `Acrobot/results/summary_<experimento>.csv`.

### D-12: Rango de las velocidades: la mitad de los límites del entorno

- **Fecha:** 2026-11-20 · **Estado:** Vigente · **Reemplaza** la parte de rangos de D-07.
- **Contexto:** con los límites teóricos solo se visitaba el 13 % de la tabla (D-07). Achicar el rango da más resolución en las velocidades frecuentes. A cambio, las velocidades que superan el rango caen todas en el bin extremo y quedan mezcladas.
- **Experimento A** (10 bins por dimensión, ángulos, 5 semillas, 5000 episodios):

  | Variante | ω1 máx. | ω2 máx. | Ancho de bin ω1 / ω2 | Pasos finales | Éxito | Pasos en la curva | Celdas visitadas |
  |---|---|---|---|---|---|---|---|
  | Teórico | 12.57 | 28.27 | 2.51 / 5.65 | 188.6 ± 53.0 | 100 % | 239.5 ± 19.1 | 14 % |
  | **Mitad** | **6.28** | **14.14** | **1.26 / 2.83** | **166.5 ± 12.4** | **100 %** | **236.4 ± 10.6** | 38 % |
  | Agente entrenado (p99) | 5.0 | 10.0 | 1.00 / 2.00 | 246.6 ± 148.0 | 89 % ± 22 % | 242.2 ± 14.0 | 53 % |
  | Agente al azar (p1–p99) | 3.4 | 5.9 | 0.68 / 1.18 | 265.8 ± 63.6 | 95 % ± 6 % | 323.1 ± 17.2 | 72 % |

- **Decisión:** ω1 ∈ [−2π, 2π] y ω2 ∈ [−4.5π, 4.5π], la mitad de los límites del entorno.
- **Justificación:**
  - Es la variante con mejor resultado final y, sobre todo, con el **menor desvío entre semillas**: 12.4 contra 53.0 de la teórica.
  - En velocidad de aprendizaje (pasos en la curva) empata con la teórica: la diferencia queda dentro del desvío. No se puede afirmar que sea mejor en ese aspecto, sino que es más estable.
  - Los rangos estrechos empeoran claramente. El del agente al azar es el peor en todas las métricas, aunque visita el 72 % de la tabla. El agente entrenado gira mucho más rápido que uno al azar (p99 de |ω2| ≈ 10 contra 5.9), así que todos los estados de alta velocidad, justo los del balanceo final, caen en el bin extremo y se mezclan (*aliasing*). El rango de 5/10 tuvo una semilla que colapsó (44 % de éxito).
  - **Visitar más celdas no implica aprender mejor:** lo que importa es que la resolución esté donde están las decisiones.
- **Evidencia:** `Acrobot/results/summary_rangos.csv` y las corridas `Acrobot/results/rangos_*`.

### D-13: Cantidad de bins: 8 por ángulo y 12 por velocidad

- **Fecha:** 2026-11-20 · **Estado:** Vigente · **Reemplaza** la parte de bins de D-07.
- **Contexto:** es el compromiso de la discretización. Con pocos bins, estados que piden acciones distintas comparten celda (*aliasing*). Con muchos, cada celda se visita poco y, con un presupuesto fijo de episodios, el aprendizaje es más lento. Además la tabla no generaliza entre celdas vecinas.
- **Experimento B** (ángulos, velocidades en la mitad de los límites, 5 semillas, 5000 episodios). Las listas son bins por dimensión en el orden (θ1, θ2, ω1, ω2):

  | Variante | Celdas | Pasos finales | Éxito | Pasos en la curva | Celdas visitadas |
  |---|---|---|---|---|---|
  | 6 | 1296 | 174.5 ± 29.2 | 99 % ± 2 % | 251.3 ± 30.2 | 59 % |
  | 8 | 4096 | 165.4 ± 16.9 | 100 % | 231.3 ± 19.8 | 46 % |
  | 10 | 10 000 | 166.5 ± 12.4 | 100 % | 236.4 ± 10.6 | 38 % |
  | 12 | 20 736 | 196.0 ± 24.8 | 100 % | 268.3 ± 19.4 | 33 % |
  | 14 | 38 416 | 225.4 ± 51.3 | 100 % ± 1 % | 282.8 ± 16.5 | 28 % |
  | [12, 12, 8, 8] | 9216 | 239.0 ± 71.1 | 98 % ± 3 % | 245.8 ± 8.6 | 39 % |
  | **[8, 8, 12, 12]** | **9216** | **173.4 ± 19.2** | **99 % ± 1 %** | **220.9 ± 9.2** | 39 % |

- **Decisión:** 8 bins para cada ángulo y 12 para cada velocidad (9216 celdas). Es la configuración por defecto de `train.py` desde el 2026-11-20.
- **Justificación:**
  - **Forma de U:** con 6 bins la discretización es demasiado gruesa (más variable), y con 12 o 14 el aprendizaje se vuelve más lento con el mismo presupuesto (peor en las dos métricas). El mejor rango está entre 8 y 10 bins.
  - **La resolución de las velocidades importa más que la de los ángulos.** Con el mismo tamaño de tabla (9216), dar más bins a las velocidades es la variante que **aprende más rápido** de todo el experimento (220.9 ± 9.2 pasos en la curva). Darlos a los ángulos empeora el resultado final (239.0 ± 71.1). Tiene sentido físico: para subir hay que bombear energía en el momento justo del balanceo, y eso depende sobre todo de la velocidad.
  - En el resultado final, [8, 8, 12, 12], 8 y 10 están empatados: las diferencias quedan dentro del desvío. Se eligió [8, 8, 12, 12] por la velocidad de aprendizaje, que es la métrica menos ruidosa (D-11): unos 15 pasos menos que 10 bins, más de 2 errores estándar con 5 semillas.
- **Nota de reproducibilidad:** la variante "10" de B y "ángulos 10" de C repiten exactamente la variante "mitad" de A, semilla por semilla (por ejemplo, 144.4 pasos con la semilla 2 en las tres). Con la misma semilla, el código es determinista.
- **Nota de tiempos:** las variantes que aprenden peor tardan más (446 s con 14 bins, contra unos 260 s), porque sus episodios duran más pasos.
- **Evidencia:** `Acrobot/results/summary_bins.csv` y las corridas `Acrobot/results/bins_*`.

## COUNTER (Connect Four)

_Sin decisiones todavía._
