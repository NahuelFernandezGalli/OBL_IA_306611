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
| [D-13](#d-13-cantidad-de-bins-8-por-ángulo-y-12-por-velocidad) | Acrobot | Cantidad de bins: 8 por ángulo y 12 por velocidad | Reemplazada por D-18 (con Q₀ = −100) |
| [D-14](#d-14-barridos-de-α-γ-y-ε-de-a-un-parámetro) | Acrobot | Barridos de α, γ y ε, de a un parámetro | Vigente (resultados; la elección final va en D-15) |
| [D-15](#d-15-hiperparámetros-finales-y-modelo-final-de-q-learning) | Acrobot | Hiperparámetros y modelo final de Q-Learning | Reemplazada por D-17 y D-18 (se entrenó con Q₀ = 0) |
| [D-16](#d-16-base-congelada-para-los-experimentos) | Acrobot | Base congelada para los experimentos | Vigente |
| [D-17](#d-17-inicialización-de-la-tabla-q) | Acrobot | Inicialización de la tabla Q: Q₀ = −100 (pesimista) | Vigente |
| [D-18](#d-18-re-ajuste-de-hiperparámetros-con-q₀--100) | Acrobot | Re-ajuste de hiperparámetros con Q₀ = −100 y modelo final | Vigente |
| [D-19](#d-19-gráficos-con-matplotlib-generados-desde-results) | General | Gráficos con matplotlib, generados desde `results/` | Vigente |

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

### D-19: Gráficos con matplotlib, generados desde `results/`

- **Fecha:** 2026-11-22 · **Estado:** Vigente
- **Contexto:** la letra pide gráficos claros y comentados en el informe. Los proyectos de la cátedra no traen ninguna librería de gráficos.
- **Decisión:**
  - Se agrega **`matplotlib>=3.9,<3.10`** (instalada la 3.9.4) a `Acrobot/` y a `ConnectFour/` con `poetry add`.
  - Las figuras se generan con un script por proyecto (`plots.py`) que **lee `results/`** y guarda en `figures/` en PNG a 150 dpi, con el nombre del experimento. El notebook de entrega llama a ese script; no grafica desde variables en memoria.
  - Cada algoritmo o variante tiene un color y un estilo de línea fijos, definidos una sola vez, para que se reconozca en todas las figuras y se lea también impreso en blanco y negro.
- **Alternativas:**
  - Otras librerías (seaborn, plotly): agregan dependencias sin aportar nada necesario para estas figuras.
  - La versión más nueva de matplotlib (3.10+): exige Python ≥ 3.10 y Poetry no la acepta con el `requires-python = ">=3.9"` de la cátedra, salvo que se cambie ese rango. La 3.9 es compatible con todo el rango y con `numpy<2`.
  - Graficar desde el notebook con variables en memoria: las figuras dependerían de haber corrido todo en una misma sesión y no se podrían rastrear hasta su registro.
- **Justificación:** matplotlib es la librería estándar con `numpy`. Generar las figuras desde los archivos de `results/` hace que cada figura se pueda reproducir y rastrear hasta las corridas que la originan (D-03), sin volver a entrenar.

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
    **Nota (2026-11-20):** el 2026-11-17 el docente indicó intentar resolver el problema *sin inicialización optimista*. Esta elección queda en revisión en D-17.
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
- **Agregado (2026-11-20):** para el modelo final hay además un conjunto de **prueba** de 1000 estados iniciales (semillas desde 2 000 000), que no se usa para elegir nada (`evaluate_model.py`, ver D-15).
- **Nota:** la evaluación greedy tiene una pequeña componente aleatoria, el desempate en celdas con valores iguales. Al recargar el modelo base se obtuvieron 196.0 pasos en vez de 196.6.

### D-10: Formato del modelo guardado

- **Fecha:** 2026-11-19 · **Estado:** Vigente
- **Decisión:** `models/<run_id>.pkl` es un `pickle` con la tabla Q (`numpy`), la configuración del discretizador, los hiperparámetros y la configuración completa de la corrida. `qlearning.load_model` devuelve el agente (con ε = 0) y el discretizador. El `run_id` coincide con el de `results/`, así cada modelo se puede rastrear hasta su registro.
- **Justificación:** la tabla Q no sirve sin el discretizador que define qué celda es cada fila. Guardarlos juntos evita cargar un modelo con una discretización equivocada.
- **Primer modelo:** `Acrobot/models/qlearning_base_20261119-183046_14cb.pkl`, 100 % de éxito y 196.6 pasos medios en la evaluación greedy.
- **Modelo final con Q₀ = 0 (2026-11-20):** `Acrobot/models/qlearning_final_20261120-154018_3f81.pkl` (ver D-15).
- **Modelo final (2026-11-20, Q₀ = −100):** `Acrobot/models/qlearning_final_pes_20261120-204635_c4c6.pkl` (ver D-18). Los anteriores se conservan como referencia.

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

- **Fecha:** 2026-11-20 · **Estado:** Reemplazada por D-18 (2026-11-20). Era la mejor opción con Q₀ = 0; con la inicialización pesimista conviene una tabla más chica.
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

### D-14: Barridos de α, γ y ε, de a un parámetro

- **Fecha:** 2026-11-20 · **Estado:** Vigente. Registra los resultados; la elección final, después de probar combinaciones, va en D-15.
- **Contexto:** con la discretización ya elegida (D-12 y D-13), falta ajustar los hiperparámetros del agente. Se cambia uno por vez respecto de α = 0.1, γ = 0.99 y ε de 1 a 0.05 con decay 0.995, con 5 semillas y 5000 episodios (D-11).
- **α (tasa de aprendizaje):**

  | α | 0.05 | **0.1** | 0.2 | 0.3 | 0.5 |
  |---|---|---|---|---|---|
  | Pasos finales | 187.4 ± 37.6 | **173.4 ± 19.2** | 208.5 ± 42.8 | 202.6 ± 35.4 | 231.7 ± 37.1 |
  | Pasos en la curva | 242.3 ± 22.5 | 220.9 ± 9.2 | **212.2 ± 4.8** | 227.4 ± 10.4 | 283.3 ± 23.2 |

  Con α chico (0.05) aprende más lento. Con α grande (0.5) empeora en todo: cada transición pisa casi la mitad del valor anterior y, como la discretización hace que la misma celda y acción lleven a resultados distintos, la tabla se vuelve muy ruidosa. Entre 0.1 y 0.2 hay un compromiso. Con 0.2 aprende más rápido (mejor curva y con poco desvío), pero la foto final es peor y más variable: con α constante la tabla nunca deja de moverse, y una α más alta hace que la política greedy oscile más de un momento a otro.
- **γ (factor de descuento):**

  | γ | 0.9 | 0.95 | 0.99 | 0.999 | 1.0 |
  |---|---|---|---|---|---|
  | Pasos finales | 171.0 ± 26.6 | 251.9 ± 73.7 | 173.4 ± 19.2 | **170.6 ± 23.7** | 192.9 ± 26.5 |
  | Pasos en la curva | 233.3 ± 11.7 | 228.2 ± 17.3 | 220.9 ± 9.2 | **218.6 ± 12.0** | 226.0 ± 7.7 |

  Casi no influye: las diferencias quedan dentro del desvío (el 0.95 tiene una semilla mala). Coincide con la teoría. Como la recompensa es −1 en cada paso, el valor de llegar en k pasos es −(1−γᵏ)/(1−γ), que es monótono en k para cualquier γ: menos pasos siempre vale más, y la política óptima no cambia. Lo único que cambia γ es la escala de los valores y cuánto se distinguen los estados lejanos del objetivo. Con γ = 0.9 el horizonte efectivo es de unos 10 pasos, pero las diferencias pequeñas siguen alcanzando para ordenar las acciones.
- **ε (exploración):**

  | Variante | Episodio en que ε llega al piso | Pasos finales | Pasos en la curva |
  |---|---|---|---|
  | decay 0.99, piso 0.05 | ~300 | 187.5 ± 24.2 | 246.4 ± 19.5 |
  | decay 0.995, piso 0.05 | ~600 | 173.4 ± 19.2 | 220.9 ± 9.2 |
  | **decay 0.999, piso 0.05** | **~3000** | **163.0 ± 7.1** | **195.2 ± 3.7** |
  | decay 0.995, piso 0 | ~600 (después, puramente greedy) | 203.0 ± 43.8 | 235.7 ± 11.3 |
  | decay 0.995, piso 0.01 | ~600 | 211.0 ± 46.5 | 223.5 ± 10.7 |
  | decay 0.995, piso 0.1 | ~600 | 199.7 ± 60.3 | 226.0 ± 16.8 |

  **El decaimiento es el hiperparámetro que más pesa.** Con decay 0.999 la exploración dura unos 3000 episodios en vez de 600, y es la mejor variante de todos los barridos en las dos métricas, con el desvío más chico (7.1 y 3.7). Cuando ε baja rápido, el agente se queda con la primera política que funciona y deja sin explorar regiones de la tabla (el "agente greedy" de AIMA 22.3.1). El piso de ε, en cambio, no muestra un efecto claro entre 0 y 0.1.
- **Advertencia sobre los tiempos:** no sirven para comparar variantes de lotes distintos. Por ejemplo, γ = 1.0 figura con 169 s contra unos 240 s del resto, pero eso pasa porque corrió en el último lote, con menos procesos compitiendo por la CPU. Solo son comparables dentro de un mismo lote de 10 corridas.
- **Evidencia:** `Acrobot/results/summary_alpha.csv`, `summary_gamma.csv` y `summary_epsilon.csv`.

### D-15: Hiperparámetros finales y modelo final de Q-Learning

- **Fecha:** 2026-11-20 · **Estado:** Reemplazada por D-17 y D-18 (2026-11-20). Todo lo de esta decisión se hizo con Q₀ = 0. Al pasar a la inicialización pesimista (D-17), los hiperparámetros se vuelven a ajustar (D-18), y este modelo queda como referencia de lo que se logra con Q₀ = 0.
- **Contexto:** los barridos de a un parámetro (D-14) no muestran interacciones entre hiperparámetros. Por eso se probaron combinaciones de los dos mejores valores de cada uno y después la cantidad de episodios.
- **Combinación** (α × decay de ε × γ, 5 semillas, 5000 episodios):

  | α | decay | γ | Pasos finales | Éxito | Pasos en la curva |
  |---|---|---|---|---|---|
  | **0.1** | **0.999** | **0.99** | **163.0 ± 7.1** | **100 %** | **195.2 ± 3.7** |
  | 0.1 | 0.999 | 0.999 | 190.5 ± 30.7 | 99 % | 209.8 ± 10.1 |
  | 0.1 | 0.9995 | 0.99 | 167.3 ± 22.0 | 99 % ± 1 % | 208.8 ± 24.0 |
  | 0.1 | 0.9995 | 0.999 | 164.8 ± 14.8 | 100 % | 196.7 ± 19.1 |
  | 0.2 | 0.999 | 0.99 | 269.0 ± 111.7 | 94 % ± 10 % | 219.3 ± 7.1 |
  | 0.2 | 0.999 | 0.999 | 184.4 ± 8.6 | 99 % ± 1 % | 220.3 ± 20.0 |
  | 0.2 | 0.9995 | 0.99 | 180.0 ± 19.2 | 99 % ± 1 % | 247.0 ± 32.2 |
  | 0.2 | 0.9995 | 0.999 | 232.5 ± 101.4 | 94 % ± 12 % | 213.5 ± 17.1 |

  α = 0.2 no se combina bien con la exploración larga: en las cuatro combinaciones es peor o más inestable que α = 0.1. Un decay de 0.9995 no mejora respecto de 0.999 y, con 5000 episodios, ε ni siquiera llega al piso.
- **Cantidad de episodios** (mejor combinación, 5 semillas). Los pasos en la curva no son comparables entre presupuestos distintos, porque la media incluye más o menos episodios del arranque. Por eso se comparan bloques de 1000 episodios de la curva, promediados entre semillas:

  | Episodios | Pasos finales | Curva en los últimos 2000 episodios | Tiempo (en paralelo) |
  |---|---|---|---|
  | 5000 | 163.0 ± 7.1 | 187.5 ± 8.7 | 246 s |
  | 10 000 | 156.2 ± 14.1 | 165.3 ± 12.5 | 430 s |
  | 20 000 | 162.6 ± 33.7 | 170.9 ± 26.1 | 593 s |

  Por bloques de 1000 episodios, la curva pasa de unos 207 pasos al principio a unos 185 cerca del episodio 3000, a unos 165 cerca del 10 000 y a unos 150–155 en la mayoría de los bloques después del 15 000. **Sigue mejorando, pero lento y con picos de oscilación**: un bloque de 233 en el episodio 13 000 y uno de 185 en el 20 000. Con α constante, la tabla nunca se queda quieta y la política greedy cambia de un momento a otro. Por eso la foto final de una semilla puede caer en un pico (la semilla 4 terminó en 229.9 pasos, contra unos 145 de las otras cuatro).
- **Decisión:** α = 0.1, γ = 0.99, ε de 1.0 a 0.05 con decay 0.999 por episodio, y **20 000 episodios**. Es la configuración por defecto de `train.py` (`python train.py --save-model`).
- **Modelo final:** `Acrobot/models/qlearning_final_20261120-154018_3f81.pkl`, entrenado con la semilla 0 en 303 s, sin otros procesos corriendo. Reproduce exactamente el resultado de esa semilla en el barrido: 143.81 pasos y 100 % de éxito sobre los 100 estados de evaluación.
- **Evaluación sin sesgo** (`evaluate_model.py`): los hiperparámetros se eligieron mirando los 100 estados de evaluación (semillas desde 1 000 000), así que esa medida es optimista. El modelo final se mide además sobre **1000 estados de prueba nuevos** (semillas desde 2 000 000), que no se usaron para elegir nada:

  | Modelo | Pasos medios | Desvío | Éxito | Pasos (solo exitosos) |
  |---|---|---|---|---|
  | Primero (D-10, configuración inicial, 3000 episodios) | 199.8 | 87.6 | 99.0 % | 194.7 |
  | **Final** | **142.1** | **51.4** | **99.6 %** | **139.9** |

  El modelo final llega al objetivo con un **29 % menos de pasos** y con menos variabilidad. Falla en 4 de los 1000 estados iniciales. Sobre los estados de evaluación había tenido 143.8 pasos y sobre los de prueba tiene 142.1: el sesgo de selección resultó despreciable.
- **Sobre la semilla:** no se eligió la mejor de las 5. La semilla 0 es la semilla por defecto y la primera del barrido. Las otras dieron 144.8, 148.9, 145.7 y 229.9 pasos, así que el modelo es representativo del resultado típico.
- **Mejoras posibles, no aplicadas:** una α decreciente por visitas, α(n), como en AIMA Fig. 22.8, reduciría la oscilación del final. Guardar la tabla en el mejor punto de la curva también ayudaría, pero habría que validarlo con un conjunto de estados distinto para no introducir sesgo.
- **Evidencia:** `Acrobot/results/summary_combinacion.csv`, `summary_episodios.csv`, `Acrobot/results/qlearning_final_20261120-154018_3f81.json` y, para la prueba, `Acrobot/results/qlearning_final_20261120-154018_3f81_test.json` y `qlearning_base_20261119-183046_14cb_test.json`.

### D-16: Base congelada para los experimentos

- **Fecha:** 2026-11-20 · **Estado:** Vigente
- **Contexto:** al principio, `sweep.py` armaba cada variante como cambios sobre `DEFAULT_CONFIG`. Pero `DEFAULT_CONFIG` cambió dos veces, con D-13 (discretización) y con D-15 (hiperparámetros). Volver a correr un experimento viejo, por ejemplo el A, habría dado otros números, porque las variantes heredaban los valores nuevos.
- **Decisión:** `sweep.py` define una base propia y fija (`BASE`), igual a la configuración inicial de `train.py`. Cada variante declara explícitamente lo que cambia respecto de ella. Los experimentos de hiperparámetros parten de la discretización elegida (`CHOSEN_DISC`). `DEFAULT_CONFIG` queda solo para entrenar con la configuración final.
- **Verificación:** se reconstruyó la configuración de las 31 variantes de los 8 experimentos y se comparó con la que quedó registrada en `results/<run_id>.json`: coinciden todas. Como el código es determinista con la misma semilla, volver a correr cualquier experimento reproduce sus números.

### D-17: Inicialización de la tabla Q

- **Fecha:** 2026-11-20 · **Estado:** Vigente. Se decidió **Q₀ = −100** el 2026-11-20, para cumplir la indicación del docente. Los experimentos anteriores (D-11 a D-16) se hicieron con Q₀ = 0.
- **Contexto:** el 2026-11-17 el docente indicó intentar resolver Acrobot *sin reward shaping ni inicialización optimista*. El reward shaping no se usa: la recompensa es la original del entorno, −1 por paso. La inicialización, en cambio, es un problema. Todas las recompensas son ≤ 0, así que el valor real de cualquier par (s, a) está entre −1/(1−γ) = −100, que es lo que vale no llegar nunca con γ = 0.99, y 0. **Inicializar en 0, que es lo estándar (AIMA, Fig. 22.8: "initially zero"), es optimista en la práctica.**
- **Implementación:** `QLearningAgent` acepta `q_init`, el valor inicial de toda la tabla. Por defecto es 0, para que los experimentos y modelos anteriores se reproduzcan igual.
- **Experimento** (configuración final de D-15, 5 semillas, 5000 episodios):

  | Q₀ | Interpretación | Pasos finales | Éxito | Pasos en la curva | Celdas visitadas |
  |---|---|---|---|---|---|
  | 0 | optimista (por encima de todo valor real) | 163.0 ± 7.1 | 100 % | 195.2 ± 3.7 | 38 % |
  | −50 | intermedio: por encima del valor real de los estados lejanos al objetivo, que con unos 140 pasos vale cerca de −75 | 134.0 ± 4.5 | 100 % | 171.9 ± 15.6 | 37 % |
  | −100 | pesimista: el peor valor posible | 689.7 ± 6.6 | 6 % ± 4 % | 695.7 ± 4.0 | 19 % |

- **Por qué falla Q₀ = −100:** −100 es justamente el punto fijo de la actualización cuando no se llega al objetivo. Con Q(s', ·) = −100, el objetivo de la actualización vale −1 + 0.99 · (−100) = −100, igual al valor actual, y la tabla **no cambia**. Lo mismo pasa en los cortes por tiempo. Las acciones probadas y las no probadas valen lo mismo, así que la política greedy es un desempate al azar. La única información que mueve la tabla es llegar al objetivo, donde el objetivo de la actualización es 0. Pero un agente que actúa al azar no llega nunca en 700 pasos (D-05: 0 de 50). Sin una primera llegada no hay señal que propagar hacia atrás, y el aprendizaje casi no arranca. Es el problema de las recompensas escasas (*sparse rewards*, AIMA 22.1 y p. 802: Q-Learning "has no means of looking into the future").
- **Por qué Q₀ = 0 explora mejor:** cada acción probada baja su valor por debajo de 0, y las no probadas quedan mejor que todas. El agente prueba sistemáticamente lo que no conoce, en lugar de depender solo de ε. Con −50 el optimismo es más moderado: sigue empujando a explorar, pero sin forzar a probar cada acción en cada celda, y en este experimento le fue mejor.
- **Opciones:**
  1. **Mantener Q₀ = 0** y justificarlo. Es la inicialización estándar del libro y no se eligió para forzar la exploración. El experimento muestra que la alternativa estrictamente no optimista no aprende con ε-greedy en este entorno.
  2. **Usar Q₀ = −50.** Da el mejor resultado, pero sigue siendo optimista para la mayoría de los estados, y elegirla por rendimiento sería, justamente, ajustar el optimismo.
  3. **Q₀ = −100**, cumpliendo estrictamente la indicación. Requiere muchísimo más entrenamiento (ver la prueba larga).
- **Decisión (2026-11-20):** **Q₀ = −100 = −1/(1−γ)**, la opción 3. Es la única estrictamente no optimista: ningún valor real está por debajo. Se acepta que el aprendizaje sea más lento, y los hiperparámetros se vuelven a ajustar para esta inicialización (D-18), porque los elegidos con Q₀ = 0 respondían a otra dinámica de aprendizaje. `DEFAULT_CONFIG` usa `q_init = -100` y `train.py` acepta `--q-init`. **Consecuencia:** si se cambia γ, el valor pesimista cambia con él (con γ = 0.999 sería −1000). Por eso γ se mantiene en 0.99.
- **Prueba larga con Q₀ = −100** (20 000 episodios, 5 semillas):

  | Decay de ε | Pasos finales | Éxito | Curva por bloques de 2000 episodios |
  |---|---|---|---|
  | 0.999 | 456.2 ± 52.5 | 73 % ± 14 % | 700 → 696 → 685 → 667 → 645 → 611 → 591 → 554 → 517 → 477 |
  | 0.9998 | 520.5 ± 118.5 | 48 % ± 26 % | 700 → 700 → 694 → 677 → 656 → 638 → 607 → 556 → 556 → 501 |

  **Sí aprende, pero muy despacio.** A los 20 000 episodios sigue en 456 pasos y 73 % de éxito, contra 143.8 pasos y 100 % con Q₀ = 0 y el mismo presupuesto. La curva sigue bajando, así que con muchos más episodios probablemente siga mejorando. Cada corrida tardó unos 2500 s en paralelo, porque los episodios fallidos duran los 700 pasos. **Explorar más no ayuda:** con un decay más lento (0.9998) le va peor. Las acciones al azar no llegan al objetivo, y lo que hace avanzar el aprendizaje es que el valor de las pocas llegadas se propaga hacia atrás, celda por celda, desde el objetivo.
- **Evidencia:** `Acrobot/results/summary_inicializacion.csv` y `Acrobot/results/summary_pesimista_larga.csv`.

### D-18: Re-ajuste de hiperparámetros con Q₀ = −100

- **Fecha:** 2026-11-20 · **Estado:** Vigente · **Reemplaza** a D-13 (bins) y a D-15 (hiperparámetros) para la configuración final.
- **Contexto:** con la inicialización pesimista (D-17), la dinámica de aprendizaje es otra. Con Q₀ = 0, el agente exploraba porque lo no probado parecía mejor. Con Q₀ = −100, la tabla solo se mueve cuando el valor de alguna llegada al objetivo se propaga hacia atrás, celda por celda (D-17). Los valores elegidos con Q₀ = 0 no tienen por qué seguir siendo los mejores. Se vuelven a ajustar uno por vez, en este orden: α, bins, ε. Se usan 5 semillas y 10 000 episodios, porque con 5000 casi no aprende. La "curva por bloques" es la media de la evaluación greedy en cada bloque de 1000 episodios.
- **α** (bins (8, 8, 12, 12), decay 0.999):

  | α | Pasos finales | Éxito | Pasos en la curva | Curva por bloques de 1000 episodios |
  |---|---|---|---|---|
  | 0.1 | 626.7 ± 33.4 | 32 % ± 13 % | 678.6 ± 9.6 | 700 → … → 633 |
  | 0.2 | 573.5 ± 67.9 | 42 % ± 18 % | 652.2 ± 17.4 | 700 → … → 600 |
  | 0.3 | 539.2 ± 58.7 | 49 % ± 15 % | 617.6 ± 10.8 | 700 → … → 522 |
  | **0.5** | **348.6 ± 62.6** | **93 % ± 6 %** | 487.6 ± 16.4 | 687 → 542 → 444 → 388 → **361** |
  | 0.7 | 408.4 ± 100.5 | 89 % ± 16 % | 442.9 ± 11.1 | 662 → 444 → 401 → 388 → 401 |
  | 0.9 | 355.7 ± 72.2 | 95 % ± 4 % | 481.9 ± 10.0 | 690 → 515 → 450 → 424 → 390 |
  | 1.0 | 628.1 ± 45.8 | 36 % ± 16 % | 639.8 ± 6.4 | 655 → 605 → 666 → 666 → 667 |

  **Con Q₀ = −100, α importa muchísimo, al revés que con Q₀ = 0 (D-14), donde convenía α chico.** Una α alta lleva el valor de cada llegada al objetivo a las celdas anteriores en pocas actualizaciones, y eso es lo que destraba el aprendizaje. En el otro extremo, **α = 1 colapsa**: la tabla se queda solo con la última muestra, sin promediar, y por el *aliasing* de la discretización la misma celda con la misma acción lleva a resultados distintos. Se eligió **α = 0.5**: es el mejor resultado final y la curva sigue bajando al final. Con 0.7 y 0.9 aprende más rápido al principio, pero se estanca cerca de 390.
- **Bins** (α = 0.5, decay 0.999). Las listas son bins por dimensión en el orden (θ1, θ2, ω1, ω2):

  | Bins | Celdas | Pasos finales | Éxito | Pasos en la curva |
  |---|---|---|---|---|
  | 4 | 256 | 302.9 ± 137.7 | 93 % ± 12 % | 408.8 ± 19.8 |
  | 5 | 625 | 685.3 ± 8.8 | 4 % ± 3 % | 692.7 ± 1.8 |
  | 6 | 1296 | 255.4 ± 63.8 | 100 % | 383.5 ± 14.2 |
  | **[6, 6, 10, 10]** | **3600** | **256.0 ± 45.4** | **100 %** | **387.1 ± 20.9** |
  | 8 | 4096 | 312.6 ± 76.2 | 97 % ± 2 % | 419.1 ± 15.5 |
  | [8, 8, 12, 12] | 9216 | 348.6 ± 62.6 | 93 % ± 6 % | 487.6 ± 16.4 |

  - **Con Q₀ = −100 conviene una tabla más chica que con Q₀ = 0.** Hay menos celdas por las que el valor del objetivo tiene que propagarse, y cada una se visita más. Con 4 bins la discretización ya es demasiado gruesa: el resultado medio todavía es aceptable, pero el desvío es muy grande.
  - **Hallazgo: la paridad de la cantidad de bins importa.** Con 5 bins no aprende nada (4 % de éxito), pero con 4 y con 6 sí. Con una cantidad impar, el bin del medio contiene el 0 y mezcla en una misma celda velocidades positivas y negativas, y ángulos a un lado y al otro de la posición colgando. Justo en el punto más bajo, donde hay que empujar a favor del movimiento para ganar energía, el agente no puede saber hacia dónde se mueve. Con una cantidad par hay un borde exactamente en 0 y el signo queda distinguido. Todas las variantes de los experimentos anteriores tenían cantidades pares.
  - Se eligió **[6, 6, 10, 10]**. Empata con 6 en las dos métricas, pero tiene menos desvío en el resultado final y le da más resolución a las velocidades, que con Q₀ = 0 resultaron lo más importante (D-13).
- **ε** (α = 0.5, bins [6, 6, 10, 10]):

  | Decay | Piso | Pasos finales | Éxito | Pasos en la curva | Últimos bloques |
  |---|---|---|---|---|---|
  | 0.99 | 0.05 | 261.1 ± 40.1 | 100 % | 442.4 ± 30.0 | 362 → 340 → 278 |
  | 0.995 | 0.05 | 360.7 ± 166.1 | 82 % ± 36 % | 414.0 ± 27.9 | 332 → 329 → 316 |
  | 0.999 | 0.05 | 256.0 ± 45.4 | 100 % | 387.1 ± 20.9 | 304 → 301 → 325 |
  | **0.999** | **0.01** | **259.5 ± 29.6** | **99 %** | **368.3 ± 16.4** | **273 → 269 → 271** |

  El resultado final empata entre decay 0.99 y 0.999. Con decay 0.999 aprende más rápido, y con un piso de 0.01 en vez de 0.05 la curva termina más baja y con menos desvío. Con inicialización pesimista, las acciones al azar no llevan al objetivo, así que una vez que la política funciona solo agregan ruido. Se eligió **decay 0.999 y piso 0.01**.
- **γ** se mantiene en 0.99. Cambiarlo cambia también el valor pesimista −1/(1−γ) (D-17), así que no es un ajuste independiente.
- **Evidencia:** `Acrobot/results/summary_pes_alpha.csv`, `summary_pes_alpha_alto.csv`, `summary_pes_bins.csv`, `summary_pes_bins_gruesos.csv` y `summary_pes_epsilon.csv`.
- **Cantidad de episodios** (configuración final, 5 semillas). Los primeros 10 000 episodios de cada corrida coinciden exactamente con la variante equivalente de `pes_epsilon`, porque el código es determinista:

  | Episodios | Pasos finales | Éxito | Curva en los últimos 2000 episodios | Tiempo (en paralelo) |
  |---|---|---|---|---|
  | 10 000 | 259.5 ± 29.6 | 99 % | ~270 | 845 s |
  | 20 000 | 272.7 ± 54.7 | 99 % ± 2 % | 258.3 ± 42.0 | 1365 s |
  | **40 000** | **212.5 ± 40.4** | **100 %** | **223.9 ± 30.4** | 2051 s |

  Por bloques de 2000 episodios, la curva baja de 575 a 312 en los primeros 6000 episodios, a unos 250 cerca del episodio 20 000 y a unos 225–250 hacia el 40 000. Mejora cada vez menos y oscila (con α = 0.5 la tabla se mueve mucho de una actualización a otra). Se eligieron **40 000 episodios**: más episodios dan cada vez menos mejora por unidad de tiempo.
- **Decisión** (configuración por defecto de `train.py` desde el 2026-11-20): ángulos con bins (6, 6, 10, 10), ω1 ∈ ±2π, ω2 ∈ ±4.5π, **Q₀ = −100**, α = 0.5, γ = 0.99, ε de 1.0 a 0.01 con decay 0.999 y **40 000 episodios**.
- **Modelo final:** `Acrobot/models/qlearning_final_pes_20261120-204635_c4c6.pkl`, entrenado con la semilla 0 (la semilla por defecto, no la mejor de las 5) en 947 s, sin otros procesos corriendo. Reproduce exactamente el resultado de esa semilla en el barrido: 220.89 pasos y 100 % de éxito sobre los 100 estados de evaluación. Sobre los **1000 estados de prueba** (`evaluate_model.py`):

  | Modelo | Q₀ | Pasos medios | Desvío | Éxito |
  |---|---|---|---|---|
  | Primero (D-10) | 0 | 199.8 | 87.6 | 99.0 % |
  | Final con Q₀ = 0 (D-15, referencia) | 0 | 142.1 | 51.4 | 99.6 % |
  | **Final (esta decisión)** | **−100** | **218.2** | **65.9** | **99.9 %** |

- **Lectura honesta:** cumplir la indicación del docente tiene un costo claro. Con el doble de episodios, el modelo pesimista tarda un 54 % más de pasos en llegar que el mejor modelo con Q₀ = 0, y queda incluso detrás del primer modelo optimista, que se entrenó con 3000 episodios. A cambio, llega al objetivo desde casi todos los estados iniciales (falla en 1 de 1000). La diferencia de pasos es el precio de que la exploración salga solo de ε-greedy: el valor del objetivo tiene que propagarse hacia atrás desde las pocas llegadas, en lugar de que la tabla empuje a probar lo desconocido.
- **Evidencia:** además de las anteriores, `Acrobot/results/summary_pes_episodios.csv`, `Acrobot/results/qlearning_final_pes_20261120-204635_c4c6.json` y `Acrobot/results/qlearning_final_pes_20261120-204635_c4c6_test.json`.

## COUNTER (Connect Four)

_Sin decisiones todavía._
