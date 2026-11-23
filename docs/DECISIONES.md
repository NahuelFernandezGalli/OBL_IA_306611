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
| [D-03](#d-03-formato-del-registro-de-experimentos) | General | Formato del registro de experimentos | Vigente (en Connect Four, el índice es por serie: D-26) |
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
| [D-20](#d-20-implementación-de-dyna-q) | Acrobot | Implementación de Dyna-Q | Vigente |
| [D-21](#d-21-tiempo-de-entrenamiento-por-episodio-y-comparación-en-el-mismo-lote) | Acrobot | Tiempo por episodio y comparación en el mismo lote | Vigente |
| [D-22](#d-22-barrido-de-n-pasos-de-planning) | Acrobot | Barrido de *n* (pasos de *planning*) | Vigente (resultados; α se revisa en D-24) |
| [D-23](#d-23-medición-del-aliasing-del-modelo) | Acrobot | Medición del *aliasing* del modelo | Vigente |
| [D-24](#d-24-α-con-dyna-q-α-bajo-y-n-chico) | Acrobot | α con Dyna-Q: α bajo y *n* chico | Vigente (resultados; la elección va en D-25) |
| [D-25](#d-25-configuración-y-modelo-final-de-dyna-q) | Acrobot | Configuración y modelo final de Dyna-Q | Vigente |
| [D-26](#d-26-registro-de-connect-four-una-corrida-es-una-serie-de-partidas) | Connect Four | Registro: una corrida es una serie de partidas | Vigente |
| [D-27](#d-27-implementación-de-minimax-alpha-beta-y-expectimax) | Connect Four | Implementación de Minimax, Alpha-Beta y Expectimax | Vigente |
| [D-28](#d-28-funciones-de-evaluación-componentes-y-combinaciones) | Connect Four | Funciones de evaluación: componentes y combinaciones | Vigente (pesos provisionales, se exploran en los experimentos) |
| [D-29](#d-29-rival-stratego-y-diversidad-de-partidas-con-aperturas-forzadas) | Connect Four | Rival Stratego y diversidad de partidas con aperturas forzadas | Vigente |
| [D-30](#d-30-e1-impacto-de-la-poda-alpha-beta) | Connect Four | E1: impacto de la poda Alpha-Beta | Vigente |

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
- **Cambio (2026-11-22):** en Connect Four, `runs.csv` tiene una fila por **serie** de partidas y no por partida; las partidas van en `<run_id>_games.csv`. Ver D-26.
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
- **Modelo final de Dyna-Q (2026-11-22):** `Acrobot/models/dynaq_final_20261122-182339_bbc2.pkl` (ver D-25). Mismo formato: guarda solo la tabla Q, y `n_planning` va entre los hiperparámetros. Se carga con el mismo `qlearning.load_model`.

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

### D-20: Implementación de Dyna-Q

- **Fecha:** 2026-11-22 · **Estado:** Vigente
- **Contexto:** la fase 2 pide Dyna-Q como componente de investigación (Sutton y Barto, *Reinforcement Learning: An Introduction*, 8.1 y 8.2). El algoritmo agrega a Q-Learning un modelo del entorno y *n* actualizaciones de *planning* por cada paso real.
- **Decisión** (`Acrobot/dynaq.py`, `DynaQAgent` hereda de `QLearningAgent`):
  - **Pseudocódigo del libro tal cual.** En cada paso real: (d) actualización de Q-Learning con la transición real (el `update` heredado, sin cambios), (e) `Model(S, A) ← (R, S')` y (f) *n* veces: un estado ya visitado al azar, una acción ya probada en ese estado al azar, `(R, S') ← Model(S, A)` y la misma actualización de Q-Learning.
  - **Modelo determinista, última transición vista.** Tres arreglos de `numpy` de tamaño estados × acciones: recompensa, s' y si la transición terminó el episodio. Con la discretización final son 3600 × 3 entradas, así que la memoria no es un problema.
  - **Muestreo del *planning* en dos pasos, como el libro:** primero un estado uniforme entre los visitados y después una acción uniforme entre las probadas en él. Se guardan una lista de estados visitados y, por estado, la lista de acciones probadas.
  - **Terminal contra corte por tiempo:** el modelo guarda `terminated` (llegada al objetivo). En el *planning*, una transición terminal tiene objetivo `r`, sin *bootstrap*. El corte a los 700 pasos (`truncated`) no se guarda como terminal, con el mismo criterio que D-08.
  - **Generador aleatorio propio para el *planning*,** separado del de la exploración. Así, con *n* = 0 la exploración consume exactamente los mismos números que en Q-Learning y la corrida es idéntica.
  - **Q₀ = −100 y el resto de la configuración final de Q-Learning** (D-17, D-18) como punto de partida. Los hiperparámetros se vuelven a revisar con Dyna-Q.
  - El `.pkl` guarda solo la tabla Q (con `n_planning` entre los hiperparámetros): para actuar no hace falta el modelo. `qlearning.load_model` carga los dos tipos de modelo.
- **Pseudocódigo de Sutton y Barto (8.2) contra el código:**

  | Paso del libro | Código |
  |---|---|
  | Inicializar Q(s, a) y Model(s, a) | `QLearningAgent.__init__`: `Q = q_init` (−100, D-17). `DynaQAgent.__init__`: `model_reward`, `model_next`, `model_terminal` |
  | (a) S ← estado actual (no terminal) | `train.run_episode`: `s = discretizer(obs)` |
  | (b) A ← ε-greedy(S, Q) | `agent.act(s)` (`qlearning.py`) |
  | (c) Ejecutar A; observar R, S' | `env.step(a)` y `s_next = discretizer(obs)` |
  | (d) Q(S,A) ← Q(S,A) + α[R + γ maxₐ Q(S',a) − Q(S,A)] | `DynaQAgent.update` → `super().update(...)` = `QLearningAgent.update`. Si `terminated`, el objetivo es solo R |
  | (e) Model(S,A) ← R, S' (entorno determinista) | `DynaQAgent.update_model`: guarda R, S' y `terminated`, y registra S y A como vistos |
  | (f) Repetir n veces: S ← estado visitado al azar | `DynaQAgent.planning`: `observed_states[int(u_state * n_observed)]` |
  | A ← acción probada en S al azar | `tried_actions[s][int(u_action * len(actions))]` |
  | R, S' ← Model(S, A) | `model_reward[s, a]`, `model_next[s, a]` (y `model_terminal[s, a]`) |
  | Misma actualización que en (d) | `super().update(...)`, con objetivo R si la transición guardada era terminal |

  El libro no distingue la llegada a un estado terminal del corte por tiempo, porque en su ejemplo no hay cortes. Aquí `terminated` se guarda en el modelo y `truncated` no (D-08).
- **Alternativas:**
  - Muestrear uniforme entre los pares (s, a) vistos, en un solo paso. Es más simple, pero no es lo que dice el libro: le da más peso a los estados donde se probaron más acciones.
  - Un modelo estocástico (contar transiciones y muestrear s' según su frecuencia), que atacaría el problema del *aliasing* (ver la hipótesis). Va más allá del 8.1–8.2 y queda fuera del alcance.
  - Priorizar los pares con más error (*prioritized sweeping*, 8.4). También queda fuera del alcance.
- **Hipótesis principal:** con Q₀ = −100 la tabla solo cambia cuando el valor de una llegada al objetivo se propaga hacia atrás, un paso por visita (D-17). El *planning* repite esas actualizaciones sin volver a jugar, así que Dyna-Q debería necesitar muchos menos episodios.
- **Hipótesis secundaria (*aliasing*):** Acrobot es determinista, pero la discretización no: una celda agrupa muchos estados reales y el mismo (s, a) lleva a celdas s' distintas. El modelo se queda con una sola de esas transiciones, que puede no ser representativa, y con *n* grande ese error se repite muchas veces. Puede pasar que a partir de cierto *n* el rendimiento empeore.
- **Verificación** (`Acrobot/test_dynaq.py`, pasan los 5 tests):
  - una transición terminal se planifica sin *bootstrap*, y una no terminal con `r + γ max Q(s')`;
  - el *planning* solo modifica pares (s, a) vistos de verdad;
  - el modelo se queda con la última transición;
  - **con *n* = 0, Dyna-Q reproduce exactamente a Q-Learning** (episodios, curva y tabla Q, 300 episodios).
- **Primera prueba** (una sola semilla, 500 episodios, solo orientativa): con *n* = 5 la evaluación greedy ya da 327 pasos y 95 % de éxito, cuando Q-Learning a esa altura sigue en unos 700 pasos. Con *n* = 10 da 590 pasos y 58 % de éxito, y con *n* = 25 y *n* = 50 no llega nunca. Es consistente con las dos hipótesis, pero hay que confirmarlo con 5 semillas (experimento `dyna_n` de `sweep.py`).

### D-21: Tiempo de entrenamiento por episodio y comparación en el mismo lote

- **Fecha:** 2026-11-22 · **Estado:** Vigente
- **Contexto:** cada paso de Dyna-Q hace 1 + *n* actualizaciones, así que la comparación con Q-Learning tiene que hacerse por episodios (eficiencia muestral) y también por tiempo de cómputo. Hasta la fase 1 solo se guardaba el tiempo total, que además incluye las evaluaciones periódicas. Esas evaluaciones tardan más cuanto peor es la política, porque los episodios fallidos duran 700 pasos. Los tiempos de la fase 1 se midieron con hasta 10 procesos en paralelo y solo son comparables dentro de un mismo lote (D-14).
- **Decisión** (`train.py`, `sweep.py`):
  - `_episodes.csv` tiene una columna nueva, **`elapsed_s`**: el tiempo acumulado **solo de entrenamiento** (episodios con aprendizaje, sin las evaluaciones). La métrica `learn_time_s` es su valor final. `train_time_s` se mantiene como antes (incluye las evaluaciones), para que siga siendo comparable con la fase 1.
  - **El experimento `dyna_n` incluye *n* = 0 (Q-Learning) en el mismo lote** que los demás *n*. Las 30 corridas compiten por la CPU en las mismas condiciones, así que sus tiempos se pueden comparar entre sí. Esto cubre la re-corrida R3 de `PLAN.md` con el presupuesto de 10 000 episodios.
- **Verificación:** medir el tiempo no cambia los resultados. Una re-corrida de Q-Learning con la configuración de `pes_epsilon` (semilla 0) repite paso a paso los primeros 500 episodios del CSV guardado en la fase 1. La variante *n* = 0 de `dyna_n` tiene que repetir los números de `pes_epsilon` / `decay0.999_min0.01`.
- **Costo medido** (una corrida sola, 500 episodios, casi todos de 700 pasos): unos 80 µs por paso en Q-Learning y unos 90, 100, 140 y 200 µs con *n* = 5, 10, 25 y 50. Cada actualización de *planning* cuesta unos 2.5 µs, bastante menos que un paso del entorno.

### D-22: Barrido de *n* (pasos de *planning*)

- **Fecha:** 2026-11-22 · **Estado:** Vigente como resultado. La elección de *n* y α se hace en D-24, porque con α = 0.5 el resultado cambia mucho.
- **Experimento:** `dyna_n` de `sweep.py`, con la configuración final de Q-Learning (Q₀ = −100, α = 0.5, bins (6, 6, 10, 10), ε de 1 a 0.01 con decay 0.999), 10 000 episodios y 5 semillas. Las 30 corridas se hicieron en un mismo lote de 10 procesos (D-21), en unos 4360 s.
- **Control:** *n* = 0 repite exactamente los números de `pes_epsilon` / `decay0.999_min0.01` (259.5 ± 29.6 y 368.3 ± 16.4), como predice la equivalencia de D-20.
- **Resultados:**

  | *n* | Pasos finales | Éxito | Pasos en la curva | Tiempo de entrenamiento (`learn_time_s`) | Finales por semilla |
  |---|---|---|---|---|---|
  | 0 (Q-Learning) | 259.5 ± 29.6 | 99 % | 368.3 ± 16.4 | 676 s | 286, 230, 220, 267, 294 |
  | **2** | 270.8 ± 83.1 | 98 % ± 4 % | **332.9 ± 14.1** | 732 s | 199, 267, 213, 430, 245 |
  | 5 | 347.3 ± 184.0 | 79 % ± 40 % | 373.3 ± 32.9 | 837 s | 700, 260, 331, 168, 277 |
  | 10 | 257.2 ± 48.0 | 99 % ± 1 % | 406.7 ± 11.3 | 1050 s | 208, 263, 315, 303, 198 |
  | 25 | 440.8 ± 194.9 | 65 % ± 42 % | 469.2 ± 16.6 | 1691 s | 651, 225, 700, 319, 309 |
  | 50 | 511.9 ± 147.7 | 59 % ± 43 % | 520.7 ± 17.5 | 2350 s | 700, 384, 407, 384, 685 |

  Curva greedy por bloques de 1000 episodios (media de 5 semillas):

  | *n* | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 0 | 629 | 521 | 468 | 309 | 331 | 292 | 321 | 273 | 269 | 271 |
  | 2 | 513 | 426 | 363 | 315 | 273 | 297 | 276 | 245 | 323 | 298 |
  | 10 | 525 | 499 | 480 | 433 | 398 | 355 | 293 | 396 | 353 | 335 |
  | 50 | 629 | 570 | 521 | 497 | 571 | 448 | 502 | 516 | 477 | 477 |

- **Lectura:**
  - **La hipótesis principal se cumple solo con *n* chico.** Con *n* = 2, Dyna-Q aprende más rápido que Q-Learning al principio: 513 contra 629 pasos en los primeros 1000 episodios y 363 contra 468 en el tercer bloque. Es la mejor curva del barrido. El resultado final empata con Q-Learning.
  - **Con *n* grande el aprendizaje empeora, y mucho:** con 25 y 50 la curva queda por encima de 450 pasos y aparecen semillas que no aprenden nada (700 pasos). El desvío entre semillas crece de 30 a casi 200 pasos. Es lo que predice la hipótesis del *aliasing* (D-20). Una explicación posible: la celda guarda una sola transición, muchas veces una que se queda en la misma celda o se aleja del objetivo. Al repetirla decenas de veces por paso con α = 0.5, el valor vuelve hacia −100 y se borra lo que se había propagado desde la meta. Q-Learning, en cambio, promedia las transiciones según la frecuencia con que ocurren de verdad. **Falta medirlo directamente** (cuántas celdas s' distintas hay por (s, a)).
  - **Costo:** el tiempo de entrenamiento crece casi lineal con *n*: unos 34 s más por cada paso de *planning* en 10 000 episodios. A igual tiempo de reloj, solo *n* = 2 compite con Q-Learning.
  - Todas las variantes con *planning* tocan más celdas (46 % contra 37 %): el *planning* propaga valores a pares que la experiencia real visitó pocas veces.
- **Pendiente:**
  1. Medir el *aliasing* del modelo.
  2. Revisar α con Dyna-Q. Con Q-Learning y Q₀ = −100 se eligió α = 0.5 porque la propagación era lenta (D-18). Con *planning* la propagación ya es rápida, así que un α menor podría promediar mejor el *aliasing* y permitir *n* más grandes.
- **Evidencia:** `Acrobot/results/summary_dyna_n.csv` y `Acrobot/results/dyna_n_*`.

### D-23: Medición del *aliasing* del modelo

- **Fecha:** 2026-11-22 · **Estado:** Vigente
- **Contexto:** el barrido de *n* (D-22) muestra que con mucho *planning* Dyna-Q aprende peor. La explicación propuesta es que el modelo determinista de Sutton y Barto (8.2) guarda una sola transición por (s, a), y la discretización hace que el mismo (s, a) lleve a celdas distintas. Hacía falta medirlo, no solo suponerlo.
- **Decisión** (`Acrobot/aliasing.py`): se juegan 200 episodios con la política del modelo final de Q-Learning más ε = 0.1, desde estados iniciales propios (semillas desde 3 000 000), y se cuentan las transiciones reales de cada (s, a). Se reportan:
  - cuántas celdas s' distintas tiene cada par (solo pares con al menos 5 visitas);
  - la **probabilidad de que la transición guardada coincida con la próxima real**. Si el par lleva a s'ᵢ con probabilidad pᵢ, la guardada es s'ᵢ con probabilidad pᵢ y la próxima también, así que coinciden con probabilidad Σ pᵢ². Vale 1 si el par es determinista. Se reporta la media pesada por visitas;
  - cuántas transiciones se quedan en la misma celda y cuántos pares a veces llegan al objetivo y a veces no.
- **Alternativas:** medirlo dentro del entrenamiento de Dyna-Q. Se descartó porque mezclaría la medición con una política que va cambiando. Con un modelo fijo, el resultado depende solo de la discretización y de qué estados visita una política que funciona.
- **Resultados** (modelo `qlearning_final_pes_20261120-204635_c4c6.pkl`, 45 597 transiciones, 1984 pares vistos, 995 con al menos 5 visitas):

  | Medida | Valor |
  |---|---|
  | Pares que llevan a más de una celda | **99.3 %** |
  | Celdas s' distintas por par (media / máximo) | **6.4** / 19 |
  | Probabilidad de que la transición guardada coincida con la real (pesada por visitas) | **28 %** |
  | Transiciones que se quedan en la misma celda | 11 % |
  | Pares que a veces se quedan en la celda y a veces salen | 31 % |
  | Pares que a veces llegan al objetivo y a veces no | 71 |

- **Lectura:** **el modelo de "última transición" no representa el entorno discretizado.** En el 72 % de los pasos reales, la transición que el modelo guarda para ese par es distinta de la que de verdad ocurre. Cada actualización de *planning* aplica, entonces, una transición que en general no es la típica del par, y con *n* grande ese error pesa más que la experiencia real. Hay dos casos especialmente dañinos:
  - **Pares que a veces llegan al objetivo y a veces no (71).** Si el modelo guardó la llegada, el *planning* lleva Q(s, a) hacia −1, un valor muy optimista. Si guardó la no llegada, lo baja. El valor salta según cuál fue la última visita.
  - **Pares que a veces se quedan en la celda (31 %).** Si el modelo guardó "me quedo en la misma celda", el *planning* repite una actualización que tira el valor hacia −100.

  Q-Learning no tiene este problema en la misma medida, porque cada transición real entra una sola vez y, con α < 1, la tabla promedia según la frecuencia real. Esto confirma la hipótesis secundaria de D-20 y explica por qué en D-22 conviene un *n* chico.
- **Evidencia:** `Acrobot/results/qlearning_final_pes_20261120-204635_c4c6_aliasing.json`.

### D-24: α con Dyna-Q: α bajo y *n* chico

- **Fecha:** 2026-11-22 · **Estado:** Vigente como resultado. La elección final está en D-25.
- **Contexto:** con Q-Learning y Q₀ = −100 se eligió α = 0.5, porque una α alta propaga rápido el valor de las llegadas al objetivo (D-18). Con Dyna-Q esa propagación la hace el *planning*. Además, el modelo acierta solo el 28 % de las transiciones (D-23), y una α baja promedia más muestras antes de mover la tabla. La hipótesis es que con *planning* conviene una α más baja.
- **Experimento:** `dyna_alpha` de `sweep.py`, con α ∈ {0.1, 0.2, 0.3} y *n* ∈ {0, 2, 10}, sobre la configuración final de Q-Learning, 10 000 episodios, 5 semillas y un mismo lote de 45 corridas. *n* = 0 (Q-Learning con las mismas α) separa el efecto de α del efecto del *planning*. α = 0.5 está en `dyna_n` (D-22), que corrió en otro lote, así que su tiempo no es comparable con este.
- **Resultados:**

  | *n* | α | Pasos finales | Éxito | Pasos en la curva | `learn_time_s` | Curva por bloques de 1000 episodios |
  |---|---|---|---|---|---|---|
  | 0 | 0.1 | 554.5 ± 68.9 | 52 % ± 21 % | 625.1 ± 40.8 | 1702 s | 700 → 667 → 624 → 616 → 565 |
  | 0 | 0.2 | 306.5 ± 35.8 | 92 % ± 4 % | 457.3 ± 55.2 | 1183 s | 693 → 553 → 410 → 356 → 318 |
  | 0 | 0.3 | 271.8 ± 56.0 | 96 % ± 4 % | 389.2 ± 24.2 | 1056 s | 683 → 491 → 319 → 303 → 266 |
  | 0 | 0.5 (`dyna_n`) | 259.5 ± 29.6 | 99 % | 368.3 ± 16.4 | (otro lote) | 629 → 468 → 331 → 321 → 271 |
  | 2 | 0.1 | 187.0 ± 19.1 | 100 % | **200.1 ± 10.3** | 641 s | **271** → 199 → 187 → 177 → 202 |
  | **2** | **0.2** | **186.0 ± 5.4** | **100 %** | 222.4 ± 7.4 | **555 s** | 371 → 244 → 234 → 182 → 192 |
  | 2 | 0.3 | 278.6 ± 144.6 | 91 % ± 18 % | 251.3 ± 12.6 | 585 s | 372 → 262 → 239 → 207 → 221 |
  | 10 | 0.1 | 203.6 ± 22.0 | 100 % | 247.7 ± 12.4 | 988 s | 360 → 232 → 236 → 218 → 216 |
  | 10 | 0.2 | 221.9 ± 21.5 | 100 % | 302.0 ± 14.0 | 1198 s | 482 → 345 → 300 → 248 → 251 |
  | 10 | 0.3 | 264.9 ± 124.0 | 96 % ± 6 % | 324.2 ± 23.4 | 991 s | 483 → 413 → 266 → 230 → 256 |

  (La curva por bloques muestra los bloques 1, 3, 5, 7 y 9.)

- **Lectura:**
  - **α y *planning* se complementan.** Sin *planning*, bajar α empeora: con α = 0.1, Q-Learning casi no aprende (52 % de éxito), porque el valor del objetivo se propaga demasiado lento (D-17). Con *planning*, bajar α mejora. **El *planning* se encarga de propagar y la α baja se encarga de promediar el *aliasing*.**
  - **Dyna-Q con *n* = 2 y α = 0.1–0.2 supera claramente a Q-Learning:** 186 pasos y 100 % de éxito con todas las semillas, contra 259.5 del mejor Q-Learning con el mismo presupuesto. Incluso mejora al Q-Learning final entrenado con 40 000 episodios (212.5 ± 40.4, D-18). Con α = 0.1 aprende casi todo en los primeros 1000 episodios (271 pasos, cuando Q-Learning sigue en 629).
  - **También gana en tiempo de reloj.** En este lote, *n* = 2 tarda unos 600 s, contra más de 1000 s de Q-Learning. El *planning* cuesta poco por paso (D-21), y como aprende antes, los episodios son mucho más cortos (los fallidos duran 700 pasos).
  - **Con α ≥ 0.3 vuelve la inestabilidad** (semillas que fallan: 564 y 505 pasos), igual que con *n* grande en D-22. Las dos cosas aumentan cuánto pesa cada transición del modelo, que en general no es la típica (D-23).
  - **El mejor α con *planning* es el más bajo probado,** así que puede haber algo mejor más abajo. También puede pasar que con α bajo los *n* grandes dejen de fallar. Por eso se corre la continuación `dyna_alpha_bajo`: (*n*, α) ∈ {(2, 0.05), (5, 0.05), (5, 0.1), (10, 0.05), (25, 0.05), (25, 0.1)}.
- **Continuación `dyna_alpha_bajo`** (mismas condiciones, otro lote de 30 corridas):

  | *n* | α | Pasos finales | Éxito | Pasos en la curva | `learn_time_s` | Finales por semilla |
  |---|---|---|---|---|---|---|
  | **2** | **0.05** | **169.5 ± 11.5** | **100 %** | **172.5 ± 7.1** | 509 s | 180, 160, 186, 159, 161 |
  | 5 | 0.05 | 210.5 ± 63.1 | 97 % ± 5 % | 210.3 ± 20.4 | 659 s | 289, 187, 282, 151, 144 |
  | 5 | 0.1 | 286.7 ± 207.6 | 80 % ± 40 % | 234.9 ± 21.3 | 967 s | 181, 168, 164, 220, 700 |
  | 10 | 0.05 | 196.9 ± 14.3 | 100 % | 239.7 ± 21.4 | 1165 s | 188, 219, 184, 208, 184 |
  | 25 | 0.05 | 206.2 ± 31.2 | 100 % | 259.5 ± 11.8 | 1574 s | 219, 227, 171, 168, 246 |
  | 25 | 0.1 | 248.2 ± 40.2 | 100 % ± 1 % | 283.6 ± 9.2 | 1536 s | 202, 297, 244, 291, 207 |

  - **Con *n* = 2, bajar α a 0.05 vuelve a mejorar:** 169.5 pasos con el menor desvío del barrido, y la curva ya está en 207 pasos en los primeros 1000 episodios. Desde el tercer bloque se mantiene plana, cerca de 160–175.
  - **Con α bajo, los *n* grandes dejan de fallar:** con *n* = 25 y α = 0.5 había semillas en 700 pasos (D-22). Con α = 0.05 y 0.1, todas las semillas llegan al 100 %. Es la confirmación más directa de que el problema de los *n* grandes era el peso del modelo con *aliasing* (D-23). Lo que importa es **cuánto mueve la tabla cada transición del modelo**, que depende de *n* y de α juntos. Aun así, ningún *n* grande supera a *n* = 2, y cuestan el triple de tiempo.
  - *n* = 5 queda en una zona inestable (una semilla en 700 con α = 0.1). Con 5 semillas, el desvío alto de esa variante no permite descartar que sea ruido.
  - **El mejor α sigue en el borde de la grilla (0.05).** Por eso se probó más abajo.
- **Última continuación `dyna_alpha_minimo`** (otro lote de 25 corridas). *n* = 1 es el mínimo de *planning* posible:

  | *n* | α | Pasos finales | Éxito | Pasos en la curva | Finales por semilla |
  |---|---|---|---|---|---|
  | 1 | 0.02 | 196.8 ± 26.9 | 100 % | 212.4 ± 19.1 | 183, 204, 241, 159, 197 |
  | 1 | 0.03 | 169.2 ± 22.6 | 100 % | 188.7 ± 13.2 | 135, 160, 165, 200, 186 |
  | 1 | 0.05 | 166.6 ± 19.5 | 100 % | 179.5 ± 6.8 | 147, 160, 152, 174, 201 |
  | 2 | 0.02 | 159.6 ± 23.5 | 100 % | 190.0 ± 19.6 | 132, 155, 159, 149, 203 |
  | 2 | 0.03 | 168.8 ± 16.6 | 100 % | 169.1 ± 16.2 | 154, 190, 145, 177, 179 |
  | 2 | 0.05 (`dyna_alpha_bajo`) | 169.5 ± 11.5 | 100 % | 172.5 ± 7.1 | 180, 160, 186, 159, 161 |

  - **Se encontró la meseta.** Con *n* ∈ {1, 2} y α entre 0.03 y 0.05, todas las variantes quedan entre 160 y 170 pasos finales y entre 169 y 189 en la curva. Las diferencias son más chicas que el desvío entre semillas (11 a 23 pasos), así que con 5 semillas no se distinguen.
  - **Con α = 0.02 la curva empieza a empeorar** (190–212): el aprendizaje es más lento al principio (unos 320 pasos en los primeros 1000 episodios, contra 265–300). Bajar más α ya no ayuda.
  - Entre *n* = 1 y *n* = 2 no hay una diferencia clara. Lo que importa es tener *planning*, no cuánto, siempre que sea poco.
  - Los tiempos de los distintos lotes no se comparan entre sí (D-21).
- **Elección:** ver D-25.
- **Evidencia:** `Acrobot/results/summary_dyna_alpha.csv`, `summary_dyna_alpha_bajo.csv`, `summary_dyna_alpha_minimo.csv` y los `dyna_alpha*` de cada experimento.

### D-25: Configuración y modelo final de Dyna-Q

- **Fecha:** 2026-11-22 · **Estado:** Vigente
- **Contexto:** los barridos de *n* (D-22) y de α (D-24) dejan una meseta: *n* ∈ {1, 2} con α entre 0.03 y 0.05, todas entre 160 y 170 pasos.
- **Decisión** (`DYNAQ_CONFIG` en `train.py`, se usa con `python train.py --algorithm dynaq`):
  - **n = 2 y α = 0.05.** Dentro de la meseta es la variante con menos desvío entre semillas (169.5 ± 11.5 pasos finales, 172.5 ± 7.1 en la curva) y la única cuyas 5 semillas quedan todas por debajo de 190 pasos.
  - El resto, igual que en Q-Learning (D-18): ángulos con bins (6, 6, 10, 10), ω1 ∈ ±2π, ω2 ∈ ±4.5π, **Q₀ = −100**, γ = 0.99, ε de 1.0 a 0.01 con decay 0.999. Así, la única diferencia con Q-Learning es el *planning* y su α.
  - **10 000 episodios.** La curva se estabiliza alrededor del episodio 3000 (unos 160–175 pasos desde el tercer bloque, D-24). No hace falta llegar a los 40 000 de Q-Learning.
- **Alternativas:**
  - *n* = 2 con α = 0.03 (mejor curva media, 169.1) o con α = 0.02 (mejor resultado final medio, 159.6). Tienen más desvío, y con 5 semillas las diferencias no son significativas.
  - *n* = 1 con α = 0.05 (166.6 ± 19.5). Es equivalente. Se prefirió *n* = 2 por el menor desvío.
  - No re-ajustar bins ni ε para Dyna-Q. Para no multiplicar barridos, la discretización y la exploración se toman de Q-Learning, y la comparación queda limpia: mismos estados y misma exploración, con y sin *planning*. Queda anotado como limitación.
- **Modelo final:** `Acrobot/models/dynaq_final_20261122-182339_bbc2.pkl`, entrenado con la semilla 0 en 593 s de entrenamiento (628 s con las evaluaciones), sin otros procesos corriendo. Reproduce exactamente el resultado de esa semilla en `dyna_alpha_bajo` / `n2_a0.05`: 180.28 pasos y 100 % de éxito en los 100 estados de evaluación. Sobre los **1000 estados de prueba** (`evaluate_model.py`):

  | Modelo | Algoritmo | Episodios | Tiempo de entrenamiento | Pasos medios | Desvío | Éxito |
  |---|---|---|---|---|---|---|
  | `qlearning_final_pes_20261120-204635_c4c6.pkl` (D-18) | Q-Learning, α = 0.5 | 40 000 | 947 s | 218.2 | 65.9 | 99.9 % |
  | **`dynaq_final_20261122-182339_bbc2.pkl`** | **Dyna-Q, n = 2, α = 0.05** | **10 000** | **593 s** | **183.2** | **56.4** | **100 %** |
  | `qlearning_final_20261120-154018_3f81.pkl` (referencia, Q₀ = 0) | Q-Learning | 20 000 | 303 s | 142.1 | 51.4 | 99.6 % |

  Los dos tiempos se midieron con un solo proceso, pero en días distintos. El de Q-Learning es el tiempo total de `train.py`, que incluye las evaluaciones periódicas.
- **Lectura:**
  - **Con Q₀ = −100, Dyna-Q es mejor que Q-Learning en las tres medidas de la comparación:**
    - **eficiencia muestral:** un cuarto de los episodios;
    - **costo computacional:** menos tiempo total, porque *planning* con *n* = 2 cuesta poco y los episodios se acortan antes;
    - **rendimiento final:** 16 % menos pasos y ningún fallo en 1000 estados.
  - Recupera parte del costo de cumplir la indicación del docente (D-18), aunque sigue por detrás del Q-Learning con inicialización optimista (142.1).
  - La mejora no viene de "más *planning* es mejor", sino de una combinación: **un poco de *planning*, que propaga el valor del objetivo, y una α baja, que promedia el *aliasing* del modelo** (D-22, D-23, D-24).
- **Evidencia:** `Acrobot/results/dynaq_final_20261122-182339_bbc2.json`, `Acrobot/results/dynaq_final_20261122-182339_bbc2_test.json` y `Acrobot/results/summary_dyna_*.csv`.

## COUNTER (Connect Four)

### D-26: Registro de Connect Four: una corrida es una serie de partidas

- **Fecha:** 2026-11-22 · **Estado:** Vigente
- **Contexto:** D-03 preveía una fila de `runs.csv` por partida. Pero lo que se compara es una configuración (agente, rival, turno) sobre varias partidas, igual que en Acrobot se compara una corrida sobre varios episodios.
- **Decisión** (`ConnectFour/match.py`, `ConnectFour/experiment_log.py`):
  - Una corrida es una **serie**: nuestro agente contra un rival, desde un turno fijo ("primero" o "segundo") y sobre un conjunto de aperturas (D-29).
  - `results/<run_id>.json` guarda la configuración de los dos agentes, el turno, las aperturas y las métricas agregadas: victorias, empates, derrotas, `win_rate`, `score` (1 por victoria y 0.5 por empate), jugadas por partida, tiempo medio y máximo por jugada, nodos por jugada y tiempo del rival.
  - `results/<run_id>_games.csv` tiene una fila por partida: apertura, número de jugador, ganador, resultado (V/E/D), jugadas, tiempo y nodos.
  - `runs.csv` tiene una fila por serie.
  - El turno nunca se mezcla: cada serie es de un solo turno.
  - *(2026-11-23)* Los experimentos que no son partidas usan el mismo esquema con otro sufijo: E1 guarda una fila por búsqueda en `<run_id>_searches.csv` (D-30).
  - El tiempo por jugada se mide en `play_game`, alrededor de `next_action`, con el mismo criterio para todos los agentes (incluidos Stratego y Random).
- **Alternativas:** una fila por partida en `runs.csv`, como decía D-03. El índice crecería mucho y para comparar configuraciones habría que agrupar siempre.
- **Justificación:** es el mismo esquema que Acrobot (corrida → episodios), y `experiment_log.py` es una copia del de Acrobot con un único cambio: el sufijo del CSV de filas (`_games`).

### D-27: Implementación de Minimax, Alpha-Beta y Expectimax

- **Fecha:** 2026-11-22 · **Estado:** Vigente
- **Contexto:** la letra pide Minimax con Alpha-Beta (analizando el impacto de la poda) y Expectimax, para decidir qué técnica conviene.
- **Decisión** (`search_base.py`, `minimax_agent.py`, `expectimax_agent.py`, `random_agent.py`):
  - **Forma estándar de AIMA** (5.2, 5.3 y 5.5): `max_value` / `min_value` con α y β; en Expectimax, `max_value` / `chance_value`, donde el nodo de azar es el promedio uniforme sobre las columnas legales del rival.
  - **Profundidad en plies** (jugadas individuales), igual que Stratego: `depth = 4` mira la jugada propia y las 3 siguientes.
  - **Flag `use_pruning`** en `MinimaxAgent`: con `False` se recorre el árbol completo. Así el mismo código mide con y sin poda.
  - **Nodos visitados:** cada llamada a la búsqueda cuenta un nodo, incluidos la raíz y las hojas. Queda en `last_stats` junto con el valor de la raíz, la columna y el tiempo.
  - **Orden de las columnas fijo**, por defecto central (3, 2, 4, 1, 5, 0, 6); también `left` (0 a 6) para medir el efecto del orden en la poda (AIMA 5.3).
  - **Desempate:** en la raíz se elige con `>` estricto, así que queda la primera columna en ese orden. Con eso la poda devuelve **el mismo valor y la misma jugada** que Minimax sin poda: las columnas anteriores a la primera óptima valen menos, la primera óptima devuelve su valor exacto (porque es mayor que α) y las siguientes devuelven a lo sumo α.
  - **Turno por paridad:** las capturas no cambian la cantidad de casillas vacías, así que el jugador al que le toca sale de la paridad de las fichas jugadas. El agente recibe su número de jugador y falla si no le toca a él.
- **Verificación** (`python test_search.py`, 9 tests, unos 9 s):
  - con y sin poda dan el mismo valor y la misma columna en 25 posiciones × profundidades 1 a 4 × 3 heurísticas, y con poda se visitan menos nodos o los mismos;
  - con `use_pruning=False` los nodos coinciden exactamente con un conteo independiente del árbol completo;
  - con poda se ahorran nodos en al menos 8 de 10 posiciones con profundidad 4;
  - gana en una jugada (Minimax y Expectimax) y bloquea en una (Minimax con profundidad ≥ 2);
  - Expectimax con profundidad 2 coincide con el promedio calculado a mano.
- **Alternativas:** negamax (más corto, pero se aleja de la presentación del curso con MAX y MIN separados); ordenar las jugadas por la heurística o profundización iterativa (fuera del alcance pedido; el orden central fijo ya es el ordenamiento simple de AIMA).
- **Justificación:** el código sigue el pseudocódigo del curso y se puede defender línea a línea. Los flags y los contadores permiten medir el impacto de la poda sin duplicar código.

### D-28: Funciones de evaluación: componentes y combinaciones

- **Fecha:** 2026-11-22 · **Estado:** Vigente (pesos provisionales, se exploran en los experimentos)
- **Contexto:** con profundidad limitada, los estados no terminales necesitan una evaluación. La letra pide probar varias, combinarlas y ponderarlas. El tablero tiene la regla de captura de `board.py`.
- **Decisión** (`heuristics.py`): la evaluación es una suma ponderada `h = Σ wₖ·fₖ` de componentes **simétricos** (valor propio − valor rival):
  - `n2` y `n3`: ventanas de 4 (las 69 del tablero) con 2 o 3 fichas propias y el resto vacías, sin fichas rivales.
  - `amenazas`: ventanas con 3 propias cuyo hueco se puede jugar ya.
  - `centro`: fichas en la columna 3.
  - `capturas`: fichas rivales que se capturarían con una sola jugada (horizontal y diagonal, igual que `board.py`).
  - **Terminales:** ganar vale `+(WIN + profundidad restante)` y perder `−(WIN + profundidad restante)`, con `WIN = 10⁶`, mayor que cualquier valor heurístico (hay un test con una cota holgada). Entre dos victorias se prefiere la más rápida, y entre dos derrotas la más lenta.
  - **Combinaciones iniciales:** `cero` (solo terminales, como línea base), `lineas`, `lineas_centro`, `amenazas`, `capturas` y `completa`. Los pesos (2, 5, 3, 20, 4) son un punto de partida y se exploran en los experimentos.
- **Tres criterios del curso (AIMA 5.4.1):**
  1. **Ordena los terminales igual que la utilidad:** los terminales no usan la heurística y quedan por encima o por debajo de cualquier no terminal (hay un test).
  2. **Barata:** una pasada por 69 ventanas, la columna central y 7 casillas jugables, sin búsqueda.
  3. **Correlacionada con ganar:** cada componente mide una forma concreta de acercarse a 4 en línea o de cambiar el tablero con una captura.
- **Alternativas:** copiar la heurística de Stratego (cuenta líneas de fichas con huecos de forma poco clara, y no daría información nueva); pesos aprendidos (fuera del alcance del curso).
- **Justificación:** son componentes simples, explicables y que se pueden activar de a uno, lo que permite medir el aporte de cada uno contra Stratego.

### D-29: Rival Stratego y diversidad de partidas con aperturas forzadas

- **Fecha:** 2026-11-22 · **Estado:** Vigente
- **Contexto:**
  - `play.py` carga `Stratego(2, 4)` como rival. Desofuscado, Stratego es Minimax **sin poda** de profundidad 4, con la heurística `1000·L4 + 3·L3 + 2·L2` (líneas propias menos rivales) y desempate con `>=` (elige la última columna empatada). Es determinista.
  - **El primer argumento no es su número de jugador:** `Stratego(n, d)` se considera el jugador `(n % 2) + 1`. El `Stratego(2, 4)` de la cátedra se cree jugador 1 aunque juegue segundo.
  - Como los dos agentes son deterministas, repetir una partida con el mismo turno da siempre la misma partida.
- **Decisión:**
  - El rival oficial es **`Stratego(2, 4)`, construido igual que en `play.py`, en los dos turnos**. Es el "campeón" que entrega la cátedra. `Stratego(1, 4)` (que sí se considera jugador 2) queda como dato complementario.
  - **Aperturas forzadas:** cada serie se juega sobre las 7 aperturas de una jugada o las 49 de dos jugadas (`--openings 1` o `2`), o solo desde el tablero vacío (`--openings 0`). Las jugadas de la apertura no las elige ningún agente.
  - Random usa una semilla distinta en cada partida (semilla base + índice de la partida).
- **Alternativas:** desempatar al azar con semilla (cambiaría el agente que se entrega y Stratego seguiría siendo determinista); empezar con jugadas al azar (es equivalente pero menos controlable: la enumeración es exhaustiva y reproducible).
- **Justificación:** las aperturas dan partidas distintas sin tocar a ninguno de los dos agentes, y enumerarlas todas evita elegir una muestra.
- **Hallazgo preliminar** (prueba de humo del 2026-11-22, sin registrar; se confirma en los experimentos):
  - Contra Random (10 partidas), `Stratego(2, 4)` gana 10 de 10 jugando primero y 9 de 10 jugando segundo. `Stratego(1, 4)` gana 8 de 10 en cada turno.
  - Desde el tablero vacío, Minimax d4 `lineas_centro` jugando primero le gana en 5 jugadas a `Stratego(2, 4)`: Stratego pone su ficha entre dos fichas nuestras y se la capturan. `Stratego(1, 4)` también pierde: apila fichas en la columna 6 sin bloquear.
  - Con las 7 aperturas de una jugada: Minimax d4 gana 7 de 7 jugando primero, pero solo 5 de 7 jugando segundo (pierde 2). **Stratego es mucho más fuerte cuando juega primero**, así que el turno es la variable más importante de los experimentos.

### D-30: E1: impacto de la poda Alpha-Beta

- **Fecha:** 2026-11-23 · **Estado:** Vigente
- **Contexto:** la letra pide analizar el impacto de Alpha-Beta. Hay que medir nodos y tiempo con y sin poda en varias profundidades, y mostrar con evidencia que la poda no cambia la decisión.
- **Decisión** (`python experiments.py poda`):
  - **50 posiciones fijas de mitad de partida:** entre 4 y 20 jugadas al azar (semilla 0), descartando las que ya terminaron. Se guardan como secuencias de columnas en el JSON, así que se reconstruyen exactamente.
  - **Profundidades 1 a 7** y tres variantes: `sin_poda`, `poda_centro` (columnas centrales primero) y `poda_izq` (de izquierda a derecha). Heurística `lineas_centro` en todas.
  - Se registran por búsqueda los nodos, el tiempo, el valor de la raíz y la columna (`results/<run_id>_searches.csv`, 1050 filas).
  - **Factor de ramificación efectivo** b\* = (nodos medios)^(1/d), para comparar con la teoría.
  - Corre en paralelo con 6 procesos (en una máquina de 14 hilos), para que la competencia por la CPU sea baja. Los tiempos son comparables dentro del lote. El costo por nodo fue parejo en todas las profundidades (53 a 66 µs), así que el tiempo es proporcional a los nodos.
- **Resultados** (medias sobre las 50 posiciones):

  | d | Nodos sin poda | Nodos con poda (centro) | Ahorro | Con poda (izq) | Ahorro | Tiempo sin poda | Tiempo con poda (centro) |
  |---|---|---|---|---|---|---|---|
  | 1 | 8 | 8 | 0 % | 8 | 0 % | 0.5 ms | 0.6 ms |
  | 2 | 52 | 24 | 53.9 % | 32 | 39.0 % | 3.3 ms | 1.3 ms |
  | 3 | 322 | 86 | 73.4 % | 133 | 58.6 % | 21 ms | 4.9 ms |
  | 4 | 2 011 | 233 | 88.4 % | 458 | 77.2 % | 128 ms | 13.6 ms |
  | 5 | 12 365 | 807 | 93.5 % | 1 701 | 86.2 % | 0.75 s | 45 ms |
  | 6 | 76 024 | 2 090 | 97.2 % | 5 708 | 92.5 % | 4.9 s | 103 ms |
  | 7 | 461 213 | 7 534 | 98.4 % | 19 664 | 95.7 % | 29.9 s | 490 ms |

- **Conclusiones:**
  - **La poda no cambia la decisión:** con el orden central, el valor y la columna coinciden con Minimax sin poda en las **350 búsquedas** (50 posiciones × 7 profundidades). Confirma en la práctica el argumento de D-27.
  - **El ahorro crece con la profundidad:** 54 % en d = 2 y 98.4 % en d = 7 (unas 61 veces menos nodos y tiempo). En d = 1 no hay nada que podar, porque solo se miran las hojas de la raíz. Por posición, el ahorro en d = 7 va de 93.2 % a 100 % (mediana 98.9 %): en algunas posiciones hay una victoria forzada que se encuentra enseguida.
  - **Efecto práctico:** con un presupuesto de un segundo por jugada, sin poda se llega a d = 5 (0.75 s) y con poda a d = 7 (0.49 s). La poda compra **dos niveles más de profundidad** con el mismo tiempo. Stratego, sin poda y con d = 4, gasta unos 128 ms por jugada, lo mismo que nuestro agente con poda en d = 6 (103 ms).
  - **El orden de las jugadas importa, como dice AIMA 5.3:** en d = 7, b\* baja de 6.44 sin poda a 3.58 con el orden central y a 4.11 con el orden de izquierda a derecha. Expresado como potencia de b: el orden central da b^0.68, entre el caso perfecto (b^0.5 ≈ 2.5) y el caso aleatorio (b^0.75 ≈ 4.0); el orden de izquierda a derecha queda en b^0.76, como un orden al azar. Mirar primero el centro poda 2.6 veces más que mirar de izquierda a derecha.
  - **Los empates existen y los desempata el orden, no la poda:** con el orden de izquierda a derecha el valor coincide siempre (350 de 350), pero la columna cambia en 62 de 350 búsquedas (18 %). Son jugadas distintas con el mismo valor minimax. Por eso la comparación de jugadas se hace con el mismo orden.
- **Alternativas:** posiciones al azar de `BoardSpace.sample` de la cátedra (muchas están casi llenas o terminadas, donde casi no hay nada que buscar); llegar a d = 8 (sin poda serían unos 3 millones de nodos y 3 minutos por posición, unas 2.5 horas en total, sin cambiar las conclusiones).
- **Evidencia:** `ConnectFour/results/poda_20261123-085728_087f.json`, `ConnectFour/results/poda_20261123-085728_087f_searches.csv` y `ConnectFour/results/summary_poda.csv`. Corrida total: 319 s con 6 procesos.
- **Para la figura C1:** nodos por jugada en escala logarítmica (las tres variantes) y tiempo por jugada en función de d, desde `summary_poda.csv`, con la coincidencia del 100 % escrita en la figura.
