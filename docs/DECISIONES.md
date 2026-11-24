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
| [D-12](#d-12-rango-de-las-velocidades-la-mitad-de-los-límites-del-entorno) | Acrobot | Rango de velocidades: la mitad de los límites del entorno | Vigente con Q₀ = 0; reemplazada por D-38 con Q₀ = −100 |
| [D-13](#d-13-cantidad-de-bins-8-por-ángulo-y-12-por-velocidad) | Acrobot | Cantidad de bins: 8 por ángulo y 12 por velocidad | Reemplazada por D-18 (con Q₀ = −100) |
| [D-14](#d-14-barridos-de-α-γ-y-ε-de-a-un-parámetro) | Acrobot | Barridos de α, γ y ε, de a un parámetro | Vigente (resultados; la elección final va en D-15) |
| [D-15](#d-15-hiperparámetros-finales-y-modelo-final-de-q-learning) | Acrobot | Hiperparámetros y modelo final de Q-Learning | Reemplazada por D-17 y D-18 (se entrenó con Q₀ = 0) |
| [D-16](#d-16-base-congelada-para-los-experimentos) | Acrobot | Base congelada para los experimentos | Vigente |
| [D-17](#d-17-inicialización-de-la-tabla-q) | Acrobot | Inicialización de la tabla Q: Q₀ = −100 (pesimista) | Vigente |
| [D-18](#d-18-re-ajuste-de-hiperparámetros-con-q₀--100) | Acrobot | Re-ajuste de hiperparámetros con Q₀ = −100 y modelo final | Vigente; el rango, los episodios y el modelo cambian en D-38 |
| [D-19](#d-19-gráficos-con-matplotlib-generados-desde-results) | General | Gráficos con matplotlib, generados desde `results/` | Vigente |
| [D-20](#d-20-implementación-de-dyna-q) | Acrobot | Implementación de Dyna-Q | Vigente |
| [D-21](#d-21-tiempo-de-entrenamiento-por-episodio-y-comparación-en-el-mismo-lote) | Acrobot | Tiempo por episodio y comparación en el mismo lote | Vigente |
| [D-22](#d-22-barrido-de-n-pasos-de-planning) | Acrobot | Barrido de *n* (pasos de *planning*) | Vigente (resultados; α se revisa en D-24) |
| [D-23](#d-23-medición-del-aliasing-del-modelo) | Acrobot | Medición del *aliasing* del modelo | Vigente |
| [D-24](#d-24-α-con-dyna-q-α-bajo-y-n-chico) | Acrobot | α con Dyna-Q: α bajo y *n* chico | Vigente (resultados; la elección va en D-25) |
| [D-25](#d-25-configuración-y-modelo-final-de-dyna-q) | Acrobot | Configuración y modelo final de Dyna-Q | Vigente; el rango y el modelo cambian en D-38 |
| [D-26](#d-26-registro-de-connect-four-una-corrida-es-una-serie-de-partidas) | Connect Four | Registro: una corrida es una serie de partidas | Vigente |
| [D-27](#d-27-implementación-de-minimax-alpha-beta-y-expectimax) | Connect Four | Implementación de Minimax, Alpha-Beta y Expectimax | Vigente |
| [D-28](#d-28-funciones-de-evaluación-componentes-y-combinaciones) | Connect Four | Funciones de evaluación: componentes y combinaciones | Vigente (comparadas en D-31: se elige `completa`) |
| [D-29](#d-29-rival-stratego-y-diversidad-de-partidas-con-aperturas-forzadas) | Connect Four | Rival Stratego y diversidad de partidas con aperturas forzadas | Vigente |
| [D-30](#d-30-e1-impacto-de-la-poda-alpha-beta) | Connect Four | E1: impacto de la poda Alpha-Beta | Vigente |
| [D-31](#d-31-e2-funciones-de-evaluación-contra-stratego) | Connect Four | E2: funciones de evaluación contra Stratego (`completa`) | Vigente |
| [D-32](#d-32-con-la-regla-de-captura-el-que-empieza-tiene-victoria-forzada) | Connect Four | Con la regla de captura, el que empieza tiene victoria forzada | Vigente |
| [D-33](#d-33-e3-profundidad-contra-rendimiento) | Connect Four | E3: profundidad contra rendimiento (d = 5) | Vigente |
| [D-34](#d-34-e4-minimax-contra-expectimax-y-elección-de-la-técnica) | Connect Four | E4: Minimax contra Expectimax; se elige Minimax con Alpha-Beta | Vigente |
| [D-35](#d-35-prueba-con-los-pasos-de-cada-estado-y-agente-al-azar-sobre-los-mismos-estados) | Acrobot | Prueba con los pasos de cada estado y agente al azar sobre los mismos estados | Vigente |
| [D-36](#d-36-rangos-y-representación-repetidos-con-q₀--100) | Acrobot | Rangos y representación repetidos con Q₀ = −100 | Vigente (llevó a D-38) |
| [D-37](#d-37-figuras-de-acrobot-plotspy) | Acrobot | Figuras de Acrobot (`plots.py`) | Vigente |
| [D-38](#d-38-rango-510-y-10-000-episodios-para-la-configuración-final) | Acrobot | Rango 5/10 y 10 000 episodios para la configuración final; modelos nuevos | Vigente |

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
- **Agregado (2026-11-23):** sobre los 1000 estados de prueba, el agente al azar llega el **5.8 %** de las veces (D-35). Los 50 estados de esta línea base eran pocos para ver esos casos. El piso de los gráficos sigue siendo 700.

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

- **Fecha:** 2026-11-20 · **Estado:** Vigente para Q₀ = 0. **Reemplazada por D-38** para la configuración final con Q₀ = −100 (2026-11-23). · **Reemplaza** la parte de rangos de D-07.
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
- **Agregado (2026-11-23):** todo esto es con Q₀ = 0. Repetido con Q₀ = −100 y la configuración final, **el mejor rango es el del agente entrenado (5 / 10), no la mitad** (D-36).

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

- **Fecha:** 2026-11-20 · **Estado:** Vigente (α, bins y ε). **Cambiados en D-38 (2026-11-23):** el rango de las velocidades (5/10), los episodios (10 000) y el modelo entregado. · **Reemplaza** a D-13 (bins) y a D-15 (hiperparámetros) para la configuración final.
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

- **Fecha:** 2026-11-22 · **Estado:** Vigente (n = 2 y α = 0.05). **Cambiados en D-38 (2026-11-23):** el rango de las velocidades (5/10) y el modelo entregado.
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

- **Fecha:** 2026-11-22 · **Estado:** Vigente (comparadas en D-31: se elige `completa`)
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
- **Complemento (2026-11-23):** las aperturas de dos jugadas no son neutrales: en 25 de las 49 el jugador 1 ya tiene victoria forzada, y desde el tablero vacío también (D-32). Las aperturas se siguen usando, pero los resultados se reportan además separados por clase de apertura. La diferencia entre turnos no es solo que Stratego juegue mejor de primero: el juego favorece al que empieza.
- **Confirmación del número de jugador (2026-11-23,** `python experiments.py stratego_arg1`**):** el agente final (Minimax d5, `completa`) contra `Stratego(1, 4)`, que se cree jugador 2, sobre las 49 aperturas:
  - nosotros primero (Stratego juega segundo, el turno que coincide con lo que cree): 46/0/3, con las 3 derrotas en aperturas abiertas (21/0/3);
  - nosotros segundo (Stratego juega primero, sin coincidir): 48/0/1.
  
  Es el espejo de `Stratego(2, 4)`, que contra el mismo agente da 49/0/0 jugando nosotros primero y 26/0/23 jugando segundo (D-34). **Stratego solo juega bien en el turno que coincide con el número de jugador que cree tener.** El `Stratego(2, 4)` de la cátedra, en `play.py`, juega segundo, que es justo su turno débil. Por eso la prueba exigente es la combinación de los dos: jugando segundo contra `Stratego(2, 4)` (21 de 24 abiertas) y jugando primero contra `Stratego(1, 4)` (21 de 24 abiertas). Evidencia: `ConnectFour/results/summary_stratego_arg1.csv` y `summary_stratego_arg1_por_apertura.csv`.

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

### D-31: E2: funciones de evaluación contra Stratego

- **Fecha:** 2026-11-23 · **Estado:** Vigente
- **Contexto:** la letra pide experimentar con distintas funciones de evaluación, combinándolas y ponderándolas. Hay que compararlas contra Stratego, porque contra Random ganan todas.
- **Decisión** (`python experiments.py heuristicas`):
  - **Minimax d = 4 con poda** (la misma profundidad que Stratego) y cada una de las 10 heurísticas: las 6 combinaciones de D-28 más 4 variaciones de pesos de `completa`, cambiando un peso por vez (`n3` × 2, `amenazas` 50, `capturas` 10 y `centro` 6).
  - Contra `Stratego(2, 4)`, en los dos turnos, sobre las 49 aperturas de dos jugadas: 980 partidas en 214 s con 6 procesos.
  - Los resultados se separan además por la clase de la apertura (D-32): `python experiments.py heuristicas --split`.
- **Resultados** (V/E/D; "abiertas" son las 24 aperturas sin victoria forzada de nadie, D-32):

  | Heurística | 1º (49) | 2º (49) | 2º, abiertas (24) | 2º, forzadas para J1 (25) | ms/jugada |
  |---|---|---|---|---|---|
  | `cero` | 49/0/0 | 17/0/32 | 15/0/9 | 2/0/23 | 6 |
  | `lineas` | 49/0/0 | 18/0/31 | 16/0/8 | 2/0/23 | 22 |
  | `lineas_centro` | 49/0/0 | 17/0/32 | 14/0/10 | 3/0/22 | 20 |
  | `amenazas` | 49/0/0 | 21/0/28 | 18/0/6 | 3/0/22 | 23 |
  | `capturas` | 49/0/0 | 22/0/27 | 17/0/7 | 5/0/20 | 20 |
  | **`completa`** | 49/0/0 | **24/0/25** | **20/0/4** | 4/0/21 | 23 |
  | `completa_n3x2` | 49/0/0 | 23/0/26 | 20/0/4 | 3/0/22 | 23 |
  | `completa_amenazas50` | 49/0/0 | 24/0/25 | 20/0/4 | 4/0/21 | 23 |
  | `completa_capturas10` | 48/0/1 | 22/0/27 | 18/0/6 | 4/0/21 | 24 |
  | `completa_centro6` | 49/0/0 | 23/0/26 | 19/0/5 | 4/0/21 | 22 |

- **Conclusiones:**
  - **Jugando primero no se distinguen:** todas ganan las 49 (salvo una derrota de `completa_capturas10`). Stratego juega mal de segundo (D-29) y además en 25 aperturas el jugador 1 tiene victoria forzada (D-32). Las heurísticas se comparan jugando segundo.
  - **Jugando segundo, en las aperturas abiertas, `completa` es la mejor:** 20 de 24, contra 14 de `lineas_centro` y 15 de `cero`. Sobre las mismas aperturas, `completa` gana 8 que `lineas_centro` pierde, y `lineas_centro` gana solo 1 que `completa` pierde. Contra `cero` también es 8 a 1.
  - **Aporte de cada componente:** sumarle `amenazas` (+4) o `capturas` (+3) a `lineas_centro` mejora, y las dos juntas mejoran más (+6). `centro` sobre `lineas` no aporta (16 → 14). La regla de captura es la que más pesa en este juego (D-32).
  - **Con d = 4, sin heurística ya se juega razonable:** `cero` solo ve victorias y derrotas, y en partidas tan cortas y tácticas eso alcanza para casi todo. La heurística se nota en las aperturas abiertas, que son las que duran más.
  - **Los pesos importan poco cerca de `completa`:** las 4 variaciones dan entre 18 y 20 de 24. Subir `capturas` a 10 empeora un poco (18 de 24 y la única derrota jugando primero): sobrevalorar una captura posible hace que se descuide el resto.
  - **Costo:** todas las heurísticas cuestan lo mismo (20–24 ms por jugada con d = 4). `cero` es más barata (6 ms), porque muchos valores empatan en 0 y la poda corta mucho más.
- **Elección:** **`completa`** queda como función de evaluación por defecto para E3 y E4.
- **Limitación:** son 24 partidas por heurística en la comparación que discrimina; diferencias de 1 o 2 partidas no son concluyentes. La de `completa` contra `lineas_centro` y `cero` (8 a 1 sobre las mismas aperturas) sí es clara.
- **Evidencia:** `ConnectFour/results/summary_heuristicas.csv`, `ConnectFour/results/summary_heuristicas_por_apertura.csv` y los 20 `ConnectFour/results/heuristicas_20261123-1143*.json` (con sus `_games.csv`).

### D-32: Con la regla de captura, el que empieza tiene victoria forzada

- **Fecha:** 2026-11-23 · **Estado:** Vigente
- **Contexto:** en E2 aparecieron derrotas jugando segundo en solo 5 jugadas, y 23 de las 49 aperturas las perdían todas las heurísticas. Revisando una (apertura `01`): la `X` forzada queda pegada a la `O`; Stratego juega en la columna 2, captura y arma `OOO`, y si se bloquea en la 3, juega en la 4, captura otra vez y conecta 4. Nuestro agente lo había visto (valor −1 000 002, derrota segura). No es un error del agente: la apertura ya estaba perdida.
- **Decisión** (`python experiments.py aperturas`): se clasifican el tablero vacío y las 7 + 49 aperturas con Minimax con poda **sin heurística** (`cero`), con profundidades 8 y 10. Un valor de ±WIN solo puede salir de estados terminales reales, así que **es una victoria forzada demostrada** dentro de esa profundidad, no una estimación.
- **Resultados** (iguales con d = 8 y d = 10, 81 s en total):
  - **Tablero vacío: el jugador 1 gana forzado jugando en la columna 3**, en a lo sumo 7 jugadas (valor WIN + 1 con d = 8, WIN + 3 con d = 10 y WIN + 5 con d = 12, que corresponden a la misma victoria en la jugada 7).
  - **Aperturas de una jugada:** solo la `3` es victoria forzada (para J1); las otras 6 quedan abiertas dentro de la profundidad 10.
  - **Aperturas de dos jugadas:** 25 son victoria forzada para J1, ninguna para J2, y 24 quedan abiertas.
- **Consecuencias:**
  - Con esta regla de captura, Connect Four es **victoria del que empieza**, y en muy pocas jugadas. Nuestro agente, jugando primero con d ≥ 7, debería ganar siempre a cualquier rival (se verifica en E3).
  - **Jugando segundo contra un rival perfecto no se puede ganar.** Contra Stratego, lo que se mide es cuánto se aprovechan sus errores.
  - Las comparaciones entre agentes se reportan **separadas por clase de apertura** (forzada para J1 / abierta). Así una apertura que ya decide la partida no esconde diferencias entre agentes.
  - Stratego, jugando primero con d = 4, a veces no encuentra la victoria forzada: perdimos entre 2 y 5 de las 25 aperturas forzadas para J1 (D-31).
- **Alternativas:** buscar con más profundidad (d = 12 tarda 34 s por posición y no cambió nada); resolver el juego completo (fuera del alcance del curso y no hace falta para el análisis).
- **Evidencia:** `ConnectFour/results/aperturas_20261123-114823_0be6.json`, `ConnectFour/results/aperturas_20261123-114823_0be6_openings.csv` y `ConnectFour/results/summary_aperturas.csv`.

### D-33: E3: profundidad contra rendimiento

- **Fecha:** 2026-11-23 · **Estado:** Vigente
- **Contexto:** hay que elegir la profundidad del agente: más profundidad ve más lejos, pero el costo crece como b\*^d (D-30). Además hay que verificar lo que predice D-32: jugando primero y con suficiente profundidad, se gana siempre.
- **Decisión** (`python experiments.py profundidad` y `profundidad_vacio`):
  - Minimax con poda y `completa` (D-31), con **profundidades 1 a 8**, contra `Stratego(2, 4)` en los dos turnos.
  - Sobre las 49 aperturas (784 partidas, 460 s) y además desde el tablero vacío (16 partidas, 35 s). Todo con 6 procesos.
- **Resultados** (V/E/D; "abiertas" son las 24 aperturas sin victoria forzada, D-32; tiempos con 6 procesos en paralelo):

  | d | 1º (49) | 2º (49) | 2º, abiertas (24) | Tiempo medio por jugada | Tiempo máximo por jugada | Desde el vacío, 1º / 2º |
  |---|---|---|---|---|---|---|
  | 1 | 44/0/5 | 9/0/40 | 8/0/16 | 0.7 ms | 2 ms | V / D |
  | 2 | 45/0/4 | 14/0/35 | 10/0/14 | 2 ms | 35 ms | V / V |
  | 3 | 49/0/0 | 17/0/32 | 14/0/10 | 8 ms | 23 ms | V / V |
  | 4 | 49/0/0 | 24/0/25 | 20/0/4 | 25 ms | 0.1 s | V / V |
  | **5** | 49/0/0 | 26/0/23 | **21/0/3** | **0.11 s** | **0.8 s** | V / V |
  | 6 | 49/0/0 | 25/0/24 | 20/0/4 | 0.30 s | 2.1 s | V / V |
  | 7 | 49/0/0 | 27/0/22 | 22/0/2 | 1.0 s | 6.9 s | V / V |
  | 8 | 49/0/0 | 23/0/26 | 19/0/5 | 2.6 s | 36.7 s | V / V |

  Stratego (d = 4, sin poda) tarda unos 0.3 s por jugada en este lote.
- **Conclusiones:**
  - **Jugando primero, d ≥ 3 gana las 49.** Con d = 1 y 2 pierde algunas: no ve las capturas que arman una línea dos jugadas más adelante.
  - **Desde el tablero vacío se gana en los dos turnos desde d = 2.** Jugando segundo también: Stratego, de primero, no encuentra su victoria forzada (D-32).
  - **Jugando segundo, el rendimiento mejora hasta d = 4 y después se estanca:** 20 a 22 de 24 en las abiertas entre d = 4 y d = 7. Las diferencias de 1 o 2 partidas no son concluyentes.
  - **Más profundidad no siempre es mejor contra un rival que se equivoca.** d = 8 rinde menos que d = 7 (19 contra 22 de 24). Hay 3 aperturas abiertas que d = 7 gana y d = 8 pierde, y ninguna al revés. Revisando una (apertura `15`):
    - d = 8 detecta en su segunda jugada que la posición está **perdida contra un rival perfecto** (todas las columnas valen −WIN). Desde ahí elige la derrota más lenta (D-28), y Stratego la aprovecha.
    - d = 7 no llega a ver esa derrota, sigue jugando por la heurística, y Stratego termina equivocándose.
    - La evaluación de d = 8 es correcta. Lo que falla es el **supuesto de Minimax de que el rival es óptimo**: cuando todas las jugadas pierden contra un rival perfecto, Minimax "se rinde", en vez de elegir la que le deja más chances de error al rival. Es justo lo que modela Expectimax (E4).
  - **El costo crece unas 3 a 4 veces por nivel**, como en E1. El tiempo máximo crece más rápido que el medio: llega a 37 s por jugada con d = 8, en posiciones abiertas de mitad de partida.
- **Elección:** **d = 5** como profundidad por defecto para E4 y para el agente que se entrega:
  - gana todo jugando primero;
  - de segundo está en la meseta (21 de 24 en las abiertas, lo mismo que d = 4 a d = 7, salvo diferencias no concluyentes);
  - cuesta 0.11 s por jugada en promedio y 0.8 s como máximo, menos que Stratego.
  
  d = 7 dio el mejor número (22 de 24), pero tarda 9 veces más por jugada (hasta 7 s) por una diferencia de una partida.
- **Alternativas:** profundización iterativa con límite de tiempo (fuera de lo pedido); elegir la profundidad solo por resultado (d = 7) sin mirar el costo.
- **Evidencia:** `ConnectFour/results/summary_profundidad.csv`, `ConnectFour/results/summary_profundidad_por_apertura.csv`, `ConnectFour/results/summary_profundidad_vacio.csv`, y las series `ConnectFour/results/profundidad_2026112*.json` y `profundidad_vacio_2026112*.json` (con sus `_games.csv`).
- **Para la figura C4:** V/E/D jugando segundo en las abiertas y tiempo medio y máximo por jugada (escala logarítmica), en función de d.

### D-34: E4: Minimax contra Expectimax, y elección de la técnica

- **Fecha:** 2026-11-23 · **Estado:** Vigente
- **Contexto:** la letra pide implementar Minimax y Expectimax para decidir cuál es la mejor técnica en este juego. En teoría:
  - Expectimax supone que el rival juega al azar (nodo de azar con promedio), así que debería ser mejor contra Random.
  - Minimax supone un rival óptimo, y Stratego es un minimax determinista, así que debería ser mejor contra Stratego.
  - Además, en E3 apareció la hipótesis de que Expectimax no "se rinde" ante una derrota forzada y podría aprovechar mejor los errores de Stratego (D-33).
- **Decisión** (`python experiments.py expectimax_random`, `expectimax_stratego` y `minimax_vs_expectimax`):
  - Tres agentes, todos con `completa` (D-31):
    - **Minimax d5** con poda (D-33);
    - **Expectimax d5**, a la misma profundidad;
    - **Expectimax d4**, con un tiempo por jugada parecido, porque Expectimax no poda.
    
    Es la misma idea que en Dyna-Q: comparar con el mismo presupuesto de búsqueda y con el mismo tiempo.
  - Cada uno juega contra Random, contra `Stratego(2, 4)` y contra Minimax d5 (en este último caso, Minimax d5 contra sí mismo es la referencia). Siempre en los dos turnos y sobre las 49 aperturas: 882 partidas en 296 s con 6 procesos.
- **Resultados** (V/E/D; entre paréntesis, las 24 aperturas abiertas, D-32):

  | Agente | ms/jugada | Random, 1º | Random, 2º | Stratego, 1º | Stratego, 2º | Minimax d5, 1º | Minimax d5, 2º |
  |---|---|---|---|---|---|---|---|
  | **Minimax d5** | **~95** | 49/0/0 | 48/0/1 | 49/0/0 | **26/0/23 (21/0/3)** | 37/0/12 (12/0/12) | 12/0/37 (12/0/12) |
  | Expectimax d5 | ~1150 | 49/0/0 | 47/0/2 | 47/0/2 | 15/0/34 (15/0/9) | 35/0/14 (10/0/14) | 6/0/43 (6/0/18) |
  | Expectimax d4 | ~175 | 49/0/0 | 47/0/2 | 49/0/0 | 11/0/38 (11/0/13) | 29/0/20 (10/0/14) | 6/0/43 (6/0/18) |

  Jugadas por partida contra Random: Minimax 7.1 (1º) y 8.0 (2º); Expectimax d5 6.4 y 7.4; Expectimax d4 6.5 y 7.4.
- **Conclusiones:**
  - **Contra Random ganan todos**: las pocas derrotas son aperturas con victoria forzada para el jugador 1, en las que Random encontró la captura ganadora por casualidad. **Expectimax gana más rápido** (alrededor de 0.6 jugadas menos por partida). Es lo que predice la teoría: contra un rival al azar, apuesta a jugadas que ganan "en promedio", aunque un rival óptimo las podría frenar.
  - **Contra Stratego, Minimax es claramente mejor:** jugando segundo, en las aperturas abiertas gana 21 de 24, contra 15 (Expectimax d5) y 11 (Expectimax d4). Jugando primero, Expectimax d5 incluso pierde 2 aperturas abiertas. El modelo del rival importa: Stratego no juega al azar, castiga las jugadas arriesgadas, y el promedio de Expectimax subestima ese riesgo.
  - **Mano a mano, Minimax también gana:** en las aperturas abiertas, Minimax contra sí mismo queda 12 a 12. Expectimax d5 gana solo 10 de 24 jugando primero y 6 de 24 jugando segundo. Expectimax d4, además, no convierte 6 de las 25 aperturas con victoria forzada jugando primero.
  - **Se refuta la hipótesis de E3:** en las aperturas que ya están perdidas jugando segundo contra Stratego, Expectimax ganó 0 de 25 y Minimax 5 de 25. Expectimax no aprovecha mejor los errores de Stratego, porque espera errores al azar y Stratego no se equivoca así.
  - **Costo:** a la misma profundidad, Expectimax visita unas 10 veces más nodos (16 000 contra 1 500 por jugada) y tarda unas 11 veces más, porque un promedio necesita todos los hijos y no se puede podar. A igual tiempo (Expectimax d4), es todavía peor.
- **Elección de la técnica:** **Minimax con poda Alpha-Beta**, d = 5 y `completa`. Es mejor o igual en todos los enfrentamientos, salvo en la velocidad para ganarle a Random, y unas 10 veces más barato. Su modelo del rival (óptimo) es el correcto contra Stratego, que es el rival que pide la letra. Expectimax solo convendría si el rival jugara realmente al azar.
- **Alternativas:** Expectimax con un modelo del rival que no sea uniforme (por ejemplo, dándole más probabilidad a las jugadas que la heurística considera buenas). Va más allá del Expectimax estándar del curso y no se probó.
- **Evidencia:** `ConnectFour/results/summary_expectimax_random.csv`, `summary_expectimax_stratego.csv`, `summary_minimax_vs_expectimax.csv` y sus `_por_apertura.csv`, más las 18 series `expectimax_random_*`, `expectimax_stratego_*` y `minimax_vs_expectimax_*` en `ConnectFour/results/`.
- **Para la figura C2:** V/E/D en barras apiladas por rival y por turno, para los tres agentes, separando las aperturas abiertas.

### D-35: Prueba con los pasos de cada estado y agente al azar sobre los mismos estados

- **Fecha:** 2026-11-23 · **Estado:** Vigente
- **Contexto:** la figura A4 (distribución de pasos en la prueba) necesita los pasos de cada uno de los 1000 estados de prueba. `evaluate_model.py` guardaba solo agregados (media, desvío y éxito). Además, la línea base al azar (D-05) usaba otros 50 estados, así que no se podía comparar estado por estado con los modelos.
- **Decisión:**
  - `train.evaluate(..., per_episode=True)` agrega `min_steps`, `max_steps`, `steps_per_episode` y `success_per_episode`. Por defecto está apagado, así que los JSON de entrenamiento (`eval_curve` y la evaluación final) no cambian.
  - `evaluate_model.py` lo usa siempre, y con `--random` evalúa un agente al azar (semilla 0) con la misma función y sobre los **mismos 1000 estados de prueba** (semillas desde 2 000 000). Se guarda en `results/random_test.json`.
  - Se volvieron a evaluar los cuatro modelos guardados.
- **Verificación:** en los cuatro modelos, la media, el desvío, la tasa de éxito y los pasos de los episodios exitosos coinciden **exactamente** con los `_test.json` anteriores. La evaluación greedy es reproducible: el desempate al azar (D-09) arranca de la misma semilla al cargar el modelo.
- **Resultados** (1000 estados de prueba):

  | Agente | Pasos medios | Mín. | Máx. | Éxito |
  |---|---|---|---|---|
  | Dyna-Q final (`dynaq_final_20261122-182339_bbc2`) | 183.2 | 98 | 476 | 100 % |
  | Q-Learning final, Q₀ = −100 (`qlearning_final_pes_20261120-204635_c4c6`) | 218.2 | 121 | 700 | 99.9 % |
  | Q-Learning, Q₀ = 0 (`qlearning_final_20261120-154018_3f81`) | 142.1 | 88 | 700 | 99.6 % |
  | Primer modelo (`qlearning_base_20261119-183046_14cb`) | 199.8 | 99 | 700 | 99.0 % |
  | Al azar | 693.2 | 320 | 700 | 5.8 % |

- **Hallazgo:** el agente al azar **a veces llega** (58 de 1000, con 583 pasos de media entre los que llegan). En los 50 estados de D-05 no había pasado nunca. No cambia ninguna conclusión, pero la línea base correcta para el informe es esta: 5.8 % de éxito sobre los mismos estados que los modelos.
- **Observación:** Dyna-Q es el único modelo que llega en todos los estados de prueba, y también es el que tiene el peor caso más bajo (476 pasos).
- **Evidencia:** `Acrobot/results/*_test.json` (los cuatro modelos y `random_test.json`).

### D-36: Rangos y representación repetidos con Q₀ = −100

- **Fecha:** 2026-11-23 · **Estado:** Vigente. La representación queda confirmada, y el rango cambia a 5/10 en D-38.
- **Contexto:** el rango de las velocidades (D-12) y la representación (D-06, experimento C) se eligieron con Q₀ = 0. Después del cambio a Q₀ = −100 (D-17) se re-ajustaron α, bins y ε (D-18), pero no estos dos. Si el modelo entregado usa Q₀ = −100, las figuras que justifican su discretización tienen que salir de corridas con Q₀ = −100. Si no, en la defensa queda la pregunta "¿por qué estos rangos, si los elegiste con otra inicialización?".
- **Decisión:** dos barridos nuevos en `sweep.py`, sobre la configuración final de Q-Learning (`FINAL_PES`: Q₀ = −100, bins (6, 6, 10, 10), α = 0.5, γ = 0.99, ε de 1 a 0.01 con decay 0.999, 10 000 episodios), con las **mismas variantes** que los originales:
  - `pes_rangos`: teórico (12.6 / 28.3), mitad (6.3 / 14.1, la elegida), agente entrenado (5 / 10) y agente al azar (3.4 / 5.9).
  - `pes_representacion`: ángulos (6, 6, 10, 10) = 3600 celdas, contra cos/sin con 4, 6 y 10 bins por dimensión (4096, 46 656 y 10⁶ celdas). La comparación más justa es con `crudo_4`, que tiene casi el mismo tamaño de tabla.
- **Verificación:** la variante que coincide con `FINAL_PES` (mitad y ángulos) repite **exactamente** `pes_epsilon` (`decay0.999_min0.01`): 259.47 ± 29.61 pasos, 99.4 % de éxito y 368.34 pasos en la curva, en los dos barridos.
- **Tiempo:** 40 corridas en dos lotes de 10 procesos, unos 70 minutos en total. Las corridas que no aprenden tardan más (`crudo_6` y `crudo_10`, unos 1450 s cada una).
- **Resultados, representación** (`pes_representacion`, 5 semillas):

  | Variante | Celdas | Pasos finales | Éxito | Pasos en la curva |
  |---|---|---|---|---|
  | **Ángulos (6, 6, 10, 10)** | 3600 | **259.5 ± 29.6** | **99 %** | **368.3 ± 16.4** |
  | cos/sin, 4 bins | 4096 | 319.2 ± 63.6 | 97 % ± 3 % | 472.9 ± 11.6 |
  | cos/sin, 6 bins | 46 656 | 652.9 ± 33.6 | 22 % ± 14 % | 670.1 ± 6.5 |
  | cos/sin, 10 bins | 10⁶ | 695.3 ± 2.7 | 2 % ± 1 % | 697.7 ± 0.8 |

  **Se confirma D-06 con Q₀ = −100, y con más diferencia que con Q₀ = 0.** A igual tamaño de tabla (`crudo_4`), los ángulos aprenden más rápido (368 contra 473 en la curva) y terminan mejor. Las tablas grandes de cos/sin directamente no aprenden: con inicialización pesimista, el valor del objetivo se tiene que propagar celda por celda, y con 46 656 o 10⁶ celdas casi ninguna se visita lo suficiente (9 % y 1 %).
- **Resultados, rangos** (`pes_rangos`, 5 semillas; la última columna promedia las 8 evaluaciones greedy de los últimos 2000 episodios):

  | Variante | Pasos finales | Éxito | Pasos en la curva | Celdas visitadas | Greedy, últimos 2000 |
  |---|---|---|---|---|---|
  | Teórico (12.6 / 28.3) | 326.4 ± 185.8 | 82 % ± 30 % | 359.4 ± 13.4 | 15 % | 290 |
  | Mitad (6.3 / 14.1), la actual | 259.5 ± 29.6 | 99 % | 368.3 ± 16.4 | 37 % | 270 |
  | **Agente entrenado (5 / 10)** | **212.9 ± 38.8** | **100 %** | 353.2 ± 12.1 | 53 % | **231** |
  | Agente al azar (3.4 / 5.9) | 286.7 ± 124.1 | 93 % ± 14 % | **309.5 ± 14.2** | 82 % | 239 |

  Por semilla, pasos finales: mitad 286 / 230 / 220 / 267 / 294; 5/10: 178 / 195 / 288 / 208 / 195.
- **Hallazgo: con Q₀ = −100 cambia el mejor rango.** Con Q₀ = 0 la mitad era la mejor y la más estable (D-12). Con Q₀ = −100:
  - 5/10 termina mejor en 4 de las 5 semillas, llega en el 100 % de los estados con todas las semillas y es mejor en la evaluación greedy de los últimos 2000 episodios (231 contra 270).
  - En la curva de entrenamiento empatan dentro del desvío (353 contra 368).
  - El rango del agente al azar es el que aprende **más rápido** (309 en la curva), pero tiene una semilla que colapsa (532 pasos y 64 % de éxito), igual que con Q₀ = 0.
  - **Interpretación:** con Q₀ = 0 el agente explora por optimismo y llega a velocidades altas, que un rango estrecho mezcla en el bin extremo. Con Q₀ = −100 el aprendizaje depende de que el valor del objetivo se propague por las celdas que de verdad se visitan, y **más resolución donde está la mayoría de las visitas** (53 % de celdas visitadas contra 37 %) ayuda más que distinguir las velocidades extremas. Es la misma lógica que hizo preferir tablas más chicas en D-18.
  - Con 5 semillas, la diferencia de los pasos finales (47 pasos, cerca de 1.3 desvíos) es moderada. La de la curva greedy tardía (39 pasos) va en el mismo sentido.
- **Decisión:** se cambia el rango a 5/10 (D-38).
- **Evidencia:** `Acrobot/results/summary_pes_rangos.csv`, `summary_pes_representacion.csv`, las corridas `pes_rangos_*` y `pes_representacion_*`, y las figuras `Acrobot/figures/A2_pes_rangos.png` y `A2_pes_representacion.png`.

### D-37: Figuras de Acrobot (`plots.py`)

- **Fecha:** 2026-11-23 · **Estado:** Vigente
- **Contexto:** el informe necesita figuras claras y comentadas (lo pide la letra), y cada número tiene que poder rastrearse hasta un registro de `results/`.
- **Decisión** (`python plots.py [A1 A2 A3 A4 A5]`, guarda en `Acrobot/figures/` en PNG a 150 dpi):
  - **Solo lee `results/`** (los JSON por corrida, los `_episodes.csv` y los `_test.json`). Nunca usa variables en memoria de un notebook. Si una semilla se corrió más de una vez, toma la última.
  - **Estilo fijo por algoritmo:** Q-Learning con Q₀ = −100 en azul con línea sólida, Dyna-Q en naranja rayada, Q-Learning con Q₀ = 0 en verde agua punteada, Q₀ = −50 en violeta y el agente al azar en gris, en todas las figuras. En los barridos, las variantes toman los colores en orden fijo.
  - **Paleta validada** para daltonismo en pares adyacentes. Además del color, **cada serie tiene su propio estilo de línea**, para que se lea impresa en blanco y negro. Tres colores tienen poco contraste sobre el fondo blanco, y eso se compensa con la leyenda y con las etiquetas directas.
  - **Curvas:** media entre semillas de la media móvil de los pasos de entrenamiento (ventana de 1/40 de los episodios, o 500 en las de 40 000), con una banda de ±1 desvío entre semillas cuando hay pocas curvas. En los barridos se omite la banda, para que no se tapen las curvas, y el desvío se muestra en el panel de barras.
  - **Resultado final:** evaluación greedy sobre los 100 estados fijos (D-09), media ± desvío entre semillas, con la variante elegida en negrita.
  - **Agente al azar:** la referencia es la de los 1000 estados de prueba (693 pasos, 5.8 % de éxito, D-35), no el −700 de D-05.
- **Figuras:**
  - **A1** `A1_curva_modelo_final.png`: `rango_final` / `ql_10000` (D-38; antes, `pes_episodios` con 40 000 episodios). Curva de entrenamiento más los puntos de la evaluación greedy cada 250 episodios.
  - **A2** `A2_pes_*.png` (Q₀ = −100: α, bins, ε, rangos y representación) y `A2_cero_*.png` (Q₀ = 0: rangos, representación, bins, α, γ y ε). Cada figura dice en el título con qué Q₀ se corrió. Las de rangos y representación con Q₀ = −100 salen de R2 (D-36).
  - **A3** `A3_inicializacion.png`: a) misma configuración (`inicializacion`), donde con −100 la curva queda plana en 700 (punto fijo, D-17); b) cada una ajustada (`episodios` 20 000 contra `rango_final` / `ql_10000`, D-38), con el resultado del modelo guardado en la prueba.
  - **A4** `A4_distribucion_prueba.png`: cajas de los pasos en los 1000 estados de prueba de Dyna-Q, Q-Learning con Q₀ = −100, Q-Learning con Q₀ = 0 y el agente al azar, con media, mínimo, máximo y éxito escritos.
  - **A5** `A5_dyna_n.png` (n = 0 a 50 con α = 0.5, mismo lote, así que los tiempos son comparables) y `A5_qlearning_vs_dynaq.png` (cada uno con su configuración final). Las dos tienen un eje x en episodios y otro en tiempo de reloj (`elapsed_s`, D-21), interpolado a una grilla común hasta la semilla más corta.
- **Advertencia para el informe** *(resuelta el 2026-11-23, D-38)*: en la primera versión, `A5_qlearning_vs_dynaq` comparaba lotes distintos (`dyna_n` con n = 0 y `dyna_alpha_bajo`), y el tiempo era orientativo. Ahora usa `rango_final`, donde los dos algoritmos corrieron en el mismo lote, así que los tiempos son comparables. `A5_dyna_n` sigue con el rango anterior (la mitad), porque su objetivo es mostrar el efecto de *n*, y el pie de la figura lo dice.
- **Alternativas:** graficar la curva greedy en vez de la de entrenamiento. Se descartó como curva principal porque son 20 estados cada 250 episodios y es muy ruidosa (D-09). En A1 se muestra como puntos, para que se vea la diferencia entre lo que hace el agente explorando y lo que aprendió.
- **Evidencia:** `Acrobot/figures/`.

### D-38: Rango 5/10 y 10 000 episodios para la configuración final

- **Fecha:** 2026-11-23 · **Estado:** Vigente · **Reemplaza** a D-12 (rango) para Q₀ = −100. Cambia el rango, los episodios y el modelo de D-18, y el rango y el modelo de D-25.
- **Contexto:** al repetir el barrido de rangos con Q₀ = −100 (D-36), el rango del agente entrenado (ω1 ∈ ±5, ω2 ∈ ±10) resultó mejor que la mitad de los límites: 213 contra 259 pasos finales, y 100 % de éxito en las 5 semillas. La mitad se había elegido con Q₀ = 0 (D-12).
- **Decisión 1: rango 5/10** en `DEFAULT_CONFIG` y `DYNAQ_CONFIG` (`train.py`). El resto de la configuración no cambia: ángulos, bins (6, 6, 10, 10), Q₀ = −100, γ = 0.99, ε de 1 a 0.01 con decay 0.999, α = 0.5 para Q-Learning, y n = 2 con α = 0.05 para Dyna-Q.
- **Verificación** (`python sweep.py rango_final`, 15 corridas con 10 procesos):

  | Variante | Pasos finales | Éxito | Pasos en la curva | Tiempo (lote) |
  |---|---|---|---|---|
  | Q-Learning, 10 000 episodios | 212.9 ± 38.8 | 100 % | 353.2 ± 12.1 | 716 s |
  | **Dyna-Q, 10 000 episodios** | **163.4 ± 19.1** | **100 %** | **158.6 ± 6.2** | **428 s** |
  | Q-Learning, 40 000 episodios | 288.4 ± 146.7 | 90 % ± 20 % | 253.3 ± 1.9 | 1530 s |

  - `ql_10000` repite **exactamente** `pes_rangos / entrenado_5_10`, semilla por semilla.
  - **Dyna-Q también mejora con 5/10:** 163.4 ± 19.1 contra 169.5 ± 11.5 en los pasos finales con la mitad (D-25), y 158.6 contra 172.5 en la curva. El rango elegido para Q-Learning le sirve también a Dyna-Q, y la comparación sigue siendo limpia: mismos estados y misma exploración.
- **Hallazgo: con 40 000 episodios, la foto final puede caer en una oscilación.**
  - El modelo de Q-Learning con 40 000 episodios (semilla 0, `qlearning_final_rango_20261123-210910_f304.pkl`) terminó con **583.5 pasos y 51 % de éxito** en la prueba.
  - Su evaluación greedy venía estable cerca de 200 pasos (179, 171, … 234) y salta a 547 justo en la última evaluación, la del episodio 40 000.
  - Es la oscilación de α constante (α = 0.5) ya vista en D-15: una sola actualización puede cambiar la acción greedy en estados clave y hacer que el péndulo quede dando vueltas.
  - Con la mitad del rango, la misma semilla tuvo un salto parecido (589) a mitad del entrenamiento, pero no cayó en la última foto. Las otras 4 semillas de 40 000 terminan entre 176 y 238 pasos.
- **Decisión 2: 10 000 episodios para Q-Learning.**
  - Con 5/10, pasar de 10 000 a 40 000 episodios casi no mejora: la evaluación greedy de los últimos 2000 episodios pasa de unos 231 a unos 200 pasos. A cambio, cuadruplica el tiempo y la cantidad de fotos en las que puede colapsar.
  - Con 10 000 episodios, las 5 semillas llegan al 100 % de éxito y el desvío es menor.
  - Q-Learning y Dyna-Q quedan comparados con el mismo presupuesto, 10 000 episodios cada uno.
  - La elección se basa en la media de las 5 semillas, no en la foto de la semilla 0. Los modelos se entrenan siempre con la semilla 0.
- **Alternativas descartadas:**
  - Mantener 40 000 episodios y entregar el modelo de 51 %, o elegir otra semilla. Elegir la semilla por su resultado sería elegirlo a dedo.
  - Guardar la mejor foto según la evaluación greedy (*checkpoint*). Es una práctica estándar, pero agrega un mecanismo que el curso no vio, y con 20 estados por evaluación la elección es ruidosa.
  - Una α decreciente, que reduciría la oscilación. Es otro hiperparámetro para ajustar, y el resultado con 10 000 episodios ya es bueno.
- **Modelos entregados** (1000 estados de prueba; entrenamiento con un solo proceso):

  | Modelo | Algoritmo | Episodios | Entrenamiento | Pasos | Mín. / Máx. | Éxito |
  |---|---|---|---|---|---|---|
  | **`qlearning_final_rango10k_20261123-211904_5a2f.pkl`** | Q-Learning, α = 0.5 | 10 000 | **294 s** | **172.0** | 86 / 700 | 99.9 % |
  | **`dynaq_final_rango_20261123-211208_4a66.pkl`** | Dyna-Q, n = 2, α = 0.05 | 10 000 | **175 s** | **140.6** | 86 / 700 | 99.9 % |
  | `qlearning_final_pes_20261120-204635_c4c6.pkl` (anterior, mitad) | Q-Learning | 40 000 | 947 s | 218.2 | 121 / 700 | 99.9 % |
  | `dynaq_final_20261122-182339_bbc2.pkl` (anterior, mitad) | Dyna-Q | 10 000 | 593 s | 183.2 | 98 / 476 | 100 % |
  | `qlearning_final_20261120-154018_3f81.pkl` (referencia, Q₀ = 0) | Q-Learning | 20 000 | 303 s | 142.1 | 88 / 700 | 99.6 % |

  Los dos modelos nuevos reproducen exactamente su semilla 0 en `rango_final`: 177.78 y 142.49 pasos en los 100 estados de evaluación.
- **Lectura:**
  - **Con Dyna-Q, el costo de cumplir la indicación del docente casi desaparece:** con Q₀ = −100, Dyna-Q (140.6) queda a la par de Q-Learning con Q₀ = 0 (142.1), con poco más de la mitad del tiempo de entrenamiento. Q-Learning con Q₀ = −100 queda en 172.0.
  - **Dyna-Q entrena más rápido en tiempo de reloj** (175 contra 294 s), aunque hace 3 actualizaciones por paso en vez de 1: aprende antes, y sus episodios se acortan antes. La figura A5 lo muestra en el mismo lote.
  - **Los dos modelos nuevos fallan en 1 de los 1000 estados de prueba** (llegan a 700), contra 0 del Dyna-Q anterior. El peor caso de Dyna-Q empeora (476 → 700), aunque la media mejora mucho (183 → 141). Hay que decirlo en el informe.
  - **Para la defensa:** el rango óptimo depende de la inicialización. Con Q₀ = 0, el agente explora por optimismo y llega a velocidades altas, que un rango estrecho mezcla en el bin extremo. Con Q₀ = −100, el valor del objetivo se propaga por las celdas que se visitan, así que conviene más resolución donde está la mayoría de las visitas.
- **Pendiente:** la medición del *aliasing* de D-23 se hizo con el modelo anterior (la mitad). No cambia la conclusión: con 5/10 las celdas son más finas en las velocidades frecuentes y más gruesas en las extremas, y el modelo determinista sigue siendo inexacto. Si el informe necesita el número actualizado, se puede repetir con `python aliasing.py models/dynaq_final_rango_20261123-211208_4a66.pkl`.
- **Evidencia:** `Acrobot/results/summary_rango_final.csv`, las corridas `rango_final_*`, los `_test.json` de los cinco modelos y las figuras A1, A3, A4 y `A5_qlearning_vs_dynaq` regeneradas.
