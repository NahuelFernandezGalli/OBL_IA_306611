# OBL_IA_306611

Obligatorio de Inteligencia Artificial (ORT, Agosto 2026).

- **ACROBAT** (`Acrobot/`): Q-Learning y Dyna-Q tabulares sobre `Acrobot-v1` de Gymnasium.
- **COUNTER** (`ConnectFour/`): Minimax con poda Alpha-Beta y Expectimax sobre Connect Four con regla de captura.

Las decisiones técnicas y su justificación están en [`docs/DECISIONES.md`](docs/DECISIONES.md).

## Estructura

```
Acrobot/
├── agent.ipynb        notebook base de la cátedra
├── experiment_log.py  registro de corridas en results/
├── baseline.py        línea base: agente al azar
├── discretizer.py     discretización de la observación
├── qlearning.py       agente Q-Learning tabular, guardar/cargar modelos
├── train.py           entrenamiento y evaluación greedy
├── sweep.py           barridos con varias semillas, en paralelo
├── evaluate_model.py  evaluación de un modelo sobre estados de prueba (y render)
├── models/            modelos entrenados (.pkl)
├── results/           registros de experimentos (JSON/CSV)
└── figures/           gráficos para el informe
ConnectFour/
├── board.py, agent.py, connect_four_env.py, play.py, stratego_agent.py   (cátedra)
├── Connect_four.ipynb notebook base de la cátedra
├── experiment_log.py  registro de series en results/ (copia del de Acrobot)
├── heuristics.py      funciones de evaluación (componentes y combinaciones)
├── search_base.py     base común de los agentes de búsqueda
├── minimax_agent.py   Minimax con o sin poda Alpha-Beta
├── expectimax_agent.py Expectimax (rival como nodo de azar uniforme)
├── random_agent.py    rival al azar
├── match.py           partidas y series contra Stratego, Random u otros agentes
├── test_search.py     verificaciones de la búsqueda y de las heurísticas
├── experiments.py     experimentos (E1: impacto de la poda Alpha-Beta)
├── results/
└── figures/
```

Cada proyecto es un entorno Poetry independiente (`pyproject.toml` + `poetry.lock`), con el entorno virtual dentro de la carpeta (`.venv/`).

## Instalación

Requiere Python **3.9 a 3.12** (los proyectos no soportan 3.13+) y [Poetry](https://python-poetry.org/) 2.x.

```bash
cd Acrobot
poetry env use <ruta a python 3.11 o 3.12>
poetry install
```

Lo mismo en `ConnectFour/`. Para los notebooks, seleccionar como kernel el `.venv` de cada carpeta.

## Uso (Acrobot)

Desde `Acrobot/`, con el entorno activado (o anteponiendo `poetry run`):

```bash
python baseline.py --episodes 50                  # línea base al azar
python train.py --save-model                      # Q-Learning con la configuración final (10000 episodios)
python train.py --algorithm dynaq --save-model    # Dyna-Q con la configuración final
python evaluate_model.py --random                 # agente al azar sobre los 1000 estados de prueba
python plots.py                                   # figuras en figures/, leídas de results/
python train.py --bins 10 --omega1-max 6 --omega2-max 12 --alpha 0.2 --tag prueba
python sweep.py bins                              # un experimento: variantes x 5 semillas
python sweep.py bins --summary                    # solo la tabla resumen de lo ya corrido
python evaluate_model.py models/<run_id>.pkl --render 3   # evaluar y ver el modelo
```

Modelos entregados, los dos con inicialización pesimista Q₀ = −100 por indicación del docente. Los números son sobre 1000 estados de prueba:

- Q-Learning: `models/qlearning_final_rango10k_20261123-211904_5a2f.pkl` (172 pasos medios y 99.9 % de éxito).
- Dyna-Q: `models/dynaq_final_rango_20261123-211208_4a66.pkl` (141 pasos y 99.9 %).

`models/qlearning_final_20261120-154018_3f81.pkl` es la referencia con Q₀ = 0 (142 pasos y 99.6 %). Los demás modelos de `models/` son versiones anteriores (D-38). El porqué de cada decisión está en `docs/DECISIONES.md`.

`python train.py --help` lista todos los parámetros. Dyna-Q: `python train.py --algorithm dynaq --n-planning 10`. Verificaciones de Dyna-Q: `python test_dynaq.py`. Cada corrida queda registrada en `results/` y, con `--save-model`, el modelo queda en `models/<run_id>.pkl`. Para cargarlo: `agent, discretizer = qlearning.load_model(ruta)`.

## Uso (Connect Four)

Desde `ConnectFour/`:

```bash
python test_search.py                                              # verificaciones
python experiments.py poda                                         # E1: nodos y tiempo con y sin poda (unos 5 min)
python experiments.py aperturas                                    # victorias forzadas del tablero vacío y de las aperturas
python experiments.py heuristicas                                  # E2: 10 heurísticas contra Stratego (unos 4 min)
python experiments.py profundidad                                  # E3: profundidades 1 a 8 contra Stratego
python experiments.py profundidad_vacio                            # E3 desde el tablero vacío
python experiments.py expectimax_stratego                          # E4 (también expectimax_random y minimax_vs_expectimax)
python match.py --depth 4 --heuristic completa --rival stratego --openings 1
python match.py --ours expectimax --depth 3 --rival random --openings 2
python match.py --no-pruning --depth 4 --rival stratego --seats primero
```

El rival oficial es `Stratego(2, 4)`, construido como en `play.py` (D-29). `--openings 1` o `2` juega las 7 o 49 aperturas forzadas, y cada turno es una serie aparte.

## Registro de experimentos

Cada corrida se guarda en `results/` del proyecto correspondiente:

- `results/<run_id>.json`: configuración completa (hiperparámetros, discretización, semilla), métricas finales y tiempo de ejecución.
- `results/<run_id>_episodes.csv` (Acrobot): una fila por episodio con recompensa, pasos, si se alcanzó el objetivo, ε y `elapsed_s` (tiempo de entrenamiento acumulado, sin las evaluaciones; existe desde la fase 2, D-21).
- `results/runs.csv`: índice con una fila por corrida (configuración resumida y métricas), usado para comparar y graficar.

En Connect Four cada corrida es una **serie** de partidas (D-26): `results/<run_id>.json` (agentes, profundidad, heurística, turno, aperturas y métricas), `results/<run_id>_games.csv` (una fila por partida: resultado, jugadas, tiempo y nodos por jugada) y una fila por serie en `results/runs.csv`.
