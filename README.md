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
├── board.py, agent.py, connect_four_env.py, play.py, stratego_agent.py
├── Connect_four.ipynb notebook base de la cátedra
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
python train.py --save-model                      # Q-Learning con la configuración final (40000 episodios)
python train.py --bins 10 --omega1-max 6 --omega2-max 12 --alpha 0.2 --tag prueba
python sweep.py bins                              # un experimento: variantes x 5 semillas
python sweep.py bins --summary                    # solo la tabla resumen de lo ya corrido
python evaluate_model.py models/<run_id>.pkl --render 3   # evaluar y ver el modelo
```

Modelo final de Q-Learning: `models/qlearning_final_pes_20261120-204635_c4c6.pkl`, con inicialización pesimista Q₀ = −100 por indicación del docente (218 pasos medios y 99.9 % de éxito sobre 1000 estados de prueba). `models/qlearning_final_20261120-154018_3f81.pkl` es la referencia con Q₀ = 0 (142 pasos y 99.6 %). El porqué de cada decisión está en `docs/DECISIONES.md`.

`python train.py --help` lista todos los parámetros. Dyna-Q: `python train.py --algorithm dynaq --n-planning 10`. Verificaciones de Dyna-Q: `python test_dynaq.py`. Cada corrida queda registrada en `results/` y, con `--save-model`, el modelo queda en `models/<run_id>.pkl`. Para cargarlo: `agent, discretizer = qlearning.load_model(ruta)`.

## Registro de experimentos

Cada corrida se guarda en `results/` del proyecto correspondiente:

- `results/<run_id>.json`: configuración completa (hiperparámetros, discretización, semilla), métricas finales y tiempo de ejecución.
- `results/<run_id>_episodes.csv` (Acrobot): una fila por episodio con recompensa, pasos, si se alcanzó el objetivo, ε y `elapsed_s` (tiempo de entrenamiento acumulado, sin las evaluaciones; existe desde la fase 2, D-21).
- `results/runs.csv`: índice con una fila por corrida (configuración resumida y métricas), usado para comparar y graficar.

En Connect Four cada fila de `results/runs.csv` es una partida: agentes, profundidad, heurística, quién empieza, resultado y tiempo por jugada.
