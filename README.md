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
python train.py --episodes 3000 --save-model      # Q-Learning con la configuración por defecto
python train.py --bins 8 --omega1-max 6 --omega2-max 12 --alpha 0.2 --tag prueba
```

`python train.py --help` lista todos los parámetros. Cada corrida queda registrada en `results/` y, con `--save-model`, el modelo queda en `models/<run_id>.pkl`. Para cargarlo: `agent, discretizer = qlearning.load_model(ruta)`.

## Registro de experimentos

Cada corrida se guarda en `results/` del proyecto correspondiente:

- `results/<run_id>.json`: configuración completa (hiperparámetros, discretización, semilla), métricas finales y tiempo de ejecución.
- `results/<run_id>_episodes.csv` (Acrobot): una fila por episodio con recompensa, pasos y si se alcanzó el objetivo.
- `results/runs.csv`: índice con una fila por corrida (configuración resumida y métricas), usado para comparar y graficar.

En Connect Four cada fila de `results/runs.csv` es una partida: agentes, profundidad, heurística, quién empieza, resultado y tiempo por jugada.
