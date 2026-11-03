# OBL_IA_306611

Obligatorio de Inteligencia Artificial (ORT, Agosto 2026).

- **ACROBAT** (`Acrobot/`): Q-Learning y Dyna-Q tabulares sobre `Acrobot-v1` de Gymnasium.
- **COUNTER** (`ConnectFour/`): Minimax con poda Alpha-Beta y Expectimax sobre Connect Four con regla de captura.

## Estructura

```
Acrobot/
├── agent.ipynb        notebook base de la cátedra
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

## Registro de experimentos

Cada corrida se guarda en `results/` del proyecto correspondiente:

- `results/<run_id>.json`: configuración completa (hiperparámetros, discretización, semilla), métricas finales y tiempo de ejecución.
- `results/<run_id>_episodes.csv` (Acrobot): una fila por episodio con recompensa, pasos y si se alcanzó el objetivo.
- `results/runs.csv`: índice con una fila por corrida (configuración resumida y métricas), usado para comparar y graficar.

En Connect Four cada fila de `results/runs.csv` es una partida: agentes, profundidad, heurística, quién empieza, resultado y tiempo por jugada.
