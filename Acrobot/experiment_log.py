"""Registro de experimentos (formato descrito en el README y en D-03).

Cada corrida genera:
- results/<run_id>.json: configuración completa, métricas y tiempo.
- results/<run_id>_episodes.csv: una fila por episodio (opcional).
- una fila en results/runs.csv, el índice para comparar corridas.
"""

import csv
import json
import uuid
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def new_run_id(prefix):
    """Identificador único y legible: <prefix>_<fecha-hora>_<4 hex>."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:4]}"


def _flatten(d, parent=""):
    """Aplana un dict anidado: {"a": {"b": 1}} -> {"a.b": 1}. Las listas pasan a texto."""
    flat = {}
    for key, value in d.items():
        name = f"{parent}.{key}" if parent else key
        if isinstance(value, dict):
            flat.update(_flatten(value, name))
        elif isinstance(value, (list, tuple)):
            flat[name] = json.dumps(value)
        else:
            flat[name] = value
    return flat


def _append_index_row(path, row):
    """Agrega una fila a runs.csv. Si aparecen columnas nuevas, reescribe el archivo
    con la unión de columnas (las filas viejas quedan vacías en esas columnas)."""
    rows, columns = [], []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            columns = list(reader.fieldnames or [])
            rows = list(reader)
    new_columns = [c for c in row if c not in columns]
    if path.exists() and not new_columns:
        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=columns).writerow(row)
        return
    columns += new_columns
    rows.append(row)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_run(run_id, config, metrics, elapsed_s, episodes=None, extra=None, results_dir=RESULTS_DIR):
    """Guarda una corrida completa.

    config: hiperparámetros, discretización, semilla, etc. (puede ser anidado).
    metrics: métricas finales (valores simples, van también al índice).
    episodes: lista de dicts, una por episodio, todas con las mismas claves.
    extra: datos que van solo al JSON (por ejemplo, la curva de evaluación).
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")

    record = {
        "run_id": run_id,
        "timestamp": timestamp,
        "elapsed_s": round(elapsed_s, 2),
        "config": config,
        "metrics": metrics,
    }
    if extra:
        record.update(extra)
    with open(results_dir / f"{run_id}.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    if episodes:
        with open(results_dir / f"{run_id}_episodes.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(episodes[0]))
            writer.writeheader()
            writer.writerows(episodes)

    row = {"run_id": run_id, "timestamp": timestamp, "elapsed_s": round(elapsed_s, 2)}
    row.update(_flatten(config, "config"))
    row.update(_flatten(metrics, "metrics"))
    _append_index_row(results_dir / "runs.csv", row)
    return results_dir / f"{run_id}.json"
