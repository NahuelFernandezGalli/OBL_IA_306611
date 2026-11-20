"""Barridos: corre cada variante de un experimento con varias semillas, en paralelo.

Cada corrida queda en results/ como cualquier otra (con config.experiment y
config.variant para agruparlas) y al final se imprime un resumen por variante:
media y desvío entre semillas de las métricas principales.

Uso:
    python sweep.py rangos                # corre el experimento y resume
    python sweep.py rangos --summary      # solo resume lo que ya está en results/
"""

import argparse
import copy
import csv
import time
from multiprocessing import Pool

import numpy as np

from discretizer import OMEGA1_MAX, OMEGA2_MAX
from experiment_log import RESULTS_DIR
from train import DEFAULT_CONFIG, log_run, run_training

SEEDS = [0, 1, 2, 3, 4]
EPISODES = 5000
WORKERS = 10


def merge(base, overrides):
    """Copia base y pisa las claves de overrides (recursivo en dicts anidados)."""
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def disc(**kwargs):
    return {"discretization": kwargs}


# Cada experimento es una lista de (nombre de la variante, cambios sobre DEFAULT_CONFIG).
EXPERIMENTS = {
    # A: rangos de las velocidades (10 bins por dimensión, ángulos).
    "rangos": [
        ("teorico_12.6_28.3", disc(omega1_max=OMEGA1_MAX, omega2_max=OMEGA2_MAX)),
        ("mitad_6.3_14.1", disc(omega1_max=OMEGA1_MAX / 2, omega2_max=OMEGA2_MAX / 2)),
        ("entrenado_5_10", disc(omega1_max=5.0, omega2_max=10.0)),
        ("azar_3.4_5.9", disc(omega1_max=3.4, omega2_max=5.9)),
    ],
    # B: cantidad de bins, con el rango elegido en A (mitad de los límites teóricos).
    # Las listas son bins por dimensión en el orden (θ1, θ2, ω1, ω2).
    "bins": [
        ("6", disc(bins=6, omega1_max=OMEGA1_MAX / 2, omega2_max=OMEGA2_MAX / 2)),
        ("8", disc(bins=8, omega1_max=OMEGA1_MAX / 2, omega2_max=OMEGA2_MAX / 2)),
        ("10", disc(bins=10, omega1_max=OMEGA1_MAX / 2, omega2_max=OMEGA2_MAX / 2)),
        ("12", disc(bins=12, omega1_max=OMEGA1_MAX / 2, omega2_max=OMEGA2_MAX / 2)),
        ("14", disc(bins=14, omega1_max=OMEGA1_MAX / 2, omega2_max=OMEGA2_MAX / 2)),
        ("angulos12_vel8", disc(bins=[12, 12, 8, 8], omega1_max=OMEGA1_MAX / 2, omega2_max=OMEGA2_MAX / 2)),
        ("angulos8_vel12", disc(bins=[8, 8, 12, 12], omega1_max=OMEGA1_MAX / 2, omega2_max=OMEGA2_MAX / 2)),
    ],
    # C: representación. "raw" discretiza las 6 componentes (cos y sin por separado).
    "representacion": [
        ("angulos_10", disc(representation="angles", bins=10,
                            omega1_max=OMEGA1_MAX / 2, omega2_max=OMEGA2_MAX / 2)),
        ("crudo_4", disc(representation="raw", bins=4, omega1_max=OMEGA1_MAX / 2, omega2_max=OMEGA2_MAX / 2)),
        ("crudo_6", disc(representation="raw", bins=6, omega1_max=OMEGA1_MAX / 2, omega2_max=OMEGA2_MAX / 2)),
        ("crudo_10", disc(representation="raw", bins=10, omega1_max=OMEGA1_MAX / 2, omega2_max=OMEGA2_MAX / 2)),
    ],
}


def run_one(job):
    """Trabajo de un proceso: entrena una variante con una semilla y devuelve el resultado
    (sin el agente, que no hace falta mandar de vuelta)."""
    experiment, variant, seed, overrides = job
    config = merge(DEFAULT_CONFIG, overrides)
    config.update(seed=seed, episodes=EPISODES, experiment=experiment, variant=variant)
    result = run_training(config, verbose=False)
    result.pop("agent")
    return config, result


def run_experiment(name):
    jobs = [(name, variant, seed, overrides)
            for variant, overrides in EXPERIMENTS[name] for seed in SEEDS]
    print(f"{name}: {len(jobs)} corridas ({len(EXPERIMENTS[name])} variantes x {len(SEEDS)} semillas), "
          f"{EPISODES} episodios cada una")
    start = time.perf_counter()
    # Los procesos entrenan; solo el proceso principal escribe en results/,
    # así no hay dos procesos escribiendo runs.csv a la vez.
    with Pool(min(WORKERS, len(jobs))) as pool:
        for config, result in pool.imap_unordered(run_one, jobs):
            log_run(result, config, tag=f"{name}_{config['variant']}_s{config['seed']}")
            m = result["metrics"]
            print(f"  {config['variant']:<22} semilla {config['seed']}: pasos={m['mean_steps']:.1f} "
                  f"exito={m['success_rate']:.0%} ({time.perf_counter() - start:.0f}s)")


SUMMARY_METRICS = ["mean_steps", "success_rate", "curve_mean_steps", "visited_states", "train_time_s"]


def summarize(name):
    """Agrupa por variante las corridas del experimento que hay en runs.csv."""
    with open(RESULTS_DIR / "runs.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("config.experiment") == name]
    variants = {}
    for r in rows:
        variants.setdefault(r["config.variant"], []).append(r)

    out_rows = []
    for variant, _ in EXPERIMENTS.get(name, [(v, None) for v in variants]):
        runs = variants.get(variant, [])
        if not runs:
            continue
        row = {"variant": variant, "n_seeds": len(runs), "n_states": runs[0]["metrics.n_states"]}
        for metric in SUMMARY_METRICS:
            values = np.array([float(r[f"metrics.{metric}"]) for r in runs])
            row[f"{metric}_mean"] = round(float(values.mean()), 3)
            row[f"{metric}_std"] = round(float(values.std()), 3)
        out_rows.append(row)

    with open(RESULTS_DIR / f"summary_{name}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\n{name} (media ± desvío entre semillas)")
    print("| variante | celdas | pasos finales | éxito | pasos en la curva | celdas visitadas | tiempo (s) |")
    print("|---|---|---|---|---|---|---|")
    for r in out_rows:
        def fmt(m, pct=False):
            mean, std = r[f"{m}_mean"], r[f"{m}_std"]
            return f"{mean:.0%} ± {std:.0%}" if pct else f"{mean:.1f} ± {std:.1f}"
        print(f"| {r['variant']} | {r['n_states']} | {fmt('mean_steps')} | {fmt('success_rate', True)} | "
              f"{fmt('curve_mean_steps')} | {fmt('visited_states', True)} | {fmt('train_time_s')} |")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("experiment", choices=sorted(EXPERIMENTS))
    p.add_argument("--summary", action="store_true", help="solo resumir, sin correr")
    args = p.parse_args()
    if not args.summary:
        run_experiment(args.experiment)
    summarize(args.experiment)
