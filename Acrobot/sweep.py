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
from train import log_run, run_training

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


def agent(**kwargs):
    return {"agent": kwargs}


# Base congelada de los experimentos: la configuración inicial de train.py (10 bins,
# rangos teóricos, α = 0.1, γ = 0.99, ε de 1 a 0.05 con decay 0.995). No depende de
# DEFAULT_CONFIG, que cambia a medida que se toman decisiones, para que cada
# experimento se pueda volver a correr y dé exactamente los mismos números.
BASE = {
    "algorithm": "qlearning",
    "seed": 0,
    "episodes": EPISODES,
    "max_steps": 700,
    "discretization": {"representation": "angles", "bins": 10,
                       "omega1_max": OMEGA1_MAX, "omega2_max": OMEGA2_MAX},
    "agent": {"alpha": 0.1, "gamma": 0.99, "epsilon_start": 1.0,
              "epsilon_min": 0.05, "epsilon_decay": 0.995},
    "eval": {"every": 250, "episodes": 20, "final_episodes": 100},
}

# Discretización elegida en los experimentos A-C (D-12 y D-13).
CHOSEN_DISC = disc(bins=[8, 8, 12, 12], omega1_max=OMEGA1_MAX / 2, omega2_max=OMEGA2_MAX / 2)


def chosen(**agent_kwargs):
    """Discretización elegida más cambios en el agente."""
    return merge(CHOSEN_DISC, agent(**agent_kwargs))


# Configuración final de Q-Learning con Q0 = -100 (D-18), con 10000 episodios.
FINAL_PES = merge(CHOSEN_DISC, merge(disc(bins=[6, 6, 10, 10]), merge(
    agent(alpha=0.5, gamma=0.99, epsilon_decay=0.999, epsilon_min=0.01, q_init=-100.0), {"episodes": 10000})))


# FINAL_PES con el rango de velocidades del agente entrenado (5 / 10), D-38.
FINAL_RANGO = merge(FINAL_PES, disc(omega1_max=5.0, omega2_max=10.0))


def dyna(n, **overrides):
    """Dyna-Q con n pasos de planning sobre FINAL_PES, más otros cambios."""
    return merge(FINAL_PES, merge({"algorithm": "dynaq", "agent": {"n_planning": n}}, overrides))


# Cada experimento es una lista de (nombre de la variante, cambios sobre BASE).
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
    # Hiperparámetros, con la discretización elegida.
    # Se cambia uno por vez respecto de α = 0.1, γ = 0.99, ε: 1 -> 0.05 con decay 0.995.
    "alpha": [(str(a), chosen(alpha=a)) for a in [0.05, 0.1, 0.2, 0.3, 0.5]],
    "gamma": [(str(g), chosen(gamma=g)) for g in [0.9, 0.95, 0.99, 0.999, 1.0]],
    # decay 0.99 / 0.995 / 0.999: ε llega al piso de 0.05 cerca del episodio 300 / 600 / 3000.
    "epsilon": [
        ("decay0.99_min0.05", chosen(epsilon_decay=0.99, epsilon_min=0.05)),
        ("decay0.995_min0.05", chosen(epsilon_decay=0.995, epsilon_min=0.05)),
        ("decay0.999_min0.05", chosen(epsilon_decay=0.999, epsilon_min=0.05)),
        ("decay0.995_min0", chosen(epsilon_decay=0.995, epsilon_min=0.0)),
        ("decay0.995_min0.01", chosen(epsilon_decay=0.995, epsilon_min=0.01)),
        ("decay0.995_min0.1", chosen(epsilon_decay=0.995, epsilon_min=0.1)),
    ],
    # Combinación de los dos mejores valores de cada barrido anterior (interacciones).
    # Con decay 0.9995, ε todavía vale ~0.08 al episodio 5000 (no llega al piso).
    "combinacion": [
        (f"a{a}_d{d}_g{g}", chosen(alpha=a, epsilon_decay=d, gamma=g))
        for a in [0.1, 0.2] for d in [0.999, 0.9995] for g in [0.99, 0.999]
    ],
    # Cantidad de episodios, con la mejor combinación (α = 0.1, γ = 0.99, decay 0.999).
    "episodios": [
        (str(n), merge(chosen(alpha=0.1, gamma=0.99, epsilon_decay=0.999), {"episodes": n}))
        for n in [5000, 10000, 20000]
    ],
    # Inicialización de la tabla Q (indicación del docente: sin inicialización optimista).
    # Con recompensa -1 por paso, todo valor real está entre -1/(1-γ) = -100 y 0:
    # Q0 = 0 es optimista, -50 es intermedio y -100 es pesimista (el peor valor posible).
    "inicializacion": [
        (str(q), chosen(alpha=0.1, gamma=0.99, epsilon_decay=0.999, q_init=q))
        for q in [0.0, -50.0, -100.0]
    ],
    # ¿La inicialización pesimista aprende si se le da más tiempo y más exploración?
    "pesimista_larga": [
        (f"d{d}", merge(chosen(alpha=0.1, gamma=0.99, epsilon_decay=d, q_init=-100.0), {"episodes": 20000}))
        for d in [0.999, 0.9998]
    ],
    # Re-ajuste con Q0 = -100 (decisión del 2026-11-20, D-17). Con inicialización pesimista
    # el aprendizaje depende de que el valor de las llegadas al objetivo se propague hacia
    # atrás, así que los hiperparámetros elegidos con Q0 = 0 pueden no servir.
    # Presupuesto: 10000 episodios. Se ajusta uno por vez, en este orden: α, bins, ε.
    "pes_alpha": [
        (str(a), merge(chosen(alpha=a, gamma=0.99, epsilon_decay=0.999, q_init=-100.0), {"episodes": 10000}))
        for a in [0.1, 0.2, 0.3, 0.5]
    ],
    # Continuación: en pes_alpha el resultado mejoraba hasta el valor más alto probado.
    "pes_alpha_alto": [
        (str(a), merge(chosen(alpha=a, gamma=0.99, epsilon_decay=0.999, q_init=-100.0), {"episodes": 10000}))
        for a in [0.7, 0.9, 1.0]
    ],
    # Bins con Q0 = -100 y α = 0.5 (elegido en pes_alpha). Con inicialización pesimista,
    # una tabla más chica tiene menos celdas por las que propagar el valor del objetivo.
    "pes_bins": [
        (name, merge(CHOSEN_DISC, merge(disc(bins=b), merge(
            agent(alpha=0.5, gamma=0.99, epsilon_decay=0.999, q_init=-100.0), {"episodes": 10000}))))
        for name, b in [("6", 6), ("6_6_10_10", [6, 6, 10, 10]), ("8", 8), ("8_8_12_12", [8, 8, 12, 12])]
    ],
    # Continuación: en pes_bins mejoraba hasta la tabla más chica probada (6 bins).
    "pes_bins_gruesos": [
        (name, merge(CHOSEN_DISC, merge(disc(bins=b), merge(
            agent(alpha=0.5, gamma=0.99, epsilon_decay=0.999, q_init=-100.0), {"episodes": 10000}))))
        for name, b in [("4", 4), ("5", 5)]
    ],
    # ε con Q0 = -100, α = 0.5 y bins (6, 6, 10, 10) (elegidos en pes_alpha y pes_bins).
    "pes_epsilon": [
        (name, merge(CHOSEN_DISC, merge(disc(bins=[6, 6, 10, 10]), merge(
            agent(alpha=0.5, gamma=0.99, epsilon_decay=d, epsilon_min=m, q_init=-100.0), {"episodes": 10000}))))
        for name, d, m in [("decay0.99_min0.05", 0.99, 0.05), ("decay0.995_min0.05", 0.995, 0.05),
                           ("decay0.999_min0.05", 0.999, 0.05), ("decay0.999_min0.01", 0.999, 0.01)]
    ],
    # Cantidad de episodios con la configuración final para Q0 = -100. La de 10000 episodios
    # es la variante decay0.999_min0.01 de pes_epsilon (mismas semillas, mismos números).
    "pes_episodios": [
        (str(n), merge(CHOSEN_DISC, merge(disc(bins=[6, 6, 10, 10]), merge(
            agent(alpha=0.5, gamma=0.99, epsilon_decay=0.999, epsilon_min=0.01, q_init=-100.0), {"episodes": n}))))
        for n in [20000, 40000]
    ],
    # R2 (fase 4, D-36): rangos y representación se habían elegido con Q0 = 0 (D-12, D-13).
    # Se repiten con la configuración final de Q0 = -100 (FINAL_PES), con las mismas variantes,
    # para que todos los barridos sean consistentes con el modelo entregado. La variante que
    # coincide con FINAL_PES tiene que repetir exactamente pes_epsilon (decay0.999_min0.01).
    "pes_rangos": [
        ("teorico_12.6_28.3", merge(FINAL_PES, disc(omega1_max=OMEGA1_MAX, omega2_max=OMEGA2_MAX))),
        ("mitad_6.3_14.1", FINAL_PES),
        ("entrenado_5_10", merge(FINAL_PES, disc(omega1_max=5.0, omega2_max=10.0))),
        ("azar_3.4_5.9", merge(FINAL_PES, disc(omega1_max=3.4, omega2_max=5.9))),
    ],
    # "crudo" discretiza las 6 componentes; crudo_4 (4096 celdas) tiene casi el mismo tamaño
    # que la tabla de ángulos final (3600), así que es la comparación más justa.
    "pes_representacion": [
        ("angulos_6_6_10_10", FINAL_PES),
        ("crudo_4", merge(FINAL_PES, disc(representation="raw", bins=4))),
        ("crudo_6", merge(FINAL_PES, disc(representation="raw", bins=6))),
        ("crudo_10", merge(FINAL_PES, disc(representation="raw", bins=10))),
    ],
    # Configuración final con el rango 5/10 (D-38), elegido en pes_rangos. Q-Learning y Dyna-Q
    # (n = 2, α = 0.05) en el mismo lote, para que los tiempos de reloj sean comparables (figura
    # A5), y Q-Learning con 40000 episodios para la curva del modelo final (A1). ql_10000 tiene que
    # repetir exactamente pes_rangos / entrenado_5_10.
    "rango_final": [
        ("ql_10000", FINAL_RANGO),
        ("dynaq_10000", merge(FINAL_RANGO, {"algorithm": "dynaq", "agent": {"n_planning": 2, "alpha": 0.05}})),
        ("ql_40000", merge(FINAL_RANGO, {"episodes": 40000})),
    ],
    # Dyna-Q (fase 2, D-20 y D-21): pasos de planning n, sobre la configuración final de
    # Q-Learning (Q0 = -100) y el mismo presupuesto de 10000 episodios que los barridos pes_*.
    # n = 0 es Q-Learning: tiene que repetir los números de pes_epsilon (decay0.999_min0.01) y,
    # al correr en el mismo lote, da tiempos de reloj comparables con los de Dyna-Q.
    "dyna_n": [(str(n), dyna(n)) for n in [0, 2, 5, 10, 25, 50]],
    # α con Dyna-Q (D-22). Con planning el valor del objetivo se propaga rápido, así que un α
    # menor que 0.5 podría promediar mejor el aliasing. n = 0 (Q-Learning con los mismos α)
    # separa el efecto de α del efecto del planning. α = 0.5 ya está en dyna_n.
    "dyna_alpha": [(f"n{n}_a{a}", dyna(n, agent={"alpha": a}))
                   for n in [0, 2, 10] for a in [0.1, 0.2, 0.3]],
    # Continuación: en dyna_alpha el mejor α con planning fue el más bajo probado (0.1).
    # Se baja α y se agregan n intermedios y grandes, para ver si con α bajo el aliasing
    # deja de castigar a los n grandes (D-24).
    "dyna_alpha_bajo": [(f"n{n}_a{a}", dyna(n, agent={"alpha": a}))
                        for n, a in [(2, 0.05), (5, 0.05), (5, 0.1), (10, 0.05), (25, 0.05), (25, 0.1)]],
    # Última continuación: el mejor α (0.05, con n = 2) seguía en el borde de la grilla.
    # n = 1 es el mínimo de planning posible (D-24).
    "dyna_alpha_minimo": [(f"n{n}_a{a}", dyna(n, agent={"alpha": a}))
                          for n, a in [(1, 0.02), (1, 0.03), (1, 0.05), (2, 0.02), (2, 0.03)]],
}


def run_one(job):
    """Trabajo de un proceso: entrena una variante con una semilla y devuelve el resultado
    (sin el agente, que no hace falta mandar de vuelta)."""
    experiment, variant, seed, overrides = job
    config = merge(BASE, overrides)
    # Presupuesto fijo de EPISODES, salvo que la variante defina el suyo.
    config.update(seed=seed, episodes=overrides.get("episodes", EPISODES),
                  experiment=experiment, variant=variant)
    result = run_training(config, verbose=False)
    result.pop("agent")
    return config, result


def run_experiment(name):
    jobs = [(name, variant, seed, overrides)
            for variant, overrides in EXPERIMENTS[name] for seed in SEEDS]
    print(f"{name}: {len(jobs)} corridas ({len(EXPERIMENTS[name])} variantes x {len(SEEDS)} semillas)")
    start = time.perf_counter()
    # Los procesos entrenan; solo el proceso principal escribe en results/,
    # así no hay dos procesos escribiendo runs.csv a la vez.
    with Pool(min(WORKERS, len(jobs))) as pool:
        for config, result in pool.imap_unordered(run_one, jobs):
            log_run(result, config, tag=f"{name}_{config['variant']}_s{config['seed']}")
            m = result["metrics"]
            print(f"  {config['variant']:<22} semilla {config['seed']}: pasos={m['mean_steps']:.1f} "
                  f"exito={m['success_rate']:.0%} ({time.perf_counter() - start:.0f}s)")


SUMMARY_METRICS = ["mean_steps", "success_rate", "curve_mean_steps", "visited_states", "train_time_s",
                   "learn_time_s"]


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
            if not runs[0].get(f"metrics.{metric}"):
                continue  # métrica que no existía cuando se corrió el experimento (learn_time_s)
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
