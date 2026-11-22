"""Mide el aliasing de la discretización: para un mismo par (celda, acción), ¿a cuántas
celdas distintas se llega? (hipótesis de D-20 y D-22).

El modelo de Dyna-Q (Sutton y Barto 8.2) guarda una sola transición por (s, a). Si el par
lleva a varias celdas, esa transición es solo una de las posibles. Se juega con la política
de un modelo guardado más ε-greedy, se cuentan las transiciones reales y se calcula:
- cuántas celdas s' distintas tiene cada par;
- la probabilidad de que la transición guardada coincida con la próxima real: si el par lleva
  a s'_i con probabilidad p_i, la guardada es s'_i con probabilidad p_i y la próxima también,
  así que coinciden con probabilidad Σ p_i² (1 si el par es determinista);
- cuántas transiciones se quedan en la misma celda (s' = s).

Uso: python aliasing.py models/<run_id>.pkl [--episodes 200 --epsilon 0.1]
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiment_log import RESULTS_DIR
from qlearning import load_model
from train import make_env

# Semillas distintas de las de entrenamiento, evaluación (1 000 000) y prueba (2 000 000).
ALIASING_SEED_BASE = 3_000_000
MIN_VISITS = 5  # pares con menos visitas no alcanzan para estimar a dónde llevan


def collect(agent, discretizer, n_episodes, epsilon):
    """Juega n_episodes con ε-greedy y cuenta, para cada (s, a), los (s', terminal) observados."""
    agent.epsilon = epsilon
    env = make_env(700)
    counts = defaultdict(Counter)
    for i in range(n_episodes):
        obs, _ = env.reset(seed=ALIASING_SEED_BASE + i)
        s = discretizer(obs)
        terminated = truncated = False
        while not (terminated or truncated):
            a = agent.act(s)
            obs, _, terminated, truncated, _ = env.step(a)
            s_next = discretizer(obs)
            counts[(s, a)][(s_next, bool(terminated))] += 1
            s = s_next
    env.close()
    return counts


def summarize(counts):
    pairs = {sa: c for sa, c in counts.items() if sum(c.values()) >= MIN_VISITS}
    visits = np.array([sum(c.values()) for c in pairs.values()])
    distinct = np.array([len(c) for c in pairs.values()])
    match = np.array([sum((n / sum(c.values())) ** 2 for n in c.values()) for c in pairs.values()])
    total = sum(sum(c.values()) for c in counts.values())
    self_loops = sum(n for (s, _), c in counts.items() for (s2, _), n in c.items() if s2 == s)
    # Pares que a veces se quedan en la celda y a veces salen: el modelo puede guardar
    # "me quedo" y el planning repite una transición que no acerca al objetivo.
    mixed_loop = np.mean([any(s2 == sa[0] for s2, _ in c) and any(s2 != sa[0] for s2, _ in c)
                          for sa, c in pairs.items()])
    # Pares que a veces llegan al objetivo y a veces no.
    mixed_terminal = sum(1 for c in counts.values() if len({t for _, t in c}) == 2)
    return {
        "transitions": int(total),
        "pairs_observed": len(counts),
        "pairs_min_visits": len(pairs),
        "min_visits": MIN_VISITS,
        "frac_pairs_more_than_one_next": float(np.mean(distinct > 1)),
        "mean_distinct_next": float(distinct.mean()),
        "max_distinct_next": int(distinct.max()),
        # Media de Σ p² pesada por visitas: la probabilidad de que, en un paso real cualquiera,
        # la transición guardada en el modelo coincida con la que de verdad ocurre.
        "match_prob_weighted": float(np.average(match, weights=visits)),
        "match_prob_unweighted": float(match.mean()),
        "frac_self_loop_transitions": float(self_loops / total),
        "frac_pairs_mixed_self_loop": float(mixed_loop),
        "pairs_mixed_terminal": int(mixed_terminal),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--epsilon", type=float, default=0.1)
    args = p.parse_args()

    agent, discretizer = load_model(args.model)
    counts = collect(agent, discretizer, args.episodes, args.epsilon)
    record = {"model": Path(args.model).name, "episodes": args.episodes, "epsilon": args.epsilon,
              "seed_base": ALIASING_SEED_BASE, **summarize(counts)}
    out = RESULTS_DIR / f"{Path(args.model).stem}_aliasing.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
