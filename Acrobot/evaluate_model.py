"""Evalúa un modelo guardado sobre estados iniciales de prueba y, si se pide, lo muestra.

Los estados de prueba (semillas desde TEST_SEED_BASE) no se usaron para elegir
hiperparámetros ni modelos, así que miden el rendimiento sin sesgo de selección.
El JSON guarda, además de los agregados, los pasos y el éxito de cada estado (D-35).

Uso:
    python evaluate_model.py models/<run_id>.pkl              # 1000 episodios de prueba
                                                              # (se guarda en results/<run_id>_test.json)
    python evaluate_model.py models/<run_id>.pkl --render 3   # además, 3 episodios en pantalla
    python evaluate_model.py --random                         # agente al azar sobre los mismos estados
                                                              # (se guarda en results/random_test.json)
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from experiment_log import RESULTS_DIR
from qlearning import load_model
from train import DEFAULT_CONFIG, TEST_SEED_BASE, build, evaluate, make_env, run_episode


class RandomAgent:
    """Elige una de las 3 acciones al azar, ignore o no el estado. Misma interfaz que
    QLearningAgent.act, para evaluarlo con la misma función y los mismos estados."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, s, greedy=False):
        return int(self.rng.integers(3))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", nargs="?")
    p.add_argument("--random", action="store_true", help="evaluar el agente al azar")
    p.add_argument("--seed", type=int, default=0, help="semilla del agente al azar")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--render", type=int, default=0, help="episodios a mostrar en pantalla")
    args = p.parse_args()
    if args.random == bool(args.model):
        p.error("pasar un modelo o --random (uno de los dos)")

    if args.random:
        # El discretizador no influye en las acciones; se usa el de la configuración final
        # solo porque run_episode lo pide.
        _, discretizer = build(DEFAULT_CONFIG)
        agent, name, stem = RandomAgent(args.seed), "random", "random"
    else:
        agent, discretizer = load_model(args.model)
        name, stem = Path(args.model).name, Path(args.model).stem

    env = make_env(700)
    metrics = evaluate(env, agent, discretizer, args.episodes, seed_base=TEST_SEED_BASE,
                       per_episode=True)
    env.close()
    record = {"model": name, "test_episodes": args.episodes, "test_seed_base": TEST_SEED_BASE}
    if args.random:
        record["random_seed"] = args.seed
    record.update(metrics)
    # Queda como evidencia en results/<run_id del modelo>_test.json.
    out = RESULTS_DIR / f"{stem}_test.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    summary = {k: v for k, v in record.items() if not k.endswith("_per_episode")}
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.render:
        env = gym.make("Acrobot-v1", render_mode="human", max_episode_steps=700)
        for i in range(args.render):
            _, steps, success, _ = run_episode(env, agent, discretizer, learn=False, seed=TEST_SEED_BASE + i)
            print(f"episodio {i}: {steps} pasos, {'llegó' if success else 'no llegó'}")
        env.close()


if __name__ == "__main__":
    main()
