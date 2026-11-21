"""Evalúa un modelo guardado sobre estados iniciales de prueba y, si se pide, lo muestra.

Los estados de prueba (semillas desde TEST_SEED_BASE) no se usaron para elegir
hiperparámetros ni modelos, así que miden el rendimiento sin sesgo de selección.

Uso:
    python evaluate_model.py models/<run_id>.pkl              # 1000 episodios de prueba
                                                              # (se guarda en results/<run_id>_test.json)
    python evaluate_model.py models/<run_id>.pkl --render 3   # además, 3 episodios en pantalla
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym

from experiment_log import RESULTS_DIR
from qlearning import load_model
from train import TEST_SEED_BASE, evaluate, make_env, run_episode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--render", type=int, default=0, help="episodios a mostrar en pantalla")
    args = p.parse_args()

    agent, discretizer = load_model(args.model)
    env = make_env(700)
    metrics = evaluate(env, agent, discretizer, args.episodes, seed_base=TEST_SEED_BASE)
    env.close()
    record = {"model": Path(args.model).name, "test_episodes": args.episodes,
              "test_seed_base": TEST_SEED_BASE, **metrics}
    # Queda como evidencia en results/<run_id del modelo>_test.json.
    out = RESULTS_DIR / f"{Path(args.model).stem}_test.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(json.dumps(record, indent=2, ensure_ascii=False))

    if args.render:
        env = gym.make("Acrobot-v1", render_mode="human", max_episode_steps=700)
        for i in range(args.render):
            _, steps, success, _ = run_episode(env, agent, discretizer, learn=False, seed=TEST_SEED_BASE + i)
            print(f"episodio {i}: {steps} pasos, {'llegó' if success else 'no llegó'}")
        env.close()


if __name__ == "__main__":
    main()
