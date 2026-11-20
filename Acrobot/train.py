"""Entrenamiento y evaluación de agentes tabulares sobre Acrobot.

- Entrena con ε-greedy y cada `eval.every` episodios evalúa la política greedy
  (ε = 0, sin aprender) sobre estados iniciales fijos, para la curva de aprendizaje.
- Al final hace una evaluación greedy más larga, que es la métrica para comparar corridas.
- Guarda la corrida en results/ y, si se pide, el modelo en models/.

Uso: python train.py --episodes 3000 --bins 10 --alpha 0.1 --gamma 0.99 --save-model
"""

import argparse
import copy
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from discretizer import OMEGA1_MAX, OMEGA2_MAX, make_discretizer
from experiment_log import new_run_id, save_run
from qlearning import QLearningAgent, save_model

MODELS_DIR = Path(__file__).parent / "models"
# Semillas de evaluación: fijas y distintas de las de entrenamiento, para que todas las
# configuraciones se comparen sobre los mismos estados iniciales.
EVAL_SEED_BASE = 1_000_000

DEFAULT_CONFIG = {
    "algorithm": "qlearning",
    "seed": 0,
    "episodes": 3000,
    "max_steps": 700,
    "discretization": {"representation": "angles", "bins": 10,
                       "omega1_max": OMEGA1_MAX, "omega2_max": OMEGA2_MAX},
    "agent": {"alpha": 0.1, "gamma": 0.99, "epsilon_start": 1.0,
              "epsilon_min": 0.05, "epsilon_decay": 0.995},
    "eval": {"every": 250, "episodes": 20, "final_episodes": 100},
}


def make_env(max_steps):
    return gym.make("Acrobot-v1", render_mode=None, max_episode_steps=max_steps)


def run_episode(env, agent, discretizer, learn, seed=None):
    """Juega un episodio. Si learn=True actúa con ε-greedy y actualiza Q; si no, actúa greedy.
    Devuelve (recompensa total, pasos, éxito, velocidades observadas)."""
    obs, _ = env.reset(seed=seed)
    s = discretizer(obs)
    total, steps, terminated, truncated = 0.0, 0, False, False
    velocities = []
    while not (terminated or truncated):
        a = agent.act(s, greedy=not learn)
        obs, r, terminated, truncated, _ = env.step(a)
        s_next = discretizer(obs)
        if learn:
            agent.update(s, a, r, s_next, terminated)
        velocities.append(obs[4:6])
        s = s_next
        total += r
        steps += 1
    return total, steps, terminated, velocities


def evaluate(env, agent, discretizer, n_episodes):
    """Evalúa la política greedy sobre n_episodes estados iniciales fijos."""
    results = [run_episode(env, agent, discretizer, learn=False, seed=EVAL_SEED_BASE + i)
               for i in range(n_episodes)]
    steps = np.array([r[1] for r in results])
    success = np.array([r[2] for r in results])
    velocities = np.concatenate([r[3] for r in results])
    return {
        "mean_reward": float(np.mean([r[0] for r in results])),
        "mean_steps": float(steps.mean()),
        "std_steps": float(steps.std()),
        "success_rate": float(success.mean()),
        # Pasos medios contando solo los episodios exitosos (NaN si no hubo ninguno).
        "mean_steps_success": float(steps[success].mean()) if success.any() else float("nan"),
        "omega1_abs_p99": float(np.percentile(np.abs(velocities[:, 0]), 99)),
        "omega2_abs_p99": float(np.percentile(np.abs(velocities[:, 1]), 99)),
    }


def build(config):
    d = config["discretization"]
    discretizer = make_discretizer(d["representation"], d["bins"], d["omega1_max"], d["omega2_max"])
    agent = QLearningAgent(discretizer.n_states, 3, seed=config["seed"], **config["agent"])
    return agent, discretizer


def train(config, save=False, tag=None, verbose=True):
    """Entrena según config, registra la corrida y devuelve (run_id, métricas, agente, discretizador)."""
    agent, discretizer = build(config)
    train_env, eval_env = make_env(config["max_steps"]), make_env(config["max_steps"])
    ev = config["eval"]
    episodes, curve = [], []
    start = time.perf_counter()

    for ep in range(config["episodes"]):
        # Solo el primer reset lleva semilla; los siguientes siguen la secuencia del entorno.
        seed = config["seed"] if ep == 0 else None
        reward, steps, success, _ = run_episode(train_env, agent, discretizer, learn=True, seed=seed)
        episodes.append({"episode": ep, "reward": reward, "steps": steps,
                         "success": int(success), "epsilon": round(agent.epsilon, 4)})
        agent.end_episode(ep)
        if (ep + 1) % ev["every"] == 0:
            m = evaluate(eval_env, agent, discretizer, ev["episodes"])
            curve.append({"episode": ep + 1, "mean_steps": m["mean_steps"], "success_rate": m["success_rate"]})
            if verbose:
                # Solo ASCII: la consola de Windows (cp1252) no imprime ε.
                print(f"ep {ep + 1:5d}  eps={agent.epsilon:.3f}  eval: pasos={m['mean_steps']:.1f}  "
                      f"exito={m['success_rate']:.0%}  ({time.perf_counter() - start:.0f}s)")

    train_time = time.perf_counter() - start
    metrics = evaluate(eval_env, agent, discretizer, ev["final_episodes"])
    elapsed = time.perf_counter() - start
    train_env.close()
    eval_env.close()

    # Fracción de celdas de la tabla que el agente visitó al menos una vez (alguna Q distinta de 0).
    metrics["visited_states"] = float(np.mean(np.any(agent.Q != 0, axis=1)))
    metrics["n_states"] = discretizer.n_states
    metrics["train_time_s"] = round(train_time, 2)
    metrics["train_last100_mean_steps"] = float(np.mean([e["steps"] for e in episodes[-100:]]))

    run_id = new_run_id(tag or config["algorithm"])
    model_path = None
    if save:
        MODELS_DIR.mkdir(exist_ok=True)
        model_path = MODELS_DIR / f"{run_id}.pkl"
        save_model(model_path, agent, discretizer, config)
    save_run(run_id, config, metrics, elapsed, episodes,
             extra={"eval_curve": curve, "model": model_path.name if model_path else None})
    if verbose:
        print(run_id, metrics)
    return run_id, metrics, agent, discretizer


def config_from_args(args):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["seed"] = args.seed
    config["episodes"] = args.episodes
    config["discretization"].update(representation=args.representation, bins=args.bins,
                                    omega1_max=args.omega1_max, omega2_max=args.omega2_max)
    config["agent"].update(alpha=args.alpha, gamma=args.gamma, epsilon_min=args.epsilon_min,
                           epsilon_decay=args.epsilon_decay)
    return config


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=DEFAULT_CONFIG["episodes"])
    p.add_argument("--representation", choices=["angles", "raw"], default="angles")
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--omega1-max", type=float, default=OMEGA1_MAX)
    p.add_argument("--omega2-max", type=float, default=OMEGA2_MAX)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon-min", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=float, default=0.995)
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--tag", default=None)
    args = p.parse_args()
    train(config_from_args(args), save=args.save_model, tag=args.tag)
