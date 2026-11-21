"""Entrenamiento y evaluación de agentes tabulares sobre Acrobot.

- Entrena con ε-greedy y cada `eval.every` episodios evalúa la política greedy
  (ε = 0, sin aprender) sobre estados iniciales fijos, para la curva de aprendizaje.
- Al final hace una evaluación greedy más larga, que es la métrica para comparar corridas.
- Guarda la corrida en results/ y, si se pide, el modelo en models/.

Uso: python train.py --save-model                 (configuración final, D-18)
     python train.py --episodes 3000 --bins 10 --alpha 0.2 --tag prueba
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
# Semillas de prueba: estados iniciales que no se usaron para elegir nada, para medir
# el modelo final sin el sesgo de haberlo elegido con las semillas de evaluación.
TEST_SEED_BASE = 2_000_000

DEFAULT_CONFIG = {
    "algorithm": "qlearning",
    "seed": 0,
    # Configuración final (DECISIONES.md): q_init = -100 = -1/(1-γ), inicialización pesimista
    # por indicación del docente (D-17); el resto, re-ajustado para esa inicialización (D-18).
    "episodes": 40000,
    "max_steps": 700,
    # Ángulos, bins (θ1, θ2, ω1, ω2) = (6, 6, 10, 10) y velocidades en la mitad de los
    # límites del entorno (D-06, D-12 y D-18).
    "discretization": {"representation": "angles", "bins": [6, 6, 10, 10],
                       "omega1_max": OMEGA1_MAX / 2, "omega2_max": OMEGA2_MAX / 2},
    "agent": {"alpha": 0.5, "gamma": 0.99, "epsilon_start": 1.0,
              "epsilon_min": 0.01, "epsilon_decay": 0.999, "q_init": -100.0},
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


def evaluate(env, agent, discretizer, n_episodes, seed_base=EVAL_SEED_BASE):
    """Evalúa la política greedy sobre n_episodes estados iniciales fijos."""
    results = [run_episode(env, agent, discretizer, learn=False, seed=seed_base + i)
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


def run_training(config, verbose=True):
    """Entrena y evalúa según config, sin escribir nada en disco.
    Devuelve un dict con el agente, el discretizador, las métricas, los episodios y la curva."""
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

    # Fracción de celdas de la tabla que el agente visitó al menos una vez
    # (alguna Q distinta del valor inicial).
    metrics["visited_states"] = float(np.mean(np.any(agent.Q != agent.q_init, axis=1)))
    metrics["n_states"] = discretizer.n_states
    metrics["train_time_s"] = round(train_time, 2)
    metrics["train_last100_mean_steps"] = float(np.mean([e["steps"] for e in episodes[-100:]]))
    # Media de los pasos greedy a lo largo de la curva: resume qué tan rápido aprende
    # (dos corridas con el mismo resultado final se distinguen por cuánto tardaron en llegar).
    metrics["curve_mean_steps"] = float(np.mean([c["mean_steps"] for c in curve])) if curve else float("nan")
    return {"agent": agent, "discretizer": discretizer, "metrics": metrics,
            "episodes": episodes, "curve": curve, "elapsed": elapsed}


def log_run(result, config, tag=None, save=False):
    """Registra en results/ una corrida devuelta por run_training y, si save=True, guarda el modelo."""
    run_id = new_run_id(tag or config["algorithm"])
    model_path = None
    if save:
        MODELS_DIR.mkdir(exist_ok=True)
        model_path = MODELS_DIR / f"{run_id}.pkl"
        save_model(model_path, result["agent"], result["discretizer"], config)
    save_run(run_id, config, result["metrics"], result["elapsed"], result["episodes"],
             extra={"eval_curve": result["curve"], "model": model_path.name if model_path else None})
    return run_id


def train(config, save=False, tag=None, verbose=True):
    """Entrena, registra la corrida y devuelve (run_id, métricas, agente, discretizador)."""
    result = run_training(config, verbose)
    run_id = log_run(result, config, tag, save)
    if verbose:
        print(run_id, result["metrics"])
    return run_id, result["metrics"], result["agent"], result["discretizer"]


def config_from_args(args):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["seed"] = args.seed
    config["episodes"] = args.episodes
    bins = args.bins[0] if len(args.bins) == 1 else args.bins
    config["discretization"].update(representation=args.representation, bins=bins,
                                    omega1_max=args.omega1_max, omega2_max=args.omega2_max)
    config["agent"].update(alpha=args.alpha, gamma=args.gamma, epsilon_min=args.epsilon_min,
                           epsilon_decay=args.epsilon_decay, q_init=args.q_init)
    return config


def parse_args(argv=None):
    """Parámetros de línea de comandos; los valores por defecto salen de DEFAULT_CONFIG."""
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=DEFAULT_CONFIG["episodes"])
    p.add_argument("--representation", choices=["angles", "raw"], default="angles")
    # Un número (igual en todas las dimensiones) o uno por dimensión: --bins 8 8 12 12
    p.add_argument("--bins", type=int, nargs="+", default=DEFAULT_CONFIG["discretization"]["bins"])
    p.add_argument("--omega1-max", type=float, default=OMEGA1_MAX / 2)
    p.add_argument("--omega2-max", type=float, default=OMEGA2_MAX / 2)
    p.add_argument("--alpha", type=float, default=DEFAULT_CONFIG["agent"]["alpha"])
    p.add_argument("--gamma", type=float, default=DEFAULT_CONFIG["agent"]["gamma"])
    p.add_argument("--epsilon-min", type=float, default=DEFAULT_CONFIG["agent"]["epsilon_min"])
    p.add_argument("--epsilon-decay", type=float, default=DEFAULT_CONFIG["agent"]["epsilon_decay"])
    p.add_argument("--q-init", type=float, default=DEFAULT_CONFIG["agent"]["q_init"])
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--tag", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    train(config_from_args(args), save=args.save_model, tag=args.tag)
