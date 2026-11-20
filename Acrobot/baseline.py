"""Línea base: agente que elige acciones al azar, sin render.

Mide recompensa, pasos y éxito por episodio, el tiempo por episodio sin render,
y los rangos que alcanzan las velocidades angulares (insumo de la discretización).

Uso: python baseline.py --episodes 50
"""

import argparse
import time

import gymnasium as gym
import numpy as np

from experiment_log import new_run_id, save_run

MAX_STEPS = 700


def run_baseline(n_episodes, seed):
    env = gym.make("Acrobot-v1", render_mode=None, max_episode_steps=MAX_STEPS)
    rng = np.random.default_rng(seed)
    episodes, velocities = [], []
    start = time.perf_counter()
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total, steps, done, terminated = 0.0, 0, False, False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(int(rng.integers(3)))
            velocities.append(obs[4:6])
            total += reward
            steps += 1
            done = terminated or truncated
        episodes.append({"episode": ep, "reward": total, "steps": steps, "success": int(terminated)})
    elapsed = time.perf_counter() - start
    env.close()

    velocities = np.array(velocities)
    rewards = np.array([e["reward"] for e in episodes])
    metrics = {
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "mean_steps": float(np.mean([e["steps"] for e in episodes])),
        "seconds_per_episode": elapsed / n_episodes,
    }
    velocity_stats = {
        name: {
            "min": float(velocities[:, i].min()),
            "p1": float(np.percentile(velocities[:, i], 1)),
            "p99": float(np.percentile(velocities[:, i], 99)),
            "max": float(velocities[:, i].max()),
        }
        for i, name in enumerate(["omega1", "omega2"])
    }
    config = {"algorithm": "random", "episodes": n_episodes, "seed": seed, "max_steps": MAX_STEPS}
    run_id = new_run_id("random")
    save_run(run_id, config, metrics, elapsed, episodes, extra={"velocity_stats": velocity_stats})
    return run_id, metrics, velocity_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_id, metrics, velocity_stats = run_baseline(args.episodes, args.seed)
    print(run_id)
    print(metrics)
    print(velocity_stats)
