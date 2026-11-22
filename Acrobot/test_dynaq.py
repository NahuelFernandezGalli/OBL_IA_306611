"""Verificaciones de Dyna-Q (D-20). Uso: python test_dynaq.py"""

import copy

import numpy as np

from dynaq import DynaQAgent
from train import DEFAULT_CONFIG, run_training


def test_planning_terminal_sin_bootstrap():
    """Una transición terminal guardada en el modelo se planifica con objetivo r, sin γ max Q(s')."""
    agent = DynaQAgent(n_states=3, n_actions=1, n_planning=0, alpha=1.0, gamma=0.99, q_init=-100.0)
    agent.Q[2, 0] = -50.0  # si se usara bootstrap, el objetivo dependería de este valor
    agent.update_model(0, 0, -1.0, 2, terminated=True)
    agent.n_planning = 1
    agent.planning()
    assert agent.Q[0, 0] == -1.0, agent.Q[0, 0]


def test_planning_no_terminal_con_bootstrap():
    """Una transición no terminal (incluido el corte por tiempo) usa r + γ max Q(s')."""
    agent = DynaQAgent(n_states=3, n_actions=1, n_planning=0, alpha=1.0, gamma=0.5, q_init=0.0)
    agent.Q[2, 0] = -10.0
    agent.update_model(0, 0, -1.0, 2, terminated=False)
    agent.n_planning = 1
    agent.planning()
    assert agent.Q[0, 0] == -1.0 + 0.5 * -10.0, agent.Q[0, 0]


def test_planning_solo_pares_vistos():
    """El planning solo toca pares (s, a) que se vieron de verdad."""
    agent = DynaQAgent(n_states=10, n_actions=3, n_planning=200, alpha=0.5, q_init=-100.0)
    agent.update(4, 1, -1.0, 5, terminated=True)
    agent.update(7, 2, -1.0, 4, terminated=False)
    changed = np.argwhere(agent.Q != -100.0)
    assert {tuple(x) for x in changed} <= {(4, 1), (7, 2)}, changed


def test_modelo_guarda_ultima_transicion():
    """El modelo determinista se queda con la última transición vista (S y B 8.2, paso e)."""
    agent = DynaQAgent(n_states=10, n_actions=3, n_planning=0)
    agent.update(1, 0, -1.0, 2, terminated=False)
    agent.update(1, 0, -1.0, 3, terminated=False)
    assert agent.model_next[1, 0] == 3
    assert agent.observed_states == [1] and agent.tried_actions[1] == [0]


def test_n0_igual_a_qlearning():
    """Con n = 0, Dyna-Q reproduce exactamente la corrida de Q-Learning (misma semilla)."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(episodes=300, eval={"every": 100, "episodes": 5, "final_episodes": 10})
    q = run_training(config, verbose=False)
    config_dyna = copy.deepcopy(config)
    config_dyna["algorithm"] = "dynaq"
    config_dyna["agent"]["n_planning"] = 0
    d = run_training(config_dyna, verbose=False)
    strip = lambda eps: [{k: v for k, v in e.items() if k != "elapsed_s"} for e in eps]
    assert strip(q["episodes"]) == strip(d["episodes"])
    assert q["curve"] == d["curve"]
    assert np.array_equal(q["agent"].Q, d["agent"].Q)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
