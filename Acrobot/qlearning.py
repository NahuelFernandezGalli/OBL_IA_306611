"""Agente Q-Learning tabular con política ε-greedy (AIMA 22.3.3, ec. 22.7)."""

import pickle

import numpy as np

from discretizer import Discretizer


class QLearningAgent:
    def __init__(self, n_states, n_actions=3, alpha=0.1, gamma=0.99,
                 epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.995, seed=0):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.rng = np.random.default_rng(seed)
        # Q en 0: como todas las recompensas son <= 0, es una inicialización optimista
        # y empuja al agente a probar las acciones que todavía no probó.
        self.Q = np.zeros((n_states, n_actions))

    def greedy_action(self, s):
        """argmax de Q(s, ·), con desempate al azar (si no, siempre ganaría la acción 0)."""
        q = self.Q[s]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))

    def act(self, s, greedy=False):
        """ε-greedy: con probabilidad ε una acción al azar, si no la mejor según Q."""
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy_action(s)

    def update(self, s, a, r, s_next, terminated):
        """Q(s,a) <- Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)].

        Si el episodio terminó porque se llegó al objetivo, s' es terminal y vale 0.
        Si se cortó por tiempo (truncated), s' no es terminal y se sigue usando su valor.
        """
        target = r if terminated else r + self.gamma * self.Q[s_next].max()
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def end_episode(self, episode):
        """Decaimiento exponencial de ε por episodio, con piso en epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon_start * self.epsilon_decay ** (episode + 1))

    def hyperparams(self):
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }


def save_model(path, agent, discretizer, config):
    """Guarda la tabla Q junto con el discretizador: sin él la tabla no sirve."""
    with open(path, "wb") as f:
        pickle.dump({"Q": agent.Q, "discretizer": discretizer.to_dict(),
                     "hyperparams": agent.hyperparams(), "config": config}, f)


def load_model(path):
    """Devuelve (agente, discretizador). El agente queda listo para actuar en modo greedy."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    discretizer = Discretizer.from_dict(data["discretizer"])
    agent = QLearningAgent(discretizer.n_states, data["Q"].shape[1], **data["hyperparams"])
    agent.Q = data["Q"]
    agent.epsilon = 0.0
    return agent, discretizer
