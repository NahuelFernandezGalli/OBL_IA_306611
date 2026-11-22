"""Agente Dyna-Q tabular (Sutton y Barto, Reinforcement Learning: An Introduction, 8.2).

En cada paso real hace, en este orden (las letras son las del pseudocódigo del libro):
  (d) RL directo: la actualización de Q-Learning con la transición real.
  (e) Aprendizaje del modelo: Model(S, A) <- (R, S'), la última transición vista.
  (f) Planning: n veces, elige un estado ya visitado y una acción ya probada en él,
      le pide (R, S') al modelo y aplica la misma actualización de Q-Learning.

Con n_planning = 0 es exactamente Q-Learning (ver DECISIONES.md, D-20).
"""

import numpy as np

from qlearning import QLearningAgent


class DynaQAgent(QLearningAgent):
    def __init__(self, n_states, n_actions=3, n_planning=10, seed=0, **kwargs):
        super().__init__(n_states, n_actions, seed=seed, **kwargs)
        self.n_planning = n_planning
        # Generador propio para el planning: así la exploración (self.rng) consume los
        # mismos números que en Q-Learning y con n = 0 la corrida es idéntica.
        self.planning_rng = np.random.default_rng((seed, 1))

        # Modelo determinista: para cada (s, a), la última recompensa, el último s' y si
        # esa transición terminó el episodio (llegada al objetivo).
        self.model_reward = np.zeros((n_states, n_actions))
        self.model_next = np.full((n_states, n_actions), -1, dtype=int)
        self.model_terminal = np.zeros((n_states, n_actions), dtype=bool)
        # Estados visitados y, para cada uno, las acciones probadas (listas de Python,
        # más rápidas que numpy para elegir un elemento al azar de a uno).
        self.observed_states = []
        self.tried_actions = {}

    def update(self, s, a, r, s_next, terminated):
        """Un paso real de Dyna-Q: (d), (e) y (f)."""
        super().update(s, a, r, s_next, terminated)   # (d) RL directo
        self.update_model(s, a, r, s_next, terminated)  # (e) modelo
        self.planning()                                 # (f) planning

    def update_model(self, s, a, r, s_next, terminated):
        """Guarda la transición. El corte por tiempo (truncated) no se guarda como terminal:
        es un límite del episodio, no una propiedad del entorno (igual que en D-08)."""
        if s not in self.tried_actions:
            self.observed_states.append(s)
            self.tried_actions[s] = []
        if a not in self.tried_actions[s]:
            self.tried_actions[s].append(a)
        self.model_reward[s, a] = r
        self.model_next[s, a] = s_next
        self.model_terminal[s, a] = terminated

    def planning(self):
        """n actualizaciones con experiencia simulada por el modelo. Como en el libro, primero
        se elige un estado visitado al azar y después una acción probada en ese estado."""
        if self.n_planning == 0:
            return
        picks = self.planning_rng.random((self.n_planning, 2))
        n_observed = len(self.observed_states)
        for u_state, u_action in picks:
            s = self.observed_states[int(u_state * n_observed)]
            actions = self.tried_actions[s]
            a = actions[int(u_action * len(actions))]
            super().update(s, a, self.model_reward[s, a], self.model_next[s, a], self.model_terminal[s, a])

    def hyperparams(self):
        return {**super().hyperparams(), "n_planning": self.n_planning}
