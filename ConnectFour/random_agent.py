"""Agente al azar: elige una columna legal con probabilidad uniforme (rival de referencia)."""

import random

from agent import Agent


class RandomAgent(Agent):
    name = "random"

    def __init__(self, player, seed=0):
        self.player = player
        self.seed = seed
        self.rng = random.Random(seed)
        self.last_stats = {}

    def config(self):
        return {"type": self.name, "seed": self.seed}

    def next_action(self, obs):
        return self.rng.choice(obs.get_posible_actions())

    def heuristic_utility(self, board):
        return 0
