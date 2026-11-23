"""Base común de los agentes de búsqueda (Minimax y Expectimax).

- `player` es el número de jugador del agente en el tablero (1 = O, empieza; 2 = X).
- `depth` es la profundidad en jugadas individuales (plies), igual que Stratego:
  depth = 4 mira la jugada propia y las 3 siguientes.
- El orden en que se prueban las columnas es fijo (D-27): con "center" se prueban
  primero las centrales (3, 2, 4, 1, 5, 0, 6); con "left", de izquierda a derecha.
  En los empates se queda la primera columna en ese orden.
- Después de cada jugada quedan en `last_stats` los nodos visitados, el valor de la
  raíz, la columna elegida y el tiempo, para el registro de los experimentos.
"""

import time

from agent import Agent
from heuristics import HEURISTICS, evaluate, other

MOVE_ORDERS = {
    "center": [3, 2, 4, 1, 5, 0, 6],
    "left": [0, 1, 2, 3, 4, 5, 6],
}


def player_to_move(board):
    """Jugador al que le toca: las capturas no cambian la cantidad de casillas vacías,
    así que la paridad de las fichas jugadas indica el turno."""
    moves_played = board.heigth * board.length - board._empty_spaces
    return 1 if moves_played % 2 == 0 else 2


class SearchAgent(Agent):
    name = "search"

    def __init__(self, player, depth=4, heuristic="lineas_centro", move_order="center"):
        if player not in (1, 2):
            raise ValueError("player tiene que ser 1 o 2")
        self.player = player
        self.rival = other(player)
        self.depth = depth
        self.heuristic = heuristic
        self.weights = HEURISTICS[heuristic]
        self.move_order = move_order
        self._order = MOVE_ORDERS[move_order]
        self.nodes = 0
        self.last_stats = {}

    def config(self):
        return {"type": self.name, "depth": self.depth, "heuristic": self.heuristic,
                "move_order": self.move_order}

    def ordered_actions(self, board):
        possible = set(board.get_posible_actions())
        return [a for a in self._order if a in possible]

    def child(self, board, action, player):
        new_board = board.clone()
        new_board.add_tile(action, player)
        return new_board

    def is_cutoff(self, board, depth):
        return depth == 0 or board.is_final()

    def evaluate(self, board, depth_left):
        return evaluate(board, self.player, self.weights, depth_left)

    def heuristic_utility(self, board):
        return self.evaluate(board, 0)

    def next_action(self, obs):
        if player_to_move(obs) != self.player:
            raise ValueError(f"El agente es el jugador {self.player}, pero le toca al {player_to_move(obs)}")
        self.nodes = 0
        start = time.perf_counter()
        value, action = self.search(obs)
        self.last_stats = {"nodes": self.nodes, "value": value, "action": action,
                           "time_s": time.perf_counter() - start}
        return action

    def search(self, board):
        """Devuelve (valor de la raíz, columna elegida)."""
        raise NotImplementedError
