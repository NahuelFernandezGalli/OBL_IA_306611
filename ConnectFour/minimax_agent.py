"""Minimax con profundidad limitada, con o sin poda Alpha-Beta (AIMA 5.2 y 5.3).

El agente es MAX y el rival es MIN (se lo supone óptimo). Con use_pruning=True se
llevan las cotas α (lo mejor que MAX ya tiene asegurado en el camino) y β (lo mejor
que MIN ya tiene asegurado) y se dejan de mirar los hermanos cuando v >= β (en MAX)
o v <= α (en MIN). Con use_pruning=False se recorre el árbol completo hasta `depth`.

La poda no cambia el valor de la raíz. Además, como en la raíz se elige con `>`
estricto y en el mismo orden de columnas, también se elige la misma jugada (D-27):
las columnas anteriores a la primera óptima valen menos que el óptimo, la primera
óptima devuelve su valor exacto, y las siguientes devuelven a lo sumo ese valor.
"""

import math

from search_base import SearchAgent


class MinimaxAgent(SearchAgent):
    name = "minimax"

    def __init__(self, player, depth=5, heuristic="completa", use_pruning=True, move_order="center"):
        super().__init__(player, depth, heuristic, move_order)
        self.use_pruning = use_pruning

    def config(self):
        return {**super().config(), "use_pruning": self.use_pruning}

    def search(self, board):
        self.nodes += 1  # la raíz
        alpha, beta = -math.inf, math.inf
        best_value, best_action = -math.inf, None
        for action in self.ordered_actions(board):
            value = self.min_value(self.child(board, action, self.player), self.depth - 1, alpha, beta)
            if value > best_value:  # estricto: en un empate queda la primera columna
                best_value, best_action = value, action
            if self.use_pruning:
                alpha = max(alpha, best_value)
        return best_value, best_action

    def max_value(self, board, depth, alpha, beta):
        self.nodes += 1
        if self.is_cutoff(board, depth):
            return self.evaluate(board, depth)
        v = -math.inf
        for action in self.ordered_actions(board):
            v = max(v, self.min_value(self.child(board, action, self.player), depth - 1, alpha, beta))
            if self.use_pruning:
                if v >= beta:
                    return v  # MIN, más arriba, ya tiene algo mejor (menor): poda
                alpha = max(alpha, v)
        return v

    def min_value(self, board, depth, alpha, beta):
        self.nodes += 1
        if self.is_cutoff(board, depth):
            return self.evaluate(board, depth)
        v = math.inf
        for action in self.ordered_actions(board):
            v = min(v, self.max_value(self.child(board, action, self.rival), depth - 1, alpha, beta))
            if self.use_pruning:
                if v <= alpha:
                    return v  # MAX, más arriba, ya tiene algo mejor (mayor): poda
                beta = min(beta, v)
        return v
