"""Expectimax con profundidad limitada (AIMA 5.5, nodos de azar).

El agente es MAX. En los turnos del rival hay un nodo de azar: se supone que el rival
elige cualquier columna legal con la misma probabilidad, y el valor del nodo es el
promedio de sus hijos. No hay poda: para calcular un promedio hay que ver todos los hijos.

A diferencia de Minimax, la escala de la evaluación importa: Expectimax promedia
valores, así que cambiar los pesos (aunque se mantenga el orden de los estados)
puede cambiar la jugada elegida.
"""

import math

from search_base import SearchAgent


class ExpectimaxAgent(SearchAgent):
    name = "expectimax"

    def search(self, board):
        self.nodes += 1  # la raíz
        best_value, best_action = -math.inf, None
        for action in self.ordered_actions(board):
            value = self.chance_value(self.child(board, action, self.player), self.depth - 1)
            if value > best_value:
                best_value, best_action = value, action
        return best_value, best_action

    def max_value(self, board, depth):
        self.nodes += 1
        if self.is_cutoff(board, depth):
            return self.evaluate(board, depth)
        return max(self.chance_value(self.child(board, a, self.player), depth - 1)
                   for a in self.ordered_actions(board))

    def chance_value(self, board, depth):
        self.nodes += 1
        if self.is_cutoff(board, depth):
            return self.evaluate(board, depth)
        actions = self.ordered_actions(board)
        return sum(self.max_value(self.child(board, a, self.rival), depth - 1) for a in actions) / len(actions)
