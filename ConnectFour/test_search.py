"""Verificaciones de Minimax, Alpha-Beta, Expectimax y las heurísticas (D-27, D-28).

Uso: python test_search.py
"""

import numpy as np

from board import Board
from connect_four_env import BoardSpace
from expectimax_agent import ExpectimaxAgent
from heuristics import HEURISTICS, WIN, evaluate, features
from minimax_agent import MinimaxAgent
from search_base import player_to_move

SYMBOLS = {".": 0, "O": 1, "X": 2}


def board_from_rows(rows):
    """Tablero a partir de 6 strings de arriba hacia abajo ('.', 'O', 'X'), sin aplicar capturas."""
    board = Board(6, 7)
    board._grid = np.array([[SYMBOLS[ch] for ch in row] for row in rows], dtype=np.int64)
    board._empty_spaces = int((board._grid == 0).sum())
    return board


def sample_positions(n, seed=0):
    """Posiciones no terminales generadas con jugadas al azar (BoardSpace de la cátedra)."""
    space = BoardSpace(6, 7)
    space.seed(seed)
    positions = []
    while len(positions) < n:
        board = space.sample()
        if not board.is_final():
            positions.append(board)
    return positions


def full_tree_nodes(board, player, depth):
    """Cantidad de nodos del árbol completo hasta `depth` (contados por separado, sin el agente)."""
    if depth == 0 or board.is_final():
        return 1
    total = 1
    for action in board.get_posible_actions():
        child = board.clone()
        child.add_tile(action, player)
        total += full_tree_nodes(child, 2 if player == 1 else 1, depth - 1)
    return total


def test_poda_mismo_valor_y_jugada():
    """Con y sin poda se obtienen el mismo valor y la misma columna, y con poda se visitan menos nodos."""
    for heuristic in ["lineas_centro", "completa", "cero"]:
        for board in sample_positions(25, seed=1):
            player = player_to_move(board)
            for depth in [1, 2, 3, 4]:
                with_pruning = MinimaxAgent(player, depth, heuristic, use_pruning=True)
                without_pruning = MinimaxAgent(player, depth, heuristic, use_pruning=False)
                with_pruning.next_action(board)
                without_pruning.next_action(board)
                a, b = with_pruning.last_stats, without_pruning.last_stats
                assert a["value"] == b["value"] and a["action"] == b["action"], (heuristic, depth, a, b)
                assert a["nodes"] <= b["nodes"]


def test_sin_poda_recorre_el_arbol_completo():
    """Con use_pruning=False los nodos visitados son exactamente los del árbol completo."""
    for board in sample_positions(10, seed=2):
        player = player_to_move(board)
        for depth in [1, 2, 3]:
            agent = MinimaxAgent(player, depth, "lineas_centro", use_pruning=False)
            agent.next_action(board)
            assert agent.nodes == full_tree_nodes(board, player, depth), (agent.nodes, depth)


def test_con_poda_poda_algo():
    """En profundidad 4 la poda tiene que ahorrar nodos en casi todas las posiciones."""
    savings = []
    for board in sample_positions(10, seed=3):
        player = player_to_move(board)
        agent = MinimaxAgent(player, 4, "lineas_centro", use_pruning=True)
        agent.next_action(board)
        savings.append(agent.nodes < full_tree_nodes(board, player, 4))
    assert sum(savings) >= 8, savings


def test_gana_en_una_jugada():
    """Si puede conectar 4 ya, lo hace (Minimax y Expectimax, cualquier profundidad)."""
    board = board_from_rows([".......", ".......", ".......", ".......",
                             "....X..",
                             "OOO.XX."])
    assert player_to_move(board) == 1
    for depth in [1, 2, 3]:
        assert MinimaxAgent(1, depth).next_action(board) == 3
        assert ExpectimaxAgent(1, depth).next_action(board) == 3


def test_bloquea_en_una_jugada():
    """Si el rival gana en la próxima jugada, Minimax con profundidad >= 2 lo bloquea."""
    board = board_from_rows([".......", ".......", ".......", ".......", ".......",
                             "OOO.XX."])
    assert player_to_move(board) == 2
    for depth in [2, 3, 4]:
        assert MinimaxAgent(2, depth, "cero").next_action(board) == 3


def test_expectimax_promedia():
    """Con profundidad 2, el valor de cada jugada es el promedio sobre las respuestas del rival."""
    board = sample_positions(1, seed=4)[0]
    player = player_to_move(board)
    rival = 2 if player == 1 else 1
    agent = ExpectimaxAgent(player, 2, "lineas_centro")
    agent.next_action(board)
    expected = -np.inf
    for a in board.get_posible_actions():
        b1 = board.clone()
        b1.add_tile(a, player)
        if b1.is_final():
            value = evaluate(b1, player, HEURISTICS["lineas_centro"], 1)
        else:
            replies = []
            for r in b1.get_posible_actions():
                b2 = b1.clone()
                b2.add_tile(r, rival)
                replies.append(evaluate(b2, player, HEURISTICS["lineas_centro"], 0))
            value = sum(replies) / len(replies)
        expected = max(expected, value)
    assert abs(agent.last_stats["value"] - expected) < 1e-9


def test_heuristica_simetrica():
    """Cada componente vale lo mismo con signo opuesto desde el punto de vista del rival."""
    for board in sample_positions(20, seed=5):
        f1, f2 = features(board, 1), features(board, 2)
        assert all(f1[k] == -f2[k] for k in f1), (f1, f2)


def test_terminales_por_encima_de_la_heuristica():
    """Ganar vale más que cualquier posición no terminal, y perder menos (criterio 1 de AIMA 5.4.1)."""
    worst_case = sum(abs(w) for w in HEURISTICS["completa"].values()) * 69 * 2
    assert worst_case < WIN
    board = board_from_rows([".......", ".......", ".......", ".......", ".......",
                             "OOOOXXX"])
    board._winner = 1
    assert evaluate(board, 1, HEURISTICS["completa"]) >= WIN
    assert evaluate(board, 2, HEURISTICS["completa"]) <= -WIN


def test_jugador_equivocado_da_error():
    board = Board(6, 7)
    try:
        MinimaxAgent(2).next_action(board)
    except ValueError:
        return
    raise AssertionError("debería fallar: en el tablero vacío le toca al jugador 1")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
