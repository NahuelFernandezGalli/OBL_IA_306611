"""Funciones de evaluación para Connect Four (con la regla de captura de board.py).

Una heurística es una suma ponderada de componentes: h(s) = sum_k w_k * f_k(s).
Cada componente es simétrico: (valor del jugador) - (valor del rival), así que
h(s) > 0 significa que la posición es buena para `player` (D-28).

Componentes (todos se calculan sobre la grilla actual, sin buscar):
- n2: ventanas de 4 con exactamente 2 fichas propias y 2 huecos (sin fichas rivales).
- n3: ventanas de 4 con 3 fichas propias y 1 hueco.
- amenazas: ventanas con 3 fichas propias cuyo hueco se puede jugar ya
  (es la fila de abajo o la casilla de abajo está ocupada): ganan en una jugada.
- centro: fichas en la columna central, que es la que participa en más ventanas.
- capturas: fichas rivales que se capturarían con una sola jugada propia
  (ficha rival encerrada en horizontal o diagonal, regla de board.py).

Los estados terminales no usan la heurística: ganar vale +(WIN + profundidad
restante) y perder -(WIN + profundidad restante), con WIN mayor que cualquier
valor heurístico posible. Así se ordenan igual que la utilidad real, y entre
dos victorias se prefiere la más rápida (y entre dos derrotas, la más lenta).
"""

HEIGHT, LENGTH = 6, 7
WIN = 1_000_000

# Las 69 ventanas de 4 casillas consecutivas: 24 horizontales, 21 verticales y 24 diagonales.
WINDOWS = []
for _r in range(HEIGHT):
    for _c in range(LENGTH):
        for _dr, _dc in [(0, 1), (1, 0), (1, 1), (-1, 1)]:
            cells = [(_r + i * _dr, _c + i * _dc) for i in range(4)]
            if all(0 <= r < HEIGHT and 0 <= c < LENGTH for r, c in cells):
                WINDOWS.append(tuple(cells))
assert len(WINDOWS) == 69

CENTER_COL = LENGTH // 2
# Direcciones de la regla de captura de board.py (horizontal y diagonales, sin vertical).
CAPTURE_DIRS = [(0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

FEATURES = ("n2", "n3", "amenazas", "centro", "capturas")

# Combinaciones a comparar (los pesos se exploran en los experimentos).
HEURISTICS = {
    "cero": {},  # solo distingue estados terminales: línea base sin evaluación
    "lineas": {"n2": 2, "n3": 5},
    "lineas_centro": {"n2": 2, "n3": 5, "centro": 3},
    "amenazas": {"n2": 2, "n3": 5, "centro": 3, "amenazas": 20},
    "capturas": {"n2": 2, "n3": 5, "centro": 3, "capturas": 4},
    "completa": {"n2": 2, "n3": 5, "centro": 3, "amenazas": 20, "capturas": 4},
    # Variaciones de pesos de "completa", cambiando un peso por vez (E2, D-31).
    "completa_n3x2": {"n2": 2, "n3": 10, "centro": 3, "amenazas": 20, "capturas": 4},
    "completa_amenazas50": {"n2": 2, "n3": 5, "centro": 3, "amenazas": 50, "capturas": 4},
    "completa_capturas10": {"n2": 2, "n3": 5, "centro": 3, "amenazas": 20, "capturas": 10},
    "completa_centro6": {"n2": 2, "n3": 5, "centro": 6, "amenazas": 20, "capturas": 4},
}


def other(player):
    return 2 if player == 1 else 1


def _playable_cells(grid):
    """Casilla donde caería una ficha en cada columna no llena: {columna: fila}."""
    cells = {}
    for c in range(LENGTH):
        for r in range(HEIGHT - 1, -1, -1):
            if grid[r][c] == 0:
                cells[c] = r
                break
    return cells


def _is_playable(grid, r, c):
    return grid[r][c] == 0 and (r == HEIGHT - 1 or grid[r + 1][c] != 0)


def _captures(grid, player, playable):
    """Fichas rivales que `player` capturaría jugando en alguna de las casillas jugables."""
    rival = other(player)
    total = 0
    for c, r in playable.items():
        for dr, dc in CAPTURE_DIRS:
            r1, c1, r2, c2 = r + dr, c + dc, r + 2 * dr, c + 2 * dc
            if 0 <= r2 < HEIGHT and 0 <= c2 < LENGTH \
                    and grid[r1][c1] == rival and grid[r2][c2] == player:
                total += 1
    return total


def features(board, player):
    """Valor de cada componente desde el punto de vista de `player` (propio - rival)."""
    grid = board._grid.tolist()  # listas de Python: más rápido de recorrer que numpy celda a celda
    rival = other(player)
    f = dict.fromkeys(FEATURES, 0)

    for window in WINDOWS:
        mine = theirs = 0
        empty_cell = None
        for r, c in window:
            v = grid[r][c]
            if v == player:
                mine += 1
            elif v == rival:
                theirs += 1
            else:
                empty_cell = (r, c)
        if theirs == 0 and mine >= 2:
            sign, count = 1, mine
        elif mine == 0 and theirs >= 2:
            sign, count = -1, theirs
        else:
            continue
        if count == 2:
            f["n2"] += sign
        elif count == 3:
            f["n3"] += sign
            if _is_playable(grid, *empty_cell):
                f["amenazas"] += sign

    for r in range(HEIGHT):
        v = grid[r][CENTER_COL]
        if v == player:
            f["centro"] += 1
        elif v == rival:
            f["centro"] -= 1

    playable = _playable_cells(grid)
    f["capturas"] = _captures(grid, player, playable) - _captures(grid, rival, playable)
    return f


def heuristic_value(board, player, weights):
    """Suma ponderada de los componentes, para un estado no terminal."""
    if not weights:
        return 0
    f = features(board, player)
    return sum(w * f[name] for name, w in weights.items())


def evaluate(board, player, weights, depth_left=0):
    """Valor de un estado (terminal o no) desde el punto de vista de `player`."""
    if board.winner != 0:
        return (WIN + depth_left) if board.winner == player else -(WIN + depth_left)
    if board.is_full():
        return 0
    return heuristic_value(board, player, weights)
