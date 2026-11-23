"""Experimentos de Connect Four (fase 3). Por ahora: E1, impacto de la poda Alpha-Beta.

E1 (`poda`, D-30): sobre un conjunto fijo de posiciones de mitad de partida, se
busca la mejor jugada con profundidades 1 a `max_depth` en tres variantes:
- sin_poda:      Minimax sin poda (el orden de columnas no cambia los nodos);
- poda_centro:   Alpha-Beta con las columnas centrales primero (3, 2, 4, 1, 5, 0, 6);
- poda_izq:      Alpha-Beta con las columnas de izquierda a derecha (0 a 6).
Para cada búsqueda se guardan los nodos visitados, el tiempo, el valor de la raíz y
la columna elegida. Con eso se mide cuánto ahorra la poda, cuánto influye el orden
de las jugadas y si se elige la misma jugada que sin poda.

Uso (desde ConnectFour/):
    python experiments.py poda                      # 50 posiciones, profundidades 1 a 7
    python experiments.py poda --positions 5 --max-depth 4 --tag prueba
"""

import argparse
import csv
import random
import statistics
import time
from multiprocessing import Pool

from board import Board
from experiment_log import RESULTS_DIR, new_run_id, save_run
from minimax_agent import MinimaxAgent
from search_base import player_to_move

WORKERS = 6
HEURISTIC = "lineas_centro"
VARIANTS = {
    "sin_poda": {"use_pruning": False, "move_order": "center"},
    "poda_centro": {"use_pruning": True, "move_order": "center"},
    "poda_izq": {"use_pruning": True, "move_order": "left"},
}


def make_positions(n, seed=0, min_moves=4, max_moves=20):
    """Posiciones de mitad de partida: entre min_moves y max_moves jugadas al azar,
    descartando las que ya terminaron. Se guardan como la secuencia de columnas,
    así se pueden reconstruir exactamente (las capturas se aplican al reproducirlas)."""
    rng = random.Random(seed)
    positions = []
    while len(positions) < n:
        board, moves, player = Board(6, 7), [], 1
        for _ in range(rng.randint(min_moves, max_moves)):
            action = rng.choice(board.get_posible_actions())
            board.add_tile(action, player)
            moves.append(action)
            player = 2 if player == 1 else 1
            if board.is_final():
                break
        if not board.is_final():
            positions.append(moves)
    return positions


def board_from_moves(moves):
    board, player = Board(6, 7), 1
    for action in moves:
        board.add_tile(action, player)
        player = 2 if player == 1 else 1
    return board


def run_search(job):
    """Una búsqueda: (índice de posición, jugadas, profundidad, variante) -> fila de resultados."""
    index, moves, depth, variant = job
    board = board_from_moves(moves)
    agent = MinimaxAgent(player_to_move(board), depth, HEURISTIC, **VARIANTS[variant])
    agent.next_action(board)
    s = agent.last_stats
    return {"position": index, "depth": depth, "variant": variant, "nodes": s["nodes"],
            "time_s": round(s["time_s"], 6), "value": s["value"], "action": s["action"]}


def summarize_poda(rows, depths):
    """Por profundidad: nodos y tiempo medios de cada variante, ahorro de la poda,
    factor de ramificación efectivo y coincidencia de valor y jugada con sin_poda."""
    by_key = {(r["position"], r["depth"], r["variant"]): r for r in rows}
    positions = sorted({r["position"] for r in rows})
    summary = []
    for d in depths:
        line = {"depth": d}
        for v in VARIANTS:
            sel = [by_key[(p, d, v)] for p in positions]
            nodes = statistics.fmean(r["nodes"] for r in sel)
            line[f"{v}_nodes"] = round(nodes, 1)
            line[f"{v}_time_ms"] = round(1000 * statistics.fmean(r["time_s"] for r in sel), 3)
            # factor de ramificación efectivo b*: b*^d = nodos medios (aproximación simple)
            line[f"{v}_bstar"] = round(nodes ** (1 / d), 3)
        for v in ("poda_centro", "poda_izq"):
            line[f"{v}_ahorro_pct"] = round(100 * (1 - line[f"{v}_nodes"] / line["sin_poda_nodes"]), 2)
            same_value = [by_key[(p, d, v)]["value"] == by_key[(p, d, "sin_poda")]["value"] for p in positions]
            same_action = [by_key[(p, d, v)]["action"] == by_key[(p, d, "sin_poda")]["action"] for p in positions]
            line[f"{v}_mismo_valor_pct"] = round(100 * sum(same_value) / len(positions), 1)
            line[f"{v}_misma_jugada_pct"] = round(100 * sum(same_action) / len(positions), 1)
        summary.append(line)
    return summary


def experiment_poda(n_positions, max_depth, seed, workers, tag):
    positions = make_positions(n_positions, seed)
    depths = list(range(1, max_depth + 1))
    # Las búsquedas más caras primero, para repartir mejor la carga entre procesos.
    jobs = [(i, moves, d, v) for d in reversed(depths) for v in VARIANTS for i, moves in enumerate(positions)]
    start = time.perf_counter()
    with Pool(min(workers, len(jobs))) as pool:
        rows = []
        for k, row in enumerate(pool.imap_unordered(run_search, jobs, chunksize=1), 1):
            rows.append(row)
            if k % 100 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} búsquedas  [{time.perf_counter() - start:.0f}s]", flush=True)
    elapsed = time.perf_counter() - start
    rows.sort(key=lambda r: (r["depth"], r["variant"], r["position"]))

    summary = summarize_poda(rows, depths)
    config = {"experiment": "poda", "tag": tag, "heuristic": HEURISTIC, "positions": n_positions,
              "position_seed": seed, "position_moves": [4, 20], "depths": depths,
              "variants": VARIANTS, "workers": workers}
    last = summary[-1]
    metrics = {
        "max_depth": max_depth,
        "ahorro_pct_max_depth": last["poda_centro_ahorro_pct"],
        "ahorro_izq_pct_max_depth": last["poda_izq_ahorro_pct"],
        "mismo_valor_pct": min(s["poda_centro_mismo_valor_pct"] for s in summary),
        "misma_jugada_pct": min(s["poda_centro_misma_jugada_pct"] for s in summary),
        "misma_jugada_izq_pct": min(s["poda_izq_misma_jugada_pct"] for s in summary),
    }
    run_id = new_run_id(f"poda_{tag}" if tag else "poda")
    save_run(run_id, config, metrics, elapsed, episodes=rows,
             extra={"summary": summary, "positions_moves": positions}, rows_suffix="searches")
    summary_path = RESULTS_DIR / (f"summary_poda_{tag}.csv" if tag else "summary_poda.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    print(f"\n{run_id}  ({elapsed:.0f}s, {workers} procesos)")
    print(f"{'d':>2} {'sin poda':>10} {'poda centro':>12} {'ahorro':>8} {'poda izq':>10} {'ahorro':>8}"
          f" {'t sin (ms)':>11} {'t centro':>9} {'misma jug.':>10} {'misma izq':>10}")
    for s in summary:
        print(f"{s['depth']:>2} {s['sin_poda_nodes']:>10.0f} {s['poda_centro_nodes']:>12.0f}"
              f" {s['poda_centro_ahorro_pct']:>7.1f}% {s['poda_izq_nodes']:>10.0f} {s['poda_izq_ahorro_pct']:>7.1f}%"
              f" {s['sin_poda_time_ms']:>11.1f} {s['poda_centro_time_ms']:>9.1f}"
              f" {s['poda_centro_misma_jugada_pct']:>9.0f}% {s['poda_izq_misma_jugada_pct']:>9.0f}%")
    print(f"Resumen: {summary_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", choices=["poda"])
    p.add_argument("--positions", type=int, default=50)
    p.add_argument("--max-depth", type=int, default=7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=WORKERS)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    if args.experiment == "poda":
        experiment_poda(args.positions, args.max_depth, args.seed, args.workers, args.tag)


if __name__ == "__main__":
    main()
