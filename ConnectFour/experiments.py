"""Experimentos de Connect Four (fase 3).

E1 (`poda`, D-30): sobre un conjunto fijo de posiciones de mitad de partida, se
busca la mejor jugada con profundidades 1 a `max_depth` en tres variantes:
- sin_poda:      Minimax sin poda (el orden de columnas no cambia los nodos);
- poda_centro:   Alpha-Beta con las columnas centrales primero (3, 2, 4, 1, 5, 0, 6);
- poda_izq:      Alpha-Beta con las columnas de izquierda a derecha (0 a 6).
Para cada búsqueda se guardan los nodos visitados, el tiempo, el valor de la raíz y
la columna elegida. Con eso se mide cuánto ahorra la poda, cuánto influye el orden
de las jugadas y si se elige la misma jugada que sin poda.

Experimentos de partidas (series, D-26): cada variante de nuestro agente juega una
serie por turno contra el rival del experimento, sobre las aperturas forzadas (D-29).
Las partidas corren en paralelo y al final se registra una serie por variante y turno.
- Aperturas (`aperturas`, D-32): clasifica el tablero vacío y las 7 + 49 aperturas
  forzadas según si algún jugador tiene victoria forzada (búsqueda sin heurística).
- E2 (`heuristicas`, D-31): Minimax d4 con poda y cada función de evaluación de
  heuristics.py contra Stratego(2, 4), con las 49 aperturas de dos jugadas.

Uso (desde ConnectFour/):
    python experiments.py poda                      # 50 posiciones, profundidades 1 a 7
    python experiments.py poda --positions 5 --max-depth 4 --tag prueba
    python experiments.py heuristicas               # 10 heurísticas x 2 turnos x 49 partidas
    python experiments.py heuristicas --openings 1 --tag prueba
    python experiments.py aperturas                 # victorias forzadas, profundidades 8 y 10
    python experiments.py heuristicas --split       # resultados por clase de apertura
"""

import argparse
import csv
import random
import statistics
import time
from multiprocessing import Pool

from board import Board
from experiment_log import RESULTS_DIR, new_run_id, save_run
from match import openings_list, play_series_game, save_series
from minimax_agent import MinimaxAgent
from heuristics import WIN
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


# ---------------------------------------------------------------- experimentos de partidas

STRATEGO = {"type": "stratego", "arg": 2, "depth": 4}  # como en play.py (D-29)


def minimax(depth=4, heuristic="lineas_centro"):
    return {"type": "minimax", "depth": depth, "heuristic": heuristic,
            "use_pruning": True, "move_order": "center"}


SERIES_EXPERIMENTS = {
    "heuristicas": {
        "rival": STRATEGO,
        "openings": 2,
        "variants": {h: minimax(4, h) for h in [
            "cero", "lineas", "lineas_centro", "amenazas", "capturas", "completa",
            "completa_n3x2", "completa_amenazas50", "completa_capturas10", "completa_centro6"]},
    },
}
SEATS = ("primero", "segundo")


def _play_job(job):
    variant, ours, rival, seat, opening, index = job
    return variant, seat, play_series_game(ours, rival, seat, opening, index)


def run_series_experiment(name, workers, tag, openings=None):
    spec = SERIES_EXPERIMENTS[name]
    openings = spec["openings"] if openings is None else openings
    opening_moves = openings_list(openings)
    rival = spec["rival"]
    jobs = [(v, ours, rival, seat, opening, i)
            for v, ours in spec["variants"].items() for seat in SEATS
            for i, opening in enumerate(opening_moves)]
    print(f"{name}: {len(spec['variants'])} variantes x {len(SEATS)} turnos x {len(opening_moves)} partidas"
          f" = {len(jobs)} partidas, {workers} procesos", flush=True)
    start = time.perf_counter()
    games = {}
    with Pool(min(workers, len(jobs))) as pool:
        for k, (variant, seat, row) in enumerate(pool.imap_unordered(_play_job, jobs, chunksize=1), 1):
            games.setdefault((variant, seat), []).append(row)
            if k % 50 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} partidas  [{time.perf_counter() - start:.0f}s]", flush=True)
    print(f"Total: {time.perf_counter() - start:.0f}s")
    print()

    summary = []
    for variant, ours in spec["variants"].items():
        for seat in SEATS:
            rows = games[(variant, seat)]
            # tiempo de la serie: suma de lo que tardó cada partida (corrieron en paralelo)
            elapsed = sum(r["game_s"] for r in rows)
            res = save_series(ours, rival, seat, openings, rows, elapsed, experiment=name, tag=tag,
                              extra_config={"variant": variant, "workers": workers})
            m = res["metrics"]
            summary.append({"variant": variant, "seat": seat, "games": m["games"], "wins": m["wins"],
                            "draws": m["draws"], "losses": m["losses"], "score": m["score"],
                            "moves_mean": m["moves_mean"],
                            "time_per_move_ms": round(1000 * m["our_time_per_move_s"], 2),
                            "nodes_per_move": m["our_nodes_per_move"], "run_id": res["run_id"]})
    path = RESULTS_DIR / (f"summary_{name}_{tag}.csv" if tag else f"summary_{name}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    print()
    print(f"{'variante':<22} {'1o V/E/D':>10} {'score':>6}   {'2o V/E/D':>10} {'score':>6}   {'ms/jug':>7}")
    for variant in spec["variants"]:
        a, b = [s for s in summary if s["variant"] == variant]
        print(f"{variant:<22} {a['wins']:>3}/{a['draws']}/{a['losses']:<4} {a['score']:>6.3f}"
              f"   {b['wins']:>3}/{b['draws']}/{b['losses']:<4} {b['score']:>6.3f}"
              f"   {(a['time_per_move_ms'] + b['time_per_move_ms']) / 2:>7.1f}")
    print(f"Resumen: {path}")


# ---------------------------------------------------------------- clasificación de aperturas

OUTCOME_NAMES = {1: "J1", -1: "J2", 0: "abierta"}


def _classify_job(job):
    """Busca desde la posición de la apertura (le toca a J1 si la apertura tiene largo par)
    sin heurística: un valor de ±WIN solo sale de estados terminales reales, así que es
    una victoria forzada demostrada dentro de esa profundidad."""
    moves, depth = job
    board = board_from_moves(list(moves))
    player = player_to_move(board)
    agent = MinimaxAgent(player, depth, "cero", use_pruning=True)
    agent.next_action(board)
    value = agent.last_stats["value"]
    outcome = 0
    if abs(value) >= WIN:
        winner = player if value > 0 else (2 if player == 1 else 1)
        outcome = 1 if winner == 1 else -1
    return {"opening": "".join(map(str, moves)), "n_moves": len(moves), "depth": depth,
            "outcome": OUTCOME_NAMES[outcome], "value": value, "action": agent.last_stats["action"],
            "nodes": agent.nodes, "time_s": round(agent.last_stats["time_s"], 3)}


def experiment_aperturas(depths, workers, tag):
    """Clasifica el tablero vacío y las 7 + 49 aperturas forzadas según si alguno de los
    jugadores tiene victoria forzada dentro de `depth` jugadas (D-32)."""
    openings = [()] + openings_list(1) + openings_list(2)
    jobs = [(o, d) for d in depths for o in openings]
    start = time.perf_counter()
    with Pool(min(workers, len(jobs))) as pool:
        rows = pool.map(_classify_job, jobs, chunksize=1)
    elapsed = time.perf_counter() - start
    max_depth = max(depths)
    final = {r["opening"]: r for r in rows if r["depth"] == max_depth}
    summary = []
    for n in (0, 1, 2):
        sel = [r for r in final.values() if r["n_moves"] == n]
        summary.append({"n_moves": n, "openings": len(sel),
                        **{k: sum(r["outcome"] == k for r in sel) for k in OUTCOME_NAMES.values()}})
    metrics = {"depth": max_depth, "vacio": final[""]["outcome"],
               "vacio_valor": final[""]["value"], "vacio_columna": final[""]["action"],
               **{f"aperturas2_{k}": summary[2][k] for k in OUTCOME_NAMES.values()}}
    config = {"experiment": "aperturas", "tag": tag, "depths": depths, "heuristic": "cero", "workers": workers}
    run_id = new_run_id(f"aperturas_{tag}" if tag else "aperturas")
    save_run(run_id, config, metrics, elapsed, episodes=rows,
             extra={"summary": summary, "classification": {o: r["outcome"] for o, r in final.items()}},
             rows_suffix="openings")
    path = RESULTS_DIR / (f"summary_aperturas_{tag}.csv" if tag else "summary_aperturas.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["opening", "n_moves", "outcome", "value", "action"])
        writer.writeheader()
        for o in openings:
            r = final["".join(map(str, o))]
            writer.writerow({k: r[k] for k in writer.fieldnames})
    print(f"{run_id}  ({elapsed:.0f}s)")
    for d in depths:
        by_d = {r["opening"]: r["outcome"] for r in rows if r["depth"] == d}
        print(f"  d={d}: vacío -> {by_d['']}; aperturas de 2: "
              + ", ".join(f"{k}={sum(by_d[o] == k for o in by_d if len(o) == 2)}" for k in OUTCOME_NAMES.values()))
    print(f"Resumen: {path}")


def load_opening_classes(path=None):
    path = path or RESULTS_DIR / "summary_aperturas.csv"
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return {r["opening"]: r["outcome"] for r in csv.DictReader(f)}


def split_by_openings(name, tag=""):
    """Resultados de un experimento de partidas separados por la clase de la apertura
    (victoria forzada de J1, de J2, o abierta), a partir de summary_aperturas.csv."""
    classes = load_opening_classes()
    if classes is None:
        print("Falta results/summary_aperturas.csv: correr antes `python experiments.py aperturas`.")
        return
    with open(RESULTS_DIR / "runs.csv", newline="", encoding="utf-8") as f:
        runs = [r for r in csv.DictReader(f)
                if r.get("config.experiment") == name and r.get("config.tag", "") == tag]
    summary = []
    for run in runs:
        with open(RESULTS_DIR / f"{run['run_id']}_games.csv", newline="", encoding="utf-8") as f:
            games = list(csv.DictReader(f))
        for cls in OUTCOME_NAMES.values():
            sel = [g for g in games if classes[g["opening"]] == cls]
            if not sel:
                continue
            summary.append({"variant": run["config.variant"], "seat": run["config.seat"], "openings": cls,
                            "games": len(sel), "wins": sum(g["result"] == "V" for g in sel),
                            "draws": sum(g["result"] == "E" for g in sel),
                            "losses": sum(g["result"] == "D" for g in sel), "run_id": run["run_id"]})
    path = RESULTS_DIR / (f"summary_{name}_{tag}_por_apertura.csv" if tag else f"summary_{name}_por_apertura.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"{'variante':<22} {'turno':<8} " + " ".join(f"{c:>12}" for c in OUTCOME_NAMES.values()))
    for variant in SERIES_EXPERIMENTS[name]["variants"]:
        for seat in SEATS:
            cells = {s["openings"]: f"{s['wins']}/{s['draws']}/{s['losses']}" for s in summary
                     if s["variant"] == variant and s["seat"] == seat}
            print(f"{variant:<22} {seat:<8} " + " ".join(f"{cells.get(c, '-'):>12}" for c in OUTCOME_NAMES.values()))
    print(f"Resumen (V/E/D por clase de apertura): {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", choices=["poda", "aperturas", *SERIES_EXPERIMENTS])
    p.add_argument("--split", action="store_true",
                   help="solo separar los resultados ya registrados por clase de apertura")
    p.add_argument("--positions", type=int, default=50)
    p.add_argument("--max-depth", type=int, default=7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=WORKERS)
    p.add_argument("--openings", type=int, choices=[0, 1, 2], default=None,
                   help="aperturas forzadas (solo experimentos de partidas; por defecto, las del experimento)")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    if args.experiment == "poda":
        experiment_poda(args.positions, args.max_depth, args.seed, args.workers, args.tag)
    elif args.experiment == "aperturas":
        experiment_aperturas([8, 10], args.workers, args.tag)
    elif args.split:
        split_by_openings(args.experiment, args.tag)
    else:
        run_series_experiment(args.experiment, args.workers, args.tag, args.openings)
        if load_opening_classes() is not None:
            print()
            split_by_openings(args.experiment, args.tag)


if __name__ == "__main__":
    main()
