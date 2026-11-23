"""Partidas y series entre agentes, con registro en results/ (D-03, D-26).

Una serie enfrenta a nuestro agente con un rival desde un turno fijo ("primero" o
"segundo") y sobre un conjunto de aperturas (D-29):
- openings = 0: solo la partida desde el tablero vacío;
- openings = 1: las 7 aperturas de una jugada forzada;
- openings = 2: las 49 aperturas de dos jugadas forzadas (una de cada jugador).
Las jugadas de la apertura no las elige ningún agente; después juega cada uno.

Cada serie queda en results/<run_id>.json (configuración y métricas) y en
results/<run_id>_games.csv (una fila por partida), más una fila en results/runs.csv.

Ejemplos (desde ConnectFour/):
    python match.py --depth 4 --heuristic completa --rival stratego --openings 1
    python match.py --ours expectimax --depth 3 --rival random --openings 2
    python match.py --rival minimax --rival-depth 4 --seats primero
"""

import argparse
import itertools
import statistics
import time

from connect_four_env import ConnectFourBaseEnv
from expectimax_agent import ExpectimaxAgent
from experiment_log import new_run_id, save_run
from minimax_agent import MinimaxAgent
from random_agent import RandomAgent
from stratego_agent import Stratego

SEAT_PLAYER = {"primero": 1, "segundo": 2}
RESULT_NAMES = {1: "V", 0: "E", -1: "D"}  # desde el punto de vista de nuestro agente


class StrategoAdapter:
    """Envuelve a Stratego para medir su tiempo por jugada con la misma interfaz.

    Se construye como lo hace la cátedra en play.py: Stratego(2, 4) (D-29). El número
    que recibe el constructor no es su número de jugador: Stratego(n, d) se considera
    el jugador (n % 2) + 1. Por eso se registra `arg` tal cual.
    """
    name = "stratego"

    def __init__(self, player, arg=2, depth=4):
        self.player = player
        self.arg = arg
        self.depth = depth
        self.agent = Stratego(arg, depth)
        self.last_stats = {}

    def config(self):
        return {"type": self.name, "arg": self.arg, "depth": self.depth}

    def next_action(self, obs):
        start = time.perf_counter()
        action = self.agent.next_action(obs)
        self.last_stats = {"time_s": time.perf_counter() - start}
        return action


def build_agent(spec, player, game_index=0):
    """Crea un agente a partir de su especificación (dict) y su número de jugador."""
    spec = dict(spec)
    kind = spec.pop("type")
    if kind == "minimax":
        return MinimaxAgent(player, **spec)
    if kind == "expectimax":
        return ExpectimaxAgent(player, **spec)
    if kind == "random":
        # cada partida usa otra semilla, para que las partidas al azar no se repitan
        return RandomAgent(player, seed=spec.get("seed", 0) + game_index)
    if kind == "stratego":
        return StrategoAdapter(player, **spec)
    raise ValueError(f"Tipo de agente desconocido: {kind}")


def openings_list(n_moves, length=7):
    return [tuple(o) for o in itertools.product(range(length), repeat=n_moves)]


def play_game(agents, opening=()):
    """Juega una partida en el entorno de la cátedra (sin render).

    agents: {1: agente, 2: agente}. Devuelve el ganador (0 = empate), la cantidad de
    jugadas (incluida la apertura) y las estadísticas de cada jugada de cada agente.
    """
    env = ConnectFourBaseEnv(render_mode=None)
    obs, _ = env.reset()
    current, done = 1, False
    for action in opening:
        obs, _, done, _, _ = env.step(action)
        current = 2 if current == 1 else 1
    stats = {1: [], 2: []}
    while not done:
        move_start = time.perf_counter()
        action = agents[current].next_action(obs)
        move_stats = dict(agents[current].last_stats)
        move_stats["time_s"] = time.perf_counter() - move_start  # mismo criterio para todos los agentes
        stats[current].append(move_stats)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        current = 2 if current == 1 else 1
    return {"winner": obs.winner, "moves": len(opening) + len(stats[1]) + len(stats[2]), "stats": stats}


def _mean(values):
    return statistics.fmean(values) if values else 0.0


def run_series(ours, rival, seat, openings=1, experiment="match", tag="", save=True, verbose=True):
    """Serie de partidas de `ours` contra `rival` con nuestro agente en el turno `seat`."""
    our_player = SEAT_PLAYER[seat]
    rival_player = 2 if our_player == 1 else 1
    games = []
    start = time.perf_counter()
    for i, opening in enumerate(openings_list(openings)):
        agents = {our_player: build_agent(ours, our_player, i), rival_player: build_agent(rival, rival_player, i)}
        game = play_game(agents, opening)
        winner = game["winner"]
        result = 0 if winner == 0 else (1 if winner == our_player else -1)
        our_stats, rival_stats = game["stats"][our_player], game["stats"][rival_player]
        our_times = [s["time_s"] for s in our_stats]
        games.append({
            "game": i,
            "opening": "".join(map(str, opening)),
            "our_player": our_player,
            "winner": winner,
            "result": RESULT_NAMES[result],
            "moves": game["moves"],
            "our_moves": len(our_stats),
            "our_time_mean_s": round(_mean(our_times), 5),
            "our_time_max_s": round(max(our_times, default=0.0), 5),
            "our_nodes_mean": round(_mean([s["nodes"] for s in our_stats if "nodes" in s]), 1),
            "rival_time_mean_s": round(_mean([s["time_s"] for s in rival_stats]), 5),
        })
    elapsed = time.perf_counter() - start

    n = len(games)
    wins = sum(g["result"] == "V" for g in games)
    draws = sum(g["result"] == "E" for g in games)
    losses = sum(g["result"] == "D" for g in games)
    all_times = [g["our_time_mean_s"] for g in games]
    metrics = {
        "games": n, "wins": wins, "draws": draws, "losses": losses,
        "win_rate": round(wins / n, 4),
        "score": round((wins + 0.5 * draws) / n, 4),  # 1 por victoria, 0.5 por empate
        "moves_mean": round(_mean([g["moves"] for g in games]), 2),
        "our_time_per_move_s": round(_mean(all_times), 5),
        "our_time_max_s": round(max(g["our_time_max_s"] for g in games), 5),
        "our_nodes_per_move": round(_mean([g["our_nodes_mean"] for g in games]), 1),
        "rival_time_per_move_s": round(_mean([g["rival_time_mean_s"] for g in games]), 5),
    }
    config = {"experiment": experiment, "tag": tag, "ours": ours, "rival": rival,
              "seat": seat, "openings": openings}

    run_id = None
    if save:
        run_id = new_run_id(f"{experiment}_{tag}" if tag else experiment)
        save_run(run_id, config, metrics, elapsed, episodes=games)
    if verbose:
        print(f"{_label(ours)} vs {_label(rival)} ({seat}, {n} partidas): "
              f"V={wins} E={draws} D={losses}  jugadas={metrics['moves_mean']}  "
              f"t/jugada={metrics['our_time_per_move_s']:.4f}s  nodos/jugada={metrics['our_nodes_per_move']}"
              f"  [{elapsed:.1f}s]" + (f"  -> {run_id}" if run_id else ""))
    return {"run_id": run_id, "config": config, "metrics": metrics, "games": games}


def _label(spec):
    parts = [spec["type"]]
    if "depth" in spec:
        parts.append(f"d{spec['depth']}")
    if "heuristic" in spec:
        parts.append(spec["heuristic"])
    if spec.get("use_pruning") is False:
        parts.append("sin_poda")
    return "_".join(parts)


def agent_spec(kind, depth, heuristic, use_pruning, move_order):
    if kind == "minimax":
        return {"type": "minimax", "depth": depth, "heuristic": heuristic,
                "use_pruning": use_pruning, "move_order": move_order}
    if kind == "expectimax":
        return {"type": "expectimax", "depth": depth, "heuristic": heuristic, "move_order": move_order}
    if kind == "random":
        return {"type": "random", "seed": 0}
    if kind == "stratego":
        return {"type": "stratego", "arg": 2, "depth": depth}
    raise ValueError(kind)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ours", default="minimax", choices=["minimax", "expectimax"])
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heuristic", default="lineas_centro")
    p.add_argument("--no-pruning", action="store_true")
    p.add_argument("--move-order", default="center", choices=["center", "left"])
    p.add_argument("--rival", default="stratego", choices=["stratego", "random", "minimax", "expectimax"])
    p.add_argument("--rival-depth", type=int, default=4)
    p.add_argument("--rival-heuristic", default="lineas_centro")
    p.add_argument("--seats", nargs="+", default=["primero", "segundo"], choices=list(SEAT_PLAYER))
    p.add_argument("--openings", type=int, default=1, choices=[0, 1, 2])
    p.add_argument("--experiment", default="match")
    p.add_argument("--tag", default="")
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()

    ours = agent_spec(args.ours, args.depth, args.heuristic, not args.no_pruning, args.move_order)
    rival = agent_spec(args.rival, args.rival_depth, args.rival_heuristic, True, "center")
    for seat in args.seats:
        run_series(ours, rival, seat, args.openings, args.experiment, args.tag, save=not args.no_save)


if __name__ == "__main__":
    main()
