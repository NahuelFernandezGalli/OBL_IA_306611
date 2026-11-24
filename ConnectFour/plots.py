"""Figuras de Connect Four para el informe. Leen solo lo que hay en results/ (los summary_*.csv
y runs.csv que escribe experiments.py) y guardan en figures/ (D-39).

Uso:
    python plots.py              # todas las figuras
    python plots.py C1 C3        # solo algunas (C1, C2, C3, C4)
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: solo se guardan archivos
import matplotlib.pyplot as plt
import numpy as np

from experiment_log import RESULTS_DIR

FIGURES_DIR = Path(__file__).parent / "figures"
DPI = 150

# --- Estilo común (el mismo que Acrobot/plots.py) ---------------------------------------
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
GRAY = "#8a8985"
TEXT_SECONDARY = "#52514e"
# Resultado de la partida: verde y rojo porque significan bueno y malo; siempre con el número escrito.
WIN, DRAW, LOSS = "#0ca30c", "#c3c2b7", "#d03b3b"

# Variantes de búsqueda: mismo color y estilo en todas las figuras.
SEARCH_STYLE = {
    "sin_poda": {"color": PALETTE[1], "ls": "--", "marker": "s", "label": "Minimax sin poda"},
    "poda_centro": {"color": PALETTE[0], "ls": "-", "marker": "o", "label": "Alpha-Beta, centro primero"},
    "poda_izq": {"color": PALETTE[2], "ls": ":", "marker": "^", "label": "Alpha-Beta, izquierda a derecha"},
}
AGENT_LABEL = {"minimax_d5": "Minimax d5", "expectimax_d5": "Expectimax d5", "expectimax_d4": "Expectimax d4"}
SEAT_LABEL = {"primero": "Nuestro agente juega primero", "segundo": "Nuestro agente juega segundo"}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#b5b4ae",
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#e6e5e0",
    "grid.linewidth": 0.8,
    "lines.linewidth": 1.8,
    "legend.frameon": False,
    "savefig.bbox": "tight",
})


# --- Lectura de results/ ---------------------------------------------------------------
def read_csv(name):
    with open(RESULTS_DIR / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def series(name):
    """{(variante, turno): fila} de summary_<name>.csv."""
    return {(r["variant"], r["seat"]): r for r in read_csv(f"summary_{name}.csv")}


def by_opening(name):
    """{(variante, turno, clase de apertura): fila} de summary_<name>_por_apertura.csv."""
    return {(r["variant"], r["seat"], r["openings"]): r for r in read_csv(f"summary_{name}_por_apertura.csv")}


def wdl(row):
    return int(row["wins"]), int(row["draws"]), int(row["losses"])


def footnote(fig, text, y=-0.02):
    fig.text(0.01, y, text, ha="left", va="top", fontsize=8.5, color=TEXT_SECONDARY, wrap=True)


def save(fig, name):
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  {path.relative_to(Path(__file__).parent)}")


def stacked_wdl(ax, ys, results, total, height=0.6):
    """Barras horizontales apiladas de victorias, empates y derrotas, con una separación
    blanca entre segmentos y el número de victorias escrito."""
    for y, (w, d, l) in zip(ys, results):
        left = 0
        for value, color in [(w, WIN), (d, DRAW), (l, LOSS)]:
            if value:
                ax.barh(y, value, left=left, height=height, color=color, edgecolor="white", linewidth=1.5)
            left += value
        # El número va dentro del segmento solo si entra; si no, queda en el texto de al lado.
        min_width = 0.06 * total
        if w >= min_width:
            ax.text(w / 2, y, str(w), ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
        if l >= min_width:
            ax.text(w + d + l / 2, y, str(l), ha="center", va="center", fontsize=8.5, color="white")
    ax.set_xlim(0, total)
    ax.grid(axis="y", visible=False)


def wdl_legend(fig, y=1.0):
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (WIN, DRAW, LOSS)]
    fig.legend(handles, ["Victorias", "Empates", "Derrotas"], loc="upper right", ncol=3, fontsize=9,
               bbox_to_anchor=(0.99, y))


# --- C1: impacto de Alpha-Beta ---------------------------------------------------------
def figure_c1():
    rows = read_csv("summary_poda.csv")
    depth = np.array([int(r["depth"]) for r in rows])
    fig, (ax_n, ax_t) = plt.subplots(1, 2, figsize=(11, 4.4))
    for key, st in SEARCH_STYLE.items():
        nodes = np.array([float(r[f"{key}_nodes"]) for r in rows])
        times = np.array([float(r[f"{key}_time_ms"]) for r in rows])
        kw = {"color": st["color"], "ls": st["ls"], "marker": st["marker"], "markersize": 6, "label": st["label"]}
        ax_n.plot(depth, nodes, **kw)
        ax_t.plot(depth, times, **kw)
        # Factor de ramificación efectivo b* en la profundidad máxima (D-30).
        ax_n.annotate(f"b* ≈ {float(rows[-1][f'{key}_bstar']):.1f}", (depth[-1], nodes[-1]),
                      xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=TEXT_SECONDARY)
    for ax in (ax_n, ax_t):
        ax.set_yscale("log")
        ax.set_xlabel("Profundidad de búsqueda d")
        ax.set_xticks(depth)
        ax.grid(axis="y", which="major")
    ax_n.set_xlim(0.6, depth[-1] + 1.1)
    ax_n.set_ylabel("Nodos expandidos por jugada (escala log)")
    ax_t.set_ylabel("Tiempo por jugada, ms (escala log)")
    ax_n.set_title("a) Nodos")
    ax_t.set_title("b) Tiempo")
    ax_n.legend(loc="upper left", fontsize=8.5)

    same_move = [float(r["poda_centro_misma_jugada_pct"]) for r in rows]
    same_value = [float(r["poda_izq_mismo_valor_pct"]) for r in rows]
    saving = float(rows[-1]["poda_centro_ahorro_pct"])
    n_searches = 50 * len(rows)
    checks = (f"Alpha-Beta con el centro primero:\n"
              f"misma jugada que sin poda en {round(n_searches * np.mean(same_move) / 100)} de {n_searches} "
              f"búsquedas\n{saving:.1f} % menos nodos con d = {depth[-1]}")
    # Abajo a la derecha no hay curvas: todas crecen hacia arriba.
    ax_t.text(0.98, 0.04, checks, transform=ax_t.transAxes, ha="right", va="bottom", fontsize=8.5,
              color=TEXT_SECONDARY)
    fig.suptitle("Impacto de la poda Alpha-Beta en Minimax", fontsize=12, x=0.01, ha="left")
    footnote(fig, "Experimento poda (D-30): 50 posiciones fijas, promedio por jugada; heurística lineas_centro. "
                  "La poda no cambia el valor minimax: con los dos órdenes, el valor coincide en el 100 % "
                  f"({min(same_value):.0f} % mínimo) de las búsquedas. Con el orden de izquierda a derecha la "
                  "jugada puede cambiar solo en los empates, porque se desempata en otro orden (D-27). "
                  "b*: factor de ramificación efectivo, la b tal que b^d = nodos.")
    fig.tight_layout()
    save(fig, "C1_poda")


# --- C2: Minimax contra Expectimax, por rival y por turno -------------------------------
def figure_c2():
    rivals = [("expectimax_random", "contra Random"), ("expectimax_stratego", "contra Stratego(2, 4)"),
              ("minimax_vs_expectimax", "contra Minimax d5")]
    agents = list(AGENT_LABEL)
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.4), sharex=True, sharey=True)
    ys = np.arange(len(agents))[::-1]
    for col, (exp, rival_label) in enumerate(rivals):
        summary, openings = series(exp), by_opening(exp)
        for row, seat in enumerate(["primero", "segundo"]):
            ax = axes[row, col]
            stacked_wdl(ax, ys, [wdl(summary[(a, seat)]) for a in agents], 49)
            for y, a in zip(ys, agents):
                w, _, _ = wdl(openings[(a, seat, "abierta")])
                ax.text(50, y, f"{w}/24", va="center", fontsize=8.5, color=TEXT_SECONDARY)
            ax.set_yticks(ys, [AGENT_LABEL[a] for a in agents])
            if row == 0:
                ax.set_title(rival_label)
            if col == 0:
                ax.set_ylabel(SEAT_LABEL[seat].replace("Nuestro agente juega ", "Juega ").capitalize(),
                              fontsize=10)
            if row == 1:
                ax.set_xlabel("Partidas (49 aperturas)")
    for ax in axes[0]:
        ax.text(50, len(agents) - 0.35, "abiertas", fontsize=8, color=TEXT_SECONDARY)
    fig.suptitle("Minimax contra Expectimax: victorias, empates y derrotas por rival y por turno",
                 fontsize=12, x=0.01, ha="left")
    wdl_legend(fig, y=0.97)
    footnote(fig, "Experimentos expectimax_random, expectimax_stratego y minimax_vs_expectimax (D-34), todos con la "
                  "heurística completa: 49 partidas por barra, una por cada apertura forzada de dos jugadas. A la "
                  "derecha, victorias en las 24 aperturas abiertas; en las otras 25 el jugador 1 tiene victoria "
                  "forzada (D-32). Expectimax d4 tarda por jugada lo mismo que Minimax d5. No hubo empates.")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "C2_minimax_vs_expectimax")


# --- C3: funciones de evaluación contra Stratego ----------------------------------------
def figure_c3():
    summary, openings = series("heuristicas"), by_opening("heuristicas")
    variants = list(dict.fromkeys(v for v, _ in summary))
    chosen = "completa"
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ys = np.arange(len(variants))[::-1]
    stacked_wdl(ax, ys, [wdl(openings[(v, "segundo", "abierta")]) for v in variants], 24)
    for y, v in zip(ys, variants):
        w, _, _ = wdl(summary[(v, "segundo")])
        ax.text(24.4, y, f"{w}/49 en total", va="center", fontsize=8.5, color=TEXT_SECONDARY)
    labels = [f"{v} (elegida)" if v == chosen else v for v in variants]
    ax.set_yticks(ys, labels)
    for tick, v in zip(ax.get_yticklabels(), variants):
        if v == chosen:
            tick.set_fontweight("bold")
    ax.set_xlabel("Partidas en las 24 aperturas abiertas, jugando segundo")
    ax.set_title("Funciones de evaluación contra Stratego(2, 4)", loc="left")
    first = [wdl(summary[(v, "primero")])[0] for v in variants]
    wdl_legend(fig, y=1.02)
    footnote(fig, "Experimento heuristicas (D-31): Minimax con poda, d = 4, contra Stratego(2, 4), sobre las 49 "
                  "aperturas de dos jugadas. Se muestra solo el caso que distingue entre heurísticas: jugando "
                  "segundo, en las aperturas abiertas. Jugando primero ganan casi todas las partidas "
                  f"({min(first)} a {max(first)} de 49). 'cero' es la heurística nula: solo ve las victorias "
                  "dentro del horizonte.")
    save(fig, "C3_heuristicas")


# --- C4: profundidad contra rendimiento y tiempo ---------------------------------------
def figure_c4():
    summary, openings = series("profundidad"), by_opening("profundidad")
    runs = {(r["config.variant"], r["config.seat"]): r for r in read_csv("runs.csv")
            if r.get("config.experiment") == "profundidad"}
    depths = sorted({int(v[1:]) for v, _ in summary})
    chosen = 5
    fig, (ax_r, ax_t) = plt.subplots(1, 2, figsize=(11, 4.4))

    def pct(row):
        w, _, _ = wdl(row)
        return 100 * w / int(row["games"])

    lines = [("primero, 49 aperturas", [pct(summary[(f"d{d}", "primero")]) for d in depths], PALETTE[0], "-", "o"),
             ("segundo, 49 aperturas", [pct(summary[(f"d{d}", "segundo")]) for d in depths], PALETTE[1], "--", "s"),
             ("segundo, 24 abiertas", [pct(openings[(f"d{d}", "segundo", "abierta")]) for d in depths],
              PALETTE[2], ":", "^")]
    for label, values, color, ls, marker in lines:
        ax_r.plot(depths, values, color=color, ls=ls, marker=marker, markersize=6, label=label)
    ax_r.set_ylim(0, 105)
    ax_r.set_ylabel("Partidas ganadas (%)")
    ax_r.set_title("a) Victorias contra Stratego(2, 4)")
    ax_r.legend(loc="lower right", fontsize=8.5, title="Nuestro agente juega", title_fontsize=8.5)

    mean_ms = [np.mean([float(summary[(f"d{d}", s)]["time_per_move_ms"]) for s in ("primero", "segundo")])
               for d in depths]
    max_ms = [1000 * max(float(runs[(f"d{d}", s)]["metrics.our_time_max_s"]) for s in ("primero", "segundo"))
              for d in depths]
    ax_t.plot(depths, mean_ms, color=PALETTE[0], marker="o", markersize=6, label="promedio por jugada")
    ax_t.plot(depths, max_ms, color=PALETTE[6], ls="--", marker="D", markersize=5, label="peor jugada")
    ax_t.set_yscale("log")
    ax_t.set_ylabel("Tiempo por jugada, ms (escala log)")
    ax_t.set_title("b) Costo")
    ax_t.legend(loc="upper left", fontsize=8.5)

    for ax in (ax_r, ax_t):
        ax.set_xticks(depths)
        ax.set_xlabel("Profundidad de búsqueda d")
        ax.axvline(chosen, color=GRAY, lw=1)
    ax_t.text(chosen + 0.1, 0.03, "d = 5 (elegida)", transform=ax_t.get_xaxis_transform(), fontsize=8.5,
              color=TEXT_SECONDARY)
    fig.suptitle("Profundidad de Minimax con Alpha-Beta: rendimiento contra costo", fontsize=12, x=0.01, ha="left")
    footnote(fig, "Experimento profundidad (D-33): heurística completa, 49 aperturas de dos jugadas por turno y "
                  "profundidad, contra Stratego(2, 4). Jugando segundo, la mejora se estanca desde d = 4, y con "
                  "d = 8 empeora: ante una derrota forzada contra un rival óptimo, Minimax juega cualquier "
                  "jugada (D-33). Peor jugada: máximo entre todas las jugadas de las dos series. Tiempos medidos "
                  "con 6 procesos en paralelo.")
    fig.tight_layout()
    save(fig, "C4_profundidad")


FIGURES = {"C1": figure_c1, "C2": figure_c2, "C3": figure_c3, "C4": figure_c4}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("figures", nargs="*", help=f"figuras a generar ({', '.join(FIGURES)}); por defecto, todas")
    args = p.parse_args()
    unknown = set(args.figures) - set(FIGURES)
    if unknown:
        p.error(f"figuras desconocidas: {', '.join(sorted(unknown))}")
    for name in args.figures or list(FIGURES):
        print(name)
        FIGURES[name]()
