"""Figuras de Acrobot para el informe. Leen solo lo que hay en results/ y guardan en figures/.

Nunca se grafica desde variables en memoria: cada figura se puede regenerar desde los
registros de los experimentos (D-37).

Uso:
    python plots.py              # todas las figuras
    python plots.py A1 A4        # solo algunas (A1, A2, A3, A4, A5)
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: solo se guardan archivos
import matplotlib.pyplot as plt
import numpy as np

from experiment_log import RESULTS_DIR
from sweep import EXPERIMENTS

FIGURES_DIR = Path(__file__).parent / "figures"
DPI = 150

# --- Estilo común --------------------------------------------------------------------
# Paleta categórica validada (orden fijo, apta para daltonismo en pares adyacentes).
# Además del color, cada serie tiene su estilo de línea, para que se lea impresa en gris.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
LINESTYLES = ["-", "--", ":", "-.", (0, (5, 1, 1, 1, 1, 1)), (0, (1, 3)), (0, (8, 2))]
GRAY = "#8a8985"
TEXT_SECONDARY = "#52514e"

# Cada algoritmo o variante tiene siempre el mismo color y estilo en todas las figuras.
STYLE = {
    "ql_pes": {"color": PALETTE[0], "ls": "-", "label": "Q-Learning, Q₀ = −100"},
    "dynaq": {"color": PALETTE[1], "ls": "--", "label": "Dyna-Q (n = 2, α = 0.05)"},
    "ql_cero": {"color": PALETTE[2], "ls": ":", "label": "Q-Learning, Q₀ = 0"},
    "q_medio": {"color": PALETTE[6], "ls": "-.", "label": "Q-Learning, Q₀ = −50"},
    "random": {"color": GRAY, "ls": "-", "label": "Agente al azar"},
}

Y_STEPS = "Pasos hasta la meta (menos es mejor)"

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
_INDEX = None


def run_index():
    """Todas las corridas de results/ (un JSON por corrida), leídas una sola vez."""
    global _INDEX
    if _INDEX is None:
        _INDEX = []
        for path in RESULTS_DIR.glob("*.json"):
            if path.name.endswith("_test.json"):
                continue
            with open(path, encoding="utf-8") as f:
                _INDEX.append(json.load(f))
    return _INDEX


def load_experiment(experiment):
    """{variante: [corridas ordenadas por semilla]} en el orden en que el experimento define
    las variantes en sweep.py. Si una semilla se corrió más de una vez, queda la última."""
    latest = {}
    for run in run_index():
        config = run.get("config", {})
        if config.get("experiment") != experiment:
            continue
        key = (config["variant"], config["seed"])
        if key not in latest or run["timestamp"] > latest[key]["timestamp"]:
            latest[key] = run
    variants = {}
    for (variant, _), run in sorted(latest.items(), key=lambda kv: kv[0][1]):
        variants.setdefault(variant, []).append(run)
    order = [v for v, _ in EXPERIMENTS.get(experiment, [])]
    return {v: variants[v] for v in order if v in variants}


def load_episodes(run):
    """Columnas del CSV por episodio de una corrida, como arrays de numpy."""
    path = RESULTS_DIR / f"{run['run_id']}_episodes.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def latest_model(prefix):
    """Nombre (sin .pkl) del último modelo de models/ con ese prefijo y su evaluación de prueba."""
    stems = sorted(p.stem for p in (Path(__file__).parent / "models").glob(f"{prefix}_2*.pkl")
                   if (RESULTS_DIR / f"{p.stem}_test.json").exists())
    if not stems:
        raise FileNotFoundError(f"no hay un modelo evaluado con prefijo {prefix}")
    return stems[-1]


# Modelos entregados (D-38) y la referencia con Q0 = 0 (D-15).
MODEL_QL = "qlearning_final_rango10k"
MODEL_DYNAQ = "dynaq_final_rango"
MODEL_QL_CERO = "qlearning_final_20261120-154018_3f81"


def load_test(stem):
    with open(RESULTS_DIR / f"{stem}_test.json", encoding="utf-8") as f:
        return json.load(f)


def moving_average(values, window):
    return np.convolve(values, np.ones(window) / window, mode="valid")


def default_window(n_episodes):
    return max(50, n_episodes // 40)


def seed_curves(runs, window, x="episode"):
    """Media móvil de los pasos de cada semilla, sobre un eje x común.
    x="episode": por episodio. x="time": por tiempo de reloj (elapsed_s), interpolado
    a una grilla común que llega hasta la semilla más corta."""
    data = [load_episodes(r) for r in runs]
    curves = [moving_average(d["steps"], window) for d in data]
    if x == "episode":
        n = min(len(c) for c in curves)
        xs = data[0]["episode"][window - 1:window - 1 + n] + 1
        return xs, np.array([c[:n] for c in curves])
    times = [d["elapsed_s"][window - 1:] for d in data]
    grid = np.linspace(max(t[0] for t in times), min(t[-1] for t in times), 400)
    return grid, np.array([np.interp(grid, t, c) for t, c in zip(times, curves)])


def plot_band(ax, xs, curves, color, ls, label, band=True):
    mean = curves.mean(axis=0)
    ax.plot(xs, mean, color=color, ls=ls, label=label)
    if band:
        std = curves.std(axis=0)
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
    return mean


def final_steps(runs):
    """Pasos greedy finales (100 estados de evaluación) de cada semilla."""
    return np.array([r["metrics"]["mean_steps"] for r in runs])


def footnote(fig, text):
    fig.text(0.01, -0.02, text, ha="left", va="top", fontsize=8.5, color=TEXT_SECONDARY, wrap=True)


def save(fig, name):
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  {path.relative_to(Path(__file__).parent)}")


def random_reference(ax, x_text):
    """Recta del agente al azar sobre los 1000 estados de prueba (D-35), con etiqueta directa."""
    random = load_test("random")
    y = random["mean_steps"]
    ax.axhline(y, color=GRAY, lw=1.2)
    ax.text(x_text, y - 12, f"agente al azar: {y:.0f} pasos ({random['success_rate']:.1%} de éxito)",
            color=TEXT_SECONDARY, fontsize=8.5, va="top")


# --- A1: curva de aprendizaje del modelo final -----------------------------------------
def figure_a1():
    runs = load_experiment("rango_final")["ql_10000"]
    window = default_window(10000)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    xs, curves = seed_curves(runs, window)
    st = STYLE["ql_pes"]
    plot_band(ax, xs, curves, st["color"], st["ls"], "Entrenamiento (ε-greedy), media móvil")
    # Evaluación greedy (ε = 0) cada 250 episodios, promedio entre semillas.
    ev_x = np.array([p["episode"] for p in runs[0]["eval_curve"]])
    ev_y = np.mean([[p["mean_steps"] for p in r["eval_curve"]] for r in runs], axis=0)
    ax.scatter(ev_x, ev_y, s=9, color="#1c5cab", zorder=3, label="Evaluación greedy (ε = 0), cada 250 episodios")
    random_reference(ax, 1500)
    ax.set_ylim(0, 720)
    ax.set_xlim(0, 10000)
    ax.set_xlabel("Episodio de entrenamiento")
    ax.set_ylabel(Y_STEPS)
    ax.set_title("Curva de aprendizaje del modelo final de Q-Learning (Q₀ = −100)")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.86))
    footnote(fig, f"Experimento rango_final, ql_10000: 5 semillas, 10 000 episodios, configuración final (D-18 y D-38). Línea: media entre semillas de la "
                  f"media móvil de {window} episodios; banda: ±1 desvío entre semillas. Puntos: evaluación greedy "
                  f"sobre 20 estados fijos, promedio entre semillas. Recta gris: agente al azar sobre los 1000 "
                  f"estados de prueba.")
    save(fig, "A1_curva_modelo_final")


# --- A2: un barrido por figura ---------------------------------------------------------
SWEEPS = [
    # (nombre de la figura, experimentos, parámetro, Q0 con que se corrió, variante elegida,
    #  orden de las variantes o None para el de sweep.py)
    ("A2_pes_alpha", ["pes_alpha", "pes_alpha_alto"], "α (tasa de aprendizaje)", "−100", "0.5", None),
    ("A2_pes_bins", ["pes_bins", "pes_bins_gruesos"], "bins (θ1, θ2, ω1, ω2)", "−100", "6_6_10_10",
     ["4", "5", "6", "6_6_10_10", "8", "8_8_12_12"]),
    ("A2_pes_epsilon", ["pes_epsilon"], "exploración ε", "−100", "decay0.999_min0.01", None),
    ("A2_pes_rangos", ["pes_rangos"], "rango de las velocidades", "−100", "entrenado_5_10", None),
    ("A2_pes_representacion", ["pes_representacion"], "representación del estado", "−100",
     "angulos_6_6_10_10", None),
    ("A2_cero_rangos", ["rangos"], "rango de las velocidades", "0", "mitad_6.3_14.1", None),
    ("A2_cero_representacion", ["representacion"], "representación del estado", "0", "angulos_10", None),
    ("A2_cero_bins", ["bins"], "bins", "0", "angulos8_vel12", None),
    ("A2_cero_alpha", ["alpha"], "α (tasa de aprendizaje)", "0", "0.1", None),
    ("A2_cero_gamma", ["gamma"], "γ (descuento)", "0", "0.99", None),
    ("A2_cero_epsilon", ["epsilon"], "exploración ε", "0", "decay0.999_min0.05", None),
]


def sweep_figure(name, experiments, parameter, q0, chosen, order):
    variants = {}
    for exp in experiments:
        variants.update(load_experiment(exp))
    if order:
        variants = {v: variants[v] for v in order if v in variants}
    if not variants:
        print(f"  {name}: sin corridas en results/ (falta correr {', '.join(experiments)})")
        return
    if len(variants) > len(PALETTE):
        raise ValueError(f"{name}: más variantes que colores")
    n_episodes = next(iter(variants.values()))[0]["config"]["episodes"]
    window = default_window(n_episodes)
    n_seeds = sorted({len(r) for r in variants.values()})

    fig, (ax_curve, ax_bar) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
    names = list(variants)
    labels = [f"{v} (elegida)" if v == chosen else v for v in names]
    for i, (runs, label) in enumerate(zip(variants.values(), labels)):
        xs, curves = seed_curves(runs, window)
        ax_curve.plot(xs, curves.mean(axis=0), color=PALETTE[i], ls=LINESTYLES[i], label=label)
    ax_curve.set_xlabel("Episodio de entrenamiento")
    ax_curve.set_ylabel(Y_STEPS)
    ax_curve.set_ylim(0, 720)
    # Abajo a la izquierda las curvas ya bajaron poco o todavía están arriba: es la zona libre.
    ax_curve.legend(title=parameter, fontsize=8, title_fontsize=8.5, loc="lower left",
                    ncol=2 if len(names) > 4 else 1)
    ax_curve.set_title("Aprendizaje (media móvil, promedio entre semillas)")

    means = [final_steps(variants[v]).mean() for v in names]
    stds = [final_steps(variants[v]).std() for v in names]
    ys = np.arange(len(names))[::-1]
    ax_bar.barh(ys, means, xerr=stds, height=0.6, color=PALETTE[:len(names)],
                error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1, "capsize": 3})
    for y, m, s in zip(ys, means, stds):
        ax_bar.text(min(m + s, 700) + 12, y, f"{m:.0f}", va="center", fontsize=8.5)
    ax_bar.set_yticks(ys, labels, fontsize=8.5)
    for tick, variant in zip(ax_bar.get_yticklabels(), names):
        if variant == chosen:
            tick.set_fontweight("bold")
    ax_bar.set_xlim(0, 790)
    ax_bar.grid(axis="y", visible=False)
    ax_bar.grid(axis="x", visible=True)
    ax_bar.set_xlabel("Pasos greedy finales (media ± desvío)")
    ax_bar.set_title("Resultado final")

    fig.suptitle(f"Barrido de {parameter}, con Q₀ = {q0}", fontsize=12, x=0.01, ha="left")
    seeds = " o ".join(str(s) for s in n_seeds)
    footnote(fig, f"Experimento{'s' if len(experiments) > 1 else ''} {', '.join(experiments)}: {seeds} semillas "
                  f"por variante, {n_episodes} episodios. Izquierda: media móvil de {window} episodios de "
                  f"entrenamiento. Derecha: evaluación greedy final sobre 100 estados fijos; barras de error: "
                  f"±1 desvío entre semillas.")
    fig.tight_layout()
    save(fig, name)


def figure_a2():
    for sweep in SWEEPS:
        sweep_figure(*sweep)


# --- A3: Q0 = 0 contra Q0 = -100 -------------------------------------------------------
def figure_a3():
    fig, (ax_same, ax_tuned) = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)

    init = load_experiment("inicializacion")
    window = default_window(5000)
    for variant, key in [("0.0", "ql_cero"), ("-50.0", "q_medio"), ("-100.0", "ql_pes")]:
        st = STYLE[key]
        xs, curves = seed_curves(init[variant], window)
        plot_band(ax_same, xs, curves, st["color"], st["ls"], st["label"].split(", ")[1])
    ax_same.set_title("a) Misma configuración (la elegida con Q₀ = 0)")
    ax_same.set_xlabel("Episodio de entrenamiento")
    ax_same.set_ylabel(Y_STEPS)
    ax_same.set_ylim(0, 720)
    ax_same.legend(loc="center right")

    window = 250
    # El último elemento es el desplazamiento de la etiqueta (en puntos), para que no pise la otra curva.
    tuned = [("ql_cero", load_experiment("episodios")["20000"], MODEL_QL_CERO, -26),
             ("ql_pes", load_experiment("rango_final")["ql_10000"], latest_model(MODEL_QL), 30)]
    for key, runs, model, dy in tuned:
        st = STYLE[key]
        xs, curves = seed_curves(runs, window)
        mean = plot_band(ax_tuned, xs, curves, st["color"], st["ls"], st["label"])
        test = load_test(model)
        ax_tuned.annotate(f"prueba: {test['mean_steps']:.0f} pasos", (xs[-1], mean[-1]),
                          xytext=(0, dy), textcoords="offset points", ha="right", fontsize=8.5,
                          color=TEXT_SECONDARY)
    ax_tuned.set_title("b) Cada una con sus hiperparámetros ajustados")
    ax_tuned.set_xlabel("Episodio de entrenamiento")
    ax_tuned.legend(loc="upper right")

    fig.suptitle("Inicialización de la tabla Q: optimista (Q₀ = 0) contra pesimista (Q₀ = −100)",
                 fontsize=12, x=0.01, ha="left")
    footnote(fig, "5 semillas por curva; banda: ±1 desvío entre semillas. a) Experimento inicializacion: α = 0.1, "
                  "bins (8, 8, 12, 12), ε con decay 0.999, 5000 episodios, media móvil de 125. Con Q₀ = −100 la "
                  "tabla no se mueve hasta la primera llegada (punto fijo, D-17). b) Q₀ = 0: 20 000 episodios "
                  "(D-15); Q₀ = −100: 10 000 episodios (D-18 y D-38); media móvil de 250. 'prueba': modelo guardado "
                  "sobre los 1000 estados de prueba.")
    fig.tight_layout()
    save(fig, "A3_inicializacion")


# --- A4: distribución de pasos en la prueba --------------------------------------------
def figure_a4():
    agents = [("dynaq", latest_model(MODEL_DYNAQ)),
              ("ql_pes", latest_model(MODEL_QL)),
              ("ql_cero", MODEL_QL_CERO),
              ("random", "random")]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ys = np.arange(len(agents))[::-1]
    for y, (key, stem) in zip(ys, agents):
        test = load_test(stem)
        steps = np.array(test["steps_per_episode"])
        st = STYLE[key]
        ax.boxplot(steps, positions=[y], vert=False, widths=0.5, patch_artist=True,
                   boxprops={"facecolor": st["color"], "alpha": 0.35, "edgecolor": st["color"]},
                   medianprops={"color": st["color"], "linewidth": 2},
                   whiskerprops={"color": st["color"]}, capprops={"color": st["color"]},
                   flierprops={"marker": "o", "markersize": 3, "markerfacecolor": st["color"],
                               "markeredgecolor": "none", "alpha": 0.6})
        ax.text(712, y, f"media {test['mean_steps']:.0f} · mín. {test['min_steps']} · máx. {test['max_steps']}"
                        f" · éxito {test['success_rate']:.1%}", va="center", fontsize=8.5)
    ax.set_yticks(ys, [STYLE[k]["label"] for k, _ in agents])
    ax.axvline(700, color=GRAY, lw=1, ls="--")
    ax.text(700, len(agents) - 0.45, "corte (700)", ha="center", fontsize=8, color=TEXT_SECONDARY)
    ax.set_xlim(0, 705)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("Pasos hasta la meta en cada estado de prueba (menos es mejor)")
    ax.set_title("Distribución de pasos sobre los 1000 estados de prueba", loc="left")
    footnote(fig, "Política greedy de cada modelo guardado sobre los mismos 1000 estados iniciales de prueba "
                  "(semillas desde 2 000 000), que no se usaron para elegir nada (D-09, D-35). Caja: cuartiles y "
                  "mediana; bigotes: 1.5 veces el rango intercuartil; puntos: estados atípicos. Un episodio que "
                  "llega a 700 pasos es un fallo.")
    save(fig, "A4_distribucion_prueba")


# --- A5: Q-Learning contra Dyna-Q ------------------------------------------------------
def two_axes_figure(series, title, note, name):
    """Dos paneles con las mismas curvas: eje x en episodios y eje x en tiempo de reloj."""
    fig, (ax_ep, ax_t) = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    window = default_window(10000)
    for runs, color, ls, label in series:
        xs, curves = seed_curves(runs, window, x="episode")
        plot_band(ax_ep, xs, curves, color, ls, label, band=len(series) <= 3)
        ts, curves_t = seed_curves(runs, window, x="time")
        plot_band(ax_t, ts / 60, curves_t, color, ls, label, band=len(series) <= 3)
    ax_ep.set_xlabel("Episodio de entrenamiento")
    ax_ep.set_ylabel(Y_STEPS)
    ax_ep.set_title("a) Por episodios (eficiencia muestral)")
    ax_t.set_xlabel("Tiempo de reloj de entrenamiento (minutos)")
    ax_t.set_title("b) Por tiempo de cómputo")
    ax_ep.set_ylim(0, 720)
    ax_ep.legend(loc="lower left", fontsize=8.5)  # abajo no hay curvas: ninguna baja de 150 pasos
    fig.suptitle(title, fontsize=12, x=0.01, ha="left")
    footnote(fig, note + f" Media móvil de {window} episodios; promedio entre semillas.")
    fig.tight_layout()
    save(fig, name)


def figure_a5():
    dyna_n = load_experiment("dyna_n")
    series = []
    for i, (variant, runs) in enumerate(dyna_n.items()):
        label = "n = 0 (Q-Learning)" if variant == "0" else f"n = {variant}"
        series.append((runs, PALETTE[i], LINESTYLES[i], label))
    two_axes_figure(series, "Pasos de planning n con α = 0.5 (la α de Q-Learning)",
                    "Experimento dyna_n: Q₀ = −100, bins (6, 6, 10, 10), velocidades en la mitad de los límites "
                    "(el rango anterior a D-38), 10 000 episodios, 5 semillas por n, "
                    "todas en el mismo lote de 10 procesos, así que los tiempos son comparables entre sí (D-21, "
                    "D-22). Con α = 0.5, más planning no ayuda y con n ≥ 25 hay semillas que no aprenden (aliasing, "
                    "D-23).", "A5_dyna_n")

    final = load_experiment("rango_final")
    series = [(final["ql_10000"], STYLE["ql_pes"]["color"], STYLE["ql_pes"]["ls"], "Q-Learning (α = 0.5)"),
              (final["dynaq_10000"], STYLE["dynaq"]["color"], STYLE["dynaq"]["ls"], STYLE["dynaq"]["label"])]
    two_axes_figure(series, "Q-Learning contra Dyna-Q, cada uno con su configuración final",
                    "Experimento rango_final: Q₀ = −100, bins (6, 6, 10, 10), velocidades en ±5 y ±10 (D-38), mismo "
                    "ε, 10 000 episodios, 5 semillas; banda: ±1 desvío entre semillas. Los dos algoritmos corrieron "
                    "en el mismo lote de 10 procesos, así que los tiempos son comparables.",
                    "A5_qlearning_vs_dynaq")


FIGURES = {"A1": figure_a1, "A2": figure_a2, "A3": figure_a3, "A4": figure_a4, "A5": figure_a5}

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
