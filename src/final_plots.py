import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from generator.points import (
    reset_seed,
    circle_points,
    polygon_points,
    random_points,
)
from model.base import Circle
from plot.plot_points import compare_cost_histories, plot_solutions_overlay
from algorithms.algorithms import hill_climbing, simulated_annealing, genetic_algorithm

SEED = 42

# the figures are saved here, next to this file
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# one column of the journal is 80 mm wide, the figures are made in that size
WIDTH = 3.15
DPI = 600

# method names as they appear in the legend, with the parameters of each run
METHODS = {
    "Hegymászó": (hill_climbing,
                  dict(max_iter=300, step0=0.1, restarts=20, patience=30)),
    "Szimulált hűtés": (simulated_annealing,
                        dict(max_iter=300, T0=0.5, alpha=0.99, step0=0.1,
                             chains=20)),
    "Genetikus algoritmus": (genetic_algorithm,
                             dict(pop_size=50, max_gen=200, pc=0.8, pm=0.2,
                                  sigma0=0.1, n_elite=2, k_tourn=3)),
}

# number of circle evaluations in one iteration, used for the x axis
EVALS_PER_ITER = {"Hegymászó": 20, "Szimulált hűtés": 20,
                  "Genetikus algoritmus": 50}

# the plot functions write their legend in English, this maps it to Hungarian
LEGEND = {"points": "pontok", "reference": "referencia",
          "known optimum": "pontos optimum"}

# the sets that get figures, remove a name to skip that set
SELECTED = ["clean", "circle", "arc", "polygon", "random"]


def make_test_sets():
    """
    Build the four test point sets, every set starts from the same seed

    Parameters:
    None

    Returns:
    dict: Maps the file name of the set to a tuple of the points, the
        circle the points came from (None if there is no such circle) and
        the radius of the exact optimum (None if it is not known)
    """
    sets = {}

    # no noise, so the generating circle is the exact smallest enclosing circle
    reset_seed(SEED)
    sets["clean"] = (circle_points(r=5, n=50, center=(1, -2), noise=0),
                     Circle(1, -2, 5), 5.0)

    reset_seed(SEED)
    sets["circle"] = (circle_points(r=5, n=50, center=(1, -2), noise=0.3),
                      Circle(1, -2, 5), None)

    reset_seed(SEED)
    sets["arc"] = (circle_points(r=5, n=50, noise=0.3, span=np.pi),
                   Circle(0, 0, 5), None)

    reset_seed(SEED)
    sets["polygon"] = (polygon_points(sides=8, r=5, n=80, noise=0.1),
                       Circle(0, 0, 5), None)

    reset_seed(SEED)
    sets["random"] = (random_points(n=50, x_range=(-5, 5), y_range=(-5, 5)),
                      None, None)
    return sets


def run_methods(pts):
    """
    Run all three methods on a point set, every method gets its own
    generator with the same seed

    Parameters:
    pts (np.ndarray): An array of shape (N, 2) with the points

    Returns:
    histories (dict): Maps the method name to its cost history
    solutions (dict): Maps the method name to the circle it found
    """
    histories, solutions = {}, {}
    for name, (method, params) in METHODS.items():
        rng = np.random.default_rng(SEED)
        best, history = method(pts, rng=rng, **params)
        histories[name] = history
        solutions[name] = best
    return histories, solutions


def use_decimal_comma(axis):
    """
    Make the tick labels of an axis use a decimal comma

    Parameters:
    axis (matplotlib.axis.Axis): The axis to change

    Returns:
    None
    """
    old = axis.get_major_formatter()
    axis.set_major_formatter(
        FuncFormatter(lambda v, pos: old(v, pos).replace(".", ",")))


def translate_legend(ax, **kwargs):
    """
    Rebuild the legend with Hungarian labels and decimal commas

    Parameters:
    ax (matplotlib.axes.Axes): The Axes whose legend is rebuilt
    kwargs: Extra options passed to ax.legend

    Returns:
    None
    """
    handles, labels = ax.get_legend_handles_labels()
    labels = [LEGEND.get(t, t).replace(".", ",") for t in labels]
    ax.legend(handles, labels, **kwargs)


def save_history(histories, path, optimum=None):
    """
    Save the cost history comparison of the three methods

    Parameters:
    histories (dict): Maps the method name to its cost history
    path (str): The output file
    optimum (float): The cost of the exact optimum, drawn as a horizontal
        line, None to skip it

    Returns:
    None
    """
    fig, ax = plt.subplots(figsize=(WIDTH, 2.35))
    compare_cost_histories(histories, evals_per_iter=EVALS_PER_ITER,
                           logy=True, reference_cost=optimum, title="", ax=ax,
                           show=False)
    ax.set_xlabel("költségkiértékelések száma")
    ax.set_ylabel("legjobb költség")
    use_decimal_comma(ax.xaxis)
    translate_legend(ax, fontsize=5.5, loc="upper right")
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def save_overlay(pts, solutions, reference, path):
    """
    Save the circles of the three methods drawn over the point set

    Parameters:
    pts (np.ndarray): An array of shape (N, 2) with the points
    solutions (dict): Maps the method name to the circle it found
    reference (Circle): The circle the points came from, None to skip it
    path (str): The output file

    Returns:
    None
    """
    fig, ax = plt.subplots(figsize=(WIDTH, 3.0))
    plot_solutions_overlay(pts, solutions, true_circle=reference, title="",
                           ax=ax, show=False)
    use_decimal_comma(ax.xaxis)
    use_decimal_comma(ax.yaxis)
    translate_legend(ax, fontsize=5.5, loc="upper center",
                     bbox_to_anchor=(0.5, -0.07), ncol=2, frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def main():
    """
    Run the three methods on the selected test sets and save two figures per
    set as figures/<set>_history.png and figures/<set>_overlay.png

    Parameters:
    None

    Returns:
    None
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 6.5,
        "axes.linewidth": 0.5,
        "lines.linewidth": 0.9,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
    })
    os.makedirs(OUT_DIR, exist_ok=True)

    for name, (pts, reference, optimum) in make_test_sets().items():
        if name not in SELECTED:
            continue
        histories, solutions = run_methods(pts)
        save_history(histories, os.path.join(OUT_DIR, f"{name}_history.png"),
                     optimum)
        save_overlay(pts, solutions, reference,
                     os.path.join(OUT_DIR, f"{name}_overlay.png"))


if __name__ == "__main__":
    main()
