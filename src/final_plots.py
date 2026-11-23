
import itertools
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from generator.points import (
    reset_seed,
    circle_points,
    ellipse_points,
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
SELECTED = ["clean", "circle", "arc", "polygon", "random", "ellipse"]


def exact_circle(pts):
    """
    Find the exact smallest enclosing circle by checking every circle that
    is fixed by two or three points, it is meant for small sets (N < 200)

    Parameters:
    pts (np.ndarray): An array of shape (N, 2) with the points

    Returns:
    tuple: The centre x, the centre y and the radius of the circle
    """
    best = None

    def covers(c, r):
        return np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]).max() <= r + 1e-9

    for i, j in itertools.combinations(range(len(pts)), 2):
        c = (pts[i] + pts[j]) / 2
        r = np.linalg.norm(pts[i] - c)
        if (best is None or r < best[2]) and covers(c, r):
            best = (c[0], c[1], r)
    for i, j, k in itertools.combinations(range(len(pts)), 3):
        a, b, d = pts[i], pts[j], pts[k]
        den = 2 * (a[0] * (b[1] - d[1]) + b[0] * (d[1] - a[1])
                   + d[0] * (a[1] - b[1]))
        if abs(den) < 1e-12:
            continue
        ux = (a @ a * (b[1] - d[1]) + b @ b * (d[1] - a[1])
              + d @ d * (a[1] - b[1])) / den
        uy = (a @ a * (d[0] - b[0]) + b @ b * (a[0] - d[0])
              + d @ d * (b[0] - a[0])) / den
        r = np.hypot(a[0] - ux, a[1] - uy)
        if (best is None or r < best[2]) and covers((ux, uy), r):
            best = (ux, uy, r)
    return best


def make_test_sets():
    """
    Build the five test point sets, every set starts from the same seed

    Parameters:
    None

    Returns:
    dict: Maps the file name of the set to a tuple of the points, the
        exact smallest enclosing circle of the points (the reference) and
        the radius of that circle, which is also the cost of the optimum
    """
    points = {}

    reset_seed(SEED)
    points["clean"] = circle_points(r=5, n=50, center=(1, -2), noise=0)

    reset_seed(SEED)
    points["circle"] = circle_points(r=5, n=50, center=(1, -2), noise=0.3)

    reset_seed(SEED)
    points["arc"] = circle_points(r=5, n=50, noise=0.3, span=np.pi)

    reset_seed(SEED)
    points["polygon"] = polygon_points(sides=8, r=5, n=80, noise=0.1)

    reset_seed(SEED)
    points["random"] = random_points(n=50, x_range=(-5, 5), y_range=(-5, 5))

    reset_seed(SEED)
    points["ellipse"] = ellipse_points(
        a=7, b=4, n=80,
        center=(1, -2),
        rotation=np.deg2rad(25),
        noise=0.1
    )

    # the reference is the exact optimum of every set, not the circle the
    # points were generated from, because noise moves the optimum away from it
    sets = {}
    for name, pts in points.items():
        cx, cy, r = exact_circle(pts)
        sets[name] = (pts, Circle(cx, cy, r), float(r))
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
    axis.set_major_formatter(
        FuncFormatter(lambda v, pos: f"{v:g}".replace(".", ",")))


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
    reference (Circle): The exact smallest enclosing circle, None to skip it
    path (str): The output file
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
    set as figures/<set>_history.png and figures/<set>_overlay.png, the
    radius of the exact optimum and of every found circle is printed too

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

        # the radii are needed for the numbers in the text of the paper
        found = ", ".join(f"{m}: {c.r:.3f}" for m, c in solutions.items())
        print(f"{name}: optimum {optimum:.3f}, {found}")


if __name__ == "__main__":
    main()
