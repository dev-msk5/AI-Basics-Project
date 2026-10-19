import time

import numpy as np

from generator.points import (
    reset_seed,
    circle_points,
    ellipse_points,
    polygon_points,
    random_points,
    add_outliers,
)
from model.base import Circle
from plot.plot_points import compare_cost_histories, plot_solutions_overlay
from algorithms.algorithms import hill_climbing, simulated_annealing, genetic_algorithm

SEED = 42


def run_method(name, method, pts, **kwargs):
    """
    Run one search method with its own seeded generator and measure the time

    Parameters:
    name (str): The name shown in the printout
    method (callable): The search function, called as method(pts, rng=rng, **kwargs)
    pts (np.ndarray): An array of shape (N, 2) with the points
    kwargs: The parameters of the method

    Returns:
    best (Circle): The best circle found
    history (np.ndarray): The best cost after each iteration
    elapsed (float): The run time in seconds
    """
    rng = np.random.default_rng(SEED)
    start = time.perf_counter()
    best, history = method(pts, rng=rng, **kwargs)
    elapsed = time.perf_counter() - start
    print(f"{name}: center=({best.cx:.3f}, {best.cy:.3f}), "
          f"radius={best.r:.3f}, cost={best.cost:.4f}, time={elapsed:.3f} s")
    return best, history, elapsed


reset_seed(SEED)

# pick one test set, the others stay here to switch quickly
pts = circle_points(r=5, n=50, center=(1, -2), noise=0.3)
# the circle the points came from, only drawn as a reference
reference = Circle(1, -2, 5)
pts = circle_points(r=5, n=50, noise=0.3, span=np.pi)
# pts = ellipse_points(a=6, b=3, n=80, rotation=0.5, noise=0.1)
# pts = polygon_points(sides=8, r=5, n=80, noise=0.1)
pts = add_outliers(circle_points(r=5, n=100, noise=0.1), ratio=0.05, scale=10)
pts = random_points(n=50, x_range=(-5, 5), y_range=(-5, 5))

hc_best, hc_hist, hc_time = run_method(
    "Hill climbing", hill_climbing, pts,
    max_iter=300, step0=0.1, restarts=20, patience=30)

sa_best, sa_hist, sa_time = run_method(
    "Simulated annealing", simulated_annealing, pts,
    max_iter=300, T0=0.5, alpha=0.99, step0=0.1, chains=20)

ga_best, ga_hist, ga_time = run_method(
    "Genetic algorithm", genetic_algorithm, pts,
    pop_size=50, max_gen=200, pc=0.8, pm=0.2, sigma0=0.1,
    n_elite=2, k_tourn=3)

results = {
    "Hill climbing": hc_best,
    "Simulated annealing": sa_best,
    "Genetic algorithm": ga_best,
}

compare_cost_histories(
    {"Hill climbing": hc_hist,
     "Simulated annealing": sa_hist,
     "Genetic algorithm": ga_hist},
    evals_per_iter={"Hill climbing": 20,
                    "Simulated annealing": 20,
                    "Genetic algorithm": 50},
    logy=True)

plot_solutions_overlay(pts, results, true_circle=reference)