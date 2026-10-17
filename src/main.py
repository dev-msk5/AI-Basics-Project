from generator.points import (
    generate_circle_points,
    generate_ellipse_points,
    generate_polygon_points,
    generate_random_points,
)
from plot.plot_points import plot_circle_points, plot_circle_points_with_prediction, plot_cost_history
from model.base import Circle
from algorithms.algorithms import hill_climbing

import matplotlib.pyplot as plt
import numpy as np


generated_random_points = generate_polygon_points(
    num_sides=8, radius=5, num_points=50, noise_level=0.3)

# generated_random_points = generate_random_points(
#     x_range=(-5, 5), y_range=(-5, 5), num_points=50)

# generated_circle_points = generate_circle_points(
#     radius=5, num_points=50, noise_level=0.0)

test_circle = Circle(cx=0, cy=0, r=5)
# print(test_circle.evaluate_cost(generated_random_points))
# print(test_circle.evaluate_cost(generated_polygon_points))
# print(test_circle.evaluate_cost(generated_circle_points))

best, history = hill_climbing(generated_random_points, max_iter=100, step0=0.2, restarts=5, patience=50, rng=np.random.default_rng())
plot_cost_history(history)
plot_circle_points_with_prediction(generated_random_points, title='Best-fitting Circle Points', radius=5, predicted_circle=best)
print(f"Best-fitting circle parameters: Center=({best.cx:.2f}, {best.cy:.2f}), Radius={best.r:.2f}, Cost={best.cost:.4f}")