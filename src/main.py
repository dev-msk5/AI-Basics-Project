from generator.points import (
    generate_circle_points,
    generate_ellipse_points,
    generate_polygon_points,
    generate_random_points,
)
from plot.plot_points import plot_circle_points
from model.base import Circle

import matplotlib.pyplot as plt


generated_polygon_points = generate_polygon_points(
    num_sides=8, radius=5, num_points=50, noise_level=0)

generated_random_points = generate_random_points(
    x_range=(-5, 5), y_range=(-5, 5), num_points=50)

generated_circle_points = generate_circle_points(
    radius=5, num_points=50, noise_level=0.0)

test_circle = Circle(cx=0, cy=0, r=5)
print(test_circle.evaluate_cost(generated_random_points))
print(test_circle.evaluate_cost(generated_polygon_points))
print(test_circle.evaluate_cost(generated_circle_points))

plot_circle_points(generated_circle_points,
                   title='Circle Points with Noise at 360 Degrees', radius=5)
