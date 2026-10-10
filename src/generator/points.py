"Generate points for circle fitting, 360,180,90,45 degrees"

import numpy as np
import random as rd

seed = 42
rd.seed(seed)

def generate_circle_points(radius, num_points):
    """
    Generate points on a circle with a given radius and number of points.

    Parameters:
    radius (float): The radius of the circle.
    num_points (int): The number of points to generate.

    Returns:
    np.ndarray: An array of shape (num_points, 2) containing the x and y coordinates of the points.
    """
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.column_stack((x, y))

print("Generating points for circle fitting at 360, 180, 90, and 45 degrees.")
print("Generating points for 360 degrees:")
points_360 = generate_circle_points(radius=5, num_points=20)
print(points_360)
