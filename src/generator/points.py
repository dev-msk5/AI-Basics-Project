"ponthalmaz generálás (szabályos kör, ellipszis, poligon stb.)"

import numpy as np
import random as rd

seed = 42
rd.seed(seed)


def generate_random_points(num_points, x_range, y_range):
    """
    Generate random points within a specified range.

    Parameters:
    num_points (int): The number of random points to generate.
    x_range (tuple): A tuple specifying the range for the x-coordinates (min, max).
    y_range (tuple): A tuple specifying the range for the y-coordinates (min, max).

    Returns:
    np.ndarray: An array of shape (num_points, 2) containing the x and y coordinates of the random points.
    """
    x = np.random.uniform(x_range[0], x_range[1], num_points)
    y = np.random.uniform(y_range[0], y_range[1], num_points)
    return np.column_stack((x, y))


def generate_circle_points(radius, num_points, noise_level=0):
    """
    Generate points on a circle with a given radius and number of points, adding random noise (optional).

    Parameters:
    radius (float): The radius of the circle.
    num_points (int): The number of points to generate.
    noise_level (float): The standard deviation of the Gaussian noise to add to the points.

    Returns:
    np.ndarray: An array of shape (num_points, 2) containing the x and y coordinates of the points with noise.
    """
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = radius * np.cos(angles) + np.random.normal(0, noise_level, num_points)
    y = radius * np.sin(angles) + np.random.normal(0, noise_level, num_points)
    return np.column_stack((x, y))


def generate_ellipse_points(a, b, num_points, noise_level=0):
    """
    Generate points on an ellipse with given semi-major axis (a) and semi-minor axis (b), adding random noise.

    Parameters:
    a (float): The semi-major axis of the ellipse.
    b (float): The semi-minor axis of the ellipse.
    num_points (int): The number of points to generate.
    noise_level (float): The standard deviation of the Gaussian noise to add to the points.

    Returns:
    np.ndarray: An array of shape (num_points, 2) containing the x and y coordinates of the points with noise.
    """
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = a * np.cos(angles) + np.random.normal(0, noise_level, num_points)
    y = b * np.sin(angles) + np.random.normal(0, noise_level, num_points)
    return np.column_stack((x, y))


def generate_polygon_points(num_sides, radius, num_points, noise_level=0):
    """
    Generate points on a regular polygon with a given number of sides and radius, adding random noise.

    Parameters:
    num_sides (int): The number of sides of the polygon.
    radius (float): The radius of the circumscribed circle of the polygon.
    num_points (int): The number of points to generate.
    noise_level (float): The standard deviation of the Gaussian noise to add to the points.

    Returns:
    np.ndarray: An array of shape (num_points, 2) containing the x and y coordinates of the points with noise.
    """
    angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
    vertices = np.column_stack(
        (radius * np.cos(angles), radius * np.sin(angles)))

    # Generate points along the edges of the polygon
    points = []
    for i in range(num_sides):
        start_vertex = vertices[i]
        end_vertex = vertices[(i + 1) % num_sides]
        edge_points = np.linspace(
            start_vertex, end_vertex, num_points // num_sides, endpoint=False)
        points.append(edge_points)

    points = np.vstack(points)

    # Add noise
    points += np.random.normal(0, noise_level, points.shape)

    return points
