from generator.points import generate_circle_points,generate_ellipse_points,generate_polygon_points
import matplotlib.pyplot as plt

def plot_circle_points(points, title, radius):
    """
    Plot the generated circle points.

    Parameters:
    points (np.ndarray): An array of shape (num_points, 2) containing the x and y coordinates of the points.
    title (str): The title of the plot.
    radius (float): The radius of the circle.
    """
    plt.figure(figsize=(radius*1.2, radius*1.2))
    plt.scatter(points[:, 0], points[:, 1], color='blue', label='Circle Points')
    plt.title(title)
    plt.xlabel('X-tengely')
    plt.ylabel('Y-tengely')
    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.show()


# generated_points = generate_circle_points(radius=5, num_points=50, noise_level=0.1)
# plot_circle_points(generated_points, title='Circle Points with Noise at 360 Degrees', radius=5)

# generate_ellipse_points = generate_ellipse_points(a=5, b=3, num_points=50, noise_level=0)
# plot_circle_points(generate_ellipse_points, title='Ellipse Points with Noise', radius=5)

generated_polygon_points = generate_polygon_points(num_sides=8, radius=5, num_points=50, noise_level=0)
plot_circle_points(generated_polygon_points, title='Polygon Points with Noise', radius=5)
