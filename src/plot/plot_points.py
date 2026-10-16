import matplotlib.pyplot as plt
import numpy as np  


def plot_circle_points(points, title, radius):
    """
    Plot the generated circle points.

    Parameters:
    points (np.ndarray): An array of shape (num_points, 2) containing the x and y coordinates of the points.
    title (str): The title of the plot.
    radius (float): The radius of the circle.
    """
    plt.figure(figsize=(radius*1.2, radius*1.2))
    plt.scatter(points[:, 0], points[:, 1],
                color='blue', label='Circle Points')
    plt.title(title)
    plt.xlabel('X-tengely')
    plt.ylabel('Y-tengely')
    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.show()

def plot_circle_points_with_prediction(points, title, radius, predicted_circle):
    """
    Plot the generated circle points along with the predicted circle.

    Parameters:
    points (np.ndarray): An array of shape (num_points, 2) containing the x and y coordinates of the points.
    title (str): The title of the plot.
    radius (float): The radius of the circle.
    predicted_circle (Circle): The predicted circle object containing center and radius.
    """
    plt.figure(figsize=(radius*1.2, radius*1.2))
    plt.scatter(points[:, 0], points[:, 1],
                color='blue', label='Circle Points')
    
    # Plot the predicted circle
    theta = np.linspace(0, 2 * np.pi, 100)
    x_pred = predicted_circle.cx + predicted_circle.r * np.cos(theta)
    y_pred = predicted_circle.cy + predicted_circle.r * np.sin(theta)
    plt.plot(x_pred, y_pred, color='red', label='Predicted Circle', linewidth=2)

    plt.title(title)
    plt.xlabel('X-tengely')
    plt.ylabel('Y-tengely')
    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.show()

def plot_cost_history(history):
    """
    Plot the cost history during the optimization process.

    Parameters:
    history (list): A list of the best costs found during the search.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history, color='red', label='Cost History')
    plt.title('Cost History During Optimization')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid()
    plt.legend()
    plt.show()

# generated_points = generate_circle_points(radius=5, num_points=50, noise_level=0.1)
# plot_circle_points(generated_points, title='Circle Points with Noise at 360 Degrees', radius=5)

# generate_ellipse_points = generate_ellipse_points(a=5, b=3, num_points=50, noise_level=0)
# plot_circle_points(generate_ellipse_points, title='Ellipse Points with Noise', radius=5)
