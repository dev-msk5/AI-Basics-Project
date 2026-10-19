"""
Plot helpers for the circle fitting task

Every function draws on a given Axes or makes a new figure, and returns the Axes
so the plot can be changed afterwards, set show=False to skip plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as CirclePatch


def prepare_axes(ax, figsize=(7, 6)):
    """
    Return the given Axes or create a new figure with one

    Parameters:
    ax (matplotlib.axes.Axes): The Axes to draw on, or None for a new figure
    figsize (tuple): The size of the new figure in inches

    Returns:
    matplotlib.axes.Axes: The Axes to draw on
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def draw_circle(ax, cx, cy, r, label=None, **style):
    """
    Draw the outline of a circle and mark its center

    Parameters:
    ax (matplotlib.axes.Axes): The Axes to draw on
    cx (float): The x coordinate of the center
    cy (float): The y coordinate of the center
    r (float): The radius
    label (str): The legend text of the circle
    style: Extra matplotlib style options such as color, linestyle, linewidth

    Returns:
    None
    """
    color = style.pop("color", None)
    patch = CirclePatch((cx, cy), r, fill=False, label=label,
                        color=color, **style)
    ax.add_patch(patch)
    ax.plot(cx, cy, marker="+", color=patch.get_edgecolor(), markersize=8)


def finish_axes(ax, title, show):
    """
    Apply the common look to an Axes and optionally show the figure

    Parameters:
    ax (matplotlib.axes.Axes): The Axes to finish
    title (str): The title of the plot
    show (bool): If True plt.show() is called

    Returns:
    None
    """
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale_view()
    ax.grid(alpha=0.3)
    ax.legend()
    if show:
        plt.show()


def plot_circle_points(pts, title="Points", ax=None, show=True):
    """
    Plot a point set

    Parameters:
    pts (np.ndarray): An array of shape (N, 2) with the points
    title (str): The title of the plot
    ax (matplotlib.axes.Axes): The Axes to draw on, or None for a new figure
    show (bool): If True plt.show() is called

    Returns:
    matplotlib.axes.Axes: The Axes with the plot
    """
    ax = prepare_axes(ax)
    ax.scatter(pts[:, 0], pts[:, 1], s=14, color="tab:blue", label="points")
    finish_axes(ax, title, show)
    return ax


def plot_circle_points_with_prediction(pts, title="Best-fitting circle",
                                       radius=None, center=(0, 0),
                                       predicted_circle=None, ax=None,
                                       show=True):
    """
    Plot a point set, the fitted circle and optionally the circle the points
    were generated from

    Parameters:
    pts (np.ndarray): An array of shape (N, 2) with the points
    title (str): The title of the plot
    radius (float): The radius of the generating circle, None to skip it
    center (tuple): The (x, y) center of the generating circle
    predicted_circle (Circle): The fitted circle, None to skip it
    ax (matplotlib.axes.Axes): The Axes to draw on, or None for a new figure
    show (bool): If True plt.show() is called

    Returns:
    matplotlib.axes.Axes: The Axes with the plot
    """
    ax = prepare_axes(ax)
    ax.scatter(pts[:, 0], pts[:, 1], s=14, color="tab:blue", label="points")
    if radius is not None:
        draw_circle(ax, center[0], center[1], radius, label="generating circle",
                    color="tab:green", linestyle="--")
    if predicted_circle is not None:
        c = predicted_circle
        draw_circle(ax, c.cx, c.cy, c.r, color="tab:red", linewidth=1.8,
                    label=f"fitted circle (r={c.r:.3f})")
    finish_axes(ax, title, show)
    return ax


def plot_cost_history(history, title="Cost history", ax=None, show=True):
    """
    Plot the best cost against the iteration for one run

    Parameters:
    history (np.ndarray): The best cost after each iteration, shape (n,)
    title (str): The title of the plot
    ax (matplotlib.axes.Axes): The Axes to draw on, or None for a new figure
    show (bool): If True plt.show() is called

    Returns:
    matplotlib.axes.Axes: The Axes with the plot
    """
    ax = prepare_axes(ax, figsize=(7, 4))
    ax.plot(np.arange(1, len(history) + 1), history, label="best cost")
    ax.set_xlabel("iteration")
    ax.set_ylabel("cost")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    if show:
        plt.show()
    return ax


def compare_cost_histories(histories, evals_per_iter=None, logy=False,
                           reference_cost=None, title="Cost history comparison",
                           ax=None, show=True):
    """
    Plot the cost histories of several methods on the same axes

    One iteration of different methods costs a different number of cost
    evaluations, so pass evals_per_iter to compare them on equal terms

    Parameters:
    histories (dict): Maps the method name to its history array
    evals_per_iter (dict): Maps the method name to the number of cost
        evaluations in one iteration, None plots against the iteration number
    logy (bool): If True the cost axis is logarithmic
    reference_cost (float): The cost of the known optimum, drawn as a
        horizontal line, None to skip it
    title (str): The title of the plot
    ax (matplotlib.axes.Axes): The Axes to draw on, or None for a new figure
    show (bool): If True plt.show() is called

    Returns:
    matplotlib.axes.Axes: The Axes with the plot
    """
    ax = prepare_axes(ax, figsize=(8, 5))
    for name, history in histories.items():
        steps = np.arange(1, len(history) + 1)
        if evals_per_iter is not None:
            steps = steps * evals_per_iter[name]
        ax.plot(steps, history, label=name)
    if reference_cost is not None:
        ax.axhline(reference_cost, color="black", linestyle=":",
                   label="known optimum")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("iteration" if evals_per_iter is None
                  else "cost evaluations")
    ax.set_ylabel("best cost")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    if show:
        plt.show()
    return ax


def plot_solutions_overlay(pts, solutions, true_circle=None,
                           title="Solutions on top of each other", ax=None,
                           show=True):
    """
    Draw the circles found by several methods over the same point set

    Parameters:
    pts (np.ndarray): An array of shape (N, 2) with the points
    solutions (dict): Maps the method name to its fitted Circle
    true_circle (Circle): The reference circle, for example the exact smallest
        enclosing circle or the generating circle, None to skip it
    title (str): The title of the plot
    ax (matplotlib.axes.Axes): The Axes to draw on, or None for a new figure
    show (bool): If True plt.show() is called

    Returns:
    matplotlib.axes.Axes: The Axes with the plot
    """
    ax = prepare_axes(ax)
    ax.scatter(pts[:, 0], pts[:, 1], s=14, color="gray", label="points")
    if true_circle is not None:
        draw_circle(ax, true_circle.cx, true_circle.cy, true_circle.r,
                    label="reference", color="black", linestyle=":",
                    linewidth=2)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (name, c) in enumerate(solutions.items()):
        draw_circle(ax, c.cx, c.cy, c.r, color=colors[i % len(colors)],
                    linewidth=1.5, label=f"{name} (r={c.r:.3f})")
    finish_axes(ax, title, show)
    return ax