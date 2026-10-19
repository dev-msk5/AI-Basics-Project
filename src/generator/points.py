import numpy as np

SEED = 42
rng = np.random.default_rng(SEED)
 
def reset_seed(seed=SEED):
    """
    Recreate the global random generator so a run can be repeated exactly
 
    Parameters:
    seed (int): The seed of the new generator
 
    Returns:
    None
    """
    global rng
    rng = np.random.default_rng(seed)



def get_angles(n, start, span, random_angles):
    """
    Create the parameter angles that place the points along a curve

    Parameters:
    n (int): The number of angles to create
    start (float): The first angle in radians
    span (float): The angle size of the covered part in radians,
        2 * pi means the full curve
    random_angles (bool): If True the angles are drawn randomly from the
        covered part, otherwise they are evenly spaced

    Returns:
    np.ndarray: An array of shape (n,) with the angles in radians
    """
    if random_angles:
        return start + rng.uniform(0, span, n)
    full = np.isclose(span, 2 * np.pi)
    return start + np.linspace(0, span, n, endpoint=not full)


def rotate(points, angle):
    """
    Rotate points around the origin counterclockwise

    Parameters:
    points (np.ndarray): An array of shape (n, 2) with the x and y coordinates
    angle (float): The rotation angle in radians

    Returns:
    np.ndarray: An array of shape (n, 2) with the rotated points
    """
    c, s = np.cos(angle), np.sin(angle)
    return points @ np.array([[c, s], [-s, c]])


def add_noise(points, noise):
    """
    Add independent Gaussian noise to both coordinates of the points

    Parameters:
    points (np.ndarray): An array of shape (n, 2) with the x and y coordinates
    noise (float): The standard deviation of the Gaussian noise,
        0 returns the points unchanged

    Returns:
    np.ndarray: A new array of shape (n, 2) with the noisy points
    """
    return points + rng.normal(0, noise, points.shape)


def add_outliers(points, ratio, scale):
    """
    Move a random part of the points away from their original place

    Parameters:
    points (np.ndarray): An array of shape (n, 2) with the x and y coordinates
    ratio (float): The part of the points to move, between 0 and 1
    scale (float): The largest shift per coordinate, the shift is drawn
        uniformly from (-scale, scale)

    Returns:
    np.ndarray: A new array of shape (n, 2), the input is not modified
    """
    n = int(len(points) * ratio)
    if n == 0:
        return points
    points = points.copy()
    idx = rng.choice(len(points), n, replace=False)
    points[idx] += rng.uniform(-scale, scale, (n, 2))

    return points


def circle_points(r, n, center=(0, 0), noise=0, start=0,
                  span=2 * np.pi, random_angles=False):
    """
    Generate points on a circle or on an arc of it

    Parameters:
    r (float): The radius of the circle
    n (int): The number of points to generate
    center (tuple): The (x, y) center of the circle
    noise (float): The standard deviation of the Gaussian noise
    start (float): The angle where the arc begins in radians
    span (float): The angular size of the arc in radians,
        2 * pi gives the full circle
    random_angles (bool): If True the points are placed at random angles,
        otherwise they are evenly spaced

    Returns:
    np.ndarray: An array of shape (n, 2) with the x and y coordinates
    """
    t = get_angles(n, start, span, random_angles)
    pts = np.column_stack((r * np.cos(t), r * np.sin(t)))

    return add_noise(pts, noise) + center


def ellipse_points(a, b, n, center=(0, 0), rotation=0, noise=0,
                   start=0, span=2 * np.pi, random_angles=False):
    """
    Generate points on an ellipse or on an arc of it

    Parameters:
    a (float): The semi axis along the x direction before rotation
    b (float): The semi axis along the y direction before rotation
    n (int): The number of points to generate
    center (tuple): The (x, y) center of the ellipse
    rotation (float): The rotation of the ellipse in radians
    noise (float): The standard deviation of the Gaussian noise
    start (float): The parameter angle where the arc begins in radians
    span (float): The size of the covered parameter range in radians,
        2 * pi gives the full ellipse
    random_angles (bool): If True the parameter angles are random,
        otherwise they are evenly spaced

    Returns:
    np.ndarray: An array of shape (n, 2) with the x and y coordinates
    """
    t = get_angles(n, start, span, random_angles)
    pts = np.column_stack((a * np.cos(t), b * np.sin(t)))
    pts = rotate(pts, rotation)

    return add_noise(pts, noise) + center


def polygon_points(sides, r, n, center=(0, 0), rotation=0, noise=0,
                   coverage=1.0):
    """
    Generate points along the perimeter of a regular polygon

    The points are evenly spaced along the perimeter, so the corners are
    only hit exactly when n is a multiple of sides

    Parameters:
    sides (int): The number of sides of the polygon
    r (float): The radius of the circumscribed circle
    n (int): The number of points to generate
    center (tuple): The (x, y) center of the polygon
    rotation (float): The rotation of the polygon in radians
    noise (float): The standard deviation of the Gaussian noise
    coverage (float): The part of the perimeter that gets points,
        between 0 and 1, 1 means the whole polygon

    Returns:
    np.ndarray: An array of shape (n, 2) with the x and y coordinates
    """
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    vx = np.append(r * np.cos(ang), r)
    vy = np.append(r * np.sin(ang), 0)
    seg = np.hypot(np.diff(vx), np.diff(vy))
    cum = np.concatenate(([0], np.cumsum(seg)))
    s = np.linspace(0, cum[-1] * coverage, n, endpoint=(coverage < 1))
    pts = np.column_stack((np.interp(s, cum, vx), np.interp(s, cum, vy)))
    pts = rotate(pts, rotation)

    return add_noise(pts, noise) + center


def random_points(n, x_range, y_range):
    """
    Generate uniformly distributed points inside a rectangle,
    used as a structureless test case

    Parameters:
    n (int): The number of points to generate
    x_range (tuple): The (min, max) range of the x coordinates
    y_range (tuple): The (min, max) range of the y coordinates

    Returns:
    np.ndarray: An array of shape (n, 2) with the x and y coordinates
    """
    x = rng.uniform(x_range[0], x_range[1], n)
    y = rng.uniform(y_range[0], y_range[1], n)

    return np.column_stack((x, y))