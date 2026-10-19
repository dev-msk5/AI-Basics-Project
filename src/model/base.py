import numpy as np

# global lambda, penatly weight
LAM = 2.0


class Circle:
    """
    A circle given by its center and radius, with an optional stored cost

    Parameters:
    cx (float): The x coordinate of the center
    cy (float): The y coordinate of the center
    r (float): The radius
    cost (float): The cost of the circle if it was already computed
    """

    __slots__ = ("cx", "cy", "r", "cost")

    def __init__(self, cx, cy, r, cost=None):
        """
        Create a circle

        Parameters:
        cx (float): The x coordinate of the center
        cy (float): The y coordinate of the center
        r (float): The radius
        cost (float): The cost of the circle if it was already computed

        Returns:
        None
        """
        self.cx = cx
        self.cy = cy
        self.r = r
        self.cost = cost

    def params(self):
        """
        Pack the circle into a parameter vector

        Parameters:
        None

        Returns:
        np.ndarray: An array of shape (3,) with cx, cy, r
        """
        return np.array([self.cx, self.cy, self.r])

    @classmethod
    def from_params(cls, p, cost=None):
        """
        Build a circle from a parameter vector

        Parameters:
        p (np.ndarray): An array of shape (3,) with cx, cy, r
        cost (float): The cost of the circle if it was already computed

        Returns:
        Circle: The new circle
        """
        return cls(p[0], p[1], p[2], cost)

    def distances(self, pts):
        """
        Measure the distance of every point from the center

        Parameters:
        pts (np.ndarray): An array of shape (N, 2) with the points

        Returns:
        np.ndarray: An array of shape (N,) with the distances
        """
        return np.hypot(pts[:, 0] - self.cx, pts[:, 1] - self.cy)

    def evaluate_cost(self, pts, lam=LAM):
        """
        Compute the radius plus lam times the total overshoot of the points
        that are outside, the result is also stored in self.cost

        Parameters:
        pts (np.ndarray): An array of shape (N, 2) with the points
        lam (float): The penalty weight, it must be above 1

        Returns:
        float: The cost of the circle
        """
        d = self.distances(pts)
        out = np.maximum(d - self.r, 0.0).sum()
        self.cost = float(self.r + lam * out)
        return self.cost

    def tighten(self, pts):
        """
        Set the radius to the distance of the farthest point, so the circle
        is valid and as small as possible for its center

        Parameters:
        pts (np.ndarray): An array of shape (N, 2) with the points

        Returns:
        Circle: The same circle with the updated radius and cost
        """
        self.r = float(self.distances(pts).max())
        self.cost = self.r
        return self


class CostBatch:
    """
    Cost of many circles at once, with buffers that are allocated only once
    Speeds up the evaluation of many circles in one call

    Parameters:
    pts (np.ndarray): An array of shape (N, 2) with the points
    restarts (int): The number of circles evaluated in one call, the input of
        a call must have exactly this many rows
    lam (float): The penalty weight, it must be above 1
    """

    def __init__(self, pts, restarts, lam=LAM):
        """
        Prepare the point columns and the work buffers

        Parameters:
        pts (np.ndarray): An array of shape (N, 2) with the points
        restarts (int): The number of circles evaluated in one call
        lam (float): The penalty weight, it must be above 1

        Returns:
        None
        """
        self.px = np.ascontiguousarray(pts[:, 0], dtype=float)
        self.py = np.ascontiguousarray(pts[:, 1], dtype=float)
        self.lam = lam
        n = len(self.px)
        self.a = np.empty((restarts, n))
        self.b = np.empty((restarts, n))

    def __call__(self, X):
        """
        Compute the cost of every circle in the batch

        Parameters:
        X (np.ndarray): An array of shape (restarts, 3) with the columns
            cx, cy, r

        Returns:
        np.ndarray: A new array of shape (restarts,) with the costs
        """
        a, b = self.a, self.b
        # offset of each point from the center of each circle
        np.subtract(self.px, X[:, 0:1], out=a)
        np.subtract(self.py, X[:, 1:2], out=b)
        # pythagorean distance
        np.multiply(a, a, out=a)
        np.multiply(b, b, out=b)
        np.add(a, b, out=a)
        np.sqrt(a, out=a)
        # overshoot of each point, zero when it is inside
        np.subtract(a, X[:, 2:3], out=a)
        np.maximum(a, 0.0, out=a)

        # overshoots summed and multiplied by lam, plus the radius
        total = a.sum(axis=1)
        total *= self.lam
        total += X[:, 2]
        return total