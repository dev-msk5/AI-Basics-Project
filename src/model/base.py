import numpy as np

LAM = 2.0


class Circle:
    __slots__ = ("cx", "cy", "r", "cost")

    def __init__(self, cx, cy, r, cost=None):
        self.cx = cx
        self.cy = cy
        self.r = r
        self.cost = cost

    def params(self):
        return np.array([self.cx, self.cy, self.r])

    @classmethod
    def from_params(cls, p, cost=None):
        return cls(p[0], p[1], p[2], cost)

    def distances(self, pts):
        return np.hypot(pts[:, 0] - self.cx, pts[:, 1] - self.cy)

    def evaluate_cost(self, pts, lam=LAM):
        """
        Radius plus lam times the total overshoot of the points outside

        pts: array of shape (N, 2)
        Returns the cost, also stored in self.cost
        """
        d = self.distances(pts)
        out = np.maximum(d - self.r, 0.0).sum()
        self.cost = float(self.r + lam * out)
        return self.cost

    def tighten(self, pts):
        """Shrink or grow r to the farthest point, so the circle is valid
        
        Parameters:
        pts: array of shape (N, 2)

        Returns:
        self (Circle): the circle with updated radius and cost
        
        """
        self.r = float(self.distances(pts).max())
        self.cost = self.r
        return self


class CostBatch:
    """
    Cost of many circles at once, with buffers that are allocated only one time
    X has shape (R, 3) with columns cx, cy, r, the result has shape (R,)
    """

    def __init__(self, pts, restarts, lam=LAM):
        self.px = np.ascontiguousarray(pts[:, 0], dtype=float)
        self.py = np.ascontiguousarray(pts[:, 1], dtype=float)
        self.lam = lam
        n = len(self.px)
        self.a = np.empty((restarts, n))
        self.b = np.empty((restarts, n))

    def __call__(self, X):
        # compute the distance of each point to the center of each circle
        a, b = self.a, self.b
        np.subtract(self.px, X[:, 0:1], out=a)
        np.subtract(self.py, X[:, 1:2], out=b)
        # pythagorean distance
        np.multiply(a, a, out=a)
        np.multiply(b, b, out=b)
        np.add(a, b, out=a)
        np.sqrt(a, out=a)
        # overshoot of each point, zero when it is inside
        np.subtract(a, X[:, 2:3], out=a)
        np.maximum(a, 0.0, out=a) # clip negative distance to zero

        # all overshoots summed, plus the radius, times lam
        total = a.sum(axis=1)
        total *= self.lam
        total += X[:, 2]
        return total