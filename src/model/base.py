import numpy as np


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
 
    def evaluate_cost(self, pts):
        """
        Sum of absolute distances from every point to the circle
 
        pts: array of shape (N, 2)
        Returns the cost, also stored in self.cost
        """
        d = np.hypot(pts[:, 0] - self.cx, pts[:, 1] - self.cy)
        self.cost = float(np.abs(d - self.r).sum())
        return self.cost
 
 
class CostBatch:
    """
    Cost of many circles at once, with buffers that are allocated only one time
    X has shape (R, 3) with columns cx, cy, r, the result has shape (R,)
    """
 
    def __init__(self, pts, restarts):
        self.px = np.ascontiguousarray(pts[:, 0], dtype=float)
        self.py = np.ascontiguousarray(pts[:, 1], dtype=float)
        n = len(self.px)
        self.a = np.empty((restarts, n))
        self.b = np.empty((restarts, n))
 
    def __call__(self, X):
        a, b = self.a, self.b
        np.subtract(self.px, X[:, 0:1], out=a)
        np.subtract(self.py, X[:, 1:2], out=b)
        # square, add, sqrt is faster than hypot
        np.multiply(a, a, out=a)
        np.multiply(b, b, out=b)
        np.add(a, b, out=a)
        np.sqrt(a, out=a)
        np.subtract(a, X[:, 2:3], out=a)
        np.abs(a, out=a)
        return a.sum(axis=1)